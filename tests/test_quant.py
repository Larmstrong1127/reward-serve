"""Tests for the quantization study.

The expensive parts (a full pass over 1,987 pairs) are not run here. What is
run here is everything that could silently invalidate that pass: the variants
build and score, int8-dynamic really is quantized, the metrics are right on
hand-checkable inputs, the accuracy runner is deterministic, and the latency
runner refuses a contended box.

The anchor test reads the committed artifact rather than recomputing it: if
`results/quantization/accuracy.json` ever stops agreeing with the published
0.7026 on 1,987 pairs, the study is measuring something other than the shipped
model and this suite says so.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from reward_serve.quant import eval_split, metrics, quantize, run_accuracy, run_latency

REPO_ROOT = Path(__file__).resolve().parents[1]
ACCURACY_JSON = REPO_ROOT / "results" / "quantization" / "accuracy.json"

PAIRS: list[tuple[str, str]] = [
    ("Define latency.", "Latency is the time between a request and its response."),
    ("Define latency.", "no"),
    ("Define throughput.", "Throughput counts completed requests per unit of time."),
]


# --- metrics ------------------------------------------------------------------


def test_pairwise_accuracy_counts_ties_as_wrong() -> None:
    assert metrics.pairwise_accuracy([1.0, -1.0, 0.0, 2.0]) == 0.5


def test_kendall_tau_is_one_for_an_identical_ordering() -> None:
    xs = [0.5, -1.0, 3.0, 2.25]
    assert metrics.kendall_tau_b(xs, xs) == pytest.approx(1.0)


def test_kendall_tau_is_minus_one_for_a_reversed_ordering() -> None:
    xs = [1.0, 2.0, 3.0, 4.0]
    assert metrics.kendall_tau_b(xs, [-v for v in xs]) == pytest.approx(-1.0)


def test_kendall_tau_matches_a_hand_counted_case() -> None:
    """3 pairs: (1,2) concordant, (1,3) discordant, (2,3) discordant -> -1/3."""
    assert metrics.kendall_tau_b([1.0, 2.0, 3.0], [1.0, 2.0, 0.0]) == pytest.approx(-1.0 / 3.0)


def test_flipped_fraction_counts_sign_changes_only() -> None:
    # Third margin changes size but not sign; second flips.
    assert metrics.flipped_fraction([1.0, -1.0, 9.0], [1.0, 1.0, 0.5]) == pytest.approx(1 / 3)


def test_mean_abs_delta() -> None:
    assert metrics.mean_abs_delta([1.0, 2.0], [1.5, 4.0]) == pytest.approx(1.25)


# --- variants -----------------------------------------------------------------


@pytest.fixture(scope="module")
def fp32() -> Any:
    return quantize.build_variant("fp32")


@pytest.mark.parametrize("variant", ["fp32", "bf16", "int8-dynamic"])
def test_every_variant_loads_and_returns_one_score_per_pair(variant: str, fp32: Any) -> None:
    build = quantize.build_variant(variant) if variant != "fp32" else fp32
    scores = build.scorer.score_pairs(PAIRS)
    assert len(scores) == len(PAIRS)
    assert all(isinstance(s, float) for s in scores)
    assert all(s == s for s in scores)  # no NaN
    # Same interface as reward_serve.model's scorer: the server could load it.
    assert build.scorer.max_length == fp32.scorer.max_length
    assert build.scorer.temperature == fp32.scorer.temperature
    assert build.scorer.score_pairs([]) == []


def test_int8_dynamic_actually_has_quantized_linear_modules() -> None:
    build = quantize.build_variant("int8-dynamic")
    assert build.quantized_linear_modules > 0
    assert build.plain_linear_modules == 0
    kinds = {type(m).__module__ for m in build.scorer.model.modules() if _is_linear_like(m)}
    assert any(k.startswith("torch.ao.nn.quantized") for k in kinds)


def _is_linear_like(module: nn.Module) -> bool:
    return type(module).__name__ == "Linear"


def test_int8_dynamic_is_smaller_than_fp32_but_the_embedding_is_untouched(fp32: Any) -> None:
    """quantize_dynamic only touches Linear, so the vocab embedding stays fp32.

    This is the whole reason the on-disk saving is far short of 4x, and the
    server-side reading of the result depends on it, so it is asserted.
    """
    int8 = quantize.build_variant("int8-dynamic")
    assert int8.size_mb < fp32.size_mb
    assert int8.size_mb > fp32.size_mb / 4


def test_bf16_matmul_is_supported_on_this_torch_build() -> None:
    assert quantize.bf16_matmul_supported() is True


def test_unknown_variant_is_rejected() -> None:
    with pytest.raises(KeyError):
        quantize.build_variant("int4-imaginary")


# --- the anchor ---------------------------------------------------------------


@pytest.mark.skipif(not ACCURACY_JSON.exists(), reason="accuracy.json not produced yet")
def test_fp32_reproduces_the_published_accuracy_on_the_published_split() -> None:
    data = json.loads(ACCURACY_JSON.read_text(encoding="utf-8"))
    assert data["n_pairs_evaluated"] == eval_split.PUBLISHED_N_PAIRS
    assert data["max_length_tokens"] == 512
    assert data["split"] == "test_prefs"
    assert data["variants"]["fp32"]["pairwise_accuracy"] == eval_split.PUBLISHED_FP32_ACCURACY


@pytest.mark.skipif(not ACCURACY_JSON.exists(), reason="accuracy.json not produced yet")
def test_every_variant_in_the_artifact_is_measured_on_the_same_pairs() -> None:
    data = json.loads(ACCURACY_JSON.read_text(encoding="utf-8"))
    fp32 = data["variants"]["fp32"]
    assert fp32["kendall_tau_margins_vs_fp32"] == 1.0
    assert fp32["flipped_fraction_vs_fp32"] == 0.0
    assert data["device"] == "cpu"


# --- the accuracy runner ------------------------------------------------------


class _FakeScorer:
    """Deterministic stand-in with the RewardModel interface."""

    def __init__(self, jitter: float = 0.0) -> None:
        self.max_length = 512
        self.temperature = 1.1668
        self.tokenizer = object()
        self.model = nn.Linear(2, 1)
        self.jitter = jitter

    def score_pairs(self, pairs: Any) -> list[float]:
        return [len(r) / 10.0 + self.jitter * len(p) for p, r in pairs]


def _fake_pairs(n: int = 12) -> list[eval_split.PreferencePair]:
    return [
        eval_split.PreferencePair(
            prompt=f"prompt {i}", chosen="a longer answer here", rejected="short" if i % 3 else "x"
        )
        for i in range(n)
    ]


@pytest.fixture
def patched_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_build(variant: str, model_id: str = "") -> Any:
        return quantize.VariantBuildResult(
            variant=variant,
            scorer=_FakeScorer(jitter=0.0 if variant == "fp32" else 0.01),  # type: ignore[arg-type]
            torch_api=quantize.TORCH_API.get(variant, "stub"),
            size_mb=1.0,
            quantized_linear_modules=0 if variant == "fp32" else 1,
            plain_linear_modules=1 if variant == "fp32" else 0,
        )

    def fake_load(tokenizer: Any, max_length: int, limit: int | None = None, **kw: Any) -> Any:
        pairs = _fake_pairs()
        stats = eval_split.AuditStats(total=13, dropped_identical=1)
        return (pairs[:limit] if limit else pairs), stats

    monkeypatch.setattr(run_accuracy, "build_variant", fake_build)
    monkeypatch.setattr(run_accuracy, "load_eval_pairs", fake_load)


def test_accuracy_runner_is_deterministic_across_two_runs(
    patched_runner: None, tmp_path: Path
) -> None:
    kwargs: dict[str, Any] = {
        "variants": ["fp32", "int8-dynamic"],
        "use_cache": False,
        "measure_box": False,
    }
    first = run_accuracy.run(out_path=tmp_path / "a" / "accuracy.json", **kwargs)
    second = run_accuracy.run(out_path=tmp_path / "b" / "accuracy.json", **kwargs)
    assert first["variants"] == second["variants"]
    assert first["n_pairs_evaluated"] == second["n_pairs_evaluated"]


def test_accuracy_runner_always_includes_fp32_as_the_baseline(
    patched_runner: None, tmp_path: Path
) -> None:
    out = run_accuracy.run(
        variants=["int8-dynamic"],
        out_path=tmp_path / "accuracy.json",
        use_cache=False,
        measure_box=False,
    )
    assert "fp32" in out["variants"]
    assert out["variants"]["fp32"]["accuracy_delta_vs_fp32"] == 0.0


def test_accuracy_runner_merges_a_later_variant_into_the_same_artifact(
    patched_runner: None, tmp_path: Path
) -> None:
    out_path = tmp_path / "accuracy.json"
    run_accuracy.run(
        variants=["fp32"], out_path=out_path, use_cache=False, measure_box=False
    )
    run_accuracy.run(
        variants=["int8-dynamic"], out_path=out_path, use_cache=False, measure_box=False
    )
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert set(data["variants"]) == {"fp32", "int8-dynamic"}


def test_score_split_returns_one_score_per_side_per_pair() -> None:
    pairs = _fake_pairs(7)
    chosen, rejected = run_accuracy.score_split(_FakeScorer(), pairs, batch_size=3)  # type: ignore[arg-type]
    assert len(chosen) == len(rejected) == 7


# --- the latency runner -------------------------------------------------------


def test_latency_runner_refuses_when_box_quiet_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Charter rule 3: a contended box produces a refusal, never a number."""
    report = tmp_path / "box_quiet_latency.json"
    out = tmp_path / "latency.json"
    out.write_text("{}", encoding="utf-8")  # a stale artifact must not survive

    import scripts.box_quiet as box_quiet

    def refuse(path: Path | None = None) -> dict[str, Any]:
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"passed": False, "cpu_mean_pct": 91.0}), encoding="utf-8")
        raise SystemExit("box_quiet FAIL")

    monkeypatch.setattr(box_quiet, "require_quiet", refuse)

    called: list[str] = []
    monkeypatch.setattr(
        run_latency, "build_variant", lambda *a, **k: called.append("built")  # type: ignore[arg-type,return-value]
    )

    rc = run_latency.run(variants=["fp32"], out_path=out, quiet_report=report)
    assert rc == 1
    assert not out.exists(), "a refused sweep must leave no latency.json"
    assert json.loads(report.read_text(encoding="utf-8"))["passed"] is False
    assert called == [], "nothing may be built or measured after a refusal"


def test_percentile_is_an_exact_order_statistic() -> None:
    values = [float(v) for v in range(1, 101)]
    assert run_latency.percentile(values, 0.50) == 51.0
    assert run_latency.percentile(values, 0.95) == 95.0
    with pytest.raises(ValueError):
        run_latency.percentile([], 0.5)


def test_latency_filler_fills_the_whole_sequence_budget(fp32: Any) -> None:
    prompt, response = run_latency.budget_filling_pair(fp32.scorer)
    encoded = fp32.scorer.tokenizer(
        prompt, response, truncation="longest_first", max_length=fp32.scorer.max_length
    )
    assert len(encoded["input_ids"]) == fp32.scorer.max_length


# --- the split ----------------------------------------------------------------


def test_truncation_filter_drops_only_pairs_that_collide_after_truncation() -> None:
    class Tok:
        def __call__(self, a: str, b: str, truncation: bool = True, max_length: int = 8) -> Any:
            return {"input_ids": [ord(c) for c in f"{a}|{b}"][:max_length]}

    keep = eval_split.PreferencePair("p", "chosen answer", "rejected")
    collide = eval_split.PreferencePair("a very long shared prompt", "xxxx", "yyyy")
    kept, stats = audit(Tok(), [keep, collide])
    assert stats.total == 2
    assert stats.dropped_identical == 1
    assert kept == [keep]


def audit(tok: Any, pairs: list[eval_split.PreferencePair]) -> Any:
    return eval_split.audit_and_filter_pairs(pairs, tok, max_length=8)


def test_cpu_only_is_enforced_in_the_quant_package() -> None:
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert not torch.cuda.is_available()


def test_score_split_resumes_from_a_checkpoint_instead_of_rescoring(tmp_path: Path) -> None:
    """A killed pass must cost the current chunk, not the whole variant."""
    pairs = _fake_pairs(24)
    ckpt = tmp_path / "partial.json"

    full = _FakeScorer()
    chosen, rejected = run_accuracy.score_split(full, pairs, batch_size=8, checkpoint=ckpt)  # type: ignore[arg-type]
    assert ckpt.exists()

    class _Counting(_FakeScorer):
        calls = 0

        def score_pairs(self, pairs: Any) -> list[float]:
            _Counting.calls += 1
            return super().score_pairs(pairs)

    again_c, again_r = run_accuracy.score_split(_Counting(), pairs, batch_size=8, checkpoint=ckpt)  # type: ignore[arg-type]
    assert _Counting.calls == 0, "a complete checkpoint must not be rescored"
    assert (again_c, again_r) == (chosen, rejected)


def test_score_split_checkpoint_survives_a_truncated_side(tmp_path: Path) -> None:
    """A kill between the chosen and rejected write must not misalign the pairs."""
    ckpt = tmp_path / "partial.json"
    ckpt.write_text(json.dumps({"chosen": [1.0, 2.0, 3.0], "rejected": [1.0, 2.0]}), "utf-8")
    chosen, rejected = run_accuracy.score_split(
        _FakeScorer(), _fake_pairs(8), batch_size=8, checkpoint=ckpt  # type: ignore[arg-type]
    )
    assert len(chosen) == len(rejected) == 8


def test_completed_run_writes_both_accuracy_json_and_box_quiet_json(
    patched_runner: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The artifact contract: a finished pass leaves both files, in one directory.

    Verified against a fresh directory, because the reader clones cold and the
    box_quiet report is written before the results file.
    """
    import scripts.box_quiet as box_quiet

    report = {"passed": False, "cpu_mean_pct": 44.0}
    monkeypatch.setattr(box_quiet, "measure", lambda *a, **k: report)
    out = tmp_path / "brand" / "new" / "accuracy.json"

    payload = run_accuracy.run(
        variants=["fp32", "int8-dynamic"], out_path=out, use_cache=False, measure_box=True
    )

    assert out.exists()
    box = out.parent / "box_quiet.json"
    assert box.exists()
    assert json.loads(box.read_text(encoding="utf-8"))["passed"] is False
    assert payload["box_quiet_passed"] is False
    written = json.loads(out.read_text(encoding="utf-8"))
    assert set(written["variants"]) == {"fp32", "int8-dynamic"}
    for key in ("n_pairs_evaluated", "max_length_tokens", "torch_num_threads", "command"):
        assert key in written


def test_a_partial_pass_is_never_written_as_a_complete_one(
    patched_runner: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Never round a partial into a done: a short pass must raise, not publish."""
    monkeypatch.setattr(
        run_accuracy, "score_split", lambda *a, **k: ([1.0, 2.0], [0.5, 0.25])
    )
    out = tmp_path / "accuracy.json"
    with pytest.raises(RuntimeError, match="refusing to write a partial"):
        run_accuracy.run(variants=["fp32"], out_path=out, use_cache=False, measure_box=False)
    assert not out.exists()


DOC = REPO_ROOT / "docs" / "quantization.md"


@pytest.mark.skipif(not ACCURACY_JSON.exists(), reason="accuracy.json not produced yet")
def test_the_doc_has_no_placeholders_left_once_the_artifact_exists() -> None:
    """The doc is pre-structured with placeholders; the artifact landing retires them."""
    assert "TODO-FROM-accuracy.json" not in DOC.read_text(encoding="utf-8")


@pytest.mark.skipif(not ACCURACY_JSON.exists(), reason="accuracy.json not produced yet")
def test_every_accuracy_quoted_in_the_doc_appears_in_the_artifact() -> None:
    """Charter rule 2: a figure in prose must be readable in a committed artifact."""
    data = json.loads(ACCURACY_JSON.read_text(encoding="utf-8"))
    text = DOC.read_text(encoding="utf-8")
    for name, row in data["variants"].items():
        assert f"{row['pairwise_accuracy']:.4f}" in text, f"{name} accuracy missing from the doc"


def test_latency_doc_claims_no_number_while_the_gate_has_not_passed() -> None:
    """No latency.json means no latency figure may appear in the doc."""
    latency = REPO_ROOT / "results" / "quantization" / "latency.json"
    if latency.exists():
        pytest.skip("latency.json exists; this guard only covers the un-measured state")
    section = DOC.read_text(encoding="utf-8").split("## Latency")[1].split("## Threats")[0]
    assert "not measured" in section.lower()
    assert not re.search(r"\d+(\.\d+)?\s*(ms|s)", section), "a latency figure leaked in"

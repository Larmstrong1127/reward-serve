"""Quantized variants of the served reward model, CPU only.

Every builder returns a `RewardModel` - the same object `reward_serve.server`
loads - so the server could adopt any variant by swapping one call. The score
semantics are unchanged in all of them: raw logit divided by the checkpoint's
`reward_temperature`, ranking-only.

Variants and the exact torch API behind each:

    fp32          the shipped loader, `RewardModel(model_id)`; no change
    bf16          `RewardModel.model.to(torch.bfloat16)`
    int8-dynamic  `torch.ao.quantization.quantize_dynamic(m, {nn.Linear},
                  dtype=torch.qint8)`
    int8-static   `torch.ao.quantization.quantize_fx.prepare_fx` /
                  `convert_fx` - attempted, see `attempt_int8_static`

CPU-only is not incidental: `CUDA_VISIBLE_DEVICES` is emptied before torch is
imported, so no GPU int8 kernel is exercised anywhere in this module.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import io  # noqa: E402
from collections.abc import Callable, Sequence  # noqa: E402
from dataclasses import dataclass  # noqa: E402

import torch  # noqa: E402
from torch import nn  # noqa: E402

from reward_serve.model import DEFAULT_MODEL_ID, RewardModel  # noqa: E402

__all__ = [
    "BUILDERS",
    "VariantBuildResult",
    "attempt_int8_static",
    "bf16_matmul_supported",
    "build_variant",
    "count_quantized_linear",
    "load_bf16",
    "load_fp32",
    "load_int8_dynamic",
    "state_dict_size_mb",
]


@dataclass(frozen=True)
class VariantBuildResult:
    """What a variant is and what it costs to hold."""

    variant: str
    scorer: RewardModel
    torch_api: str
    size_mb: float
    quantized_linear_modules: int
    plain_linear_modules: int
    notes: str = ""


def state_dict_size_mb(model: nn.Module) -> float:
    """Serialized weight size in MB.

    Measured by serializing `state_dict()` to an in-memory buffer rather than
    by summing `numel * element_size`: dynamically quantized Linear layers hold
    packed int8 params that a naive tensor walk does not see at all.
    """
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return round(buf.getbuffer().nbytes / (1024 * 1024), 3)


def count_quantized_linear(model: nn.Module) -> tuple[int, int]:
    """(quantized Linear modules, plain nn.Linear modules) in the module tree."""
    quantized = 0
    plain = 0
    for module in model.modules():
        qualified = type(module).__module__
        if qualified.startswith(("torch.ao.nn.quantized", "torch.nn.quantized")):
            # Exact name only: a quantized Linear also owns a child
            # `LinearPackedParams`, which would double the count.
            if type(module).__name__ == "Linear":
                quantized += 1
        elif isinstance(module, nn.Linear):
            plain += 1
    return quantized, plain


def bf16_matmul_supported() -> bool:
    """Verify torch can actually run a bf16 matmul on this CPU build."""
    try:
        a = torch.ones(4, 4, dtype=torch.bfloat16)
        return bool(torch.isfinite(a @ a).all())
    except Exception:
        return False


def load_fp32(model_id: str = DEFAULT_MODEL_ID) -> RewardModel:
    """Baseline: the shipped loader, untouched."""
    return RewardModel(model_id)


def load_bf16(model_id: str = DEFAULT_MODEL_ID) -> RewardModel:
    """Weights and activations in bfloat16. Scores are cast back to float32.

    `RewardModel.score_pairs` already calls `.float()` on the logits, so the
    returned score type is identical to fp32's.
    """
    if not bf16_matmul_supported():
        raise RuntimeError("this torch build cannot run a bfloat16 matmul on CPU")
    scorer = RewardModel(model_id)
    scorer.model.to(torch.bfloat16)
    scorer.model.eval()
    return scorer


def load_int8_dynamic(model_id: str = DEFAULT_MODEL_ID) -> RewardModel:
    """int8 weights, activations quantized per-batch at runtime.

    `quantize_dynamic` replaces every `nn.Linear` with a quantized Linear whose
    weight is int8; activations are observed and quantized inside the op call,
    so there is no calibration pass. This is the classic CPU path and the one a
    server can adopt with no data plumbing at all.
    """
    scorer = RewardModel(model_id)
    scorer.model = torch.ao.quantization.quantize_dynamic(
        scorer.model, {nn.Linear}, dtype=torch.qint8
    )
    scorer.model.eval()
    return scorer


def attempt_int8_static(
    calibration_pairs: Sequence[tuple[str, str]],
    model_id: str = DEFAULT_MODEL_ID,
) -> tuple[RewardModel | None, str]:
    """Try FX-graph-mode static quantization. Returns (scorer or None, reason).

    Static quantization needs activation ranges from a calibration pass, which
    means the model must be captured as a graph (`prepare_fx`) or hand-annotated
    with Quant/DeQuant stubs. DeBERTa-v2's disentangled attention is written with
    Python control flow, so symbolic tracing is the part that decides whether
    this is available at all. This function reports what happened instead of
    guessing; the caller records the reason verbatim.
    """
    from torch.ao.quantization import get_default_qconfig
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    if not calibration_pairs:
        return None, "no calibration pairs supplied"
    scorer = RewardModel(model_id)

    def encode(chunk: Sequence[tuple[str, str]]) -> dict[str, torch.Tensor]:
        enc = scorer.tokenizer(
            [p for p, _ in chunk],
            [r for _, r in chunk],
            padding=True,
            truncation="longest_first",
            max_length=scorer.max_length,
            return_tensors="pt",
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    example = encode(calibration_pairs[:1])
    qconfig_mapping = {"": get_default_qconfig("x86")}
    try:
        prepared = prepare_fx(
            scorer.model,
            qconfig_mapping,
            example_inputs=(example["input_ids"], example["attention_mask"]),
        )
    except Exception as exc:  # tracing is the gate; say so rather than fake it
        return None, f"prepare_fx failed: {type(exc).__name__}: {exc}"
    try:
        with torch.inference_mode():
            for start in range(0, len(calibration_pairs), 8):
                enc = encode(calibration_pairs[start : start + 8])
                prepared(enc["input_ids"], enc["attention_mask"])
        scorer.model = convert_fx(prepared)
        scorer.model.eval()
    except Exception as exc:
        return None, f"calibration or convert_fx failed: {type(exc).__name__}: {exc}"
    return scorer, "converted"


BUILDERS: dict[str, Callable[[str], RewardModel]] = {
    "fp32": load_fp32,
    "bf16": load_bf16,
    "int8-dynamic": load_int8_dynamic,
}

TORCH_API: dict[str, str] = {
    "fp32": "AutoModelForSequenceClassification.from_pretrained(...)  # no quantization",
    "bf16": "model.to(torch.bfloat16)",
    "int8-dynamic": (
        "torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)"
    ),
    "int8-static": (
        "torch.ao.quantization.quantize_fx.prepare_fx(...) -> calibrate -> convert_fx(...)"
    ),
}


def build_variant(variant: str, model_id: str = DEFAULT_MODEL_ID) -> VariantBuildResult:
    """Build one variant and measure what it costs to hold in memory."""
    if variant not in BUILDERS:
        raise KeyError(f"unknown variant {variant!r}; known: {sorted(BUILDERS)}")
    scorer = BUILDERS[variant](model_id)
    quantized, plain = count_quantized_linear(scorer.model)
    return VariantBuildResult(
        variant=variant,
        scorer=scorer,
        torch_api=TORCH_API[variant],
        size_mb=state_dict_size_mb(scorer.model),
        quantized_linear_modules=quantized,
        plain_linear_modules=plain,
    )

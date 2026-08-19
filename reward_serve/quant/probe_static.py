"""Attempt int8 static (post-training) quantization and record what happened.

Static PTQ is the variant that would buy the most on CPU: activations are
quantized once from calibration statistics instead of re-observed on every
forward. It needs two things this study can supply - a calibration set held out
of the eval split, and a CPU - and one thing it turns out it cannot get on this
stack: a traceable graph.

The calibration set is the first 256 pairs of UltraFeedback `train_prefs` after
the same truncation filter the eval split uses. It is disjoint from the 1,987
evaluated pairs by construction: those come from `test_prefs`, a different split
of the same dataset, and no pair is shared between them.

    python -m reward_serve.quant.probe_static

Writes `results/quantization/int8_static_attempt.json`. A failure is the result
here, not an error: the alternative is fabricating a static number.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import torch  # noqa: E402
import transformers  # noqa: E402

from reward_serve.model import DEFAULT_MODEL_ID  # noqa: E402
from reward_serve.quant.eval_split import load_calibration_pairs  # noqa: E402
from reward_serve.quant.quantize import TORCH_API, attempt_int8_static, build_variant  # noqa: E402

__all__ = ["main", "run"]

DEFAULT_OUT = Path("results/quantization/int8_static_attempt.json")
CALIBRATION_N = 256


def run(out_path: Path = DEFAULT_OUT, n_calibration: int = CALIBRATION_N) -> dict[str, Any]:
    fp32 = build_variant("fp32", DEFAULT_MODEL_ID)
    print(f"[calib] taking {n_calibration} pairs from train_prefs", flush=True)
    calib = load_calibration_pairs(
        fp32.scorer.tokenizer, fp32.scorer.max_length, n_pairs=n_calibration
    )
    # Calibrate on the chosen side: it is what a served response looks like.
    texts = [(p.prompt, p.chosen) for p in calib]
    del fp32

    started = time.perf_counter()
    scorer, reason = attempt_int8_static(texts, DEFAULT_MODEL_ID)
    elapsed = round(time.perf_counter() - started, 2)

    payload: dict[str, Any] = {
        "_comment": (
            "Record of an attempt at int8 static post-training quantization. If "
            "'succeeded' is false, no static accuracy number exists in this study and "
            "none should be quoted."
        ),
        "model_id": DEFAULT_MODEL_ID,
        "torch_api_attempted": TORCH_API["int8-static"],
        "calibration_split": "UltraFeedback train_prefs",
        "calibration_n_pairs": len(texts),
        "calibration_side": "chosen response",
        "calibration_disjoint_from_eval": (
            "yes - the eval split is test_prefs; no pair is shared with train_prefs"
        ),
        "device": "cpu",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": sys.version.split()[0],
        "elapsed_s": elapsed,
        "succeeded": scorer is not None,
        "reason": reason,
        "command": "python -m reward_serve.quant.probe_static",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-calibration", type=int, default=CALIBRATION_N)
    args = parser.parse_args(argv)
    run(args.out, args.n_calibration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

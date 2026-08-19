"""Quantization study for the served reward model. CPU only, by design.

The whole study runs on CPU: `CUDA_VISIBLE_DEVICES` is emptied before torch is
imported, in this package and in `reward_serve.model`. GPU int8 kernels are
therefore out of scope for every number produced here.
"""

from __future__ import annotations

__all__ = ["VARIANTS"]

VARIANTS = ("fp32", "bf16", "int8-dynamic", "int8-static")

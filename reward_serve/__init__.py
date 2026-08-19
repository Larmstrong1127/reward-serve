"""reward-serve: a continuous-batching inference server for one machine."""

from __future__ import annotations

import os

# The GPU on this machine belongs to other work. Enforced before torch imports.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

__version__ = "0.1.0"

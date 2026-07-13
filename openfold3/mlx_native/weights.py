"""Load the torch OpenFold3 checkpoint into MLX arrays."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np


def load_state_dict(path: str | Path) -> dict:
    """Load the released .pt checkpoint into a flat dict[str, mx.array] (float32)."""
    import torch

    sd = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    out = {}
    for k, v in sd.items():
        if hasattr(v, "detach"):
            out[k] = mx.array(v.detach().to(torch.float32).numpy())
    return out


def subtree(sd: dict, prefix: str) -> dict:
    """Return the sub-dict under ``prefix`` with the prefix stripped from keys."""
    if not prefix.endswith("."):
        prefix = prefix + "."
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def torch_params_to_mlx(module) -> dict:
    """Convert a torch module's named parameters to a dict[str, mx.array]."""
    import torch

    return {
        name: mx.array(p.detach().to(torch.float32).numpy())
        for name, p in module.named_parameters()
    }


def to_mlx(t) -> mx.array:
    import torch

    return mx.array(t.detach().to(torch.float32).numpy())


def to_np(a: mx.array) -> np.ndarray:
    return np.array(a)

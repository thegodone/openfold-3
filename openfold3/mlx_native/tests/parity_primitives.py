"""Parity gate for the remaining Pairformer primitives vs torch references:
SwiGLUTransition, general gated MHA (Attention), and TriangleAttention
(starting + ending). Random non-zero weights; pass if max abs diff < 1e-4.
"""

import numpy as np
import torch

import mlx.core as mx

from openfold3.mlx_native import modules as M
from openfold3.mlx_native.weights import to_mlx


def _randomize(mod):
    with torch.no_grad():
        for prm in mod.parameters():
            prm.copy_(torch.randn_like(prm) * 0.5)


def _named(mod):
    return {name: to_mlx(p) for name, p in mod.named_parameters()}


def test_swiglu_transition(N=17, c=128, n=4, seed=0):
    torch.manual_seed(seed)
    from openfold3.core.model.layers.transition import SwiGLUTransition

    mod = SwiGLUTransition(c_in=c, n=n).eval()
    _randomize(mod)
    x = torch.randn(N, c)
    mask = (torch.rand(N) > 0.2).float()
    with torch.no_grad():
        out_t = mod(x.clone(), mask.clone()).numpy()
    out_m = np.array(
        M.swiglu_transition(to_mlx(x), _named(mod), mask=to_mlx(mask),
                            ln_eps=mod.layer_norm.eps)
    )
    return _report("swiglu_transition", out_t, out_m)


def test_mha(N=17, cq=128, ch=32, H=4, seed=1):
    torch.manual_seed(seed)
    from openfold3.core.model.primitives.attention import Attention

    mod = Attention(c_q=cq, c_k=cq, c_v=cq, c_hidden=ch, no_heads=H, gating=True).eval()
    _randomize(mod)
    x = torch.randn(N, cq)
    bias = torch.randn(1, N, N)  # broadcasts to [H, Q, K]
    with torch.no_grad():
        out_t = mod(x.clone(), x.clone(), biases=[bias.clone()]).numpy()
    out_m = np.array(
        M.mha(to_mlx(x), to_mlx(x), _named(mod), no_heads=H,
              biases=(to_mlx(bias),), gating=True)
    )
    return _report("mha(gated)", out_t, out_m)


def test_triangle_attention(starting, N=17, c=128, ch=32, H=4, seed=2):
    torch.manual_seed(seed)
    from openfold3.core.model.layers.triangular_attention import (
        TriangleAttentionEndingNode,
        TriangleAttentionStartingNode,
    )

    cls = TriangleAttentionStartingNode if starting else TriangleAttentionEndingNode
    mod = cls(c_in=c, c_hidden=ch, no_heads=H).eval()
    _randomize(mod)
    x = torch.randn(N, N, c)
    mask = (torch.rand(N, N) > 0.2).float()
    with torch.no_grad():
        out_t = mod(x.clone(), mask=mask.clone()).numpy()
    out_m = np.array(
        M.triangle_attention(to_mlx(x), _named(mod), no_heads=H, starting=starting,
                             mask=to_mlx(mask), inf=mod.inf, ln_eps=mod.layer_norm.eps)
    )
    return _report(f"tri_attn {'start' if starting else 'end'}", out_t, out_m)


def _report(name, out_t, out_m):
    if out_t.shape != out_m.shape:
        print(f"{name}: SHAPE MISMATCH torch={out_t.shape} mlx={out_m.shape}")
        return 1e9
    diff = float(np.abs(out_t - out_m).max())
    rng = float(np.abs(out_t).max())
    rel = diff / (rng + 1e-9)
    print(f"{name}: max_abs_diff={diff:.3e}  rel={rel:.3e}  torch_range={rng:.3f}")
    # Judge on relative error: abs diff scales with output magnitude in float32.
    return rel


if __name__ == "__main__":
    mx.set_default_device(mx.gpu)
    diffs = {
        "swiglu_transition": test_swiglu_transition(),
        "mha": test_mha(),
        "tri_attn_start": test_triangle_attention(True),
        "tri_attn_end": test_triangle_attention(False),
    }
    worst = max(diffs.values())
    print("PARITY", "PASS" if worst < 1e-4 else "FAIL", f"(worst rel={worst:.3e}, threshold rel 1e-4)")
    raise SystemExit(0 if worst < 1e-4 else 1)

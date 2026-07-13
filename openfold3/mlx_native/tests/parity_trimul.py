"""Parity gate: MLX triangle-multiplicative-update vs the torch reference module.

Builds the torch TriangleMultiplication{Outgoing,Incoming} with random weights,
loads the same weights into the MLX port, and compares outputs on a random input.
Passes if max abs diff < 1e-4 (float32).
"""

import numpy as np
import torch

import mlx.core as mx

from openfold3.core.model.layers.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from openfold3.mlx_native.modules import tri_mul
from openfold3.mlx_native.weights import to_mlx


def run(outgoing: bool, N: int = 17, c: int = 128, seed: int = 0) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)

    cls = TriangleMultiplicationOutgoing if outgoing else TriangleMultiplicationIncoming
    mod = cls(c_z=c, c_hidden=c).eval()

    # AF2 zero-inits the final projections, so a fresh module outputs all zeros
    # (a vacuous parity check). Fill every parameter with random values so the
    # comparison actually exercises the math.
    with torch.no_grad():
        for prm in mod.parameters():
            prm.copy_(torch.randn_like(prm) * 0.5)

    z = torch.randn(N, N, c)
    mask = (torch.rand(N, N) > 0.2).float()

    with torch.no_grad():
        out_t = mod(z.clone(), mask.clone(), inplace_safe=False).numpy()

    # Gather the torch weights this module actually holds.
    p = {name: to_mlx(param) for name, param in mod.named_parameters()}
    out_m = np.array(
        tri_mul(to_mlx(z), p, outgoing=outgoing, mask=to_mlx(mask), ln_eps=mod.layer_norm_in.eps)
    )

    diff = float(np.abs(out_t - out_m).max())
    rel = diff / (float(np.abs(out_t).max()) + 1e-9)
    print(
        f"tri_mul {'outgoing' if outgoing else 'incoming'}: "
        f"max_abs_diff={diff:.3e}  rel={rel:.3e}  torch_range=[{out_t.min():.3f},{out_t.max():.3f}]"
    )
    return diff


if __name__ == "__main__":
    mx.set_default_device(mx.gpu)
    d_out = run(outgoing=True)
    d_in = run(outgoing=False)
    ok = max(d_out, d_in) < 1e-4
    print("PARITY", "PASS" if ok else "FAIL", f"(threshold 1e-4)")
    raise SystemExit(0 if ok else 1)

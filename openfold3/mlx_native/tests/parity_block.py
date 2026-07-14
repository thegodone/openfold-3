"""Parity gate for the full Pairformer block AND the 48-block stack, using the
REAL released checkpoint weights.

Instantiates the torch PairFormerBlock with the checkpoint's dims, loads real
block weights, and compares against the MLX port on random s/z inputs.
"""

import sys

import numpy as np
import torch

import mlx.core as mx

from openfold3.core.model.latent.pairformer import PairFormerBlock
from openfold3.mlx_native import modules as M
from openfold3.mlx_native.weights import load_state_dict, subtree, to_mlx

CKPT = "/Users/tgg/.openfold3/of3_ft3_v1.pt"

# From checkpoint inspection (pairformer_stack.blocks.N.*):
CFG = dict(
    c_s=384, c_z=128,
    c_hidden_pair_bias=24, no_heads_pair_bias=16,   # 24*16 = 384
    c_hidden_mul=128,
    c_hidden_pair_att=32, no_heads_pair=4,          # 32*4  = 128
    transition_type="swiglu", transition_n=4,
    pair_dropout=0.0, fuse_projection_weights=False, inf=1e9,
)


def build_torch_block(sd_torch, prefix):
    blk = PairFormerBlock(**CFG).eval()
    want = {k[len(prefix) + 1:]: v for k, v in sd_torch.items() if k.startswith(prefix + ".")}
    missing, unexpected = blk.load_state_dict(want, strict=False)
    missing = [m for m in missing if "dropout" not in m]
    if missing or unexpected:
        print(f"  load_state_dict missing={missing[:4]} unexpected={unexpected[:4]}")
    return blk


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    # torch checkpoint (raw) for building the reference module
    raw = torch.load(CKPT, map_location="cpu", weights_only=False)
    raw = raw.get("state_dict", raw)
    # MLX checkpoint
    sd = load_state_dict(CKPT)

    N = 76  # ubiquitin length
    s = torch.randn(N, CFG["c_s"]) * 0.5
    z = torch.randn(N, N, CFG["c_z"]) * 0.5
    single_mask = (torch.rand(N) > 0.1).float()
    pair_mask = (torch.rand(N, N) > 0.1).float()

    # ---- single block parity (block 0) ----
    prefix0 = "pairformer_stack.blocks.0"
    blk = build_torch_block(raw, prefix0)
    with torch.no_grad():
        s_t, z_t = blk(s.clone(), z.clone(), single_mask.clone(), pair_mask.clone(),
                       inplace_safe=False)
    p0 = subtree(sd, prefix0)
    s_m, z_m = M.pairformer_block(
        to_mlx(s), to_mlx(z), p0, to_mlx(single_mask), to_mlx(pair_mask),
        no_heads_pair_bias=CFG["no_heads_pair_bias"], no_heads_pair=CFG["no_heads_pair"],
        inf=CFG["inf"],
    )
    ds = float(np.abs(s_t.numpy() - np.array(s_m)).max()) / (float(np.abs(s_t.numpy()).max()) + 1e-9)
    dz = float(np.abs(z_t.numpy() - np.array(z_m)).max()) / (float(np.abs(z_t.numpy()).max()) + 1e-9)
    print(f"block0 parity: rel_s={ds:.3e}  rel_z={dz:.3e}")

    # ---- full 48-block stack parity ----
    n_blocks = 1 + max(
        int(k.split("blocks.")[1].split(".")[0])
        for k in raw if k.startswith("pairformer_stack.blocks.")
    )
    print(f"stack has {n_blocks} blocks; running torch + MLX...")

    st, zt = s.clone(), z.clone()
    with torch.no_grad():
        for i in range(n_blocks):
            b = build_torch_block(raw, f"pairformer_stack.blocks.{i}")
            st, zt = b(st, zt, single_mask.clone(), pair_mask.clone(), inplace_safe=False)

    p_stack = subtree(sd, "pairformer_stack")
    sm, zm = M.pairformer_stack(
        to_mlx(s), to_mlx(z), p_stack, to_mlx(single_mask), to_mlx(pair_mask),
        n_blocks=n_blocks, no_heads_pair_bias=CFG["no_heads_pair_bias"],
        no_heads_pair=CFG["no_heads_pair"], inf=CFG["inf"],
    )
    mx.eval(sm, zm)
    ds = float(np.abs(st.numpy() - np.array(sm)).max()) / (float(np.abs(st.numpy()).max()) + 1e-9)
    dz = float(np.abs(zt.numpy() - np.array(zm)).max()) / (float(np.abs(zt.numpy()).max()) + 1e-9)
    print(f"stack({n_blocks}) parity: rel_s={ds:.3e}  rel_z={dz:.3e}")

    ok = max(ds, dz) < 1e-3  # 48 blocks accumulate fp32 error; 1e-3 rel is tight
    print("PARITY", "PASS" if ok else "FAIL", f"(rel threshold 1e-3)")
    return 0 if ok else 1


if __name__ == "__main__":
    mx.set_default_device(mx.gpu)
    sys.exit(main())

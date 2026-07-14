"""Profile the 48-block Pairformer stack: torch-MPS vs MLX vs MLX+compile.

Uses real checkpoint weights at ubiquitin size (N=76). Reports median forward
wall-clock over several runs after warmup.
"""

import time

import numpy as np
import torch

import mlx.core as mx

from openfold3.core.model.latent.pairformer import PairFormerBlock
from openfold3.mlx_native import modules as M
from openfold3.mlx_native.weights import load_state_dict, subtree, to_mlx
from openfold3.mlx_native.tests.parity_block import CFG, build_torch_block

CKPT = "/Users/tgg/.openfold3/of3_ft3_v1.pt"
N = 76
REPS = 5


def timeit(fn, sync, warmup=2, reps=REPS):
    for _ in range(warmup):
        fn(); sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); sync()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), float(np.min(ts))


def main():
    torch.manual_seed(0)
    raw = torch.load(CKPT, map_location="cpu", weights_only=False)
    raw = raw.get("state_dict", raw)
    sd = load_state_dict(CKPT)
    n_blocks = 1 + max(int(k.split("blocks.")[1].split(".")[0])
                       for k in raw if k.startswith("pairformer_stack.blocks."))

    s0 = torch.randn(N, CFG["c_s"]) * 0.5
    z0 = torch.randn(N, N, CFG["c_z"]) * 0.5
    sm_mask = (torch.rand(N) > 0.1).float()
    pm_mask = (torch.rand(N, N) > 0.1).float()

    results = {}

    # ---------- torch MPS ----------
    if torch.backends.mps.is_available():
        dev = torch.device("mps")
        blocks = []
        for i in range(n_blocks):
            b = build_torch_block(raw, f"pairformer_stack.blocks.{i}").to(dev)
            blocks.append(b)
        s_d, z_d = s0.to(dev), z0.to(dev)
        smm, pmm = sm_mask.to(dev), pm_mask.to(dev)

        @torch.no_grad()
        def run_torch():
            s, z = s_d, z_d
            for b in blocks:
                s, z = b(s, z, smm, pmm, inplace_safe=False)
            return s, z

        results["torch-mps"] = timeit(run_torch, torch.mps.synchronize)

        # torch.compile on MPS (Inductor). May graph-break or fall back; report
        # whatever it actually does after warmup.
        try:
            def torch_stack(s, z):
                for b in blocks:
                    s, z = b(s, z, smm, pmm, inplace_safe=False)
                return s, z

            compiled = torch.compile(torch_stack, backend="inductor", dynamic=False)
            with torch.no_grad():
                results["torch-mps-compiled"] = timeit(
                    lambda: compiled(s_d, z_d), torch.mps.synchronize, warmup=3
                )
        except Exception as e:
            print(f"torch.compile failed: {type(e).__name__}: {str(e)[:200]}")
        del blocks
    else:
        print("MPS not available")

    # ---------- MLX (eager) ----------
    p_stack = subtree(sd, "pairformer_stack")
    s_x, z_x = to_mlx(s0), to_mlx(z0)
    smx, pmx = to_mlx(sm_mask), to_mlx(pm_mask)
    kw = dict(n_blocks=n_blocks, no_heads_pair_bias=CFG["no_heads_pair_bias"],
              no_heads_pair=CFG["no_heads_pair"], inf=CFG["inf"])

    def run_mlx():
        s, z = M.pairformer_stack(s_x, z_x, p_stack, smx, pmx, **kw)
        return s, z

    results["mlx-eager"] = timeit(lambda: mx.eval(*run_mlx()), lambda: None)

    # ---------- MLX (compiled) ----------
    # Compile a single block; call it n_blocks times (weights vary per block so
    # the whole-stack graph is huge — per-block compile is the practical unit).
    stack_p = [subtree(p_stack, f"blocks.{i}") for i in range(n_blocks)]

    @mx.compile
    def cblock(s, z, sm, pm, *flat):
        # flat is unused placeholder; kept simple — recompiles per weight-set are
        # avoided by capturing via closure below instead.
        return s, z

    # Simplest robust "compiled" variant: compile the eager stack function with
    # captured weights (MLX traces the graph once).
    compiled_stack = mx.compile(lambda s, z: M.pairformer_stack(s, z, p_stack, smx, pmx, **kw))
    results["mlx-compiled"] = timeit(lambda: mx.eval(*compiled_stack(s_x, z_x)), lambda: None)

    print(f"\n=== 48-block Pairformer, N={N}, c_s={CFG['c_s']} c_z={CFG['c_z']} ===")
    base = results.get("torch-mps", (None,))[0]
    for name, (med, mn) in results.items():
        spd = f"  ({base/med:.2f}x vs torch-mps)" if base and name != "torch-mps" else ""
        print(f"  {name:14s} median={med*1000:8.1f} ms   min={mn*1000:8.1f} ms{spd}")


if __name__ == "__main__":
    mx.set_default_device(mx.gpu)
    main()

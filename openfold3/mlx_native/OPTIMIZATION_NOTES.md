# MLX Pairformer optimization — what helps and what doesn't

All measured on M3 Max, 48-block Pairformer stack, real checkpoint weights,
torch 2.13 / MLX 0.32. Times are median compiled unless noted.

## The one real win: `mx.compile`
| backend | N=76 |
|---|---|
| torch-MPS eager | 153 ms |
| torch-MPS `torch.compile` | 245 ms (regresses; Inductor-MPS immature) |
| MLX eager | 161 ms |
| **MLX `mx.compile`** | **121 ms (1.27× vs torch-MPS)** |

## What does NOT help (measured, not assumed)

**1. Fused SDPA + fused LayerNorm (`mx.fast.*`)** — 0.94–0.98× compiled (N=76→384).
Helps eager ~8% but `mx.compile` already captures it. The stack is not
attention-bound: per-component at N=256, pair_transition 17 ms and the triangle-
attention *projections* 12 ms×2 dominate; softmax is a sliver (per-head dim 24–32).

**2. cuEquivariance "concat projections" trick** — 0.98× (exact parity).
Concatenating linear_a_g|a_p|b_g|b_p|g into one matmul gives nothing: MLX is not
kernel-launch-bound the way CUDA is. The projection `[65536,128]@[128,640]` runs
at ~1 TFLOP/s on a ~14 TFLOP machine — **memory-bandwidth-bound**, not launch- or
compute-bound.

**3. einsum → explicit batched matmul for the O(N³) contraction** — 1.0× at N≥256.
The contraction is cheap (1–2.5 ms); it's not the bottleneck. The 6 projections are.

**4. Naive low precision** — bf16/fp16 give ~1.5× but destroy accuracy:
full-bf16 rel_z 0.86 vs fp32; fp16 NaNs (1e9 mask overflow). Even bf16-matmul-only
(fp32 everywhere else) is 1.05–1.09× and still rel_z 0.7+ — 48 residual blocks +
the tri_mul contraction compound bf16 error. Real AF3 bf16 inference needs a
co-designed mixed-precision recipe (fp32 residual/LN/softmax, finite mask, and
ideally bf16-robust weights); post-hoc casting of fp32 weights is not enough.

## Why cuEquivariance doesn't transfer
cuEquivariance's win on CUDA is **tiling/fusion to avoid materializing the N²×c
intermediates** (keep tiles in shared memory), plus launch-overhead amortization.
MLX's tuned GEMM + `mx.compile` already handle the launch/elementwise side, and
the remaining cost is DRAM traffic on the N²×c tensors — which only a hand-tiled
fused Metal kernel (`mx.fast.metal_kernel`) that keeps intermediates in
threadgroup memory could reduce. That is a large, uncertain-payoff project.

## Verdict
- Ship `mx.compile` (1.27× over torch-MPS, free).
- Generic fused ops and the cueq concat trick don't help in MLX.
- The two remaining levers are (a) a careful mixed-precision bf16 pass validated
  on real structures, and (b) a hand-tiled fused triangle Metal kernel. Both are
  real projects, not quick wins.

## Addendum: kernel-level analysis of the cuEquivariance approach

cuEquivariance's `triangle_multiplicative_update` requires `hidden_dim % 32 == 0`
=> it tiles over channel-blocks of 32 (warp width), fusing LN + gated projections
+ the O(N^3) contraction + output, keeping tiles in shared memory to avoid
materializing the N^2 x c intermediates.

Tested whether this can transfer to Metal/MLX:

1. Custom Metal contraction kernel (mx.fast.metal_kernel, correct, maxdiff 0):
     N=128  0.42x   N=256  0.13x   N=384  0.05x   vs MLX's batched GEMM.
   The contraction IS a GEMM; MLX runs it on simdgroup_matrix. A naive custom
   kernel is 2-20x slower. To win you must reimplement a tiled simdgroup_matrix
   GEMM and fuse the epilogue.

2. Bandwidth ceiling on fusion (N=256, c=128): the a,b intermediates are 33.5 MB
   each; fusing projection->contraction saves ~134 MB round-trip. At ~400 GB/s
   that's ~0.33 ms on a ~5 ms op => ~6% ceiling, and only if the fused GEMM
   matches MLX's tuned one.

Why it wins on NVIDIA but not Apple:
- Tensor cores make the GEMM fast enough that DRAM/launch overhead dominates ->
  fusion pays. Apple has no exposed tensor-core GEMM at this level, so the work
  is GEMM-bound, not overhead-bound.
- CUDA per-op launch overhead is higher; MLX already batches dispatch.
- Warp register files + 48-228 KB shared mem vs Apple's 32 KB threadgroup mem ->
  smaller tiles, smaller fusion benefit.

Conclusion: the cuEquivariance strategy is a tensor-core/launch-overhead play with
little headroom on Apple silicon. mx.compile (1.27x) is the win; a hand-tiled
fused GEMM+epilogue is weeks of work for a single-digit % ceiling here.

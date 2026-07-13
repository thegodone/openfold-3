"""Core MLX primitives for the OpenFold3 port.

Each function mirrors the corresponding torch primitive in
``openfold3/core/model/primitives`` and ``.../layers`` exactly, so weights load
1:1 and outputs match to float32 precision.
"""

from __future__ import annotations

import mlx.core as mx


# ---------------------------------------------------------------------------
# Elementary primitives
# ---------------------------------------------------------------------------
def linear(x: mx.array, weight: mx.array, bias: mx.array | None = None) -> mx.array:
    """torch ``Linear``: weight is (out, in); computes ``x @ weight.T + bias``."""
    y = x @ weight.T
    if bias is not None:
        y = y + bias
    return y


def layer_norm(
    x: mx.array,
    weight: mx.array | None,
    bias: mx.array | None,
    eps: float = 1e-5,
) -> mx.array:
    """LayerNorm over the last dim, matching torch ``F.layer_norm`` semantics."""
    mu = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    y = (x - mu) * mx.rsqrt(var + eps)
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y


def sigmoid(x: mx.array) -> mx.array:
    return mx.sigmoid(x)


# ---------------------------------------------------------------------------
# Triangle multiplicative update (AF2 Alg. 11/12, AF3 Alg. 12/13)
# ---------------------------------------------------------------------------
def tri_mul(
    z: mx.array,
    p: dict,
    outgoing: bool,
    mask: mx.array | None = None,
    ln_eps: float = 1e-5,
) -> mx.array:
    """Triangle multiplicative update.

    Faithful to ``TriangleMultiplicativeUpdate.forward`` (non-inplace path):

        mask = mask[..., None]
        z    = layer_norm_in(z)
        a    = mask * sigmoid(linear_a_g(z)) * linear_a_p(z)
        b    = mask * sigmoid(linear_b_g(z)) * linear_b_p(z)
        x    = combine(a, b)                # outgoing:  ikc,jkc->ijc
                                            # incoming:  kic,kjc->ijc
        x    = linear_z(layer_norm_out(x))
        g    = sigmoid(linear_g(z))
        return x * g

    ``p`` maps the torch submodule names to MLX arrays:
        layer_norm_in.{weight,bias}, layer_norm_out.{weight,bias},
        linear_a_g.weight, linear_a_p.weight, linear_b_g.weight,
        linear_b_p.weight, linear_z.weight, linear_g.weight
    (these linears are bias-free in the checkpoint).
    """
    if mask is None:
        mask = mx.ones(z.shape[:-1], dtype=z.dtype)
    mask = mask[..., None]

    zn = layer_norm(z, p["layer_norm_in.weight"], p["layer_norm_in.bias"], ln_eps)

    a = mask * sigmoid(linear(zn, p["linear_a_g.weight"])) * linear(zn, p["linear_a_p.weight"])
    b = mask * sigmoid(linear(zn, p["linear_b_g.weight"])) * linear(zn, p["linear_b_p.weight"])

    if outgoing:
        x = mx.einsum("...ikc,...jkc->...ijc", a, b)
    else:
        x = mx.einsum("...kic,...kjc->...ijc", a, b)

    x = layer_norm(x, p["layer_norm_out.weight"], p["layer_norm_out.bias"], ln_eps)
    x = linear(x, p["linear_z.weight"])
    g = sigmoid(linear(zn, p["linear_g.weight"]))
    return x * g

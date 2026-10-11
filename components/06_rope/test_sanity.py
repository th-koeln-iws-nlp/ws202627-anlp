"""Limit-case checks. Passing both this and test_shapes.py is
necessary, not sufficient, for your implementation to be correct."""

import torch

from starter import apply_rope, precompute_rope


def test_rope_preserves_norm():
    """A rotation cannot change a vector's length: applying RoPE at
    any position must leave each (query or key) vector's L2 norm
    unchanged."""
    torch.manual_seed(0)
    head_dim = 8
    context_length = 20
    x = torch.randn(1, 1, context_length, head_dim)
    cos, sin = precompute_rope(head_dim, context_length)
    rotated = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), rotated.norm(dim=-1), atol=1e-5)


def test_position_zero_is_identity():
    """At position 0 every rotation angle is 0 (cos=1, sin=0), so
    RoPE must leave position 0's vector unchanged."""
    head_dim = 8
    context_length = 5
    x = torch.randn(1, 1, context_length, head_dim)
    cos, sin = precompute_rope(head_dim, context_length)
    rotated = apply_rope(x, cos, sin)
    assert torch.allclose(rotated[:, :, 0, :], x[:, :, 0, :], atol=1e-6)


def test_dot_product_depends_only_on_relative_position():
    """The defining property of RoPE: the dot product of a rotated
    query at position i and a rotated key at position j depends only
    on i - j, not on i and j separately. Rotate the same underlying
    query/key pair at two different absolute positions that share the
    same offset, and check the dot products agree."""
    torch.manual_seed(0)
    head_dim = 8
    context_length = 200
    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)
    cos, sin = precompute_rope(head_dim, context_length)

    def rotated_dot(pos_q, pos_k):
        rq = apply_rope(q, cos[pos_q : pos_q + 1], sin[pos_q : pos_q + 1])
        rk = apply_rope(k, cos[pos_k : pos_k + 1], sin[pos_k : pos_k + 1])
        return (rq * rk).sum()

    dot_near = rotated_dot(5, 3)  # offset 2
    dot_far = rotated_dot(105, 103)  # offset 2, far from position 0
    assert torch.allclose(dot_near, dot_far, atol=1e-3)

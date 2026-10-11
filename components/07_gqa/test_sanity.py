"""Limit-case checks. Passing both this and test_shapes.py is
necessary, not sufficient, for your implementation to be correct."""

import torch

from starter import GroupedQueryAttention, MultiHeadAttention


def test_num_kv_heads_equal_to_num_heads_matches_baseline():
    """num_kv_heads == num_heads means group_size == 1: repeat_interleave
    by 1 is the identity, so GQA must reduce to ordinary multi-head
    attention exactly, given the same weights."""
    torch.manual_seed(0)
    baseline = MultiHeadAttention(d_in=16, d_out=16, context_length=10, dropout=0.0, num_heads=4)
    gqa = GroupedQueryAttention(d_in=16, d_out=16, context_length=10, dropout=0.0, num_heads=4, num_kv_heads=4)
    gqa.load_state_dict(baseline.state_dict())  # W_key/W_value are the same shape when num_kv_heads == num_heads

    x = torch.randn(2, 10, 16)
    assert torch.allclose(baseline(x), gqa(x), atol=1e-6)


def test_num_kv_heads_one_shares_a_single_key_value_head():
    """With num_kv_heads=1, every query head must attend using the
    exact same key and value projection: only one KV head exists to
    repeat_interleave across all of them."""
    att = GroupedQueryAttention(d_in=16, d_out=16, context_length=10, dropout=0.0, num_heads=4, num_kv_heads=1)
    assert att.W_key.out_features == att.head_dim
    assert att.W_value.out_features == att.head_dim

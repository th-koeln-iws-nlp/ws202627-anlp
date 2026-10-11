"""Shape-only: confirms the code is wired up correctly, not that it is
correct. Run this first."""

import torch

from starter import GroupedQueryAttention, ablation_config, build_model


def test_gqa_shape():
    att = GroupedQueryAttention(d_in=16, d_out=16, context_length=10, dropout=0.0, num_heads=4, num_kv_heads=2)
    x = torch.randn(2, 10, 16)
    assert att(x).shape == x.shape


def test_gqa_shape_num_kv_heads_one():
    att = GroupedQueryAttention(d_in=16, d_out=16, context_length=10, dropout=0.0, num_heads=4, num_kv_heads=1)
    x = torch.randn(2, 10, 16)
    assert att(x).shape == x.shape


def test_gqa_key_value_projection_shrinks():
    att = GroupedQueryAttention(d_in=16, d_out=16, context_length=10, dropout=0.0, num_heads=4, num_kv_heads=2)
    assert att.W_key.out_features == 2 * (16 // 4)  # num_kv_heads * head_dim
    assert att.W_query.out_features == 16  # unchanged


def test_build_model_shape():
    cfg = {**ablation_config(), "vocab_size": 100, "n_layers": 2}  # smaller, for a fast test
    model = build_model(cfg, num_kv_heads=2)
    ids = torch.randint(0, 100, (2, 10))
    assert model(ids).shape == (2, 10, 100)

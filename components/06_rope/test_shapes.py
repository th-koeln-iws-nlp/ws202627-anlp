"""Shape-only: confirms the code is wired up correctly, not that it is
correct. Run this first."""

import torch

from starter import (
    RotaryMultiHeadAttention,
    ZeroPositionalEmbedding,
    ablation_config,
    apply_rope,
    build_model,
    precompute_rope,
    rotate_half,
)


def test_precompute_rope_shape():
    cos, sin = precompute_rope(head_dim=8, context_length=10)
    assert cos.shape == (10, 8)
    assert sin.shape == (10, 8)


def test_rotate_half_shape():
    x = torch.randn(2, 4, 10, 8)  # (b, num_heads, num_tokens, head_dim)
    assert rotate_half(x).shape == x.shape


def test_apply_rope_shape():
    x = torch.randn(2, 4, 10, 8)
    cos, sin = precompute_rope(head_dim=8, context_length=10)
    assert apply_rope(x, cos, sin).shape == x.shape


def test_rotary_attention_shape():
    att = RotaryMultiHeadAttention(d_in=16, d_out=16, context_length=10, dropout=0.0, num_heads=2)
    x = torch.randn(2, 10, 16)
    assert att(x).shape == x.shape


def test_zero_positional_embedding_shape():
    zero_pos = ZeroPositionalEmbedding(emb_dim=16)
    position_ids = torch.arange(10)
    out = zero_pos(position_ids)
    assert out.shape == (10, 16)
    assert torch.equal(out, torch.zeros(10, 16))


def test_build_model_shape():
    cfg = {**ablation_config(), "vocab_size": 100, "n_layers": 2}  # smaller, for a fast test
    model = build_model(cfg)
    ids = torch.randint(0, 100, (2, 10))
    assert model(ids).shape == (2, 10, 100)

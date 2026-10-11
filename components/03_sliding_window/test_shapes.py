"""Shape-only: confirms the code is wired up correctly, not that it is
correct. Run this first."""

import torch

from starter import SlidingWindowAttention, ablation_config, build_model


def test_sliding_window_attention_shape():
    att = SlidingWindowAttention(
        d_in=16, d_out=16, context_length=10, dropout=0.0, num_heads=2, window_size=4
    )
    x = torch.randn(2, 10, 16)
    assert att(x).shape == x.shape


def test_sliding_window_attention_shape_with_padding_mask():
    att = SlidingWindowAttention(
        d_in=16, d_out=16, context_length=10, dropout=0.0, num_heads=2, window_size=4
    )
    x = torch.randn(2, 10, 16)
    attn_mask = torch.ones(2, 10)
    attn_mask[:, :3] = 0  # left-padded
    assert att(x, attn_mask=attn_mask).shape == x.shape


def test_build_model_shape():
    cfg = {**ablation_config(), "vocab_size": 100, "n_layers": 2}  # smaller, for a fast test
    model = build_model(cfg, window_size=32)
    ids = torch.randint(0, 100, (2, 10))
    assert model(ids).shape == (2, 10, 100)

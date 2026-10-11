"""Shape-only: confirms the code is wired up correctly, not that it is
correct. Run this first."""

import torch

from starter import SinusoidalPositionalEmbedding, ablation_config, build_model


def test_sinusoidal_embedding_shape():
    pe = SinusoidalPositionalEmbedding(context_length=20, emb_dim=16)
    position_ids = torch.arange(10)
    assert pe(position_ids).shape == (10, 16)


def test_build_model_shape():
    cfg = {**ablation_config(), "vocab_size": 100, "n_layers": 2}  # smaller, for a fast test
    model = build_model(cfg)
    ids = torch.randint(0, 100, (2, 10))
    assert model(ids).shape == (2, 10, 100)

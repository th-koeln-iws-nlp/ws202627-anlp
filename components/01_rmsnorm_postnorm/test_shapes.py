"""Shape-only: confirms the code is wired up correctly, not that it is
correct. Run this first."""

import torch

from starter import (
    PostNormTransformerBlock,
    RMSNorm,
    ablation_config,
    build_model_a,
    build_model_b,
)


def test_rmsnorm_shape():
    norm = RMSNorm(32)
    x = torch.randn(2, 5, 32)
    assert norm(x).shape == x.shape


def test_postnorm_block_shape():
    cfg = {"emb_dim": 32, "context_length": 16, "n_heads": 4, "drop_rate": 0.0, "qkv_bias": False}
    block = PostNormTransformerBlock(cfg)
    x = torch.randn(2, 5, 32)
    assert block(x).shape == x.shape


def test_build_model_a_shape():
    cfg = {**ablation_config(), "vocab_size": 100, "n_layers": 2}  # smaller vocab and depth, for a fast test
    model = build_model_a(cfg)
    ids = torch.randint(0, 100, (2, 10))
    assert model(ids).shape == (2, 10, 100)


def test_build_model_b_shape():
    cfg = {**ablation_config(), "vocab_size": 100, "n_layers": 2}
    model = build_model_b(cfg)
    ids = torch.randint(0, 100, (2, 10))
    assert model(ids).shape == (2, 10, 100)

"""Shape-only: confirms the code is wired up correctly, not that it is
correct. Run this first."""

import torch

from starter import MATCHED_HIDDEN_DIM, SwiGLUFeedForward, ablation_config, build_model


def test_swiglu_ff_shape():
    cfg = ablation_config()
    ff = SwiGLUFeedForward(cfg, hidden_dim=MATCHED_HIDDEN_DIM)
    x = torch.randn(2, 5, cfg["emb_dim"])
    assert ff(x).shape == x.shape


def test_build_model_shape():
    cfg = {**ablation_config(), "vocab_size": 100, "n_layers": 2}  # smaller, for a fast test
    model = build_model(cfg)
    ids = torch.randint(0, 100, (2, 10))
    assert model(ids).shape == (2, 10, 100)

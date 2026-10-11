"""Shape-only: confirms the code is wired up correctly, not that it is
correct. Run this first."""

import torch

from starter import (
    InstrumentedMultiHeadAttention,
    QKNormAttention,
    ablation_config,
    build_model,
)


def test_qk_norm_attention_shape():
    att = QKNormAttention(d_in=16, d_out=16, context_length=10, dropout=0.0, num_heads=2)
    x = torch.randn(2, 10, 16)
    assert att(x).shape == x.shape
    assert isinstance(att.last_max_logit, float)


def test_instrumented_attention_shape():
    att = InstrumentedMultiHeadAttention(d_in=16, d_out=16, context_length=10, dropout=0.0, num_heads=2)
    x = torch.randn(2, 10, 16)
    assert att(x).shape == x.shape
    assert isinstance(att.last_max_logit, float)


def test_build_model_shape():
    cfg = {**ablation_config(), "vocab_size": 100, "n_layers": 2}  # smaller, for a fast test
    model = build_model(cfg)
    ids = torch.randint(0, 100, (2, 10))
    assert model(ids).shape == (2, 10, 100)

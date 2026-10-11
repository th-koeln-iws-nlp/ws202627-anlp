"""Limit-case checks. Passing both this and test_shapes.py is
necessary, not sufficient, for your implementation to be correct."""

import torch

from starter import InstrumentedMultiHeadAttention, MultiHeadAttention, QKNormAttention


def test_qk_norm_bounds_attention_logits():
    """With unit-RMS queries and keys, the dot product of two head_dim
    vectors is bounded by head_dim, so scaling by 1/sqrt(head_dim)
    bounds the logit by sqrt(head_dim), regardless of how large the
    raw Q/K projections are. Feed in a deliberately huge input and
    check the logit stays small."""
    torch.manual_seed(0)
    att = QKNormAttention(d_in=8, d_out=8, context_length=6, dropout=0.0, num_heads=2)
    x = torch.randn(1, 6, 8) * 1000  # deliberately huge, to try to blow up logits
    att(x)
    max_logit = att.last_max_logit
    assert isinstance(max_logit, float)
    head_dim = 4
    assert abs(max_logit) < head_dim**0.5 + 1e-3


def test_instrumented_attention_matches_baseline_exactly():
    """InstrumentedMultiHeadAttention must be a numerically identical
    drop-in for the baseline: same parameter names (so a checkpoint's
    state_dict loads with strict=True), same output given the same
    weights and input."""
    torch.manual_seed(0)
    baseline = MultiHeadAttention(d_in=8, d_out=8, context_length=6, dropout=0.0, num_heads=2)
    instrumented = InstrumentedMultiHeadAttention(d_in=8, d_out=8, context_length=6, dropout=0.0, num_heads=2)
    instrumented.load_state_dict(baseline.state_dict())  # strict=True by default

    x = torch.randn(2, 6, 8)
    assert torch.equal(baseline(x), instrumented(x))
    assert isinstance(instrumented.last_max_logit, float)

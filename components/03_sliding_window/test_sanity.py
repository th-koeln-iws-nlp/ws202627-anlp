"""Limit-case checks. Passing both this and test_shapes.py is
necessary, not sufficient, for your implementation to be correct."""

import torch

from starter import MultiHeadAttention, SlidingWindowAttention


def test_window_size_one_attends_only_to_self():
    """With window_size=1, a query can only see a key j with j <= i
    (causal) and i - j < 1 (window), which together force j == i: the
    mask must block every off-diagonal entry."""
    att = SlidingWindowAttention(
        d_in=8, d_out=8, context_length=6, dropout=0.0, num_heads=2, window_size=1
    )
    off_diagonal = ~torch.eye(6, dtype=torch.bool)
    assert att.mask[off_diagonal].all(), "every off-diagonal entry must be masked"
    assert not att.mask.diagonal().any(), "the diagonal itself must not be masked"


def test_window_size_at_least_context_length_matches_full_causal():
    """A window at least as wide as the context sees the full causal
    history, so with identical weights, SlidingWindowAttention must
    match the baseline MultiHeadAttention exactly."""
    torch.manual_seed(0)
    context_length = 6
    baseline = MultiHeadAttention(d_in=8, d_out=8, context_length=context_length, dropout=0.0, num_heads=2)
    windowed = SlidingWindowAttention(
        d_in=8, d_out=8, context_length=context_length, dropout=0.0, num_heads=2, window_size=context_length
    )
    windowed.load_state_dict(baseline.state_dict())

    x = torch.randn(2, context_length, 8)
    assert torch.allclose(baseline(x), windowed(x), atol=1e-6)

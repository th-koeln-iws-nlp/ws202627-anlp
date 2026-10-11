"""Limit-case checks. Passing both this and test_shapes.py is
necessary, not sufficient, for your implementation to be correct."""

import torch

from starter import MATCHED_HIDDEN_DIM, FeedForward, SwiGLUFeedForward, ablation_config


def test_param_count_matches_baseline():
    """Parameter count within 1% of the baseline feed-forward, at the
    matched hidden_dim."""
    cfg = ablation_config()
    baseline_params = sum(p.numel() for p in FeedForward(cfg).parameters())
    swiglu_params = sum(p.numel() for p in SwiGLUFeedForward(cfg, hidden_dim=MATCHED_HIDDEN_DIM).parameters())
    diff_fraction = abs(swiglu_params - baseline_params) / baseline_params
    assert diff_fraction < 0.01



def test_closed_gate_blocks_the_output():
    """With the gate branch zeroed (weights and bias), nothing may pass through: the output
    is only the down projection's bias, whatever the input."""
    cfg = ablation_config()
    ff = SwiGLUFeedForward(cfg, hidden_dim=MATCHED_HIDDEN_DIM)
    with torch.no_grad():
        ff.gate.weight.zero_()
        ff.gate.bias.zero_()
    x = torch.randn(2, 5, cfg["emb_dim"])
    assert torch.allclose(ff(x), ff.down.bias.expand(2, 5, -1), atol=1e-6)


def test_zeroed_up_branch_blocks_the_output():
    """Both branches must take part: with the up branch zeroed, the output is again only
    the down projection's bias."""
    cfg = ablation_config()
    ff = SwiGLUFeedForward(cfg, hidden_dim=MATCHED_HIDDEN_DIM)
    with torch.no_grad():
        ff.up.weight.zero_()
        ff.up.bias.zero_()
    x = torch.randn(2, 5, cfg["emb_dim"])
    assert torch.allclose(ff(x), ff.down.bias.expand(2, 5, -1), atol=1e-6)

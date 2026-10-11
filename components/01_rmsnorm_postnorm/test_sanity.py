"""Limit-case checks. Passing both this and test_shapes.py is
necessary, not sufficient, for your implementation to be correct."""

import torch

from starter import PostNormTransformerBlock, RMSNorm, TransformerBlock


def test_rmsnorm_output_has_unit_rms():
    """RMSNorm output has root mean square 1 per position when the
    weight is ones, the default init."""
    norm = RMSNorm(32)
    x = torch.randn(4, 6, 32) * 10  # arbitrary scale, should not matter
    out = norm(x)
    rms = out.pow(2).mean(dim=-1).sqrt()  # (4, 6)
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_postnorm_block_matches_baseline_shape():
    """The post-norm block returns the same shape as the baseline
    (pre-norm) block."""
    cfg = {"emb_dim": 32, "context_length": 16, "n_heads": 4, "drop_rate": 0.0, "qkv_bias": False}
    post_block = PostNormTransformerBlock(cfg)
    pre_block = TransformerBlock(cfg)
    x = torch.randn(2, 5, 32)
    assert post_block(x).shape == pre_block(x).shape


def test_postnorm_block_output_is_normalised():
    """Post-norm ends every sublayer with a LayerNorm, so with its default scale and shift
    every output vector of the block has mean 0 and variance 1, whatever the input scale."""
    torch.manual_seed(0)
    cfg = {"emb_dim": 32, "context_length": 16, "n_heads": 4, "drop_rate": 0.0, "qkv_bias": False}
    block = PostNormTransformerBlock(cfg)
    out = block(torch.randn(2, 5, 32) * 10)  # (2, 5, 32)
    assert torch.allclose(out.mean(dim=-1), torch.zeros(2, 5), atol=1e-4)
    assert torch.allclose(out.var(dim=-1, unbiased=False), torch.ones(2, 5), atol=1e-2)

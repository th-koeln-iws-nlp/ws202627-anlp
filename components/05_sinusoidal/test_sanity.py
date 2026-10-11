"""Limit-case checks. Passing both this and test_shapes.py is
necessary, not sufficient, for your implementation to be correct."""

import torch

from starter import SinusoidalPositionalEmbedding


def test_pe_is_a_buffer_not_a_parameter():
    """Nothing here is learned: pe must not show up in
    model.parameters(), only in model.buffers()."""
    pe_module = SinusoidalPositionalEmbedding(context_length=20, emb_dim=16)
    assert list(pe_module.parameters()) == []
    buffer_names = [name for name, _ in pe_module.named_buffers()]
    assert "pe" in buffer_names


def test_position_zero_is_sin_zero_cos_zero():
    """At position 0, every sin term is sin(0) = 0 and every cos term
    is cos(0) = 1, regardless of frequency."""
    pe_module = SinusoidalPositionalEmbedding(context_length=20, emb_dim=16)
    row0 = pe_module(torch.tensor([0]))[0]
    assert torch.allclose(row0[0::2], torch.zeros(8), atol=1e-6)
    assert torch.allclose(row0[1::2], torch.ones(8), atol=1e-6)


def test_values_are_bounded():
    pe_module = SinusoidalPositionalEmbedding(context_length=512, emb_dim=256)
    assert pe_module.pe.abs().max() <= 1.0 + 1e-6


def test_first_pair_is_sin_and_cos_of_the_position():
    """The fastest pair of dimensions has a wavelength of 2 pi: dimension 0 is sin(pos),
    dimension 1 is cos(pos)."""
    positions = torch.arange(20)
    out = SinusoidalPositionalEmbedding(context_length=20, emb_dim=16)(positions)  # (20, 16)
    assert torch.allclose(out[:, 0], torch.sin(positions.float()), atol=1e-5)
    assert torch.allclose(out[:, 1], torch.cos(positions.float()), atol=1e-5)


def test_frequencies_fall_from_pair_to_pair():
    """Later pairs of dimensions oscillate more slowly. At position 1 every angle is below one
    radian, so the sine values must fall strictly from the first pair to the last."""
    out = SinusoidalPositionalEmbedding(context_length=20, emb_dim=16)(torch.tensor([1]))[0]  # (16,)
    sines = out[0::2]
    assert torch.all(sines[:-1] > sines[1:])

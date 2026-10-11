"""Component 06: RoPE, rotary position embeddings.

Injects position information directly into attention, by rotating
queries and keys, instead of adding a position vector at the embedding
layer.
"""

import torch
import torch.nn as nn

from ablation_harness import ablation_config
from baseline import GPTModel, MultiHeadAttention


def precompute_rope(head_dim, context_length, theta_base=10000.0):
    """cos and sin tables, each (context_length, head_dim).

    Uses the rotate-half convention (GPT-NeoX, Llama): dimension i and
    dimension i + head_dim/2 are paired and rotated together, not
    dimension 2i and 2i+1 as in the original RoPE paper's presentation.
    precompute_rope and apply_rope must agree on this, always use them
    as a pair. Mixing rotate-half angles with interleaved-pairs
    rotation (or vice versa) does not crash, it silently computes a
    wrong rotation, the classic RoPE bug.

    The pair (i, i + head_dim/2) at position pos is rotated by the
    angle pos * theta_i, with theta_i = theta_base^(-2i / head_dim) for
    i = 0, ..., head_dim/2 - 1. Both halves of the table therefore hold
    the same angles.
    """
    raise NotImplementedError("implement precompute_rope")


def rotate_half(x):
    """The helper behind the rotate-half convention: split the last
    dimension into two halves (x1, x2), and return the halves swapped,
    with the new first half negated: (-x2, x1)."""
    raise NotImplementedError("implement rotate_half")


def apply_rope(x, cos, sin):
    """x: (b, num_heads, num_tokens, head_dim). cos, sin: (num_tokens,
    head_dim), broadcast over batch and heads.

    Rotate every vector by its position's angle. Think of each
    dimension pair as a point in 2D, and use cos, sin and rotate_half."""
    raise NotImplementedError("implement apply_rope")


class RotaryMultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False, theta_base=10000.0):
        # The same submodules as the baseline MultiHeadAttention, plus
        # the cos and sin tables from precompute_rope, computed once
        # and registered as buffers ("cos" and "sin") so they move with
        # .to(device) but are never trained.
        raise NotImplementedError("implement RotaryMultiHeadAttention.__init__")

    def forward(self, x, attn_mask=None):
        # x: (b, num_tokens, d_in)
        #
        # The baseline's forward pass, with queries and keys rotated by
        # their positions before they are multiplied (values are not
        # rotated). The tables cover context_length positions, this
        # batch may be shorter. Masking and scaling stay as in the
        # baseline.
        raise NotImplementedError("implement RotaryMultiHeadAttention.forward")


class ZeroPositionalEmbedding(nn.Module):
    """A drop-in stand-in for GPTModel.pos_emb that always returns
    zeros. RoPE injects position information inside attention
    directly, not by adding a position vector to the token embedding,
    so that addition in GPTModel.forward must contribute nothing."""

    def __init__(self, emb_dim):
        # Store emb_dim. No parameters, no buffers, nothing learned.
        raise NotImplementedError("implement ZeroPositionalEmbedding.__init__")

    def forward(self, position_ids):
        # Return a (len(position_ids), emb_dim) tensor of zeros, on
        # the same device as position_ids (so it works whether
        # position_ids is on CPU or GPU).
        raise NotImplementedError("implement ZeroPositionalEmbedding.forward")


def build_model(cfg):
    """Baseline GPTModel with pos_emb zeroed out and every attention
    replaced by RotaryMultiHeadAttention."""
    model = GPTModel(cfg)
    model.pos_emb = ZeroPositionalEmbedding(cfg["emb_dim"])
    for block in model.trf_blocks:
        block.att = RotaryMultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
    return model


# The variants the ablation notebook can run: label -> build_model_fn(cfg).
VARIANTS = {"rope": build_model}

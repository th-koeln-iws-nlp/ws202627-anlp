"""Component 05: sinusoidal positional embeddings.

Replaces the baseline's learned position embedding table with a fixed,
non-learned function of position.
"""

import math

import torch
import torch.nn as nn

from ablation_harness import ablation_config
from baseline import GPTModel


class SinusoidalPositionalEmbedding(nn.Module):
    """A fixed (context_length, emb_dim) table, computed once and
    registered as a buffer, not a parameter: nothing here is learned.
    Called exactly like the baseline's nn.Embedding(context_length,
    emb_dim): forward(position_ids) with position_ids a 1D tensor of
    integer positions, returns (len(position_ids), emb_dim)."""

    def __init__(self, context_length, emb_dim):
        # Build the (context_length, emb_dim) table once, here, and
        # register it as a buffer named "pe": it is not a parameter, it
        # must never get a gradient or show up in model.parameters().
        #
        # The definition (Vaswani et al. 2017, section 3.5), for
        # position pos and dimension pair (2i, 2i + 1):
        #   pe[pos, 2i]     = sin(pos / 10000^(2i / emb_dim))
        #   pe[pos, 2i + 1] = cos(pos / 10000^(2i / emb_dim))
        # Computing the frequencies in log space avoids huge powers.
        raise NotImplementedError("implement SinusoidalPositionalEmbedding.__init__")

    def forward(self, position_ids):
        # position_ids: a 1D tensor of integer positions, exactly what
        # GPTModel.forward passes to self.pos_emb. Return the table rows
        # for those positions, (len(position_ids), emb_dim).
        raise NotImplementedError("implement SinusoidalPositionalEmbedding.forward")


def build_model(cfg):
    """Baseline GPTModel with pos_emb replaced by
    SinusoidalPositionalEmbedding. GPTModel.forward calls
    self.pos_emb(torch.arange(seq_len, ...)) exactly like an
    nn.Embedding, so no other change is needed."""
    model = GPTModel(cfg)
    model.pos_emb = SinusoidalPositionalEmbedding(cfg["context_length"], cfg["emb_dim"])
    return model


# The variants the ablation notebook can run: label -> build_model_fn(cfg).
VARIANTS = {"sinusoidal": build_model}

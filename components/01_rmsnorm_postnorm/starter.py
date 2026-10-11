"""Component 01: RMSNorm and norm placement.

Two independent variants against the same baseline, never combined.
Variant A replaces every LayerNorm with RMSNorm. Variant B keeps
LayerNorm but moves it from pre-norm to post-norm placement.
"""

import torch
import torch.nn as nn

from ablation_harness import ablation_config
from baseline import (
    FeedForward,
    GPTModel,
    LayerNorm,
    MultiHeadAttention,
    TransformerBlock,
)


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        # Keep eps, and one learnable scale per dimension, initialised
        # to ones. RMSNorm has no shift parameter.
        raise NotImplementedError("implement RMSNorm.__init__")

    def forward(self, x):
        # x: (b, num_tokens, emb_dim)
        #
        # Rescale every position's vector by its own root mean square
        # over the last dimension (eps keeps it away from zero), then
        # apply the learnable scale. Unlike LayerNorm, the mean is not
        # subtracted. The output has the same shape as x.
        raise NotImplementedError("implement RMSNorm.forward")


class PostNormTransformerBlock(nn.Module):
    def __init__(self, cfg):
        # The same submodules as the baseline TransformerBlock, under
        # the same attribute names (your day 1 TransformerBlock):
        # attention, feed-forward, two LayerNorms and the shortcut
        # dropout. Nothing here is new, only forward() changes.
        raise NotImplementedError("implement PostNormTransformerBlock.__init__")

    def forward(self, x, attn_mask=None):
        # The baseline normalises first, runs the sublayer, then adds
        # the shortcut. Post-norm changes the order: the sublayer sees
        # the un-normalised input, the shortcut is added to its
        # output, and the sum is normalised. Do this for both
        # sublayers, attention first. Same signature as the baseline.
        raise NotImplementedError("implement PostNormTransformerBlock.forward")


def build_model_a(cfg):
    """Baseline GPTModel with every LayerNorm replaced by RMSNorm: norm1
    and norm2 in every block, and final_norm. Same eps as the baseline
    (1e-5), so only the normalization itself differs."""
    model = GPTModel(cfg)
    for block in model.trf_blocks:
        block.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-5)
        block.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-5)
    model.final_norm = RMSNorm(cfg["emb_dim"], eps=1e-5)
    return model


def build_model_b(cfg):
    """Baseline GPTModel with every TransformerBlock replaced by
    PostNormTransformerBlock. final_norm stays LayerNorm, unchanged."""
    model = GPTModel(cfg)
    model.trf_blocks = nn.ModuleList([PostNormTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
    return model


# The variants the ablation notebook can run: label -> build_model_fn(cfg).
VARIANTS = {"A_rmsnorm": build_model_a, "B_postnorm": build_model_b}

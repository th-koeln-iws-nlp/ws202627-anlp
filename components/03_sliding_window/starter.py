"""Component 03: sliding-window attention.

Restricts each query to a local window of keys instead of the full
causal history.
"""

from functools import partial

import torch
import torch.nn as nn

from ablation_harness import ablation_config
from baseline import GPTModel, MultiHeadAttention


class SlidingWindowAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, window_size, qkv_bias=False):
        # The same submodules as the baseline MultiHeadAttention
        # (W_query, W_key, W_value, out_proj, dropout), plus window_size.
        # Build the (context_length, context_length) boolean mask as a
        # buffer named `mask` (the tests read it), True wherever
        # attention must be blocked. A query may see itself and the
        # window_size - 1 positions before it, nothing in the future
        # and nothing older.
        raise NotImplementedError("implement SlidingWindowAttention.__init__")

    def forward(self, x, attn_mask=None):
        # x: (b, num_tokens, d_in)
        #
        # The baseline MultiHeadAttention.forward, except that it uses
        # the windowed mask. Keep the padding-mask handling, including
        # the fully masked row edge case: a padding query can now be
        # blocked by causality, window and padding all at once.
        raise NotImplementedError("implement SlidingWindowAttention.forward")


def build_model(cfg, window_size):
    """Baseline GPTModel with every block's attention replaced by
    SlidingWindowAttention at the given window_size. Same d_in/d_out/
    num_heads/dropout/qkv_bias as the baseline, window_size is the only
    new argument."""
    model = GPTModel(cfg)
    for block in model.trf_blocks:
        block.att = SlidingWindowAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            window_size=window_size,
            qkv_bias=cfg["qkv_bias"],
        )
    return model


# The variants the ablation notebook can run: label -> build_model_fn(cfg).
VARIANTS = {f"window{size}": partial(build_model, window_size=size) for size in (64, 128, 256)}

"""Component 07: grouped-query attention (GQA).

Shares each key/value head across a group of query heads, instead of
giving every query head its own key/value head.
"""

from functools import partial

import torch
import torch.nn as nn

from ablation_harness import ablation_config
from baseline import GPTModel, MultiHeadAttention


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, num_kv_heads, qkv_bias=False):
        # The same submodules and attribute names as the baseline
        # MultiHeadAttention (the tests read head_dim, W_query and
        # W_key), with one change: W_key and W_value project to
        # num_kv_heads * head_dim instead of num_heads * head_dim.
        # That shrink is the entire parameter saving. Also keep how
        # many query heads share one KV head, and check that
        # num_heads is divisible by num_kv_heads.
        raise NotImplementedError("implement GroupedQueryAttention.__init__")

    def forward(self, x, attn_mask=None):
        # x: (b, num_tokens, d_in)
        #
        # The baseline's forward pass, except that keys and values have
        # num_kv_heads heads while queries have num_heads. Before the
        # dot product, expand keys and values so that they line up with
        # the queries: query head h must read KV head h // group_size,
        # where group_size is the number of query heads per KV head.
        # Mind the order in which you repeat. Everything after that is
        # the baseline.
        raise NotImplementedError("implement GroupedQueryAttention.forward")


def build_model(cfg, num_kv_heads):
    """Baseline GPTModel with every attention replaced by
    GroupedQueryAttention at the given num_kv_heads. num_kv_heads ==
    num_heads (8, at the ablation config's n_heads) recovers ordinary
    multi-head attention exactly."""
    model = GPTModel(cfg)
    for block in model.trf_blocks:
        block.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            num_kv_heads=num_kv_heads,
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
    return model


# The variants the ablation notebook can run: label -> build_model_fn(cfg).
VARIANTS = {f"kv{n}": partial(build_model, num_kv_heads=n) for n in (4, 2, 1)}

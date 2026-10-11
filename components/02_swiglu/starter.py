"""Component 02: SwiGLU feed-forward.

Replaces the baseline two-matrix feed-forward with a gated, three-matrix
one, at matched parameter count.
"""

from functools import partial

import torch.nn as nn

from ablation_harness import ablation_config
from baseline import FeedForward, GPTModel

# For d = 256 (the ablation config's emb_dim): 8 * 256 / 3 = 682.7,
# rounded to the nearest multiple of 8, 680. This is the recipe from the
# SwiGLU paper and from Llama: without it, SwiGLU's three matrices at the
# baseline's own hidden size (4d) would just be a bigger model, and any
# effect would be a parameter-count effect, not a gating-structure effect.
MATCHED_HIDDEN_DIM = 680


class SwiGLUFeedForward(nn.Module):
    def __init__(self, cfg, hidden_dim):
        # Three linear layers, all with bias (the default for
        # nn.Linear, same as the baseline FeedForward): gate and up,
        # both cfg["emb_dim"] -> hidden_dim, and down, hidden_dim ->
        # cfg["emb_dim"].
        raise NotImplementedError("implement SwiGLUFeedForward.__init__")

    def forward(self, x):
        # x: (b, num_tokens, emb_dim)
        #
        # The gate branch goes through SiLU (torch.nn.functional.silu)
        # and then controls, elementwise, how much of the up branch
        # passes through. Project the result back down to emb_dim.
        # Both branches are (b, num_tokens, hidden_dim).
        raise NotImplementedError("implement SwiGLUFeedForward.forward")


def build_model(cfg, hidden_dim=MATCHED_HIDDEN_DIM):
    """Baseline GPTModel with every FeedForward replaced by
    SwiGLUFeedForward. hidden_dim defaults to the parameter-matched
    680, pass a different value only for the optional unmatched-
    comparison run."""
    model = GPTModel(cfg)
    for block in model.trf_blocks:
        block.ff = SwiGLUFeedForward(cfg, hidden_dim=hidden_dim)
    return model


# The variants the ablation notebook can run: label -> build_model_fn(cfg).
VARIANTS = {
    "swiglu": build_model,
    "swiglu_1024": partial(build_model, hidden_dim=1024),  # optional, unmatched
}

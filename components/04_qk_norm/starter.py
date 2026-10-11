"""Component 04: QK-Norm.

Normalises queries and keys per head, before the dot product, so the
scale of Q and K cannot grow unboundedly and blow up attention logits.
"""

import torch
import torch.nn as nn

from ablation_harness import ablation_config
from baseline import GPTModel, MultiHeadAttention


class RMSNorm(nn.RMSNorm):
    """Given complete: PyTorch's RMSNorm over the last dimension, with a learnable scale.
    Writing an RMSNorm by hand is component 01's job, here it is only a building block."""

    def __init__(self, dim, eps=1e-6):
        super().__init__(dim, eps=eps)


class QKNormAttention(nn.Module):
    """Same as the baseline MultiHeadAttention, with one addition:
    queries and keys are each normalised by their own RMSNorm(head_dim),
    per head, after splitting into heads and before the dot product.
    Note: the original QK-Norm paper (Henry et al. 2020) also learns a
    scalar temperature to replace the fixed 1/sqrt(head_dim) scaling.
    We do not implement that here, only the normalisation itself, to
    isolate its effect."""

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        # The same submodules as the baseline MultiHeadAttention,
        # plus one RMSNorm over head_dim for the queries (q_norm) and
        # one for the keys (k_norm). Also self.last_max_logit = None,
        # it is written on every forward call.
        raise NotImplementedError("implement QKNormAttention.__init__")

    def forward(self, x, attn_mask=None):
        # x: (b, num_tokens, d_in)
        #
        # The baseline's forward pass, with queries and keys normalised
        # per head before they are multiplied. Masking and scaling stay
        # as in the baseline.
        #
        # Before the softmax, set self.last_max_logit to the largest
        # scaled, masked attention logit of this forward pass (a Python
        # float, ignore the -inf entries): it is this component's
        # secondary metric.
        raise NotImplementedError("implement QKNormAttention.forward")


class InstrumentedMultiHeadAttention(nn.Module):
    """Numerically identical to the baseline MultiHeadAttention (same
    parameter names, so a trained baseline checkpoint's state_dict
    loads onto it with strict=True), with one addition: records the
    largest scaled attention logit seen on the last forward pass, so
    you can measure logit growth on the baseline itself without ever
    touching 02_attention.py."""

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        # Exactly the baseline MultiHeadAttention's __init__ (same
        # five submodule names, same mask buffer), plus
        # self.last_max_logit = None.
        raise NotImplementedError("implement InstrumentedMultiHeadAttention.__init__")

    def forward(self, x, attn_mask=None):
        # Exactly the baseline MultiHeadAttention.forward, with one
        # addition: before the softmax, set self.last_max_logit to the
        # largest scaled, masked attention logit this forward pass
        # produced (a Python float, ignore the -inf entries). The
        # returned tensor must be bit-for-bit what the baseline would
        # return given the same weights and the same input.
        raise NotImplementedError("implement InstrumentedMultiHeadAttention.forward")


def build_model(cfg):
    """Baseline GPTModel with every attention replaced by
    QKNormAttention. This is the ablation arm."""
    model = GPTModel(cfg)
    for block in model.trf_blocks:
        block.att = QKNormAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
    return model


def instrument_baseline(model):
    """Swap every attention in an already-built baseline GPTModel for
    InstrumentedMultiHeadAttention, in place, preserving weights.
    InstrumentedMultiHeadAttention has the same parameter names as
    MultiHeadAttention, so load_state_dict(strict=True) succeeds on
    each block's attention. Use this on a trained baseline checkpoint
    to measure max attention logits without retraining and without
    touching 02_attention.py."""
    for block in model.trf_blocks:
        instrumented = InstrumentedMultiHeadAttention(
            d_in=block.att.W_query.in_features,
            d_out=block.att.d_out,
            context_length=block.att.mask.shape[0],
            num_heads=block.att.num_heads,
            dropout=block.att.dropout.p,
            qkv_bias=block.att.W_query.bias is not None,
        )
        instrumented.load_state_dict(block.att.state_dict())
        block.att = instrumented
    return model


# The variants the ablation notebook can run: label -> build_model_fn(cfg).
VARIANTS = {"qk_norm": build_model}

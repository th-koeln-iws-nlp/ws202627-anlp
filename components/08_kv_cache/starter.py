"""Component 08: KV cache.

Different in kind from the other seven: this does not change what the
model computes, only how generation reuses past work. There is no
validation loss here, a KV-cache model must produce the exact same
tokens as the uncached model given the same weights. The metric is
generation throughput.
"""

import statistics
import time

import torch
import torch.nn as nn

from baseline import FeedForward, GPTModel, LayerNorm


class KVCacheMultiHeadAttention(nn.Module):
    """Same parameter names as the baseline MultiHeadAttention
    (W_query, W_key, W_value, out_proj), so a trained baseline
    checkpoint's state_dict loads onto it with strict=True: this
    component never trains anything, it only changes how generation
    reuses keys and values already computed.

    forward takes only x and use_cache, no attn_mask: KV-cache
    generation here is single-sequence, there is no padding to mask.

    KVCacheGPTModel (given complete, below) relies on two things from
    this class beyond forward itself: a self.cache_k / self.cache_v
    pair of attributes (None until the first use_cache=True call, then
    (b, num_heads, seq_len_so_far, head_dim)), and a reset_cache()
    method that sets both back to None. Keep those names exactly."""

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        # The same submodules as the baseline MultiHeadAttention,
        # including its "mask" buffer. forward below will not read that
        # buffer, it only has to exist so the state_dict keys match the
        # baseline's and a trained checkpoint loads with strict=True.
        # Also self.cache_k = None and self.cache_v = None.
        raise NotImplementedError("implement KVCacheMultiHeadAttention.__init__")

    def reset_cache(self):
        # Forget everything cached.
        raise NotImplementedError("implement KVCacheMultiHeadAttention.reset_cache")

    def forward(self, x, use_cache=False):
        # x: (b, num_new_tokens, d_in), only the NEW tokens this call
        # (the whole prompt on the first, prefill call; one single
        # token on every following, decode call).
        #
        # With use_cache=False this is the baseline attention. With
        # use_cache=True, the keys and values of earlier calls are
        # reused instead of recomputed: keep them in cache_k / cache_v
        # (shape (b, num_heads, seq_len_so_far, head_dim)), and let the
        # new queries attend over everything cached plus the new
        # tokens.
        #
        # The causal mask can no longer be the fixed square buffer. The
        # new queries sit at absolute positions past_len, past_len + 1,
        # ..., and each may only see keys up to its own position. Work
        # out what the mask looks like for the prefill call and for a
        # single token decode call.
        raise NotImplementedError("implement KVCacheMultiHeadAttention.forward")


class KVCacheTransformerBlock(nn.Module):
    """Same four submodule names as the baseline TransformerBlock
    (att, ff, norm1, norm2), so a trained baseline checkpoint's
    state_dict loads with strict=True. Only att is a different class,
    and forward passes use_cache through to it."""

    def __init__(self, cfg):
        super().__init__()
        self.att = KVCacheMultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def reset_cache(self):
        self.att.reset_cache()

    def forward(self, x, use_cache=False):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, use_cache=use_cache)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class KVCacheGPTModel(nn.Module):
    """Same submodule names as the baseline GPTModel (tok_emb, pos_emb,
    drop_emb, trf_blocks, final_norm, out_head), so a trained baseline
    checkpoint's full state_dict loads onto it directly, with
    strict=True. Never trained itself, see load_baseline_checkpoint."""

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.ModuleList([KVCacheTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        self.out_head.weight = self.tok_emb.weight  # weight tying, matches the baseline

    def reset_cache(self):
        for block in self.trf_blocks:
            block.reset_cache()

    def forward(self, in_idx, use_cache=False):
        batch_size, num_new_tokens = in_idx.shape  # (b, num_new_tokens)
        first_att = self.trf_blocks[0].att
        past_len = first_att.cache_k.shape[2] if use_cache and first_att.cache_k is not None else 0

        tok_embeds = self.tok_emb(in_idx)  # (b, num_new_tokens, emb_dim)
        position_ids = torch.arange(past_len, past_len + num_new_tokens, device=in_idx.device)
        pos_embeds = self.pos_emb(position_ids)  # (num_new_tokens, emb_dim)
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        for block in self.trf_blocks:
            x = block(x, use_cache=use_cache)
        x = self.final_norm(x)
        logits = self.out_head(x)  # (b, num_new_tokens, vocab_size)
        return logits


def load_baseline_checkpoint(cfg, checkpoint_path):
    """Build a KVCacheGPTModel and load a trained baseline GPTModel
    checkpoint into it directly (strict=True): every parameter name in
    KVCacheGPTModel matches the baseline exactly, only the attention
    class differs in behaviour, not in its weights' names or shapes."""
    model = KVCacheGPTModel(cfg)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model


@torch.no_grad()
def generate_greedy(model, idx, max_new_tokens):
    """Greedy, uncached generation: re-runs the full sequence seen so
    far through the model at every single step. Works against either
    GPTModel or KVCacheGPTModel (with use_cache left False, the
    default), since it never asks for caching. This is the baseline
    generate_greedy_cached is compared against."""
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(idx)  # (b, seq_len, vocab_size), the whole sequence, every step
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (b, 1)
        idx = torch.cat([idx, next_token], dim=1)
    return idx


@torch.no_grad()
def generate_greedy_cached(model, idx, max_new_tokens):
    """Greedy generation using the KV cache: the prompt is fed once
    (the prefill step, which builds the cache), and every following
    step feeds only the single newest token. model must be a
    KVCacheGPTModel. Reset its cache at the start, so the function is
    safe to call repeatedly on fresh prompts. It must return exactly
    the tokens generate_greedy returns, only faster.
    """
    raise NotImplementedError("implement generate_greedy_cached")


def measure_generation_throughput(model, idx, max_new_tokens, use_cache, num_repeats=5):
    """Median tokens/second generating max_new_tokens continuations of
    idx: generate_greedy_cached if use_cache else generate_greedy.
    Follows the timing protocol from components/README.md:
    torch.cuda.synchronize() around the measurement when on a CUDA
    device (otherwise this times how fast Python queues work, not the
    GPU), discard the first (warmup) measurement, repeat num_repeats
    times, report the median."""
    device = next(model.parameters()).device
    generate_fn = generate_greedy_cached if use_cache else generate_greedy

    timings = []
    for _ in range(num_repeats + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        generate_fn(model, idx, max_new_tokens)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append(time.time() - start)

    timings = timings[1:]  # discard the first, it includes one-time warmup cost
    median_seconds = statistics.median(timings)
    return max_new_tokens / median_seconds


# The ablation notebook trains only the baseline for this component, there is no variant to compare.
VARIANTS = {}

"""Limit-case checks. Passing both this and test_shapes.py is
necessary, not sufficient, for your implementation to be correct.

Run entirely in fp32 (the module default here, no autocast involved):
bf16 rounding is coarse enough to occasionally flip which token wins
an argmax, which would make cached and uncached generation diverge
for a reason that has nothing to do with a caching bug. The real
correctness check always belongs in fp32.
"""

import torch

from starter import GPTModel, KVCacheGPTModel, generate_greedy, generate_greedy_cached


def _tiny_cfg():
    return {
        "vocab_size": 50,
        "context_length": 20,
        "emb_dim": 16,
        "n_layers": 2,
        "n_heads": 2,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }


def test_kv_cache_model_state_dict_matches_baseline_exactly():
    """The entire point of keeping identical parameter names: a
    trained baseline GPTModel's state_dict must load onto a fresh
    KVCacheGPTModel with strict=True, no missing keys, no unexpected
    keys."""
    cfg = _tiny_cfg()
    baseline = GPTModel(cfg)
    kv_model = KVCacheGPTModel(cfg)
    kv_model.load_state_dict(baseline.state_dict())  # strict=True by default, raises on any mismatch


def test_cached_and_uncached_generation_agree():
    """The one property that matters most: caching must not change
    what the model generates, only how fast. Same weights, same
    prompt, greedy decoding, fp32: the token sequences must match
    exactly."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    baseline = GPTModel(cfg)
    kv_model = KVCacheGPTModel(cfg)
    kv_model.load_state_dict(baseline.state_dict())

    idx = torch.randint(0, cfg["vocab_size"], (1, 5))
    uncached = generate_greedy(kv_model, idx.clone(), max_new_tokens=8)
    cached = generate_greedy_cached(kv_model, idx.clone(), max_new_tokens=8)
    assert torch.equal(uncached, cached)

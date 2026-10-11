"""Shape-only: confirms the code is wired up correctly, not that it is
correct. Run this first."""

import torch

from starter import (
    KVCacheGPTModel,
    KVCacheMultiHeadAttention,
    generate_greedy,
    generate_greedy_cached,
)


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


def test_kv_cache_attention_shape_no_cache():
    att = KVCacheMultiHeadAttention(d_in=16, d_out=16, context_length=20, dropout=0.0, num_heads=2)
    x = torch.randn(2, 6, 16)
    assert att(x).shape == x.shape


def test_kv_cache_attention_shape_prefill_then_decode():
    att = KVCacheMultiHeadAttention(d_in=16, d_out=16, context_length=20, dropout=0.0, num_heads=2)
    prompt = torch.randn(2, 6, 16)
    out = att(prompt, use_cache=True)
    assert out.shape == prompt.shape
    assert att.cache_k.shape == (2, 2, 6, 8)  # (b, num_heads, seq_len, head_dim)

    one_new_token = torch.randn(2, 1, 16)
    out2 = att(one_new_token, use_cache=True)
    assert out2.shape == one_new_token.shape
    assert att.cache_k.shape == (2, 2, 7, 8)  # cache grew by one


def test_kv_cache_gpt_model_shape():
    model = KVCacheGPTModel(_tiny_cfg())
    ids = torch.randint(0, 50, (2, 6))
    assert model(ids).shape == (2, 6, 50)


def test_generate_greedy_shape():
    model = KVCacheGPTModel(_tiny_cfg())
    idx = torch.randint(0, 50, (1, 4))
    out = generate_greedy(model, idx, max_new_tokens=5)
    assert out.shape == (1, 9)


def test_generate_greedy_cached_shape():
    model = KVCacheGPTModel(_tiny_cfg())
    idx = torch.randint(0, 50, (1, 4))
    out = generate_greedy_cached(model, idx, max_new_tokens=5)
    assert out.shape == (1, 9)

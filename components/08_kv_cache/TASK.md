# Component 08: KV cache

This component is different in kind from the other seven. It does not change what the model computes, so there is no validation loss to report and no variant to compare against the baseline. It changes how generation reuses work already done, and the thing you measure is speed. Like every other group, you start by training the baseline, because you measure the cache on a trained model.

## 1. What it is

Generating one token at a time, the uncached way, re-runs the entire sequence so far through the model at every single step: token 50 gets recomputed, from scratch, forty-nine more times before generation ends. Almost all of that is wasted, a causal model's keys and values for position i never change once position i has been computed. A KV cache stores them the first time and reuses them, so each new step only has to compute the one new token's query, key, and value, and attend it against everything cached so far.

## 2. Where it appears

Every production LLM serving stack (vLLM, TGI, llama.cpp, every hosted API you have ever called) relies on KV caching, it is not optional at any real scale. The cache itself has to be stored and read back at every step, so its size matters too: grouped-query attention, another component of this course, exists only to shrink it. You do not need that component for this task.

## 3. What it replaces in our GPT

`MultiHeadAttention` becomes `KVCacheMultiHeadAttention`, `TransformerBlock` becomes `KVCacheTransformerBlock` (given complete, only the different attention class), `GPTModel` becomes `KVCacheGPTModel` (given complete). No architecture changes, no training changes. These three classes keep identical parameter names and shapes to the baseline, so the number of parameters is unchanged and a trained baseline checkpoint loads with `strict=True`, see `load_baseline_checkpoint`, given complete.

## 4. What you implement

In `starter.py`:

- `KVCacheMultiHeadAttention(d_in, d_out, context_length, dropout, num_heads, qkv_bias=False)`, a `nn.Module` with `forward(x, use_cache=False)` (no `attn_mask`, this component never deals with padding) and a `reset_cache()` method
- `generate_greedy_cached(model, idx, max_new_tokens)`, given `generate_greedy` (the uncached version) complete for comparison

The given classes rely on your `KVCacheMultiHeadAttention` exposing exactly `self.cache_k`, `self.cache_v` (`None` until first populated), and `reset_cache()`, that contract is spelled out in the class's docstring. Run the tests first: `uv run pytest components/08_kv_cache/`.

Train the baseline first, like every other group: run `uv run marimo run components/run_ablation.py`, pick `08_kv_cache` and the variant `baseline`, and press **Train seeds 0, 1, 2** (see `components/README.md`). You only need the weights of seed 0, `components/08_kv_cache/runs/baseline_seed0.pt`, which is written after the first seed, about 5 to 7 minutes in.

Then open the measurement notebook, `uv run marimo run components/08_kv_cache/measure_kv_cache.py`. It loads those weights, checks that `generate_greedy` and `generate_greedy_cached` produce identical tokens, and measures the throughput of both for 50 and 500 new tokens on the device you pick. Run it once on the GPU and once on the CPU, the table at the bottom keeps both. The notebook uses your `starter.py`, so it works once your `KVCacheMultiHeadAttention` and `generate_greedy_cached` do.

## 5. The measurements

No seeds, no training:

| Measurement | Where |
| --- | --- |
| Correctness: `generate_greedy` vs `generate_greedy_cached` produce identical tokens | fp32, CPU or GPU, your trained baseline |
| Throughput: `measure_generation_throughput`, cached vs uncached | GPU |
| Throughput: `measure_generation_throughput`, cached vs uncached | CPU |

## 6. Primary metric

Generation throughput (tokens/second), `measure_generation_throughput(model, idx, max_new_tokens, use_cache)`, given complete, run once with `use_cache=False` and once with `use_cache=True`, on both GPU and CPU. Report all four numbers and the cached/uncached speedup ratio on each device.

**Before measuring speed, verify correctness**, and always in fp32: run `generate_greedy` and `generate_greedy_cached` from the same weights and the same prompt, and confirm the token sequences are identical. Do this check in fp32 specifically. bf16's rounding is coarse enough to occasionally flip which token wins an argmax between two nearly-tied logits, which would make cached and uncached generation appear to disagree for a reason that has nothing to do with a caching bug. A real mismatch only means something in the cache logic is wrong; do not chase it in bf16.

## 7. Secondary metric

How the speedup ratio changes with `max_new_tokens`: the uncached cost per step grows with sequence length (it reprocesses everything so far), the cached cost per step does not, so the gap should widen as `max_new_tokens` grows. Try at least two lengths, for example 50 and 500, and report both ratios.

## 8. What you should be able to explain

- Why the uncached path's cost per generated token grows with sequence length, and the cached path's does not (in principle, ignoring the attention sum itself growing).
- What exactly gets stored in the cache, and why its size per token is `2 * n_layers * n_heads * head_dim * bytes_per_value`: one key vector and one value vector for every layer and every head. For our model in bf16 or fp16 (2 bytes per value) that is `2 * 6 * 8 * 32 * 2` = 6,144 bytes per token.
- Why the correctness check has to run in fp32, concretely, in terms of what bf16 rounding could do to an argmax.
- Whether you would expect the cached/uncached speedup to look different on CPU versus GPU, and why, before you look at your own numbers.

## 9. Reading

**Read first**

- Hugging Face Transformers documentation, Cache strategies (the KV cache explained), https://huggingface.co/docs/transformers/main/en/kv_cache

**Afterwards, once your own implementation runs**

- Kwon et al. 2023, Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM), the industrial-strength version of the same idea, https://arxiv.org/abs/2309.06180

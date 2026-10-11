# Component 07: grouped-query attention (GQA)

## 1. What it is

The baseline gives every query head its own key and value head: 8 query heads, 8 key heads, 8 value heads. GQA groups query heads together and gives each group a single, shared key/value head: with `num_kv_heads=2`, 8 query heads share only 2 key/value heads, 4 query heads per group. `num_kv_heads=1` is the extreme case, every query head shares one single key/value head (also known as multi-query attention). `num_kv_heads=num_heads` recovers the baseline exactly.

## 2. Where it appears

Llama 2 70B, Mistral, and most current serving-oriented LLMs use GQA. Its motivation is inference, not training: when a model generates text, it keeps the keys and values of all earlier tokens in memory (the KV cache) instead of recomputing them at every step, and that cache is what has to be stored and re-read at every step. It shrinks by a factor of `num_heads / num_kv_heads`, with no change to the query side at all. Component 08 implements the cache and measures its speed, you do not need it for this task.

## 3. What it replaces in our GPT

Every `MultiHeadAttention` instance: `block.att` in every `TransformerBlock`.

## 4. What you implement

In `starter.py`:

- `GroupedQueryAttention(d_in, d_out, context_length, dropout, num_heads, num_kv_heads, qkv_bias=False)`, a `nn.Module` with a `forward(x, attn_mask=None)`, keeping the baseline `MultiHeadAttention` signature plus `num_kv_heads`

`build_model(cfg, num_kv_heads)` and `VARIANTS` are given complete. Run the tests first: `uv run pytest components/07_gqa/`.

## 5. The runs

| Variant (name in the runner) | Seeds |
| --- | --- |
| `baseline` (`num_kv_heads=8`, ordinary multi-head attention) | 0, 1, 2 |
| `kv2` (`num_kv_heads=2`) | 0, 1, 2 |
| `kv4` (`num_kv_heads=4`) | 0, 1, 2 |
| `kv1` (`num_kv_heads=1`, multi-query attention) | 0, 1, 2 |

Twelve runs, about 60 to 80 minutes. See `components/README.md` for how to start them.

**If time is short.** Never cut seeds: a variant with fewer than three seeds cannot be compared against the noise floor. Cut variants instead. Run the variants in this order: `kv2`, `kv1`, `kv4`, after the baseline at 9:30. `kv4` is closest to the baseline and the least informative, so drop it first, `kv1` (multi-query attention) is the extreme end of the trade-off. The KV cache sizes of all four settings are computed from the shapes and need no run, so the size side of the trade-off is complete whichever runs you skip. Two GPU sessions let you run two variants in parallel.

Parameter count: `W_key` and `W_value` shrink from `emb_dim -> emb_dim` to `emb_dim -> num_kv_heads * head_dim`. At `num_kv_heads=1`, that is an 8x reduction in those two matrices specifically (the ablation config has 8 heads). `W_query` and everything else stays the same size. Report the exact parameter count for every variant (`num_params` in each run's json).

## 6. Primary metric

Validation loss, `evaluate_full_validation`, see `components/README.md`.

## 7. Secondary metric

`kv_cache_bytes_per_token(cfg, n_kv_heads)` from `ablation_harness`, for each of `n_kv_heads` in `8, 4, 2, 1`: report the KV cache size in bytes per token for every variant, alongside its validation loss, so the write-up can show the size/quality tradeoff directly. Per token, the cache holds one key vector and one value vector for every layer and every KV head: `2 * n_layers * n_kv_heads * head_dim * bytes_per_value`. That is all the function computes. The number is analytical, computed from the shapes rather than measured, and it assumes a bf16 or fp16 cache (2 bytes per value).

## 8. What you should be able to explain

- Why GQA's parameter and compute savings are asymmetric: `W_query` is untouched, only `W_key` and `W_value` shrink.
- Why `repeat_interleave` (not `repeat`) is the right way to expand `num_kv_heads` back up to `num_heads`, and what would go subtly wrong with the other grouping.
- Why GQA's real motivating cost is inference-time cache size, not training-time compute, and how that shows up (or does not) in a validation-loss-only ablation like this one.
- Where the tradeoff between `num_kv_heads=8`, `4`, `2`, `1` seems to bend in your own results, if anywhere within the noise floor.

## 9. Reading

**Read first**

- Ainslie et al. 2023, GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints, https://arxiv.org/abs/2305.13245

**Afterwards, once your own implementation runs**

- Shazeer 2019, Fast Transformer Decoding: One Write-Head is All You Need (the original multi-query attention paper), https://arxiv.org/abs/1911.02150

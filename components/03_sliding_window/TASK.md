# Component 03: sliding-window attention

## 1. What it is

Full causal attention lets query position i see every key from 0 to i. Sliding-window attention caps that: query i can only see keys in `[i - window_size + 1, i]`, a fixed-size local window that slides along with the query. Every position still attends causally, just to less of the past.

## 2. Where it appears

Mistral 7B used a 4096-token window (with a receptive field that grows across layers). Longformer uses it directly for long documents, and GPT-3 alternates dense layers with locally banded ones. Modern long-context models increasingly interleave a few full-attention layers with many windowed ones.

## 3. What it replaces in our GPT

Every `MultiHeadAttention` instance: `block.att` in every `TransformerBlock`.

## 4. What you implement

In `starter.py`:

- `SlidingWindowAttention(d_in, d_out, context_length, dropout, num_heads, window_size, qkv_bias=False)`, a `nn.Module` with a `forward(x, attn_mask=None)`, keeping the same signature as the baseline `MultiHeadAttention` plus `window_size`

`build_model(cfg, window_size)` and `VARIANTS` are given complete. Run the tests first: `uv run pytest components/03_sliding_window/`.

## 5. The runs

| Variant (name in the runner) | Seeds |
| --- | --- |
| `baseline` (full causal, effectively window 512) | 0, 1, 2 |
| `window128` | 0, 1, 2 |
| `window64` | 0, 1, 2 |
| `window256` | 0, 1, 2 |

Twelve runs, about 60 to 80 minutes. See `components/README.md` for how to start them.

**If time is short.** Never cut seeds: a variant with fewer than three seeds cannot be compared against the noise floor. Cut variants instead. Run the variants in this order: `window128`, `window64`, `window256`, after the baseline at 9:30. Windows of 128 or 64 show the clearest gap against the baseline, `window256` is closest to it and the least informative, so drop that one first. Two GPU sessions let you run two windows in parallel.

Parameter count: unchanged. `window_size` changes the fixed `mask` buffer, not any learnable weight.

## 6. Primary metric

Validation loss, `evaluate_full_validation`, see `components/README.md`.

## 7. Secondary metric

`loss_by_position(model, n_buckets=8)`, on the models of your finished runs (load them with `load_run_model`, see `components/README.md`). With `context_length=512` and windows of 128 or 64, positions past the window size should show the clearest gap against the baseline, since those are exactly the positions where the baseline can see context the windowed model cannot.

**Do not measure speed.** At `context_length=512`, a window of 64 or 128 removes only a small fraction of the attention matrix's total compute, and our attention is not written to skip masked-out positions, it still computes and then discards them. Any wall-clock difference you might observe here is noise, not a real speedup. Say this in your report so nobody reports "no speedup" as a finding: at this scale there is nothing to measure, by construction. Sliding-window's real speed benefit only shows up at context lengths much longer than the window itself.

## 8. What you should be able to explain

- Why the baseline's own `MultiHeadAttention` is not simply "correct" for a windowed model, i.e. what specifically the mask condition adds to the causal one.
- Why smaller windows should hurt validation loss more at late positions than at early ones.
- Why no speedup shows up at `context_length=512`, and roughly what context length would need to be reached before one would.
- What Mistral's layer-wise receptive field trick buys over a single fixed window everywhere.

## 9. Reading

**Read first**

- Beltagy, Peters and Cohan 2020, Longformer: The Long-Document Transformer, https://arxiv.org/abs/2004.05150
- Jiang et al. 2023, Mistral 7B, https://arxiv.org/abs/2310.06825

**Afterwards, once your own implementation runs**

- Child et al. 2019, Generating Long Sequences with Sparse Transformers, https://arxiv.org/abs/1904.10509

# Component 04: QK-Norm

## 1. What it is

Normalise queries and keys, per head, before computing their dot product. Without it, nothing bounds how large Q and K can grow during training, so attention logits can grow with them, pushing softmax towards a near one-hot distribution (a handful of positions get all the weight, everything else gets none). QK-Norm keeps logits bounded regardless of what the rest of the network does upstream.

## 2. Where it appears

Henry et al. 2020 introduced it (with L2 normalisation and a learnable scale) so that the softmax does not saturate, in low-resource translation. ViT-22B (Google's 22-billion-parameter vision Transformer) is the paper that brought it to scale, with LayerNorm on queries and keys, reporting that without it the training loss diverges. Gemma 3, Qwen3 and OLMo 2 use it today.

## 3. What it replaces in our GPT

Every `MultiHeadAttention` instance: `block.att` in every `TransformerBlock`.

## 4. What you implement

In `starter.py`:

- `QKNormAttention(d_in, d_out, context_length, dropout, num_heads, qkv_bias=False)`, a `nn.Module` with a `forward(x, attn_mask=None)`, keeping the baseline `MultiHeadAttention` signature, that also records `self.last_max_logit` (a Python float) on every forward call
- `InstrumentedMultiHeadAttention(d_in, d_out, context_length, dropout, num_heads, qkv_bias=False)`, numerically identical to the baseline `MultiHeadAttention` (same parameter names, same output given the same weights), also recording `self.last_max_logit`

`RMSNorm(dim, eps=1e-6)` is given complete, it wraps PyTorch's `nn.RMSNorm`: writing one by hand is component 01's job, here it is only a building block. `build_model(cfg)` (the ablation arm), `VARIANTS` and `instrument_baseline(model)` are given complete too. The last one swaps every attention in an already-trained baseline `GPTModel` for `InstrumentedMultiHeadAttention`, in place, keeping its weights, so you can measure `last_max_logit` on your baseline runs without retraining. Run the tests first: `uv run pytest components/04_qk_norm/`.

## 5. The runs

| Variant (name in the runner) | Seeds |
| --- | --- |
| `baseline` | 0, 1, 2 |
| `qk_norm` | 0, 1, 2 |

Six runs, about 30 to 40 minutes. See `components/README.md` for how to start them.

Parameter count: QK-Norm adds two `RMSNorm(head_dim)` per attention layer, `2 * head_dim` parameters each, negligible next to the rest of the model. Report the exact numbers anyway (`num_params` in each run's json).

## 6. Primary metric

Validation loss, `evaluate_full_validation`, see `components/README.md`.

## 7. Secondary metric

Max attention logit at the end of training. Load each of your three baseline runs with `load_run_model` (see `components/README.md`), call `instrument_baseline` on it, run a forward pass over a validation batch (`next(iter_full_validation_batches(device))` from `ablation_harness`), and read off `last_max_logit` for every layer (`model.trf_blocks[i].att.last_max_logit`). Do the same on your trained QK-Norm models, no wrapper needed. Plot both, layer by layer, or take the max over layers as one number per run. QK-Norm should show visibly smaller, flatter logits.

## 8. What you should be able to explain

- What "attention entropy collapse" means and why unbounded logits cause it.
- Why QK-Norm bounds the logit regardless of what the rest of the network does, using the Cauchy-Schwarz argument from the sanity test.
- What the original QK-Norm paper's learnable temperature was for, and why we left it out here.
- Whether QK-Norm changing validation loss (or not) tells you anything about whether it would matter at a much larger scale, like ViT-22B's.

## 9. Reading

**Read first**

- Henry et al. 2020, Query-Key Normalization for Transformers, https://arxiv.org/abs/2010.04245

**Afterwards, once your own implementation runs**

- Dehghani et al. 2023, Scaling Vision Transformers to 22 Billion Parameters, https://arxiv.org/abs/2302.05442
- Gemma Team 2025, Gemma 3 Technical Report (replaces Gemma 2's logit soft-capping with QK-norm), https://arxiv.org/abs/2503.19786

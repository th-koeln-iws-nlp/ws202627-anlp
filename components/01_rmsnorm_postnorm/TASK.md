# Component 01: RMSNorm and norm placement

This component is two independent variants, each changing one thing against the baseline, **never combined**. It gets two variants because each implementation is short.

## 1. What it is

**Variant A, RMSNorm.** A simplified LayerNorm: divide each position's vector by its own root mean square, then scale by a learnable weight. No mean subtraction, no shift, no bias.

**Variant B, post-norm.** The same LayerNorm, moved. Pre-norm (our baseline) normalises before each sublayer, and adds the shortcut afterwards. Post-norm normalises after the sublayer and the shortcut have already been added together.

## 2. Where it appears

RMSNorm: Llama, Mistral, Gemma, and nearly every current open-weight model. Post-norm: the original Transformer used it, pre-norm became the default for training stability at depth, and OLMo 2 moved back towards a post-norm-style placement.

## 3. What it replaces in our GPT

Variant A replaces every `LayerNorm` instance: `norm1` and `norm2` in every `TransformerBlock`, and `final_norm` in `GPTModel`.

Variant B replaces `TransformerBlock` itself with a differently-ordered block, still built from `LayerNorm`. `final_norm` is untouched.

## 4. What you implement

In `starter.py`:

- `RMSNorm(emb_dim, eps=1e-5)`, a `nn.Module` with a `forward(x)` that returns a tensor the same shape as `x`
- `PostNormTransformerBlock(cfg)`, a `nn.Module` with a `forward(x, attn_mask=None)`, keeping the same signature as the baseline `TransformerBlock`

`build_model_a`, `build_model_b` and `VARIANTS` are given complete. Run the tests first: `uv run pytest components/01_rmsnorm_postnorm/`.

## 5. The runs

| Variant (name in the runner) | Seeds |
| --- | --- |
| `baseline` (pre-norm, LayerNorm) | 0, 1, 2 |
| `A_rmsnorm` | 0, 1, 2 |
| `B_postnorm` | 0, 1, 2 |

Nine runs, about 45 to 60 minutes. See `components/README.md` for how to start them.

**If time is short.** Never cut seeds: a variant with fewer than three seeds cannot be compared against the noise floor. Cut variants instead. The baseline always comes first, at 9:30. Implement `A_rmsnorm` first, it is the shorter one, and let its runs go while you implement `B_postnorm`. If you can afford only one variant, drop `A_rmsnorm`: whether post-norm trains at all with only 20 warmup steps is the less predictable and more informative result. If you have two GPU sessions, run the two variants in parallel.

Parameter count: variant A removes the shift parameters LayerNorm has and RMSNorm does not, a few thousand parameters. Variant B is unchanged, same modules, different order. Report both counts either way, they are `num_params` in each run's json.

## 6. Primary metric

Validation loss, `evaluate_full_validation`, see `components/README.md`.

## 7. Secondary metric

**Variant A.** Step time against the baseline, the `step_time_ms` field of each run's json.

**Variant B.** Whether it trains stably at all with only 20 warmup steps: plot the loss curve, and the maximum gradient norm over training. Both are in each run's `<variant>_seed<N>_history.json` (`train_loss`, `val_loss`, `grad_norm` per step, the gradient norm is the one before clipping).

## 8. What you should be able to explain

- Why pre-norm became the default for training deep Transformers.
- Why OLMo 2 moved back towards post-norm.
- What RMSNorm drops from LayerNorm, and why that might not matter.
- What happens to gradients early in training if warmup is too short and the norm placement is wrong for it.

## 9. Reading

**Read first**

- Zhang and Sennrich 2019, Root Mean Square Layer Normalization, https://arxiv.org/abs/1910.07467
- Xiong et al. 2020, On Layer Normalization in the Transformer Architecture, https://arxiv.org/abs/2002.04745

**Afterwards, once your own implementation runs**

- Team OLMo et al. 2024, 2 OLMo 2 Furious (the OLMo 2 technical report), the reordered norm, https://arxiv.org/abs/2501.00656
- Raschka, The Big LLM Architecture Comparison, OLMo 2 section, https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

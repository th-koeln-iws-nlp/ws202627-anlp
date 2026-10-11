# Component 02: SwiGLU feed-forward

## 1. What it is

A gated feed-forward network. The baseline feed-forward is two matrices with a GELU in between: up-project, activate, down-project. SwiGLU is three matrices: one branch gates the other, elementwise, before the down-projection. The activation itself also changes, from GELU to SiLU (the "Swi" in SwiGLU is short for Swish, another name for SiLU).

## 2. Where it appears

Llama, Mistral, Gemma, PaLM, and most current open-weight models use SwiGLU (or a close variant) in place of a plain GELU feed-forward.

## 3. What it replaces in our GPT

Every `FeedForward` instance: one per `TransformerBlock`, `cfg["n_layers"]` of them in total.

## 4. What you implement

In `starter.py`:

- `SwiGLUFeedForward(cfg, hidden_dim)`, a `nn.Module` with a `forward(x)` that returns a tensor the same shape as `x`. Three `nn.Linear` layers: `gate` and `up` (both `emb_dim -> hidden_dim`), and `down` (`hidden_dim -> emb_dim`), all with bias, matching the baseline's own use of bias.

`build_model` and `VARIANTS` are given complete. Run the tests first: `uv run pytest components/02_swiglu/`.

## 5. The runs

| Variant (name in the runner) | Seeds |
| --- | --- |
| `baseline` (GELU feed-forward) | 0, 1, 2 |
| `swiglu`, `hidden_dim=680` (matched) | 0, 1, 2 |
| `swiglu_1024`, the baseline's own hidden size (optional, unmatched) | 0, 1, 2 |

Six runs, about 30 to 40 minutes, plus 15 to 20 minutes for the optional one, which shows the effect of extra parameters on its own. See `components/README.md` for how to start them.

**If time is short.** The optional `swiglu_1024` run comes last. Only start it once the baseline and `swiglu` are done and you still have time, otherwise skip it. Never cut seeds: a variant with fewer than three seeds cannot be compared against the noise floor.

Parameter count: `MATCHED_HIDDEN_DIM = 680` is chosen so the two feed-forwards are within about 0.3% of each other in parameter count (the classic `8d/3` recipe, rounded to the nearest multiple of 8, for `d = 256`). Report both counts either way (`num_params` in each run's json), and note that `swiglu_1024` does not have matched parameters.

## 6. Primary metric

Validation loss, `evaluate_full_validation`, see `components/README.md`.

## 7. Secondary metric

Step time against the baseline, the `step_time_ms` field of each run's json: SwiGLU does three matrix multiplies where the baseline does two, at a smaller hidden dimension each.

## 8. What you should be able to explain

- What the gate branch does that a plain two-matrix feed-forward cannot.
- Why `hidden_dim` has to shrink to keep parameter count matched, given a third matrix.
- What SiLU is, and how it differs from GELU.
- If SwiGLU wins, whether that is more likely the gating, the activation, or the parameter budget.

## 9. Reading

**Read first**

- Shazeer 2020, GLU Variants Improve Transformer, https://arxiv.org/abs/2002.05202

**Afterwards, once your own implementation runs**

- Touvron et al. 2023, LLaMA: Open and Efficient Foundation Language Models (uses SwiGLU with a hidden size of 2/3 of 4d, the `8d/3` sizing), https://arxiv.org/abs/2302.13971
- Raschka, The Big LLM Architecture Comparison, https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

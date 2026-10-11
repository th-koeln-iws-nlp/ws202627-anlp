# Component 06: RoPE, rotary position embeddings

## 1. What it is

Instead of adding a position vector to the token embedding once, at the bottom of the network, RoPE rotates each query and key vector by an angle that depends on its position, right before the attention dot product, in every layer. The rotation is designed so that the dot product of a rotated query at position i and a rotated key at position j depends only on the relative offset i - j, never on i and j separately.

## 2. Where it appears

GPT-NeoX, Llama, Mistral, Qwen, and nearly every current open-weight decoder-only model use RoPE (or a direct variant of it).

## 3. What it replaces in our GPT

`model.pos_emb` (zeroed out, contributes nothing) and every `MultiHeadAttention` instance (`block.att` in every `TransformerBlock`).

## 4. What you implement

In `starter.py`:

- `precompute_rope(head_dim, context_length, theta_base=10000.0)`, returns `(cos, sin)`, each `(context_length, head_dim)`
- `rotate_half(x)`, returns a tensor the same shape as `x`
- `apply_rope(x, cos, sin)`, returns a tensor the same shape as `x`
- `RotaryMultiHeadAttention(d_in, d_out, context_length, dropout, num_heads, qkv_bias=False, theta_base=10000.0)`, a `nn.Module` with a `forward(x, attn_mask=None)`, keeping the baseline `MultiHeadAttention` signature plus `theta_base`
- `ZeroPositionalEmbedding(emb_dim)`, a `nn.Module` with a `forward(position_ids)` that always returns zeros

**This implementation uses the rotate-half convention** (GPT-NeoX, Llama): dimension i is paired with dimension `i + head_dim/2`, not with `i+1` as in the original paper's own presentation. `precompute_rope` and `apply_rope` have to agree with each other on this. If you look at another RoPE implementation for reference, check which convention it uses before copying pieces across: mixing the two conventions does not crash, it silently computes a wrong rotation, the single most common RoPE bug.

`build_model(cfg)` and `VARIANTS` are given complete. Run the tests first: `uv run pytest components/06_rope/`.

## 5. The runs

| Variant (name in the runner) | Seeds |
| --- | --- |
| `baseline` (learned position embedding) | 0, 1, 2 |
| `rope` | 0, 1, 2 |

Six runs, about 30 to 40 minutes. See `components/README.md` for how to start them.

Parameter count: removes `context_length * emb_dim` learned parameters (the position embedding table). RoPE's own `cos`/`sin` tables are buffers, not parameters, and cost nothing to train. Report both counts (`num_params` in each run's json).

## 6. Primary metric

Validation loss, `evaluate_full_validation`, see `components/README.md`.

## 7. Secondary metric

`loss_by_position(model, n_buckets=8)`, on the models of your finished runs (load them with `load_run_model`, see `components/README.md`): plot both variants' buckets together. RoPE's relative-position property is often argued to help most at positions far from the start of the window.

## 8. What you should be able to explain

- What "depends only on relative position" buys over an absolute position embedding, concretely.
- Why the rotate-half convention and the interleaved-pairs convention compute the same underlying idea but are not interchangeable code.
- Why `head_dim` must be even, and what would break if it were not.
- Whether RoPE's relative-position property gives it a real advantage at `context_length=512` after only 2,000 steps, or whether that advantage should mostly show up at much longer sequences and much longer training, if at all in this ablation.

## 9. Reading

**Read first**

- Su et al. 2021 (revised 2023), RoFormer: Enhanced Transformer with Rotary Position Embedding, https://arxiv.org/abs/2104.09864

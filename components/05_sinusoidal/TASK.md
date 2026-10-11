# Component 05: sinusoidal positional embeddings

## 1. What it is

The baseline learns a position embedding table, `context_length` separate vectors, one per position, trained like any other parameter. Sinusoidal embeddings replace that table with a fixed function of position: a mix of sine and cosine waves at geometrically increasing wavelengths, computed once and never updated.

## 2. Where it appears

The original Transformer (Vaswani et al. 2017) used it. Most current decoder-only LLMs (GPT-2 onwards) moved to learned absolute embeddings, then to RoPE (component 06); sinusoidal is now mostly of historical and pedagogical interest, useful here as the "nothing learned, no relative structure either" baseline.

## 3. What it replaces in our GPT

`model.pos_emb`, the single `nn.Embedding(context_length, emb_dim)` in `GPTModel`. `GPTModel.forward` calls it with `torch.arange(seq_len)`, exactly as it would call an `nn.Embedding`, so nothing else changes.

## 4. What you implement

In `starter.py`:

- `SinusoidalPositionalEmbedding(context_length, emb_dim)`, a `nn.Module` with a `forward(position_ids)` that returns `(len(position_ids), emb_dim)`, called exactly like the baseline's `nn.Embedding`

`build_model(cfg)` and `VARIANTS` are given complete. Run the tests first: `uv run pytest components/05_sinusoidal/`.

## 5. The runs

| Variant (name in the runner) | Seeds |
| --- | --- |
| `baseline` (learned position embedding) | 0, 1, 2 |
| `sinusoidal` | 0, 1, 2 |

Six runs, about 30 to 40 minutes. See `components/README.md` for how to start them.

Parameter count: removes `context_length * emb_dim` learned parameters (the position embedding table), replaced by a same-shape buffer that costs memory but not gradient computation or optimizer state. Report both counts (`num_params` in each run's json).

## 6. Primary metric

Validation loss, `evaluate_full_validation`, see `components/README.md`.

## 7. Secondary metric

`loss_by_position(model, n_buckets=8)`, on the models of your finished runs (load them with `load_run_model`, see `components/README.md`): plot both variants' buckets together. A fixed, non-learned position signal may behave differently at the positions farthest from the ones seen most often during the (short, 2,000-step) training run.

## 8. What you should be able to explain

- What information a learned position embedding can capture that a fixed sinusoidal one cannot, in principle.
- Why the frequencies are chosen geometrically (from very fast to very slow) rather than evenly spaced.
- Why removing gradient updates from the position signal might matter more, or less, over only 2,000 steps than over a full pretraining run.
- What "extrapolation to unseen lengths" means, and which of the two, learned or sinusoidal, has a stronger a priori argument for it (whether or not our 2,000-step run is enough to observe it).

## 9. Reading

**Read first**

- Vaswani et al. 2017, Attention Is All You Need, section 3.5, https://arxiv.org/abs/1706.03762

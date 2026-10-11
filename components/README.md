# The shared protocol

One protocol, identical for every pair and every component. What you compare against and how the numbers are computed must never be something you have to guess.

## The baseline

The baseline is **the GPT-2 you built on days 1 and 2**, specifically the unmodified `GPTModel` from `solutions/04_gpt_solution.py`, built with `ablation_config()` from `ablation_harness.py`, next to this file. Not your day 2 checkpoint. The day 2 model trained on a 12 minute wall clock for one seed; the baseline here is retrained under the fixed protocol below so that it can actually be compared against.

Every pair trains the baseline themselves, at 9:30 on day 3, on molab or the cluster, before the framing talk. Three runs, about 15 to 20 minutes. The baseline rows in your component's `TASK.md` are these same runs: train them once, the runner keeps them in your component's folder.

## The fixed protocol

Identical for every baseline run and every component run, for every pair:

| Setting | Value |
| --- | --- |
| Model | `emb_dim` 256, `n_layers` 6, `n_heads` 8, `context_length` 512, `drop_rate` 0.1, `qkv_bias` False |
| Data | `train.bin` and `valid.bin` from `Richard-Sieg-TH-Koln/anlp-tinystories-gpt2` |
| Steps | 2,000, fixed. Not a wall clock budget |
| Batch size | 64 |
| Optimizer | AdamW, lr 1e-3, weight decay 0.1, betas (0.9, 0.95), grad clip 1.0 |
| Schedule | 20 warmup steps, cosine decay to 1e-4 over the 2,000 steps |
| Precision | bf16 autocast for training |
| Seeds | 0, 1 and 2 |

About 5 to 7 minutes per run, depending on the GPU: roughly 5 on molab's RTX PRO 6000, 7 on the cluster's A100.

"Same seed" does not mean identical initial weights once the architecture changes, because a different model consumes the random number generator differently. That is expected, and it is one of the reasons we use three seeds and compare means rather than single runs.

## How to run

1. **Test your implementation first**, one folder at a time: `uv run pytest components/06_rope/`. All folders share file names, so `pytest components` as a whole fails at collection. `test_shapes.py` checks that your code is wired up correctly, `test_sanity.py` checks limit cases. Passing both is necessary, not sufficient.
2. **Quick check, about a minute.** Start the runner with `uv run marimo run components/run_ablation.py`, pick your component and a variant (the names are in your `TASK.md`), and press **Quick check**. It trains your variant and the baseline for 200 steps on seed 0 and shows both loss curves. A correct variant starts near the untrained loss of about 10.8, goes down, and ends in the same range as the baseline. The notebook warns you if the loss stays flat, turns to NaN, or starts far from 10.8. Passing it does not prove that your implementation is correct, a slightly wrong one still trains, but a broken one usually shows here.
3. **Start the runs.** In the same notebook, press **Train seeds 0, 1, 2**. It trains the three seeds under the protocol above and shows the loss curves, and for every variant after the baseline the effect against the noise floor. The notebook loads your `starter.py` again each time you press a button, so you can keep it open while you edit.

**If time is short**, never cut seeds: a variant with fewer than three seeds cannot be compared against the noise floor. Cut variants instead. Components with several variants list in their `TASK.md` which ones to prioritise.

Each run leaves three files in `components/<your folder>/runs/`: `<variant>_seed<N>.pt` (the weights), `<variant>_seed<N>.json` (the result, see below) and `<variant>_seed<N>_history.json` (train loss, validation loss, step time and gradient norm for every step).

## The primary metric: validation loss

**Definition.** Mean cross-entropy over the **entire** `valid.bin`, split into non-overlapping windows of 512 tokens, computed once after the final training step, with the model in `eval()` mode. Computed by `evaluate_full_validation(model)` in `ablation_harness.py`, shared by every component. **Do not write your own.** Every number in the class must come from the same tokens and the same code.

Why the full set rather than a sample: a sampled validation set changes with the seed, which would add evaluation noise to training noise and inflate the noise floor. The full set is identical for every run.

## The noise floor

The spread of the three baseline validation losses: highest minus lowest. It is how much the number moves for no reason at all.

## The decision rule

- **Effect** = mean validation loss of the component runs minus mean of the baseline runs. Negative means the component lowered the loss.
- If the absolute effect is **not larger than the noise floor**, the result is **inconclusive at this scale**. That is a complete, correct answer.
- If it is larger, report the direction and the size, and still say what the scale of the experiment means for how far the result travels.
- Also report the spread of your own component runs. If it is much larger than the baseline spread, your component made training less stable, which is itself a finding.

## Always report

Every run writes one result file, `<variant>_seed<N>.json`, into `components/<your folder>/runs/`. This is what gets collected into the class results table. It is written by the harness automatically, you do not fill it in by hand. The fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `component` | string | the component folder name, e.g. `01_rmsnorm_postnorm` |
| `variant` | string | the run label, e.g. `baseline`, `A_rmsnorm`, `B_postnorm` |
| `seed` | int | the seed this run used |
| `num_steps` | int | the number of training steps this run completed |
| `val_loss` | float | the primary metric, from `evaluate_full_validation(model)` |
| `num_params` | int | total parameter count of this run's model |
| `tokens_per_second` | float | training throughput over the whole run |
| `step_time_ms` | float | median milliseconds per training step from step 100 on |
| `config` | object | the full config this run's model was built with |
| `secondary` | object | free-form, the component-specific secondary metric, empty if a run has none to attach |
| `timestamp` | string | ISO 8601 UTC, when this run finished |

## Reference baseline

My own three baseline runs, computed with the same harness:

| Seed | Validation loss |
| --- | --- |
| 0 | 1.8733 |
| 1 | 1.8906 |
| 2 | 1.8685 |

Noise floor on the reference setup: 0.022. Your baseline should land within that of these numbers. If it does not, something in your setup differs from the protocol. Find it before you run your component.

## Secondary metrics

Every component also has a secondary metric, defined in its `TASK.md`, matching what that component was actually designed to improve. Most of these components were not designed to lower the loss of a 17M parameter model, so judged on loss alone most of them would look useless. The secondary metric is how you can see whether the mechanism is doing anything at all.

Some of them are computed on the models your runs leave behind. Reload one with `load_run_model`, the second argument is the variant name from your `TASK.md`, the last one is the function the run was trained with:

```python
import sys

sys.path[:0] = ["components", "components/03_sliding_window"]  # from the repo root

from ablation_harness import load_run_model, loss_by_position
from starter import VARIANTS

model = load_run_model("03_sliding_window", "window128", 0, VARIANTS["window128"])
loss_by_position(model)
```

The baseline is `baseline.GPTModel`. Some secondary metrics are analytical rather than measured, computed from the shapes alone (the KV cache size per token in component 07).

## Timing measurements

Several secondary metrics are timings, and timings are noisy too. Same logic as the loss noise floor:

- call `torch.cuda.synchronize()` before starting and before stopping the clock on GPU, otherwise you are timing how fast Python queues work, not how fast the GPU does it
- discard the first measurement, it includes warm-up
- repeat five times, report the median and the spread

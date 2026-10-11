"""Day 3 ablation harness: the fixed protocol from components/README.md,
shared by every component. Do not edit it, every number in the class
results table has to come from the same code. Standalone: it needs only
the packages in pyproject.toml, nothing from solutions/ or notebooks/.

    run_seeds(build_model_fn, component, label)   train one variant, seeds 0 to 2
    evaluate_full_validation(model)               the primary metric
    load_run_model(component, label, seed, fn)    reload a finished run's model
    summarize(baseline_losses, component_losses)  effect, noise floor, verdict

plus the secondary-metric helpers loss_by_position and
kv_cache_bytes_per_token.
"""

import functools
import json
import math
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import marimo as mo
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

DATASET_REPO_ID = "Richard-Sieg-TH-Koln/anlp-tinystories-gpt2"

# The optimizer and schedule constants are the fixed protocol table in
# components/README.md, and the same values the pretraining notebook
# settled on.
MAX_LR = 1e-3
MIN_LR = 1e-4
WARMUP_STEPS = 20
GRAD_CLIP = 1.0
EVAL_EVERY_STEPS = 100  # periodic validation for the logged curve
EVAL_ITERS = 20  # batches per periodic validation

CONTEXT_LENGTH = 512
BATCH_SIZE = 64

COMPONENTS_DIR = Path(__file__).resolve().parent

# Settled config, do not re-derive: 2,000 fixed steps per run. Fixed
# step count, not a wall clock budget: a faster component would
# otherwise simply see more tokens in the same time, and a throughput
# difference would get reported as a loss difference.
NUM_STEPS = 2000
SEEDS = [0, 1, 2]

# Fixed, independent of any run's own seed: see train_for_ablation.
VAL_BATCH_SEED = 1234


def get_batch(data, batch_size, context_length, device):
    """Random windows into a token memmap, nanoGPT style. Never loads the full array."""
    max_start = len(data) - context_length - 1
    starts = torch.randint(max_start, (batch_size,))  # (batch_size,)
    x = torch.stack(
        [torch.from_numpy(data[i : i + context_length].astype(np.int64)) for i in starts]
    )  # (batch_size, context_length)
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + 1 + context_length].astype(np.int64)) for i in starts]
    )  # (batch_size, context_length)
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, batches):
    """Mean cross-entropy over a list of (x, y) batches, used for the periodic curve logged
    during training. The final, reported number always comes from evaluate_full_validation
    instead, on the entire validation set rather than a handful of sampled batches."""
    model.eval()
    losses = []
    for x, y in batches:
        logits = model(x)  # (batch_size, context_length, vocab_size)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def ablation_config():
    """The fixed day 3 protocol config, identical for the baseline and
    every component: emb_dim 256, 6 layers, 8 heads, context 512, drop
    0.1, no qkv bias. See components/README.md."""
    return {
        "vocab_size": 50257,
        "context_length": CONTEXT_LENGTH,
        "emb_dim": 256,
        "n_layers": 6,
        "n_heads": 8,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }


@functools.lru_cache(maxsize=1)
def load_ablation_data():
    """Train and validation data for the fixed protocol, downloaded and
    memory-mapped once per process, then cached."""
    train_bin_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="train.bin", repo_type="dataset")
    valid_bin_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="valid.bin", repo_type="dataset")
    train_data = np.memmap(train_bin_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(valid_bin_path, dtype=np.uint16, mode="r")
    return train_data, val_data


def ablation_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_for_ablation(model, train_data, val_data, device, batch_size, context_length, num_steps, seed):
    """Train one model for num_steps, under the fixed day 3 protocol.

    The periodic validation batches used for the logged curve are drawn
    under a fixed seed (VAL_BATCH_SEED), independent of `seed`:
    torch.random.fork_rng saves and restores the RNG state around that
    one draw, so every run's curve is measured on the identical
    windows, and the training run itself still consumes the RNG stream
    seeded by `seed` exactly as if this draw had not happened. Returns
    (history, elapsed_seconds, tokens_seen).
    """
    torch.manual_seed(seed)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=0.1, betas=(0.9, 0.95))

    with torch.random.fork_rng():
        torch.manual_seed(VAL_BATCH_SEED)
        val_batches = [get_batch(val_data, batch_size, context_length, device) for _ in range(EVAL_ITERS)]

    history = []
    tokens_seen = 0
    use_bf16 = device.type == "cuda"
    start_time = time.time()

    for step in range(num_steps):
        if step < WARMUP_STEPS:
            lr = MAX_LR * (step + 1) / WARMUP_STEPS
        else:
            progress = min((step - WARMUP_STEPS) / max(num_steps - WARMUP_STEPS, 1), 1.0)
            lr = MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1 + math.cos(math.pi * progress))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if device.type == "cuda":
            torch.cuda.synchronize()  # otherwise this times how fast Python queues work, not the GPU
        step_start = time.time()

        x, y = get_batch(train_data, batch_size, context_length, device)  # (batch_size, context_length) each

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(x)  # (batch_size, context_length, vocab_size)
            train_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)  # norm before clipping
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        step_ms = (time.time() - step_start) * 1000

        tokens_seen += x.numel()

        val_loss = None
        if step % EVAL_EVERY_STEPS == 0 or step == num_steps - 1:
            val_loss = evaluate(model, val_batches)

        history.append(
            {
                "step": step,
                "train_loss": train_loss.item(),
                "val_loss": val_loss,
                "step_ms": step_ms,
                "lr": lr,
                "grad_norm": grad_norm.item(),
            }
        )

    elapsed_seconds = time.time() - start_time
    return history, elapsed_seconds, tokens_seen


def iter_full_validation_batches(device, batch_size=64):
    """Every non-overlapping 512-token window in valid.bin, batched.

    Shared by evaluate_full_validation (and loss_by_position, where a
    component includes it) so both read the exact same tokens, in the
    exact same order, the exact same way: every window needs
    context_length + 1 raw tokens (the extra one is the shifted target
    for the window's last position), so window i occupies raw tokens
    [i * (context_length + 1), (i + 1) * (context_length + 1)), with no
    gap and no overlap between windows.
    """
    _, val_data = load_ablation_data()
    window_stride = CONTEXT_LENGTH + 1
    n_windows = (len(val_data) - 1) // window_stride
    for batch_start in range(0, n_windows, batch_size):
        indices = range(batch_start, min(batch_start + batch_size, n_windows))
        x = torch.stack(
            [
                torch.from_numpy(val_data[i * window_stride : i * window_stride + CONTEXT_LENGTH].astype(np.int64))
                for i in indices
            ]
        ).to(device)  # (batch, CONTEXT_LENGTH)
        y = torch.stack(
            [
                torch.from_numpy(
                    val_data[i * window_stride + 1 : i * window_stride + 1 + CONTEXT_LENGTH].astype(np.int64)
                )
                for i in indices
            ]
        ).to(device)  # (batch, CONTEXT_LENGTH)
        yield x, y


@torch.no_grad()
def evaluate_full_validation(model):
    """Mean cross-entropy over the ENTIRE valid.bin, split into
    non-overlapping 512-token windows, batched 64 at a time, model in
    eval() mode. The primary metric for every component: do not write
    your own, every number in the class results table has to come from
    the same tokens and the same code.

    Non-overlapping and exhaustive rather than a random sample, on
    purpose: a sampled validation set changes with the seed, which
    would add evaluation noise on top of training noise and inflate
    the noise floor. The full set is identical for every run.
    """
    device = next(model.parameters()).device
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    for x, y in iter_full_validation_batches(device, batch_size=64):
        logits = model(x)  # (batch, CONTEXT_LENGTH, vocab_size)
        # sum, not mean, then divide once at the end: the last batch is
        # very likely a different size than 64, and averaging per-batch
        # means before averaging those means would silently overweight
        # the small last batch relative to a token-level average.
        batch_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_loss += batch_loss.item()
        total_tokens += y.numel()

    model.train()
    return total_loss / total_tokens


def step_time_ms(history):
    """Median milliseconds per training step from step 100 on, excluding
    warm-up. Same reasoning as the loss noise floor: a single step's
    timing is noisy, report the median of many, not one number."""
    timed_steps = [row["step_ms"] for row in history if row["step"] >= 100]
    return statistics.median(timed_steps) if timed_steps else float("nan")


def run_dir(component):
    return COMPONENTS_DIR / component / "runs"


def run_seeds(build_model_fn, component, label, seeds=SEEDS, num_steps=NUM_STEPS, cfg=None):
    """Train build_model_fn(cfg) once per seed under the fixed protocol.

    component is the folder name, e.g. "03_sliding_window", label names
    the variant, e.g. "baseline" or "window128". For each seed this
    writes, into components/<component>/runs/, next to each other:
      <label>_seed<N>.pt            the weights
      <label>_seed<N>.json          the result, fields listed in components/README.md
      <label>_seed<N>_history.json  train/val loss, step time, grad norm per step
    Returns the list of per-seed validation losses, in the order of
    `seeds`. seeds, num_steps and cfg only exist for quick smoke tests,
    the protocol uses the defaults.
    """
    cfg = ablation_config() if cfg is None else cfg
    train_data, val_data = load_ablation_data()
    device = ablation_device()
    runs_dir = run_dir(component)
    runs_dir.mkdir(parents=True, exist_ok=True)

    val_losses = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = build_model_fn(cfg).to(device)
        num_params = sum(p.numel() for p in model.parameters())

        history, elapsed_seconds, tokens_seen = train_for_ablation(
            model,
            train_data,
            val_data,
            device,
            batch_size=BATCH_SIZE,
            context_length=CONTEXT_LENGTH,
            num_steps=num_steps,
            seed=seed,
        )
        val_loss = evaluate_full_validation(model)
        val_losses.append(val_loss)

        torch.save(model.state_dict(), runs_dir / f"{label}_seed{seed}.pt")
        (runs_dir / f"{label}_seed{seed}_history.json").write_text(json.dumps(history))

        result = {
            "component": component,
            "variant": label,
            "seed": seed,
            "num_steps": num_steps,
            "val_loss": val_loss,
            "num_params": num_params,
            "tokens_per_second": tokens_seen / elapsed_seconds,
            "step_time_ms": step_time_ms(history),
            "config": cfg,
            "secondary": {},
            "timestamp": datetime.now(UTC).isoformat(),
        }
        (runs_dir / f"{label}_seed{seed}.json").write_text(json.dumps(result, indent=2))

    return val_losses


def load_val_losses(component, label):
    """Validation losses of every finished seed of one variant, by seed."""
    result_files = sorted(run_dir(component).glob(f"{label}_seed[0-9].json"))
    return [json.loads(path.read_text())["val_loss"] for path in result_files]


def load_run_model(component, label, seed, build_model_fn):
    """Rebuild a finished run's model from its saved weights, in eval
    mode on the ablation device. build_model_fn is the one the run was
    trained with."""
    model = build_model_fn(ablation_config())
    state_dict = torch.load(run_dir(component) / f"{label}_seed{seed}.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(ablation_device()).eval()


def summarize(baseline_losses, component_losses):
    """Baseline mean/spread, component mean/spread, the effect, and the
    decision-rule verdict, as one markdown table. Also flags a
    component spread much larger than the baseline spread, itself a
    finding about training stability, not just about the mean.
    """
    baseline_mean = sum(baseline_losses) / len(baseline_losses)
    baseline_spread = max(baseline_losses) - min(baseline_losses)
    component_mean = sum(component_losses) / len(component_losses)
    component_spread = max(component_losses) - min(component_losses)
    effect = component_mean - baseline_mean

    if abs(effect) <= baseline_spread:
        verdict = "**Inconclusive at this scale**: the effect does not exceed the noise floor."
    elif effect < 0:
        verdict = f"**Component lowered val loss** by {abs(effect):.4f}, larger than the noise floor of {baseline_spread:.4f}."
    else:
        verdict = f"**Component raised val loss** by {effect:.4f}, larger than the noise floor of {baseline_spread:.4f}."

    stability_note = (
        f" Component spread ({component_spread:.4f}) is much larger than the baseline spread "
        f"({baseline_spread:.4f}): this component made training less stable, that is itself a finding."
        if component_spread > 2 * baseline_spread
        else ""
    )

    return mo.md(
        f"""
        | | mean val loss | spread |
        |---|---|---|
        | baseline | {baseline_mean:.4f} | {baseline_spread:.4f} |
        | component | {component_mean:.4f} | {component_spread:.4f} |

        effect (component mean - baseline mean): `{effect:+.4f}`

        {verdict}{stability_note}
        """
    )


@torch.no_grad()
def loss_by_position(model, n_buckets=8):
    """Per-token cross-entropy at each of the 512 positions in a
    window, averaged over every validation window, then grouped into
    n_buckets consecutive buckets (512 / n_buckets positions each).
    Returns n_buckets numbers.

    Positions are positions within the 512-token window, not within a
    story: a window can start and end anywhere in the validation set
    and cuts across story boundaries freely, so "position 500" here
    just means the 500th token this particular window happened to
    contain, not the 500th token of whatever story it came from.
    """
    device = next(model.parameters()).device
    model.eval()

    position_loss_sum = torch.zeros(CONTEXT_LENGTH, device=device)  # (CONTEXT_LENGTH,)
    position_count = 0
    for x, y in iter_full_validation_batches(device, batch_size=64):
        logits = model(x)  # (batch, CONTEXT_LENGTH, vocab_size)
        per_token_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none"
        ).view(x.shape[0], CONTEXT_LENGTH)  # (batch, CONTEXT_LENGTH)
        position_loss_sum += per_token_loss.sum(dim=0)  # (CONTEXT_LENGTH,)
        position_count += x.shape[0]

    model.train()
    position_mean = (position_loss_sum / position_count).cpu()  # (CONTEXT_LENGTH,)
    bucket_size = CONTEXT_LENGTH // n_buckets
    return [position_mean[i * bucket_size : (i + 1) * bucket_size].mean().item() for i in range(n_buckets)]


def kv_cache_bytes_per_token(cfg, n_kv_heads, bytes_per_value=2):
    """Analytical KV cache size per token: 2 (keys and values) times
    n_layers times n_kv_heads times head_dim times bytes_per_value. Not
    measured, since the size follows directly from the shapes involved.
    The default of 2 bytes per value is a bf16 or fp16 cache."""
    head_dim = cfg["emb_dim"] // cfg["n_heads"]
    return 2 * cfg["n_layers"] * n_kv_heads * head_dim * bytes_per_value

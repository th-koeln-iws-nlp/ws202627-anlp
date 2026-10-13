import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import importlib
    import json
    import math
    import os
    import time
    from pathlib import Path

    import numpy as np
    import plotly.graph_objects as go
    import tiktoken
    import torch
    import torch.nn.functional as F
    import marimo as mo
    from dotenv import load_dotenv
    from huggingface_hub import HfApi, hf_hub_download

    # get_batch is defined in exactly one place, 01_text_data.py.
    _text_data = importlib.import_module("01_text_data")
    get_batch = _text_data.get_batch
    DATASET_REPO_ID = _text_data.DATASET_REPO_ID

    # GPTModel, build_config and generate are defined in exactly one
    # place, 04_gpt.py.
    _gpt = importlib.import_module("04_gpt")
    GPTModel = _gpt.GPTModel
    build_config = _gpt.build_config
    generate = _gpt.generate
    chat_window = _gpt.chat_window

    # Settled config, do not re-derive: small (emb_dim 256, 6 layers, 8
    # heads, 17,731,328 params), context 512, batch 64. Measured on
    # molab's RTX PRO 6000: a 12 minute run does about 213k tokens per
    # second, 4,680 steps, val loss 1.62, peak VRAM 28.7 GB.
    CONTEXT_LENGTH = 512
    BATCH_SIZE = 64
    MAX_LR = 1e-3
    MIN_LR = 1e-4
    WARMUP_STEPS = 20
    GRAD_CLIP = 1.0
    EVAL_EVERY_STEPS = 100
    EVAL_ITERS = 20


@app.cell(hide_code=True)
def _():
    mo.md("""
    # 05. Pretraining

    Raschka chapter 5.1-5.2: the loss, and the loop that drives it down.
    Everything before this notebook was architecture with no training
    signal. This is where the model actually learns TinyStories.

    In this notebook you write one function, `evaluate`. Everything else
    is given: read it, run it, and watch the loss fall.
    """)
    return


@app.cell
def _():
    gpt2_tokenizer = tiktoken.get_encoding("gpt2")
    return (gpt2_tokenizer,)


@app.cell
def _():
    train_bin_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="train.bin", repo_type="dataset")
    valid_bin_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="valid.bin", repo_type="dataset")

    train_data = np.memmap(train_bin_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(valid_bin_path, dtype=np.uint16, mode="r")
    return train_data, val_data


@app.cell
def _():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (device,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Evaluating: cross-entropy and perplexity, Raschka 5.1

    Cross-entropy is the negative log probability the model assigned to
    the actual next token, averaged over a batch. An untrained model
    should score close to `ln(vocab_size)` &asymp; 10.8: that is exactly
    the loss of a uniform random guess over 50,257 tokens, which is all
    an untrained (but correctly initialized) model can do.
    `torch.no_grad()` and `model.eval()` matter here: no gradient
    bookkeeping, and dropout off, since we want a stable measurement, not
    a training step.
    """)
    return


@app.function
@torch.no_grad()
def evaluate(model, batches):
    # Torch tools for this gap: notebooks/torch_toolbox.py, section 05.
    # Goal: the mean cross-entropy loss of the model over a list of (x, y)
    # batches, with dropout switched off.
    # - Switch the model to eval mode and keep an empty list for the losses.
    # - For every (x, y): run the model on x. The logits have the shape
    #   (batch_size, context_length, vocab_size).
    # - F.cross_entropy wants a flat list of predictions and a flat list of
    #   targets. Reshape the logits to (batch_size * context_length,
    #   vocab_size) and y to (batch_size * context_length,). Tip: .view with
    #   -1 for one dimension lets PyTorch work that size out for you.
    # - Store loss.item() in the list: a plain float, not a tensor.
    # - Switch back to train mode and return the mean of the list, that is
    #   the sum divided by the number of batches.
    raise NotImplementedError("implement evaluate, Raschka 5.1")


@app.cell
def _(device, val_data):
    torch.manual_seed(0)
    eval_smoke_model = GPTModel(build_config()).to(device)
    eval_smoke_batches = [get_batch(val_data, 8, CONTEXT_LENGTH, device) for _ in range(4)]
    eval_smoke_loss = evaluate(eval_smoke_model, eval_smoke_batches)
    assert math.isfinite(eval_smoke_loss) and eval_smoke_loss > 0
    mo.md(f"evaluate() on an untrained model: {eval_smoke_loss:.3f} nats (expect close to ln(50257) &asymp; 10.8)")
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## The learning rate schedule, Raschka 5.2 (given)

    Linear warmup from 0 up to `MAX_LR` over the first `WARMUP_STEPS`
    steps, then cosine decay from `MAX_LR` down to `MIN_LR` as
    `progress` runs from 0 to 1. `progress` is computed by the loop
    that calls `lr_at`, not by `lr_at` itself: from steps remaining in
    step mode, from wall clock time remaining in budget mode, so the
    same function serves both.
    """)
    return


@app.function
def lr_at(step, progress):
    if step < WARMUP_STEPS:
        # warmup: the rate grows linearly, and even step 0 gets a small rate above zero
        return MAX_LR * (step + 1) / WARMUP_STEPS
    # cosine decay: cos(pi * progress) falls from 1 to -1, so the rate falls smoothly from MAX_LR to MIN_LR
    return MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1 + math.cos(math.pi * progress))


@app.cell
def _():
    assert lr_at(0, 0.0) == MAX_LR * (0 + 1) / WARMUP_STEPS, "lr_at(0, 0.0) should be the first warmup value"
    assert lr_at(WARMUP_STEPS, 0.0) == MAX_LR, "lr_at(WARMUP_STEPS, 0.0) should be MAX_LR, the warmup/decay boundary"
    assert lr_at(WARMUP_STEPS, 1.0) == MIN_LR, "lr_at(WARMUP_STEPS, 1.0) should be MIN_LR, progress fully run out"
    mo.md("lr_at() asserts: first warmup value, MAX_LR at the warmup/decay boundary, MIN_LR once progress reaches 1.0")
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## One training step, Raschka 5.2 (given)

    Forward pass under bf16 autocast on GPU (plain fp32 on CPU),
    cross-entropy on the flattened logits and targets, zero the
    gradients, backward, clip the gradient norm to `GRAD_CLIP`, step
    the optimizer. Returns the loss as a plain float, not a tensor:
    the training loop below only ever needs the number, and holding
    onto the tensor instead would keep its autograd graph alive for no
    reason.
    """)
    return


@app.function
def train_step(model, optimizer, x, y, device):
    # x, y: (batch_size, context_length) each
    use_bf16 = device.type == "cuda"  # lower precision only on the GPU
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
        logits = model(x)  # (batch_size, context_length, vocab_size)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))  # the same loss as in evaluate()

    optimizer.zero_grad(set_to_none=True)  # forget the gradients of the last step
    loss.backward()  # gradient of the loss for every parameter
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)  # keep one bad batch from pushing too hard
    optimizer.step()  # change the parameters
    return loss.item()  # a plain float, not a tensor, so the computation graph can be freed


@app.cell
def _(device, val_data):
    torch.manual_seed(0)
    train_step_smoke_model = GPTModel(build_config()).to(device)
    train_step_smoke_optimizer = torch.optim.AdamW(train_step_smoke_model.parameters(), lr=MAX_LR)
    train_step_smoke_x, train_step_smoke_y = get_batch(val_data, 8, CONTEXT_LENGTH, device)

    train_step_losses = [
        train_step(train_step_smoke_model, train_step_smoke_optimizer, train_step_smoke_x, train_step_smoke_y, device)
        for _ in range(5)
    ]
    assert math.isfinite(train_step_losses[0])
    assert train_step_losses[-1] < train_step_losses[0], "five steps on the same batch should lower the loss"
    mo.md(f"train_step() smoke test: loss {train_step_losses[0]:.3f} -> {train_step_losses[-1]:.3f} over 5 steps on the same batch")
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## The training loop, Raschka 5.2 (given, you do not need to read the code)

    `train_model` is a loop that ties the pieces together, its code is
    hidden. In plain words, for every step:

    1. ask `lr_at` for this step's learning rate
    2. draw a batch with `get_batch` from notebook 01
    3. let `train_step` learn from it
    4. every 100 steps, measure the validation loss with your `evaluate`
    5. every 250 steps, let the model write a short story, so you can
       watch it improve

    It stops when the time is up (this notebook's 12 minute run) or after a
    fixed number of steps (day 3's ablation harness uses that mode).
    """)
    return


@app.function(hide_code=True)
def train_model(
    model,
    train_data,
    val_data,
    device,
    batch_size,
    context_length,
    num_steps=None,
    budget_seconds=None,
    seed=42,
    sample_every_steps=None,
    sample_prompt_ids=None,
    sample_tokens=200,
    tokenizer=None,
):
    """Train for a fixed step count, or for a wall clock budget.

    Pass exactly one of `num_steps` or `budget_seconds`.
    """
    assert (num_steps is None) != (budget_seconds is None), "pass exactly one of num_steps or budget_seconds"

    torch.manual_seed(seed)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=0.1, betas=(0.9, 0.95))

    val_batches = [get_batch(val_data, batch_size, context_length, device) for _ in range(EVAL_ITERS)]

    history = []
    samples = []
    tokens_seen = 0
    step = 0
    warmup_end_time = None

    start_time = time.time()
    # In step mode there is no deadline; math.inf keeps deadline a plain
    # float instead of float | None, and is never actually compared
    # against since every branch below checks num_steps first.
    deadline = start_time + budget_seconds if budget_seconds is not None else math.inf

    while (step < num_steps) if num_steps is not None else (time.time() < deadline):
        now = time.time()
        if step < WARMUP_STEPS:
            progress = 0.0  # unused by lr_at's warmup branch, kept defined for clarity
        elif num_steps is not None:
            progress = (step - WARMUP_STEPS) / max(num_steps - WARMUP_STEPS, 1)
            progress = min(progress, 1.0)
        else:
            if warmup_end_time is None:
                warmup_end_time = now
            progress = (now - warmup_end_time) / max(deadline - warmup_end_time, 1e-6)
            progress = min(progress, 1.0)
        lr = lr_at(step, progress)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = get_batch(train_data, batch_size, context_length, device)  # (batch_size, context_length) each
        train_loss = train_step(model, optimizer, x, y, device)

        tokens_seen += x.numel()
        elapsed = time.time() - start_time
        at_last_step = (step == num_steps - 1) if num_steps is not None else (time.time() >= deadline)

        val_loss = None
        if step % EVAL_EVERY_STEPS == 0 or at_last_step:
            val_loss = evaluate(model, val_batches)

        history.append(
            {
                "step": step,
                "elapsed_seconds": elapsed,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "tokens_seen": tokens_seen,
                "lr": lr,
            }
        )

        if sample_every_steps and step % sample_every_steps == 0 and sample_prompt_ids is not None:
            sample_ids = generate(model, sample_prompt_ids, sample_tokens, context_length)
            sample_text = tokenizer.decode(sample_ids[0].tolist())
            samples.append({"step": step, "elapsed_seconds": elapsed, "text": sample_text})

        step += 1

    total_elapsed = time.time() - start_time
    final_metrics = {
        "steps": step,
        "elapsed_seconds": total_elapsed,
        "tokens_seen": tokens_seen,
        "tokens_per_second": tokens_seen / total_elapsed,
        "train_loss": history[-1]["train_loss"],
        "val_loss": evaluate(model, val_batches),
    }
    return history, samples, final_metrics


@app.cell
def _(device, train_data, val_data):
    torch.manual_seed(0)
    train_smoke_model = GPTModel(build_config()).to(device)
    train_smoke_history, _train_smoke_samples, train_smoke_final = train_model(
        train_smoke_model, train_data, val_data, device,
        batch_size=8, context_length=CONTEXT_LENGTH, num_steps=5,
    )
    assert train_smoke_final["steps"] == 5
    assert len(train_smoke_history) == 5
    assert math.isfinite(train_smoke_final["train_loss"])
    mo.md(f"train_model() smoke test: {train_smoke_final['steps']} steps, final train loss {train_smoke_final['train_loss']:.3f}")
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## The 12 minute run

    Training only starts when you press the run button, or automatically
    when this file runs as a script. Without that gate, marimo's
    reactivity would restart a multi-minute training run every time an
    unrelated cell changed. Sampling every 250 steps from a fixed prompt
    lets you watch coherence appear.
    """)
    return


@app.cell(hide_code=True)
def _():
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell(hide_code=True)
def _():
    run_seconds = mo.ui.number(value=720, start=60, stop=3600, label="Training seconds (12 min default)")
    run_prompt = mo.ui.text(value="Once upon a time", label="Generation prompt")
    run_button = mo.ui.run_button(label="Run pretraining")
    mo.vstack([run_seconds, run_prompt, run_button])
    return run_button, run_prompt, run_seconds


@app.cell(hide_code=True)
def _():
    # Preserves the run's result across reactive re-runs that are not a
    # fresh button click, see 99_calibration.py for the full reasoning.
    get_run_result, set_run_result = mo.state(None)
    return get_run_result, set_run_result


@app.cell(hide_code=True)
def _(
    device,
    gpt2_tokenizer,
    is_script_mode,
    run_button,
    run_prompt,
    run_seconds,
    set_run_result,
    train_data,
    val_data,
):
    if is_script_mode or run_button.value:
        torch.manual_seed(42)
        _cfg = build_config()
        _model = GPTModel(_cfg).to(device)

        _prompt_ids = torch.tensor([gpt2_tokenizer.encode_ordinary(run_prompt.value)], device=device)

        _history, _samples, _final = train_model(
            _model,
            train_data,
            val_data,
            device,
            batch_size=BATCH_SIZE,
            context_length=CONTEXT_LENGTH,
            budget_seconds=run_seconds.value,
            sample_every_steps=250,
            sample_prompt_ids=_prompt_ids,
            sample_tokens=200,
            tokenizer=gpt2_tokenizer,
        )

        set_run_result({"cfg": _cfg, "model": _model, "history": _history, "samples": _samples, "final": _final})
    return


@app.cell(hide_code=True)
def _(get_run_result):
    _result = get_run_result()
    if _result is None:
        samples_md = mo.md("Run the training above to see generation samples here.")
    else:
        samples_md = mo.md(
            "\n\n---\n\n".join(
                f"**step {s['step']}** ({s['elapsed_seconds']:.0f}s)\n\n```\n{s['text']}\n```"
                for s in _result["samples"]
            )
        )
    samples_md
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Loss curve

    Train and validation loss together, as always. If train loss sits
    above val loss here, that is expected: train loss is a single noisy
    batch measured with dropout on, val loss is dropout off, averaged
    over EVAL_ITERS batches.
    """)
    return


@app.cell(hide_code=True)
def _(get_run_result):
    _result = get_run_result()
    _history = _result["history"] if _result is not None else []
    _steps = [row["step"] for row in _history]
    _train_loss = [row["train_loss"] for row in _history]
    _val_steps = [row["step"] for row in _history if row["val_loss"] is not None]
    _val_loss = [row["val_loss"] for row in _history if row["val_loss"] is not None]

    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(x=_steps, y=_train_loss, name="train loss", mode="lines"))
    loss_fig.add_trace(go.Scatter(x=_val_steps, y=_val_loss, name="val loss", mode="lines+markers"))
    loss_fig.update_layout(title="Pretraining loss", xaxis_title="step", yaxis_title="cross-entropy loss")
    loss_fig
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Chat, checkpoint 3 of 5: your pretrained model

    Run the training above first. Expect a children's story that
    ignores the question: pretraining only ever taught next-token
    prediction on raw TinyStories, nothing about treating a prompt as
    an instruction yet.
    """)
    return


@app.cell(hide_code=True)
def _(get_run_result, gpt2_tokenizer):
    _result = get_run_result()
    if _result is None:
        chat_checkpoint_3 = mo.md("Run the training above to chat with your pretrained model here.")
    else:
        chat_checkpoint_3 = chat_window(
            _result["model"], gpt2_tokenizer, _result["cfg"]["context_length"],
            prompts=["Once upon a time", "The best way to learn a new language is"],
        )
    chat_checkpoint_3
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Push to the Hub

    The config and the loss history go up next to the checkpoint, not
    just the weights: a checkpoint without the config it was trained with
    is not reproducible.

    The repository name is derived, not typed: `checkpoint_repo_id()`
    asks the Hub who the current `HF_TOKEN` belongs to and builds
    `{username}/anlp-day2-pretrained` from that, so there is no
    `CHECKPOINT_REPO_ID` to edit by hand here and again in `07_sft.py`,
    and nothing to get out of sync between the two. Private, not
    public: there is no reason for every student's checkpoint to be
    world-readable.
    """)
    return


@app.function(hide_code=True)
def checkpoint_repo_id():
    """f"{your Hub username}/anlp-day2-pretrained", derived from whoever
    HF_TOKEN belongs to. Not in the setup cell: 07_sft.py imports this
    notebook for train_model, and the setup cell runs on import, a
    network call there would hit the Hub, with a token, on every
    import.
    """
    load_dotenv()
    hf_token = os.environ["HF_TOKEN"]
    username = HfApi().whoami(token=hf_token)["name"]
    return f"{username}/anlp-day2-pretrained"


@app.cell(hide_code=True)
def _(get_run_result):
    _result = get_run_result()
    hub_api = None
    repo_id = None
    if _result is not None:
        load_dotenv()
        hf_token = os.environ["HF_TOKEN"]
        repo_id = checkpoint_repo_id()

        checkpoint_path = Path("day2_checkpoint.pt")
        torch.save({"model_state_dict": _result["model"].state_dict()}, checkpoint_path)

        config_path = Path("day2_config.json")
        config_path.write_text(json.dumps(_result["cfg"], indent=2))

        history_path = Path("day2_history.json")
        history_path.write_text(json.dumps(_result["history"], indent=2))

        hub_api = HfApi(token=hf_token)
        hub_api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
        for _local_path, _repo_path in [
            (checkpoint_path, "checkpoint.pt"),
            (config_path, "config.json"),
            (history_path, "history.json"),
        ]:
            hub_api.upload_file(
                path_or_fileobj=_local_path, path_in_repo=_repo_path, repo_id=repo_id, repo_type="model"
            )
    return hub_api, repo_id


@app.cell(hide_code=True)
def _(hub_api, repo_id):
    if hub_api is None:
        push_md = mo.md("Run the training above to push a checkpoint.")
    else:
        push_md = mo.md(f"Pushed checkpoint, config and loss history to https://huggingface.co/{repo_id} (private)")
    push_md
    return


if __name__ == "__main__":
    app.run()

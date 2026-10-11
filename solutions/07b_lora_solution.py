import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import copy
    import importlib

    import numpy as np
    import plotly.graph_objects as go
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import marimo as mo

    # GPTModel, generate, get_instruct_batch, masked_mean and the data
    # loading are all defined in exactly one place, 07_sft.py (which
    # itself pulls GPTModel and generate from 04_gpt.py). Reused here,
    # not duplicated, so the fine-tuning loop below is provably the
    # same code block 07 already ran, with LoRA the only thing swapped
    # in.
    _sft = importlib.import_module("07_sft_solution")
    GPTModel = _sft.GPTModel
    generate = _sft.generate
    get_instruct_batch = _sft.get_instruct_batch
    masked_mean = _sft.masked_mean
    load_train_data = _sft.load_train_data
    # This demo always starts from the instructor's calibration checkpoint,
    # whose config is embedded in checkpoint.pt (see the loading cell below).
    CHECKPOINT_REPO_ID = _sft.INSTRUCTOR_CHECKPOINT_REPO_ID
    MAX_LEN = _sft.MAX_LEN
    FT_MAX_LR = _sft.FT_MAX_LR
    FT_NUM_STEPS = _sft.FT_NUM_STEPS
    FT_BATCH_SIZE = _sft.FT_BATCH_SIZE
    hf_hub_download = _sft.hf_hub_download
    hf_token = _sft.hf_token

    _gpt = importlib.import_module("04_gpt_solution")
    build_config = _gpt.build_config

    # Rank 8, alpha 16, all four attention projections (query, key,
    # value, output, not just query and value as some LoRA write-ups
    # default to): a reasonable, unturned default, not chosen to
    # flatter the result further down.
    LORA_RANK = 8
    LORA_ALPHA = 16


@app.cell(hide_code=True)
def _():
    torch.manual_seed(0)
    _preview_model = GPTModel(build_config())
    _total_params = sum(p.numel() for p in _preview_model.parameters())
    del _preview_model

    mo.md(f"""
    # 07b. LoRA

    An optional, self-paced extension. Block 07 only has slide time to
    show LoRA, not to build it. This notebook builds it by hand, reruns
    block 07's exact fine-tune with LoRA swapped in for full
    fine-tuning, and checks honestly whether it keeps up. It is meant
    to be read alone, with no one talking over it, so every claim below
    is something this notebook checks in front of you rather than
    something to take on trust.

    The honest part first, before any code makes it easy to skate past:
    this course's model has about {_total_params / 1e6:.0f} million
    parameters. It fits in memory, gradients and all, many times over
    on any machine that can run this notebook. LoRA's actual selling
    point, cutting optimizer memory for a model too large to fully
    fine-tune, does not apply at this size, at all. Nothing below will
    look dramatically cheaper, because at this scale it genuinely is
    not. What this notebook can show instead: the mechanism, verified
    on a model you built yourself, and the arithmetic for the scale
    where the argument is real.
    """)
    return


@app.cell(hide_code=True)
def _():
    _d = build_config()["emb_dim"]
    _r = LORA_RANK
    _full_update_params = _d * _d
    _lora_params = 2 * _d * _r

    mo.md(f"""
    ## The idea

    Full fine-tuning learns a weight update, &Delta;W, the same shape
    as the weight it updates. One attention projection in this course's
    model is a {_d} by {_d} matrix, so &Delta;W for it is
    {_full_update_params:,} numbers, with no constraint on any of them.

    LoRA's claim: that update does not need full rank. Approximate it
    as the product of two thin matrices instead, A (rank {_r} by
    {_d}) and B ({_d} by rank {_r}), so &Delta;W &asymp; B @ A. Storing
    and training A and B is {_lora_params:,} numbers instead of
    {_full_update_params:,}, about {_full_update_params / _lora_params:.0f}&times;
    fewer, for this one projection. Whether that approximation is
    actually good enough to match full fine-tuning is not assumed
    here, it gets checked below, on the real loss curves, not on this
    arithmetic.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## `LoRALinear`

    Wraps an existing `nn.Linear`: freezes its weight, adds two small
    trainable matrices `A` and `B` alongside it, and adds their product
    (scaled by `alpha / r`) to the base layer's own output. `B` starts
    at zero, so at the moment of injection, before a single training
    step, the adapted layer computes exactly what the base layer
    already computed. Checked below, not assumed.
    """)
    return


@app.class_definition
class LoRALinear(nn.Module):
    def __init__(self, base_linear, r=LORA_RANK, alpha=LORA_ALPHA):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        d_in = base_linear.in_features
        d_out = base_linear.out_features
        self.A = nn.Parameter(torch.empty(r, d_in))  # (r, d_in)
        nn.init.kaiming_uniform_(self.A, a=5**0.5)  # same scheme nn.Linear itself uses for its own weight
        self.B = nn.Parameter(torch.zeros(d_out, r))  # (d_out, r), zero init: adapted model starts identical to base
        self.scaling = alpha / r

    def forward(self, x):
        base_out = self.base(x)  # (..., d_out)
        lora_update = (x @ self.A.T) @ self.B.T  # (..., d_in) @ (d_in, r) -> (..., r), then @ (r, d_out) -> (..., d_out)
        return base_out + self.scaling * lora_update


@app.cell
def _():
    torch.manual_seed(0)
    _base_linear = nn.Linear(16, 16)
    _lora_linear = LoRALinear(_base_linear, r=LORA_RANK, alpha=LORA_ALPHA)
    _x = torch.randn(2, 5, 16)  # (b=2, num_tokens=5, d_in=16)
    assert torch.equal(_base_linear(_x), _lora_linear(_x)), "zero-initialized B must make the adapted layer start identical to the base"
    mo.md("LoRALinear() at init: output identical to the wrapped base layer, confirmed (B starts at zero)")
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Injection

    Load block 07's checkpoint, walk every transformer block, replace
    its four attention projections (`W_query`, `W_key`, `W_value`,
    `out_proj`) with `LoRALinear` wrappers around the already-loaded
    weights, and freeze everything that is not part of an adapter.
    `attn_mask` still works exactly as before: `LoRALinear` presents
    the same `forward(x)` interface as the `nn.Linear` it replaces, so
    nothing in `MultiHeadAttention` needs to know the difference.
    """)
    return


@app.function
def inject_lora(model, r=LORA_RANK, alpha=LORA_ALPHA):
    """Replace every attention projection with a LoRALinear wrapper, in
    place, and freeze every parameter that is not a LoRA A or B.
    """
    for param in model.parameters():
        param.requires_grad = False

    for block in model.trf_blocks:
        block.att.W_query = LoRALinear(block.att.W_query, r=r, alpha=alpha)
        block.att.W_key = LoRALinear(block.att.W_key, r=r, alpha=alpha)
        block.att.W_value = LoRALinear(block.att.W_value, r=r, alpha=alpha)
        block.att.out_proj = LoRALinear(block.att.out_proj, r=r, alpha=alpha)

    return model


@app.cell
def _():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (device,)


@app.cell
def _():
    # Same data as block 07, loaded the same way, imported rather than
    # re-downloaded with a second copy of the same code.
    train_tokens, train_loss_mask, train_attn_mask = load_train_data()
    return train_attn_mask, train_loss_mask, train_tokens


@app.cell
def _(device):
    lora_checkpoint_path = hf_hub_download(
        repo_id=CHECKPOINT_REPO_ID, filename="checkpoint.pt", repo_type="model", token=hf_token
    )
    lora_checkpoint = torch.load(lora_checkpoint_path, map_location="cpu", weights_only=True)
    # The calibration checkpoint embeds its config inside checkpoint.pt
    # itself, see the same note in 07_sft.py: your own anlp-day2-pretrained
    # repo saves config.json separately instead.
    lora_cfg = lora_checkpoint["config"]

    reference_model = GPTModel(lora_cfg)
    reference_model.load_state_dict(lora_checkpoint["model_state_dict"])
    reference_model.to(device)
    reference_num_params = sum(p.numel() for p in reference_model.parameters())

    mo.md(f"loaded {reference_num_params:,} parameters from `{CHECKPOINT_REPO_ID}`, config: {lora_cfg}")
    return (reference_model,)


@app.cell
def _(device, reference_model):
    # Inject into a copy, not reference_model itself: reference_model
    # stays the plain, un-adapted checkpoint, reused below both for
    # this assert and for the full fine-tune rerun to compare against.
    # Injected on CPU, then moved as one assembled model: A and B are
    # freshly created tensors with no device of their own yet, moving
    # the base and the adapter together in one .to(device) call avoids
    # ever having a model half on one device and half on another.
    lora_model = inject_lora(copy.deepcopy(reference_model).cpu(), r=LORA_RANK, alpha=LORA_ALPHA).to(device)

    _trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    _total = sum(p.numel() for p in lora_model.parameters())

    _test_x = torch.randint(0, lora_model.tok_emb.num_embeddings, (2, 32), device=device)  # (2, 32), synthetic, no real data needed
    lora_model.eval()
    reference_model.eval()
    with torch.no_grad():
        _reference_logits = reference_model(_test_x)  # (2, 32, vocab_size)
        _lora_logits = lora_model(_test_x)  # (2, 32, vocab_size)
    lora_model.train()
    reference_model.train()
    assert torch.allclose(_reference_logits, _lora_logits, atol=1e-4), "freshly injected LoRA (B=0) must match the un-injected checkpoint"

    mo.md(
        f"trainable: {_trainable:,} / {_total:,} ({_trainable / _total:.2%}). "
        "Injected model matches the un-injected checkpoint exactly, confirmed: B starts at zero, at model scale, not just for one toy layer."
    )
    return (lora_model,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Run it

    Same 500 steps, same learning rate, same seed, on the same data, as
    block 07's full fine-tune, rerun here so the comparison is against
    a curve this exact code just produced, not a number copied from
    another notebook. Both runs share one seed for the model side of
    things (dropout) and one seed for the data side (which rows
    `get_instruct_batch` draws), so any gap between the two curves is
    LoRA, not batch luck.

    Training only starts when you press the run button, or
    automatically when this file runs as a script.
    """)
    return


@app.cell
def _():
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell
def _():
    lora_run_button = mo.ui.run_button(label="Run both fine-tunes")
    lora_run_button
    return (lora_run_button,)


@app.cell
def _():
    get_lora_result, set_lora_result = mo.state(None)
    return get_lora_result, set_lora_result


@app.cell
def _(
    device,
    is_script_mode,
    lora_model,
    lora_run_button,
    reference_model,
    set_lora_result,
    train_attn_mask,
    train_loss_mask,
    train_tokens,
):
    def _run_finetune(model, trainable_params, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)  # get_instruct_batch draws rows via np.random, a separate generator from torch's
        model.train()
        optimizer = torch.optim.AdamW(trainable_params, lr=FT_MAX_LR, weight_decay=0.1, betas=(0.9, 0.95))
        history = []
        for step in range(FT_NUM_STEPS):
            x, y, y_loss_mask, x_attn_mask = get_instruct_batch(
                train_tokens, train_loss_mask, train_attn_mask, FT_BATCH_SIZE, device
            )
            logits = model(x, attn_mask=x_attn_mask)  # (batch_size, MAX_LEN - 1, vocab_size)
            per_token_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none"
            )  # (batch_size * (MAX_LEN - 1),)
            response_loss = masked_mean(per_token_loss, y_loss_mask)

            optimizer.zero_grad(set_to_none=True)
            response_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            history.append({"step": step, "response_loss": response_loss.item()})
        return history

    if is_script_mode or lora_run_button.value:
        _full_model = copy.deepcopy(reference_model)
        _full_history = _run_finetune(_full_model, list(_full_model.parameters()), seed=42)

        _lora_trainable = [p for p in lora_model.parameters() if p.requires_grad]
        _lora_history = _run_finetune(lora_model, _lora_trainable, seed=42)

        set_lora_result({"full_history": _full_history, "lora_history": _lora_history})
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Full fine-tune vs LoRA
    """)
    return


@app.cell
def _(get_lora_result):
    _result = get_lora_result()
    _full_history = _result["full_history"] if _result is not None else []
    _lora_history = _result["lora_history"] if _result is not None else []

    _full_steps = [row["step"] for row in _full_history]
    _full_loss = [row["response_loss"] for row in _full_history]
    _lora_steps = [row["step"] for row in _lora_history]
    _lora_loss = [row["response_loss"] for row in _lora_history]

    comparison_fig = go.Figure()
    comparison_fig.add_trace(go.Scatter(x=_full_steps, y=_full_loss, name="full fine-tune", mode="lines"))
    comparison_fig.add_trace(go.Scatter(x=_lora_steps, y=_lora_loss, name=f"LoRA (r={LORA_RANK})", mode="lines"))
    comparison_fig.update_layout(
        title="Response loss: full fine-tune vs LoRA", xaxis_title="step", yaxis_title="cross-entropy loss"
    )
    comparison_fig
    return


@app.cell(hide_code=True)
def _(get_lora_result):
    _result = get_lora_result()
    if _result is None:
        verdict_md = mo.md("Run both fine-tunes above to see the comparison and the verdict.")
    else:
        _full_first = _result["full_history"][0]["response_loss"]
        _full_last = _result["full_history"][-1]["response_loss"]
        _lora_last = _result["lora_history"][-1]["response_loss"]
        _gap = _lora_last - _full_last
        _full_drop = _full_first - _full_last

        if _gap <= 0.02:
            _verdict = (
                f"LoRA matched full fine-tuning: final loss {_lora_last:.3f} against {_full_last:.3f}, "
                f"a gap of {_gap:.3f}, small enough to be ordinary run-to-run noise rather than a real difference."
            )
        elif _gap <= 0.1:
            _verdict = (
                f"LoRA got close but not identical: final loss {_lora_last:.3f} against {_full_last:.3f}, "
                f"a gap of {_gap:.3f}. Bigger than step-to-step noise, still small next to the {_full_drop:.3f} "
                "full fine-tuning gained overall."
            )
        else:
            _verdict = (
                f"LoRA fell short: final loss {_lora_last:.3f} against {_full_last:.3f}, a gap of {_gap:.3f}, "
                f"a real fraction of the {_full_drop:.3f} full fine-tuning gained overall. Rank {LORA_RANK} on "
                "attention alone was not enough here. That is the finding, not a bug to fix by re-running until "
                "it looks better, see the suggestions at the end of this notebook for what to try instead."
            )

        verdict_md = mo.md(f"""
        {_verdict}

        Judged from the numbers above, not from how the plot looks:
        report what happened, not what was hoped for.
        """)
    verdict_md
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Merging

    A trained adapter does not have to stay separate from the base
    model. `B @ A * alpha / r` is a matrix the same shape as the base
    weight, so it can be folded directly into it: add it once, and the
    result is a plain `nn.Linear` again, no `A`, no `B`, no extra
    matmul at inference. Checked below: if merging changed the model's
    output at all, this notebook would be presenting "adapted" and
    "merged" as equivalent when they were not.
    """)
    return


@app.function
def merge_lora(lora_linear):
    """Fold a trained LoRA adapter into its base nn.Linear, returning a
    plain nn.Linear with no adapter machinery left over.
    """
    merged = nn.Linear(
        lora_linear.base.in_features, lora_linear.base.out_features, bias=lora_linear.base.bias is not None
    )
    delta_w = lora_linear.scaling * (lora_linear.B @ lora_linear.A)  # (d_out, r) @ (r, d_in) -> (d_out, d_in)
    merged.weight.data = lora_linear.base.weight.data + delta_w  # (d_out, d_in)
    if lora_linear.base.bias is not None:
        merged.bias.data = lora_linear.base.bias.data.clone()
    return merged


@app.function
def merge_lora_model(model):
    """Replace every LoRALinear in a model's attention blocks with its
    merged, plain nn.Linear equivalent. In place.
    """
    for block in model.trf_blocks:
        block.att.W_query = merge_lora(block.att.W_query)
        block.att.W_key = merge_lora(block.att.W_key)
        block.att.W_value = merge_lora(block.att.W_value)
        block.att.out_proj = merge_lora(block.att.out_proj)
    return model


@app.cell
def _(device, get_lora_result, lora_model):
    _result = get_lora_result()
    if _result is None:
        merge_md = mo.md("Run fine-tuning above first, merging an untrained adapter would just repeat the assert already done at injection.")
    else:
        lora_model.eval()
        _test_x = torch.randint(0, lora_model.tok_emb.num_embeddings, (2, 32), device=device)  # (2, 32), synthetic
        with torch.no_grad():
            _unmerged_logits = lora_model(_test_x)  # (2, 32, vocab_size)

        _merged_model = merge_lora_model(copy.deepcopy(lora_model).cpu()).to(device)
        _merged_model.eval()
        with torch.no_grad():
            _merged_logits = _merged_model(_test_x)  # (2, 32, vocab_size)
        lora_model.train()

        assert torch.allclose(_unmerged_logits, _merged_logits, atol=1e-4)
        merge_md = mo.md(
            "Merged vs unmerged logits, after training: identical, confirmed. A merged adapter is a normal "
            "`nn.Linear` again, no extra parameters, no extra matmul, nothing left over at inference that was not "
            "already there before fine-tuning. This is why adapters get merged before serving in production rather "
            "than kept separate."
        )
    merge_md
    return


@app.cell(hide_code=True)
def _(lora_model):
    _trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    _total = sum(p.numel() for p in lora_model.parameters())
    _bytes_per_state = 4  # fp32
    _states_per_param = 2  # AdamW keeps a first and a second moment per trainable parameter

    def _optimizer_gb(num_params):
        return num_params * _states_per_param * _bytes_per_state / 1e9

    _this_full_gb = _optimizer_gb(_total)
    _this_lora_gb = _optimizer_gb(_trainable)

    _seven_b = 7_000_000_000
    _seven_b_fraction = _trainable / _total  # this model's own trainable fraction, used as a stand-in for a 7B model's
    _seven_b_trainable = _seven_b * _seven_b_fraction
    _seven_b_full_gb = _optimizer_gb(_seven_b)
    _seven_b_lora_gb = _optimizer_gb(_seven_b_trainable)

    mo.md(f"""
    ## The arithmetic at scale

    AdamW keeps two optimizer states per trainable parameter (a first
    and a second moment), in fp32, 4 bytes each: 8 bytes of optimizer
    state per trainable parameter, on top of the parameter itself and
    its gradient. That scales with trainable parameters, not total
    parameters, which is the entire argument for LoRA:

    | | trainable params | optimizer state |
    |---|---|---|
    | this model, full fine-tune | {_total:,} | {_this_full_gb * 1000:.1f} MB |
    | this model, LoRA (r={LORA_RANK}) | {_trainable:,} | {_this_lora_gb * 1000:.2f} MB |
    | 7B model, full fine-tune | {_seven_b:,} | {_seven_b_full_gb:.0f} GB |
    | 7B model, LoRA (same {_seven_b_fraction:.2%} trainable fraction) | {_seven_b_trainable:,.0f} | {_seven_b_lora_gb:.2f} GB |

    (the 7B row assumes the same trainable fraction measured on this
    model, a stand-in for a real deployment's own choice of rank and
    target modules, not a specific published config)

    At this model's scale, the LoRA row is a rounding error either way,
    both fit trivially, which is exactly what the honest problem at the
    top of this notebook said would happen. At 7B, full fine-tuning's
    optimizer state alone is {_seven_b_full_gb:.0f} GB, before the model
    weights, activations or gradients are even counted, often the
    difference between fitting on one GPU and needing several. That
    gap, not the one two rows up, is what LoRA is actually for.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## The same thing, with PEFT

    Not run here, `peft` is not a dependency of this notebook, just
    what the three lines above look like with the library people
    actually reach for at work instead of a hand-rolled `LoRALinear`:

    ```python
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(r=8, lora_alpha=16, target_modules=["W_query", "W_key", "W_value", "out_proj"])
    model = get_peft_model(model, config)
    ```

    Same idea as `inject_lora` above: freeze the base, attach A and B
    to named modules, train only those. `peft` handles far more than
    this notebook does (more architectures, quantized base weights,
    several adapters loaded at once), which is exactly why it is the
    one to reach for outside a course built to show the mechanism by
    hand.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## If you have time left

    No solutions here, just places to poke:

    - Vary `r` (4, 16, 32, ...) and watch where the loss curve stops
      moving. Somewhere it should flatten: past some rank, the
      approximation stops being the bottleneck and something else is.
    - Adapt the feed-forward layers (`block.ff.layers[0]` and
      `block.ff.layers[2]`) as well as attention. Most of a
      transformer's parameters live there, not in attention, so this
      changes both the trainable fraction and, possibly, the result.
    - Train two adapters on two different slices of the data, keep one
      frozen base, and swap which adapter is attached without touching
      the base weights at all. This is most of the reason LoRA composes
      well in production: one base model, many small adapters.
    """)
    return


if __name__ == "__main__":
    app.run()

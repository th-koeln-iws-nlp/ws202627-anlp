import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import copy
    import importlib
    import json
    import os
    from pathlib import Path

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import tiktoken
    import torch
    import torch.nn.functional as F
    import marimo as mo
    from dotenv import load_dotenv
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import RepositoryNotFoundError

    # GPTModel and generate are defined in exactly one place, 04_gpt.py.
    _gpt = importlib.import_module("04_gpt")
    GPTModel = _gpt.GPTModel
    generate = _gpt.generate
    chat_window = _gpt.chat_window

    # checkpoint_repo_id is defined in exactly one place, 05_pretraining.py.
    _pretraining = importlib.import_module("05_pretraining")
    checkpoint_repo_id = _pretraining.checkpoint_repo_id

    load_dotenv()
    hf_token = os.environ["HF_TOKEN"]

    DATASET_REPO_ID = "Richard-Sieg-TH-Koln/anlp-tinystories-instruct-gpt2"
    MAX_LEN = 512  # matches CONTEXT_LENGTH everywhere else in the course

    # Fallback if your own pretraining run failed: load the
    # instructor's calibration checkpoint instead of your own, so you
    # can still work through fine-tuning. Flip back to False once your
    # own checkpoint exists.
    USE_INSTRUCTOR_CHECKPOINT = False
    INSTRUCTOR_CHECKPOINT_REPO_ID = "Richard-Sieg-TH-Koln/anlp-calibration-checkpoint-small"

    # Settled: 500 steps, lr 1e-4, batch 64, response loss 2.08 to
    # 1.69. 1,500 steps was also measured and only reaches 1.62, see
    # the markdown cell below for why that is not worth running here.
    FT_MAX_LR = 1e-4
    FT_NUM_STEPS = 500
    FT_BATCH_SIZE = 64


@app.cell(hide_code=True)
def _():
    mo.md("""
    # 07. Supervised fine-tuning

    Raschka chapter 7. Same model, same cross-entropy loss as
    pretraining (block 05), different data: prompt/response pairs
    instead of raw story text, plus a mask over the loss so only the
    response is trained on. Padding, and the two masks it forces, is
    the only genuinely new machinery, everything else is day 2 again.

    In this notebook you write one function, `masked_mean`. Everything
    else is given: read it, run it, and look at what the masks do.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## The data

    Prompt/response pairs built from TinyStoriesInstruct, tokenized,
    right-padded to 512, with a `loss_mask` (1 on response tokens) and
    an `attn_mask` (1 on every real, non-padding token) sitting next to
    the tokens. Loaded straight from the Hub, already built, no
    re-tokenizing here.
    """)
    return


@app.cell
def _():
    gpt2_tokenizer = tiktoken.get_encoding("gpt2")
    return (gpt2_tokenizer,)


@app.function(hide_code=True)
def load_train_data():
    """Download and load the tokenized instruct dataset's train split."""
    train_tokens = np.load(
        hf_hub_download(repo_id=DATASET_REPO_ID, filename="train_tokens.npy", repo_type="dataset", token=hf_token)
    )
    train_loss_mask = np.load(
        hf_hub_download(repo_id=DATASET_REPO_ID, filename="train_loss_mask.npy", repo_type="dataset", token=hf_token)
    )
    train_attn_mask = np.load(
        hf_hub_download(repo_id=DATASET_REPO_ID, filename="train_attn_mask.npy", repo_type="dataset", token=hf_token)
    )
    return train_tokens, train_loss_mask, train_attn_mask


@app.cell(hide_code=True)
def _():
    train_tokens, train_loss_mask, train_attn_mask = load_train_data()
    mo.md(f"loaded {train_tokens.shape[0]:,} tokenized train examples from `{DATASET_REPO_ID}`")
    return train_attn_mask, train_loss_mask, train_tokens


@app.cell
def _():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (device,)


@app.cell(hide_code=True)
def _(gpt2_tokenizer, train_attn_mask, train_loss_mask, train_tokens):
    def _aligned_rows(tokens, loss_mask, attn_mask, start, end):
        """Per-token text, loss_mask and attn_mask, column-aligned in a monospace block."""
        token_strs = [gpt2_tokenizer.decode([int(t)]).replace("\n", "\\n") for t in tokens[start:end]]
        widths = [max(len(s), 1) for s in token_strs]
        token_row = " ".join(s.ljust(w) for s, w in zip(token_strs, widths))
        loss_row = " ".join(str(int(v)).ljust(w) for v, w in zip(loss_mask[start:end], widths))
        attn_row = " ".join(str(int(v)).ljust(w) for v, w in zip(attn_mask[start:end], widths))
        return f"text: {token_row}\nloss: {loss_row}\nattn: {attn_row}"

    _row = 0
    _tokens = train_tokens[_row]
    _loss_mask = train_loss_mask[_row]
    _attn_mask = train_attn_mask[_row]
    _prompt_end = int((_loss_mask == 0).sum() - (_attn_mask == 0).sum())  # first response token
    _real_end = int(_attn_mask.sum())  # first padding token

    _decoded_prompt = gpt2_tokenizer.decode(_tokens[:_prompt_end].tolist())
    _decoded_response = gpt2_tokenizer.decode(_tokens[_prompt_end:_real_end].tolist())
    _boundary_1 = _aligned_rows(_tokens, _loss_mask, _attn_mask, max(0, _prompt_end - 6), _prompt_end + 6)
    _boundary_2 = _aligned_rows(
        _tokens, _loss_mask, _attn_mask, max(0, _real_end - 6), min(len(_tokens), _real_end + 6)
    )

    mo.md(
        f"""
        Example 0 from `train`, decoded in full, then both masks lined
        up token by token at the two places they actually change:
        prompt turning into response, and real tokens turning into
        padding.

        ```
        PROMPT:
        {_decoded_prompt}
        RESPONSE:
        {_decoded_response}
        ```

        prompt &rarr; response boundary (`loss_mask` flips 0 to 1, `attn_mask` stays 1, still a real token, just no longer one the loss looks at):

        ```
        {_boundary_1}
        ```

        real &rarr; padding boundary (`attn_mask` flips 1 to 0, `loss_mask` was already 0 here too):

        ```
        {_boundary_2}
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## A new batch sampler (given)

    Day 2's `get_batch` (`01_text_data.py`) slices random windows out of
    one flat token stream. That does not apply here, there is no
    stream: every row in `train_tokens` is already one complete, padded
    example.

    `get_instruct_batch` picks `batch_size` random rows and reads `x`
    and `y` off them shifted by one token, the same next-token
    prediction shift `get_batch` uses. `loss_mask` and `attn_mask` are
    sliced the same way, in step with the tokens, so all four still
    line up row for row.
    """)
    return


@app.function
def get_instruct_batch(tokens, loss_mask, attn_mask, batch_size, device):
    """Random rows from a padded instruct dataset, not random windows.

    Returns (x, y, y_loss_mask, x_attn_mask): x and y are tokens shifted
    by one for next-token prediction, the two masks are shifted and/or
    sliced to line up with them (y_loss_mask against the targets,
    x_attn_mask against the input, since that is what MultiHeadAttention
    needs to know which key positions to ignore).
    """
    n_examples = tokens.shape[0]
    row_indices = np.random.randint(0, n_examples, size=batch_size)  # (batch_size,), the same rows for all four arrays

    x = torch.from_numpy(tokens[row_indices, :-1].astype(np.int64)).to(device)  # (batch_size, MAX_LEN - 1)
    y = torch.from_numpy(tokens[row_indices, 1:].astype(np.int64)).to(device)  # (batch_size, MAX_LEN - 1)
    y_loss_mask = torch.from_numpy(loss_mask[row_indices, 1:].astype(np.int64)).to(device)  # shifted like y
    x_attn_mask = torch.from_numpy(attn_mask[row_indices, :-1].astype(np.int64)).to(device)  # sliced like x

    return x, y, y_loss_mask, x_attn_mask


@app.cell
def _(device, train_attn_mask, train_loss_mask, train_tokens):
    _demo_x, _demo_y, _demo_loss_mask, _demo_attn_mask = get_instruct_batch(
        train_tokens, train_loss_mask, train_attn_mask, batch_size=4, device=device
    )
    assert _demo_x.shape == _demo_y.shape == _demo_loss_mask.shape == _demo_attn_mask.shape == (4, MAX_LEN - 1)
    mo.md(f"get_instruct_batch() shapes: {tuple(_demo_x.shape)}")
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## The masked loss, your gap

    Pretraining's loss, in full, two lines:

    ```
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    ```

    Every position in every example contributes equally. Fine-tuning
    changes exactly one thing: only response positions should
    contribute. `attn_mask` already keeps the model itself from
    attending to padding (`04_gpt.py`), this is a separate mask, over
    the loss, that additionally excludes the prompt.

    Implement `masked_mean` first: the mean of a tensor of per-position
    values, restricted to the positions where a same-shaped mask is 1.

    Then the loss becomes three lines instead of two: run the model
    with `attn_mask`, compute the per-token loss with
    `reduction="none"` instead of letting `cross_entropy` average it
    for you, then `masked_mean` that against `loss_mask`.
    """)
    return


@app.function
def masked_mean(per_token_values, mask):
    """Mean of per_token_values over positions where mask is 1."""
    # Tools for this gap: notebooks/torch_toolbox.py, section 07.
    # per_token_values and mask have the same shape. The result is one number:
    # the average of per_token_values over the positions where mask is 1.
    # - Flatten the mask to 1D and turn it into floats (.reshape(-1), .float()).
    # - Multiply the flattened values by the mask: masked-out positions become
    #   0. Add everything up with .sum().
    # - Divide by the number of positions that count (the sum of the mask), not
    #   by the total number of positions.
    # - An all-zero mask would divide by zero: clamp that count to at least 1
    #   (Tensor.clamp with min=1).
    raise NotImplementedError("implement masked_mean, Raschka 7.3")


@app.cell
def _(device):
    torch.manual_seed(0)
    _logits = torch.randn(4, 6, 50, device=device)  # (batch, tokens, vocab), synthetic, no real model needed
    _targets = torch.randint(0, 50, (4, 6), device=device)
    _per_token_loss = F.cross_entropy(_logits.reshape(-1, 50), _targets.reshape(-1), reduction="none")

    _all_ones_mask = torch.ones(4, 6, device=device)
    _plain_loss = F.cross_entropy(_logits.reshape(-1, 50), _targets.reshape(-1))
    assert torch.allclose(masked_mean(_per_token_loss, _all_ones_mask), _plain_loss, atol=1e-5)

    _half_mask = torch.zeros(4, 6, device=device)
    _half_mask[:, :3] = 1
    assert torch.allclose(masked_mean(_per_token_loss, _half_mask), _per_token_loss.view(4, 6)[:, :3].mean(), atol=1e-5)

    mo.md(
        "masked_mean() asserts: a mask of all ones reproduces plain cross_entropy, "
        "a partial mask averages only the marked positions"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Before and after

    Load the base checkpoint (before any fine-tuning), generate from a
    held-out prompt in the fine-tuning format, not itself part of the
    dataset. Fine-tune below, then come back and generate from the same
    prompt again. The base model already writes fluent stories on its
    own, from a plain prompt, so this is the one place in the block
    where the actual difference fine-tuning makes is visible, rather
    than assumed from a loss number going down.
    """)
    return


@app.cell(hide_code=True)
def _(device):
    ft_repo_id = INSTRUCTOR_CHECKPOINT_REPO_ID if USE_INSTRUCTOR_CHECKPOINT else checkpoint_repo_id()
    try:
        ft_checkpoint_path = hf_hub_download(
            repo_id=ft_repo_id, filename="checkpoint.pt", repo_type="model", token=hf_token
        )
        ft_checkpoint = torch.load(ft_checkpoint_path, map_location="cpu", weights_only=True)
        if USE_INSTRUCTOR_CHECKPOINT:
            # The instructor's checkpoint embeds its config inside
            # checkpoint.pt itself. Your own anlp-day2-pretrained push
            # (05_pretraining.py) instead saves it as a separate
            # config.json, checkpoint.pt there only has model_state_dict.
            ft_cfg = ft_checkpoint["config"]
        else:
            ft_config_path = hf_hub_download(
                repo_id=ft_repo_id, filename="config.json", repo_type="model", token=hf_token
            )
            ft_cfg = json.loads(Path(ft_config_path).read_text())
    except RepositoryNotFoundError as e:
        raise RuntimeError(
            f"No checkpoint found at `{ft_repo_id}`: push one from 05_pretraining.py first, or set "
            "USE_INSTRUCTOR_CHECKPOINT = True above to work from the instructor's checkpoint instead."
        ) from e

    ft_model = GPTModel(ft_cfg).to(device)
    ft_model.load_state_dict(ft_checkpoint["model_state_dict"])
    ft_model
    return ft_cfg, ft_model, ft_repo_id


@app.cell(hide_code=True)
def _(ft_cfg, ft_model, ft_repo_id):
    ft_num_params = sum(p.numel() for p in ft_model.parameters())
    mo.md(f"loaded {ft_num_params:,} parameters from `{ft_repo_id}`, config: {ft_cfg}")
    return


@app.cell(hide_code=True)
def _():
    held_out_prompt = (
        "Summary: A brave little fox helps a lost duckling find its way back to the pond.\n"
        "Features: Dialogue\n"
        "Story:\n"
    )
    return (held_out_prompt,)


@app.cell(hide_code=True)
def _(device, ft_model, gpt2_tokenizer, held_out_prompt):
    _before_ids = torch.tensor([gpt2_tokenizer.encode_ordinary(held_out_prompt)], device=device)
    _before_output = generate(ft_model, _before_ids, max_new_tokens=200, context_length=MAX_LEN)
    before_text = gpt2_tokenizer.decode(_before_output[0].tolist())
    return (before_text,)


@app.cell(hide_code=True)
def _(before_text, device, get_ft_result, gpt2_tokenizer, held_out_prompt):
    _result = get_ft_result()
    if _result is None:
        _after_block = "*(run fine-tuning below, this cell fills in once it finishes)*"
    else:
        _after_ids = torch.tensor([gpt2_tokenizer.encode_ordinary(held_out_prompt)], device=device)
        _after_output = generate(_result["model"], _after_ids, max_new_tokens=200, context_length=MAX_LEN)
        _after_block = f"```\n{gpt2_tokenizer.decode(_after_output[0].tolist())}\n```"

    mo.md(
        f"""
        PROMPT:
        ```
        {held_out_prompt}
        ```

        BEFORE (base checkpoint, never seen an instruction):
        ```
        {before_text}
        ```

        AFTER (fine-tuned):
        {_after_block}

        The base model treats `Story:` as one more labeled field and
        continues in the same descriptive style as the prompt. The
        fine-tuned model should write the story itself, and honour
        `Features: Dialogue` (look for quoted speech in the output).
        Judge the actual text above, that is the real test here, not
        the loss curve below.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(f"""
    ## Fine-tuning

    {FT_NUM_STEPS} steps, lr {FT_MAX_LR}, batch {FT_BATCH_SIZE}, width
    {MAX_LEN}. Response loss goes 2.08 to 1.69. 1,500 steps was also
    measured and only reaches 1.62: three times the steps for a
    fraction of the remaining gain, not worth it in an 85 minute block.
    Stopping once a curve visibly flattens, rather than chasing a
    target number, is the same judgment call day 3's ablations need.

    Training only starts when you press the run button, or
    automatically when this file runs as a script.
    """)
    return


@app.cell(hide_code=True)
def _():
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell(hide_code=True)
def _():
    ft_run_button = mo.ui.run_button(label="Run fine-tuning")
    ft_run_button
    return (ft_run_button,)


@app.cell(hide_code=True)
def _():
    # mo.state, not a plain returned variable: preserves the
    # fine-tuning result across reactive re-runs that are not a fresh
    # button click, which a plain return would otherwise lose.
    get_ft_result, set_ft_result = mo.state(None)
    return get_ft_result, set_ft_result


@app.cell
def _(
    device,
    ft_model,
    ft_run_button,
    is_script_mode,
    set_ft_result,
    train_attn_mask,
    train_loss_mask,
    train_tokens,
):
    if is_script_mode or ft_run_button.value:
        torch.manual_seed(42)
        # A copy, not ft_model itself: the before/after cell above reads
        # ft_model directly for the "before" text, and that has to stay
        # the untouched base checkpoint no matter what order marimo
        # happens to run cells in, not depend on whether it ran before
        # or after this cell mutated ft_model in place.
        _model = copy.deepcopy(ft_model).to(device)
        _model.train()
        _optimizer = torch.optim.AdamW(_model.parameters(), lr=FT_MAX_LR, weight_decay=0.1, betas=(0.9, 0.95))

        _history = []
        for _step in range(FT_NUM_STEPS):
            _x, _y, _y_loss_mask, _x_attn_mask = get_instruct_batch(
                train_tokens, train_loss_mask, train_attn_mask, FT_BATCH_SIZE, device
            )
            _prompt_mask = (_x_attn_mask == 1) & (_y_loss_mask == 0)  # real but not response: (batch_size, MAX_LEN - 1)

            _logits = _model(_x, attn_mask=_x_attn_mask)  # (batch_size, MAX_LEN - 1, vocab_size)
            _per_token_loss = F.cross_entropy(
                _logits.reshape(-1, _logits.size(-1)), _y.reshape(-1), reduction="none"
            )  # (batch_size * (MAX_LEN - 1),)

            _response_loss = masked_mean(_per_token_loss, _y_loss_mask)
            _prompt_loss = masked_mean(_per_token_loss, _prompt_mask)

            _optimizer.zero_grad(set_to_none=True)
            _response_loss.backward()
            torch.nn.utils.clip_grad_norm_(_model.parameters(), 1.0)
            _optimizer.step()

            _history.append(
                {"step": _step, "response_loss": _response_loss.item(), "prompt_loss": _prompt_loss.item()}
            )

        set_ft_result({"model": _model, "history": _history})
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Loss curves

    Response loss (what fine-tuning actually optimizes) and prompt loss
    (diagnostic only) on separate panels, sharing steps on the x-axis:
    response loss sits around 1.6 to 2.1, prompt loss around 5.5 to
    6.1, on one shared axis the prompt trace would be a flat line at
    the bottom, not a finding.
    """)
    return


@app.cell(hide_code=True)
def _(get_ft_result):
    def _rolling_mean(values, window):
        """Trailing mean over the last `window` steps (fewer at the very start)."""
        return [sum(values[max(0, i - window + 1) : i + 1]) / min(i + 1, window) for i in range(len(values))]

    _result = get_ft_result()
    _history = _result["history"] if _result is not None else []
    _steps = [row["step"] for row in _history]
    _response_loss = [row["response_loss"] for row in _history]
    _prompt_loss = [row["prompt_loss"] for row in _history]
    _response_loss_smoothed = _rolling_mean(_response_loss, window=20)

    loss_fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("response loss (trained on)", "prompt loss (diagnostic only)"),
    )
    loss_fig.add_trace(
        go.Scatter(
            x=_steps, y=_response_loss, name="response loss, raw", mode="lines",
            line=dict(width=1, color="rgba(31,119,180,0.25)"),
        ),
        row=1, col=1,
    )
    loss_fig.add_trace(
        go.Scatter(
            x=_steps, y=_response_loss_smoothed, name="response loss, rolling mean (20 steps)", mode="lines",
            line=dict(width=2, color="rgba(31,119,180,1.0)"),
        ),
        row=1, col=1,
    )
    loss_fig.add_trace(
        go.Scatter(x=_steps, y=_prompt_loss, name="prompt loss", mode="lines", line=dict(color="firebrick")),
        row=2, col=1,
    )
    loss_fig.update_yaxes(title_text="cross-entropy loss", row=1, col=1)
    loss_fig.update_yaxes(title_text="cross-entropy loss", row=2, col=1)
    loss_fig.update_xaxes(title_text="step", row=2, col=1)
    loss_fig.update_layout(title="Fine-tuning loss", height=600)
    loss_fig
    return


@app.cell(hide_code=True)
def _(get_ft_result):
    _result = get_ft_result()
    if _result is None:
        _numbers_line = ""
    else:
        _history = _result["history"]
        _first, _last = _history[0], _history[-1]
        _numbers_line = (
            f"\n\nThis run: response loss {_first['response_loss']:.3f} -> {_last['response_loss']:.3f}, "
            f"prompt loss {_first['prompt_loss']:.3f} -> {_last['prompt_loss']:.3f}.\n"
        )

    mo.md(
        f"""
        The raw response loss (faint line above) swings by roughly 0.3
        between adjacent steps, ordinary batch-to-batch noise from only
        64 examples at a time. The rolling mean makes the actual trend
        visible through that noise. Scale matters: the entire extra
        gain from running fine-tuning three times longer, 500 steps to
        1,500, was 0.07, smaller than a single step-to-step wiggle.
        Reading anything into one run's raw curve, or into one seed's
        result in a day 3 ablation, means mistaking noise for a
        finding, which is exactly why the ablations run several seeds
        against a noise floor instead of trusting a single number.
        {_numbers_line}
        Prompt loss rising is not a broken run, it is the mask visibly
        doing its job: fine-tuning only ever computes a gradient from
        response tokens, so the model specializes toward the response
        distribution and drifts away from the descriptive prompt style
        it no longer trains on. A rising prompt loss next to a falling
        response loss is stronger evidence the mask works than a flat
        prompt loss would have been, a flat prompt loss would only mean
        "unaffected", not "confirmed excluded".
        """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Chat, checkpoint 4 of 5: after fine-tuning

    Run the fine-tuning above first. Expect a story that follows the
    instruction: your message becomes the `Summary:` field of the same
    prompt format the before/after comparison above uses, and the
    fine-tuned model should write the story itself rather than
    continue the prompt descriptively.
    """)
    return


@app.cell(hide_code=True)
def _(get_ft_result, gpt2_tokenizer):
    def _format_prompt(user_message):
        return f"Summary: {user_message}\nFeatures: Dialogue\nStory:\n"

    _result = get_ft_result()
    if _result is None:
        chat_checkpoint_4 = mo.md("Run the fine-tuning above to chat with the fine-tuned model here.")
    else:
        chat_checkpoint_4 = chat_window(
            _result["model"], gpt2_tokenizer, MAX_LEN,
            prompts=["A brave little fox helps a lost duckling find its way back to the pond."],
            format_prompt=_format_prompt,
        )
    chat_checkpoint_4
    return


if __name__ == "__main__":
    app.run()

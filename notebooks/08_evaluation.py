import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import importlib
    import json
    import math
    import os
    import re
    from pathlib import Path

    import numpy as np
    import plotly.graph_objects as go
    import tiktoken
    import torch
    import torch.nn as nn
    import marimo as mo
    from dotenv import load_dotenv
    from huggingface_hub import hf_hub_download

    # GPTModel and generate are defined in exactly one place, 04_gpt.py.
    _gpt = importlib.import_module("04_gpt")
    GPTModel = _gpt.GPTModel
    generate = _gpt.generate

    # get_batch and DATASET_REPO_ID (the plain TinyStories corpus, not the
    # instruct one) are defined in exactly one place, 01_text_data.py.
    _text_data = importlib.import_module("01_text_data")
    get_batch = _text_data.get_batch
    DATASET_REPO_ID = _text_data.DATASET_REPO_ID

    # evaluate is defined in exactly one place, 05_pretraining.py.
    _pretraining = importlib.import_module("05_pretraining")
    evaluate = _pretraining.evaluate

    load_dotenv()
    hf_token = os.environ["HF_TOKEN"]

    MAX_LEN = 512  # matches CONTEXT_LENGTH everywhere else in the course
    EOT = 50256

    INSTRUCT_DATASET_REPO_ID = "Richard-Sieg-TH-Koln/anlp-tinystories-instruct-gpt2"  # from 98_tokenize_instruct.py

    # The course's three checkpoints, same lineage as 96_build_preference_pairs.py
    # and 07c_dpo.py: pretrained -> SFT -> DPO. "embedded_config" records which of
    # the two checkpoint formats this course has used (see load_checkpoint_model
    # below): the calibration checkpoint embeds its config in checkpoint.pt
    # itself, the newer pushes (97_sft_sanity.py, 07c_dpo.py) save it separately.
    PRETRAINED_CHECKPOINT_REPO_ID = "Richard-Sieg-TH-Koln/anlp-calibration-checkpoint-small"
    SFT_CHECKPOINT_REPO_ID = "Richard-Sieg-TH-Koln/anlp-sft-sanity-checkpoint"
    DPO_CHECKPOINT_REPO_ID = "Richard-Sieg-TH-Koln/anlp-dpo-checkpoint"

    N_TASK_PROMPTS = 100  # held-out instruct prompts for the task metric
    GEN_MAX_NEW_TOKENS = 150
    GEN_BATCH_SIZE = 25


@app.cell(hide_code=True)
def _():
    mo.md("""
    # 08. Evaluating LLMs

    The same three models, the pretrained checkpoint, the SFT'd one,
    the DPO'd one, judged a few different ways. You have talked to all
    three already in the chat interface, the question here is only how
    to measure what you already have an intuition for.

    Perplexity and the required-word inclusion rate both run live, on
    the same three models, and disagree with each other about which
    one is best. That disagreement is the actual content of this
    block. A benchmark gets one cell, discussed rather than run.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## The three models
    """)
    return


@app.function
def load_checkpoint_model(repo_id, embedded_config, device):
    """Load a GPTModel from a Hub checkpoint repo.

    embedded_config: True if config.json is embedded inside checkpoint.pt
    (the calibration checkpoint's format), False if it is a separate
    config.json file next to it (97_sft_sanity.py's and 07c_dpo.py's
    format). Returns (model, config).
    """
    checkpoint_path = hf_hub_download(repo_id=repo_id, filename="checkpoint.pt", repo_type="model", token=hf_token)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if embedded_config:
        cfg = checkpoint["config"]
    else:
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json", repo_type="model", token=hf_token)
        cfg = json.loads(Path(config_path).read_text())

    model = GPTModel(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, cfg


@app.cell
def _():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (device,)


@app.cell
def _(device):
    pretrained_model, pretrained_cfg = load_checkpoint_model(PRETRAINED_CHECKPOINT_REPO_ID, embedded_config=True, device=device)
    sft_model, sft_cfg = load_checkpoint_model(SFT_CHECKPOINT_REPO_ID, embedded_config=False, device=device)
    dpo_model, dpo_cfg = load_checkpoint_model(DPO_CHECKPOINT_REPO_ID, embedded_config=False, device=device)

    for _model in (pretrained_model, sft_model, dpo_model):
        _model.eval()

    mo.md(
        f"""
        loaded all three:

        - pretrained: `{PRETRAINED_CHECKPOINT_REPO_ID}`, {sum(p.numel() for p in pretrained_model.parameters()):,} params
        - SFT: `{SFT_CHECKPOINT_REPO_ID}`, {sum(p.numel() for p in sft_model.parameters()):,} params
        - DPO: `{DPO_CHECKPOINT_REPO_ID}`, {sum(p.numel() for p in dpo_model.parameters()):,} params
        """
    )
    return dpo_model, pretrained_model, sft_model


@app.cell
def _():
    gpt2_tokenizer = tiktoken.get_encoding("gpt2")
    return (gpt2_tokenizer,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 1. Perplexity

    `exp(val_loss)`, the same `evaluate` from `05_pretraining.py`, on
    held-out TinyStories, the plain, unformatted corpus every one of
    these three models was originally pretrained on.
    """)
    return


@app.cell
def _():
    valid_bin_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="valid.bin", repo_type="dataset")
    tinystories_val_data = np.memmap(valid_bin_path, dtype=np.uint16, mode="r")
    mo.md(f"{len(tinystories_val_data):,} held-out TinyStories tokens")
    return (tinystories_val_data,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### Gap 1: perplexity across the three models

    You already have both pieces: `get_batch` samples windows from the
    held-out data, `evaluate` runs a model over a list of `(x, y)`
    batches and returns the mean cross-entropy loss. Perplexity is
    just that loss exponentiated, `exp(loss)`, the number of choices
    the model was effectively as uncertain as, if it had guessed
    uniformly among that many tokens.

    Implement `compute_perplexity`: build `n_batches` batches with
    `get_batch`, evaluate the model on them, exponentiate the result.
    """)
    return


@app.function
def compute_perplexity(model, val_data, device, batch_size, context_length, n_batches=20):
    """Perplexity on held-out TinyStories: exp(mean cross-entropy loss)."""
    # 1. val_batches = [get_batch(val_data, batch_size, context_length, device) for _ in range(n_batches)]
    #      a list of n_batches (x, y) pairs of random windows
    # 2. val_loss = evaluate(model, val_batches)      the mean cross-entropy over those batches
    # 3. return math.exp(val_loss)                    perplexity is the exponential of the loss
    raise NotImplementedError("implement compute_perplexity")


@app.cell
def _(device, pretrained_model, tinystories_val_data):
    _ppl = compute_perplexity(pretrained_model, tinystories_val_data, device, batch_size=8, context_length=64, n_batches=4)
    assert math.isfinite(_ppl) and _ppl > 1, "perplexity must be a finite number greater than 1"
    mo.md(f"compute_perplexity() smoke test on the pretrained model: {_ppl:.1f}")
    return


@app.cell
def _(device, dpo_model, pretrained_model, sft_model, tinystories_val_data):
    pretrained_ppl = compute_perplexity(pretrained_model, tinystories_val_data, device, batch_size=32, context_length=MAX_LEN)
    sft_ppl = compute_perplexity(sft_model, tinystories_val_data, device, batch_size=32, context_length=MAX_LEN)
    dpo_ppl = compute_perplexity(dpo_model, tinystories_val_data, device, batch_size=32, context_length=MAX_LEN)

    mo.md(
        f"""
        | model | perplexity |
        |---|---|
        | pretrained | {pretrained_ppl:.2f} |
        | SFT | {sft_ppl:.2f} |
        | DPO | {dpo_ppl:.2f} |

        The pretrained model should come out on top (lowest perplexity):
        it was trained on exactly this distribution, plain TinyStories,
        and SFT then DPO both moved the model away from it, toward
        following instructions instead of just continuing text. The
        model that follows instructions worst scores best here.
        Perplexity measures fit to a distribution, not usefulness, and
        that gap is the whole point of this block.
        """
    )
    return dpo_ppl, pretrained_ppl, sft_ppl


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Batched generation, and a deliberate bug

    Everything from here on needs generated text from all three models,
    on the order of a hundred prompts each. One prompt at a time, the
    way every notebook so far has done it, is slow enough to hurt in a
    fifty minute block. Batching prompts together is the fix, and
    batching means **left padding**.

    Prompts have different lengths. Stacked into one tensor, the short
    ones need padding, and where that padding goes matters. Right
    padding (every earlier notebook's choice, prompt first then
    padding) puts the padding *after* the real content, so the model's
    next-token prediction for a short prompt sits somewhere in the
    middle of the row, not at the end. `generate`'s `logits[:, -1, :]`
    always reads the *last* position, so for a right-padded short
    prompt it reads a padding position instead, and produces something
    that looks like fluent text with no error and no warning, just the
    wrong answer to the wrong question. Left padding fixes this by
    construction: every row's real content ends at the same, last
    position, so `logits[:, -1, :]` is always the right read, for every
    row, regardless of that row's real prompt length. `attn_mask`,
    unused since `02_attention.py` introduced it, is what stops the
    model from attending to the padding on the left.
    """)
    return


@app.function
def truncate_at_eot(token_ids):
    """Cut a generated token id list at its first EOT, if any (see 96_build_preference_pairs.py)."""
    if EOT in token_ids:
        return token_ids[: token_ids.index(EOT)]
    return token_ids


@app.function
@torch.no_grad()
def generate_batch(model, prompt_id_lists, max_new_tokens, context_length, tokenizer, temperature=0.8, top_k=50):
    """Batched generation, left-padded so every row's real content ends at the same position.

    prompt_id_lists: a list of token-id lists, one per prompt, lengths may differ.
    Returns a list of decoded completion strings (new tokens only, each cut at its own first EOT).
    """
    model.eval()
    device = next(model.parameters()).device
    batch_size = len(prompt_id_lists)
    max_prompt_len = max(len(ids) for ids in prompt_id_lists)

    idx = torch.full((batch_size, max_prompt_len), EOT, dtype=torch.long, device=device)  # (batch_size, max_prompt_len)
    attn_mask = torch.zeros((batch_size, max_prompt_len), dtype=torch.long, device=device)  # (batch_size, max_prompt_len)
    for i, ids in enumerate(prompt_id_lists):
        idx[i, max_prompt_len - len(ids) :] = torch.tensor(ids, device=device)  # right-align: real tokens end at the last column
        attn_mask[i, max_prompt_len - len(ids) :] = 1

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]  # (batch_size, min(num_tokens, context_length))
        attn_cond = attn_mask[:, -context_length:]  # (batch_size, min(num_tokens, context_length))
        logits = model(idx_cond, attn_mask=attn_cond)[:, -1, :]  # (batch_size, vocab_size), always the true last position now
        top_values, _ = torch.topk(logits, top_k)
        logits[logits < top_values[:, [-1]]] = -torch.inf
        probs = torch.softmax(logits / temperature, dim=-1)  # (batch_size, vocab_size)
        next_id = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
        idx = torch.cat([idx, next_id], dim=1)
        attn_mask = torch.cat([attn_mask, torch.ones((batch_size, 1), dtype=torch.long, device=device)], dim=1)

    model.train()
    completions = []
    for i in range(batch_size):
        new_ids = truncate_at_eot(idx[i, max_prompt_len:].tolist())
        completions.append(tokenizer.decode(new_ids))
    return completions


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### The bug, on purpose

    Two prompts, deliberately different lengths, batched with right
    padding this one time, to see the failure directly rather than
    take the explanation above on faith.
    """)
    return


@app.cell
def _(device, gpt2_tokenizer, sft_model):
    _prompts = [
        "Summary: A brave little fox helps a lost duckling find its way back to the pond.\nFeatures: Dialogue\nStory:\n",
        "Summary: A boy loses his toy.\nStory:\n",
    ]
    _ids = [gpt2_tokenizer.encode_ordinary(p) for p in _prompts]
    _max_len = max(len(i) for i in _ids)

    # Right padding, on purpose: real content first, EOT padding after.
    _right_idx = torch.full((2, _max_len), EOT, dtype=torch.long, device=device)
    for _i, _tid in enumerate(_ids):
        _right_idx[_i, : len(_tid)] = torch.tensor(_tid, device=device)

    with torch.no_grad():
        _logits = sft_model(_right_idx)[:, -1, :]  # (2, vocab_size), reads column max_len - 1 for BOTH rows
    _next_token = _logits.argmax(dim=-1)
    _decoded_next = [gpt2_tokenizer.decode([t.item()]) for t in _next_token]

    right_padded_demo_md = mo.md(
        f"""
        Prompt lengths: {[len(i) for i in _ids]}. Row 1 (the shorter
        prompt) ends at position {len(_ids[1])}, but with right padding
        `logits[:, -1, :]` reads position {_max_len - 1} for both rows,
        {_max_len - len(_ids[1])} positions past where row 1's real
        content actually ends.

        Most likely next token read at that wrong position for each row:
        row 0: `{_decoded_next[0]!r}`, row 1: `{_decoded_next[1]!r}`.
        Row 1's "next token" is a prediction for what follows a stretch
        of padding `<|endoftext|>` tokens, not what follows "A boy loses
        his toy.", it is confidently answering a question nobody asked.
        """
    )
    right_padded_demo_md
    return


@app.cell(hide_code=True)
def _(dpo_model, gpt2_tokenizer):
    _prompts = [
        "Summary: A brave little fox helps a lost duckling find its way back to the pond.\nFeatures: Dialogue\nStory:\n",
        "Summary: A boy loses his toy.\nStory:\n",
    ]
    _ids = [gpt2_tokenizer.encode_ordinary(p) for p in _prompts]
    _completions = generate_batch(dpo_model, _ids, max_new_tokens=60, context_length=MAX_LEN, tokenizer=gpt2_tokenizer)

    left_padded_demo_md = mo.md(
        f"""
        Same two prompts, `generate_batch` (left padding, `attn_mask`
        passed through):

        row 0: `{_completions[0][:150]!r}`

        row 1: `{_completions[1][:150]!r}`

        Both read as an actual continuation of their own prompt, not a
        continuation of padding. This is what the rest of the notebook
        uses from here on.
        """
    )
    left_padded_demo_md
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 2. A task metric

    Required-word inclusion rate, the same measure the DPO demo used,
    now on held-out instruct prompts, for all three models.
    """)
    return


@app.function
def parse_required_words(words_field):
    """Turn a raw 'Words: a, b, c' field value into a clean list of words (see 96_build_preference_pairs.py)."""
    return [w.strip() for w in words_field.split(",") if w.strip()]


@app.function
def count_word_matches(text, required_words):
    """How many of required_words appear as whole words in text, case-insensitive (see 96_build_preference_pairs.py)."""
    text_lower = text.lower()
    return sum(1 for word in required_words if re.search(rf"\b{re.escape(word.lower())}\b", text_lower))


@app.cell(hide_code=True)
def _():
    mo.md("""
    Prompts come from the instruct dataset's own valid split
    (`98_tokenize_instruct.py`), already tokenized: the prompt portion
    (`loss_mask == 0`, `attn_mask == 1`) is decoded back to text, which
    still contains its own `Words:` line verbatim, since
    `build_prompt_response` put it there in the first place. Only rows
    that actually have one are usable.
    """)
    return


@app.cell
def _():
    instruct_valid_tokens = np.load(
        hf_hub_download(repo_id=INSTRUCT_DATASET_REPO_ID, filename="valid_tokens.npy", repo_type="dataset")
    )
    instruct_valid_loss_mask = np.load(
        hf_hub_download(repo_id=INSTRUCT_DATASET_REPO_ID, filename="valid_loss_mask.npy", repo_type="dataset")
    )
    instruct_valid_attn_mask = np.load(
        hf_hub_download(repo_id=INSTRUCT_DATASET_REPO_ID, filename="valid_attn_mask.npy", repo_type="dataset")
    )
    mo.md(f"loaded {instruct_valid_tokens.shape[0]:,} tokenized valid examples from `{INSTRUCT_DATASET_REPO_ID}`")
    return instruct_valid_attn_mask, instruct_valid_loss_mask, instruct_valid_tokens


@app.cell
def _(gpt2_tokenizer, instruct_valid_attn_mask, instruct_valid_loss_mask, instruct_valid_tokens):
    def _decode_prompt(row_idx):
        _loss_mask = instruct_valid_loss_mask[row_idx]
        _attn_mask = instruct_valid_attn_mask[row_idx]
        _prompt_end = int((_loss_mask == 0).sum() - (_attn_mask == 0).sum())
        return gpt2_tokenizer.decode(instruct_valid_tokens[row_idx, :_prompt_end].tolist())

    _rng = np.random.default_rng(seed=2)
    _row_order = _rng.permutation(instruct_valid_tokens.shape[0])

    task_prompts = []
    for _row_idx in _row_order:
        _prompt_text = _decode_prompt(int(_row_idx))
        _words_match = re.search(r"Words: (.+)", _prompt_text)
        if _words_match is None:
            continue
        task_prompts.append({"prompt": _prompt_text, "required_words": parse_required_words(_words_match.group(1))})
        if len(task_prompts) >= N_TASK_PROMPTS:
            break

    mo.md(f"{len(task_prompts):,} held-out instruct prompts with a usable `Words:` field")
    return (task_prompts,)


@app.cell
def _(device, dpo_model, gpt2_tokenizer, pretrained_model, sft_model, task_prompts):
    def _generate_all(model):
        completions = []
        for _start in range(0, len(task_prompts), GEN_BATCH_SIZE):
            _batch_prompts = task_prompts[_start : _start + GEN_BATCH_SIZE]
            _prompt_ids = [gpt2_tokenizer.encode_ordinary(p["prompt"]) for p in _batch_prompts]
            completions.extend(
                generate_batch(model, _prompt_ids, GEN_MAX_NEW_TOKENS, MAX_LEN, gpt2_tokenizer)
            )
        return completions

    pretrained_completions = _generate_all(pretrained_model)
    sft_completions = _generate_all(sft_model)
    dpo_completions = _generate_all(dpo_model)

    def _inclusion_rate(completions):
        fractions = [
            count_word_matches(c, p["required_words"]) / len(p["required_words"])
            for c, p in zip(completions, task_prompts)
        ]
        return sum(fractions) / len(fractions)

    pretrained_inclusion = _inclusion_rate(pretrained_completions)
    sft_inclusion = _inclusion_rate(sft_completions)
    dpo_inclusion = _inclusion_rate(dpo_completions)
    return dpo_inclusion, pretrained_inclusion, sft_inclusion


@app.cell(hide_code=True)
def _(
    dpo_inclusion,
    dpo_ppl,
    pretrained_inclusion,
    pretrained_ppl,
    sft_inclusion,
    sft_ppl,
):
    # Report what this run actually shows rather than assert a fixed
    # narrative: pretrained scoring worst on the task metric while
    # scoring best on perplexity is the robust, structural part of this
    # (perplexity directly rewards staying close to the pretraining
    # distribution, the task metric directly rewards moving away from
    # it). Whether DPO specifically beats SFT on the task metric is a
    # smaller effect on top of that, real but not guaranteed at every
    # training length or sample size, so it gets checked, not assumed.
    _perplexity_favors_pretrained = pretrained_ppl < sft_ppl and pretrained_ppl < dpo_ppl
    _task_metric_favors_pretrained_least = pretrained_inclusion < sft_inclusion and pretrained_inclusion < dpo_inclusion
    _dpo_beats_sft_on_task_metric = dpo_inclusion > sft_inclusion

    if _perplexity_favors_pretrained and _task_metric_favors_pretrained_least:
        _headline = (
            "Sit with this before reading on: the ranking flips. Perplexity puts the pretrained model first, "
            "the task metric puts it last, on the same three models. The two numbers measure different things, "
            "fit to a distribution versus success at a task, and neither is \"the\" score."
        )
    else:
        _headline = (
            "The clean textbook flip (pretrained best on perplexity, worst on the task metric) did not fully "
            "show up in these particular numbers. Read the table as it is, not as it was expected to be, that "
            "is the whole point of measuring rather than assuming."
        )

    _dpo_note = (
        "DPO also edges out SFT specifically on the task metric here, the effect the DPO demo's preference "
        "pairs were built to produce."
        if _dpo_beats_sft_on_task_metric
        else "SFT and DPO are close enough on the task metric here that which one is \"ahead\" should not be "
        "read as a strong claim, particularly on a small or lightly-trained run, see the noise-floor point "
        "block 07 already made."
    )

    mo.md(
        f"""
        | model | perplexity | required-word inclusion rate |
        |---|---|---|
        | pretrained | {pretrained_ppl:.2f} | {pretrained_inclusion:.1%} |
        | SFT | {sft_ppl:.2f} | {sft_inclusion:.1%} |
        | DPO | {dpo_ppl:.2f} | {dpo_inclusion:.1%} |

        {_headline}

        {_dpo_note}
        """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 3. A benchmark, discussed not run

    HellaSwag, ARC, the standard leaderboard suites: multiple-choice
    tasks designed for models with billions of parameters and broad
    world knowledge. A 19 to 30 million parameter model that has only
    ever seen TinyStories scores at chance on these, four options, 25%,
    regardless of which of the three checkpoints you pick. Running it
    would burn slides on a number that carries no information.

    The number still teaches something: benchmarks need scale to mean
    anything, which is why nobody benchmarks a model this size on
    HellaSwag for real. And at the scale where benchmarks do mean
    something, two more problems show up that do not exist at this
    size: **contamination** (the benchmark's own questions leaking
    into a model's enormous pretraining corpus, so it is recognising
    answers, not reasoning), and the general unreliability of
    leaderboard numbers once labs can choose which checkpoint, which
    prompt template and which run to report. A benchmark score is a
    measurement with its own error bars and its own failure modes,
    same as the required-word counter above.
    """)
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import copy
    import importlib
    import json
    import math
    import os
    import re
    from pathlib import Path

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import tiktoken
    import torch
    import torch.nn.functional as F
    import marimo as mo
    from dotenv import load_dotenv
    from huggingface_hub import HfApi, hf_hub_download

    # GPTModel and generate are defined in exactly one place, 04_gpt.py
    # (re-exposed by 07_sft.py, same directory, plain sibling import).
    _sft = importlib.import_module("07_sft_solution")
    GPTModel = _sft.GPTModel
    generate = _sft.generate
    chat_window = _sft.chat_window

    SOURCE_REPO_ID = "roneneldan/TinyStoriesInstruct"
    DELIMITER = "<|endoftext|>"
    EOT = 50256  # GPT-2 <|endoftext|>

    # The raw file labels each field of an example, in no fixed order. The
    # prompt always lists the fields in PROMPT_FIELD_ORDER, then "Story:".
    KNOWN_LABELS = ("Random sentence", "Words", "Features", "Summary", "Story")
    PROMPT_FIELD_ORDER = ("Summary", "Features", "Words", "Random sentence")

    # The preference pairs were built from this many prompts, sampled from the
    # valid split with seed 0. The evaluation below reproduces that sample to
    # stay clear of exactly those prompts.
    N_BUILD_PROMPTS = 3000

    load_dotenv()
    hf_token = os.environ["HF_TOKEN"]

    MAX_LEN = 512  # matches CONTEXT_LENGTH everywhere else in the course
    SFT_CHECKPOINT_REPO_ID = "Richard-Sieg-TH-Koln/anlp-sft-sanity-checkpoint"
    DPO_DATASET_REPO_ID = "Richard-Sieg-TH-Koln/anlp-tinystories-dpo-gpt2"  # preference pairs, built beforehand from the SFT checkpoint
    DPO_CHECKPOINT_REPO_ID = "Richard-Sieg-TH-Koln/anlp-dpo-checkpoint"  # this notebook's own push target, read back by the evaluation block (solutions/08_evaluation_solution.py)

    DPO_BETA = 0.1
    DPO_MAX_LR = 5e-6  # far below SFT's 1e-4: DPO nudges preferences, it does not relearn the task
    DPO_NUM_STEPS = 300
    DPO_BATCH_SIZE = 32  # smaller than SFT's 64: every DPO step is 4 forward passes (chosen/rejected x policy/reference), not 1

    N_EVAL_PROMPTS = 200  # held-out prompts for the before/after inclusion-rate comparison at the end
    GEN_MAX_NEW_TOKENS = 200
    GEN_BATCH_SIZE = 25  # prompts generated together in the evaluation, see generate_batch


@app.cell(hide_code=True)
def _():
    mo.md("""
    # 07c. Direct Preference Optimization

    A demo, not a lab: read top to bottom, or watch it run top to
    bottom. Reruns the same idea as `07_sft.py` and `07b_lora.py`, one
    more stage on top of a model this course already built by hand.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## The three stages, recapped

    Pretraining (`05_pretraining.py`) taught next-token prediction on
    raw stories: a model that continues text, nothing more. Supervised
    fine-tuning (`07_sft.py`) taught it to treat a prompt as an
    instruction rather than more text to continue, the same loss,
    masked to only the response. Preference tuning is the third stage:
    given two responses to the same prompt, learn to prefer the better
    one. You have now built the first two of these yourselves. This
    notebook is the third.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Reward models and PPO

    The original recipe for that third stage, still what "RLHF" means
    by default, needs two more models on top of the one being tuned.

    A **reward model** is an LLM with its language-modelling head
    (the linear layer producing next-token logits) replaced by a
    single output: one scalar score for how good a whole response is.
    It is trained on preference pairs directly, chosen should score
    higher than rejected, that is the entire signal.

    **PPO** (Proximal Policy Optimization) then treats generation as
    reinforcement learning: sample a response from the policy, score
    it with the reward model, and push the policy to raise the
    probability of high-scoring responses, with a penalty that keeps
    the policy from drifting too far from where it started (a KL term
    against a frozen reference), so it cannot simply learn to fool the
    reward model with degenerate text.

    That is three models in memory at once (policy, reward model,
    reference), a sampling loop that has to actually generate text
    every step, and a reinforcement learning algorithm on top, all to
    optimize something that, underneath, is still just a preference
    between two known texts.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What DPO changes

    DPO's observation: for this exact setup, the reward model and the
    sampling loop are not necessary. A preference between two *already
    known* responses can be turned directly into a loss on the policy,
    with a frozen copy of the starting model standing in for both the
    reward model and the KL penalty at once.

    $$
    \mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \left[
    \left(\log \pi_\theta(y_w \mid x) - \log \pi_{\text{ref}}(y_w \mid x)\right)
    -
    \left(\log \pi_\theta(y_l \mid x) - \log \pi_{\text{ref}}(y_l \mid x)\right)
    \right]\right)
    $$

    $y_w$ is chosen, $y_l$ is rejected, $\pi_\theta$ is the model being
    trained (the policy), $\pi_{\text{ref}}$ is a frozen copy of it
    from before this stage started. Each log-ratio
    ($\log \pi_\theta - \log \pi_{\text{ref}}$) is how much more (or
    less) likely the policy now finds a response than the reference
    did, an implicit reward, no separate reward model computes it. The
    difference of the two log-ratios is how much more that reward
    favors chosen over rejected. $\beta$ controls how hard to push,
    `logsigmoid` turns "favors chosen" into a loss to minimize, the
    same shape a classification loss would have.

    In code, next.
    """)
    return


@app.function
def sequence_logprob(model, x, y, loss_mask, attn_mask):
    """Sum of log p(y_t | x_<=t) over response positions only (loss_mask==1)."""
    logits = model(x, attn_mask=attn_mask)  # (b, seq_len, vocab_size)
    log_probs = F.log_softmax(logits, dim=-1)  # (b, seq_len, vocab_size)
    token_log_probs = log_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)  # (b, seq_len)
    return (token_log_probs * loss_mask).sum(dim=-1)  # (b,)


@app.function
def dpo_loss(
    policy_model,
    reference_model,
    chosen_x, chosen_y, chosen_loss_mask, chosen_attn_mask,
    rejected_x, rejected_y, rejected_loss_mask, rejected_attn_mask,
    beta=DPO_BETA,
):
    """DPO loss, plus the pieces the plots below need: per-example chosen/rejected
    log-probs under the policy, and the implicit reward margin."""
    policy_chosen_logp = sequence_logprob(policy_model, chosen_x, chosen_y, chosen_loss_mask, chosen_attn_mask)  # (b,)
    policy_rejected_logp = sequence_logprob(policy_model, rejected_x, rejected_y, rejected_loss_mask, rejected_attn_mask)  # (b,)

    with torch.no_grad():
        ref_chosen_logp = sequence_logprob(reference_model, chosen_x, chosen_y, chosen_loss_mask, chosen_attn_mask)  # (b,)
        ref_rejected_logp = sequence_logprob(reference_model, rejected_x, rejected_y, rejected_loss_mask, rejected_attn_mask)  # (b,)

    policy_logratio = policy_chosen_logp - policy_rejected_logp  # (b,), how much the policy now prefers chosen
    reference_logratio = ref_chosen_logp - ref_rejected_logp  # (b,), how much the reference preferred chosen
    reward_margin = beta * (policy_logratio - reference_logratio)  # (b,), implicit reward, chosen minus rejected

    loss = -F.logsigmoid(reward_margin).mean()
    return loss, policy_chosen_logp.detach(), policy_rejected_logp.detach(), reward_margin.detach()


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## The reference model

    A frozen, deep-copied snapshot of the SFT'd model, in eval mode
    (dropout off, stable log-probs), gradients off (never optimized,
    never should be). A reference model that drifts during training is
    the silent failure here: the loss above would keep computing
    without error, it would just be measuring against a moving target
    instead of a fixed one. Checked below, not assumed.
    """)
    return


@app.cell
def _():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (device,)


@app.cell
def _(device):
    sft_checkpoint_path = hf_hub_download(repo_id=SFT_CHECKPOINT_REPO_ID, filename="checkpoint.pt", repo_type="model")
    sft_config_path = hf_hub_download(repo_id=SFT_CHECKPOINT_REPO_ID, filename="config.json", repo_type="model")
    sft_checkpoint = torch.load(sft_checkpoint_path, map_location="cpu", weights_only=True)
    sft_cfg = json.loads(Path(sft_config_path).read_text())  # the SFT checkpoint repo keeps config.json next to checkpoint.pt, not embedded in it

    policy_model = GPTModel(sft_cfg).to(device)
    policy_model.load_state_dict(sft_checkpoint["model_state_dict"])

    reference_model = copy.deepcopy(policy_model)
    reference_model.eval()
    for _param in reference_model.parameters():
        _param.requires_grad = False

    _num_params = sum(p.numel() for p in policy_model.parameters())
    mo.md(f"loaded {_num_params:,} parameters from `{SFT_CHECKPOINT_REPO_ID}` into both the policy and the reference, config: {sft_cfg}")
    return policy_model, reference_model


@app.cell
def _(reference_model):
    reference_snapshot = torch.cat([p.detach().flatten() for p in reference_model.parameters()]).clone()
    return (reference_snapshot,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## The data

    Preference pairs built beforehand from the SFT'd model: two
    completions per prompt, sampled at temperature 1.0 so they can
    differ, and the one that contains more of the prompt's required
    words is chosen, the other rejected (ties are dropped). Tokenized
    to width 512 with the same padding and mask scheme as the SFT
    dataset.
    """)
    return


@app.cell
def _():
    def _load(name):
        return np.load(hf_hub_download(repo_id=DPO_DATASET_REPO_ID, filename=f"{name}.npy", repo_type="dataset", token=hf_token))

    dpo_chosen_tokens = _load("chosen_tokens")
    dpo_chosen_loss_mask = _load("chosen_loss_mask")
    dpo_chosen_attn_mask = _load("chosen_attn_mask")
    dpo_rejected_tokens = _load("rejected_tokens")
    dpo_rejected_loss_mask = _load("rejected_loss_mask")
    dpo_rejected_attn_mask = _load("rejected_attn_mask")

    mo.md(f"loaded {dpo_chosen_tokens.shape[0]:,} preference pairs from `{DPO_DATASET_REPO_ID}`")
    return (
        dpo_chosen_attn_mask,
        dpo_chosen_loss_mask,
        dpo_chosen_tokens,
        dpo_rejected_attn_mask,
        dpo_rejected_loss_mask,
        dpo_rejected_tokens,
    )


@app.function
def get_dpo_batch(
    chosen_tokens, chosen_loss_mask, chosen_attn_mask,
    rejected_tokens, rejected_loss_mask, rejected_attn_mask,
    batch_size, device,
):
    """Paired chosen/rejected rows for the same prompts, shifted by one token like get_instruct_batch."""
    n_examples = chosen_tokens.shape[0]
    row_indices = np.random.randint(0, n_examples, size=batch_size)  # (batch_size,), same rows for chosen and rejected

    def _shift(tokens, loss_mask, attn_mask):
        x = torch.from_numpy(tokens[row_indices, :-1].astype(np.int64)).to(device)  # (batch_size, MAX_LEN - 1)
        y = torch.from_numpy(tokens[row_indices, 1:].astype(np.int64)).to(device)  # (batch_size, MAX_LEN - 1)
        y_loss_mask = torch.from_numpy(loss_mask[row_indices, 1:].astype(np.int64)).to(device)  # (batch_size, MAX_LEN - 1)
        x_attn_mask = torch.from_numpy(attn_mask[row_indices, :-1].astype(np.int64)).to(device)  # (batch_size, MAX_LEN - 1)
        return x, y, y_loss_mask, x_attn_mask

    chosen_x, chosen_y, chosen_ymask, chosen_amask = _shift(chosen_tokens, chosen_loss_mask, chosen_attn_mask)
    rejected_x, rejected_y, rejected_ymask, rejected_amask = _shift(rejected_tokens, rejected_loss_mask, rejected_attn_mask)
    return chosen_x, chosen_y, chosen_ymask, chosen_amask, rejected_x, rejected_y, rejected_ymask, rejected_amask


@app.cell(hide_code=True)
def _():
    mo.md(f"""
    ## Run it

    {DPO_NUM_STEPS} steps, lr {DPO_MAX_LR} (two orders of magnitude
    below SFT's own learning rate: this stage nudges preferences, a
    higher rate would risk unlearning what SFT already taught), beta
    {DPO_BETA}, batch {DPO_BATCH_SIZE}.

    The policy runs in eval mode here, same as the reference: dropout
    off for the whole run, not just at step 0. Two reasons. It makes
    every plot below less noisy. And it makes step 0 checkable: policy
    and reference start as the same weights, and with dropout off that
    makes them the same *function*, not just the same weights, so
    every margin at step 0 is exactly zero and the loss is exactly
    `log(2)` &asymp; 0.693, the same kind of sanity check as
    pretraining starting at `ln(50257)` &asymp; 10.8. Checked below,
    not just claimed.

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
    dpo_run_button = mo.ui.run_button(label="Run DPO")
    dpo_run_button
    return (dpo_run_button,)


@app.cell
def _():
    get_dpo_result, set_dpo_result = mo.state(None)
    return get_dpo_result, set_dpo_result


@app.cell
def _(
    device,
    dpo_chosen_attn_mask,
    dpo_chosen_loss_mask,
    dpo_chosen_tokens,
    dpo_rejected_attn_mask,
    dpo_rejected_loss_mask,
    dpo_rejected_tokens,
    dpo_run_button,
    is_script_mode,
    policy_model,
    reference_model,
    set_dpo_result,
):
    if is_script_mode or dpo_run_button.value:
        torch.manual_seed(42)
        # eval mode, not train mode, for the policy too: the only
        # train/eval-sensitive layer in this architecture is dropout,
        # so this simply turns dropout off for the whole run (nothing
        # here stops gradients from flowing or the optimizer from
        # updating weights, eval/train only changes forward-pass
        # behaviour). TRL does the same by default. With dropout off,
        # policy and reference are the same function, not just the
        # same weights, at step 0, see the sanity check below.
        policy_model.eval()
        _optimizer = torch.optim.AdamW(policy_model.parameters(), lr=DPO_MAX_LR, weight_decay=0.1, betas=(0.9, 0.95))

        _history = []
        for _step in range(DPO_NUM_STEPS):
            _chosen_x, _chosen_y, _chosen_ymask, _chosen_amask, _rejected_x, _rejected_y, _rejected_ymask, _rejected_amask = get_dpo_batch(
                dpo_chosen_tokens, dpo_chosen_loss_mask, dpo_chosen_attn_mask,
                dpo_rejected_tokens, dpo_rejected_loss_mask, dpo_rejected_attn_mask,
                DPO_BATCH_SIZE, device,
            )

            _loss, _chosen_logp, _rejected_logp, _reward_margin = dpo_loss(
                policy_model, reference_model,
                _chosen_x, _chosen_y, _chosen_ymask, _chosen_amask,
                _rejected_x, _rejected_y, _rejected_ymask, _rejected_amask,
                beta=DPO_BETA,
            )

            _optimizer.zero_grad(set_to_none=True)
            _loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            _optimizer.step()

            _history.append(
                {
                    "step": _step,
                    "loss": _loss.item(),
                    "chosen_logp": _chosen_logp.mean().item(),
                    "rejected_logp": _rejected_logp.mean().item(),
                    "reward_margin": _reward_margin.mean().item(),
                }
            )

        set_dpo_result({"history": _history})
    return


@app.cell(hide_code=True)
def _(get_dpo_result):
    _result = get_dpo_result()
    if _result is None:
        ln2_md = mo.md("Run DPO above to check the step-0 sanity value.")
    else:
        _step0_loss = _result["history"][0]["loss"]
        assert abs(_step0_loss - math.log(2)) < 1e-3, (
            f"step 0 loss should be exactly ln(2) ~= {math.log(2):.6f} (policy and reference start as the same "
            f"weights and, with dropout off, the same function), got {_step0_loss:.6f}: dropout may still be "
            "active somewhere"
        )
        ln2_md = mo.md(
            f"Step 0 loss: {_step0_loss:.6f}, against `log(2)` = {math.log(2):.6f}. Policy and reference start "
            "identical and, with dropout off, compute identically: every margin at step 0 is exactly zero, and "
            "the loss must be exactly `log(2)`, confirmed."
        )
    ln2_md
    return


@app.cell(hide_code=True)
def _(get_dpo_result, reference_model, reference_snapshot):
    _result = get_dpo_result()
    if _result is None:
        drift_md = mo.md("Run DPO above to check the reference model for drift.")
    else:
        _current = torch.cat([p.detach().flatten() for p in reference_model.parameters()])
        assert torch.equal(reference_snapshot, _current), "reference model drifted during training, this is the silent failure DPO depends on avoiding"
        drift_md = mo.md("Reference model parameters after training: identical to before training, confirmed. It never saw an optimizer step.")
    drift_md
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Three plots

    DPO loss, then chosen and rejected log-probs separately (not just
    their difference), then the implicit reward margin.

    The middle panel is the one that contradicts the intuition every
    student walks in with. DPO does not make the chosen response more
    likely, in any absolute sense. It makes chosen relatively more
    likely than rejected, measured against the reference, and nothing
    in the loss stops both from falling together while that ratio
    still moves the right way: the model becomes less confident about
    everything, just less so about the responses it is supposed to
    prefer. Same lesson as the toast story: the model is not learning
    to write a better response, it is learning to shift a ratio. Shown
    below if it happens on this run, not smoothed over or assumed.

    Plotted relative to each series' own step-0 value, both lines
    starting at zero, rather than as raw log-probs: chosen and
    rejected sit only a few nats apart on an absolute scale of several
    hundred, so on a shared axis the two raw traces overlap so tightly
    the drift this panel exists to show would otherwise be invisible.
    """)
    return


@app.cell
def _(get_dpo_result):
    _result = get_dpo_result()
    _history = _result["history"] if _result is not None else []
    _steps = [row["step"] for row in _history]

    _chosen_logp = [row["chosen_logp"] for row in _history]
    _rejected_logp = [row["rejected_logp"] for row in _history]
    # Relative to each series' own step-0 value, not raw: chosen and
    # rejected differ by only a few nats on a scale of several hundred,
    # so the raw traces overlap too tightly to show the drift this
    # panel exists to show (see the markdown above).
    _chosen_logp_delta = [v - _chosen_logp[0] for v in _chosen_logp] if _chosen_logp else []
    _rejected_logp_delta = [v - _rejected_logp[0] for v in _rejected_logp] if _rejected_logp else []

    dpo_fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("DPO loss", "log-prob change since step 0, chosen vs rejected", "implicit reward margin"),
    )
    dpo_fig.add_trace(go.Scatter(x=_steps, y=[row["loss"] for row in _history], name="loss", mode="lines"), row=1, col=1)
    dpo_fig.add_trace(
        go.Scatter(x=_steps, y=_chosen_logp_delta, name="chosen log-prob (delta)", mode="lines"),
        row=2, col=1,
    )
    dpo_fig.add_trace(
        go.Scatter(x=_steps, y=_rejected_logp_delta, name="rejected log-prob (delta)", mode="lines"),
        row=2, col=1,
    )
    dpo_fig.add_trace(
        go.Scatter(x=_steps, y=[row["reward_margin"] for row in _history], name="reward margin", mode="lines"),
        row=3, col=1,
    )
    dpo_fig.update_yaxes(title_text="loss", row=1, col=1)
    dpo_fig.update_yaxes(title_text="log-prob change from step 0", row=2, col=1)
    dpo_fig.update_yaxes(title_text="margin", row=3, col=1)
    dpo_fig.update_xaxes(title_text="step", row=3, col=1)
    dpo_fig.update_layout(title="DPO training", height=800)
    dpo_fig
    return


@app.cell(hide_code=True)
def _(get_dpo_result):
    _result = get_dpo_result()
    if _result is None:
        margin_md = mo.md("Run DPO above to see whether this happened on this run.")
    else:
        _history = _result["history"]
        _first, _last = _history[0], _history[-1]
        _both_fell = _last["chosen_logp"] < _first["chosen_logp"] and _last["rejected_logp"] < _first["rejected_logp"]
        _margin_grew = _last["reward_margin"] > _first["reward_margin"]

        if _both_fell and _margin_grew:
            _note = (
                f"On this run: both log-probs fell (chosen {_first['chosen_logp']:.1f} -> {_last['chosen_logp']:.1f}, "
                f"rejected {_first['rejected_logp']:.1f} -> {_last['rejected_logp']:.1f}) while the reward margin "
                f"still grew ({_first['reward_margin']:.3f} -> {_last['reward_margin']:.3f}). Exactly the pattern "
                "described above: DPO did not make the chosen response more likely, it made it relatively more "
                "likely than rejected. The model got less confident about everything and learned the preference "
                "anyway, same lesson as the toast story: this is shifting a ratio, not learning to write better."
            )
        else:
            _note = (
                f"On this run: chosen log-prob went {_first['chosen_logp']:.1f} -> {_last['chosen_logp']:.1f}, "
                f"rejected {_first['rejected_logp']:.1f} -> {_last['rejected_logp']:.1f}, reward margin "
                f"{_first['reward_margin']:.3f} -> {_last['reward_margin']:.3f}. The both-fall-together pattern "
                "described above did not show up on this particular run, that is fine too, it is common, not guaranteed."
            )
        margin_md = mo.md(_note)
    margin_md
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## The result

    Required-word inclusion rate, before (reference) and after
    (policy), on held-out prompts the DPO pairs were never built from.
    Then one prompt, generated by both, side by side. This is the
    actual test, the plots above show whether training behaved, not
    whether it worked.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### Helpers

    The raw TinyStoriesInstruct file labels each field of an example
    (`Summary:`, `Features:`, `Words:`, `Random sentence:`, `Story:`),
    in no fixed order. These helpers split an example by label, build
    the prompt the model was fine-tuned on, and score a completion by
    how many of the required words it contains.

    The evaluation generates 200 completions per model, 200 tokens
    each, with no KV cache: one prompt at a time that is tens of
    thousands of tiny forward passes, and a small model at batch size 1
    leaves the GPU mostly idle. So `generate_batch` generates many
    prompts at once. Prompts have different lengths, so the short ones
    are padded on the **left**: every row's real content then ends at
    the last position, which is the one `generate` reads the next
    token from, and `attn_mask` keeps the model from attending to the
    padding. Padding on the right would make the model continue the
    padding instead of the prompt.
    """)
    return


@app.function
def parse_example(raw_text):
    """Split one raw TinyStoriesInstruct example into its labeled fields."""
    fields = {}
    current_label = None
    current_lines = []
    for line in raw_text.strip("\n").split("\n"):
        matched_label = None
        for label in KNOWN_LABELS:
            prefix = label + ":"
            if line.startswith(prefix):
                matched_label = label
                first_line_content = line[len(prefix) :].strip()
                break
        if matched_label is not None:
            if current_label is not None:
                fields[current_label] = "\n".join(current_lines).strip()
            current_label = matched_label
            current_lines = [first_line_content] if first_line_content else []
        elif current_label is not None:
            current_lines.append(line)
    if current_label is not None:
        fields[current_label] = "\n".join(current_lines).strip()
    return fields


@app.function
def build_prompt_response(fields):
    """Turn parsed fields into a (prompt, response) pair. Story is always the response."""
    prompt_lines = [f"{label}: {fields[label]}" for label in PROMPT_FIELD_ORDER if label in fields]
    prompt_lines.append("Story:")
    prompt = "\n".join(prompt_lines) + "\n"
    response = fields["Story"].strip()
    return prompt, response


@app.function
def parse_required_words(words_field):
    """Turn a raw 'Words: a, b, c' field value into a clean list of words."""
    return [w.strip() for w in words_field.split(",") if w.strip()]


@app.function
def count_word_matches(text, required_words):
    """How many of required_words appear as whole words in text, case-insensitive."""
    text_lower = text.lower()
    return sum(1 for word in required_words if re.search(rf"\b{re.escape(word.lower())}\b", text_lower))


@app.function
def truncate_at_eot(token_ids):
    """Cut a generated token id list at its first EOT, if any.

    generate() has no early stopping, it always produces exactly
    max_new_tokens more tokens, running past the point where the SFT'd
    model would have stopped on its own. Counting required words in
    text the model would not have written itself would inflate the score.
    """
    if EOT in token_ids:
        return token_ids[: token_ids.index(EOT)]
    return token_ids


@app.function
@torch.no_grad()
def generate_batch(model, prompt_id_lists, max_new_tokens, context_length, temperature=0.8, top_k=50):
    """Generate a continuation for every prompt at once, left-padded.

    prompt_id_lists: one token-id list per prompt, lengths may differ.
    Returns one list of new token ids per prompt, each cut at its first EOT.
    Samples like generate(): temperature 0.8 and top-k 50 by default.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    batch_size = len(prompt_id_lists)
    max_prompt_len = max(len(ids) for ids in prompt_id_lists)

    idx = torch.full((batch_size, max_prompt_len), EOT, dtype=torch.long, device=device)  # (b, max_prompt_len)
    attn_mask = torch.zeros((batch_size, max_prompt_len), dtype=torch.long, device=device)  # (b, max_prompt_len)
    for row, ids in enumerate(prompt_id_lists):
        idx[row, max_prompt_len - len(ids) :] = torch.tensor(ids, device=device)  # real tokens end at the last column
        attn_mask[row, max_prompt_len - len(ids) :] = 1  # padding on the left stays masked out

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]  # (b, min(num_tokens, context_length))
        attn_cond = attn_mask[:, -context_length:]  # (b, min(num_tokens, context_length))
        logits = model(idx_cond, attn_mask=attn_cond)[:, -1, :]  # (b, vocab_size), the last position is real for every row
        top_values, _ = torch.topk(logits, top_k)
        logits[logits < top_values[:, [-1]]] = -torch.inf  # keep only the top-k logits
        probs = torch.softmax(logits / temperature, dim=-1)  # (b, vocab_size)
        next_id = torch.multinomial(probs, num_samples=1)  # (b, 1)
        idx = torch.cat([idx, next_id], dim=1)
        attn_mask = torch.cat([attn_mask, torch.ones((batch_size, 1), dtype=torch.long, device=device)], dim=1)

    model.train(was_training)
    return [truncate_at_eot(idx[row, max_prompt_len:].tolist()) for row in range(batch_size)]


@app.cell
def _():
    valid_txt_path = hf_hub_download(repo_id=SOURCE_REPO_ID, filename="TinyStories-Instruct-valid.txt", repo_type="dataset")
    valid_raw_text = open(valid_txt_path, encoding="utf-8").read()
    _raw_examples = [e for e in valid_raw_text.split(DELIMITER) if e.strip()]
    _parsed = [parse_example(e) for e in _raw_examples]
    usable_examples = [f for f in _parsed if f.get("Story") and f.get("Words")]

    # Reproduce the sample the preference pairs were built from (same
    # seed, same N) to know exactly which examples they came from, then
    # evaluate on the ones left over instead, so this is a genuine
    # held-out check, not a readout on the model's own training data.
    _rng = np.random.default_rng(seed=0)
    _build_idx = set(_rng.choice(len(usable_examples), size=min(N_BUILD_PROMPTS, len(usable_examples)), replace=False).tolist())
    _remaining = [i for i in range(len(usable_examples)) if i not in _build_idx]

    _eval_rng = np.random.default_rng(seed=1)
    _eval_n = min(N_EVAL_PROMPTS, len(_remaining))
    _eval_idx = _eval_rng.choice(_remaining, size=_eval_n, replace=False)
    eval_fields = [usable_examples[i] for i in _eval_idx]

    mo.md(f"{len(eval_fields):,} held-out prompts for evaluation, disjoint from the {len(_build_idx):,} used to build the DPO pairs")
    return (eval_fields,)


@app.cell
def _():
    gpt2_tokenizer = tiktoken.get_encoding("gpt2")
    return (gpt2_tokenizer,)


@app.function
def inclusion_rate(model, fields_list, tokenizer):
    """Mean fraction of required words present across one generated completion per prompt.

    Prompts are generated in batches of GEN_BATCH_SIZE, shortest first, so that each batch
    needs as little padding as possible.
    """
    prompt_ids = [tokenizer.encode_ordinary(build_prompt_response(fields)[0]) for fields in fields_list]
    order = sorted(range(len(prompt_ids)), key=lambda i: len(prompt_ids[i]))  # indices, shortest prompt first

    completions = [""] * len(prompt_ids)
    for start in range(0, len(order), GEN_BATCH_SIZE):
        batch_indices = order[start : start + GEN_BATCH_SIZE]
        new_id_lists = generate_batch(model, [prompt_ids[i] for i in batch_indices], GEN_MAX_NEW_TOKENS, MAX_LEN)
        for i, new_ids in zip(batch_indices, new_id_lists):
            completions[i] = tokenizer.decode(new_ids)

    fractions = []
    for fields, completion in zip(fields_list, completions):
        required_words = parse_required_words(fields["Words"])
        fractions.append(count_word_matches(completion, required_words) / len(required_words))
    return sum(fractions) / len(fractions) if fractions else 0.0


@app.cell(hide_code=True)
def _(eval_fields, get_dpo_result, gpt2_tokenizer, policy_model, reference_model):
    # policy_model is trained in place above, not stored in
    # get_dpo_result: reference_model is a permanently separate,
    # frozen object (the "before"), so reading policy_model directly
    # here, gated on get_dpo_result having a value, is safe the same
    # way the drift-check cell above is, no risk of reading it before
    # training actually finished.
    _result = get_dpo_result()
    if _result is None:
        inclusion_md = mo.md("Run DPO above first, this compares the reference (before) against the trained policy (after).")
    else:
        before_rate = inclusion_rate(reference_model, eval_fields, gpt2_tokenizer)
        after_rate = inclusion_rate(policy_model, eval_fields, gpt2_tokenizer)
        _delta = after_rate - before_rate

        if abs(_delta) < 0.01:
            _verdict = "The rate did not move. That is a real result, not a failed run, worth saying plainly rather than reading noise as a finding."
        elif _delta > 0:
            _verdict = f"The rate moved up by {_delta:.1%}, in the direction DPO was supposed to push it."
        else:
            _verdict = f"The rate moved down by {abs(_delta):.1%}, the wrong direction, worth investigating rather than hiding."

        inclusion_md = mo.md(
            f"""
            Required-word inclusion rate on {len(eval_fields)} held-out prompts:

            before (reference, SFT'd model): {before_rate:.1%}

            after (policy, after DPO): {after_rate:.1%}

            {_verdict}
            """
        )
    inclusion_md
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## One prompt, before and after

    Same held-out prompt, generated by the reference and by the
    trained policy, side by side.
    """)
    return


@app.cell
def _(eval_fields):
    demo_prompt, _ = build_prompt_response(eval_fields[0])
    demo_required_words = parse_required_words(eval_fields[0]["Words"])
    return demo_prompt, demo_required_words


@app.cell(hide_code=True)
def _(demo_prompt, demo_required_words, device, get_dpo_result, gpt2_tokenizer, policy_model, reference_model):
    _result = get_dpo_result()
    if _result is None:
        side_by_side_md = mo.md("Run DPO above first.")
    else:
        _prompt_ids = torch.tensor([gpt2_tokenizer.encode_ordinary(demo_prompt)], device=device)

        _before_ids = truncate_at_eot(
            generate(reference_model, _prompt_ids, GEN_MAX_NEW_TOKENS, MAX_LEN)[0, _prompt_ids.shape[1] :].tolist()
        )
        _after_ids = truncate_at_eot(
            generate(policy_model, _prompt_ids, GEN_MAX_NEW_TOKENS, MAX_LEN)[0, _prompt_ids.shape[1] :].tolist()
        )
        _before_text = gpt2_tokenizer.decode(_before_ids)
        _after_text = gpt2_tokenizer.decode(_after_ids)
        _before_count = count_word_matches(_before_text, demo_required_words)
        _after_count = count_word_matches(_after_text, demo_required_words)

        side_by_side_md = mo.md(
            f"""
            PROMPT (required words: {", ".join(demo_required_words)}):
            ```
            {demo_prompt}
            ```

            BEFORE (reference, {_before_count}/{len(demo_required_words)} words):
            ```
            {_before_text}
            ```

            AFTER (policy, {_after_count}/{len(demo_required_words)} words):
            ```
            {_after_text}
            ```
            """
        )
    side_by_side_md
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Chat, checkpoint 5 of 5: after preference tuning

    Run DPO above first. Compare with checkpoint 4 in `07_sft.py`: both
    follow the instruction, DPO additionally nudges which of two valid
    continuations it prefers, the same preference signal the plots
    above show moving.
    """)
    return


@app.cell(hide_code=True)
def _(get_dpo_result, gpt2_tokenizer, policy_model, sft_cfg):
    def _format_prompt(user_message):
        return f"Summary: {user_message}\nFeatures: Dialogue\nStory:\n"

    _result = get_dpo_result()
    if _result is None:
        chat_checkpoint_5 = mo.md("Run DPO above to chat with the preference-tuned model here.")
    else:
        chat_checkpoint_5 = chat_window(
            policy_model, gpt2_tokenizer, sft_cfg["context_length"],
            prompts=["A brave little fox helps a lost duckling find its way back to the pond."],
            format_prompt=_format_prompt,
        )
    chat_checkpoint_5
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Push to the Hub

    Config and loss history go up next to the checkpoint: the
    evaluation block (`solutions/08_evaluation_solution.py`) reads this
    back as the third of the course's three checkpoints (pretrained,
    SFT, DPO), and a checkpoint without the config it was trained with
    is not reproducible.
    """)
    return


@app.cell
def _(get_dpo_result, policy_model, sft_cfg):
    _result = get_dpo_result()
    dpo_hub_api = None
    if _result is not None:
        dpo_checkpoint_path = Path("dpo_checkpoint.pt")
        torch.save({"model_state_dict": policy_model.state_dict()}, dpo_checkpoint_path)

        dpo_config_path = Path("dpo_config.json")
        dpo_config_path.write_text(json.dumps(sft_cfg, indent=2))

        dpo_history_path = Path("dpo_history.json")
        dpo_history_path.write_text(json.dumps(_result["history"], indent=2))

        dpo_hub_api = HfApi(token=hf_token)
        dpo_hub_api.create_repo(DPO_CHECKPOINT_REPO_ID, repo_type="model", private=False, exist_ok=True)
        for _local_path, _repo_path in [
            (dpo_checkpoint_path, "checkpoint.pt"),
            (dpo_config_path, "config.json"),
            (dpo_history_path, "history.json"),
        ]:
            dpo_hub_api.upload_file(
                path_or_fileobj=_local_path, path_in_repo=_repo_path, repo_id=DPO_CHECKPOINT_REPO_ID, repo_type="model"
            )
            _local_path.unlink()
    return (dpo_hub_api,)


@app.cell(hide_code=True)
def _(dpo_hub_api):
    if dpo_hub_api is None:
        dpo_push_md = mo.md("Run DPO above to push the trained checkpoint.")
    else:
        dpo_push_md = mo.md(f"Pushed checkpoint, config and loss history to https://huggingface.co/{DPO_CHECKPOINT_REPO_ID}")
    dpo_push_md
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## TRL

    Not run here, hand-rolled is the point of the whole course. At
    work, this is `trl.DPOTrainer`, which handles batching, the
    reference model, multi-GPU training, and the loss above in one
    call:

    ```python
    from trl import DPOConfig, DPOTrainer

    config = DPOConfig(beta=0.1, learning_rate=5e-6, per_device_train_batch_size=32)
    trainer = DPOTrainer(model=policy_model, ref_model=reference_model, args=config, train_dataset=dpo_dataset)
    trainer.train()
    ```
    """)
    return


if __name__ == "__main__":
    app.run()

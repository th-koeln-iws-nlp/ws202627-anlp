import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import os
    import platform

    import numpy as np
    import tiktoken
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import marimo as mo
    from dotenv import load_dotenv
    from huggingface_hub import HfApi, hf_hub_download

    DATASET_REPO_ID = "Richard-Sieg-TH-Koln/anlp-tinystories-gpt2"
    VOCAB_SIZE = 50257
    BATCH_SIZE = 4
    CONTEXT_LENGTH = 64


@app.cell(hide_code=True)
def _():
    mo.md("""
    # 00. Smoke test

    Run this once before the course starts, top to bottom, no edits.
    Submit a screenshot of the last cell. This is a handshake, not a
    lesson: it proves your environment works, it does not teach
    anything, and it does not use any code from notebooks 01-06 (those
    aren't implemented yet at this point).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Environment
    """)
    return


@app.cell
def _():
    gpu_available = torch.cuda.is_available()
    mode = "gpu" if gpu_available else "local"
    device = torch.device("cuda" if gpu_available else "cpu")

    if mode == "gpu":
        device_name = torch.cuda.get_device_name(0)
        env_md = mo.md(
            f"""
            | | |
            |---|---|
            | mode | **GPU** |
            | device | {device_name} |
            | VRAM | {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB |
            | torch | {torch.__version__} |
            | CUDA | {torch.version.cuda} |
            | bf16 supported | {torch.cuda.is_bf16_supported()} |
            """
        )
    else:
        mps_available = torch.backends.mps.is_available()
        device_name = platform.platform()
        env_md = mo.md(
            f"""
            | | |
            |---|---|
            | mode | **local** |
            | platform | {platform.platform()} |
            | torch | {torch.__version__} |
            | Apple MPS available | {mps_available} |

            No GPU here, and that is expected and fine: this notebook
            scales itself down automatically. The course itself runs on
            molab, not on this machine.
            """
        )
    env_md
    return device, device_name, mode


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Hugging Face token
    """)
    return


@app.cell
def _(mode):
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")

    hf_username = None
    if not hf_token:
        token_status = "failed"
        token_message = (
            "No HF_TOKEN found in the environment. Locally: create a file "
            "named exactly `.env` in the project root containing "
            "`HF_TOKEN=hf_...`. On molab: add HF_TOKEN in the Secrets "
            "panel, not a `.env` you create yourself there, it has to be "
            "named exactly `.env` or it gets forked along with the "
            "notebook instead of staying secret."
        )
    else:
        try:
            hf_username = HfApi(token=hf_token).whoami()["name"]
            token_status = "ok"
            token_message = f"token is valid, logged in as {hf_username}"
        except Exception as e:
            token_status = "failed"
            token_message = (
                f"HF_TOKEN was found but the Hub rejected it "
                f"({type(e).__name__}). Check it hasn't expired or been "
                f"revoked at https://huggingface.co/settings/tokens, and "
                f"that {'.env' if mode == 'local' else 'the Secrets panel'} "
                f"has the current value."
            )

    mo.md(f"token check: {token_message}" if token_status == "ok" else f"**{token_message}**")
    return hf_token, hf_username, token_status


@app.cell(hide_code=True)
def _(mode):
    mo.md(
        f"""
        ## Data

        {"On GPU this downloads both train.bin (about 1 GB) and valid.bin, a few minutes the first time this ever runs on this machine, instant on every run after (it's cached). If it looks stuck for a couple of minutes, it isn't." if mode == "gpu" else "Local mode only fetches valid.bin, about 11 MB. The full ~1 GB train.bin is deliberately skipped here, that download belongs on molab, not on your laptop before the course."}
        """
    )
    return


@app.cell
def _(hf_token, mode):
    train_data = None
    val_data = None
    train_token_count = None
    valid_token_count = None

    try:
        valid_path = hf_hub_download(
            repo_id=DATASET_REPO_ID, filename="valid.bin", repo_type="dataset", token=hf_token
        )
        val_data = np.memmap(valid_path, dtype=np.uint16, mode="r")
        valid_token_count = len(val_data)

        if mode == "gpu":
            train_path = hf_hub_download(
                repo_id=DATASET_REPO_ID, filename="train.bin", repo_type="dataset", token=hf_token
            )
            train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
            train_token_count = len(train_data)

        data_status = "ok"
        data_message = f"valid: {valid_token_count:,} tokens" + (
            f", train: {train_token_count:,} tokens" if train_token_count is not None else ""
        )
    except Exception as e:
        data_status = "failed"
        data_message = (
            f"could not download or read the dataset ({type(e).__name__}: {e}). "
            f"Check your internet connection, and that HF_TOKEN above actually "
            f"passed."
        )

    mo.md(f"data check: {data_message}" if data_status == "ok" else f"**{data_message}**")
    return (
        data_status,
        train_data,
        train_token_count,
        val_data,
        valid_token_count,
    )


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Tokenizer

    Decoding real tokens from the file just downloaded, this is the
    check that dataset, tokenizer and encoding all line up end to end.
    It should read as a children's story.
    """)
    return


@app.cell
def _(val_data):
    tokenizer_preview = None
    if val_data is None:
        tokenizer_status = "failed"
        tokenizer_message = "skipped, the data check above did not succeed"
    else:
        try:
            gpt2_tokenizer = tiktoken.get_encoding("gpt2")
            tokenizer_preview = gpt2_tokenizer.decode(val_data[:500].tolist())
            tokenizer_status = "ok"
            tokenizer_message = "decoded below"
        except Exception as e:
            tokenizer_status = "failed"
            tokenizer_message = f"tiktoken failed ({type(e).__name__}: {e})"

    mo.md(
        f"```\n{tokenizer_preview}\n```" if tokenizer_status == "ok" else f"**tokenizer check: {tokenizer_message}**"
    )
    return (tokenizer_status,)


@app.cell(hide_code=True)
def _(mode):
    mo.md(f"""
    ## A tiny training run

    A small, self-contained model, deliberately unrelated to the
    course's own from-scratch GPT (notebooks 01-06 aren't implemented
    yet at this point, so nothing here imports from them). No attention,
    on purpose, this only needs to prove that a forward and backward
    pass actually compute on this device. The first loss should be near
    `ln(50257)` &asymp; 10.8, a correctly initialized but untrained model
    guessing uniformly at random over the vocabulary. The last loss
    should be lower.

    {"50 steps, emb_dim 128, 4 layers." if mode == "gpu" else "20 steps, emb_dim 64, 2 layers, batch 4, context 64. Well under a minute on a laptop CPU."}
    """)
    return


@app.cell
def _(device, mode, train_data, val_data):
    train_source = train_data if train_data is not None else val_data
    emb_dim = 128 if mode == "gpu" else 64
    n_layers = 4 if mode == "gpu" else 2
    num_steps = 50 if mode == "gpu" else 20

    first_loss = None
    last_loss = None

    if train_source is None:
        training_status = "failed"
        training_message = "skipped, no data available to train on"
    else:
        try:

            class SmokeTestModel(nn.Module):
                def __init__(self, vocab_size, emb_dim, n_layers):
                    super().__init__()
                    self.tok_emb = nn.Embedding(vocab_size, emb_dim)
                    self.blocks = nn.Sequential(
                        *[nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.GELU()) for _ in range(n_layers)]
                    )
                    self.out_head = nn.Linear(emb_dim, vocab_size)

                def forward(self, idx):
                    x = self.tok_emb(idx)  # (b, num_tokens, emb_dim)
                    x = self.blocks(x)  # (b, num_tokens, emb_dim)
                    return self.out_head(x)  # (b, num_tokens, vocab_size)

            # One fixed window from the start of the file, repeated across
            # the batch. Not the real batch sampler you will build in
            # 01_text_data.py (that one picks a fresh random window per
            # example, per step), this only needs one real
            # (input, target) pair to prove gradients flow on this device.
            window = torch.from_numpy(train_source[: CONTEXT_LENGTH + 1].astype(np.int64))
            x = window[:-1].unsqueeze(0).expand(BATCH_SIZE, -1).to(device)  # (batch_size, context_length)
            y = window[1:].unsqueeze(0).expand(BATCH_SIZE, -1).to(device)  # (batch_size, context_length)

            torch.manual_seed(42)
            smoke_model = SmokeTestModel(VOCAB_SIZE, emb_dim, n_layers).to(device)
            optimizer = torch.optim.AdamW(smoke_model.parameters(), lr=3e-4)

            step_losses = []
            for _step in range(num_steps):
                logits = smoke_model(x)  # (batch_size, context_length, vocab_size)
                # y came from .expand(), a non-contiguous view (all rows
                # share the same memory), so .view() would refuse it;
                # .reshape() copies if needed instead of erroring.
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step_losses.append(loss.item())

            first_loss, last_loss = step_losses[0], step_losses[-1]
            training_status = "ok"
            training_message = f"first loss {first_loss:.2f}, last loss {last_loss:.2f}"
        except Exception as e:
            training_status = "failed"
            training_message = f"training step failed ({type(e).__name__}: {e})"

    mo.md(f"training check: {training_message}" if training_status == "ok" else f"**{training_message}**")
    return first_loss, last_loss, training_status


@app.cell(hide_code=True)
def _(
    data_status,
    device_name,
    first_loss,
    hf_username,
    last_loss,
    mode,
    token_status,
    tokenizer_status,
    train_token_count,
    training_status,
    valid_token_count,
):
    def _mark(ok):
        return "✓" if ok else "✗"

    all_ok = all(
        status == "ok" for status in (token_status, data_status, tokenizer_status, training_status)
    )
    verdict = (
        "**Ready for the course.**"
        if mode == "gpu"
        else "**Setup OK, the course itself runs on molab.** A missing GPU here is not a problem."
    )
    data_line = f"valid: {valid_token_count:,} tokens" if valid_token_count else "valid: n/a"
    if train_token_count:
        data_line += f", train: {train_token_count:,} tokens"

    mo.md(
        f"""
        # Smoke test result

        | check | |
        |---|---|
        | environment | {_mark(True)} mode: **{mode}**, device: {device_name} |
        | token | {_mark(token_status == "ok")} {"logged in as " + hf_username if hf_username else "failed, see above"} |
        | data | {_mark(data_status == "ok")} {data_line} |
        | tokenizer | {_mark(tokenizer_status == "ok")} decoded a real sample |
        | training | {_mark(training_status == "ok")} {f"loss {first_loss:.2f} to {last_loss:.2f}" if training_status == "ok" else "failed, see above"} |

        {verdict if all_ok else "**Something above failed, read the check that says so, it explains what to do.**"}
        """
    )
    return


if __name__ == "__main__":
    app.run()

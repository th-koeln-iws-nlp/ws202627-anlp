import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")

with app.setup:
    import numpy as np
    import tiktoken
    import torch
    import marimo as mo
    from huggingface_hub import hf_hub_download

    DATASET_REPO_ID = "Richard-Sieg-TH-Koln/anlp-tinystories-gpt2"
    CONTEXT_LENGTH = 512
    BATCH_SIZE = 64


@app.cell(hide_code=True)
def _():
    mo.md("""
    # 01. Text data

    Raschka chapter 2: turning raw text into the (input, target) batches a
    GPT trains on. Two pieces: a tokenizer (`tiktoken`, we do not implement
    BPE ourselves) and a sliding-window sampler over the tokenized corpus.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Tokenizing with tiktoken

    GPT-2's byte pair encoding, exposed by `tiktoken`. `encode_ordinary`
    skips special-token handling, which is what we want for plain story
    text.
    """)
    return


@app.cell
def _():
    gpt2_tokenizer = tiktoken.get_encoding("gpt2")
    return (gpt2_tokenizer,)


@app.cell
def _(gpt2_tokenizer):
    sample_text = "Once upon a time there was a little robot who loved stories."
    sample_token_ids = gpt2_tokenizer.encode_ordinary(sample_text)
    mo.md(
        f"""
        `{sample_text!r}`

        tokenizes to {len(sample_token_ids)} ids:

        `{sample_token_ids}`
        """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## The dataset

    `train.bin` / `valid.bin`: TinyStoriesV2-GPT4, already tokenized with
    this exact GPT-2 encoding and stored as a flat `uint16` array with
    `<|endoftext|>` (token id 50256) between stories. `np.memmap` reads it
    off disk lazily, so opening it does not load 500MB+ into RAM.
    """)
    return


@app.cell
def _():
    train_bin_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="train.bin", repo_type="dataset")
    valid_bin_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="valid.bin", repo_type="dataset")

    train_data = np.memmap(train_bin_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(valid_bin_path, dtype=np.uint16, mode="r")
    mo.md(f"train: {len(train_data):,} tokens, valid: {len(val_data):,} tokens")
    return (train_data,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## The batch sampler

    Raschka 2.6, sliding-window sampling, nanoGPT style: pick
    `batch_size` random starting offsets into the token array, and for
    each one read a `context_length` window as the input and the same
    window shifted one token to the right as the target. That shift is
    the entire supervision signal for next-token prediction: at every
    position, the target is simply the next real token.
    """)
    return


@app.function
def get_batch(data, batch_size, context_length, device):
    """Random windows into a token memmap, nanoGPT style. Never loads the full array."""
    # Pick batch_size random starting offsets into `data` (a flat array
    # of token ids). Each offset must be small enough that a
    # context_length window starting there, plus one more token past
    # the end of it, still fits inside the array.
    #
    # For each offset, read a length-context_length window as one input
    # example (x), and the same window shifted one token later as its
    # target (y), so at every position the target is simply the next
    # real token. Stack the per-offset windows into two
    # (batch_size, context_length) tensors and move both to `device`
    # before returning them as (x, y).
    raise NotImplementedError("implement get_batch, Raschka 2.6")


@app.cell
def _():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (device,)


@app.cell
def _(device, train_data):
    demo_x, demo_y = get_batch(train_data, BATCH_SIZE, CONTEXT_LENGTH, device)
    assert demo_x.shape == (BATCH_SIZE, CONTEXT_LENGTH)
    assert demo_y.shape == (BATCH_SIZE, CONTEXT_LENGTH)
    assert torch.equal(demo_x[0, 1:], demo_y[0, :-1]), "y should be x shifted by one token"
    mo.md(f"x: {tuple(demo_x.shape)}, y: {tuple(demo_y.shape)}, y is x shifted by one token: confirmed")
    return (demo_x,)


@app.cell(hide_code=True)
def _(demo_x, gpt2_tokenizer):
    mo.md(f"""
    ## Decoding a sample

    The first row of that batch, decoded back to text, should read as
    an unremarkable mid-story fragment (it starts at a random offset,
    so it will not start at the beginning of a story):

    ```
    {gpt2_tokenizer.decode(demo_x[0].tolist())}
    ```
    """)
    return


if __name__ == "__main__":
    app.run()

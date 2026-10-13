import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")

with app.setup:
    import math

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Torch toolbox

    The torch functions you need for the gaps in notebooks 03, 04, 05, 07
    and 08, each with a tiny example you can run and change. Nothing here
    is the solution: it shows what a single function does, putting them
    together is your job. Read the output of every cell, most of what
    matters is the **shape**.

    Notation: `b` is the batch size, `T` the number of tokens, `d` the
    embedding size (`emb_dim`).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Notebook 03: `FeedForward` and `TransformerBlock`

    | You need | What it does |
    |---|---|
    | `nn.Linear(n_in, n_out)` | multiplies the **last** dimension by a learned matrix and adds a bias |
    | `nn.GELU(approximate="tanh")` | a smooth ReLU, shape unchanged |
    | `nn.Sequential(a, b, c)` | runs the modules one after the other, `seq[0]` is the first module |
    | `nn.Dropout(p)` | randomly zeroes entries while training, does nothing in eval mode |
    | `self.name = module` in `__init__` | registers the module, so its parameters are trained and saved |
    | `x + shortcut` | the residual connection, both must have the same shape |
    """)
    return


@app.cell
def _():
    toy_tokens = torch.randn(2, 5, 16)  # (b=2, T=5, d=16)
    toy_linear = nn.Linear(16, 64)
    mo.md(
        f"""
        `nn.Linear(16, 64)` on `{tuple(toy_tokens.shape)}` gives `{tuple(toy_linear(toy_tokens).shape)}`:
        only the last dimension changes, every token is processed on its own.
        It has `{sum(p.numel() for p in toy_linear.parameters())}` parameters:
        a `16 x 64` weight matrix plus `64` biases.
        """
    )
    return toy_linear, toy_tokens


@app.cell
def _(toy_tokens):
    toy_stack = nn.Sequential(nn.Linear(16, 32), nn.GELU(approximate="tanh"), nn.Linear(32, 16))
    mo.md(
        f"""
        `nn.Sequential` with three modules: `{tuple(toy_stack(toy_tokens).shape)}` out,
        the first module is `{toy_stack[0]}`, the last one `{toy_stack[2]}`.
        """
    )
    return


@app.cell
def _(toy_tokens):
    toy_dropout = nn.Dropout(0.5)
    toy_dropout.train()
    dropped = toy_dropout(toy_tokens[0, 0])
    toy_dropout.eval()
    kept = toy_dropout(toy_tokens[0, 0])
    mo.md(
        f"""
        `nn.Dropout(0.5)` in train mode zeroed `{int((dropped == 0).sum())}` of 16 numbers,
        in eval mode `{int((kept == 0).sum())}`. `model.train()` and `model.eval()` switch
        every dropout layer inside the model at once.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Notebook 04: `generate`

    | You need | What it does |
    |---|---|
    | `x[:, -k:]` | the last `k` entries along the second dimension |
    | `logits[:, -1, :]` | only the last position of every sequence, `(b, T, V)` becomes `(b, V)` |
    | `torch.argmax(x, dim=-1, keepdim=True)` | the index of the largest entry per row, `keepdim` keeps the shape `(b, 1)` |
    | `torch.softmax(x, dim=-1)` | turns scores into probabilities that sum to 1 per row |
    | `torch.multinomial(probs, num_samples=1)` | draws one random index per row, likelier indices more often |
    | `torch.topk(x, k)` | returns `(values, indices)` of the `k` largest entries per row |
    | `x[mask] = -torch.inf` | sets every entry where `mask` is `True` to minus infinity, softmax makes it 0 |
    | `torch.sort(x, descending=True, dim=-1)` | returns `(sorted_values, original_indices)` |
    | `torch.cumsum(x, dim=-1)` | running total along a dimension |
    | `Tensor.scatter(dim, index, src)` | writes `src` into the positions named by `index`, used to undo a sort |
    | `torch.cat([a, b], dim=1)` | glues tensors together along a dimension |
    | `model.eval()`, `model.train()`, `@torch.no_grad()` | dropout off or on, no gradient bookkeeping |
    """)
    return


@app.cell
def _():
    toy_scores = torch.tensor([[2.0, 0.5, 1.0, -1.0, 0.0], [0.1, 0.2, 3.0, 0.3, 0.4]])  # (b=2, V=5)
    toy_probs = torch.softmax(toy_scores, dim=-1)
    mo.md(
        f"""
        `argmax` per row: `{torch.argmax(toy_scores, dim=-1, keepdim=True).tolist()}` (shape
        `{tuple(torch.argmax(toy_scores, dim=-1, keepdim=True).shape)}`, so it can be glued onto a `(b, T)` tensor).

        `softmax` rows sum to `{toy_probs.sum(dim=-1).tolist()}`.

        `multinomial` draws, run the cell again and the draw changes:
        `{torch.multinomial(toy_probs, num_samples=1).tolist()}`
        """
    )
    return toy_probs, toy_scores


@app.cell
def _(toy_scores):
    top_values, top_indices = torch.topk(toy_scores, 2)
    kth_best = top_values[:, [-1]]  # (b, 1): the 2nd best score of every row, the extra [] keeps two dimensions
    filtered_scores = toy_scores.clone()
    filtered_scores[filtered_scores < kth_best] = -torch.inf  # comparing (b, V) with (b, 1) works row by row
    mo.md(
        f"""
        `topk(scores, 2)` values `{top_values.tolist()}`, indices `{top_indices.tolist()}`.

        Everything below the 2nd best score of its row becomes minus infinity:
        `{filtered_scores.tolist()}`
        """
    )
    return


@app.cell
def _():
    unsorted_scores = torch.tensor([[0.3, 2.0, 1.0]])
    sorted_scores, sorted_positions = torch.sort(unsorted_scores, descending=True, dim=-1)
    mo.md(
        f"""
        `sort` gives the sorted values `{sorted_scores.tolist()}` and where each one came from,
        `{sorted_positions.tolist()}`. `cumsum` of `[0.5, 0.3, 0.2]` is
        `{torch.cumsum(torch.tensor([0.5, 0.3, 0.2]), dim=-1).tolist()}`.
        """
    )
    return (sorted_positions,)


@app.cell
def _(sorted_positions):
    flags_in_sorted_order = torch.tensor([[False, True, True]])  # one flag per *sorted* entry
    flags_in_original_order = flags_in_sorted_order.scatter(1, sorted_positions, flags_in_sorted_order)
    mo.md(
        f"""
        `scatter` undoes a sort: the flags `{flags_in_sorted_order.tolist()}` belong to the sorted
        order `{sorted_positions.tolist()}`, put back in the original order they read
        `{flags_in_original_order.tolist()}`. Entry `j` of the sorted flags is written to the position
        `sorted_positions[j]`.
        """
    )
    return


@app.cell
def _():
    seq_so_far = torch.tensor([[5, 6, 7]])  # (b=1, T=3)
    new_token = torch.tensor([[8]])  # (b=1, 1)
    mo.md(f"`torch.cat([seq, new], dim=1)` grows `(1, 3)` to `{tuple(torch.cat([seq_so_far, new_token], dim=1).shape)}`: `{torch.cat([seq_so_far, new_token], dim=1).tolist()}`")
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Notebook 05: `evaluate`, `lr_at`, `train_step`

    | You need | What it does |
    |---|---|
    | `F.cross_entropy(logits, targets)` | the loss, `logits` is `(N, V)`, `targets` is `(N,)`, the result is one number |
    | `x.view(-1, V)`, `y.view(-1)` | reshape, the `-1` means "work this size out yourself" |
    | `loss.item()` | the plain Python float inside a one-number tensor |
    | `math.cos`, `math.pi` | cosine in radians, for the learning rate schedule |
    | `torch.autocast(device_type=..., dtype=..., enabled=...)` | a `with` block that runs the forward pass in lower precision |
    | `optimizer.zero_grad(set_to_none=True)` | throws away the gradients of the last step |
    | `loss.backward()` | computes the gradient of the loss for every parameter |
    | `torch.nn.utils.clip_grad_norm_(params, max_norm)` | scales the gradients down if their total norm is too big |
    | `optimizer.step()` | changes the parameters using the gradients |
    """)
    return


@app.cell
def _():
    toy_logits = torch.randn(2, 3, 7)  # (b=2, T=3, V=7)
    toy_targets = torch.randint(0, 7, (2, 3))  # (b=2, T=3)
    flat_logits = toy_logits.view(-1, toy_logits.size(-1))  # (6, 7)
    flat_targets = toy_targets.view(-1)  # (6,)
    toy_loss = F.cross_entropy(flat_logits, flat_targets)
    mo.md(
        f"""
        `cross_entropy` wants flat inputs: `{tuple(toy_logits.shape)}` becomes `{tuple(flat_logits.shape)}`,
        `{tuple(toy_targets.shape)}` becomes `{tuple(flat_targets.shape)}`. The loss is the tensor
        `{toy_loss}`, `.item()` gives the float `{toy_loss.item():.4f}`. A uniform
        guess would give `ln(7)` = `{math.log(7):.4f}`, random scores are a bit worse.
        """
    )
    return


@app.cell
def _():
    toy_weight = torch.tensor([3.0, 4.0], requires_grad=True)
    toy_optimizer = torch.optim.SGD([toy_weight], lr=0.1)
    (toy_weight**2).sum().backward()
    norm_before_clipping = torch.nn.utils.clip_grad_norm_([toy_weight], max_norm=1.0)
    gradient_after_clipping = toy_weight.grad.clone()
    toy_optimizer.step()
    toy_optimizer.zero_grad(set_to_none=True)
    mo.md(
        f"""
        `backward()` filled `weight.grad` with `[6, 8]`. `clip_grad_norm_` returned their norm
        `{norm_before_clipping.item():.1f}` and scaled the gradient down to `{gradient_after_clipping.tolist()}`
        (norm 1). `step()` moved the weight to `{[round(v, 2) for v in toy_weight.tolist()]}`,
        `zero_grad(set_to_none=True)` set `weight.grad` to `{toy_weight.grad}`.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Notebook 07: `get_instruct_batch`, `masked_mean`

    | You need | What it does |
    |---|---|
    | `np.random.randint(0, n, size=k)` | `k` random whole numbers from `0` up to `n - 1` |
    | `array[rows, :-1]` | pick the rows, then every column except the last |
    | `array[rows, 1:]` | pick the rows, then every column except the first |
    | `array.astype(np.int64)` | change the number type of a numpy array |
    | `torch.from_numpy(array)` | a torch tensor that shares the numpy data |
    | `tensor.to(device)` | move a tensor to the GPU or CPU |
    | `x.reshape(-1)`, `.float()` | flatten to 1D, turn into floats |
    | `x.sum()`, `x.clamp(min=1)` | add everything up, never go below 1 |
    """)
    return


@app.cell
def _():
    toy_table = np.arange(20).reshape(4, 5)  # (4 rows, 5 columns)
    toy_rows = np.random.randint(0, 4, size=2)
    mo.md(
        f"""
        Random rows `{toy_rows.tolist()}` of the table. Without the last column:
        `{toy_table[toy_rows, :-1].tolist()}`. Without the first column:
        `{toy_table[toy_rows, 1:].tolist()}`. As a tensor:
        `{torch.from_numpy(toy_table[toy_rows, :-1].astype(np.int64)).dtype}`.
        """
    )
    return


@app.cell
def _():
    toy_values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    toy_mask = torch.tensor([[1, 1, 0], [0, 0, 0]])
    flat_mask = toy_mask.reshape(-1).float()
    mo.md(
        f"""
        Flat mask `{flat_mask.tolist()}`, values times mask summed:
        `{(toy_values.reshape(-1) * flat_mask).sum().item()}`, number of positions that count:
        `{flat_mask.sum().item()}`. For an all-zero mask that count would be 0, `clamp(min=1)` turns it into
        `{torch.zeros(3).sum().clamp(min=1).item()}`.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Notebook 08: `compute_perplexity`

    | You need | What it does |
    |---|---|
    | `[f(i) for _ in range(n)]` | a list with `n` results, one call of `f` per entry |
    | `math.exp(x)` | `e` to the power `x`, the inverse of the logarithm in the loss |
    """)
    return


@app.cell
def _():
    toy_cross_entropy = 2.0
    mo.md(f"A loss of `{toy_cross_entropy}` nats is a perplexity of `math.exp({toy_cross_entropy})` = `{math.exp(toy_cross_entropy):.2f}`: about as uncertain as a choice among that many equally likely tokens.")
    return


if __name__ == "__main__":
    app.run()

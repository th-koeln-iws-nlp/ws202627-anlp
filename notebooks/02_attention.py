import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")

with app.setup:
    import torch
    import torch.nn as nn
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md("""
    # 02. Attention

    Raschka chapter 3. Built up in three stages: raw attention scores,
    the causal mask, then wrapping both in a batched, multi-head,
    trainable layer.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Stage 1: attention scores

    For every pair of positions, how much should one token's
    representation be influenced by another's? The simplest answer: the
    dot product of their embeddings. A high dot product means "similar
    direction", which we treat as "relevant".
    """)
    return


@app.cell
def _():
    torch.manual_seed(123)
    toy_embeddings = torch.randn(1, 6, 8)  # (b=1, num_tokens=6, d_in=8)
    toy_scores = toy_embeddings @ toy_embeddings.transpose(-2, -1)  # (1, 6, 6), token-to-token similarity
    mo.md(f"scores shape: {tuple(toy_scores.shape)}, scores[0, 2] (how token 2 relates to every token): {toy_scores[0, 2].tolist()}")
    return toy_embeddings, toy_scores


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Stage 2: the causal mask

    A GPT predicts the next token from the ones before it, so position i
    must not see positions after i. We enforce that by setting every
    score above the diagonal to -inf before the softmax, so softmax
    assigns those positions exactly zero weight.
    """)
    return


@app.cell
def _(toy_scores):
    toy_mask = torch.triu(torch.ones(6, 6), diagonal=1).bool()  # (6, 6), True above the diagonal
    toy_masked_scores = toy_scores.masked_fill(toy_mask, -torch.inf)  # (1, 6, 6)
    toy_weights = torch.softmax(toy_masked_scores, dim=-1)  # (1, 6, 6), each row sums to 1
    assert torch.allclose(toy_weights[0, 2, 3:], torch.zeros(3)), "token 2 must not attend to tokens 3, 4, 5"
    mo.md(f"row 2 of the attention weights (should be zero from index 3 on): {toy_weights[0, 2].tolist()}")
    return (toy_weights,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    Row 2's weight sits mostly on token 2 itself: a vector's dot product
    with itself is its squared length, while two random vectors average
    zero against each other. Learned queries and keys exist partly to
    break that.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    The context vector: the weighted sum of the embeddings, weighted by
    attention. This is $z_i$ from the slides, with the raw embeddings
    standing in here for values, properly introduced in stage 3.
    """)
    return


@app.cell
def _(toy_embeddings, toy_weights):
    toy_context = toy_weights @ toy_embeddings  # (1, 6, 8), zᵢ: each token's context vector
    hand_computed_row2 = sum(
        (toy_weights[0, 2, j] * toy_embeddings[0, j] for j in range(toy_weights.shape[-1])),
        start=torch.zeros(toy_embeddings.shape[-1]),
    )  # same sum, written out term by term
    assert torch.allclose(toy_context[0, 2], hand_computed_row2)
    mo.md(f"context vector shape: {tuple(toy_context.shape)}, row 2 matches the hand-computed weighted sum: confirmed")
    return


@app.cell
def _(toy_scores, toy_weights):
    scaling_mask = torch.triu(torch.ones(6, 6), diagonal=1).bool()  # (6, 6), same causal mask as stage 2
    toy_scaled_weights = torch.softmax(
        toy_scores.masked_fill(scaling_mask, -torch.inf) / 8**0.5, dim=-1
    )  # (1, 6, 6), same masked scores, scaled by 1/sqrt(d_in) = 1/sqrt(8) before softmax
    mo.md(
        f"row 2, unscaled: {toy_weights[0, 2].tolist()}\n\n"
        f"row 2, scaled by 1/sqrt(8) before softmax: {toy_scaled_weights[0, 2].tolist()}"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Stage 3: multi-head, trainable, batched

    `MultiHeadAttention`, Raschka 3.5-3.6. Same two ideas as above, scaled
    up: queries/keys/values now come from learned linear projections
    instead of being the raw embeddings, and the computation runs in
    parallel across `num_heads` independent, lower-dimensional subspaces
    (`head_dim = d_out / num_heads`) rather than one big one, so different
    heads can specialize.

    `forward` also takes an optional `attn_mask` argument. It does
    nothing yet, on Wednesday the instruction data arrives padded and
    this is what will mask it out.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    `build_blocking_mask` turns the causal mask and an optional padding
    mask into the one boolean mask `forward` fills scores with. It does
    nothing until block 07, where the instruction data arrives padded:
    right padding only needs the causal and padding masks combined, the
    fully-masked-row edge case below only shows up with left padding,
    which block 08's batched generation uses.
    """)
    return


@app.function
def build_blocking_mask(causal_mask, attn_mask=None):
    """causal_mask: (num_tokens, num_tokens), True where blocked (the
    causal mask buffer, already sliced to the current length).
    attn_mask: optional (b, num_tokens), 1 for a real token, 0 for
    padding. Returns one boolean mask broadcastable to
    (b, num_heads, num_tokens, num_tokens), True where blocked.
    """
    if attn_mask is None:
        return causal_mask

    num_tokens = causal_mask.shape[-1]
    padding_bool = attn_mask[:, None, None, :] == 0  # (b, 1, 1, num_tokens), True where the key position is padding
    mask_bool = causal_mask | padding_bool  # (b, 1, num_tokens, num_tokens), broadcasts over heads and query positions

    # A query position that is itself padding can end up with every key
    # masked too (only possible with left padding, where padding queries
    # sit before any real key the causal mask would allow): softmax over
    # an all -inf row is 0/0, NaN, and 0 * NaN is NaN, not 0, so that NaN
    # would survive being multiplied by a correctly-zeroed attention
    # weight and contaminate real positions in the next layer. Its own
    # output is never used, so give those rows one valid target,
    # themselves, instead.
    fully_masked_rows = mask_bool.all(dim=-1, keepdim=True)  # (b, 1, num_tokens, 1), True where a whole row is masked
    diagonal = torch.eye(num_tokens, dtype=torch.bool, device=mask_bool.device)  # (num_tokens, num_tokens)
    mask_bool = mask_bool & ~(fully_masked_rows & diagonal)  # unmask just the diagonal for those rows
    return mask_bool


@app.class_definition
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # Store what forward() will need: d_out, num_heads, and the
        # per-head dimension (d_out must split evenly across num_heads,
        # the constructor should assert that).
        #
        # Create three linear layers that each project a d_in-dimensional
        # input into d_out dimensions, for queries, keys and values
        # (bias controlled by qkv_bias), plus a fourth linear layer,
        # out_proj, that projects the combined d_out-dimensional context
        # vector back onto itself.
        #
        # Create a dropout layer for the attention weights.
        #
        # Register a causal mask buffer: an upper-triangular
        # (context_length, context_length) matrix of ones with the
        # diagonal itself left at zero, marking which positions are
        # "in the future" and must be blocked.
        raise NotImplementedError("implement MultiHeadAttention.__init__, Raschka 3.5-3.6")

    def forward(self, x, attn_mask=None):
        # x: (b, num_tokens, d_in)
        # attn_mask: optional (b, num_tokens), 1 for a real token, 0 for
        #   padding. Unused until block 07, see the note above.
        #
        # 1. Project x through the three linear layers to get queries,
        #    keys and values, each (b, num_tokens, d_out).
        # 2. Split each one's last dimension into (num_heads, head_dim)
        #    and move the head dimension next to the batch dimension, so
        #    each head can be processed independently:
        #    (b, num_heads, num_tokens, head_dim).
        # 3. Compute raw attention scores between every pair of
        #    positions, per head: (b, num_heads, num_tokens, num_tokens).
        # 4. Get the blocking mask from build_blocking_mask (the causal
        #    mask buffer from __init__, sliced down to num_tokens, and
        #    attn_mask), and set those scores to -inf before the
        #    softmax.
        # 5. Softmax over the last dimension, scaled by 1 / sqrt(head_dim)
        #    (stabilizes gradients, see Raschka 3.4), then dropout.
        # 6. Use the attention weights to combine the values, move the
        #    head dimension back next to num_tokens, and merge the heads
        #    back into a single (b, num_tokens, d_out) tensor.
        # 7. Pass that through out_proj and return it.
        raise NotImplementedError("implement MultiHeadAttention.forward, Raschka 3.5-3.6")


@app.cell
def _():
    mha = MultiHeadAttention(d_in=8, d_out=8, context_length=6, dropout=0.0, num_heads=2)
    mha
    return (mha,)


@app.cell
def _(mha, toy_embeddings):
    mha_out = mha(toy_embeddings)
    assert mha_out.shape == toy_embeddings.shape == (1, 6, 8)

    toy_attn_mask = torch.tensor([[1, 1, 1, 1, 0, 0]])  # (1, 6), last two positions are padding
    mha_masked_out = mha(toy_embeddings, attn_mask=toy_attn_mask)
    assert mha_masked_out.shape == toy_embeddings.shape == (1, 6, 8)

    mo.md(
        f"MultiHeadAttention output shape: {tuple(mha_out.shape)} (matches input shape, as it must for "
        f"stacking blocks), unchanged with attn_mask given: {tuple(mha_masked_out.shape)}"
    )
    return


if __name__ == "__main__":
    app.run()

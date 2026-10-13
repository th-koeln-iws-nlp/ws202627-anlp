import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")

with app.setup:
    import importlib

    import torch
    import torch.nn as nn
    import marimo as mo

    # MultiHeadAttention is defined in exactly one place, 02_attention.py,
    # and reused here rather than redefined.
    _attention = importlib.import_module("02_attention")
    MultiHeadAttention = _attention.MultiHeadAttention


@app.cell(hide_code=True)
def _():
    mo.md("""
    # 03. The transformer block

    Raschka chapter 4, sections 4.2-4.5: the three pieces around attention
    that turn it into a trainable, stackable layer. LayerNorm keeps
    activations well-scaled, FeedForward gives the model a place to do
    per-position computation, and TransformerBlock wires both of those
    plus attention together with residual (shortcut) connections.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## LayerNorm, Raschka 4.2

    Normalizes each token's embedding independently to zero mean and unit
    variance, then applies a learnable scale and shift so the network can
    undo that normalization if it turns out to be useful. Pre-LN
    (normalizing before each sublayer, not after) is what makes deep
    transformers trainable without careful learning-rate warmup tricks.

    This class is given complete, read it and run the check below. You
    write `FeedForward` and `TransformerBlock` yourself.
    """)
    return


@app.class_definition
class LayerNorm(nn.Module):
    """Given complete. Read it line by line, then write FeedForward and TransformerBlock in the same style."""

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # a small constant, so we never divide by zero
        self.scale = nn.Parameter(torch.ones(emb_dim))  # (emb_dim,), learnable, starts at 1
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # (emb_dim,), learnable, starts at 0

    def forward(self, x):
        # x: (b, num_tokens, emb_dim). Every token's vector is normalized on its own, over the last dimension.
        mean = x.mean(dim=-1, keepdim=True)  # (b, num_tokens, 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # (b, num_tokens, 1), the biased estimator
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  # (b, num_tokens, emb_dim), mean 0 and variance 1
        return self.scale * norm_x + self.shift  # (b, num_tokens, emb_dim), the network may undo the normalization


@app.cell
def _():
    torch.manual_seed(123)
    toy_batch = torch.randn(2, 5, 16)  # (b=2, num_tokens=5, emb_dim=16)
    layer_norm = LayerNorm(emb_dim=16)
    layer_norm
    return layer_norm, toy_batch


@app.cell
def _(layer_norm, toy_batch):
    normed = layer_norm(toy_batch)
    assert normed.shape == toy_batch.shape
    assert torch.allclose(normed.mean(dim=-1), torch.zeros(2, 5), atol=1e-5)
    mo.md(
        f"LayerNorm output shape: {tuple(normed.shape)}, per-token mean after norm: ~0 (confirmed)"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## FeedForward, Raschka 4.3

    A two-layer MLP applied to every position independently: expand to
    `4 * emb_dim`, apply GELU, project back down to `emb_dim`. This is
    where most of each block's parameters live, two thirds of them; in
    our model, most parameters overall sit in the token embedding
    table, 12,865,792 of 17,731,328, about 73 percent.
    """)
    return


@app.class_definition
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Goal: one attribute called self.layers that holds a small network.
        # Build it with nn.Sequential and put three modules into it, in this order:
        #   1. a Linear layer that makes every token vector four times as long
        #      (emb_dim comes in, 4 * emb_dim goes out)
        #   2. a GELU activation. nn.GELU has an argument called `approximate`,
        #      GPT-2 uses the "tanh" variant
        #   3. a Linear layer that goes back from 4 * emb_dim to emb_dim
        # The name self.layers matters: notebook 04 loads GPT-2's weights into
        # the two Linear layers by their position inside it.
        # Check yourself: how many parameters do the two Linear layers have
        # together for emb_dim = 16? (weights and biases)
        raise NotImplementedError("implement FeedForward.__init__, Raschka 4.3")

    def forward(self, x):
        # x: (b, num_tokens, emb_dim). Send x through self.layers and return
        # what comes out. That is the whole method: nn.Sequential runs its
        # modules one after the other, and nn.Linear only changes the last
        # dimension, so every position is processed on its own and the
        # shape stays (b, num_tokens, emb_dim).
        raise NotImplementedError("implement FeedForward.forward, Raschka 4.3")


@app.cell
def _():
    toy_ff_cfg = {"emb_dim": 16}
    feed_forward = FeedForward(toy_ff_cfg)
    feed_forward
    return (feed_forward,)


@app.cell
def _(feed_forward, toy_batch):
    ff_out = feed_forward(toy_batch)
    assert ff_out.shape == toy_batch.shape
    mo.md(
        f"FeedForward output shape: {tuple(ff_out.shape)} (matches input, as it must)"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## TransformerBlock, Raschka 4.5

    Attention mixes information across positions, FeedForward processes
    each position, and each is wrapped in "pre-norm, sublayer, add the
    shortcut back" so gradients have a direct path through every block
    regardless of depth.
    """)
    return


@app.class_definition
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # You need five attributes. The names matter, notebook 04 loads
        # GPT-2's weights into them by name:
        #   att            the attention layer: a MultiHeadAttention from
        #                  notebook 02. Pass it the same keyword arguments you
        #                  defined there (d_in, d_out, context_length,
        #                  num_heads, dropout, qkv_bias) and read the values
        #                  from cfg. Careful with the names: cfg calls the
        #                  number of heads "n_heads" and the dropout rate
        #                  "drop_rate". d_in and d_out are both emb_dim.
        #   ff             a FeedForward, it takes the whole cfg
        #   norm1, norm2   two separate LayerNorms over emb_dim. Why two and
        #                  not one? Each LayerNorm has its own learnable
        #                  scale and shift.
        #   drop_shortcut  one nn.Dropout with the rate drop_rate, used after
        #                  both sublayers
        raise NotImplementedError("implement TransformerBlock.__init__, Raschka 4.5")

    def forward(self, x, attn_mask=None):
        # x: (b, num_tokens, emb_dim). The block has two sublayers, first
        # attention, then the feed-forward. Both follow the same recipe,
        # called pre-norm with a residual connection:
        #   (1) keep a copy of the input, this is the shortcut
        #   (2) normalize the input
        #   (3) run the sublayer on the normalized input
        #   (4) apply dropout to the result
        #   (5) add the shortcut back
        # Do the recipe once for attention, with norm1, and a second time for
        # the feed-forward, with norm2, starting from the result of the first.
        # Two details: the attention takes attn_mask as a keyword argument, the
        # feed-forward does not. Return x, its shape must still be
        # (b, num_tokens, emb_dim), that is what lets blocks stack.
        # Why the shortcut? Without it, gradients would have to pass through
        # every sublayer of every block on their way back.
        raise NotImplementedError("implement TransformerBlock.forward, Raschka 4.5")


@app.cell
def _():
    toy_block_cfg = {
        "emb_dim": 16,
        "context_length": 5,
        "n_heads": 2,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    transformer_block = TransformerBlock(toy_block_cfg)
    transformer_block
    return (transformer_block,)


@app.cell
def _(toy_batch, transformer_block):
    block_out = transformer_block(toy_batch)
    assert block_out.shape == toy_batch.shape
    mo.md(
        f"TransformerBlock output shape: {tuple(block_out.shape)} (matches input, so blocks can stack)"
    )
    return


if __name__ == "__main__":
    app.run()

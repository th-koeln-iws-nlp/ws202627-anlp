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
        # Build one attribute, self.layers, with nn.Sequential holding
        # three modules in exactly this order:
        #   1. nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"])   expands every token's vector to 4 times the size
        #   2. nn.GELU(approximate="tanh")                     the non-linearity
        #   3. nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])   projects back down to emb_dim
        # Keep the name self.layers: notebook 04 loads GPT-2's weights
        # into self.layers[0] and self.layers[2].
        raise NotImplementedError("implement FeedForward.__init__, Raschka 4.3")

    def forward(self, x):
        # x: (b, num_tokens, emb_dim). One line: return what self.layers
        # gives for x. nn.Linear only looks at the last dimension, so
        # every position is processed independently. The output has
        # the same shape as x.
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
        # Create these five attributes, with exactly these names (notebook
        # 04 loads GPT-2's weights by name):
        #   self.att = MultiHeadAttention(d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
        #                                 context_length=cfg["context_length"], num_heads=cfg["n_heads"],
        #                                 dropout=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"])
        #   self.ff = FeedForward(cfg)
        #   self.norm1 = LayerNorm(cfg["emb_dim"])      goes before the attention
        #   self.norm2 = LayerNorm(cfg["emb_dim"])      goes before the feed-forward
        #   self.drop_shortcut = nn.Dropout(cfg["drop_rate"])   used after both sublayers
        raise NotImplementedError("implement TransformerBlock.__init__, Raschka 4.5")

    def forward(self, x, attn_mask=None):
        # x: (b, num_tokens, emb_dim). Two sublayers, one after the other.
        # The attention sublayer, in five steps:
        #   1. shortcut = x
        #   2. x = self.norm1(x)
        #   3. x = self.att(x, attn_mask=attn_mask)
        #   4. x = self.drop_shortcut(x)
        #   5. x = x + shortcut         the residual connection
        # The feed-forward sublayer: the same five steps again, starting
        # from the new x, with self.norm2 and self.ff(x) in steps 2 and 3
        # (no attn_mask there).
        # Return x. Its shape is the same as the input, which is what
        # lets blocks stack.
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

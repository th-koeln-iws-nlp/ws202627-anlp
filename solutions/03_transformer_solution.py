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
    _attention = importlib.import_module("02_attention_solution")
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
    """)
    return


@app.class_definition
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # (b, num_tokens, 1)
        var = x.var(
            dim=-1, keepdim=True, unbiased=False
        )  # (b, num_tokens, 1)
        norm_x = (x - mean) / torch.sqrt(
            var + self.eps
        )  # (b, num_tokens, emb_dim)
        return self.scale * norm_x + self.shift


@app.cell
def _():
    torch.manual_seed(123)
    toy_batch = torch.randn(2, 5, 16)  # (b=2, num_tokens=5, emb_dim=16)
    layer_norm = LayerNorm(emb_dim=16)
    normed = layer_norm(toy_batch)
    assert normed.shape == toy_batch.shape
    assert torch.allclose(normed.mean(dim=-1), torch.zeros(2, 5), atol=1e-5)
    mo.md(
        f"LayerNorm output shape: {tuple(normed.shape)}, per-token mean after norm: ~0 (confirmed)"
    )
    return (toy_batch,)


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
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


@app.cell
def _(toy_batch):
    toy_ff_cfg = {"emb_dim": 16}
    feed_forward = FeedForward(toy_ff_cfg)
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
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, attn_mask=None):
        shortcut = x  # (b, num_tokens, emb_dim)
        x = self.norm1(x)
        x = self.att(x, attn_mask=attn_mask)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


@app.cell
def _(toy_batch):
    toy_block_cfg = {
        "emb_dim": 16,
        "context_length": 5,
        "n_heads": 2,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    transformer_block = TransformerBlock(toy_block_cfg)
    block_out = transformer_block(toy_batch)
    assert block_out.shape == toy_batch.shape
    mo.md(
        f"TransformerBlock output shape: {tuple(block_out.shape)} (matches input, so blocks can stack)"
    )
    return


if __name__ == "__main__":
    app.run()

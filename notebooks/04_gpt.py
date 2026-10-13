import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")

with app.setup:
    import importlib

    import tiktoken
    import torch
    import torch.nn as nn
    import marimo as mo
    from huggingface_hub import hf_hub_download

    # TransformerBlock and LayerNorm are defined in exactly one place,
    # 03_transformer.py, and reused here rather than redefined.
    _transformer = importlib.import_module("03_transformer")
    TransformerBlock = _transformer.TransformerBlock
    LayerNorm = _transformer.LayerNorm

    # Settled config for the model this course actually trains: small
    # (emb_dim 256, 6 layers, 8 heads), see 05_pretraining.py. Do not
    # re-derive these, they came out of a calibration pass on the real
    # training hardware.
    VOCAB_SIZE = 50257
    CONTEXT_LENGTH = 512
    EMB_DIM = 256
    N_LAYERS = 6
    N_HEADS = 8
    DROP_RATE = 0.1
    QKV_BIAS = False  # training from scratch, not loading OpenAI's checkpoint

    # A second, separate config: the original OpenAI GPT-2 (124M)
    # architecture, only used for the weight-loading demo below. Its shape
    # does not match the course model, that's expected, it is a different
    # (much larger) network.
    GPT2_124M_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_layers": 12,
        "n_heads": 12,
        "drop_rate": 0.0,
        "qkv_bias": True,  # the OpenAI checkpoint has bias terms on c_attn/c_proj/c_fc
    }
    GPT2_REPO_ID = "openai-community/gpt2"

    EOT = 50256  # <|endoftext|>

    # chat_window's defaults, course-wide: 100 tokens, temperature 0.8,
    # top-k 50, top-p 1.0 (no nucleus restriction by default).
    CHAT_DEFAULT_CONFIG = {"max_tokens": 100, "temperature": 0.8, "top_k": 50, "top_p": 1.0}


@app.cell(hide_code=True)
def _():
    mo.md("""
    # 04. GPT

    Raschka chapter 4.6: assembling embeddings, positional information,
    and a stack of transformer blocks into one model. Then two things the
    book does not build a full model for: loading OpenAI's real GPT-2
    weights into our from-scratch architecture, and a chat interface to
    poke at the result.
    """)
    return


@app.cell
def _():
    gpt2_tokenizer = tiktoken.get_encoding("gpt2")
    return (gpt2_tokenizer,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## GPTModel, Raschka 4.6

    Token embedding + positional embedding, dropout, `n_layers` stacked
    `TransformerBlock`s, a final `LayerNorm`, then a linear head back out
    to vocabulary size. `out_head` is tied to `tok_emb`: the same matrix
    that turns token ids into vectors also turns vectors back into
    per-token logits, which roughly halves the embedding parameter count.

    `_init_weights` matters more than it looks: PyTorch's default
    `nn.Embedding` init is `N(0, 1)`, and because of the weight tying that
    directly sets the scale of the untrained output logits. Left at the
    default, the model starts around 30+ nats of loss instead of
    `ln(vocab_size)` &asymp; 10.8, the loss any untrained classifier should
    start at. GPT-2 uses `N(0, 0.02)` for every linear and embedding
    weight, and zero for biases, that is what `_init_weights` applies via
    `self.apply(...)` before the tying happens.

    This class is given complete, read it and run the cells below. The
    function you write in this notebook is `generate`.
    """)
    return


@app.class_definition
class GPTModel(nn.Module):
    """Given complete. Read it top to bottom: it only assembles the pieces you built in 02 and 03."""

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  # one vector per token id
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # one vector per position
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # nn.ModuleList, not nn.Sequential: forward() has to pass attn_mask to every block
        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        self.apply(self._init_weights)  # runs _init_weights once for every submodule
        self.out_head.weight = self.tok_emb.weight  # weight tying: both layers share one tensor

    def _init_weights(self, module):
        # GPT-2: every linear and embedding weight from N(0, 0.02), every linear bias zero
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, in_idx, attn_mask=None):
        batch_size, seq_len = in_idx.shape  # (b, num_tokens)
        tok_embeds = self.tok_emb(in_idx)  # (b, num_tokens, emb_dim)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # (num_tokens, emb_dim)
        x = tok_embeds + pos_embeds  # (b, num_tokens, emb_dim), the position vectors broadcast over the batch
        x = self.drop_emb(x)
        for block in self.trf_blocks:
            x = block(x, attn_mask=attn_mask)  # (b, num_tokens, emb_dim)
        x = self.final_norm(x)
        logits = self.out_head(x)  # (b, num_tokens, vocab_size)
        return logits


@app.function
def build_config():
    return {
        "vocab_size": VOCAB_SIZE,
        "context_length": CONTEXT_LENGTH,
        "emb_dim": EMB_DIM,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "drop_rate": DROP_RATE,
        "qkv_bias": QKV_BIAS,
    }


@app.cell
def _():
    torch.manual_seed(42)
    course_cfg = build_config()
    gpt = GPTModel(course_cfg)
    gpt
    return gpt, course_cfg


@app.cell
def _(gpt, course_cfg):
    num_params = sum(p.numel() for p in gpt.parameters())

    dummy_ids = torch.randint(0, course_cfg["vocab_size"], (2, 10))  # (b=2, num_tokens=10)
    dummy_logits = gpt(dummy_ids)
    assert dummy_logits.shape == (2, 10, course_cfg["vocab_size"])
    mo.md(f"{num_params:,} parameters (about 17.7M, as expected), forward pass output shape: {tuple(dummy_logits.shape)}")
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Generating text, Raschka 5.3

    At each step: run the model on the current sequence (cropped to the
    last `context_length` tokens, the model has no memory beyond that
    window), keep only the last position's logits, restrict to the
    `top_k` most likely next tokens, and sample from that restricted
    distribution scaled by `temperature`. Sampling instead of always
    taking the single most likely token (greedy decoding) matters: greedy
    decoding tends to loop into repetition ("the cat. the cat."),
    especially before training, which understates how coherent a model
    actually is.
    """)
    return


@app.function
@torch.no_grad()
def generate(model, idx, max_new_tokens, context_length, temperature=0.8, top_k=50, top_p=None):
    # idx: (b, num_tokens), the prompt to continue.
    # Start with model.eval() and call model.train() just before you return.
    #
    # Then repeat max_new_tokens times, with a plain for loop:
    #   a. idx_cond = idx[:, -context_length:]        the model only sees the last context_length tokens
    #   b. logits = model(idx_cond)[:, -1, :]         (b, vocab_size), the last position is the prediction for the next token
    #   c. If temperature == 0: next_id = logits.argmax(dim=-1, keepdim=True)  (b, 1),
    #      then jump to step h (greedy decoding, no sampling).
    #   d. logits = logits / temperature
    #   e. If top_k is not None: top_values, _ = torch.topk(logits, top_k), then
    #      logits[logits < top_values[:, [-1]]] = -torch.inf   (everything below the k-th best score is out)
    #   f. If top_p is not None, in this order (do steps a, b, d, e, g, h first and come back for this one):
    #        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    #        sorted_probs = torch.softmax(sorted_logits, dim=-1)
    #        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    #        sorted_remove = cumulative_probs - sorted_probs > top_p     True for tokens beyond the top_p mass
    #        sorted_remove[:, 0] = False                                 always keep the single best token
    #        remove = sorted_remove.scatter(1, sorted_indices, sorted_remove)   back to the original token order
    #        logits[remove] = -torch.inf
    #   g. probs = torch.softmax(logits, dim=-1), then
    #      next_id = torch.multinomial(probs, num_samples=1)   (b, 1), a random draw from probs
    #   h. idx = torch.cat([idx, next_id], dim=1)      (b, num_tokens + 1)
    #
    # Return idx: the prompt plus every generated token.
    raise NotImplementedError("implement generate, Raschka 5.3")


@app.cell
def _(gpt, gpt2_tokenizer, course_cfg):
    determinism_prompt_ids = torch.tensor([gpt2_tokenizer.encode_ordinary("Once upon a time")])
    greedy_1 = generate(gpt, determinism_prompt_ids, max_new_tokens=10, context_length=course_cfg["context_length"], temperature=0)
    greedy_2 = generate(gpt, determinism_prompt_ids, max_new_tokens=10, context_length=course_cfg["context_length"], temperature=0)
    assert torch.equal(greedy_1, greedy_2), "temperature=0 must be deterministic"

    torch.manual_seed(0)
    sampled_top_p_none = generate(gpt, determinism_prompt_ids, max_new_tokens=10, context_length=course_cfg["context_length"], top_p=None)
    torch.manual_seed(0)
    sampled_top_p_one = generate(gpt, determinism_prompt_ids, max_new_tokens=10, context_length=course_cfg["context_length"], top_p=1.0)
    assert torch.equal(sampled_top_p_none, sampled_top_p_one), "top_p=1.0 must match top_p=None under the same seed"

    mo.md(
        "generate() asserts, on the untrained model so no download is needed: temperature=0 is deterministic, "
        "top_p=1.0 matches top_p=None under the same seed"
    )
    return


@app.cell
def _(gpt, gpt2_tokenizer, course_cfg):
    untrained_prompt_ids = torch.tensor([gpt2_tokenizer.encode_ordinary("Once upon a time")])
    untrained_output_ids = generate(gpt, untrained_prompt_ids, max_new_tokens=20, context_length=course_cfg["context_length"])
    mo.md(
        f"""
        Untrained model, sampled: expect gibberish, this GPTModel has
        never seen a single training step yet, that happens in
        `05_pretraining.py`.

        ```
        {gpt2_tokenizer.decode(untrained_output_ids[0].tolist())}
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## One chat window, five checkpoints

    - `chat_window` wraps `mo.ui.chat` around your own `generate`. The
      chat window sends your message to `generate` and streams the
      result back, one token at a time, so there is no second, separate
      generation implementation to keep in sync with the real one.
    - Only the last user message is used: base models have no notion of
      a conversation.
    - The window stays the same all course. Behind it sit five models,
      one per checkpoint this course produces: today the first two,
      then `05_pretraining.py`, `07_sft.py` and the DPO notebook.
    """)
    return


@app.function
def decode_delta(tokenizer, token_ids, already_yielded):
    """The new text to yield for a streaming decode, given every new
    token id generated so far and the text already yielded for them.

    Decodes the whole run of token_ids every call, since a later token
    can change how an earlier one decodes: a curly quote is two GPT-2
    tokens, and decoding the first alone ends in the replacement
    character, an incomplete UTF-8 sequence, not a real character, so
    any trailing one is held back until a later token completes it.
    Returns (new_text, updated_already_yielded).
    """
    decoded = tokenizer.decode(token_ids).rstrip("�")
    new_text = decoded[len(already_yielded) :]
    return new_text, decoded


@app.function
def chat_window(model, tokenizer, context_length, prompts=None, format_prompt=None):
    """One mo.ui.chat, reused for every checkpoint this course produces.
    Streams the student's own generate(), one token at a time: calling
    it once per token costs the same as one long call, since generate
    has no KV cache and reprocesses the whole context every step
    anyway. Only the last user message is used, base models have no
    notion of a conversation; format_prompt, if given, wraps it in an
    instruction format before it reaches the model.
    """

    def stream_model(messages, config):
        prompt = messages[-1].content
        if format_prompt is not None:
            prompt = format_prompt(prompt)

        max_new_tokens = CHAT_DEFAULT_CONFIG["max_tokens"] if config.max_tokens is None else config.max_tokens
        temperature = CHAT_DEFAULT_CONFIG["temperature"] if config.temperature is None else config.temperature
        top_k = CHAT_DEFAULT_CONFIG["top_k"] if config.top_k is None else config.top_k
        top_p = CHAT_DEFAULT_CONFIG["top_p"] if config.top_p is None else config.top_p

        device = next(model.parameters()).device
        idx = torch.tensor([tokenizer.encode_ordinary(prompt)], device=device)  # (1, prompt_len)

        new_token_ids = []
        yielded = ""
        for _ in range(max_new_tokens):
            idx = generate(
                model, idx, max_new_tokens=1, context_length=context_length,
                temperature=temperature, top_k=top_k, top_p=top_p,
            )
            next_id = idx[0, -1].item()
            if next_id == EOT:
                break
            new_token_ids.append(next_id)
            delta, yielded = decode_delta(tokenizer, new_token_ids, yielded)
            if delta:
                yield delta

    return mo.ui.chat(
        stream_model,
        prompts=prompts,
        show_configuration_controls=True,
        config=CHAT_DEFAULT_CONFIG,
    )


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Chat, checkpoint 1 of 5: random weights

    The first of five checkpoints this course's one chat window gets
    used on, across this notebook and the ones that follow: random
    weights, nothing trained yet. Expect noise.
    """)
    return


@app.cell
def _(gpt, gpt2_tokenizer, course_cfg):
    chat_checkpoint_1 = chat_window(
        gpt, gpt2_tokenizer, course_cfg["context_length"],
        prompts=["Once upon a time", "The best way to learn a new language is"],
    )
    chat_checkpoint_1
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## GPT-2 small is our model, only bigger

    | | GPT-2 small | Our model |
    |---|---|---|
    | Context | 1,024 | 512 |
    | Width | 768 | 256 |
    | Layers | 12 | 6 |
    | Heads | 12 | 8 |
    | Bias on queries, keys, values | yes | no |
    | Parameters, tied | 124 million | 17.7 million |
    | Vocabulary | 50,257 | 50,257 |

    Same `GPTModel` class, a different configuration dictionary
    (`GPT2_124M_CONFIG` instead of `build_config()`). The weights have to
    match the shapes exactly.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Loading OpenAI's GPT-2 weights


    OpenAI's checkpoint (mirrored on the Hub as plain PyTorch tensors, no
    `transformers` needed) stores the same numbers as our model, just
    differently, so loading is a translation job:

    - Every linear weight is stored transposed, `(in_features,
      out_features)` instead of PyTorch's `(out_features, in_features)`
      (called Conv1D layout). Transpose on the way in.
    - Queries, keys and values sit in one matrix per block, `c_attn`,
      three times as wide: `(768, 2304)` for GPT-2 small. Split it in
      three along the output dimension and transpose each piece to
      `(768, 768)`, one each for `W_query`, `W_key` and `W_value`.
    - The checkpoint has biases on queries, keys and values (and on
      `c_proj` and `c_fc`), so this model is built with
      `qkv_bias=True`, unlike the course model we train ourselves.
    - Loading overwrites the token table, so the output layer has to
      share it again afterwards. The code re-ties `out_head` explicitly,
      to be safe.
    """)
    return


@app.function
def load_gpt2_weights(model, state_dict):
    """Copy OpenAI's GPT-2 weights into our GPTModel, in place."""
    n_layers = len(model.trf_blocks)

    model.tok_emb.weight.data = state_dict["wte.weight"]  # (vocab_size, emb_dim), no transpose needed
    model.pos_emb.weight.data = state_dict["wpe.weight"]  # (context_length, emb_dim)

    for i in range(n_layers):
        prefix = f"h.{i}."
        block = model.trf_blocks[i]

        q_w, k_w, v_w = state_dict[prefix + "attn.c_attn.weight"].T.chunk(3, dim=0)  # each (emb_dim, emb_dim)
        q_b, k_b, v_b = state_dict[prefix + "attn.c_attn.bias"].chunk(3, dim=0)  # each (emb_dim,)
        block.att.W_query.weight.data = q_w
        block.att.W_key.weight.data = k_w
        block.att.W_value.weight.data = v_w
        block.att.W_query.bias.data = q_b
        block.att.W_key.bias.data = k_b
        block.att.W_value.bias.data = v_b

        block.att.out_proj.weight.data = state_dict[prefix + "attn.c_proj.weight"].T  # (emb_dim, emb_dim)
        block.att.out_proj.bias.data = state_dict[prefix + "attn.c_proj.bias"]

        block.ff.layers[0].weight.data = state_dict[prefix + "mlp.c_fc.weight"].T  # (4 * emb_dim, emb_dim)
        block.ff.layers[0].bias.data = state_dict[prefix + "mlp.c_fc.bias"]
        block.ff.layers[2].weight.data = state_dict[prefix + "mlp.c_proj.weight"].T  # (emb_dim, 4 * emb_dim)
        block.ff.layers[2].bias.data = state_dict[prefix + "mlp.c_proj.bias"]

        block.norm1.scale.data = state_dict[prefix + "ln_1.weight"]
        block.norm1.shift.data = state_dict[prefix + "ln_1.bias"]
        block.norm2.scale.data = state_dict[prefix + "ln_2.weight"]
        block.norm2.shift.data = state_dict[prefix + "ln_2.bias"]

    model.final_norm.scale.data = state_dict["ln_f.weight"]
    model.final_norm.shift.data = state_dict["ln_f.bias"]
    model.out_head.weight = model.tok_emb.weight  # re-tie after overwriting tok_emb
    return model


@app.cell
def _():
    gpt2_weights_path = hf_hub_download(repo_id=GPT2_REPO_ID, filename="pytorch_model.bin")
    gpt2_state_dict = torch.load(gpt2_weights_path, map_location="cpu", weights_only=True)
    return (gpt2_state_dict,)


@app.cell
def _(gpt2_state_dict):
    torch.manual_seed(42)
    gpt2_model = GPTModel(GPT2_124M_CONFIG)
    load_gpt2_weights(gpt2_model, gpt2_state_dict)
    gpt2_model
    return (gpt2_model,)


@app.cell
def _(gpt2_model):
    gpt2_num_params = sum(p.numel() for p in gpt2_model.parameters())
    mo.md(f"loaded {gpt2_num_params:,} parameters into our GPTModel")
    return


@app.cell
def _(gpt2_model, gpt2_tokenizer):
    loaded_prompt_ids = torch.tensor([gpt2_tokenizer.encode_ordinary("Once upon a time")])
    loaded_output_ids = generate(
        gpt2_model, loaded_prompt_ids, max_new_tokens=30, context_length=GPT2_124M_CONFIG["context_length"]
    )
    mo.md(
        f"""
        Same architecture, real weights: expect actual English, this
        proves the loading code above is correct, a shape mismatch or a
        missed transpose would produce garbage just like the untrained
        model above.

        ```
        {gpt2_tokenizer.decode(loaded_output_ids[0].tolist())}
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    Temperature controls how sharply sampling favors the top candidate:
    greedy (`temperature=0`) tends to loop into exact repetition, a
    temperature above 1 flattens the distribution enough that it drifts
    into nonsense.
    """)
    return


@app.cell
def _(gpt2_model, gpt2_tokenizer):
    temperature_demo_prompt_ids = torch.tensor([gpt2_tokenizer.encode_ordinary("Once upon a time")])
    temperature_demo_outputs = {
        temp: generate(
            gpt2_model, temperature_demo_prompt_ids, max_new_tokens=30,
            context_length=GPT2_124M_CONFIG["context_length"], temperature=temp,
        )
        for temp in (0, 0.8, 1.5)
    }
    mo.md(
        "\n\n".join(
            f"**temperature={temp}**\n```\n{gpt2_tokenizer.decode(ids[0].tolist())}\n```"
            for temp, ids in temperature_demo_outputs.items()
        )
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Chat, checkpoint 2 of 5: GPT-2's own weights

    GPT-2 was trained to continue text, not to follow instructions or
    hold a conversation (block 6, "The Story of LLMs", covers how
    InstructGPT changed that). Ask it a question and it writes more text
    that looks like where questions appear: fluent, but not an answer.
    """)
    return


@app.cell
def _(gpt2_model, gpt2_tokenizer):
    chat_checkpoint_2 = chat_window(
        gpt2_model, gpt2_tokenizer, GPT2_124M_CONFIG["context_length"],
        prompts=["Once upon a time", "The best way to learn a new language is"],
    )
    chat_checkpoint_2
    return


if __name__ == "__main__":
    app.run()

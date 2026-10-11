import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")

with app.setup:
    import json
    import sys
    from pathlib import Path

    import marimo as mo
    import tiktoken
    import torch

    COMPONENT_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(COMPONENT_DIR.parent))  # ablation_harness.py lives in components/

    from ablation_harness import ablation_config
    from starter import (
        generate_greedy,
        generate_greedy_cached,
        load_baseline_checkpoint,
        measure_generation_throughput,
    )

    CHECKPOINT_PATH = COMPONENT_DIR / "runs" / "baseline_seed0.pt"
    PROMPT = "Once upon a time"
    MAX_NEW_TOKENS_LIST = (50, 500)  # the speedup should grow with the length


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Component 08: KV cache

    Three steps: load your trained baseline, check that the cached
    generation produces exactly the same tokens as the uncached one,
    then measure how much faster it is. Run it once on the GPU and once
    on the CPU, the table at the bottom collects both.

    You need the weights of your baseline run,
    `runs/baseline_seed0.pt`: train the baseline first in
    `components/run_ablation.py` (component `08_kv_cache`, variant
    `baseline`), the file appears after the first seed.
    """)
    return


@app.cell
def _():
    device_names = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    device_choice = mo.ui.dropdown(device_names, value=device_names[-1], label="Device")
    device_choice
    return (device_choice,)


@app.cell
def _(device_choice):
    mo.stop(
        not CHECKPOINT_PATH.exists(),
        mo.md("No baseline weights yet. Train the baseline first, see the top of this notebook."),
    )
    device = torch.device(device_choice.value)
    model = load_baseline_checkpoint(ablation_config(), CHECKPOINT_PATH).to(device)  # fp32
    tokenizer = tiktoken.get_encoding("gpt2")
    prompt_ids = torch.tensor([tokenizer.encode_ordinary(PROMPT)], device=device)  # (1, prompt_len)
    mo.md(f"Loaded `{CHECKPOINT_PATH.name}` on `{device}`, prompt: `{PROMPT}` ({prompt_ids.shape[1]} tokens).")
    return device, model, prompt_ids, tokenizer


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Correctness first

    Same weights, same prompt, greedy decoding, fp32: the two token
    sequences must be identical. A mismatch means something in the
    cache logic is wrong. Do not run this in bf16, rounding can flip an
    argmax between two nearly tied logits and look like a bug.
    """)
    return


@app.cell
def _(model, prompt_ids, tokenizer):
    uncached_tokens = generate_greedy(model, prompt_ids.clone(), max_new_tokens=200)  # (1, prompt_len + 200)
    cached_tokens = generate_greedy_cached(model, prompt_ids.clone(), max_new_tokens=200)  # (1, prompt_len + 200)

    if torch.equal(uncached_tokens, cached_tokens):
        correctness_md = mo.md(
            f"**Identical** over 200 new tokens.\n\n```\n{tokenizer.decode(cached_tokens[0].tolist())}\n```"
        )
    else:
        first_difference = (uncached_tokens[0] != cached_tokens[0]).nonzero()[0].item()
        correctness_md = mo.md(
            f"**Different.** The first mismatch is at position {first_difference} "
            f"(the prompt has {prompt_ids.shape[1]} tokens). Fix this before you measure speed."
        )
    correctness_md
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Throughput

    Tokens per second for 50 and for 500 new tokens, uncached and
    cached, median of five runs after one warm-up run. The uncached
    cost per token grows with the sequence, the cached one does not, so
    the gap should be wider at 500. Switch the device above and measure
    again, the table keeps both.
    """)
    return


@app.cell
def _():
    measure_button = mo.ui.run_button(label="Measure throughput")
    measure_button
    return (measure_button,)


@app.cell
def _(device, measure_button, model, prompt_ids):
    if measure_button.value:
        throughput_rows = []
        with mo.status.spinner(title=f"Measuring on {device}"):
            for max_new_tokens in MAX_NEW_TOKENS_LIST:
                uncached_speed = measure_generation_throughput(model, prompt_ids, max_new_tokens, use_cache=False)
                cached_speed = measure_generation_throughput(model, prompt_ids, max_new_tokens, use_cache=True)
                throughput_rows.append(
                    {
                        "max_new_tokens": max_new_tokens,
                        "uncached_tokens_per_second": uncached_speed,
                        "cached_tokens_per_second": cached_speed,
                    }
                )
        throughput_path = COMPONENT_DIR / "runs" / f"throughput_{device.type}.json"
        throughput_path.write_text(json.dumps(throughput_rows, indent=2))
    measured = measure_button.value  # the table below re-reads the files once the measurement is done
    return (measured,)


@app.cell
def _(measured):
    throughput_files = sorted((COMPONENT_DIR / "runs").glob("throughput_*.json"))
    mo.stop(not throughput_files, mo.md("No measurements yet. Press **Measure throughput**."))

    table_lines = [
        "| device | new tokens | uncached, tokens/s | cached, tokens/s | speedup |",
        "|---|---|---|---|---|",
    ]
    for throughput_file in throughput_files:
        device_name = throughput_file.stem.removeprefix("throughput_")
        for row in json.loads(throughput_file.read_text()):
            speedup = row["cached_tokens_per_second"] / row["uncached_tokens_per_second"]
            table_lines.append(
                f"| {device_name} | {row['max_new_tokens']} | {row['uncached_tokens_per_second']:.0f} | "
                f"{row['cached_tokens_per_second']:.0f} | {speedup:.1f}x |"
            )
    mo.md("\n".join(table_lines))
    return


if __name__ == "__main__":
    app.run()

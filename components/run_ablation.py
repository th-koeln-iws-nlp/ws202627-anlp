import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")

with app.setup:
    import importlib.util
    import json
    import math
    from pathlib import Path

    import marimo as mo
    import plotly.graph_objects as go

    import ablation_harness
    import baseline

    COMPONENTS_DIR = Path(__file__).resolve().parent
    SEED_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # seeds 0, 1, 2

    # The quick check: a short run of one seed, to catch broken implementations before the real runs.
    QUICK_STEPS = 200
    QUICK_BASELINE_LABEL = "quick_baseline"
    QUICK_TAIL = 20  # the end loss is the mean train loss of the last steps, a single step is noisy
    START_LOSS = math.log(50257)  # an untrained model guesses uniformly over the vocabulary
    START_LOSS_TOLERANCE = 0.5  # every correct variant starts within 0.1 of START_LOSS, an init bug is far off


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Ablation runner

    Trains one variant of one component under the protocol in
    `components/README.md`: the config, the data, the seeds and the
    number of steps are fixed, only the component changes. Pick a
    component and a variant.

    **Quick check** (about a minute): 200 steps, seed 0, your variant
    next to the baseline. Run it first, before the real runs, to catch
    an implementation that does not train.

    **Train seeds 0, 1, 2**: the real run, 5 to 7 minutes per seed.
    Train `baseline` first, its spread is your noise floor.

    Everything lands in `components/<component>/runs/`: the weights, one
    result file and one loss history per run. The notebook reloads your
    `starter.py` every time you press a button, so you can edit it while
    this notebook stays open.
    """)
    return


@app.cell
def _():
    component_names = sorted(path.parent.name for path in COMPONENTS_DIR.glob("0*/starter.py"))
    component = mo.ui.dropdown(component_names, value=component_names[0], label="Component")
    component
    return (component,)


@app.cell
def _():
    def load_starter(component_name):
        """Load components/<component_name>/starter.py fresh, so edits made after the notebook opened count."""
        spec = importlib.util.spec_from_file_location(
            f"starter_{component_name}", COMPONENTS_DIR / component_name / "starter.py"
        )
        starter_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(starter_module)
        return starter_module

    return (load_starter,)


@app.cell
def _(component, load_starter):
    starter = load_starter(component.value)
    return (starter,)


@app.cell
def _(starter):
    variant = mo.ui.dropdown(["baseline", *starter.VARIANTS], value="baseline", label="Variant")
    quick_button = mo.ui.run_button(label="Quick check, 200 steps")
    run_button = mo.ui.run_button(label="Train seeds 0, 1, 2")
    mo.hstack([variant, quick_button, run_button], justify="start")
    return quick_button, run_button, variant


@app.cell
def _(component, load_starter, run_button, variant):
    if run_button.value:
        fresh_starter = load_starter(component.value)
        build_model_fn = baseline.GPTModel if variant.value == "baseline" else fresh_starter.VARIANTS[variant.value]
        with mo.status.spinner(title=f"Training {component.value}, {variant.value}, seeds 0, 1, 2"):
            ablation_harness.run_seeds(build_model_fn, component.value, variant.value)
    runs_checked = run_button.value  # downstream cells re-read the run files once training is done
    return (runs_checked,)


@app.cell
def _(component, load_starter, quick_button, variant):
    if quick_button.value:
        quick_runs_dir = ablation_harness.run_dir(component.value)
        quick_baseline_done = (quick_runs_dir / f"{QUICK_BASELINE_LABEL}_seed0.json").exists()
        with mo.status.spinner(title=f"Quick check, {component.value}, {variant.value}, {QUICK_STEPS} steps"):
            # the baseline quick run is the same for every variant: reuse it, unless the baseline itself is selected
            if variant.value == "baseline" or not quick_baseline_done:
                ablation_harness.run_seeds(
                    baseline.GPTModel, component.value, QUICK_BASELINE_LABEL, seeds=[0], num_steps=QUICK_STEPS
                )
            if variant.value != "baseline":
                quick_starter = load_starter(component.value)
                ablation_harness.run_seeds(
                    quick_starter.VARIANTS[variant.value],
                    component.value,
                    f"quick_{variant.value}",
                    seeds=[0],
                    num_steps=QUICK_STEPS,
                )
    quick_checked = quick_button.value
    return (quick_checked,)


@app.function
def summarize_quick_check(histories):
    """Table rows and warnings for the quick check.

    histories maps a name to a run history (the list of per-step dicts of ablation_harness).
    Returns (rows, notes): one (name, start loss, end loss, final val loss) row per run, and a
    list of warnings, empty if nothing looks wrong.
    """
    rows = []
    notes = []
    for name, history in histories.items():
        train_losses = [row["train_loss"] for row in history]
        tail = train_losses[-QUICK_TAIL:]
        start_loss = train_losses[0]
        end_loss = sum(tail) / len(tail)
        final_val_loss = [row["val_loss"] for row in history if row["val_loss"] is not None][-1]
        rows.append((name, start_loss, end_loss, final_val_loss))

        if not all(math.isfinite(loss) for loss in train_losses):
            notes.append(f"**{name}: the loss became NaN or infinite.**")
        elif end_loss >= start_loss:
            notes.append(f"**{name}: the loss did not go down.**")
        if abs(start_loss - START_LOSS) > START_LOSS_TOLERANCE:
            notes.append(
                f"**{name}: the first loss is {start_loss:.2f}, an untrained model starts near "
                f"{START_LOSS:.2f}.** Check the initialisation."
            )
    return rows, notes


@app.cell
def _(component, quick_checked, variant):
    quick_dir = ablation_harness.run_dir(component.value)
    quick_baseline_path = quick_dir / f"{QUICK_BASELINE_LABEL}_seed0_history.json"
    quick_variant_path = quick_dir / f"quick_{variant.value}_seed0_history.json"
    quick_needed_paths = [quick_baseline_path] if variant.value == "baseline" else [quick_baseline_path, quick_variant_path]
    mo.stop(
        not all(path.exists() for path in quick_needed_paths),
        mo.md("No quick check yet. Press **Quick check**."),
    )

    quick_histories = {"baseline": json.loads(quick_baseline_path.read_text())}
    if variant.value != "baseline":
        quick_histories[variant.value] = json.loads(quick_variant_path.read_text())
    quick_rows, quick_notes = summarize_quick_check(quick_histories)

    quick_table = "| | first loss | last 20 steps, mean | final val loss |\n|---|---|---|---|\n" + "\n".join(
        f"| {name} | {start_loss:.3f} | {end_loss:.3f} | {final_val_loss:.3f} |"
        for name, start_loss, end_loss, final_val_loss in quick_rows
    )
    if quick_notes:
        quick_verdict = "\n\n".join(quick_notes)
    elif variant.value == "baseline":
        quick_verdict = (
            "The baseline starts near the untrained loss and goes down, the setup works. Pick a variant "
            "and press **Quick check** again to compare it against this curve."
        )
    else:
        quick_verdict = (
            "Nothing looks wrong: both start near the untrained loss and both go down. Your variant should "
            "now be in the same range as the baseline. Much worse than the baseline after 200 steps is a bug "
            "until proven otherwise."
        )

    quick_figure = go.Figure()
    for quick_index, (quick_name, quick_history) in enumerate(quick_histories.items()):
        quick_val_rows = [row for row in quick_history if row["val_loss"] is not None]
        quick_figure.add_scatter(
            x=[row["step"] for row in quick_history],
            y=[row["train_loss"] for row in quick_history],
            name=f"train, {quick_name}",
            mode="lines",
            line={"color": SEED_COLORS[quick_index], "width": 1},
        )
        quick_figure.add_scatter(
            x=[row["step"] for row in quick_val_rows],
            y=[row["val_loss"] for row in quick_val_rows],
            name=f"val, {quick_name}",
            mode="lines+markers",
            line={"color": SEED_COLORS[quick_index], "dash": "dash"},
        )
    quick_figure.update_layout(
        title=f"Quick check, {component.value}, seed 0, {QUICK_STEPS} steps",
        xaxis_title="step",
        yaxis_title="loss",
    )
    mo.vstack([mo.md(quick_table), mo.md(quick_verdict), quick_figure])
    return


@app.cell
def _(component, runs_checked, variant):
    baseline_losses = ablation_harness.load_val_losses(component.value, "baseline")
    variant_losses = ablation_harness.load_val_losses(component.value, variant.value)

    mo.stop(not baseline_losses, mo.md("No baseline runs yet. Train `baseline` first."))

    if variant.value == "baseline":
        noise_floor = max(baseline_losses) - min(baseline_losses)
        summary = mo.md(
            f"""
            Baseline validation losses, seeds 0 to {len(baseline_losses) - 1}:
            {", ".join(f"{loss:.4f}" for loss in baseline_losses)}.

            Noise floor (highest minus lowest): `{noise_floor:.4f}`.
            """
        )
    else:
        mo.stop(not variant_losses, mo.md(f"No runs of `{variant.value}` yet."))
        summary = ablation_harness.summarize(baseline_losses, variant_losses)
    summary
    return


@app.cell
def _(component, runs_checked, variant):
    history_paths = sorted(ablation_harness.run_dir(component.value).glob(f"{variant.value}_seed[0-9]_history.json"))
    mo.stop(not history_paths)

    loss_curves = go.Figure()
    for history_path in history_paths:
        seed = int(history_path.stem.split("_seed")[1].split("_")[0])
        history = json.loads(history_path.read_text())
        val_rows = [row for row in history if row["val_loss"] is not None]
        loss_curves.add_scatter(
            x=[row["step"] for row in history],
            y=[row["train_loss"] for row in history],
            name=f"train, seed {seed}",
            mode="lines",
            line={"color": SEED_COLORS[seed], "width": 1},
        )
        loss_curves.add_scatter(
            x=[row["step"] for row in val_rows],
            y=[row["val_loss"] for row in val_rows],
            name=f"val, seed {seed}",
            mode="lines+markers",
            line={"color": SEED_COLORS[seed], "dash": "dash"},
        )
    loss_curves.update_layout(
        title=f"{component.value}, {variant.value}",
        xaxis_title="step",
        yaxis_title="loss",
    )
    loss_curves
    return


if __name__ == "__main__":
    app.run()

# Setup Checklist

⚠️ **Do this before Thursday, 8 October**, then reply to the course email with your two screenshots and your GitHub username. That leaves a few days to fix anything that does not work, instead of fixing it on the first morning.

The repository is still under construction and will change over the next days. Run `git pull` and `uv sync` again the evening before the course starts.

## 1. Accounts

- [ ] A [GitHub](https://github.com) account. The team repositories are created on day 1, once the pairs are fixed, and I need your username for that
- [ ] A [Hugging Face](https://huggingface.co/join) account. The notebooks download the dataset from the Hub, and from day 2 you upload your own trained models there
- [ ] A [molab](https://molab.marimo.io) account. molab is marimo's cloud notebook service, and it is where we run the GPU work during the course

## 2. Hugging Face token

- [ ] Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens) and create a new token of type **Write**. Read would be enough for the smoke test, but you need write access on day 2 to push your checkpoints, so create the right one now
- [ ] Copy it somewhere safe. Treat it like a password: never paste it into a notebook cell and never commit it. molab notebooks are viewable by anyone who has the link

**On your own machine:** create a file called `.env` in the project root, next to `pyproject.toml`, containing one line:

```
HF_TOKEN=hf_your_token_here
```

The notebooks read it from there. `.env` is listed in `.gitignore`, so it is never committed.

**On molab:** add `HF_TOKEN` in the notebook's **Secrets** panel. molab stores it in a `.env` file that persists between sessions and is not copied when a notebook is shared or forked. See the [molab storage documentation](https://marimo.io/pages/molab/storage) for how secrets and files behave there.

## 3. Software on your own machine

- [ ] Install [uv](https://docs.astral.sh/uv/), our package and environment manager
- [ ] Clone this repository
- [ ] Run `uv sync` in the project folder (packages are managed in [pyproject.toml](./pyproject.toml))

## 4. Smoke test, twice

Run the smoke test once on your own machine and once on molab, and send me a screenshot of the result card at the bottom of each.

**On your own machine:**

```bash
uv run marimo edit notebooks/00_smoke_test.py
```

**On molab:** open it on molab with GPU activated, so it runs on a server rather than as a preview, add your token in the Secrets panel, and run all cells.

The notebook detects whether you have a GPU and scales itself down accordingly, so running without one is a normal pass, not a problem. It checks your environment, that your Hugging Face token works, that the TinyStories dataset downloads and decodes correctly, and runs a tiny training step to prove your device computes. Nothing else needs to be downloaded separately: the notebooks pull the dataset straight from the Hub.

## 5. GPU access

GPUs for the course are provided through molab, with a university GPU cluster as a backup. If your molab session has no GPU before the course, that is fine: the smoke test passes in CPU mode. Access details follow on day 1.


## 6. Pre-reading

All textbooks are available online through the TH Köln library.

- [ ] Read [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch), Sebastian Raschka, chapter 1. Read it **without coding**, the code comes in class
- [ ] Watch [Visualizing Attention](https://www.3blue1brown.com/lessons/attention), 3Blue1Brown, a visual introduction to attention
- [ ] Optional: chapter 2 of the same book, if you want more before day 1
- [ ] Optional: skim [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/), Jay Alammar

More background reading, papers and tools are in the [README](./README.md#literature--links).
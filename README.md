# Advanced NLP Winter Term 2026/27

Welcome to the Advanced NLP block course. Three days, one model: we build GPT-2 from scratch, train it, align it, and then take it apart again to see how it became a modern LLM.

A general overview of the repository:
- [components](./components/): one folder per project component, each with a task description and a starter file
- [data](./data/): TinyStories subsets and the instruction data
- [notebooks](./notebooks/): marimo notebooks used in class, one per block
- [slides](./slides/): lecture slides and the scoring rubric
- [src](./src/): the GPT implementation we build together

## Schedule

Campus Claudiusstraße. 9:30 to 17:00 each day, with breaks at 11:00, 12:15 (lunch) and 15:05.

| **#** | **Date** | **Weekday** | **Time** | **Topic** | **Contents** | **Slides** |
|-------|----------|-------------|----------|-----------|--------------|-----------|
| 1 | 13.10. | Tuesday | 09:30–11:00 | Kick-Off & Text Data | Course overview, the three training stages, pair setup. Why subword tokenization, BPE in principle, `tiktoken`, sliding-window data loader | [PDF](./slides/01_kick_off.pdf) |
| 2 | 13.10. | Tuesday | 11:15–12:15 | Attention I | From the seq2seq bottleneck to Bahdanau attention to self-attention. Queries, keys, values, and the scaled dot product | [PDF](./slides/02_attention.pdf) |
| 3 | 13.10. | Tuesday | 13:35–15:05 | Attention II & the Transformer Block | Causal masking, multi-head attention. Positional embeddings, LayerNorm, GELU, the feed-forward expansion, residual connections, pre-LN vs post-LN | [PDF](./slides/03_transformer.pdf) |
| 4 | 13.10. | Tuesday | 15:30–17:00 | Weights & Generation | Assembling the full GPT model and loading the original GPT-2 weights into it. Greedy decoding, temperature, top-k, top-p. Building the marimo chat interface | [PDF](./slides/04_generation.pdf) |
| 5 | 14.10. | Wednesday | 09:30–11:00 | Pretraining | Cross-entropy and perplexity, train/val loss, AdamW, warmup and cosine schedules. Pretraining our model on TinyStories | [PDF](./slides/05_pretraining.pdf) |
| 6 | 14.10. | Wednesday | 11:15–12:15 | The Story of LLMs | GPT-1 to GPT-3, in-context learning, scaling laws and Chinchilla, InstructGPT, the open-weights turn, architecture drift, reasoning models. Told as changes to the model we just built | [PDF](./slides/06_llm_story.pdf) |
| 7 | 14.10. | Wednesday | 13:35–15:05 | Post-Training | Supervised fine-tuning: same loss, different data, plus the loss mask. LoRA and QLoRA. Reward models, PPO, and DPO | [PDF](./slides/07_post_training.pdf) |
| 8 | 14.10. | Wednesday | 15:30–17:00 | Evaluating LLMs | Perplexity vs benchmarks vs LLM-as-judge. Contamination, leaderboards, Goodhart's Law. Project kick-off and component assignment | [PDF](./slides/08_evaluation.pdf) |
| 9 | 15.10. | Thursday | 09:30–12:15 | 🔧 **Hackathon** | Pair work on the assigned component. Mandatory checkpoint with each pair | |
| 10 | 15.10. | Thursday | 13:35–17:00 | 🎤 **Presentations** | Pair presentations and questions | |

## Initial Set-Up

⚠️ **Do this before 13 October.** Day 1 starts with attention, not with installation.

- Packages are managed in the [pyproject.toml](./pyproject.toml)
- Install `uv` on your machine. Then run

```bash
uv sync
```

- Run the smoke test notebook and submit the screenshot as described in the pre-work issue:

```bash
uv run marimo edit notebooks/00_smoke_test.py
```

- It checks that PyTorch sees a GPU, that `tiktoken` and the GPT-2 weights are cached, and that the TinyStories subset loads.

### GPU Access

Still being clarified. We expect to work on molab, which runs Blackwell-generation NVIDIA cards. Until that is confirmed, the `pyproject.toml` may still change, so re-run `uv sync` the evening before the course starts. Details will follow in the pre-work issue.

### Pre-Work

- Read Raschka chapters 1 and 2. Read them **without coding**, the code comes in class.
- Skim the Illustrated Transformer.
- Both are linked below.

## Project

On day 2 you are assigned **one architectural component** that modern LLMs changed relative to GPT-2. On day 3 you implement it in your own day-1 code, run a controlled ablation against the unmodified baseline, and present the result.

| **#** | **Component** | **Found in** |
|-------|---------------|--------------|
| 1 | RMSNorm and norm placement | Llama, Gemma |
| 2 | SwiGLU feed-forward | Llama, Mistral |
| 3 | Sliding-window attention | Mistral |
| 4 | QK-Norm | Qwen3, Gemma 3 |
| 5 | Sinusoidal vs learned positional embeddings | original Transformer |
| 6 | Rotary positional embeddings (RoPE) | nearly everything |
| 7 | Grouped-query attention (GQA) | Llama 3, Qwen, Mistral |
| 8 | KV cache | all inference stacks |

### What is in the components folder

Every component has its own folder in [components](./components/) containing two things:

1. **`TASK.md`**, a one-page description: what the component is, which part of our GPT it replaces, what exactly you are expected to implement, and which measurement you should report.
2. **A starter Python file** with the class or function already named, the arguments already defined, and the body left empty for you to fill in. Next to it is a small test that feeds a dummy batch through your code and checks that the tensor shapes come out right.

Run the test first. It does not tell you whether your component is *correct*, only whether it is *wired up correctly*, but that alone saves you an afternoon of shape errors.

```bash
uv run pytest components/06_rope/
```

### Rules

- Rank your top three components by the end of day 2. Assignment happens overnight.
- The training script, the data and the random seed are fixed. You change **one thing**. That is the entire point.
- Budget for two or three training runs, not twenty. This is an ablation, not a hyperparameter search.
- A negative result, honestly explained, scores as well as a positive one. "Our loss got worse and here is why we think that happened" is a finding. Faking a win is not.

## Exam

### Presentation, 15 October, afternoon (60%)

- 12 minutes presentation per pair, both partners speak
- 6 minutes questions, directed at one named partner at a time
- Cover: what the component does, why it was introduced, how you implemented it, what your ablation showed, and what you would do with more time
- Your notebooks and training logs must be up and running during the questions

### Technical Report, due 21 October, 23:59 (40%)

- Maximum 4 pages plus the repository
- Contains: method, results with the loss curves, discussion, and a section on what did **not** work
- Contains an **AI-use appendix**: which tools you used, for what, and what you changed afterwards. This is expected and does not cost you points. Omitting it does.

[Scoring Rubric](./slides/rubric_anlp.pdf)

## Assignments

- Submit your work on the main branch
- Set a new upstream to fetch updates on our main repo
    - `git remote add upstream https://github.com/th-koeln-iws-nlp/anlp-ws2026.git`
    - `git pull upstream main`
- Work in pairs from day 1. One machine drives, the other navigates, and you swap at every break.
- If you have questions, create an issue in your repo and tag me `@RichardSiegTH`

## Dataset

We work with [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) throughout the whole course.

- Synthetic short stories written by GPT-3.5 and GPT-4, restricted to the vocabulary of a three- or four-year-old
- The point of the dataset: models with only a few million parameters still produce fluent, coherent, multi-paragraph English. Data quality beats model size
- [TinyStoriesInstruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct) is the same corpus reformatted as instruction and response, which we use for supervised fine-tuning
- For preference tuning we build our own dataset by sampling two completions from our fine-tuned model and ranking them

A pre-subsetted version is in the [data folder](./data/). Do not download the full 2.66 GB on the day.

## Literature & Links

### Main Textbooks

- [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch), Sebastian Raschka, Manning 2024. Our main text for days 1 and 2. Chapters 2 to 5 and 7 are the backbone of this course
- [LLMs-from-scratch repository](https://github.com/rasbt/LLMs-from-scratch), the code for the above, plus bonus material on KV caching, modern architecture conversions, and pretraining at larger scale
- [Hands-On Large Language Models](https://www.oreilly.com/library/view/hands-on-large-language/9781098150952/), Jay Alammar and Maarten Grootendorst, O'Reilly 2024. Chapter 3 for the visual explanation of the Transformer, chapter 12 for LoRA, reward models and DPO
- [a smol course](https://huggingface.co/learn/smol-course/unit1/1), Hugging Face. Instruction tuning, preference alignment and evaluation with TRL. Our reference for the tooling side

### Foundational Papers

- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473), Bahdanau et al. 2014. Attention, before the Transformer existed
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Vaswani et al. 2017
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), the GPT-2 paper
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165), GPT-3 and in-context learning
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556), the Chinchilla scaling laws
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155), InstructGPT and the RLHF pipeline
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290), Rafailov et al. 2023
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), Hu et al. 2021
- [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759), Eldan and Li 2023, the paper behind our dataset

### Component Papers

One per project component. Start here, not with a blog post.

| **Component** | **Paper** |
|---------------|-----------|
| RMSNorm | [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) |
| Pre-LN vs post-LN | [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) |
| SwiGLU | [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) |
| Sliding-window attention | [Mistral 7B](https://arxiv.org/abs/2310.06825) |
| QK-Norm | [Scaling Vision Transformers to 22 Billion Parameters](https://arxiv.org/abs/2302.05442) |
| Positional embeddings | [Attention Is All You Need](https://arxiv.org/abs/1706.03762), section 3.5 |
| RoPE | [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) |
| GQA | [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) |
| KV cache | [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) |

### Explanations and Visualizations

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/), Jay Alammar. The best visual introduction there is. Read this before day 1
- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY), Andrej Karpathy
- [Building LLMs from the Ground Up](https://www.youtube.com/watch?v=quh7z1q7-uc), Sebastian Raschka's three-hour workshop, essentially a compressed version of this course
- [nanoGPT](https://github.com/karpathy/nanoGPT), Karpathy's minimal GPT training repository
- [Ahead of AI](https://magazine.sebastianraschka.com/), Raschka's newsletter, the best ongoing source on architecture comparisons

### Tooling

- [uv](https://docs.astral.sh/uv/), our package and environment manager
- [marimo](https://docs.marimo.io/), our notebook format. See [mo.ui.chat](https://docs.marimo.io/api/inputs/chat/) for the chat interface we build on day 1
- [TRL](https://huggingface.co/docs/trl/index), Hugging Face's library for SFT, DPO and friends
- [PEFT](https://huggingface.co/docs/peft/index), parameter-efficient fine-tuning, including LoRA
- [tiktoken](https://github.com/openai/tiktoken), OpenAI's BPE tokenizer

### Further Reading

- [AI Engineering: Building Applications with Foundation Models](https://www.oreilly.com/library/view/ai-engineering/9781098166298/), Chip Huyen, O'Reilly 2025. For everything that happens after the model is trained
- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/), Jurafsky and Martin, free online
- [CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/), Stanford's semester-long version of what we do in three days

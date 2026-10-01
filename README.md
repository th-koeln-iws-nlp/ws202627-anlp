# Advanced NLP Winter Term 2026/27

Welcome to the Advanced NLP block course. Three days, one model: we build GPT-2 from scratch, train it, align it, and then take it apart again to see how it became a modern LLM.

A general overview of the repository:
- [components](./components/): one folder per project component, each with a task description and a starter file
- [notebooks](./notebooks/): marimo notebooks used in class
- [slides](./slides/): lecture slides and the scoring rubric

## Schedule

Campus Claudiusstraße. 9:30 to 17:00 each day, with breaks at 11:00, 12:15 (lunch) and 15:05.

| **#** | **Date** | **Weekday** | **Time** | **Topic** | **Contents** | **Slides** |
|-------|----------|-------------|----------|-----------|--------------|-----------|
| 1 | 13.10. | Tuesday | 09:30–11:00 | Kick-Off & Text Data | Course overview, the three training stages, pair setup. Why subword tokenization, BPE in principle, `tiktoken`, sliding-window data loader | [PDF](./slides/01_kick_off.pdf) |
| 2 | 13.10. | Tuesday | 11:15–12:15 | Attention I | From the seq2seq bottleneck to Bahdanau attention to self-attention. Queries, keys, values, and the scaled dot product | [PDF](./slides/02_attention.pdf) |
| 3 | 13.10. | Tuesday | 13:30–15:05 | Attention II & the Transformer Block | Causal masking, multi-head attention. Positional embeddings, LayerNorm, GELU, the feed-forward expansion, residual connections, pre-LN vs post-LN | [PDF](./slides/03_transformer.pdf) |
| 4 | 13.10. | Tuesday | 15:30–17:00 | Weights & Generation | Assembling the full GPT model and loading the original GPT-2 weights into it. Greedy decoding, temperature, top-k, top-p. Building the marimo chat interface | [PDF](./slides/04_generation.pdf) |
| 5 | 14.10. | Wednesday | 09:30–11:00 | Pretraining | Cross-entropy and perplexity, train/val loss, AdamW, warmup and cosine schedules. Pretraining our model on TinyStories | [PDF](./slides/05_pretraining.pdf) |
| 6 | 14.10. | Wednesday | 11:15–12:15 | The Story of LLMs | GPT-1 to GPT-3, in-context learning, scaling laws and Chinchilla, InstructGPT, the open-weights turn, architecture drift, reasoning models. Told as changes to the model we just built | [PDF](./slides/06_llm_story.pdf) |
| 7 | 14.10. | Wednesday | 13:30–15:05 | Post-Training | Supervised fine-tuning: same loss, different data, plus the loss mask. LoRA and QLoRA. Reward models, PPO, and DPO | [PDF](./slides/07_post_training.pdf) |
| 8 | 14.10. | Wednesday | 15:30–17:00 | Evaluating LLMs | Perplexity vs benchmarks vs LLM-as-judge. Contamination, leaderboards, Goodhart's Law. Project kick-off and component assignment | [PDF](./slides/08_evaluation.pdf) |
| 9 | 15.10. | Thursday | 09:30–12:15 | **Hackathon** | Pair work on the assigned component. Mandatory checkpoint with each pair | |
| 10 | 15.10. | Thursday | 13:30–17:00 | **Presentations** | Pair presentations and questions | |

## Initial Set-Up

⚠️ **Do this before 13 October.** See the [Setup Checklist](./SETUP.md): installing `uv`, running the smoke test, GPU access, and the pre-reading.

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

- The components are distributed to the pairs at the end of day 2.
- The training script and the data are fixed. Change one component, nothing else. See below for what that means exactly.
- Budget for two or three training runs per configuration, not twenty. This is an ablation, not a hyperparameter search.
- A negative or inconclusive result, honestly explained, scores as well as a positive one. "Our loss got worse and here is why we think that happened" is a finding. So is "the difference was smaller than our noise floor, so we cannot say". Faking a win is not.

### How to run the ablation

The point of day 3 is not to find out whether your component wins. It is to run a comparison that would let you tell.

**1. Measure your noise floor first.**

Run the unmodified baseline three times, changing nothing but the random seed. You will get three different validation losses, because training is stochastic: the batch order changes, the initialisation changes, the result moves.

The spread between your highest and lowest baseline run is your **noise floor**. It is how much your number moves for no reason at all. Write it down before you touch your component. On the reference setup it is 0.022, from validation losses of 1.8733, 1.8906 and 1.8685 at seeds 0, 1 and 2.

Training is deterministic for a given seed, so your three baseline runs should reproduce those numbers exactly. If they do not, something in your setup differs from the protocol. Find it before you run your component.

**2. Then run your component at the same three seeds.**

Compare the mean of your three component runs against the mean of your three baseline runs.

If the difference is **smaller than your noise floor**, you have not measured an effect. Say so. That is a complete and correct answer, and it scores as well as a positive result.

**3. What "change one thing" means.**

One *component*, plus everything that component requires. Implementing RoPE means deleting the learned positional embeddings as well as adding the rotation: that is still one change, because you cannot have one without the other.

What must stay identical between your baseline runs and your component runs:

- the dataset and the number of training steps
- the learning rate, schedule, batch size and context length
- the seeds
- everything in the model except your component

What is **not** allowed: tuning the learning rate for your component and not for the baseline, training your component for longer, or implementing two components and reporting the combination.

Some components change the parameter count. SwiGLU and GQA both do. That is fine and expected. Report the parameter count for both configurations and mention it in your discussion, since a model with more parameters has an advantage that has nothing to do with your component.

## Exam

The module grade comes from two parts, both assessed against the same rubric:

| Part | When | Weight |
|------|------|--------|
| Presentation and questions | 15 October, afternoon | 60% |
| Technical report and repository | due 21 October, 23:59 | 40% |

There is no written exam and no separate oral exam. Everything is assessed from the work you do in this course.

### How the grade is calculated

Eight criteria, each scored from 0 to 6 points, each carrying a fixed weight. The weighted average is converted to a German grade. Four criteria are scored for you individually and four for the pair, which means **33% of your grade is yours alone**.

The full form, with the description of every band, is published before the course starts: [Scoring Rubric](./slides/rubric_anlp.xlsx). 

**The result of your ablation is not graded.** You do not get a better mark because your component happened to improve the loss, and you do not lose marks because it did not. What is graded is whether the experiment was sound and whether you can say what your numbers do and do not show.

### Part 1: Presentation, 15 October

Per pair: **12 minutes presentation, then 6 minutes of questions.** Both partners speak. Slots are assigned in the morning.

A structure that works, though you are free to deviate:

| Minutes | Content |
|---|---|
| ~2 | What your component is and what problem it was introduced to solve |
| ~3 | How you implemented it: what changed in the GPT-2 you built on day 1 |
| ~4 | Experimental setup and results: baseline, seeds, what you held fixed, what you measured |
| ~3 | What you conclude, what you cannot conclude, and what you would do with another week |

**Have open and ready during the questions:** your notebooks, your training logs, and your loss curves. You will be asked to show things.

**Questions go to one partner at a time, and both of you will be asked.** Each of you is expected to be across the whole project, not only the half you wrote. Expect questions like: why does this component exist, what did it replace, show me the line where it changes the forward pass, what happens if you remove it, how do you know that difference is not noise, what would you expect at 7B parameters.

### Part 2: Report and repository, due 21 October, 23:59

**Maximum four pages** of report, excluding the AI-use appendix and references. PDF, submitted in your repository.

Required sections:

1. **The component.** What it is, where it appears in modern models, what it replaced.
2. **Implementation.** What you changed in the baseline and why you did it that way. Code goes in the repository, not the report.
3. **Experimental setup.** Baseline, seeds, step count, configuration. Enough that someone could repeat your run exactly.
4. **Results.** Loss curves and per-seed numbers, not just a final figure. Include your baseline spread.
5. **Discussion and limitations.** What your numbers support, what they do not, and what the scale of the experiment means for how far your conclusion travels.
6. **What did not work.** What you tried, what happened, what you concluded. This section carries the single highest weight in the report, and it cannot be written honestly after the fact, so keep notes as you go.

**AI-use appendix.** Required, not counted against the page limit, and it does not cost you points. Omitting it does. State which tools you used, for what, and what you changed afterwards. A sentence per item is enough:

> Claude Code: generated the first version of the RoPE rotation function. Rewrote the head-dimension handling, which was wrong for an odd number of heads. Wrote the plotting cells unchanged.

**Repository requirements.** Your component in the folder the starter file came from, the shape test passing, your training configuration and seeds recorded next to your results, and a commit history showing both partners.

## Assignments

- Submit your work on the main branch
- Set a new upstream to fetch updates on our main repo
    - `git remote add upstream https://github.com/th-koeln-iws-nlp/ws202627-anlp.git`
    - `git pull upstream main`
- Work in pairs from day 1.
- If you have questions, create an issue in your repo and tag me `@RichardSiegTH`

## Dataset

We work with [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) throughout the whole course.

- Synthetic short stories written by GPT-3.5 and GPT-4, restricted to the vocabulary of a three- or four-year-old
- The point of the dataset: models with only a few million parameters still produce fluent, coherent, multi-paragraph English. Data quality beats model size
- [TinyStoriesInstruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct) is the same corpus reformatted as instruction and response, which we use for supervised fine-tuning
- For preference tuning we build our own dataset by sampling two completions from our fine-tuned model and ranking them

We do not use the raw files. Pre-tokenized versions live on the Hugging Face Hub and are pulled in one line from the notebooks:

- [anlp-tinystories-gpt2](https://huggingface.co/datasets/Richard-Sieg-TH-Koln/anlp-tinystories-gpt2), for pretraining
- [anlp-tinystories-instruct-gpt2](https://huggingface.co/datasets/Richard-Sieg-TH-Koln/anlp-tinystories-instruct-gpt2), for supervised fine-tuning

## Literature & Links
 
### Main Textbooks
 
- [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch), Sebastian Raschka, Manning 2024. Our main text for days 1 and 2. Chapters 2 to 5 and 7 are the backbone of this course
- [LLMs-from-scratch repository](https://github.com/rasbt/LLMs-from-scratch), the code for the above, plus bonus material on KV caching, modern architecture conversions, and pretraining at larger scale
- [Hands-On Large Language Models](https://www.oreilly.com/library/view/hands-on-large-language/9781098150952/), Jay Alammar and Maarten Grootendorst, O'Reilly 2024. Chapter 3 for the visual explanation of the Transformer, chapter 12 for LoRA, reward models and DPO
- [a smol course](https://huggingface.co/learn/smol-course/unit1/1), Hugging Face. Instruction tuning, preference alignment and evaluation with TRL. Our reference for the tooling side
- [The RLHF Book](https://rlhfbook.com/), the free book-length treatment of reinforcement learning from human feedback. Our main source for block 07, because Raschka does not cover RLHF and Alammar gives it one chapter
- [The Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook), Hugging Face. The account of training SmolLM3, written as an operations log rather than a success story: the dead ends, the restart after one trillion tokens, and how ablations are actually run. The ablation chapter is required reading before day 3

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
 
One per project component. 
 
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
- [Visualizing Attention](https://www.3blue1brown.com/lessons/attention), 3Blue1Brown. Watch it alongside the Illustrated Transformer, it explains the same thing from the other direction
- [LLM Visualization](https://bbycroft.net/llm), an animated walk through every tensor operation in GPT inference. Open it next to your own code on day 1
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/), a GPT-2 running live in the browser with its attention weights on screen
- [MicroGPT Visualized](https://microgpt.jtauber.com/), the same idea at the smallest possible scale
- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY), Andrej Karpathy
- [Building LLMs from the Ground Up](https://www.youtube.com/watch?v=quh7z1q7-uc), Sebastian Raschka's three-hour workshop, essentially a compressed version of this course
- [nanoGPT](https://github.com/karpathy/nanoGPT), Karpathy's minimal GPT training repository
- [llama3-from-scratch](https://github.com/naklecha/llama3-from-scratch), Llama 3 implemented one tensor at a time. The fastest way in for the RoPE, GQA, RMSNorm and SwiGLU teams

### Ahead of AI
 
Sebastian Raschka's newsletter is the single most useful ongoing source for this course, and it is worth browsing beyond the three posts below. He writes up new open-weight architectures as they appear, which is exactly the question your project asks.
 
- [Ahead of AI](https://magazine.sebastianraschka.com/), the newsletter itself
- [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison), a walk through what current models changed and why. Read this before you start the project
- [From GPT-2 to gpt-oss](https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the), the same question this project asks, answered for one model family
- [A Visual Guide to Attention Variants in Modern LLMs](https://magazine.sebastianraschka.com/p/visual-attention-variants), for the attention components

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
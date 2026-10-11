"""The unmodified GPT every component changes one thing in.

Always loaded from solutions/, never from notebooks/: the ablation needs
a guaranteed-correct baseline, not whatever state your own day 1-2
implementation happens to be in.

The solution files are named like 04_gpt_solution.py, which a plain
`import` statement cannot spell, and they import each other by that bare
name. Putting solutions/ on sys.path and using importlib.import_module
handles both.
"""

import importlib
import sys
from pathlib import Path

SOLUTIONS_DIR = Path(__file__).resolve().parent.parent / "solutions"
if str(SOLUTIONS_DIR) not in sys.path:
    sys.path.insert(0, str(SOLUTIONS_DIR))

MultiHeadAttention = importlib.import_module("02_attention_solution").MultiHeadAttention

_transformer = importlib.import_module("03_transformer_solution")
LayerNorm = _transformer.LayerNorm
FeedForward = _transformer.FeedForward
TransformerBlock = _transformer.TransformerBlock

GPTModel = importlib.import_module("04_gpt_solution").GPTModel

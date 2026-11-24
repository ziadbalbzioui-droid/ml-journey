# 05-explore/

Exploration-first notebooks, written 2026-11-24 after feedback on `04-practice/`: those were too condensed. Most of
the work was filling in formulas, with little room to explore or to learn how ML is done in practice. This series
is lighter on algebra and heavier on **engineering judgement**. You work on real datasets, look at the data first,
beat a baseline, sanity-check before training, read loss curves, look at the mistakes, and run open-ended experiments.

## How each notebook works

| | |
|---|---|
| 📖 | short explanation of the idea |
| 🔮 | predict before running, with a confidence tag `[1]` guessing / `[2]` fairly sure / `[3]` certain |
| ✍️ | your code: usually 1–5 lines (`???`), more in 04–05 where you write whole classes |
| 🧪 | open experiment: no check, no single answer; write one line about what you saw |
| 🛠️ | an engineer's habit to keep on every project |

- Hints are folded in `▶ Hint` blocks. Open them after ~10 min stuck.
- `check` cells print ✅ / ❌. There are fewer of them than in `04-practice/`: the 🧪 cells are where you learn.
- Reference solutions, run with outputs, are in [`solutions/`](solutions/). Open one after your own attempt, not before.
- Data downloads automatically on first use into `data/` (via `mlj.datasets`). Everything runs on CPU, faster on MPS.

Run from this folder: `uv run jupyter lab`.

## The series

| # | Notebook | Dataset | You learn | Time |
|---|---|---|---|---|
| 01 | [Logistic regression on clinical data](01_logreg_diabetes.ipynb) | Pima diabetes (768 patients) | zeros-as-missing trap, split-then-preprocess, baselines, LR from scratch + gradient check, why scaling matters (by experiment), odds ratios and confounded coefficients, thresholds / sensitivity / PPV | 2–2.5 h |
| 02 | [Loss functions](02_loss_functions.ipynb) | Diabetes progression (442), Pima | MSE → mean, MAE → median, pinball → quantile; Huber vs corrupted labels; prediction intervals + coverage; BCE vs MSE saturation; what class weights really do; loss vs metric | 2 h |
| 03 | [Optimizers playground](03_optimizers_playground.ipynb) | 2-D surfaces, Fashion-MNIST | SGD / momentum / Adam written and raced; LR range test; reading loss curves; batch size; warmup + cosine | 2 h |
| 04 | [MLP from scratch](04_mlp_from_scratch.ipynb) | Fashion-MNIST (70k images) | Linear / ReLU / softmax-CE with hand backprop, gradient-checked; pre-flight checks; init measured layer by layer; confusion matrix and confident mistakes; PyTorch version; fighting overfitting | 3 h |
| 05 | [GPT from scratch that learns to add](05_transformer_addition.ipynb) | 3-digit addition as text (10⁶ pairs) | LayerNorm, multi-head causal attention, blocks, GPT; causality test; answer-only loss; greedy decoding; error analysis by carries; attention maps; ablations: reversed digits, no mask (leakage), no positions, memorisation vs generalisation | 3 h |

Order: 01 → 02 → 03 → 04 → 05. Each builds on the previous one (02 reuses 01's preprocessing through
`mlj.datasets.prepare_pima`, 05 reuses 03's schedule and 04's checks).

## Editing

The notebooks are generated from `_build/src/*.nbsrc` (one source → student + solution). After changing a source:
`uv run python notebooks/05-explore/_build/build.py --execute <name>`.

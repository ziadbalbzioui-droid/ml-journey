# Learning log

One entry per session, newest first. Written at the end of the session, before the commit.

**Still shaky** is the most valuable line: it is what the next Claude chat opens with, and
anything that survives two monthly reviews becomes the next month's first topic.

## Template

```markdown
## YYYY-MM-DD — <short title>
**Did:** what I built or read. Files touched.
**Understood:** the idea I could now explain to someone else, in my own words.
**Still shaky:** what I used without fully justifying it. Be honest — this is the point.
**Next:** the single next step, small enough to start immediately.
```

---

## 2026-09-29 — Repo, roadmap, and rules
**Did:** Turned `~/ML` into the `ml-journey` repo: existing work sorted into
`notebooks/01-handson-ml/` (Géron) and `notebooks/02-zero-to-hero/` (Karpathy), data moved
to a gitignored `data/` with `scripts/fetch_data.py` to restore it, `uv` environment with
`pyproject.toml`. Wrote `CLAUDE.md` (working rules), `ROADMAP.md` (Oct → Feb, ending on
biomedical ML), this log, and the concept-note format.
**Understood:** Not ML itself — the process. The output of a session is a working artefact
*plus* a written record of what I did not understand. Without the second half, the first
half decays.
**Still shaky:** Nothing tested yet. The roadmap assumes ~10 h/week; that assumption is
unverified and October will tell me whether it survives contact with the semester.
**Next:** Write the `Head` class in `nano-gpt/train.ipynb` — and before writing it, state
on paper what shape `q @ k.transpose(-2,-1)` produces and why it gets divided by √d_k.

---

## Before this repo

Reconstructed from file timestamps; the code predates the log.

## 2026-09-21 — nanoGPT: data pipeline and bigram baseline
**Did:** `nano-gpt/train.ipynb` — char encode/decode, 90/10 split, `get_batch` with random
offsets, `BigramLanguageModel` with token **and** position embeddings, `generate` loop,
AdamW training.
**Still shaky:** the position embeddings are in the model but do nothing useful without
attention — I built the scaffolding before understanding what it was for.

## 2026-08-30 — WaveNet in real PyTorch
**Did:** `makemore_torch.py` — the hand-rolled net rewritten with `nn.Embedding`,
`nn.Conv1d` (kernel 2, stride 2) replacing `FlattenConsecutive` + `Linear`,
`nn.BatchNorm1d`, AdamW + scheduler, `DataLoader`, MPS, grad clipping, checkpoints.
**Understood:** every PyTorch module maps onto something I had already written by hand.

## 2026-08-22 — Hyperparameter experiments
**Did:** `hparam_tuning.py` — LR range test, LR curves, capacity sweep, train/dev gap,
random search → five figures in `plots/`. Also `makemore_rnn.ipynb`: hierarchical
WaveNet-style net from my own `Linear`, `BatchNorm1D`, `Tanh`, `Embedding`,
`FlattenConsecutive`.

## 2026-08-19 — MLP character model
**Did:** `makemore_mpl.ipynb` — Bengio-style MLP over a 3-character context: embedding
table, hidden tanh layer, softmax, minibatching, train/dev/test split.

## 2026-08-17 — Bigram character model
**Did:** `makemore.ipynb` — counted bigrams into a 27×27 matrix and sampled from it, then
trained the same model by gradient descent and watched the two losses meet.

## 2026-08-16 — micrograd
**Did:** `micrograd.ipynb` — `Value` class with `+`, `*`, `**`, `tanh`, `exp`, topological
sort, `.backward()`; then `Neuron` / `Layer` / `MLP`. Backprop by hand, no torch.

## 2026-08-14 — First PyTorch tensors
**Did:** `scratch/tensor_test.ipynb` — torch / torchvision setup.

## 2026-06-02 — Kaggle House Prices
**Did:** `house-prices-kaggle.ipynb` — dropped columns >50% NaN, imputation, split,
first submission pipeline.

## 2026-03-13 — California housing
**Did:** `housing.ipynb` — EDA, histograms, feature inspection.

## 2026-03-10 — Classification
**Did:** `classification.ipynb` — MNIST with KNN + `GridSearchCV`; Titanic fetch and load.

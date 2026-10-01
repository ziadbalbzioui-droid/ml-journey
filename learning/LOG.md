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

## 2026-10-01 — Deep learning promoted to its own project
**Did:** Pulled the 2026-09-30 diagnostic rewrite from GitHub (`WORKFLOW.md`, 21-week plan)
into the local clone. Then rewrote `learning/ROADMAP.md`: added `P2 — Neural networks and
attention from scratch` (W5–W6, Core) — self-attention derived and gradient-checked in NumPy,
then finishing `nano-gpt/train.ipynb` into a trained char-GPT with ablations — funded by
cutting `P5` (high-dimensional omics), already the lowest-priority item. Renumbered
P2→P3 (clinical tabular), P3→P4 (sepsis), old P4→P5 (survival, still Stretch); P4's sequence
model week (W13) now reuses P2's attention module instead of re-deriving it. Updated
`chat-context.md` §4 to match (nanoGPT is active work, not parked).
**Understood:** The original month-by-month plan treated the transformer as the single
October deliverable; the diagnostic-driven rewrite had folded it down to one week inside the
sepsis project. Neither matched "deep learning as an important part of the plan" — it needed
its own project with its own exit check, not a subtask of something else.
**Still shaky:** Nothing technical yet — this was a planning session, no code or theory. The
two "light" weeks (now W14, W15) no longer land on Christmas/New Year; unverified whether that
still matches my actual exam calendar.
**Next:** W1 starts Monday 5 Oct — theory prep (normal equation, GD, logistic regression,
ridge, AUROC) in chat, then `src/mlj/scratch/linear.py` and `logistic.py` here.

---

## 2026-09-30 — Diagnostic test (A–D) and plan rewrite
**Did:** Closed-book diagnostic of 40 questions. Took sections A–D (Q1–22); Q18 and Q22 left
blank, the Q14 check (`z = [0,0,0]`, true class 1) unanswered. Sections E–H not taken yet:
scheduled for W2 (baseline), W8 and W21. Rewrote `learning/ROADMAP.md` as a 21-week,
project-driven plan (5 Oct 2026 → 28 Feb 2027): one foundations week, then seven projects
(XGBoost from scratch, rigorous tabular, sepsis time series, survival, omics, audit toolkit,
lab replication), and added
`learning/WORKFLOW.md` (Claude Code vs chat split, weekly loop, prompts). Corrected the level
description in `CLAUDE.md` and `chat-context.md`: beginner in classical ML.
**Understood:** Nothing verified yet. Explanations received and to be confirmed in exit checks:
ridge as a limit and the intercept (Q4), scaling per model (Q10), backprop cost argument
(Q16), softmax + cross-entropy gradient `p − y` (Q14), saturation (Q19).
**Still shaky:**
- Normal equation written as `X Xᵀ` instead of `XᵀX`; singular cases (Q6).
- Gradient-descent update written on the loss instead of the parameters (Q11).
- Binary cross-entropy formula not written; "MSE is only for regression" is imprecise (Q7).
- Decision-tree split criterion blank (Q8); KNN in high dimension (Q9).
- Feature scaling: said "yes" without separating KNN / ridge / tree (Q10).
- Softmax + CE gradient, backprop cost, initial loss of a uniform softmax, residual
  connections: blank (Q14, Q16, Q18, Q22).
- Vocabulary: saturation vs "dead neuron" (Q19); embedding table described as "coding logits"
  (Q21); BatchNorm train/eval explanation unfinished (Q20).
- Overfitting described only as "training too long" (Q1); why tuning on test breaks validity (Q2).
- Habit: no confidence tags until Q17; sentences left unfinished.
**Next:** Answer the open items (Q14 check, Q18, Q22) with confidence tags, then start W1:
linear regression from scratch. Before writing it, write the normal equation with shapes.

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

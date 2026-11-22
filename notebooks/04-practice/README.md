# 04-practice/

Hands-on exercises written for this repo (2026-11-22), after `03-ens-intro-ml/`. The ENS labs explained ML and
called libraries; these make me **write every line myself**, from plain Python up to a transformer, with each piece
checked against scikit-learn / PyTorch. Two goals at once: deep understanding for the Stanford lab (March 2027), and
the ML coding + ML fundamentals rounds of internship interviews (applications Jan–Feb 2027, start Sept 2027).

## How each notebook works

- Every function comes with its signature, a docstring and hints only. **The body is mine.**
- `✍️ Predict` cells come before experiments: write the guess and a confidence tag `[1]`/`[2]`/`[3]` *before* running.
- `check` cells print ✅ or a ❌ that says *what* is wrong, not how to fix it. No answer keys in the repo.
- No Claude / Copilot / autocomplete writing code. Claude Code only for the hint ladder, after 15 min stuck.
- Each notebook ends with a debrief to copy into `learning/LOG.md`, and most with a blank-file drill (rewrite the
  core function from memory, timed): this is the interview-readiness test.

Run from this folder: `uv run jupyter lab`. Everything uses synthetic or scikit-learn built-in data, no downloads,
CPU only (the longest cell, in 11, trains for about a minute).

## Curriculum, by increasing difficulty

| # | Notebook | Level | Time | Status |
|---|---|---|---|---|
| 00 | [Pure Python fluency](00_python_fluency.ipynb): comprehensions, recursion, classes, generators, a bigram LM, a patient-level split | 1 | 2–3 h | ✅ |
| 01 | [NumPy, linear / logistic / softmax regression](01_numpy_linear_logistic.ipynb): normal equation, GD stability threshold, ridge, Newton, gradient checking | 1 | 3 h or 1 h test | ✅ |
| 02 | [Optimizers](02_optimizers.ipynb): SGD, momentum, RMSProp, Adam, AdamW vs torch.optim; κ and convergence rates; batch size; warmup + cosine | 2 | 3 h | ✅ |
| 03 | Bias–variance and regularization: decomposition by simulation, lasso by coordinate descent | 2 | 3 h | planned |
| 04 | Probabilistic ML: MLE vs MAP, Bayesian linear regression with uncertainty, GMM by EM | 2 | 3–4 h | planned |
| 05 | PCA by SVD, kernel ridge regression, the curse of dimensionality | 2 | 3 h | planned |
| 06 | [Trees and boosting](06_trees_boosting.ipynb): vectorised split search, regression tree, Newton-leaf boosting matching sklearn | 2 | 4 h | ✅ |
| 07 | [Evaluation pitfalls](07_evaluation_pitfalls.ipynb): AUROC from ranks, PPV vs prevalence, row vs patient bootstrap, two leaks, calibration, prior shift | 3 | 3–4 h | ✅ |
| 08 | An autodiff engine on NumPy arrays (broadcasting-aware), training an MLP with it | 3 | 4 h | planned |
| 09 | Training deep nets: init by variance argument, BatchNorm / LayerNorm forward + backward, dropout, residuals | 3 | 4 h | planned |
| 10 | Convolution by im2col and a GRU, from scratch | 3 | 4 h | planned |
| 11 | [Attention and transformers](11_attention_transformer.ipynb): NumPy forward + backward, √d experiment, multi-head vs nn.MultiheadAttention, causal forecaster vs naive baselines, the mask as leakage control | 4 | 5 h | ✅ |
| 12 | Representation learning: VAE (ELBO, reparameterisation), contrastive learning on time series | 4 | 4–5 h | planned |
| 13 | Wearables end to end: subject-wise splits, GBDT vs 1D-CNN, event-level errors, interpretability | 5 | 5 h | planned |

## Suggested order (Nov 2026 → Jan 2027)

11 first while the ENS attention course is fresh, then 00 → 01 (as a test) → 02 → 07 → 06, then the rest in order.
Alongside: NeetCode 150 on LeetCode, one problem a day (the algorithms rounds are a separate skill these notebooks
do not cover), and Tomin's forecasting book.

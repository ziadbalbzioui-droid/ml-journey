# Roadmap — 5 Oct 2026 → 28 Feb 2027 (21 weeks)

Rewritten on 2026-09-30 after the diagnostic test (see `LOG.md`, entry of that date).

**Level, honestly:** strong maths (calculus, linear algebra), beginner in classical ML. I have
*followed* micrograd → makemore → nanoGPT, but the diagnostic showed I cannot yet write the
basic formulas from memory (normal equation, gradient-descent update, cross-entropy) and I
mix up some vocabulary (embeddings vs logits). So the plan starts from the foundations,
from scratch, and every week ends with a closed-book exit check.

**Goal:** arrive at the Stanford research position in March 2027 able to build, evaluate and
explain models on tabular and longitudinal data, including gradient boosting (XGBoost),
and defend the evaluation in front of a reviewer.

**Budget:** ~10 h/week. Minimum viable week = 6 h: theory prep + one build session + the exit
check. Split of a normal week and the exact prompts: [WORKFLOW.md](WORKFLOW.md).

**Tools:** *Claude Code* (in this repo) builds, runs, tests, reviews, commits and logs.
*Claude chat* (the "AI/ML preparation" Project, connected to this repo on GitHub) handles
theory, derivations, oral exams and paper reading.

## Overview

| Phase | Weeks | Theme | Main deliverable |
| --- | --- | --- | --- |
| 1 | W1–W4 (Oct) | Foundations from scratch | `src/mlj/scratch/` linear + logistic + ridge, `mlj/metrics.py` |
| 2 | W5–W7 (Nov) | Trees, boosting, **XGBoost** | from-scratch tree + GBM + XGBoost-lite, checked against the library |
| 3 | W8–W10 (Nov–Dec) | Attention and the transformer | gradient-checked attention in NumPy + char-GPT |
| 4 | W11–W16 (Dec–Jan) | Honest evaluation + a tabular project | `src/mlj/eval/` + `projects/tabular-clinical/` report |
| 5 | W17–W19 (Jan–Feb) | Time, survival, interpretability, clinical data | time-aware split, SHAP notes, one survival comparison |
| 6 | W20–W21 (Feb) | Lab-facing work | paper notes, final diagnostic, repo clean-up |

XGBoost is a three-week module (W5–W7) and is then reused as one of the models in W14–W18.

## Diagnostic gaps → where they get fixed

| Gap seen on 2026-09-30 | Fixed in |
| --- | --- |
| Normal equation, singular `XᵀX` (Q6) | W1 |
| Gradient-descent update on parameters (Q11) | W2 |
| Binary cross-entropy, softmax+CE gradient `p − y` (Q7, Q14) | W2 |
| Backprop vs numerical gradient cost (Q16) | W2 |
| Ridge as a limit, scaling per model (Q4, Q10) | W3 |
| Why tuning on test breaks validity, overfitting causes (Q1, Q2) | W3–W4 |
| Decision-tree splits, KNN in high dimension (Q8, Q9) | W5 |
| Initial loss of a uniform softmax (Q18) | W2 |
| Saturation vs "dead neuron", BatchNorm train/eval (Q19, Q20) | W9–W10 |
| Embedding table vs logits (Q21) | W8 |
| Residual connections (Q22) | W9 |
| Sections E–H never taken (evaluation, code, theory link, stretch) | first taken in W4, retaken W12, W21 |

## Every week

- **Mon** — chat: theory prep (3 questions that find the gap, then the derivation).
- **Mon–Tue** — the resource of the week.
- **Tue–Thu** — Claude Code: build session(s), with the prediction question before any code.
- **Fri** — chat: oral exit check, closed book.
- **Sat** — concept note in `learning/concepts/`, `LOG.md` entry, commit, push.

---

# Phase 1 — Foundations from scratch (October)

## W1 · 5–11 Oct · Linear regression
- **Theory (chat):** model, MSE, the normal equation `w = (XᵀX)⁻¹Xᵀy` with shapes `(n,d)`, projection view, when `XᵀX` is singular (collinear columns, `n < d`).
- **Build (Claude Code):** `src/mlj/scratch/linear.py` with `fit_normal_eq` and `fit_gd`. Check against scikit-learn `LinearRegression` on California housing (you already loaded it). Add a test that reproduces the singular case.
- **Resources:** ISLP ch. 3 (free PDF: https://www.statlearning.com).
- **Exit check:** write the normal equation with shapes from memory and explain the singular case (Q6).

## W2 · 12–18 Oct · Gradient descent and logistic regression
- **Theory (chat):** update `w ← w − α ∂L/∂w` on parameters, too-large vs too-small `α`, minibatch vs full batch; sigmoid; Bernoulli negative log-likelihood = binary cross-entropy; derive softmax + cross-entropy gradient `p − y`; why that makes a uniform-softmax initial loss `log C`.
- **Build:** `logistic.py` (binary + softmax regression, NumPy), `gradcheck.py` (numerical vs analytic gradient), a learning-rate sweep plot showing slow / good / diverging.
- **Resources:** Andrew Ng, *Machine Learning Specialization* course 1 (https://www.coursera.org/specializations/machine-learning-introduction); 3Blue1Brown neural-network series for intuition (https://www.3blue1brown.com/topics/neural-networks).
- **Exit check:** derive `p − y`, compute the `z = [0,0,0]` case by hand, write the BCE formula, explain why backprop beats finite differences in number of forward passes (Q7, Q11, Q14, Q16, Q18).

## W3 · 19–25 Oct · Regularisation and bias–variance
- **Theory (chat):** bias–variance, ridge closed form `(XᵀX + λI)⁻¹Xᵀy` and why it is invertible, lasso sparsity, intercept not penalised, why feature scaling matters for ridge and KNN but not trees.
- **Build:** ridge in NumPy, regularisation-path plot, learning curves (error vs training size) on housing; compare to `Ridge` / `Lasso`.
- **Resources:** ISLP ch. 5–6.
- **Exit check:** Q3, Q4 (including `λ → ∞` and the intercept), Q10 with a reason per model.

## W4 · 26 Oct–1 Nov · Evaluation basics + Phase 1 review
- **Theory (chat):** why three splits, optimism from tuning on test, k-fold CV, confusion matrix, precision/recall, AUROC as the probability a random positive outranks a random negative, PPV vs prevalence via Bayes.
- **Build:** `src/mlj/metrics.py` (confusion metrics, ROC/AUC from ranks), a from-scratch k-fold splitter, and a demo of the scaler-before-split leak (Q32).
- **Resources:** ISLP ch. 5; StatQuest videos on ROC/AUC and cross-validation (https://www.youtube.com/@statquest).
- **Exit check:** retake diagnostic sections A–D closed book (target ≥ 80 %), and take E–H for the first time as a baseline.

# Phase 2 — Trees, boosting and XGBoost (November)

## W5 · 2–8 Nov · Decision trees and random forests
- **Theory (chat):** impurity (Gini, entropy, variance), greedy split search, why deep trees overfit, scale invariance; bagging, the variance of an average of correlated estimators `ρσ² + (1−ρ)σ²/B`, feature subsampling, out-of-bag error; KNN and the curse of dimensionality.
- **Build:** `scratch/tree.py` (regression + classification, exact greedy split) and `scratch/forest.py`; match scikit-learn trees at equal depth.
- **Resources:** ISLP ch. 8; StatQuest "Decision Trees" and "Random Forests".
- **Exit check:** compute the best split by hand on 6 points; explain why feature subsampling lowers `ρ` (Q8, Q9).

## W6 · 9–15 Nov · Gradient boosting from scratch
- **Theory (chat):** boosting as gradient descent in function space: each tree fits the negative gradient (residuals for MSE, `p − y` for logistic loss), shrinkage, `n_estimators` vs learning rate.
- **Build:** `scratch/gbm.py` with squared then logistic loss on your own tree; train/validation curve vs number of trees; compare with scikit-learn `GradientBoosting`.
- **Resources:** StatQuest Gradient Boost series; https://explained.ai/gradient-boosting/; Friedman (2001), *Greedy function approximation* (skim).
- **Exit check:** predict, before running, the effect of learning rate 1.0 vs 0.1; explain "fit residuals" = "fit negative gradient".

## W7 · 16–22 Nov · XGBoost
- **Theory (chat):** the regularised objective `Ω = γT + ½λ‖w‖²`, second-order Taylor expansion with `gᵢ, hᵢ`, optimal leaf weight `w* = −G/(H+λ)`, split gain with `γ`, `min_child_weight`, sparsity-aware handling of missing values, column/row subsampling.
- **Build:** `scratch/xgb_lite.py` (your tree with `g, h, λ, γ`), checked against the `xgboost` library (`tree_method="exact"`, same parameters) on toy data. Then `notebooks/03-tabular/xgb_benchmark.ipynb`: logistic regression vs random forest vs XGBoost under the same CV on three public datasets, with early stopping and random-search tuning. Add the library with `uv add xgboost`.
- **Resources:** Chen & Guestrin (2016), https://arxiv.org/abs/1603.02754 (sections 2–3); the docs page "Introduction to Boosted Trees" (https://xgboost.readthedocs.io); StatQuest XGBoost series.
- **Exit check:** derive `w*` and the gain on paper; for an overfitting model, name three hyperparameters to change and the direction of each. Tag the repo `phase-2`.

# Phase 3 — Attention and the transformer (Nov–Dec)

Picks up `notebooks/02-zero-to-hero/nano-gpt/train.ipynb` exactly where it stopped.

## W8 · 23–29 Nov · Self-attention
- **Theory (chat):** embeddings (input vectors, row lookup in a `(V, d)` table) vs logits (output scores); `q @ kᵀ` shapes `(B, T, hs)`, why `/√hs`, what the causal `tril` mask enforces (applied before softmax, with `−inf`).
- **Build:** one attention head in **pure NumPy**, forward and backward, checked with `gradcheck.py`: `src/mlj/scratch/attention.py`. Then write `Head` in the notebook.
- **Resources:** Karpathy, *Let's build GPT* (https://www.youtube.com/watch?v=kCc8FmEb1nY); Vaswani et al. (https://arxiv.org/abs/1706.03762).
- **Exit check:** derive the shapes through one head and the variance of `q·k` with and without the scaling (Q21, Q37–39).

## W9 · 30 Nov–6 Dec · Multi-head, residuals, LayerNorm
- **Theory (chat):** why several small heads; residual connection gradient `∂y/∂x = I + ∂F/∂x`; LayerNorm vs the `BatchNorm1D` you wrote (per-sample vs per-batch, running statistics); tanh saturation vs dead ReLU.
- **Build:** `MultiHeadAttention`, `FeedForward`, `Block`, full GPT in the notebook.
- **Resources:** *The Annotated Transformer* (https://nlp.seas.harvard.edu/annotated-transformer/).
- **Exit check:** Q19, Q20, Q22 closed book.

## W10 · 7–13 Dec · Training craft and ablations
- **Theory (chat):** AdamW, warmup + cosine, initialisation scale, what the loss curve should look like.
- **Build:** train a char-GPT on tinyshakespeare until the text is readable. Ablations: no `/√d`, no residuals, no LayerNorm. **Write the prediction for each before running.**
- **Resources:** Karpathy, *A Recipe for Training Neural Networks* (https://karpathy.github.io/2019/04/25/recipe/); Zero to Hero index (https://karpathy.ai/zero-to-hero.html).
- **Exit check:** ablation table with predicted vs observed, and a short explanation of each mismatch. Write `concepts/self-attention.md`.

# Phase 4 — Honest evaluation and a tabular project (Dec–Jan)

## W11 · 14–20 Dec · Cross-validation done right + bootstrap
- **Theory (chat):** stratified, grouped and nested CV; why rows from the same patient or group leak (Q26); bootstrap percentile confidence intervals on a metric.
- **Build:** `src/mlj/eval/splits.py` and `bootstrap.py` with pytest tests; a synthetic demo where a row-level split looks good and a group-level split does not.
- **Resources:** scikit-learn docs, "Cross-validation: evaluating estimator performance"; Varoquaux, *Cross-validation failure: small sample sizes lead to large error bars* (search the title).
- **Exit check:** Q26, Q27, Q29; explain in three sentences why grouped CV is required.

## W12 · 21–27 Dec (light week, ~5 h) · Calibration and thresholds
- **Theory (chat):** calibration, reliability curve, Brier score, log loss as proper scoring rules; XGBoost probabilities are often miscalibrated; choosing a threshold from cost or PPV.
- **Build:** `src/mlj/eval/calibration.py` (reliability curve, Brier, ECE); calibrate an XGBoost model on a held-out fold and plot before/after.
- **Resources:** Van Calster et al. (2019), https://bmcmedicine.biomedcentral.com/articles/10.1186/s12916-019-1466-7; scikit-learn calibration docs.
- **Exit check:** retake sections E (Q23–29) closed book, target ≥ 80 %.

## W13 · 28 Dec–3 Jan (light week, ~5 h) · Project setup
- **Theory (chat):** missing data (MCAR / MAR / MNAR), the catalogue of leaks (row-level, temporal, post-outcome features), fixing the target and the prediction time before looking at any model.
- **Build:** `projects/tabular-clinical/`. Dataset: UCI *Diabetes 130-US hospitals* (30-day readmission), which has a patient id for grouped splits — Claude Code verifies the link and licence in this session. Write a data card, EDA, and a README stating question, target, unit of analysis and split protocol. **Lock the test set before any modelling.**
- **Resources:** Kaufman et al. (2012), *Leakage in Data Mining* (https://dl.acm.org/doi/10.1145/2382577.2382579).
- **Exit check:** ask Claude Code to review the README as a skeptical reviewer and list three leak risks; fix them.

## W14 · 4–10 Jan · Models and confidence intervals
- **Theory (chat):** what a fair baseline is (majority class, regularised logistic regression), tuning only inside the training folds.
- **Build:** logistic regression, random forest and XGBoost under grouped, nested CV with `mlj.eval`; results table with bootstrap CIs. Predict which model wins, and by how much, before running.
- **Resources:** XGBoost docs (parameters page, early stopping).
- **Exit check:** table with CIs committed, and the prediction-vs-result note.

## W15 · 11–17 Jan · Finish the project
- **Theory (chat):** reporting standards (TRIPOD+AI idea), model cards.
- **Build:** calibration, threshold analysis and subgroup metrics (age, sex, admission type). **Leak audit:** Claude Code plants two leaks on a branch; I find them and quantify the optimism each one buys. Write a 2-page report and a model card.
- **Resources:** TRIPOD+AI (https://www.bmj.com/content/385/bmj-2023-078378); Mitchell et al., *Model Cards for Model Reporting* (2019).
- **Exit check:** both leaks found with numbers; report committed. Tag `phase-4`.

## W16 · 18–24 Jan · Neural nets on tabular data (exam-risk week)
Check your exam calendar now. If this is exam season, do only the exit check and treat the rest as buffer.
- **Theory (chat):** why tree ensembles often beat MLPs on medium-size tabular data (irregular target functions, uninformative features, rotation invariance).
- **Build:** a PyTorch MLP on the project data (reuse the loop from `makemore_torch.py`) vs XGBoost under the same grouped CV.
- **Resources:** Grinsztajn et al. (2022), *Why do tree-based models still outperform deep learning on typical tabular data?* (https://arxiv.org/abs/2207.08815).
- **Exit check:** explain, with evidence from my own run, why the MLP lost or won.

# Phase 5 — Time, interpretability, clinical data (Jan–Feb)

## W17 · 25–31 Jan · Time-aware validation and survival
- **Theory (chat):** temporal leakage, forward-chaining splits, prediction horizon; censoring, Kaplan–Meier, Cox proportional hazards, concordance index.
- **Build:** a time-aware splitter in `mlj/eval/`; on the `lifelines` built-in Rossi dataset fit Kaplan–Meier and Cox, then XGBoost with the `survival:cox` objective, and compare concordance.
- **Resources:** https://lifelines.readthedocs.io/ ; the XGBoost survival tutorials in the docs.
- **Exit check:** explain why random K-fold is wrong for time-ordered data; define the concordance index.

## W18 · 1–7 Feb · Interpretability
- **Theory (chat):** gain/cover importances and why they mislead, permutation importance, PDP/ICE, Shapley values and TreeSHAP, the correlated-feature problem, association vs causation.
- **Build:** SHAP on the project's XGBoost model vs permutation importance; inject a proxy feature and confirm it dominates; write a one-page "what SHAP can and cannot say".
- **Resources:** Molnar, *Interpretable Machine Learning* (https://christophm.github.io/interpretable-ml-book/); Lundberg & Lee (2017), https://arxiv.org/abs/1705.07874; `shap` docs.
- **Exit check:** given a confident SHAP story, name a non-causal mechanism that could produce it.

## W19 · 8–14 Feb · Clinical data and robustness
- **Theory (chat):** what EHR data looks like (labs, vitals, codes, irregular sampling), informative missingness, distribution shift, subgroup performance.
- **Build:** a missingness experiment on the project data (simulate MCAR vs MNAR and watch XGBoost's native handling); a subgroup dashboard.
- **Resources:** MIMIC-IV demo documentation (https://physionet.org/content/mimic-iv-demo/).
- **Exit check:** explain informative missingness with a clinical example in my own words.

# Phase 6 — Lab-facing work (February)

## W20 · 15–21 Feb · Read the lab
- **Theory (chat):** collect four or five recent papers from the lab's publication page; structured reading (question, data, method, evaluation, limitations); list the methods I do not know.
- **Build:** in `projects/lab-replication/`, reproduce one baseline or key figure from a paper with public data, or from an analogous public dataset.
- **Resources:** Keshav, *How to Read a Paper*.
- **Exit check:** one page per paper with three questions to ask the lab.

## W21 · 22–28 Feb · Consolidate
- **Build:** clean the repo so a PI can read it in ten minutes; update `chat-context.md`; write the handoff note (still-shaky list, questions for the lab).
- **Exit check:** retake the full diagnostic A–H closed book and compare with 2026-09-30. Commit the score report in `LOG.md`. Tag `phase-6`.

---

## If a week slips

1. Cut the practice, never the exit check.
2. Two slipped weeks in a row → drop W16 and W19 first, then the optional parts of W20.
3. Never skip the `LOG.md` entry: the *still shaky* list is what drives the next chat session.
4. Anything in *still shaky* after two weeks becomes the first topic of the next week.

This is a plan, not a contract. If something turns up that is worth chasing, chase it and
rewrite the rest — but write down why in the log.

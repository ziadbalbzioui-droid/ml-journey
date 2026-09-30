# Roadmap — 5 Oct 2026 → 28 Feb 2027 (21 weeks)

Revised 2026-09-30 (evening), after the diagnostic test (`LOG.md`) and a change of strategy:
**one foundations week, then project-driven.** Every project must be good enough to show on a
resume and to a research group at Stanford.

**Level, honestly:** strong maths, beginner in classical ML. The diagnostic showed gaps in
formulas and vocabulary, not in ability. They are fixed in W1, and then *inside* the projects
by writing the core components from scratch. Every week still ends with a closed-book exit
check, so speed never hides a gap.

**Goal:** arrive at the Stanford research position in March 2027 with a public portfolio that
shows rigorous evaluation on clinical-style data, gradient boosting including XGBoost, time
series, survival analysis, high-dimensional data, and clear writing.

**Budget:** ~10 h/week core. Minimum viable week = 6 h. Weekly loop, tools and prompts:
[WORKFLOW.md](WORKFLOW.md). Claude Code builds and reviews; Claude chat teaches and examines.

## Portfolio

| # | Project | Weeks | What it shows | Why it matters for the lab | Priority |
| --- | --- | --- | --- | --- | --- |
| P1 | `mlj`: ML from scratch, verified (linear → trees → gradient boosting → XGBoost-lite) | W2–W4 | I understand the algorithms, not only the API | XGBoost is part of the lab's toolkit | Core |
| P2 | Clinical tabular prediction, rigorous evaluation | W5–W8 | Grouped CV, CIs, calibration, subgroups, leak audit, model card | The evaluation standard a clinical-ML group expects | Core |
| P3 | Sepsis early warning on ICU time series (PhysioNet/CinC 2019) | W9–W12 | Time-aware validation, irregular data, XGBoost vs GRU/attention, alarm-rate metrics | Longitudinal clinical data | Core |
| P4 | Survival analysis benchmark | W14–W15 | Censoring, Cox, random survival forest, XGBoost survival objectives | Time-to-event outcomes are common in medicine | Stretch |
| P5 | High-dimensional omics (p ≫ n) | W16–W17 | Elastic net, stability selection, leakage in feature selection | Omics-style data | Stretch |
| P6 | `mlj-audit`: evaluation and leakage toolkit | W18 | A reusable, tested, CI-checked Python package | Shows engineering and rigour | Core |
| P7 | Lab replication + 2-page research proposal | W20 | Reading papers, reproducing, proposing | Direct preparation for the position | Core |

If time runs short, cut in this order: P5, P4, W19. Never cut P1–P3, P6, P7, the exit checks
or the log.

**Definition of done for every project (resume-grade):**
1. Folder `projects/<name>/` with a README: question, data, method, headline result with a
   confidence interval, limitations.
2. Reproducible with `uv run` from a clean clone (data fetched by script, seeds fixed).
3. Tests for the core code (pytest).
4. A data card or model card, and one clear figure.
5. A 1-page write-up in plain language.
6. A resume bullet with a number, drafted by Claude Code and rewritten by me.

## Diagnostic gaps → where they get fixed

| Gap seen on 2026-09-30 | Fixed in |
| --- | --- |
| Normal equation, singular `XᵀX` (Q6) | W1 |
| Gradient-descent update on parameters (Q11) | W1 |
| Binary cross-entropy, softmax + CE gradient `p − y`, uniform initial loss (Q7, Q14, Q18) | W1 |
| Backprop vs numerical-gradient cost (Q16) | W1 |
| Ridge as a limit, scaling per model (Q4, Q10) | W1, W2 |
| Overfitting causes, why tuning on test breaks validity (Q1, Q2) | W1, W5 |
| Decision-tree splits, KNN in high dimension (Q8, Q9) | W2 |
| Saturation, BatchNorm vs LayerNorm, embeddings vs logits, residuals (Q19–22) | W11 |
| Attention: `/√d`, causal mask (Q37–39) | W11 |
| Sections E–H never taken (evaluation, code, theory links, stretch) | first taken W2, retaken W8 and W21 |

## Every week

- **Mon** — chat: theory prep (three questions that find the gap, then the derivation).
- **Mon–Tue** — the resource of the week.
- **Tue–Thu** — Claude Code: build sessions, with the prediction question before any code.
- **Fri** — chat: oral exit check, closed book.
- **Sat** — concept note in `learning/concepts/`, `LOG.md` entry, commit, push.

---

# Foundations sprint

## W1 · 5–11 Oct · The only foundations week
- **Theory (chat):** normal equation `w = (XᵀX)⁻¹Xᵀy` with shapes `(n,d)` and the singular case; gradient descent `w ← w − α ∂L/∂w` and learning-rate failure modes; logistic regression, Bernoulli negative log-likelihood = binary cross-entropy; derive the softmax + CE gradient `p − y` and why a uniform softmax starts at loss `log C`; ridge `(XᵀX + λI)⁻¹Xᵀy` and `λ → ∞`; train/validation/test logic; AUROC as the probability a random positive outranks a random negative; PPV vs prevalence via Bayes.
- **Build (Claude Code):** `src/mlj/scratch/linear.py` (normal equation, GD, ridge), `logistic.py` (binary + softmax, NumPy), `gradcheck.py` (numerical vs analytic), `src/mlj/metrics.py` (confusion metrics, ROC/AUC from ranks). Check each against scikit-learn.
- **Resources:** ISLP ch. 3, 4, 5 and 6.2 (free PDF, https://www.statlearning.com); Andrew Ng, *Machine Learning Specialization* course 1 at double speed (https://www.coursera.org/specializations/machine-learning-introduction); 3Blue1Brown neural-network series (https://www.3blue1brown.com/topics/neural-networks).
- **Exit check:** retake diagnostic sections A–D closed book (target ≥ 80 %), including Q14 (`z = [0,0,0]`), Q18 and Q22.

# P1 — `mlj`: ML from scratch, verified (W2–W4)

## W2 · 12–18 Oct · Trees and forests
- **Theory (chat):** impurity (Gini, entropy, variance), greedy split search, why deep trees overfit, scale invariance; bagging, the variance of an average of correlated estimators `ρσ² + (1−ρ)σ²/B`, feature subsampling, out-of-bag error; KNN and the curse of dimensionality.
- **Build:** `scratch/tree.py` (regression + classification, exact greedy split) and `scratch/forest.py`; match scikit-learn trees at equal depth.
- **Resources:** ISLP ch. 8; StatQuest "Decision Trees" and "Random Forests" (https://www.youtube.com/@statquest).
- **Exit check:** compute the best split by hand on 6 points (Q8, Q9). **On Friday, also take diagnostic sections E–H for the first time**, as the baseline.

## W3 · 19–25 Oct · Gradient boosting from scratch
- **Theory (chat):** boosting as gradient descent in function space: each tree fits the negative gradient (residuals for MSE, `p − y` for logistic loss), shrinkage, learning rate vs number of trees.
- **Build:** `scratch/gbm.py` with squared and then logistic loss on your own tree; train/validation curve vs number of trees; compare with scikit-learn `GradientBoosting`.
- **Resources:** StatQuest Gradient Boost series; https://explained.ai/gradient-boosting/; Friedman (2001), *Greedy function approximation* (skim).
- **Exit check:** predict, before running, the effect of learning rate 1.0 vs 0.1; explain why "fit residuals" is "fit the negative gradient".

## W4 · 26 Oct–1 Nov · XGBoost-lite and the benchmark
- **Theory (chat):** regularised objective `Ω = γT + ½λ‖w‖²`, second-order Taylor expansion with `gᵢ, hᵢ`, optimal leaf weight `w* = −G/(H+λ)`, split gain with `γ` and `min_child_weight`, sparsity-aware missing-value handling, column/row subsampling.
- **Build:** `scratch/xgb_lite.py` (your tree with `g, h, λ, γ`), checked against the `xgboost` library (`tree_method="exact"`, same parameters) on toy data (`uv add xgboost`). Benchmark: logistic regression vs random forest vs your GBM vs XGBoost on three public datasets, same CV, early stopping, random-search tuning.
- **Resources:** Chen & Guestrin (2016), https://arxiv.org/abs/1603.02754 (sections 2–3); the docs page "Introduction to Boosted Trees" (https://xgboost.readthedocs.io); StatQuest XGBoost series.
- **Deliverable:** `projects/p1-mlj/` README with the benchmark table and the agreement with the library. **Exit check:** derive `w*` and the gain on paper; for an overfitting model name three hyperparameters to change and the direction of each. Tag `p1`.

# P2 — Clinical tabular prediction, rigorous evaluation (W5–W8)

## W5 · 2–8 Nov · Harness, data, split protocol
- **Theory (chat):** grouped, stratified and nested CV; why rows of the same patient leak (Q26); bootstrap percentile confidence intervals; missing data (MCAR / MAR / MNAR); the catalogue of leaks; fixing the target and the prediction time before looking at any model.
- **Build:** `src/mlj/eval/splits.py` and `bootstrap.py` with pytest tests; a synthetic demo where a row-level split looks good and a patient-level split does not. Dataset: UCI *Diabetes 130-US hospitals* (30-day readmission), which has a patient id — Claude Code verifies the link and licence first (https://archive.ics.uci.edu). Data card, EDA, README with question, target, unit of analysis and split protocol. **Lock the test set before any modelling.**
- **Resources:** scikit-learn docs, "Cross-validation: evaluating estimator performance"; Kaufman et al. (2012), *Leakage in Data Mining* (https://dl.acm.org/doi/10.1145/2382577.2382579).
- **Exit check:** Q26, Q27, Q29; Claude Code reviews the README as a skeptical reviewer and lists three leak risks; fix them.

## W6 · 9–15 Nov · Baselines and tuning
- **Theory (chat):** what a fair baseline is (majority class, regularised logistic regression); tuning only inside the training folds; XGBoost native missing-value handling; why feature importance types (gain, cover, weight) mislead.
- **Build:** logistic regression, random forest and XGBoost under grouped, nested CV with `mlj.eval`; results table with bootstrap CIs. Predict which model wins, and by how much, before running.
- **Resources:** XGBoost docs (parameters page, early stopping).
- **Exit check:** table with CIs committed, plus the prediction-vs-result note.

## W7 · 16–22 Nov · Calibration, thresholds, subgroups
- **Theory (chat):** calibration, reliability curve, Brier score and log loss as proper scoring rules; tree-ensemble probabilities are often miscalibrated; Platt vs isotonic; choosing a threshold from cost or PPV; subgroup performance.
- **Build:** `src/mlj/eval/calibration.py` (reliability curve, Brier, ECE); calibrate the XGBoost model on a held-out fold, before/after plot; threshold analysis; metrics by age, sex and admission type.
- **Resources:** Van Calster et al. (2019), https://bmcmedicine.biomedcentral.com/articles/10.1186/s12916-019-1466-7; scikit-learn calibration docs.
- **Exit check:** Q23–Q25 and Q28 closed book: a model with 90 % sensitivity and specificity at 1 % prevalence, PPV by hand.

## W8 · 23–29 Nov · Leak audit, report, model card
- **Theory (chat):** reporting standards (TRIPOD+AI idea), model cards.
- **Build:** **leak audit** — Claude Code plants two leaks on an `exp/` branch; I find them and quantify the optimism each one buys. Then a 2-page report and a model card.
- **Resources:** TRIPOD+AI (https://www.bmj.com/content/385/bmj-2023-078378); Mitchell et al., *Model Cards for Model Reporting* (2019).
- **Deliverable:** `projects/p2-clinical-tabular/` meeting the definition of done. **Exit check:** retake diagnostic section E closed book (≥ 80 %). Tag `p2`.

# P3 — Sepsis early warning on ICU time series (W9–W12)

## W9 · 30 Nov–6 Dec · Data, time, labels
- **Theory (chat):** prediction horizon and prediction time; temporal leakage; irregular sampling, informative missingness (a lab is measured because the clinician suspects something).
- **Build:** `projects/p3-sepsis/`: fetch the PhysioNet/CinC 2019 challenge data (https://physionet.org/content/challenge-2019/; Claude Code checks access and licence), patient-level splits, hourly windows, label definition, EDA of missingness and sampling rates.
- **Resources:** the challenge paper and data description on that page.
- **Exit check:** README stating target, prediction time and split protocol; Claude Code lists three temporal-leak risks.

## W10 · 7–13 Dec · XGBoost baseline
- **Theory (chat):** feature design for time series (last value, rolling statistics, deltas, time since last measurement, missingness indicators).
- **Build:** XGBoost on those features with patient-grouped CV; AUROC, AUPRC, and PPV at a fixed alert rate.
- **Resources:** XGBoost docs; Harutyunyan et al., *Multitask learning and benchmarking with clinical time series* (https://arxiv.org/abs/1703.07771).
- **Exit check:** explain why AUROC alone is the wrong headline number here.

## W11 · 14–20 Dec · Sequence models and attention
- **Theory (chat):** GRU; embeddings (input vectors, row lookup in a `(V, d)` table) vs logits (output scores); `q @ kᵀ` shapes `(B, T, hs)`, why `/√hs`, the causal mask; residual connections `∂y/∂x = I + ∂F/∂x`; LayerNorm vs the `BatchNorm1D` you wrote; tanh saturation.
- **Build:** one attention head in **pure NumPy**, forward and backward, checked with `gradcheck.py` (`src/mlj/scratch/attention.py`); then a GRU and a small attention-based model in PyTorch on the sepsis windows, compared with XGBoost under the same split.
- **Resources:** Karpathy, *Let's build GPT*, the attention part (https://www.youtube.com/watch?v=kCc8FmEb1nY); Vaswani et al. (https://arxiv.org/abs/1706.03762); *The Annotated Transformer* (https://nlp.seas.harvard.edu/annotated-transformer/).
- **Exit check:** Q19–Q22 and Q37–Q39 closed book. The nanoGPT notebook stays parked; it is optional after W21.

## W12 · 21–27 Dec (light week, ~5 h) · Alarm framing and report
- **Theory (chat):** alarms per patient-day, PPV at fixed alert rate, alarm fatigue, lead time before onset, calibration over time.
- **Build:** operating-point analysis; defend one operating point to an imaginary ICU nurse (why this threshold, how many false alarms per shift, what happens to the miss rate if she asks for half the alarms); report.
- **Deliverable:** `projects/p3-sepsis/` meeting the definition of done. Tag `p3`.

## W13 · 28 Dec–3 Jan (light week, ~5 h) · Portfolio sprint
- Polish P1–P3: READMEs, figures, resume bullets, a clean clone test (`uv run` from scratch). Optional: a short blog post per project on a simple portfolio page.

# P4 — Survival analysis benchmark (W14–W15)

## W14 · 4–10 Jan · Censoring, Cox, forests, XGBoost
- **Theory (chat):** censoring, Kaplan–Meier, Cox proportional hazards, concordance index; XGBoost survival objectives (`survival:cox`, `survival:aft`).
- **Build:** `projects/p4-survival/` with scikit-survival datasets (GBSG2, WHAS500): Kaplan–Meier, Cox, random survival forest, XGBoost; concordance on a grouped/stratified split.
- **Resources:** scikit-survival docs (https://scikit-survival.readthedocs.io); lifelines docs (https://lifelines.readthedocs.io/); the XGBoost survival tutorials in the docs.
- **Exit check:** define the concordance index; explain why random K-fold is wrong for time-ordered data.

## W15 · 11–17 Jan · Survival calibration, interpretation, report
- **Theory (chat):** time-dependent AUC, integrated Brier score, calibration of survival predictions; SHAP on tree models.
- **Build:** evaluation with time-dependent metrics, SHAP on the XGBoost model, report and resume bullet.
- **Resources:** Molnar, *Interpretable Machine Learning* (https://christophm.github.io/interpretable-ml-book/); Lundberg & Lee (2017), https://arxiv.org/abs/1705.07874.
- **Deliverable:** definition of done. Tag `p4`.

# P5 — High-dimensional omics, p ≫ n (W16–W17)

Check your exam calendar now. If late January is exam season, these two weeks become buffer.

## W16 · 18–24 Jan · Data and baselines
- **Theory (chat):** why `p ≫ n` breaks ordinary CV habits; elastic net; stability selection; why feature selection must happen *inside* each CV fold.
- **Build:** `projects/p5-omics/`: a public, downloadable omics dataset (GEO or a DREAM-style challenge; Claude Code picks one and verifies the licence before you commit to it). Nested CV with elastic net and XGBoost; demonstrate the optimism from selecting features on the full data.
- **Resources:** ISLP ch. 6; the `scikit-learn` elastic-net docs.
- **Exit check:** explain, with your own numbers, how much optimism feature selection outside the fold buys.

## W17 · 25–31 Jan · Interpretation and report
- **Theory (chat):** stability of selected features, batch effects.
- **Build:** stability analysis of selected features, biological sanity check, report and resume bullet.
- **Deliverable:** definition of done. Tag `p5`.

# P6 — `mlj-audit` (W18)

## W18 · 1–7 Feb · Evaluation and leakage toolkit
- **Theory (chat):** which checks catch which leaks: group overlap between train and test, duplicate rows, a single feature with suspiciously high AUC, temporal order violations, calibration gaps.
- **Build:** refactor `mlj.eval` into an installable package `mlj-audit` with a small CLI: checks, a calibration and subgroup report, pytest, GitHub Actions CI, a README showing before/after on the leaks found in P2 and P3.
- **Resources:** pytest docs; GitHub Actions docs.
- **Deliverable:** package with tests passing in CI. **Exit check:** run it on a fresh dataset and explain every warning it raises. Tag `p6`.

# Research preparation

## W19 · 8–14 Feb · Interpretability and robustness across projects
- **Theory (chat):** permutation importance, PDP/ICE, SHAP and where each one lies (correlated features, off-manifold sampling, causal misreading); shortcut learning; distribution shift and subgroup performance.
- **Build:** apply permutation importance and SHAP to the P2 and P3 models; inject a proxy feature and confirm it dominates; write a one-page "what our explanations can and cannot say".
- **Resources:** Molnar's book; Rudin (2019), https://arxiv.org/abs/1811.10154.
- **Exit check:** given a confident SHAP story, name a non-causal mechanism that could produce it.

## W20 · 15–21 Feb · P7 — Lab replication and proposal
- **Theory (chat):** collect four or five recent papers from the lab's publication page; structured reading (question, data, method, evaluation, limitations); list the methods I do not know.
- **Build:** in `projects/p7-lab-replication/`, reproduce one baseline or key figure from a paper with public data, or from an analogous public dataset; draft a 2-page research proposal (question, data, method, evaluation plan, risks, why it matters clinically).
- **Resources:** Keshav, *How to Read a Paper*.
- **Exit check:** one page per paper with three questions to ask the lab.

## W21 · 22–28 Feb · Portfolio and consolidation
- **Build:** portfolio README at the repo root linking every project with its headline result; final resume bullets; handoff note (still-shaky list, questions for the lab); update `chat-context.md`.
- **Exit check:** retake the full diagnostic A–H closed book and compare with 2026-09-30. Commit the score report in `LOG.md`. Tag `phase-final`.

---

## Stretch (only if ahead, +4 h/week at most)

- One Kaggle tabular competition with a public write-up.
- One small pull request (docs or tests) to an open-source library used in the projects.
- An arXiv-style technical report for P3.
- Finish the nanoGPT notebook from `notebooks/02-zero-to-hero/`.

## If a week slips

1. Cut the practice, never the exit check.
2. Two slipped weeks in a row → cut in the order given under **Portfolio**.
3. Never skip the `LOG.md` entry: the *still shaky* list drives the next chat session.
4. Anything in *still shaky* after two weeks becomes the first topic of the next week.

This is a plan, not a contract. If something turns up that is worth chasing, chase it and
rewrite the rest — but write down why in the log.

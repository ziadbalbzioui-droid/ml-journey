# Roadmap — October 2026 → February 2027

~10 h/week. Target: research-grade ML on biomedical and clinical data. Depth over
breadth — one theme per month, one thing that runs, one proof I understood it.

| Month | Theme | Deliverable |
| --- | --- | --- |
| Oct | Finish the transformer + training craft | char-GPT that writes readable text, attention in NumPy |
| Nov | Statistical learning + honest evaluation | `src/mlj/eval/` harness (CV, bootstrap CI, calibration) |
| Dec | Clinical tabular ML, done properly | MIMIC-IV demo project + model card + TRIPOD checklist |
| Jan | Time series & sequences in medicine | early-warning model with time-aware validation |
| Feb | Interpretability, robustness, research move | 2-page research proposal + portfolio repo |

Rule for every month: **theory in the morning, code in the evening, one concept note per
week** in `concepts/`. If a month slips, cut the practice, never the deliverable.

---

## October 2026 — Finish the transformer, master the training loop

Picking up exactly where I stopped: `nano-gpt/train.ipynb` has token + position
embeddings and a linear head, and no attention yet.

**Theory**
- Self-attention: why query·key, why `/√d_k`, what the causal `tril` mask enforces.
- Multi-head attention: why several small heads beat one big one.
- Residual connections and LayerNorm — and how LayerNorm differs from the `BatchNorm1D`
  I hand-rolled in `makemore_rnn.ipynb` (per-sample vs per-batch, train/eval behaviour).
- Optimisation craft: AdamW, warmup + cosine schedules, initialisation scaling, why deep
  nets need residuals to train at all.

**Practice**
- Finish nanoGPT: `Head` → `MultiHeadAttention` → `FeedForward` → `Block` → full GPT.
- One attention head in **pure NumPy**, forward *and* backward, checked against a
  numerical gradient. This is the from-scratch requirement for the month.
- Ablations: remove `/√d_k`, remove residuals, remove LayerNorm. Predict each effect in
  writing *before* running it.

**Deliverable** — a char-GPT trained on tinyshakespeare producing text that scans as
English, plus `src/mlj/scratch/attention.py` (gradient-checked) and
`concepts/self-attention.md`.

**Proof of understanding** — on paper, no notes: derive the shapes through one attention
block from `(B, T, C)`, and derive the gradient of softmax. Then explain why the
√d_k scaling matters more as `d_k` grows, and show the ablation numbers that confirm it.

**Resources**
- Karpathy, *Let's build GPT: from scratch, in code, spelled out* — https://www.youtube.com/watch?v=kCc8FmEb1nY
- Vaswani et al., *Attention Is All You Need* — https://arxiv.org/abs/1706.03762
- *The Annotated Transformer* (Harvard NLP) — https://nlp.seas.harvard.edu/annotated-transformer/

---

## November 2026 — Statistical learning and evaluation I can defend

The month that turns me from someone who trains models into someone who can referee one.

**Theory**
- Bias–variance, and what regularisation really is: derive ridge as MAP with a Gaussian
  prior, lasso as Laplace. MLE vs MAP vs full Bayes.
- Cross-validation done right: stratified, grouped, nested CV; why tuning on the test set
  inflates everything.
- Uncertainty: bootstrap confidence intervals on a metric.
- Metrics under class imbalance: ROC vs precision–recall, PPV/NPV and their dependence on
  prevalence, proper scoring rules (log-loss, Brier), calibration and why it is the thing
  clinicians actually need.

**Practice**
- NumPy from scratch: linear regression (closed form *and* gradient descent), ridge,
  logistic regression. Then check each against scikit-learn — same coefficients or bust.
- Build the evaluation harness I will reuse for the rest of the roadmap.

**Deliverable** — `src/mlj/eval/`: stratified & grouped CV splitters, bootstrap CIs,
ROC/PR curves, calibration curve + Brier score decomposition, with unit tests. Demoed on
one public clinical tabular dataset.

**Proof of understanding** — take a model with AUROC 0.92 on a condition with 2%
prevalence, and compute by hand the PPV at the threshold I would deploy. Explain in three
sentences why that model may still be useless, and what I would report instead. Rederive
ridge from the MAP argument without notes.

**Resources**
- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*, ch. 3 & 7 — https://hastie.su.domains/ElemStatLearn/
- Van Calster et al., *Calibration: the Achilles heel of predictive analytics* (BMC Medicine, 2019) — https://bmcmedicine.biomedcentral.com/articles/10.1186/s12916-019-1466-7
- Guo et al., *On Calibration of Modern Neural Networks* (2017) — https://arxiv.org/abs/1706.04599

---

## December 2026 — Clinical tabular ML, done properly

Real clinical data. On tabular clinical problems, gradient boosting and a well-specified
logistic regression are the baselines to beat — and usually they win.

**Theory**
- Missing data: MCAR / MAR / MNAR, why "impute with the mean" can encode the outcome,
  and why *missingness itself* is often the strongest predictor in clinical data.
- Leakage: patient-level vs row-level splits, temporal splits, site splits, and features
  that only exist after the outcome (treatment codes, discharge fields).
- How trees and boosting actually work: splitting criteria, gradient boosting as gradient
  descent in function space.
- Reporting standards: TRIPOD+AI — what a reviewer expects to see.

**Practice**
- From scratch: a regression tree, then gradient boosting on top of it, in NumPy.
- Then the real thing: scikit-learn `HistGradientBoosting` / XGBoost vs a regularised
  logistic regression, on MIMIC-IV demo. Grouped splits by patient, always.

**Deliverable** — `projects/clinical-tabular/`: in-hospital mortality (or readmission)
prediction on the MIMIC-IV demo, with grouped splits, calibration, bootstrap CIs, a model
card, and a filled TRIPOD+AI-style checklist. Written as a short report, not a notebook dump.

**Proof of understanding** — I ask Claude to plant three leaks in my pipeline. I find all
three and quantify how much optimism each one buys. Then: explain why my boosted model
beat logistic regression, or admit it did not and say what that tells me about the data.

**Resources**
- MIMIC-IV Clinical Database Demo (PhysioNet, open access) — https://physionet.org/content/mimic-iv-demo/
- Collins et al., *TRIPOD+AI statement* (BMJ, 2024) — https://www.bmj.com/content/385/bmj-2023-078378
- Kaufman et al., *Leakage in Data Mining* (2012) — https://dl.acm.org/doi/10.1145/2382577.2382579

---

## January 2027 — Time series and sequences in medicine

Vitals and labs arrive irregularly, are censored, and get acted upon. This month is about
not fooling myself with time.

**Theory**
- Windowing and prediction horizons; temporal leakage; why the prediction time must be
  fixed before you look at anything.
- Survival analysis: censoring, Kaplan–Meier, Cox proportional hazards, concordance index
  — the right frame when the question is *when*, not *whether*.
- Sequence models for clinical series: GRU/LSTM, temporal convolutions, and transformers
  on irregular data; how masking and resampling change the problem.
- Alarm framing: alert rate, PPV at fixed alert rate, alarms per patient-day, alarm fatigue.

**Practice**
- A GRU cell from scratch in NumPy (forward + backward), then the PyTorch version.
- A sliding-window dataset with strict time-aware splits.
- A Cox model with `lifelines` on the same cohort, for contrast.

**Deliverable** — `projects/early-warning/`: deterioration or sepsis early-warning model
from vital signs, evaluated with time-aware validation, reported as PPV and alarms per
patient-day at a clinically plausible alert rate, with calibration checked over time.

**Proof of understanding** — defend one operating point to an imaginary ICU nurse:
why this threshold, how many false alarms per shift, what happens to the miss rate if she
asks for half the alarms. Explain why AUROC is the wrong headline number here.

**Resources**
- PhysioNet/CinC 2019 Challenge: *Early Prediction of Sepsis* — https://physionet.org/content/challenge-2019/
- Harutyunyan et al., *Multitask learning and benchmarking with clinical time series* — https://arxiv.org/abs/1703.07771
- `lifelines` documentation (survival analysis in Python) — https://lifelines.readthedocs.io/

---

## February 2027 — Interpretability, robustness, and the move to research

Turning five months of work into something a lab would take seriously.

**Theory**
- Permutation importance, PDP/ICE, SHAP — and precisely where each one lies: correlated
  features, off-manifold sampling, causal misreading of an associational explanation.
- Counterfactual explanations; inherently interpretable models as the alternative.
- Shortcut learning: models that solve the dataset instead of the task.
- Subgroup and fairness evaluation; performance under distribution shift; uncertainty.

**Practice**
- SHAP on the December and January models; check whether the top features are clinically
  plausible or artefacts of the pipeline.
- A deliberate shortcut experiment: inject a proxy feature (e.g. a treatment that only
  happens after diagnosis), confirm it dominates, then find it with the tools above.
- Subgroup breakdown of every metric: age, sex, care unit.

**Deliverable** — a 2-page research proposal (question, data, method, evaluation plan,
risks, why it matters clinically) to send with internship applications, plus the repo
cleaned into something a PI can read in ten minutes, plus one concept note per method.

**Proof of understanding** — I hand you a confident, plausible SHAP story about a feature.
You refute it: name the mechanism that could produce that pattern without the causal claim,
and design the experiment that would settle it.

**Resources**
- Molnar, *Interpretable Machine Learning* (SHAP, PDP, permutation chapters) — https://christophm.github.io/interpretable-ml-book/
- Rudin, *Stop explaining black box ML models for high stakes decisions* (2019) — https://arxiv.org/abs/1811.10154
- DeGrave et al., *AI for radiographic COVID-19 detection selects shortcuts over signal* (2021) — https://www.nature.com/articles/s42256-021-00338-7

---

## Monthly review (last session of each month, 30 min)

1. Did the deliverable ship? If not, what blocked it — time, or a gap I hid from myself?
2. Pass the proof-of-understanding check, out loud, no notes.
3. Re-read **Still shaky** in `LOG.md`: what cleared, what is still there after a month?
   Anything that survives two months becomes next month's first topic.
4. Update `chat-context.md` §4 and §5, then commit.

This roadmap is a plan, not a contract. If December's data work opens a question worth
chasing, chase it and rewrite the rest — but write down why in the log.

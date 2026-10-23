# Experiment log

One entry per run or session. The prediction is written **before** running.

```
### YYYY-MM-DD — <exp id> <short name>
**Rung:** 0 baseline | 1 GBDT | 2 CNN/GRU | 3 ensemble
**Hypothesis / prediction:** what I expect (score, which tolerances/events move) and why.
**Setup:** folds file, features, params, seed.
**Result:** CV mean ± sd; per-event AP; Δ vs baseline.
**Why it matched or didn't:**
**Next:**
```

---

### 2026-10-23 — setup
**Did:** Created `projects/sleep-states/`: download script, polars loaders, plotting,
submission formatter, and a re-implementation of the competition metric with 7 hand-checked
tests. No data downloaded yet (Kaggle credentials missing).
**Learned:** —
**Next:** Set up Kaggle credentials, accept the rules, run `fetch.py`. Then answer the metric
questions in the README session, before any EDA.

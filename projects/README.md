# projects/

Self-contained projects — the things I would actually show someone. One folder each, with
its own README stating the question, the data, the evaluation, and the result (including
when the result is "it did not work, here is why").

```
clinical-tabular/   Dec 2026 — mortality/readmission on MIMIC-IV demo
early-warning/      Jan 2027 — deterioration prediction from vitals
```

Every project README answers, in this order:

1. **Question** — the clinical question, and who would act on the answer.
2. **Data** — source, cohort definition, inclusion/exclusion, how missingness was handled.
3. **Split** — grouped by patient / by time / by site, and why. Leakage checked how?
4. **Baseline** — what a regularised logistic regression achieves. Beat it or explain it.
5. **Evaluation** — discrimination *and* calibration, with confidence intervals, plus the
   operating point and what it costs in false alarms.
6. **Limits** — what would break this in deployment.

A project without a stated baseline and a calibration plot is not finished.

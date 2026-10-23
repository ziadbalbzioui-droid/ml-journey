# Detecting sleep onset and wake-up from wrist accelerometry

Kaggle: [Child Mind Institute — Detect Sleep States](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states)
(2023, closed; late submissions still scored).

## 1. Question

Given a multi-day wrist accelerometer recording of a child, find the time steps at which they
fall asleep (`onset`) and wake up (`wakeup`) for each night. Who would act on it: sleep
researchers who currently rely on diaries or polysomnography to annotate actigraphy.

## 2. Data

`uv run python projects/sleep-states/fetch.py` → `data/cmi-sleep/` (~1 GB).

| File | Rows | Content |
| --- | --- | --- |
| `train_series.parquet` | ~127M | `series_id, step, timestamp, anglez, enmo` every 5 s |
| `train_events.csv` | ~14.5k | `series_id, night, event, step, timestamp`; `step` NaN if not labelled |
| `test_series.parquet` | tiny | format example only — the real test set is hidden |

**Retention: 60 days.** [cleanup.py](cleanup.py) runs as a launchd job (daily at 10:00 and at
login): it notifies during the last 7 days, then deletes `data/cmi-sleep/` 60 days after the
date in `.downloaded_at`. Check with `/usr/bin/python3 projects/sleep-states/cleanup.py --status`.

Loaders in [sleepdet/data.py](sleepdet/data.py). Data card: *to write after EDA*.

## 3. Metric

Event-detection AP averaged over 10 tolerances (1 → 30 min) and both event classes.
Re-implemented in [sleepdet/metric.py](sleepdet/metric.py), tested in
[tests/test_metric.py](tests/test_metric.py). `ap_table()` gives the per-tolerance breakdown.

## 4. Split

*To decide before the first model* — unit, number of folds, fixed seed, saved to disk so every
rung uses the same folds.

## 5. The ladder

Same folds, same metric, every rung compared with the baseline. Full history in
[EXPERIMENTS.md](EXPERIMENTS.md).

| Rung | Method | CV score (mean ± sd over folds) | Δ vs baseline |
| --- | --- | --- | --- |
| 0 | Rule-based baseline | | — |
| 1 | Hand-crafted features + GBDT | | |
| 2 | 1D-CNN / GRU | | |
| 3 | Ensemble | | |

## 6. Limits

*To write.*

## Layout

```
fetch.py            download
sleepdet/           data.py · metric.py · viz.py · submission.py   (boilerplate)
tests/              uv run pytest projects/sleep-states
notebooks/          EDA and experiments
EXPERIMENTS.md      prediction → result log, one entry per run
```

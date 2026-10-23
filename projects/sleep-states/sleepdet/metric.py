"""
Event-detection average precision, re-implemented from the competition's official metric.

For each event class e ∈ {onset, wakeup} and each tolerance τ (in steps):
  1. Per series, greedily match detections to ground-truth events: detections in decreasing
     score order, each takes the nearest still-unmatched ground truth with |Δstep| < τ (strict).
  2. Pool the matched/unmatched flags over all series, sweep a threshold on the score, and
     compute AP = Σ_k (R_k − R_{k−1}) P_k, with recall denominator = number of ground truths.
Final score = mean over τ, then mean over e.

Detections on series absent from the solution are ignored, as in the official code.
"""
import numpy as np
import pandas as pd

# 1 step = 5 s, so 12 steps = 1 min, 360 steps = 30 min
TOLERANCES = {
    "onset": [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
    "wakeup": [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
}


def match(gt_steps: np.ndarray, det_steps: np.ndarray, det_scores: np.ndarray, tol: float) -> np.ndarray:
    """Boolean array aligned with det_steps: True if that detection was matched to a ground truth."""
    matched = np.zeros(len(det_steps), dtype=bool)
    if len(gt_steps) == 0:
        return matched
    gt_steps = np.sort(gt_steps).astype(float)  # ties → earliest ground truth, as in the official code
    free = np.ones(len(gt_steps), dtype=bool)
    for i in np.argsort(-det_scores, kind="stable"):
        err = np.where(free, np.abs(gt_steps - det_steps[i]), np.inf)
        j = int(np.argmin(err))
        if err[j] < tol:
            matched[i] = True
            free[j] = False
    return matched


def average_precision(matched: np.ndarray, scores: np.ndarray, n_gt: int) -> float:
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    matched, scores = matched[order], scores[order]
    tp, fp = np.cumsum(matched), np.cumsum(~matched)
    last = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]  # tied scores = one threshold
    tp, fp = tp[last], fp[last]
    precision, recall = tp / (tp + fp), tp / n_gt
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def ap_table(solution: pd.DataFrame, submission: pd.DataFrame, tolerances=TOLERANCES) -> pd.DataFrame:
    """AP for every (event, tolerance). solution: series_id, event, step (NaN steps are dropped).
    submission: series_id, event, step, score."""
    sol = solution.dropna(subset=["step"])
    sub = submission[submission["series_id"].isin(sol["series_id"].unique())]
    sol_g = {k: g["step"].to_numpy(float) for k, g in sol.groupby(["series_id", "event"])}
    sub_g = {k: g for k, g in sub.groupby(["series_id", "event"])}
    rows = []
    for event, tols in tolerances.items():
        n_gt = int((sol["event"] == event).sum())
        if n_gt == 0:
            continue
        keys = [k for k in sub_g if k[1] == event]
        for tol in tols:
            flags, scores = [], []
            for k in keys:
                d = sub_g[k]
                steps, sc = d["step"].to_numpy(float), d["score"].to_numpy(float)
                flags.append(match(sol_g.get(k, np.empty(0)), steps, sc, tol))
                scores.append(sc)
            m = np.concatenate(flags) if flags else np.empty(0, bool)
            s = np.concatenate(scores) if scores else np.empty(0)
            rows.append({"event": event, "tolerance": tol, "ap": average_precision(m, s, n_gt)})
    return pd.DataFrame(rows)


def score(solution: pd.DataFrame, submission: pd.DataFrame, tolerances=TOLERANCES) -> float:
    t = ap_table(solution, submission, tolerances)
    return float(t.groupby("event")["ap"].mean().mean())

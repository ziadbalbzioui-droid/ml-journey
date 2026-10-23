import numpy as np
import pandas as pd
import pytest
from sleepdet.metric import average_precision, match, score

ONE_TOL = {"onset": [10], "wakeup": [10]}


def frame(rows, cols):
    return pd.DataFrame(rows, columns=cols)


def sol(*rows):
    return frame(rows, ["series_id", "event", "step"])


def sub(*rows):
    return frame(rows, ["series_id", "event", "step", "score"])


def test_perfect_predictions_score_one():
    s = sol(("a", "onset", 100), ("a", "wakeup", 500), ("b", "onset", 50), ("b", "wakeup", 900))
    d = s.assign(score=1.0)
    assert score(s, d) == pytest.approx(1.0)


def test_tolerance_is_strict():
    assert match(np.array([100]), np.array([110]), np.array([1.0]), tol=10).tolist() == [False]
    assert match(np.array([100]), np.array([109]), np.array([1.0]), tol=10).tolist() == [True]


def test_one_ground_truth_matched_only_once():
    m = match(np.array([100]), np.array([101, 99]), np.array([0.9, 0.8]), tol=10)
    assert m.tolist() == [True, False]


def test_duplicate_ranked_above_hit_halves_ap():
    s = sol(("a", "onset", 100), ("a", "wakeup", 500))
    good = sub(("a", "onset", 100, 0.9), ("a", "onset", 300, 0.1),
               ("a", "wakeup", 500, 0.9))
    bad = sub(("a", "onset", 100, 0.1), ("a", "onset", 300, 0.9),
              ("a", "wakeup", 500, 0.9))
    assert score(s, good, ONE_TOL) == pytest.approx(1.0)
    assert score(s, bad, ONE_TOL) == pytest.approx((0.5 + 1.0) / 2)


def test_missed_event_caps_recall():
    s = sol(("a", "onset", 100), ("a", "onset", 1000), ("a", "wakeup", 500))
    d = sub(("a", "onset", 100, 0.9), ("a", "wakeup", 500, 0.9))
    assert score(s, d, ONE_TOL) == pytest.approx((0.5 + 1.0) / 2)


def test_tied_scores_form_one_threshold():
    # TP and FP tied: a single point with P = 1/2, R = 1
    assert average_precision(np.array([True, False]), np.array([0.5, 0.5]), 1) == 0.5


def test_other_series_and_nan_ground_truth_ignored():
    s = sol(("a", "onset", 100), ("a", "wakeup", 500), ("a", "onset", None))
    d = sub(("a", "onset", 100, 0.9), ("a", "wakeup", 500, 0.9), ("zzz", "onset", 1, 1.0))
    assert score(s, d, ONE_TOL) == pytest.approx(1.0)

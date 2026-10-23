"""Format detections as the competition expects: row_id, series_id, step, event, score."""
import pandas as pd

COLUMNS = ["row_id", "series_id", "step", "event", "score"]


def to_submission(det: pd.DataFrame) -> pd.DataFrame:
    """det: series_id, step, event, score (any order, any extra columns)."""
    missing = {"series_id", "step", "event", "score"} - set(det.columns)
    assert not missing, f"missing columns: {missing}"
    assert det["event"].isin(["onset", "wakeup"]).all(), "event must be onset or wakeup"
    assert det["score"].notna().all() and det["step"].notna().all(), "NaN step or score"
    out = det.sort_values(["series_id", "step"]).reset_index(drop=True)
    out["step"] = out["step"].astype(int)
    out["row_id"] = range(len(out))
    return out[COLUMNS]

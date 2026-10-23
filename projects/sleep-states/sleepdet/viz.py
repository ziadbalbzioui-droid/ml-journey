"""Plot one series: anglez and enmo over time, with true events and (optionally) predictions."""
import matplotlib.pyplot as plt
import pandas as pd

COLORS = {"onset": "tab:green", "wakeup": "tab:red"}


def plot_series(series: pd.DataFrame, events: pd.DataFrame | None = None,
                preds: pd.DataFrame | None = None, start: int | None = None,
                stop: int | None = None, title: str | None = None):
    """series: one series_id, columns step/anglez/enmo. events, preds: step/event (+score).
    start/stop: step range to zoom on."""
    s = series
    if start is not None or stop is not None:
        s = s[s["step"].between(start or 0, stop or s["step"].max())]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 5), sharex=True)
    ax1.plot(s["step"], s["anglez"], lw=0.3, color="0.3")
    ax1.set_ylabel("anglez (°)")
    ax2.plot(s["step"], s["enmo"], lw=0.3, color="0.3")
    ax2.set_ylabel("enmo (g)")
    ax2.set_xlabel("step (5 s)")
    lo, hi = s["step"].min(), s["step"].max()
    for df, ls in ((events, "-"), (preds, ":")):
        if df is None:
            continue
        df = df.dropna(subset=["step"])
        df = df[df["step"].between(lo, hi)]
        for _, r in df.iterrows():
            for ax in (ax1, ax2):
                ax.axvline(r["step"], color=COLORS[r["event"]], ls=ls, lw=1.2)
    ax1.set_title(title or "solid = truth, dotted = prediction · green onset, red wakeup")
    fig.tight_layout()
    return fig

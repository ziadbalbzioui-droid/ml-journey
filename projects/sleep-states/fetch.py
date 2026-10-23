"""
Download the CMI "Detect Sleep States" competition files into data/cmi-sleep/ (~1 GB).

    uv run kaggle auth login                       # once
    uv run python projects/sleep-states/fetch.py

Also needs the competition rules accepted once:
https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/rules

Writes the download date to data/cmi-sleep/.downloaded_at (delete the folder by hand within 60 days).
"""
import datetime as dt
import sys
from pathlib import Path

import kagglehub

HANDLE = "child-mind-institute-detect-sleep-states"
DEST = Path(__file__).resolve().parents[2] / "data" / "cmi-sleep"


def main() -> None:
    if (DEST / "train_series.parquet").exists():
        print(f"already there: {DEST}")
    else:
        DEST.mkdir(parents=True, exist_ok=True)
        kagglehub.competition_download(HANDLE, output_dir=str(DEST))
        (DEST / ".downloaded_at").write_text(dt.datetime.now().astimezone().date().isoformat() + "\n")
    for f in sorted(DEST.iterdir()):
        print(f"  {f.name:30s} {f.stat().st_size / 1e6:8.1f} MB")


if __name__ == "__main__":
    sys.exit(main())

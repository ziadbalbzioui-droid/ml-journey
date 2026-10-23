"""
Download the CMI "Detect Sleep States" competition files into data/cmi-sleep/.

    uv run python projects/sleep-states/fetch.py

Needs Kaggle credentials (~/.kaggle/kaggle.json, or KAGGLE_USERNAME + KAGGLE_KEY) and the
competition rules accepted once on the website:
https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/rules
~1 GB on disk once extracted.
"""
from pathlib import Path

import kagglehub

HANDLE = "child-mind-institute-detect-sleep-states"
DEST = Path(__file__).resolve().parents[2] / "data" / "cmi-sleep"


def main() -> None:
    if (DEST / "train_series.parquet").exists():
        print(f"already there: {DEST}")
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    path = kagglehub.competition_download(HANDLE, output_dir=str(DEST))
    print("files in", path)
    for f in sorted(Path(path).iterdir()):
        print(f"  {f.name:30s} {f.stat().st_size / 1e6:8.1f} MB")


if __name__ == "__main__":
    main()

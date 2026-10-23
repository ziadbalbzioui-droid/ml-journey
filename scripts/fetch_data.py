"""
Download every dataset this repo uses into data/.

    python3 scripts/fetch_data.py

Files already present are skipped, so it is safe to re-run.
Kaggle House Prices is the one exception: it needs a Kaggle account,
so the script only tells you where to put it.
"""
import sys
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"

FILES = {
    # used by 02-zero-to-hero/makemore*.ipynb, hparam_tuning.py, makemore_torch.py
    "names.txt":
        "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt",
    # used by 02-zero-to-hero/nano-gpt/train.ipynb
    "tinyshakespeare.txt":
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    # used by 01-handson-ml/housing.ipynb
    "housing/housing.csv":
        "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv",
    # used by 01-handson-ml/classification.ipynb
    "titanic/train.csv":
        "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/titanic/train.csv",
    "titanic/test.csv":
        "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/titanic/test.csv",
}

KAGGLE_NOTE = """
Not downloaded automatically (Kaggle login required):
  data/house-prices/{train,test,sample_submission}.csv + data_description.txt
  -> https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
     or: kaggle competitions download -c house-prices-advanced-regression-techniques
Used by 01-handson-ml/house-prices-kaggle.ipynb.

  data/cmi-sleep/  -> uv run python projects/sleep-states/fetch.py
"""


def main() -> int:
    for rel, url in FILES.items():
        dest = DATA / rel
        if dest.exists():
            print(f"skip      {rel}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"download  {rel}")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as exc:                      # noqa: BLE001 - report and keep going
            print(f"  FAILED: {exc}\n  url: {url}")
    print(KAGGLE_NOTE)
    return 0


if __name__ == "__main__":
    sys.exit(main())

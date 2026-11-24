"""Dataset loaders for the 05-explore notebooks.

Each loader downloads its file once into ``data/`` (gitignored) and reads it from there afterwards.
All sources are plain files on GitHub, so no account or API key is needed.
"""

from __future__ import annotations

import gzip
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"

PIMA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
FASHION_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"

FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url} -> {dest.relative_to(DATA.parent)}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
    return dest


def load_pima() -> pd.DataFrame:
    """Pima Indians Diabetes: 768 women, 8 clinical features, `Outcome` = diabetes within 5 years.

    Returned raw, on purpose: some columns use 0 to mean "not measured".
    """
    return pd.read_csv(_download(PIMA_URL, DATA / "pima" / "diabetes.csv"))


def prepare_pima(seed: int = 0) -> dict[str, np.ndarray]:
    """Notebook 01's preprocessing, packaged: zeros -> NaN, stratified 60/20/20 split,
    train-median imputation, train-statistics standardisation. Returns X_tr, X_va, X_te
    (standardised, no bias column), y_tr, y_va, y_te and the feature names."""
    from sklearn.model_selection import train_test_split

    df = load_pima()
    cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols] = df[cols].replace(0, np.nan)
    feats = [c for c in df.columns if c != "Outcome"]
    trval, te = train_test_split(df, test_size=0.2, stratify=df["Outcome"], random_state=seed)
    tr, va = train_test_split(trval, test_size=0.25, stratify=trval["Outcome"], random_state=seed)
    med = tr[feats].median()
    mu, sd = tr[feats].fillna(med).mean(), tr[feats].fillna(med).std()
    out: dict[str, np.ndarray] = {"features": np.array(feats)}
    for name, d in [("tr", tr), ("va", va), ("te", te)]:
        out[f"X_{name}"] = ((d[feats].fillna(med) - mu) / sd).to_numpy()
        out[f"y_{name}"] = d["Outcome"].to_numpy()
    return out


def _read_idx(path: Path, offset: int) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=offset)


def load_fashion_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fashion-MNIST: 28x28 grey images of clothes, 10 classes.

    Returns X_train (60000, 784) uint8, y_train (60000,), X_test (10000, 784), y_test (10000,).
    """
    out = []
    for split in ["train", "t10k"]:
        img = _download(f"{FASHION_URL}{split}-images-idx3-ubyte.gz",
                        DATA / "fashion-mnist" / f"{split}-images-idx3-ubyte.gz")
        lab = _download(f"{FASHION_URL}{split}-labels-idx1-ubyte.gz",
                        DATA / "fashion-mnist" / f"{split}-labels-idx1-ubyte.gz")
        out += [_read_idx(img, 16).reshape(-1, 784), _read_idx(lab, 8).astype(np.int64)]
    return out[0], out[1], out[2], out[3]

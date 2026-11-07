# notebooks/

Exploration and course work. Notebooks are where I think; they are allowed to be messy,
but they must run top to bottom before being committed.

```
01-handson-ml/    scikit-learn on tabular data (Géron, Hands-On ML)
02-zero-to-hero/  neural nets from scratch (Karpathy) — micrograd → makemore → nanoGPT
03-ens-intro-ml/  ENS intro-ML labs — classification, trees, clustering, MLP, CNN, RNN, ViT
scratch/          throwaway experiments, no expectations
```

Data is read by relative path (`../../data/...`), so **run a notebook from its own
folder**. Restore the datasets with `uv run python scripts/fetch_data.py`.

Rule: once a piece of code is used twice, it moves to `src/mlj/` and the notebook imports
it. Notebooks are not a library.

# ml-journey

My machine-learning self-study, in one place: the code I write while learning, the
experiments I run, a dated progress log, and the context file I paste into Claude chat
so it always knows where I am.

Two tracks so far — classical ML on tabular data (Géron, *Hands-On ML*), then neural
nets from scratch (Karpathy, *Zero to Hero*).

## Map

```
01-handson-ml/            classical ML, scikit-learn, tabular data
├── housing.ipynb             California housing — EDA, regression
├── classification.ipynb      MNIST (KNN + GridSearchCV) and Titanic
└── house-prices-kaggle.ipynb Kaggle House Prices — NaN handling, feature prep

02-zero-to-hero/          neural nets built by hand, then in PyTorch
├── micrograd.ipynb           scalar autograd engine + MLP, backprop written by hand
├── makemore.ipynb            bigram character model (counts, then gradient descent)
├── makemore_mpl.ipynb        MLP character model (Bengio 2003 style)
├── makemore_rnn.ipynb        hand-rolled Linear / BatchNorm1D / Tanh / Embedding /
│                             FlattenConsecutive — hierarchical WaveNet-style net
├── makemore_torch.py         the same WaveNet rewritten with real nn.Modules + AdamW
├── hparam_tuning.py          LR range test, LR curves, capacity, train/dev gap,
│                             random search → plots/
├── plots/                    output of hparam_tuning.py (committed: they are results)
└── nano-gpt/train.ipynb      transformer from scratch on tinyshakespeare (in progress)

scratch/                  throwaway experiments
scripts/fetch_data.py     re-downloads every dataset into data/
data/                     gitignored — see data/README.md
PROGRESS.md               dated log: what I built, what clicked, what is still unclear
CLAUDE_CONTEXT.md         paste this into Claude chat at the start of a session
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/fetch_data.py     # pulls the datasets into data/
```

Kaggle House Prices needs a Kaggle login — `fetch_data.py` prints where to put it.

Notebooks read data through relative paths (`../data/...`), so run them from their own
folder. The two `.py` scripts resolve `data/` from their own location, so they run from
anywhere:

```bash
python3 02-zero-to-hero/hparam_tuning.py     # regenerates plots/
python3 02-zero-to-hero/makemore_torch.py
```

## Working with Claude

- **Claude chat (learning):** paste [`CLAUDE_CONTEXT.md`](CLAUDE_CONTEXT.md) into a new
  chat. It tells Claude how far I am, how I want to be taught (coach, not code
  dispenser), and what I am stuck on.
- **After a session:** add an entry to [`PROGRESS.md`](PROGRESS.md) and update the
  "where I am / what I'm stuck on" sections of `CLAUDE_CONTEXT.md`. Those two files are
  the whole point of the repo — the code is just the evidence.

## Rules I try to hold myself to

1. Write it from scratch once before importing the library version.
2. Every notebook ends with something that runs and produces a sample or a number.
3. If I cannot explain a line to someone else, I have not learned it yet.

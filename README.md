# ml-journey

Machine-learning self-study, aimed at research on biomedical and clinical data.
Plan, code, experiments, and an honest record of what I do not yet understand.

October 2026 → February 2027, ~10 h/week. Plan: [learning/ROADMAP.md](learning/ROADMAP.md).

## Layout

```
CLAUDE.md                 rules for Claude Code working in this repo
learning/
├── ROADMAP.md            month-by-month plan, ending on clinical ML
├── LOG.md                session log: did / understood / still shaky / next
├── chat-context.md       paste into claude.ai to bring a chat up to speed
└── concepts/             one short note per concept
notebooks/
├── 01-handson-ml/        scikit-learn on tabular data (Géron)
├── 02-zero-to-hero/      micrograd → makemore → nanoGPT (Karpathy)
└── scratch/              throwaway
src/mlj/                  reusable code — `scratch/` (NumPy from scratch), `eval/` (harness)
projects/                 self-contained projects, one folder each
scripts/fetch_data.py     re-downloads the datasets
data/                     gitignored
```

## Setup

```bash
uv sync                               # .venv from pyproject.toml, installs mlj editable
uv run python scripts/fetch_data.py   # datasets into data/
uv run jupyter lab
```

Kaggle House Prices needs a Kaggle login; `fetch_data.py` prints where to put it.
Notebooks read `../../data/...`, so run them from their own folder.

```bash
uv run python notebooks/02-zero-to-hero/hparam_tuning.py    # regenerates plots/
```

## How this repo is used

- **Claude Code**, in the repo: reads [CLAUDE.md](CLAUDE.md) — teach before solving,
  NumPy before libraries, flag shallow understanding, log and commit at the end of every
  session.
- **claude.ai chat**, for learning: paste [learning/chat-context.md](learning/chat-context.md)
  into the first message.
- **Me**, every session: one entry in [learning/LOG.md](learning/LOG.md), one commit.

## Principles

1. Implement it from scratch once before importing the library version.
2. A concept is not learned until there is a note in `learning/concepts/` for it.
3. Every model gets its baseline, its calibration, and its confidence interval — or it
   does not get reported.

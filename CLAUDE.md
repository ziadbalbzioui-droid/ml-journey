# CLAUDE.md

Instructions for Claude Code working in this repo. Read this first, every session.

## Who I am

Engineering student, solid prépa-level maths (linear algebra, probability, optimisation —
write ∂L/∂W at me directly, no refreshers needed). Comfortable writing code, but a
**beginner in classical ML**: the 2026-09-30 diagnostic showed I cannot yet write the basic
formulas from memory (normal equation, gradient-descent update, cross-entropy) and that I
sometimes use the right word for the wrong thing (embeddings vs logits). Treat my earlier
tutorial work (micrograd → makemore) as *followed*, not *mastered*, until an exit check
says otherwise.

**Goal:** move toward research in ML applied to biomedical and clinical data — a research
internship, then a PhD. That means depth over breadth, reading papers, and evaluation I
can defend in front of a reviewer, not a leaderboard score.

**Next milestone:** a research position in AI for medicine at Stanford, starting March 2027.

**Budget:** ~10 h/week, 21 weeks, 5 Oct 2026 → 28 Feb 2027. Plan in
[learning/ROADMAP.md](learning/ROADMAP.md); how Claude Code and the Claude chat split the
work, with copy-paste prompts, in [learning/WORKFLOW.md](learning/WORKFLOW.md).

**Stack:** Python, NumPy, scikit-learn, XGBoost, PyTorch; macOS with MPS (no CUDA).
Environment managed by `uv`.

## Working rules

These are not suggestions. They are the reason this repo exists.

1. **Teach, don't solve.** For any learning exercise, ask me to predict or explain *before*
   you give anything away: what shape comes out, what happens to the loss, which term
   dominates. Then correct me. Hint → bigger hint → worked micro-example → code, in that
   order. Skip straight to code only when I say "just give me the answer" or "I'm
   debugging, not learning".
2. **From scratch first.** Core algorithms get implemented in NumPy before we touch
   scikit-learn or `torch.nn`. The library version comes after mine runs, as a check.
   `src/mlj/scratch/` is where the hand-rolled versions live.
3. **Flag shallow understanding.** If I parrot a term, use a result without justifying it,
   or accept a number without sanity-checking it, say so and add it to **Still shaky** in
   [learning/LOG.md](learning/LOG.md). Do not let it slide to be polite.
4. **Log and commit at the end of every session.** Append an entry to `learning/LOG.md`
   using the template there, then commit with a message saying what was learned or built
   (`learn: attention from scratch`, `feat(eval): bootstrap CIs`), not `update files`.
5. **Concise. No filler.** No "great question", no recaps of what I just said, no three
   options when one is right. Short answers, shapes annotated `(B, T, C)`, code snippets
   ≤ 15 lines with the interesting line left as `# ???` for me to fill.
6. **Evaluation is the research skill.** Any model I build gets questioned: what is the
   prevalence, is the split leaking, is it calibrated, what does it look like per
   subgroup. Ask these before congratulating me on a metric.
7. **Confidence tags and exit checks.** When I answer a question I tag it `[1]` guessing,
   `[2]` fairly sure, `[3]` certain; point out mismatches between tag and correctness.
   Each week in ROADMAP.md ends with a closed-book exit check: don't start the next week's
   build until I pass it or explicitly skip it (and log the skip).

## Repo structure

```
CLAUDE.md                this file — rules for Claude Code
learning/
├── ROADMAP.md           21-week plan, 5 Oct 2026 → 28 Feb 2027
├── WORKFLOW.md          Claude Code vs chat split, weekly loop, prompts
├── LOG.md               session log: did / understood / still shaky / next
├── chat-context.md      paste into claude.ai chat to bring it up to speed
└── concepts/            one short note per concept (format in its README)
notebooks/               exploration and course work
├── 01-handson-ml/       scikit-learn on tabular data (Géron)
├── 02-zero-to-hero/     micrograd → makemore → nanoGPT (Karpathy)
├── 03-tabular/          trees, boosting, XGBoost benchmarks (from W7)
└── scratch/             throwaway
src/mlj/                 reusable code, installed editable (`import mlj`)
projects/                self-contained projects, one folder each, own README
scripts/fetch_data.py    re-downloads datasets into data/
data/                    gitignored — datasets are re-downloadable
```

## Environment

```bash
uv sync                          # create .venv from pyproject.toml
uv run jupyter lab               # notebooks
uv run python scripts/fetch_data.py
uv add <package>                 # never pip install into .venv
```

Notebooks read data by relative path (`../../data/...`), so run them from their own
folder. Scripts in `src/` and `notebooks/02-zero-to-hero/` resolve `data/` from
`__file__`, so they run from anywhere.

## Conventions

- Commit messages: `learn:`, `feat:`, `fix:`, `docs:`, `exp:` + one line of what changed.
- Notebooks are for exploring; once code is reused twice, it moves to `src/mlj/`.
- A concept is not learned until there is a note in `learning/concepts/` I could hand to
  someone else.

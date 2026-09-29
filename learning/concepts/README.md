# Concept notes

One file per concept, named in kebab-case: `self-attention.md`, `layer-norm.md`,
`calibration.md`, `censoring.md`.

A concept is not learned until it has a note here that someone else could read.

## Format

Four sections, **one page maximum**. If it runs longer, the concept is really two concepts.

```markdown
# <Concept>

## Intuition
Three sentences, no notation. What problem does this solve, and what would break
without it? If I cannot write this without the formula in front of me, I do not
understand it yet.

## Math
The one equation that matters, with every symbol named and its shape given.
Plus the one-line derivation or argument for *why* it takes that form.

## Code
Pointer to where it lives in this repo — `src/mlj/scratch/attention.py:42`, or the
notebook cell. Not a copy of the code: a pointer, so it stays true when the code changes.
At most a 5-line excerpt if one line is the whole point.

## Common mistakes
Three bullets. Mine first — the thing I actually got wrong — then the classic ones.

## Check
One question whose answer proves the concept is understood, not memorised. Usually:
"what happens to X if I change Y, and why?"
```

## Rules

- Write the note **after** the code runs, not before. The note records what I learned by
  building it, not what I planned to learn.
- Rewrite an existing note rather than adding a second one on the same topic.
- Link related notes: `[layer-norm.md](layer-norm.md)`.
- Target: one note per week.

# src/

Reusable code, installed in editable mode by `uv sync` — so `import mlj` works from any
notebook or script without path hacks.

```
mlj/
├── scratch/   from-scratch NumPy implementations (attention, trees, GRU, …)
└── eval/      the evaluation harness: CV splitters, bootstrap CIs, calibration, curves
```

Rules:

- **`scratch/` comes first.** Every core algorithm gets a NumPy version here, checked
  against a numerical gradient or against the library equivalent, before I am allowed to
  use the library version in a project.
- Anything in `eval/` gets a test. Evaluation code that is silently wrong is worse than no
  evaluation code — it produces numbers I will believe.
- No notebook imports from a notebook. Shared code lives here.

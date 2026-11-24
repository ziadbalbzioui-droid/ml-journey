"""Build the 05-explore notebooks from their sources.

    uv run python notebooks/05-explore/_build/build.py            # build student + solution notebooks
    uv run python notebooks/05-explore/_build/build.py --execute  # also run the solutions (fills outputs)

Source format (`src/*.nbsrc`): cells separated by `%% md` or `%% code` lines. Inside a code cell,

    #<<SOL
    ...solution lines...
    #==
    ...stub lines shown to the student...
    #>>

The student notebook gets the stub, the solution notebook gets the solution.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
SOL_RE = re.compile(r"^[ \t]*#<<SOL\n(.*?)^[ \t]*#==\n(.*?)^[ \t]*#>>\n", re.S | re.M)


def split_cells(text: str) -> list[tuple[str, str]]:
    cells, kind, buf = [], None, []
    for line in text.splitlines(keepends=True):
        m = re.match(r"^%% (md|code)\s*$", line)
        if m:
            if kind is not None:
                cells.append((kind, "".join(buf).strip("\n")))
            kind, buf = m.group(1), []
        else:
            buf.append(line)
    if kind is not None:
        cells.append((kind, "".join(buf).strip("\n")))
    return cells


def render(src: str, solution: bool) -> str:
    return SOL_RE.sub(lambda m: m.group(1) if solution else m.group(2), src + "\n").rstrip("\n")


def build(path: Path, solution: bool) -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for kind, src in split_cells(path.read_text()):
        if kind == "md":
            nb.cells.append(new_markdown_cell(src))
        else:
            nb.cells.append(new_code_cell(render(src, solution)))
    return nb


def main() -> None:
    execute = "--execute" in sys.argv
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    (OUT / "solutions").mkdir(exist_ok=True)
    for path in sorted((HERE / "src").glob("*.nbsrc")):
        if only and not any(o in path.stem for o in only):
            continue
        student = OUT / f"{path.stem}.ipynb"
        sol = OUT / "solutions" / f"{path.stem}_solution.ipynb"
        nbformat.write(build(path, False), student)
        nb = build(path, True)
        if execute:
            from nbclient import NotebookClient
            print(f"executing {sol.name} ...", flush=True)
            NotebookClient(nb, timeout=1800, resources={"metadata": {"path": str(OUT)}}).execute()
        nbformat.write(nb, sol)
        print(f"built {student.name} and solutions/{sol.name}")


if __name__ == "__main__":
    main()

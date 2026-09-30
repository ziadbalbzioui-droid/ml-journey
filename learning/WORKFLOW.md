# Workflow — Claude Code + Claude chat

The repo is the source of truth. Claude Code works *inside* it; Claude chat reads it through
GitHub. The plan they follow is [ROADMAP.md](ROADMAP.md).

## Who does what

| | Claude Code (in this repo) | Claude chat (Project "AI/ML preparation", GitHub connected) |
| --- | --- | --- |
| Main job | Build, run, test, review, commit, log | Theory, derivations, oral exams, paper reading |
| Typical week | 2 build sessions of ~2 h | 1.5 h theory prep + 1 h exit check |
| Does | Writes the scaffold and tests, runs experiments, checks my code against scikit-learn / XGBoost, reviews my code as a skeptical reviewer, plants leaks for audits, appends `LOG.md`, commits | Explains from a concrete shape, derives with me, quizzes me, corrects my vocabulary, reads papers with me, drafts the text of concept notes |
| Never | Hands me the solution before I predict | Writes my exit-check answers |

Rule of thumb: if it has to **run**, Claude Code. If I cannot **explain** it, chat. If I can
explain it but the code fails, Claude Code — and I say "I'm debugging, not learning".

## The weekly loop (~10 h)

| Day | Where | What |
| --- | --- | --- |
| Mon | Chat (1.5 h) | Theory prep for the week (prompt A) |
| Mon–Tue | Resource (2.5 h) | The video / chapter / paper listed in the roadmap |
| Tue–Thu | Claude Code (4 h) | Build sessions (prompts B, C, D) |
| Fri | Chat (1 h) | Oral exit check, closed book (prompt E) |
| Sat | Claude Code (1 h) | Concept note, `LOG.md`, commit, push (prompt D) |

Minimum viable week (6 h): Monday prep, one build session, Friday exit check.

## Prompts to copy

**A — Chat, theory prep**
> Week N theory prep, topic: <topic>. Start with three questions, one at a time, to find what I already know. Then derive the key result with me from a concrete example. End with one check question. Hint ladder if I'm stuck; no full solution unless I say so. Ask me for a [1]/[2]/[3] confidence tag on each answer.

**B — Claude Code, start of a session**
> Read CLAUDE.md, this week's block in learning/ROADMAP.md and the latest learning/LOG.md entry. In three lines: what is this week's exit check, and what am I still shaky on. Then ask me the prediction question for the first build step. Do not write code until I answer.

**C — Claude Code, during a session**
> Hint ladder: hint, bigger hint, worked micro-example, code. I predict first, you correct. When you check my code against scikit-learn or XGBoost, tell me *where* to look if they differ, not the fix.

**D — Claude Code, end of a session**
> Session end. Append a LOG.md entry with the template: what I built, what I can now explain, still shaky (including anything I hesitated on today), next step. Update §4 and §5 of learning/chat-context.md. Commit with a learn:/feat: message and push.

**E — Chat, oral exit check**
> Oral exam on week N of the roadmap. Closed book. One question at a time, no hints until I say "hint". Grade each answer 0–2 and compare my confidence tag with the result. At the end list my gaps in a block I can paste into LOG.md "Still shaky".

## Connecting GitHub to the chat

1. In claude.ai, open the Project "AI/ML preparation" and add this repo to its knowledge through the GitHub option (connect GitHub first if asked).
2. Push at the end of every Claude Code session, then re-sync the repo in the Project so the chat sees the new `LOG.md`.
3. If the sync looks stale, paste `learning/chat-context.md` as the first message instead.
4. Anything worth keeping from the chat (a derivation, a corrected explanation) goes back into the repo as a concept note: tell Claude Code "write learning/concepts/<topic>.md from this derivation" and paste it. I review the note; I do not just accept it.

## Git

- Commit prefixes: `learn:`, `feat:`, `fix:`, `docs:`, `exp:` + one line saying what changed.
- One commit (or a few) per session; push every session.
- Experiments that might be thrown away (leak audits, ablations) go on `exp/<name>` branches; `main` stays clean.
- Tag the end of each phase: `phase-1` … `phase-6`.

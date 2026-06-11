# CLAUDE.md

Read **AGENTS.md** first — it is the operating contract for this repo (the Task seam,
the tier ladder, paper-fidelity rules, ledger discipline). Everything there applies to
you.

Repo quick facts:

- `imagelab/` is the framework (public API = the seams in AGENTS.md); `examples/` are
  the blessed model families; `tests/` must stay CPU-only, network-free, and fast.
- Run things with `uv run lab ...` / `uv run pytest -q` / `uv run ruff check imagelab
  examples tests`.
- New experiment = new directory via `lab new task <name>`; experiments never require
  editing `imagelab/`. Framework changes are API changes — hold them to
  CONTRIBUTING.md's bar (seam stability, tests, a strong reason).
- Full training runs are expensive: stop at the cheapest tier that answers the current
  question, and ask before launching anything beyond `--tier fast`.

"""
`lab` — the experimentation CLI. Queries over runs/ (the filesystem is the database).

    lab run config.yaml --tier overfit                       # train (any trainer flag works)
    lab runs                                                 # ledger table
    lab compare 0610-1432 0609                               # config diff + metric diff
    lab board --metric val/loss                              # leaderboard
    lab gallery                                              # static HTML: grids + curves
    lab report 0610-1432 0609 --format md                    # publication-ready table
    lab sweep config.yaml --grid train.lr=1e-4,3e-4          # grid of runs
    lab check task my_idea                                   # contract probes (~seconds)
    lab new task my_idea                                     # scaffold a runnable project
    lab cite config.yaml                                     # papers behind a config

Display logic (key metric columns, leaderboard direction) comes from each ledger row —
tasks declare key_metrics/higher_is_better and the trainer stamps them into the row, so
none of these commands import model code.
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import yaml

from imagelab.lab.criteria import is_higher_better, key_metrics_of
from imagelab.lab.rundir import RUNS_ROOT, ledger_rows


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v) if v is not None else ""


def table(rows: list[dict], cols: list[str], headers: dict | None = None) -> str:
    headers = headers or {}
    head = [headers.get(c, c) for c in cols]
    cells = [[fmt(r.get(c, "")) for c in cols] for r in rows]
    widths = [max(len(h), *(len(row[i]) for row in cells)) if cells else len(h)
              for i, h in enumerate(head)]
    lines = ["  ".join(h.ljust(w) for h, w in zip(head, widths)),
             "  ".join("-" * w for w in widths)]
    lines += ["  ".join(c.ljust(w) for c, w in zip(row, widths)) for row in cells]
    return "\n".join(lines)


def metric_short(k: str) -> str:
    return k.removeprefix("val/").removeprefix("eval/").replace("combined/", "")


def find_run(query: str, rows: list[dict] | None = None) -> dict:
    rows = rows if rows is not None else ledger_rows()
    exact = [r for r in rows if r.get("run_id") == query]
    if exact:
        return exact[0]
    hits = [r for r in rows if query in r.get("run_id", "")]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise SystemExit(f"no run matching {query!r} (try `lab runs`)")
    raise SystemExit(f"{query!r} is ambiguous: {[r['run_id'] for r in hits]}")


def run_dir_of(row: dict) -> Path:
    return RUNS_ROOT / row["run_id"]


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def cmd_run(rest: list[str]) -> int:
    from imagelab.trainer import run as trainer_run
    if rest and not rest[0].startswith("-"):
        rest = ["--config", rest[0]] + rest[1:]
    return 0 if trainer_run(argv=rest) is not None else 1


def cmd_runs(args) -> int:
    rows = ledger_rows()
    if args.task:
        rows = [r for r in rows if r.get("task") == args.task]
    if not rows:
        print("no runs yet — `lab run <config> --tier smoke` to make one")
        return 0
    view = []
    for r in reversed(rows):  # newest first
        keym = key_metrics_of(r)
        view.append({
            "run": r.get("run_id"), "task": r.get("task"), "tier": r.get("tier"),
            "status": r.get("status"),
            "steps": r.get("steps", ""), "min": r.get("wall_min", ""),
            "seed": r.get("seed", ""),
            "result": "  ".join(f"{metric_short(k)} {fmt(v)}" for k, v in keym.items())
                      or (r.get("verdict") and f"overfit:{r['verdict']}") or "",
            "deltas": " ".join(f"{k.split('.')[-1]}={v}"
                               for k, v in (r.get("deltas") or {}).items())[:40],
        })
        if r.get("verdict") and view[-1]["result"]:
            view[-1]["result"] = f"overfit:{r['verdict']}  " + view[-1]["result"]
    print(table(view, ["run", "task", "tier", "status", "steps", "min", "seed",
                       "result", "deltas"]))
    return 0


def cmd_compare(args) -> int:
    rows = ledger_rows()
    a, b = find_run(args.a, rows), find_run(args.b, rows)
    ca = flatten(yaml.safe_load((run_dir_of(a) / "config.yaml").read_text()))
    cb = flatten(yaml.safe_load((run_dir_of(b) / "config.yaml").read_text()))
    diff = [{"key": k, a["run_id"]: ca.get(k, "—"), b["run_id"]: cb.get(k, "—")}
            for k in sorted(set(ca) | set(cb))
            if ca.get(k) != cb.get(k)
            and not k.startswith(("out.", "wandb."))]   # per-run bookkeeping, not the experiment
    print(f"== config delta ({len(diff)} key(s)) ==")
    print(table(diff, ["key", a["run_id"], b["run_id"]]) if diff
          else "  (identical resolved configs)")

    ma, mb = a.get("final_metrics") or {}, b.get("final_metrics") or {}
    lead = [k for k in (a.get("key_metrics") or []) + (b.get("key_metrics") or [])
            if k in ma and k in mb]
    shared = list(dict.fromkeys(lead))            # declared order, deduped
    shared += sorted((set(ma) & set(mb)) - set(shared))
    mrows = []
    for k in shared:
        va, vb = ma[k], mb[k]
        better = ""
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and va != vb:
            hib = is_higher_better(k, a, b)
            better = a["run_id"] if (va > vb) == hib else b["run_id"]
        mrows.append({"metric": k, a["run_id"]: va, b["run_id"]: vb,
                      "delta": vb - va if isinstance(va, (int, float)) else "",
                      "better": better})
    print(f"\n== metrics ({a.get('steps')} vs {b.get('steps')} steps) ==")
    print(table(mrows, ["metric", a["run_id"], b["run_id"], "delta", "better"])
          if mrows else "  (no shared final metrics)")
    if a.get("steps") != b.get("steps"):
        print("\nNOTE: step counts differ — this is not a compute-matched comparison.")
    if a.get("seed") != b.get("seed"):
        print("NOTE: seeds differ — at this scale, metric deltas of a few % can be noise.")
    if a.get("task") != b.get("task"):
        print(f"NOTE: different tasks ({a.get('task')} vs {b.get('task')}) — shared "
              f"metrics must mean the same thing for this to be a fair comparison.")
    return 0


def cmd_board(args) -> int:
    rows = [r for r in ledger_rows() if r.get("status") == "done"
            and (not args.task or r.get("task") == args.task)]
    metric = args.metric
    if not metric:
        # first key metric (declared, else heuristic) of any completed run, newest first
        for r in reversed(rows):
            keys = list(key_metrics_of(r, n=1))
            if keys:
                metric = keys[0]
                break
    scored = [r for r in rows if metric and metric in (r.get("final_metrics") or {})]
    if not scored:
        print(f"no completed runs with metric {metric!r}")
        return 0
    hib = is_higher_better(metric, *scored)
    scored.sort(key=lambda r: r["final_metrics"][metric], reverse=hib)
    view = [{"#": i + 1, "run": r["run_id"], metric: r["final_metrics"][metric],
             "task": r.get("task"), "tier": r.get("tier"), "steps": r.get("steps"),
             "seed": r.get("seed"),
             "deltas": " ".join(f"{k.split('.')[-1]}={v}"
                                for k, v in (r.get("deltas") or {}).items())[:40]}
            for i, r in enumerate(scored)]
    print(f"leaderboard by {metric} ({'higher' if hib else 'lower'} is better)\n")
    print(table(view, ["#", "run", metric, "task", "tier", "steps", "seed", "deltas"]))
    return 0


def cmd_report(args) -> int:
    rows = ledger_rows()
    chosen = [find_run(q, rows) for q in args.runs] if args.runs else \
             [r for r in rows if r.get("status") == "done"]
    if not chosen:
        print("nothing to report")
        return 0
    metrics = list(dict.fromkeys(
        k for r in chosen for k in r.get("key_metrics") or []
        if k in (r.get("final_metrics") or {})))
    if args.format == "latex":
        cols = " & ".join(["run", "steps"] + [metric_short(m) for m in metrics])
        print("\\begin{tabular}{l" + "r" * (len(metrics) + 1) + "}\n\\toprule")
        print(cols + r" \\ \midrule")
        for r in chosen:
            fm = r.get("final_metrics") or {}
            cells = [r["run_id"].replace("_", r"\_"), str(r.get("steps", ""))]
            cells += [fmt(fm.get(m, "")) for m in metrics]
            print(" & ".join(cells) + r" \\")
        print("\\bottomrule\n\\end{tabular}")
    else:
        head = ["run", "steps"] + [metric_short(m) for m in metrics]
        print("| " + " | ".join(head) + " |")
        print("|" + "|".join("---" for _ in head) + "|")
        for r in chosen:
            fm = r.get("final_metrics") or {}
            cells = [r["run_id"], str(r.get("steps", ""))] + [fmt(fm.get(m, "")) for m in metrics]
            print("| " + " | ".join(cells) + " |")
    return 0


def cmd_sweep(args) -> int:
    grids = {}
    for spec in args.grid:
        key, sep, vals = spec.partition("=")
        if not sep:
            raise SystemExit(f"--grid expects key=v1,v2,..., got {spec!r}")
        grids[key] = vals.split(",")
    combos = [dict(zip(grids, vs)) for vs in itertools.product(*grids.values())]
    before = {r["run_id"] for r in ledger_rows()}
    print(f"sweep: {len(combos)} runs (tier={args.tier})")
    for i, combo in enumerate(combos):
        sets = [f"{k}={v}" for k, v in combo.items()]
        print(f"\n--- sweep {i + 1}/{len(combos)}: {' '.join(sets)} ---")
        cmd = [sys.executable, "-m", "imagelab.trainer", "--config", args.config,
               "--tier", args.tier, "--set", *sets]
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"run {i + 1} failed (exit {r.returncode}) — continuing")
    new = [r for r in ledger_rows() if r["run_id"] not in before]
    if new:
        print("\n== sweep results ==")
        view = [{"run": r["run_id"], "status": r.get("status"),
                 **{metric_short(k): v for k, v in key_metrics_of(r).items()}}
                for r in new]
        cols = sorted({c for v in view for c in v} - {"run", "status"})
        print(table(view, ["run", "status"] + cols))
    return 0


def cmd_cite(args) -> int:
    from imagelab.registry import meta as reg_meta
    from imagelab.trainer import project_imports, resolve_task
    project_imports()
    cfg = yaml.safe_load(Path(args.config).read_text())
    spec = cfg.get("task")
    if not spec:
        print(f"{args.config} has no `task:` key")
        return 1
    cls = resolve_task(str(spec), Path(args.config).resolve().parent)
    print(f"methods used by {args.config}:")
    for kind, name in cls.cite_components(cfg):
        paper = reg_meta(kind, name).get("paper")
        if kind == "task" and not paper:
            paper = cls.paper                  # unregistered (file-path) tasks declare it
        print(f"  {kind:10s} {name:14s} {paper or '(no reference registered)'}")
    return 0


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # `lab run` forwards everything after it to the trainer's own argparse.
    if argv and argv[0] == "run":
        return cmd_run(argv[1:])

    ap = argparse.ArgumentParser(prog="lab", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Registered for --help only; `lab run ...` is short-circuited above so every
    # trainer flag (including task-declared ones) forwards untouched.
    p = sub.add_parser("run", help="train a config: lab run <config> [--tier ...] "
                                   "(all trainer flags forwarded)")
    p.add_argument("trainer_args", nargs=argparse.REMAINDER)
    p = sub.add_parser("runs", help="ledger table of all runs")
    p.add_argument("--task", default="")
    p = sub.add_parser("compare", help="config diff + metric diff between two runs")
    p.add_argument("a")
    p.add_argument("b")
    p = sub.add_parser("board", help="leaderboard over completed runs")
    p.add_argument("--metric", default="")
    p.add_argument("--task", default="")
    p = sub.add_parser("gallery", help="static HTML: every run's grids + curves")
    p.add_argument("--out", default="")
    p = sub.add_parser("report", help="markdown/latex results table")
    p.add_argument("runs", nargs="*")
    p.add_argument("--format", default="md", choices=["md", "latex"])
    p = sub.add_parser("sweep", help="grid of runs over --set values")
    p.add_argument("config")
    p.add_argument("--grid", nargs="+", required=True, metavar="KEY=V1,V2")
    p.add_argument("--tier", default="fast")
    p = sub.add_parser("check", help="contract probes for a component (~seconds)")
    p.add_argument("kind")
    p.add_argument("name")
    p.add_argument("--kwargs", default="", help="YAML dict of constructor kwargs")
    p = sub.add_parser("new", help="scaffold a component / a runnable task project")
    p.add_argument("kind")
    p.add_argument("name")
    p = sub.add_parser("cite", help="paper references for a config's components")
    p.add_argument("config")
    args = ap.parse_args(argv)

    if args.cmd == "runs":
        return cmd_runs(args)
    if args.cmd == "compare":
        return cmd_compare(args)
    if args.cmd == "board":
        return cmd_board(args)
    if args.cmd == "gallery":
        from imagelab.lab.gallery import build_gallery
        out = build_gallery(Path(args.out) if args.out else None)
        print(f"wrote {out} — open it in a browser")
        return 0
    if args.cmd == "report":
        return cmd_report(args)
    if args.cmd == "sweep":
        return cmd_sweep(args)
    if args.cmd == "check":
        from imagelab.lab.probes import run_checks
        from imagelab.trainer import project_imports
        project_imports()                      # load the project's checkers + components
        kw = yaml.safe_load(args.kwargs) if args.kwargs else {}
        return run_checks(args.kind, args.name, kw or {})
    if args.cmd == "new":
        from imagelab.lab.scaffold import scaffold
        from imagelab.trainer import project_imports
        project_imports()                      # load the project's scaffolds
        return scaffold(args.kind, args.name)
    if args.cmd == "cite":
        return cmd_cite(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""
`lab gallery` — one static HTML page over runs/: every run's sample grids, learning
curves, key metrics, and deltas, newest first. No server, no matplotlib — curves are
inline SVG polylines built from each run's metrics.jsonl, and image grids are referenced
relative to the page so it works from a file:// open or a repo checkout.

Image models get judged by eyes as much as numbers; because every run's fixed eval batch
is the same val images (see tasks/*.py), the grids here are directly comparable.
"""

from __future__ import annotations

import html
import json
import re
from collections import defaultdict
from pathlib import Path

from cvq.lab.criteria import key_metrics_of
from cvq.lab.rundir import RUNS_ROOT, ledger_rows

# Series never worth a curve in the overview (bookkeeping, not learning signal).
_SKIP = ("opt/", "sys/", "train/epoch", "train/img_per_s", "train/ar_on",
         "train/c_keep", "step", "t")
_MAX_CURVES = 6
_MAX_IMAGES = 6


def _series(jsonl: Path) -> dict[str, list[tuple[int, float]]]:
    out = defaultdict(list)
    if not jsonl.exists():
        return out
    for line in jsonl.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        s = row.get("step", 0)
        for k, v in row.items():
            if isinstance(v, (int, float)) and not any(k.startswith(p) or k == p
                                                       for p in _SKIP):
                out[k].append((s, float(v)))
    return out


def _sparkline(name: str, pts: list[tuple[int, float]], w=240, h=64) -> str:
    if len(pts) < 2:
        return ""
    xs, ys = zip(*pts)
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    sx = (w - 8) / max(1, x1 - x0)
    sy = (h - 20) / max(1e-12, y1 - y0)
    path = " ".join(f"{4 + (x - x0) * sx:.1f},{h - 14 - (y - y0) * sy:.1f}" for x, y in pts)
    last = ys[-1]
    return (f'<div class="spark"><svg width="{w}" height="{h}">'
            f'<polyline points="{path}" fill="none" stroke="#7aa2f7" stroke-width="1.5"/>'
            f'<text x="4" y="{h - 2}" class="lbl">{html.escape(name)}</text>'
            f'<text x="{w - 4}" y="{h - 2}" class="lbl val" text-anchor="end">'
            f'{last:.4g}</text></svg></div>')


def _latest_grids(sample_dir: Path) -> list[Path]:
    """Latest image per grid family (recon_all_*, gen_imagenette_*, ...)."""
    fams = {}
    for p in sorted(sample_dir.glob("*.png")):
        fam = re.sub(r"_?\d+\.png$", "", p.name)
        fams[fam] = p  # sorted() => last one (highest step) wins
    return list(fams.values())[:_MAX_IMAGES]


def build_gallery(out: Path | None = None) -> Path:
    out = out or RUNS_ROOT / "gallery.html"
    rows = ledger_rows()
    secs = []
    for r in reversed(rows):
        rid = r.get("run_id", "?")
        rdir = RUNS_ROOT / rid
        head = (f'<span class="id">{html.escape(rid)}</span> '
                f'<span class="tag">{r.get("task", "")}</span> '
                f'<span class="tag">{r.get("tier", "")}</span> '
                f'<span class="tag st-{r.get("status", "")}">{r.get("status", "")}</span>')
        if r.get("verdict"):
            head += f' <span class="tag st-{ "done" if r["verdict"] == "pass" else "crashed"}">overfit:{r["verdict"]}</span>'
        facts = []
        if r.get("deltas"):
            facts.append("Δ " + " ".join(f"{k}={v}" for k, v in r["deltas"].items()))
        for k, v in key_metrics_of(r.get("final_metrics") or {}, n=4).items():
            facts.append(f"{k} <b>{v:.4g}</b>" if isinstance(v, (int, float)) else f"{k} {v}")
        facts.append(f"steps {r.get('steps', '?')} · {r.get('wall_min', '?')} min "
                     f"· seed {r.get('seed', '?')}")
        if r.get("params"):
            total = sum(v for k, v in r["params"].items() if "." not in k)
            facts.append(f"params {total:.1f}M")

        sparks = "".join(_sparkline(k, pts) for k, pts in
                         sorted(_series(rdir / "metrics.jsonl").items(),
                                key=lambda kv: -len(kv[1]))[:_MAX_CURVES])
        imgs = "".join(
            f'<a href="{rid}/samples/{p.name}"><img src="{rid}/samples/{p.name}" '
            f'title="{p.name}"/></a>'
            for p in (_latest_grids(rdir / "samples") if (rdir / "samples").is_dir() else []))
        secs.append(f'<section><h2>{head}</h2><p class="facts">{" · ".join(facts)}</p>'
                    f'<div class="sparks">{sparks}</div>'
                    f'<div class="imgs">{imgs}</div></section>')

    page = f"""<!doctype html><meta charset="utf-8"><title>ImageLab runs</title>
<style>
  body {{ background:#16161e; color:#c0caf5; font:14px/1.5 ui-monospace,monospace;
          max-width:1100px; margin:2rem auto; padding:0 1rem; }}
  h1 {{ font-size:1.2rem; }} h2 {{ font-size:1rem; margin:0 0 .2rem; }}
  section {{ border-top:1px solid #2f334d; padding:1rem 0; }}
  .id {{ color:#e0af68; }} .facts {{ color:#9aa5ce; margin:.2rem 0 .6rem; }}
  .tag {{ background:#2f334d; border-radius:4px; padding:0 .4em; font-size:.85em; }}
  .st-done {{ background:#1f3d2e; color:#9ece6a; }}
  .st-crashed,.st-killed {{ background:#4d1f2f; color:#f7768e; }}
  .st-running {{ background:#3d3d1f; color:#e0af68; }}
  .sparks,.imgs {{ display:flex; flex-wrap:wrap; gap:.6rem; }}
  .spark svg {{ background:#1a1b26; border-radius:6px; }}
  .lbl {{ fill:#565f89; font-size:10px; }} .val {{ fill:#9ece6a; }}
  .imgs img {{ max-height:140px; image-rendering:pixelated; border-radius:6px; }}
</style>
<h1>ImageLab — {len(rows)} runs</h1>
{"".join(secs) or "<p>no runs yet</p>"}"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return out

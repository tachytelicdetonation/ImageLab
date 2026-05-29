"""
Download the full Pokemon roster (including variant forms) from PokeAPI and build
an image dataset for CVQ training.

Pipeline:
  1. Enumerate every entry from /pokemon (this list already includes variant forms
     such as 'charizard-mega-x', 'deoxys-attack', regional forms, etc.).
  2. For each entry, fetch its detail JSON (cached to disk) and pull the
     'official-artwork' front sprite URL.
  3. Download the (transparent) PNG, composite it onto a white background, resize
     to a square `--size`, and save it.
  4. Write a manifest.jsonl with one record per image: {file, id, name, caption}.

PokeAPI is a free, rate-limit-friendly API; we cache every JSON + image so re-runs
are cheap and we stay polite. Network access is the only external dependency.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

POKEAPI_INDEX = "https://pokeapi.co/api/v2/pokemon?limit=100000&offset=0"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "cvq-pokemon-dataset/0.1 (research; local)"})


# ---------------------------------------------------------------------------
# Caption construction
# ---------------------------------------------------------------------------
# This is a genuine design decision that shapes what the CAR text-to-image model
# (phase 2) learns to condition on. PokeAPI names look like:
#     'pikachu', 'charizard-mega-x', 'deoxys-attack', 'tauros-paldea-aqua-breed'
# A caption is what we will eventually feed the generator as text. Options:
#   (a) raw slug              -> "charizard-mega-x"
#   (b) hyphens -> spaces     -> "charizard mega x"
#   (c) base name only        -> "charizard"      (drops form info)
#   (d) templated             -> "a pixel art of charizard mega x"
# Trade-offs: keeping the form (b/d) lets the model learn distinct variants but
# spreads ~1300 names thin; collapsing to base (c) gives more images per caption
# but throws away the variant signal the dataset was built to capture.
def make_caption(name: str) -> str:
    """Turn a PokeAPI name slug into the text caption stored for each image.

    DEFAULT POLICY = (b): replace hyphens with spaces, so the variant form is kept
    as natural-ish text: "charizard-mega-x" -> "charizard mega x". This preserves the
    variant signal the dataset was built to capture.

    >>> This is YOUR knob. To collapse variants to the base name (policy c) for denser
        captions, return name.split("-")[0]. To use a template (policy d), return
        f"a drawing of {name.replace('-', ' ')}". Edit freely.
    """
    return name.replace("-", " ").strip()


# ---------------------------------------------------------------------------
# Networking helpers (with on-disk caching)
# ---------------------------------------------------------------------------
def get_json(url: str, cache_dir: Path) -> dict:
    """GET a JSON resource, caching the parsed body under cache_dir by URL tail."""
    key = url.rstrip("/").split("/")[-1].split("?")[0] or "index"
    cache_file = cache_dir / f"{key}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except json.JSONDecodeError:
            cache_file.unlink(missing_ok=True)  # corrupt cache; refetch
    resp = SESSION.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    cache_file.write_text(json.dumps(data))
    return data


def fetch_index(cache_dir: Path) -> list[dict]:
    """Return the full list of {name, url} entries (includes all variant forms)."""
    data = get_json(POKEAPI_INDEX, cache_dir)
    return data["results"]


def extract_artwork_url(detail: dict) -> str | None:
    """Pull the official-artwork front_default, falling back to the basic sprite."""
    sprites = detail.get("sprites", {}) or {}
    other = sprites.get("other", {}) or {}
    art = (other.get("official-artwork", {}) or {}).get("front_default")
    if art:
        return art
    return sprites.get("front_default")  # fallback for forms lacking artwork


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------
def process_image(raw: bytes, size: int, bg: tuple[int, int, int]) -> Image.Image:
    """Composite a (possibly transparent) PNG onto a solid bg and resize square."""
    img = Image.open(io.BytesIO(raw)).convert("RGBA")
    background = Image.new("RGBA", img.size, bg + (255,))
    flat = Image.alpha_composite(background, img).convert("RGB")
    # Official artwork is already square (475x475); LANCZOS keeps edges clean.
    return flat.resize((size, size), Image.LANCZOS)


def download_entry(entry: dict, cache_dir: Path, img_dir: Path, size: int,
                   bg: tuple[int, int, int]) -> dict | None:
    """Fetch one Pokemon's detail + artwork, save the processed image, return its
    manifest record (or None if it has no usable sprite)."""
    name = entry["name"]
    try:
        detail = get_json(entry["url"], cache_dir)
    except requests.RequestException:
        return None
    pkid = detail.get("id")
    art_url = extract_artwork_url(detail)
    if not art_url:
        return None

    out_path = img_dir / f"{pkid:05d}_{name}.png"
    if not out_path.exists():
        raw_cache = cache_dir / "imgs" / f"{pkid}_{name}.png"
        if raw_cache.exists():
            raw = raw_cache.read_bytes()
        else:
            try:
                r = SESSION.get(art_url, timeout=60)
                r.raise_for_status()
                raw = r.content
            except requests.RequestException:
                return None
            raw_cache.parent.mkdir(parents=True, exist_ok=True)
            raw_cache.write_bytes(raw)
        try:
            process_image(raw, size, bg).save(out_path)
        except Exception:
            return None

    return {"file": out_path.name, "id": pkid, "name": name, "caption": make_caption(name)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Download all Pokemon (incl. variants) for CVQ.")
    ap.add_argument("--out", type=Path, default=Path("data"), help="dataset root")
    ap.add_argument("--size", type=int, default=256, help="output square resolution")
    ap.add_argument("--workers", type=int, default=8, help="concurrent downloads")
    ap.add_argument("--limit", type=int, default=0, help="cap entries (0 = all, for testing)")
    ap.add_argument("--bg", type=str, default="255,255,255", help="background RGB for transparency")
    args = ap.parse_args()

    bg = tuple(int(x) for x in args.bg.split(","))
    cache_dir = args.out / "cache"
    img_dir = args.out / f"images_{args.size}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching Pokemon index from PokeAPI ...", file=sys.stderr)
    index = fetch_index(cache_dir)
    if args.limit:
        index = index[: args.limit]
    print(f"  {len(index)} entries (including variant forms)", file=sys.stderr)

    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_entry, e, cache_dir, img_dir, args.size, bg): e
            for e in index
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="downloading"):
            rec = fut.result()
            if rec is not None:
                records.append(rec)

    records.sort(key=lambda r: r["id"])
    manifest = args.out / "manifest.jsonl"
    with manifest.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\nDone: {len(records)} images -> {img_dir}", file=sys.stderr)
    print(f"Manifest: {manifest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

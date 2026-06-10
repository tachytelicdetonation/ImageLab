"""
Build the blessed benchmark dataset: ImageNette (+ optionally ImageWoof) from fastai.

    python -m cvq.data.download_imagenette --size 64                 # imagenette only
    python -m cvq.data.download_imagenette --size 64 --which both    # + imagewoof tags

Why this is the default for public results: Apache-2.0 licensed (it's the fastai
imagenette repo's subset of ImageNet), small enough for casual runs, and the
imagenette/imagewoof pair gives a built-in easy-vs-hard axis — ten visually distinct
classes vs ten dog breeds — that the GroupedEvaluator splits automatically via the
`dataset` tag. The official train/val directory split is preserved in the manifest's
`split` field, so every run holds out the same images.

Output layout (the generic manifest format every dataset uses):

    <root>/images_<size>/<wnid>_<i>.png
    <root>/manifest.jsonl   # {"file", "name", "caption", "dataset", "split"}
"""

from __future__ import annotations

import argparse
import json
import tarfile
import urllib.request
from pathlib import Path

from PIL import Image
from tqdm import tqdm

URLS = {
    # 160px-shortest-side archives: plenty for our 64-128px training sizes, ~10x smaller
    # than the full-size ones.
    "imagenette": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
    "imagewoof": "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz",
}

# WNID -> human caption (the AR model's text conditioning).
CLASSES = {
    # imagenette: 10 visually distinct classes (the "easy" axis)
    "n01440764": "tench", "n02102040": "english springer spaniel",
    "n02979186": "cassette player", "n03000684": "chain saw",
    "n03028079": "church", "n03394916": "french horn",
    "n03417042": "garbage truck", "n03425413": "gas pump",
    "n03445777": "golf ball", "n03888257": "parachute",
    # imagewoof: 10 dog breeds (the "hard" axis — fine-grained)
    "n02086240": "shih tzu", "n02087394": "rhodesian ridgeback",
    "n02088364": "beagle", "n02089973": "english foxhound",
    "n02093754": "border terrier", "n02096294": "australian terrier",
    "n02099601": "golden retriever", "n02105641": "old english sheepdog",
    "n02111889": "samoyed", "n02115641": "dingo",
}


def fetch(url: str, dest: Path):
    if dest.exists():
        print(f"cached: {dest.name}")
        return
    print(f"downloading {url} -> {dest}")
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def center_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    return img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))


def build(which: str, root: Path, size: int, max_per_class: int) -> list[dict]:
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    tgz = cache / Path(URLS[which]).name
    fetch(URLS[which], tgz)
    extract_dir = cache / tgz.name.replace(".tgz", "")
    if not extract_dir.exists():
        print(f"extracting {tgz.name}")
        with tarfile.open(tgz) as tf:
            tf.extractall(cache, filter="data")

    img_dir = root / f"images_{size}"
    img_dir.mkdir(parents=True, exist_ok=True)
    records, per_class = [], {}
    srcs = sorted(extract_dir.glob("*/*/*.JPEG"))  # <split>/<wnid>/<file>
    for src in tqdm(srcs, desc=f"{which} -> {size}px"):
        split = "val" if src.parent.parent.name == "val" else "train"
        wnid = src.parent.name
        caption = CLASSES.get(wnid)
        if caption is None:
            continue
        k = (wnid, split)
        per_class[k] = per_class.get(k, 0) + 1
        if max_per_class and per_class[k] > max_per_class:
            continue
        out_name = f"{wnid}_{per_class[k]:05d}.png"
        out_path = img_dir / out_name
        if not out_path.exists():
            img = center_square(Image.open(src).convert("RGB"))
            img.resize((size, size), Image.LANCZOS).save(out_path)
        records.append({"file": out_name, "name": caption, "caption": caption,
                        "dataset": which, "split": split})
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/imagenette")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--which", default="imagenette",
                    choices=["imagenette", "imagewoof", "both"])
    ap.add_argument("--max-per-class", type=int, default=0,
                    help="cap images per class+split (0 = all; handy for quick subsets)")
    args = ap.parse_args()

    root = Path(args.root)
    sources = ["imagenette", "imagewoof"] if args.which == "both" else [args.which]
    records = []
    for which in sources:
        records += build(which, root, args.size, args.max_per_class)

    with open(root / "manifest.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    n_val = sum(r["split"] == "val" for r in records)
    print(f"\nwrote {len(records)} records ({len(records) - n_val} train / {n_val} val) "
          f"-> {root}/manifest.jsonl")
    print(f"train: python -m cvq.trainer --config configs/tok_inet_64.yaml")


if __name__ == "__main__":
    main()

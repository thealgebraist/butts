#!/usr/bin/env python3
"""Train a cigarette-butt presence classifier and measure the split-honesty gap.

Object *detection* cannot be trained on this data: none of the 331 manual
polygons is a cigarette butt, so no boxes exist for the primary class. What can
be trained from the labels that do exist is image-level classification —
"does this frame contain a cigarette butt?".

Two datasets are built from the same images:

  * ``location``: train/val split by GPS cluster, so no site appears in both.
  * ``random``:   an ordinary random split.

Photographs taken minutes apart at one site are near-duplicates, so a random
split leaks them across the boundary and inflates the score. Training both
quantifies that gap.

    python scripts/train_cigarette_classifier.py [--epochs 40]
"""

import argparse
import collections
import json
import random
import shutil
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
JPG = ROOT / "images" / "jpg"
WORK = ROOT / "analysis" / "cls_work"
OUT = ROOT / "analysis" / "cigarette_classifier.json"

POS = "cigarette_butt"
NEG = "dontknow"
VAL_FRACTION = 0.30
IMG_SIZE = 320  # classifier input; butts are large in these handheld frames


def gps(path):
    ex = Image.open(path).getexif()
    ifd = ex.get_ifd(34853) if ex else None
    if not ifd:
        return None
    try:
        def dec(v, ref):
            d = float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
            return -d if ref in ("S", "W") else d
        return round(dec(ifd[2], ifd[1]), 3), round(dec(ifd[4], ifd[3]), 3)
    except Exception:
        return None


def location_split(paths):
    """Hold out whole GPS clusters, so no site spans train and val."""
    clusters = collections.defaultdict(list)
    for p in paths:
        clusters[gps(p) or ("nogps", len(clusters))].append(p)
    # Largest clusters to train, fill val until it reaches the target size.
    order = sorted(clusters.values(), key=len, reverse=True)
    target = len(paths) * VAL_FRACTION
    val, train = [], []
    for group in reversed(order):           # smallest first into val
        (val if len(val) + len(group) <= target else train).extend(group)
    return train, val


def random_split(paths, seed=0):
    ps = list(paths)
    random.Random(seed).shuffle(ps)
    cut = int(len(ps) * VAL_FRACTION)
    return ps[cut:], ps[:cut]


def build(name, splitter):
    root = WORK / name
    if root.exists():
        shutil.rmtree(root)
    counts = {}
    for cls in (POS, NEG):
        paths = sorted((JPG / cls).glob("*.jpg"))
        train, val = splitter(paths)
        counts[cls] = {"train": len(train), "val": len(val)}
        for sub, group in (("train", train), ("val", val)):
            d = root / sub / cls
            d.mkdir(parents=True, exist_ok=True)
            for p in group:
                shutil.copy2(p, d / p.name)
    return root, counts


def evaluate(model, root):
    """Per-class recall on the val split, plus the majority-class baseline."""
    correct = collections.Counter()
    total = collections.Counter()
    for cls in (POS, NEG):
        for p in (root / "val" / cls).glob("*.jpg"):
            pred = model.predict(p, imgsz=IMG_SIZE, verbose=False, device="mps")[0]
            total[cls] += 1
            correct[cls] += model.names[int(pred.probs.top1)] == cls
    n = sum(total.values())
    acc = sum(correct.values()) / n
    majority = max(total.values()) / n
    return {
        "val_n": n,
        "accuracy": round(acc, 3),
        "majority_baseline": round(majority, 3),
        "recall": {c: round(correct[c] / total[c], 3) for c in total},
        "counts": {c: total[c] for c in total},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    results = {}
    for name, splitter in (("location", location_split), ("random", random_split)):
        print(f"\n{'='*60}\n{name} split\n{'='*60}")
        root, counts = build(name, splitter)
        print(json.dumps(counts))
        model = YOLO("yolo11s-cls.pt")
        model.train(data=str(root), epochs=args.epochs, imgsz=IMG_SIZE,
                    device="mps", project=str(WORK / "runs"), name=name,
                    exist_ok=True, verbose=False, plots=False)
        results[name] = {"counts": counts, **evaluate(model, root)}
        print(json.dumps(results[name], indent=1))

    OUT.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {OUT.relative_to(ROOT)}")

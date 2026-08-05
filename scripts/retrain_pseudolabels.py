#!/usr/bin/env python3
"""Retrain the butt classifier on its own discoveries, verified vs unverified.

The first classifier flagged 22 images outside images/jpg/cigarette_butt as
containing a butt. Visual inspection found only two of them to be real. This
script trains three models against an identical held-out validation set:

    baseline   original labels only
    verified   + the 2 hand-confirmed discoveries
    all_found  + all 22 discoveries, taken on trust (naive pseudo-labelling)

The validation split is built first and frozen, and any discovered image that
falls inside it is skipped, so all three models are scored on the same data
with uncorrupted labels.
"""

import collections
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
JPG = ROOT / "images" / "jpg"
WORK = ROOT / "analysis" / "pseudo_work"
SCAN = ROOT / "analysis" / "pseudo_candidates.json"
OUT = ROOT / "analysis" / "pseudo_retrain.json"

POS, NEG = "cigarette_butt", "dontknow"
VAL_FRACTION = 0.30
IMG_SIZE, EPOCHS = 320, 30

# Confirmed by eye to contain a real cigarette butt.
VERIFIED = ["dontknow/IMG_2151.jpg", "plastic_piece/IMG_2728.jpg"]


def gps(path):
    from PIL import Image
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
    clusters = collections.defaultdict(list)
    for p in paths:
        clusters[gps(p) or ("nogps", len(clusters))].append(p)
    order = sorted(clusters.values(), key=len, reverse=True)
    target = len(paths) * VAL_FRACTION
    val, train = [], []
    for g in reversed(order):
        (val if len(val) + len(g) <= target else train).extend(g)
    return train, val


def build(name, extra_positives):
    """Same frozen val split every time; extras only ever enter train."""
    root = WORK / name
    if root.exists():
        shutil.rmtree(root)
    splits = {c: location_split(sorted((JPG / c).glob("*.jpg"))) for c in (POS, NEG)}
    val_names = {p.name for c in splits for p in splits[c][1]}

    for cls in (POS, NEG):
        train, val = splits[cls]
        for sub, group in (("train", train), ("val", val)):
            d = root / sub / cls
            d.mkdir(parents=True, exist_ok=True)
            for p in group:
                shutil.copy2(p, d / p.name)

    added, skipped = 0, 0
    for rel in extra_positives:
        src = JPG / rel
        if not src.exists() or src.name in val_names:
            skipped += 1          # never let a pseudo-label touch validation
            continue
        # A discovery from dontknow/ is currently a training negative: flip it.
        stale = root / "train" / NEG / src.name
        if stale.exists():
            stale.unlink()
        shutil.copy2(src, root / "train" / POS / src.name)
        added += 1
    return root, added, skipped


def evaluate(model, root):
    correct, total = collections.Counter(), collections.Counter()
    for cls in (POS, NEG):
        for p in (root / "val" / cls).glob("*.jpg"):
            r = model.predict(p, imgsz=IMG_SIZE, verbose=False, device="mps")[0]
            total[cls] += 1
            correct[cls] += model.names[int(r.probs.top1)] == cls
    n = sum(total.values())
    return {"val_n": n, "accuracy": round(sum(correct.values()) / n, 3),
            "majority_baseline": round(max(total.values()) / n, 3),
            "recall": {c: round(correct[c] / total[c], 3) for c in total}}


if __name__ == "__main__":
    found = [r["rel"] for r in json.loads(SCAN.read_text())]
    variants = {"baseline": [], "verified": VERIFIED, "all_found": found}

    results = {}
    for name, extra in variants.items():
        print(f"\n{'='*60}\n{name}: {len(extra)} candidate positives\n{'='*60}")
        root, added, skipped = build(name, extra)
        model = YOLO("yolo11s-cls.pt")
        model.train(data=str(root), epochs=EPOCHS, imgsz=IMG_SIZE, device="mps",
                    project=str(WORK / "runs"), name=name, exist_ok=True,
                    verbose=False, plots=False)
        results[name] = {"added": added, "skipped_in_val": skipped, **evaluate(model, root)}
        print(json.dumps(results[name], indent=1))

    OUT.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {OUT.relative_to(ROOT)}")

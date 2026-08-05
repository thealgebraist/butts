#!/usr/bin/env python3
"""Train a cigarette-butt detector on the manual polygon annotations.

The annotator tool (tools/annotator/src/ImageLoader.h) reads the EXIF
orientation tag but never applies it, so saved polygon coordinates are in a
rotated frame relative to the stored pixel buffer. This script corrects them
per image from each file's EXIF orientation before training:

    orientation 3 (180 deg) -> (x, y) -> (W-x, H-y)
    orientation 6  (90 deg) -> (x, y) -> (W-y, x)
    orientation 8 (270 deg) -> (x, y) -> (y, H-x)

Verified by measuring filter-orange coverage inside each polygon: the
correction lifts the mean from 0.04 to 0.24 and puts boxes visibly on target.

Train/val is split by GPS cluster so no location appears in both halves;
photos taken minutes apart at one site are near-duplicates and a random split
inflates the score badly (measured at +12 points on the earlier classifier).

    python scripts/train_detector.py [--epochs 100]
"""

import argparse
import collections
import json
import shutil
from pathlib import Path

import cv2
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
JPG = ROOT / "images" / "jpg"
HEIC = ROOT / "images" / "heic"
WORK = ROOT / "analysis" / "det_work"
OUT = ROOT / "analysis" / "detector_results.json"

BUTT = {"cigarette_butt", "butt", "old_butt"}
PACK = {"cigarette_packing", "pack", "cigarette_pack"}
NAMES = ["cigarette_butt", "cigarette_pack"]
VAL_FRACTION = 0.30
N_NEGATIVES = 60


def orient_fix(x, y, ori, W, H):
    if ori == 3:
        return W - x, H - y
    if ori == 6:
        return W - y, x
    if ori == 8:
        return y, H - x
    return x, y


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


def collect():
    """Map each annotated image to its corrected YOLO boxes."""
    jpgs = {p.stem: p for p in JPG.rglob("*.jpg")}
    per_image = collections.defaultdict(list)
    for f in HEIC.rglob("*_annot.json"):
        key = f.name.replace(".HEIC_annot.json", "").replace(".heic_annot.json", "")
        if key not in jpgs:
            continue
        try:
            entries = json.load(f.open())
        except json.JSONDecodeError:
            continue
        img = jpgs[key]
        for a in entries:
            label, poly = a.get("label"), a.get("polygon")
            cls = 0 if label in BUTT else (1 if label in PACK else None)
            if cls is None or not poly or len(poly) < 3:
                continue
            im = cv2.imread(str(img))
            if im is None:
                continue
            H, W = im.shape[:2]
            ori = Image.open(img).getexif().get(274, 1)
            pts = [orient_fix(p["x"], p["y"], ori, W, H) for p in poly]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x0, x1 = max(0, min(xs)), min(W, max(xs))
            y0, y1 = max(0, min(ys)), min(H, max(ys))
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            per_image[img].append(
                (cls, ((x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H))
            )
    return per_image


def split_by_location(paths):
    clusters = collections.defaultdict(list)
    for p in paths:
        clusters[gps(p) or ("nogps", len(clusters))].append(p)
    order = sorted(clusters.values(), key=len, reverse=True)
    target = len(paths) * VAL_FRACTION
    val, train = [], []
    for g in reversed(order):
        (val if len(val) + len(g) <= target else train).extend(g)
    return train, val


def build(per_image):
    if WORK.exists():
        shutil.rmtree(WORK)
    pos = sorted(per_image, key=str)
    train, val = split_by_location(pos)

    # Background images (no annotations) teach the model what is NOT a butt.
    negs = [p for p in sorted((JPG / "dontknow").glob("*.jpg")) if p not in per_image]
    ntr, nva = split_by_location(negs[:N_NEGATIVES])

    counts = {}
    for sub, imgs, negs_sub in (("train", train, ntr), ("val", val, nva)):
        (WORK / "images" / sub).mkdir(parents=True, exist_ok=True)
        (WORK / "labels" / sub).mkdir(parents=True, exist_ok=True)
        for p in imgs + negs_sub:
            shutil.copy2(p, WORK / "images" / sub / p.name)
            lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                     for c, (x, y, w, h) in per_image.get(p, [])]
            (WORK / "labels" / sub / f"{p.stem}.txt").write_text("\n".join(lines))
        counts[sub] = {"annotated": len(imgs), "background": len(negs_sub),
                       "boxes": sum(len(per_image.get(p, [])) for p in imgs)}

    (WORK / "data.yaml").write_text(
        f"path: {WORK}\ntrain: images/train\nval: images/val\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES))
    )
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    # Defaults are deliberately memory-lean: 960px at batch 16 exhausts RAM on
    # a laptop. Butts span 100-700 px in these frames, so 640 keeps them well
    # above the detector's ~20 px floor.
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    per_image = collect()
    boxes = sum(len(v) for v in per_image.values())
    print(f"{len(per_image)} annotated images, {boxes} boxes")
    counts = build(per_image)
    print(json.dumps(counts, indent=1))

    model = YOLO(args.model)
    model.train(data=str(WORK / "data.yaml"), epochs=args.epochs, imgsz=args.imgsz,
                device="mps", project=str(WORK / "runs"), name="det", exist_ok=True,
                verbose=False, plots=False, seed=0, batch=args.batch,
                workers=args.workers, cache=False, val=True)
    m = model.val(data=str(WORK / "data.yaml"), imgsz=args.imgsz, device="mps",
                  batch=args.batch, workers=args.workers, verbose=False)

    res = {"counts": counts, "boxes": boxes,
           "mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4),
           "precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4),
           "per_class": {NAMES[i]: {"mAP50": round(float(v), 4)}
                         for i, v in enumerate(m.box.maps if len(m.box.maps) == len(NAMES) else [])}}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))

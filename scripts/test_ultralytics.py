#!/usr/bin/env python3
"""Baseline the image set with Ultralytics YOLO — closed- and open-vocabulary.

Runs two models over images/jpg:

  * a COCO-pretrained detector (80 fixed classes, no litter classes as such),
  * YOLO-World, an open-vocabulary detector prompted with this project's own
    class names including "cigarette butt".

Both are run at two input sizes, because input resolution is the dominant
variable for objects this small.

    python scripts/test_ultralytics.py            # run everything
    python scripts/test_ultralytics.py --limit 20 # quick smoke test

Results are written to analysis/ultralytics_results.json.
"""

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
JPG = ROOT / "images" / "jpg"
OUT = ROOT / "analysis" / "ultralytics_results.json"

# Open-vocabulary prompts. Kept concrete and visual: YOLO-World grounds text,
# so "cigarette butt" works better than an abstract label like "litter".
PROMPTS = [
    "cigarette butt", "cigarette", "plastic bottle", "bottle", "can",
    "plastic bag", "wrapper", "food wrapper", "paper", "paper cup",
    "drinking straw", "bottle cap", "glove", "cloth", "shoe",
    "chair", "cardboard", "trash", "litter",
]

# COCO classes that plausibly correspond to litter in these photos.
COCO_RELEVANT = {"bottle", "cup", "chair", "bowl", "handbag", "backpack",
                 "book", "cell phone", "remote", "scissors", "toothbrush",
                 "sports ball", "frisbee", "vase", "banana", "orange"}


def image_paths(limit=None):
    ps = sorted(p for p in JPG.rglob("*.jpg"))
    return ps[:limit] if limit else ps


def run(model_name, paths, imgsz, prompts=None, conf=0.15):
    model = YOLO(model_name)
    if prompts:
        model.set_classes(prompts)
    names = model.names
    rows = {}
    t0 = time.time()
    for i, p in enumerate(paths, 1):
        r = model.predict(p, imgsz=imgsz, conf=conf, verbose=False, device="mps")[0]
        dets = []
        for b in r.boxes:
            cls = names[int(b.cls)]
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            dets.append({"cls": cls, "conf": round(float(b.conf), 3),
                         "px": round(max(x2 - x1, y2 - y1), 1)})
        rows[str(p.relative_to(JPG))] = dets
        if i % 50 == 0 or i == len(paths):
            print(f"    {i}/{len(paths)}  ({time.time()-t0:.0f}s)", flush=True)
    return rows, time.time() - t0


def summarize(tag, rows):
    n_any = sum(1 for d in rows.values() if d)
    cls = Counter(x["cls"] for d in rows.values() for x in d)
    print(f"\n=== {tag}")
    print(f"  images with >=1 detection: {n_any}/{len(rows)} ({100*n_any/len(rows):.0f}%)")
    print(f"  total detections: {sum(len(d) for d in rows.values())}")
    for c, n in cls.most_common(12):
        print(f"    {n:5}  {c}")


def per_folder(rows, target_cls=None):
    """Detection rate per source folder, optionally for one class only."""
    agg = defaultdict(lambda: [0, 0])
    for rel, dets in rows.items():
        folder = rel.split("/")[0]
        agg[folder][0] += 1
        hit = any(d["cls"] == target_cls for d in dets) if target_cls else bool(dets)
        agg[folder][1] += hit
    return agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    paths = image_paths(args.limit)
    print(f"{len(paths)} images\n")
    results = {}

    for tag, model_name, imgsz, prompts in [
        ("coco_640",   "yolo11x.pt",        640,  None),
        ("coco_1280",  "yolo11x.pt",        1280, None),
        ("world_640",  "yolov8x-worldv2.pt", 640,  PROMPTS),
        ("world_1280", "yolov8x-worldv2.pt", 1280, PROMPTS),
    ]:
        print(f"running {tag} ...")
        rows, secs = run(model_name, paths, imgsz, prompts)
        results[tag] = {"model": model_name, "imgsz": imgsz, "seconds": round(secs, 1),
                        "prompts": prompts, "rows": rows}
        summarize(tag, rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {OUT.relative_to(ROOT)}")

#!/usr/bin/env python3
"""Cut annotated objects out of their photos using the manual polygon masks.

Each images/heic/**/NAME.heic_annot.json holds a list of manual annotations:
{"label": ..., "bbox": {x,y,width,height}, "polygon": [{x,y}, ...]} in the
coordinate space of the full-resolution image.

For each selected annotation this writes analysis/crops/<label>.jpg: the region
inside the polygon, everything outside blacked out, cropped to the bounding box
(so the output's pixel size is exactly the bbox size).

    python scripts/extract_polygons.py [label ...]
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
HEIC = ROOT / "images" / "heic"
JPG = ROOT / "images" / "jpg"
OUT = ROOT / "analysis" / "crops"

DEFAULT_LABELS = [
    "cocio", "gloves", "plastic_bag", "chair",
    "paper", "dirty_cloth", "burgerking", "maoam",
]


def largest_per_label(labels: set[str]) -> dict:
    """Pick the biggest polygon for each requested label."""
    best: dict[str, tuple] = {}
    for annot in HEIC.rglob("*_annot.json"):
        jpg = JPG / annot.parent.name / annot.name.replace(".heic_annot.json", ".jpg")
        if not jpg.exists():
            continue
        try:
            entries = json.load(annot.open())
        except json.JSONDecodeError:
            continue
        for a in entries:
            label, bbox, poly = a.get("label"), a.get("bbox"), a.get("polygon")
            if label not in labels or not bbox or not poly or len(poly) < 3:
                continue
            area = bbox["width"] * bbox["height"]
            if label not in best or area > best[label][0]:
                best[label] = (area, jpg, a)
    return best


def extract(label: str, jpg: Path, a: dict) -> tuple[Path, tuple[int, int]]:
    im = Image.open(jpg).convert("RGB")
    b = a["bbox"]
    box = (
        max(0, int(b["x"])), max(0, int(b["y"])),
        min(im.width, int(b["x"] + b["width"])), min(im.height, int(b["y"] + b["height"])),
    )
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).polygon([(p["x"], p["y"]) for p in a["polygon"]], fill=255)
    masked = Image.new("RGB", im.size, (0, 0, 0))
    masked.paste(im, mask=mask)  # everything outside the polygon stays black
    crop = masked.crop(box)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{label}.jpg"
    crop.save(path, quality=92)
    return path, crop.size


if __name__ == "__main__":
    wanted = set(sys.argv[1:]) or set(DEFAULT_LABELS)
    found = largest_per_label(wanted)
    for label, (_, jpg, a) in sorted(found.items()):
        path, size = extract(label, jpg, a)
        print(f"{label:14}{str(size):14}<- {jpg.parent.name}/{jpg.name}")
    for missing in sorted(wanted - set(found)):
        print(f"{missing:14}no polygon found")

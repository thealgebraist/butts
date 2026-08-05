#!/usr/bin/env python3
"""Reclassify images into folders named after the vision model's primary_object.

Reads analysis/image_analysis.json and reorganizes both images/jpg and
images/heic (plus each image's _annot.json sibling) into <label>/ folders.
Images the model reported no waste item for go to dontknow/.

    python scripts/reclassify.py plan     # print the move plan, touch nothing
    python scripts/reclassify.py apply    # perform the moves with git mv

Every move is recorded in analysis/reclassify_manifest.json so the original
folder labels survive the reorganization and the move can be reversed.
"""

import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis" / "image_analysis.json"
MANIFEST = ROOT / "analysis" / "reclassify_manifest.json"
JPG = ROOT / "images" / "jpg"
HEIC = ROOT / "images" / "heic"
UNKNOWN = "dontknow"

# Collapse the model's free-text labels into one canonical folder per concept.
SYNONYMS = {
    "cigarette_butt": ["cigarette butt", "cigarette butts"],
    "cigarette_pack": ["cigarette pack"],
    "plastic_bottle": ["plastic bottle"],
    "bottle_cap": ["bottle cap", "plastic bottle cap"],
    "plastic_bag": ["plastic bag", "garbage bag"],
    "plastic_wrapper": ["plastic wrapper", "wrapper", "food wrapper"],
    "candy_wrapper": ["candy wrapper"],
    "plastic_straw": ["plastic straw"],
    "plastic_piece": [
        "plastic fragment",
        "piece of plastic",
        "piece of red plastic",
        "plastic sheet",
    ],
    "plastic_container": [
        "plastic container",
        "plastic bin",
        "plastic crate filled with metal waste",
        "basket",
    ],
    "can": ["aluminum can", "crushed can", "crumpled can"],
    "paper": ["crumpled paper", "piece of paper", "paper sheet", "paper fragment"],
    "paper_cup": ["paper cup"],
    "paper_bag": ["paper bag"],
    "tissue": ["tissue"],
    "textile": [
        "textile",
        "textile scrap",
        "textile cloth fragment",
        "piece of textile",
        "fabric piece",
        "cloth piece",
    ],
    "glove": ["glove", "plastic glove"],
    "chair": ["chair", "folding chair"],
    "shoe": ["shoe"],
    "rope": ["rope fragments"],
    "detergent_package": ["detergent package"],
    "pine_cone": ["pine cone"],
}
LOOKUP = {alias: canon for canon, aliases in SYNONYMS.items() for alias in aliases}


def label_for(entry: dict) -> str:
    raw = str(entry.get("primary_object") or "").strip().lower()
    if raw in ("", "none", "null"):
        return UNKNOWN
    if raw in LOOKUP:
        return LOOKUP[raw]
    # Unmapped label: fall back to a slug so nothing is silently dropped.
    return raw.replace(" ", "_").replace("/", "_")


def build_plan() -> list[dict]:
    results = json.loads(ANALYSIS.read_text())["results"]
    plan = []
    used: dict[tuple[str, str], int] = defaultdict(int)

    for rel in sorted(results):
        label = label_for(results[rel])
        src_jpg = JPG / rel
        stem = Path(rel).stem
        # Flattening can collide on basename; disambiguate with the old folder.
        name = stem
        if used[(label, stem)]:
            name = f"{Path(rel).parent.name}_{stem}"
        used[(label, stem)] += 1

        moves = []
        if src_jpg.exists():
            moves.append((src_jpg, JPG / label / f"{name}.jpg"))
        # HEIC original and its annotation sibling, if present.
        src_heic = HEIC / Path(rel).with_suffix(".heic")
        if src_heic.exists():
            moves.append((src_heic, HEIC / label / f"{name}.heic"))
            annot = src_heic.parent / (src_heic.name + "_annot.json")
            if annot.exists():
                moves.append((annot, HEIC / label / f"{name}.heic_annot.json"))

        plan.append({"rel": rel, "label": label, "original_folder": str(Path(rel).parent),
                     "moves": [(str(a.relative_to(ROOT)), str(b.relative_to(ROOT))) for a, b in moves]})
    return plan


def plan_cmd() -> None:
    plan = build_plan()
    counts = Counter(p["label"] for p in plan)
    print(f"{len(plan)} images -> {len(counts)} folders\n")
    for label, n in counts.most_common():
        print(f"  {n:4}  {label}")
    missing = [p["rel"] for p in plan if not p["moves"]]
    nheic = sum(1 for p in plan for a, _ in p["moves"] if a.endswith(".heic"))
    nannot = sum(1 for p in plan for a, _ in p["moves"] if a.endswith("_annot.json"))
    print(f"\nfiles: {sum(len(p['moves']) for p in plan)} total "
          f"({nheic} heic, {nannot} annotations)")
    if missing:
        print(f"WARNING: {len(missing)} entries had no files on disk")


def apply_cmd() -> None:
    plan = build_plan()
    for p in plan:
        for src, dst in p["moves"]:
            (ROOT / dst).parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "mv", src, dst], cwd=ROOT, check=True)
    MANIFEST.write_text(json.dumps(
        {"source": "analysis/image_analysis.json (gpt-4o, detail=low)",
         "note": "original_folder preserves the pre-reclassification human label",
         "entries": plan}, indent=2))
    # Drop directories the moves emptied out.
    subprocess.run(["find", str(JPG), str(HEIC), "-type", "d", "-empty", "-delete"], check=False)
    print(f"moved {sum(len(p['moves']) for p in plan)} files; manifest -> {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plan"
    {"plan": plan_cmd, "apply": apply_cmd}[cmd]()

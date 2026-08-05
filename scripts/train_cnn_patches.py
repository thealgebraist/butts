#!/usr/bin/env python3
"""A small from-scratch CNN that classifies image patches as butt / not-butt.

Complements the YOLO detector: instead of a pretrained detection head, this is
a plain convolutional network defined and trained from random initialisation in
PyTorch, on fixed-size crops.

  positives  crops centred on each manually annotated cigarette-butt polygon,
             jittered in position and scale for augmentation
  negatives  random crops from background images (no annotations at all) and
             from non-overlapping regions of the annotated images

Polygon coordinates are corrected for the annotator's EXIF-orientation bug
before use (see scripts/train_detector.py). Train/val is split by GPS cluster
*before* crops are cut, so no location appears on both sides.

    python scripts/train_cnn_patches.py [--epochs 30] [--patch 64]
"""

import argparse
import collections
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
JPG = ROOT / "images" / "jpg"
HEIC = ROOT / "images" / "heic"
OUT = ROOT / "analysis" / "cnn_patches_results.json"
CKPT = ROOT / "analysis" / "cnn_patches.pt"

BUTT = {"cigarette_butt", "butt", "old_butt"}
VAL_FRACTION = 0.30
POS_PER_BOX = 12          # jittered crops per annotated butt
NEG_PER_IMAGE = 14
SEED = 0


# ---------------------------------------------------------------- data

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


def load_boxes():
    """image path -> list of corrected (x0, y0, x1, y1) butt boxes."""
    jpgs = {p.stem: p for p in JPG.rglob("*.jpg")}
    out = collections.defaultdict(list)
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
            if a.get("label") not in BUTT or not a.get("polygon"):
                continue
            with Image.open(img) as im:
                W, H = im.size
                ori = im.getexif().get(274, 1)
            pts = [orient_fix(p["x"], p["y"], ori, W, H) for p in a["polygon"]]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            x0, y0 = max(0, min(xs)), max(0, min(ys))
            x1, y1 = min(W, max(xs)), min(H, max(ys))
            if x1 - x0 > 8 and y1 - y0 > 8:
                out[img].append((x0, y0, x1, y1))
    return out


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


def square(cx, cy, side, W, H):
    s = int(side)
    x0 = int(np.clip(cx - s / 2, 0, max(0, W - s)))
    y0 = int(np.clip(cy - s / 2, 0, max(0, H - s)))
    return x0, y0, min(W, x0 + s), min(H, y0 + s)


def crops_for(img_path, boxes, rng, patch):
    """Return (patches, labels) for one image."""
    im = cv2.imread(str(img_path))
    if im is None:
        return [], []
    H, W = im.shape[:2]
    xs, ys = [], []

    for (x0, y0, x1, y1) in boxes:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        base = max(x1 - x0, y1 - y0)
        for _ in range(POS_PER_BOX):
            side = base * rng.uniform(1.1, 2.2)          # scale jitter + context
            jx = cx + rng.uniform(-0.25, 0.25) * base    # position jitter
            jy = cy + rng.uniform(-0.25, 0.25) * base
            a, b, c, d = square(jx, jy, side, W, H)
            if c - a < 16 or d - b < 16:
                continue
            xs.append(cv2.resize(im[b:d, a:c], (patch, patch)))
            ys.append(1)

    # negatives: random crops that miss every annotated box
    sides = [max(x1 - x0, y1 - y0) for (x0, y0, x1, y1) in boxes] or [W * 0.06]
    for _ in range(NEG_PER_IMAGE):
        side = float(np.mean(sides)) * rng.uniform(1.1, 2.2)
        jx, jy = rng.uniform(0, W), rng.uniform(0, H)
        a, b, c, d = square(jx, jy, side, W, H)
        if c - a < 16 or d - b < 16:
            continue
        if any(not (c < x0 or a > x1 or d < y0 or b > y1) for (x0, y0, x1, y1) in boxes):
            continue                                       # overlaps a butt
        xs.append(cv2.resize(im[b:d, a:c], (patch, patch)))
        ys.append(0)
    return xs, ys


def build(patch):
    boxes = load_boxes()
    annotated = sorted(boxes, key=str)
    backgrounds = [p for p in sorted((JPG / "dontknow").glob("*.jpg")) if p not in boxes]
    tr_a, va_a = split_by_location(annotated)
    tr_b, va_b = split_by_location(backgrounds)

    data = {}
    for name, ann, bg in (("train", tr_a, tr_b), ("val", va_a, va_b)):
        rng = random.Random(SEED if name == "train" else SEED + 1)
        X, Y = [], []
        for p in ann:
            x, y = crops_for(p, boxes[p], rng, patch)
            X += x
            Y += y
        for p in bg:
            x, y = crops_for(p, [], rng, patch)
            X += x
            Y += y
        X = np.stack(X).astype(np.float32) / 255.0
        data[name] = (torch.from_numpy(X).permute(0, 3, 1, 2), torch.tensor(Y))
    # train-side sources are handed back so hard negatives can be mined from
    # exactly the images the model is allowed to see.
    data["mine_paths"] = tr_a + tr_b
    data["boxes"] = boxes
    return data


# ---------------------------------------------------------------- model

class ButtCNN(nn.Module):
    """Four conv blocks, global average pool, one hidden layer. ~200k params."""

    def __init__(self, ch=(32, 64, 96, 128)):
        super().__init__()
        layers, prev = [], 3
        for c in ch:
            layers += [nn.Conv2d(prev, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
                       nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
                       nn.MaxPool2d(2)]
            prev = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(prev, 64), nn.ReLU(),
                                  nn.Linear(64, 2))

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        return self.head(x)


def augment(xb, rng, colour=True):
    """Geometric + photometric augmentation.

    The colour part matters more than usual here. Trained without it, the
    network keys on "orange blob": on external images its peak response landed
    on reddish ground and red chalk graffiti rather than on butts. Randomising
    hue, saturation and channel order — and dropping to greyscale a third of
    the time — makes colour an unreliable cue and forces shape to carry the
    decision.
    """
    if rng.random() < 0.5:
        xb = torch.flip(xb, [3])
    if rng.random() < 0.5:
        xb = torch.flip(xb, [2])
    k = rng.randint(0, 3)
    if k:
        xb = torch.rot90(xb, k, [2, 3])

    if colour:
        if rng.random() < 0.33:                       # full greyscale
            g = (xb * torch.tensor([0.114, 0.587, 0.299],
                                   device=xb.device).view(1, 3, 1, 1)).sum(1, keepdim=True)
            xb = g.repeat(1, 3, 1, 1)
        else:
            if rng.random() < 0.5:                    # channel permutation
                xb = xb[:, torch.randperm(3, device=xb.device)]
            # desaturate towards grey by a random amount
            g = xb.mean(1, keepdim=True)
            xb = g + (xb - g) * rng.uniform(0.2, 1.3)
            # per-channel gain = hue/tint shift
            gain = torch.tensor([rng.uniform(0.8, 1.2) for _ in range(3)],
                                device=xb.device).view(1, 3, 1, 1)
            xb = xb * gain

    xb = xb * rng.uniform(0.7, 1.3)                   # brightness
    xb = (xb - 0.5) * rng.uniform(0.8, 1.25) + 0.5    # contrast
    # flip/rot90 leave the tensor non-contiguous, which breaks the flatten in
    # the backward pass; make it contiguous before it reaches the model.
    return xb.clamp(0, 1).contiguous()


def mine_hard_negatives(model, paths, boxes, rng, patch, dev, per_image=6, cap=1200):
    """Collect the background crops the current model scores most butt-like."""
    model.eval()
    found = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            continue
        H, W = im.shape[:2]
        bs = boxes.get(p, [])
        sides = [max(x1 - x0, y1 - y0) for (x0, y0, x1, y1) in bs] or [W * 0.06]
        crops = []
        for _ in range(per_image * 6):
            side = float(np.mean(sides)) * rng.uniform(1.1, 2.2)
            a, b, c, d = square(rng.uniform(0, W), rng.uniform(0, H), side, W, H)
            if c - a < 16 or d - b < 16:
                continue
            if any(not (c < x0 or a > x1 or d < y0 or b > y1) for (x0, y0, x1, y1) in bs):
                continue
            crops.append(cv2.resize(im[b:d, a:c], (patch, patch)))
        if not crops:
            continue
        X = torch.from_numpy(np.stack(crops).astype(np.float32) / 255).permute(0, 3, 1, 2)
        with torch.no_grad():
            s = model(X.to(dev)).softmax(1)[:, 1].cpu().numpy()
        for i in np.argsort(-s)[:per_image]:
            if s[i] > 0.3:                     # only genuinely confusing ones
                found.append(crops[i])
    rng.shuffle(found)
    return found[:cap]


def metrics(logits, y):
    p = logits.softmax(1)[:, 1]
    pred = (p > 0.5).long()
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"accuracy": round(float((pred == y).float().mean()), 4),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--colour-aug", dest="colour_aug", action="store_true", default=True)
    ap.add_argument("--no-colour-aug", dest="colour_aug", action="store_false")
    ap.add_argument("--mine", action="store_true", default=True)
    ap.add_argument("--no-mine", dest="mine", action="store_false")
    ap.add_argument("--mine-epoch", type=int, default=15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out:
        OUT = ROOT / "analysis" / args.out
        CKPT = ROOT / "analysis" / (Path(args.out).stem + ".pt")

    torch.manual_seed(SEED)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    data = build(args.patch)
    Xtr, Ytr = data["train"]
    Xva, Yva = data["val"]
    print(f"train {tuple(Xtr.shape)} pos={int(Ytr.sum())} neg={int((Ytr==0).sum())}")
    print(f"val   {tuple(Xva.shape)} pos={int(Yva.sum())} neg={int((Yva==0).sum())}")

    model = ButtCNN().to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    rng = random.Random(SEED)
    best, mined = None, 0

    def evaluate():
        model.eval()
        with torch.no_grad():
            lo = torch.cat([model(Xva[i:i + 128].to(dev)).cpu()
                            for i in range(0, len(Xva), 128)])
        return metrics(lo, Yva)

    for ep in range(1, args.epochs + 1):
        # Midway, add the background crops the model currently finds most
        # butt-like. These are the errors it actually makes, which is worth far
        # more than the same number of random negatives.
        if args.mine and ep == args.mine_epoch:
            hard = mine_hard_negatives(model, data["mine_paths"], data["boxes"],
                                       rng, args.patch, dev)
            if hard:
                Xh = torch.from_numpy(np.stack(hard).astype(np.float32) / 255).permute(0, 3, 1, 2)
                Xtr = torch.cat([Xtr, Xh])
                Ytr = torch.cat([Ytr, torch.zeros(len(Xh), dtype=torch.long)])
                mined = len(Xh)
                print(f"  mined {mined} hard negatives at epoch {ep}")

        w = torch.tensor([1.0, float((Ytr == 0).sum()) / max(1, int(Ytr.sum()))]).to(dev)
        model.train()
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            xb = augment(Xtr[idx].to(dev), rng, colour=args.colour_aug)
            yb = Ytr[idx].to(dev)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb, weight=w)
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        m = evaluate()
        if best is None or m["f1"] >= best["f1"]:
            best = {**m, "epoch": ep}
            torch.save(model.state_dict(), CKPT)
        if ep % 5 == 0 or ep == 1:
            print(f"  ep{ep:3} loss {tot/len(Xtr):.4f}  val {m}")

    res = {"params": n_params, "patch": args.patch, "epochs": args.epochs,
           "colour_aug": args.colour_aug, "hard_negatives_mined": mined,
           "train_pos": int(Ytr.sum()), "train_neg": int((Ytr == 0).sum()),
           "val_pos": int(Yva.sum()), "val_neg": int((Yva == 0).sum()),
           "val_majority_baseline": round(float(max((Yva == 0).float().mean(),
                                                    Yva.float().mean())), 4),
           "best": best}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))

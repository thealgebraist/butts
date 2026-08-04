#!/usr/bin/env python3
"""
download_images.py – Download images listed in external_storage/manifest.csv.

Supports backends: r2, s3, gdrive, and direct HTTP URLs.

Requirements:
    pip install requests tqdm

Usage:
    # Download everything
    python scripts/download_images.py --dest images/

    # Download only R2-hosted images
    python scripts/download_images.py --dest images/ --backend r2

    # Dry-run (list what would be downloaded)
    python scripts/download_images.py --dest images/ --dry-run
"""

import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path

import requests
from tqdm import tqdm

MANIFEST_PATH = Path(__file__).parent.parent / "external_storage" / "manifest.csv"
CHUNK_SIZE = 65536


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name, leave=False) as bar:
        for chunk in response.iter_content(CHUNK_SIZE):
            f.write(chunk)
            bar.update(len(chunk))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download images from external_storage/manifest.csv.")
    parser.add_argument("--dest", default="images", help="Root directory to save images into (default: images/).")
    parser.add_argument("--backend", choices=["r2", "gdrive", "s3", "all"], default="all",
                        help="Only download from this backend (default: all).")
    parser.add_argument("--no-verify", action="store_true", help="Skip SHA-256 integrity check after download.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be downloaded without downloading.")
    args = parser.parse_args()

    dest_root = Path(args.dest)

    if not MANIFEST_PATH.exists():
        print(f"Manifest not found: {MANIFEST_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(MANIFEST_PATH, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Filter rows whose values look like example placeholders
    rows = [r for r in rows if "XXXX" not in r.get("url", "") and "FILE_ID" not in r.get("url", "")]

    if args.backend != "all":
        rows = [r for r in rows if r.get("storage_backend") == args.backend]

    if not rows:
        print("No matching entries found in manifest.")
        return

    print(f"{'DRY RUN – ' if args.dry_run else ''}Downloading {len(rows)} image(s) to {dest_root}/")

    errors = []
    for row in rows:
        filename = row["filename"]
        category = row.get("category", "")
        subcategory = row.get("subcategory", "")
        url = row["url"]
        expected_hash = row.get("sha256", "")

        dest_path = dest_root / category / subcategory / filename

        if args.dry_run:
            status = "(exists)" if dest_path.exists() else "(missing)"
            print(f"  {status} {dest_path}  ←  {url}")
            continue

        if dest_path.exists():
            if not args.no_verify and expected_hash:
                if sha256_of_file(dest_path) == expected_hash:
                    print(f"  ✓ skip (already verified): {dest_path}")
                    continue
                else:
                    print(f"  ! hash mismatch, re-downloading: {dest_path}")
            else:
                print(f"  ✓ skip (exists): {dest_path}")
                continue

        try:
            print(f"  ↓ {dest_path}")
            download_file(url, dest_path)
            if not args.no_verify and expected_hash:
                actual = sha256_of_file(dest_path)
                if actual != expected_hash:
                    print(f"    ✗ integrity check FAILED for {filename} (got {actual})", file=sys.stderr)
                    errors.append(filename)
                else:
                    print(f"    ✓ integrity ok")
        except Exception as exc:
            print(f"    ✗ failed to download {filename}: {exc}", file=sys.stderr)
            errors.append(filename)

    if errors:
        print(f"\n{len(errors)} file(s) failed: {errors}", file=sys.stderr)
        sys.exit(1)
    elif not args.dry_run:
        print("All downloads complete.")


if __name__ == "__main__":
    main()

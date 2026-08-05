#!/usr/bin/env python3
"""Analyze every image in images/jpg with GPT-4o vision via the OpenAI Batch API.

Usage:
    python scripts/analyze_images.py build     # build the .jsonl request file
    python scripts/analyze_images.py submit    # upload + create the batch
    python scripts/analyze_images.py status    # print batch status once
    python scripts/analyze_images.py wait      # block until the batch finishes
    python scripts/analyze_images.py fetch     # download results -> analysis/image_analysis.json
    python scripts/analyze_images.py run       # build + submit + wait + fetch

The API key is read from key.txt (gitignored).
"""

import base64
import io
import json
import sys
import time
from pathlib import Path

from openai import OpenAI
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "images" / "jpg"
WORK_DIR = ROOT / "analysis" / "batch"
REQUESTS_FILE = WORK_DIR / "requests.jsonl"
BATCH_ID_FILE = WORK_DIR / "batch_id.txt"
RESULTS_FILE = ROOT / "analysis" / "image_analysis.json"

MODEL = "gpt-4o"
MAX_EDGE = 1024  # images are downscaled before upload to keep the batch file small
JPEG_QUALITY = 85

PROMPT = """You are cataloguing photographs of litter and waste items for a computer-vision dataset.

Describe what this image contains. Respond with JSON only, matching this schema:
{
  "primary_object": "short name of the main waste item, or 'none' if no waste item is present",
  "objects": ["every distinct object visible, including background elements"],
  "material": "dominant material of the primary object (plastic, paper, metal, glass, organic, textile, mixed, unknown)",
  "surface": "what the item is resting on (asphalt, grass, soil, gravel, concrete, sand, water, indoor floor, other)",
  "lighting": "bright sunlight, overcast, shadow, night, or indoor",
  "count": <integer number of distinct waste items visible>,
  "occlusion": "none, partial, or heavy",
  "condition": "brief note on wear, dirt, damage, or decay",
  "caption": "one factual sentence describing the whole image"
}"""


def load_key() -> str:
    key_path = ROOT / "key.txt"
    if not key_path.exists():
        sys.exit("key.txt not found; put your OpenAI API key there.")
    return key_path.read_text().strip()


def client() -> OpenAI:
    return OpenAI(api_key=load_key())


def image_paths() -> list[Path]:
    return sorted(p for p in IMAGE_DIR.rglob("*.jpg") if p.is_file())


def encode(path: Path) -> str:
    """Downscale to MAX_EDGE and return a base64 JPEG data URL."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((MAX_EDGE, MAX_EDGE), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    paths = image_paths()
    if not paths:
        sys.exit(f"no .jpg files under {IMAGE_DIR}")

    with REQUESTS_FILE.open("w") as fh:
        for i, path in enumerate(paths, 1):
            rel = path.relative_to(IMAGE_DIR).as_posix()
            fh.write(
                json.dumps(
                    {
                        "custom_id": rel,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": MODEL,
                            "max_tokens": 600,
                            "response_format": {"type": "json_object"},
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": PROMPT},
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": encode(path),
                                                "detail": "low",
                                            },
                                        },
                                    ],
                                }
                            ],
                        },
                    }
                )
                + "\n"
            )
            if i % 25 == 0 or i == len(paths):
                print(f"  encoded {i}/{len(paths)}", flush=True)

    size_mb = REQUESTS_FILE.stat().st_size / 1e6
    print(f"wrote {REQUESTS_FILE} ({len(paths)} requests, {size_mb:.1f} MB)")
    if size_mb > 190:
        sys.exit("batch file exceeds the 200 MB limit; lower MAX_EDGE or split it.")


def submit() -> str:
    cl = client()
    uploaded = cl.files.create(file=REQUESTS_FILE.open("rb"), purpose="batch")
    batch = cl.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "waste image vision analysis"},
    )
    BATCH_ID_FILE.write_text(batch.id)
    print(f"batch {batch.id} created (status: {batch.status})")
    return batch.id


TERMINAL = ("completed", "failed", "expired", "cancelled")


def status():
    batch = client().batches.retrieve(BATCH_ID_FILE.read_text().strip())
    c = batch.request_counts
    print(f"{batch.id}: {batch.status} — {c.completed}/{c.total} done, {c.failed} errored")
    return batch


def wait():
    """Block until the batch reaches a terminal state; exit non-zero if not completed."""
    while True:
        batch = status()
        if batch.status in TERMINAL:
            break
        time.sleep(60)
    if batch.status != "completed":
        sys.exit(f"batch ended as {batch.status}")


def fetch() -> None:
    cl = client()
    batch = cl.batches.retrieve(BATCH_ID_FILE.read_text().strip())
    if not batch.output_file_id:
        sys.exit(f"no output yet (status: {batch.status})")

    results = {}
    errors = {}
    for line in cl.files.content(batch.output_file_id).text.splitlines():
        row = json.loads(line)
        cid = row["custom_id"]
        resp = row.get("response") or {}
        if row.get("error") or resp.get("status_code") != 200:
            errors[cid] = row.get("error") or resp.get("body")
            continue
        content = resp["body"]["choices"][0]["message"]["content"]
        try:
            results[cid] = json.loads(content)
        except json.JSONDecodeError:
            errors[cid] = {"unparsable": content}

    if batch.error_file_id:
        for line in cl.files.content(batch.error_file_id).text.splitlines():
            row = json.loads(line)
            errors[row["custom_id"]] = row.get("error")

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps(
            {"model": MODEL, "batch_id": batch.id, "results": results, "errors": errors},
            indent=2,
            sort_keys=True,
        )
    )
    print(f"wrote {RESULTS_FILE}: {len(results)} analyzed, {len(errors)} errors")


def run() -> None:
    build()
    submit()
    wait()
    fetch()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {
        "build": build,
        "submit": submit,
        "status": status,
        "wait": wait,
        "fetch": fetch,
        "run": run,
    }[cmd]()

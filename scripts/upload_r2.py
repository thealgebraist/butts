#!/usr/bin/env python3
"""
upload_r2.py – Upload a local image to Cloudflare R2 and print the public URL.

Requirements:
    pip install boto3

Environment variables (or use a .env file with python-dotenv):
    R2_ACCOUNT_ID      – Cloudflare account ID
    R2_ACCESS_KEY_ID   – R2 API access key
    R2_SECRET_ACCESS_KEY – R2 API secret key
    R2_BUCKET          – Bucket name
    R2_PUBLIC_BASE_URL – Base URL for public access, e.g. https://pub-XXXX.r2.dev

Usage:
    python scripts/upload_r2.py path/to/image.jpg aerial/high_altitude/image.jpg
"""

import argparse
import hashlib
import os
import sys

import boto3
from botocore.config import Config


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload an image to Cloudflare R2.")
    parser.add_argument("local_path", help="Path to the local image file.")
    parser.add_argument("remote_key", help="Object key in the R2 bucket, e.g. aerial/high_altitude/image.jpg.")
    args = parser.parse_args()

    account_id = os.environ["R2_ACCOUNT_ID"]
    access_key = os.environ["R2_ACCESS_KEY_ID"]
    secret_key = os.environ["R2_SECRET_ACCESS_KEY"]
    bucket = os.environ["R2_BUCKET"]
    public_base = os.environ.get("R2_PUBLIC_BASE_URL", "").rstrip("/")

    endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
    )

    file_size = os.path.getsize(args.local_path)
    checksum = sha256_of_file(args.local_path)

    print(f"Uploading {args.local_path} → s3://{bucket}/{args.remote_key} …")
    client.upload_file(args.local_path, bucket, args.remote_key)

    public_url = f"{public_base}/{args.remote_key}" if public_base else "(set R2_PUBLIC_BASE_URL)"
    print(f"Done.")
    print(f"  public_url : {public_url}")
    print(f"  size_bytes : {file_size}")
    print(f"  sha256     : {checksum}")
    print()
    print("Add this row to external_storage/manifest.csv:")
    filename = os.path.basename(args.local_path)
    parts = args.remote_key.split("/")
    category = parts[0] if len(parts) > 0 else ""
    subcategory = parts[1] if len(parts) > 1 else ""
    print(f"{filename},{category},{subcategory},r2,{public_url},{file_size},{checksum},,")


if __name__ == "__main__":
    main()

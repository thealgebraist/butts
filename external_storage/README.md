# External Storage

This directory contains the **manifest** that tracks high-resolution images stored outside the Git repository on cloud storage backends.

## Why external storage?

High-resolution drone images and RAW files can easily be 10–50 MB each. Storing thousands of them directly in Git (even with LFS) can become expensive or slow. The manifest approach lets you:

- Keep the repo lightweight and fast to clone.
- Store originals on cheap object storage (Cloudflare R2, Google Drive, Amazon S3, etc.).
- Reproduce the full dataset on any machine with a single download command.

## manifest.csv columns

| Column            | Description                                                      |
|-------------------|------------------------------------------------------------------|
| `filename`        | Original filename (matches naming convention in `images/`)       |
| `category`        | Top-level category (`aerial`, `close_up`, etc.)                  |
| `subcategory`     | Subcategory folder name                                          |
| `storage_backend` | `r2`, `gdrive`, `s3`, or `lfs`                                   |
| `url`             | Public (or signed) download URL                                  |
| `size_bytes`      | File size in bytes                                               |
| `sha256`          | SHA-256 hash for integrity verification                          |
| `date_captured`   | ISO 8601 date the photo was taken                                |
| `notes`           | Free-text field for capture conditions or equipment used         |

## Adding a new image

1. Upload the file to your chosen backend (see `../scripts/`).
2. Append a row to `manifest.csv` with the returned public URL and metadata.
3. Commit only the updated `manifest.csv` — **not the image file itself**.

## Downloading images

```bash
pip install -r scripts/requirements.txt
python scripts/download_images.py --dest images/
```

Use `--backend r2` / `--backend gdrive` / `--backend s3` to download only from one source.

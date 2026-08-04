# butts

A dataset of cigarette butts photographed "in the wild", intended for training a neural network to detect them across a wide range of conditions — from drone altitude down to ground level, on various surfaces, under different lighting and weather, and when partially obscured by grass or other debris.

The end goal is to power an autonomous detection-and-retrieval system: a drone identifies butts from the air, and a small ground vehicle navigates to them and picks them up.

## Repository Structure

```
butts/
├── images/              # Raw images, organised by capture scenario (small/preview files)
│   ├── aerial/              # High- and low-altitude drone shots
│   ├── close_up/            # Ground-level macro shots
│   ├── partial_occlusion/   # Butts hidden by grass, leaves, soil, debris
│   ├── lighting/            # Varying light conditions (sun, shade, night, …)
│   ├── surfaces/            # Different ground types (pavement, grass, gravel, …)
│   └── weather/             # Dry, rain, and snow conditions
│
├── annotations/         # Bounding-box / segmentation labels mirroring images/
│
├── external_storage/    # Manifest tracking high-res images on cloud storage
│   └── manifest.csv         # One row per external image (URL, hash, metadata)
│
└── scripts/             # Upload/download helpers for external storage
    ├── upload_r2.py         # Upload to Cloudflare R2
    ├── upload_gdrive.py     # Upload to Google Drive
    ├── download_images.py   # Download all images from manifest
    └── requirements.txt
```

See [`images/README.md`](images/README.md) for full folder descriptions and the naming convention, and [`annotations/README.md`](annotations/README.md) for label formats.

## Large File / High-Resolution Image Storage

Full-resolution images (RAW, high-res JPEG/TIFF) can be very large. Three options are supported:

### Option 1 – Git LFS (recommended for small teams)

All image and video file types are already configured in `.gitattributes` to use [Git Large File Storage](https://git-lfs.com). Install Git LFS once and it works transparently:

```bash
git lfs install   # one-time setup
git add images/aerial/high_altitude/my_image.jpg
git commit -m "add aerial image"
git push          # LFS handles the large file automatically
```

GitHub offers 1 GB of free LFS storage; additional storage can be purchased.

### Option 2 – Cloudflare R2 (recommended for large datasets)

Upload images to a Cloudflare R2 bucket (S3-compatible, generous free tier, no egress fees):

```bash
pip install -r scripts/requirements.txt
export R2_ACCOUNT_ID=… R2_ACCESS_KEY_ID=… R2_SECRET_ACCESS_KEY=… R2_BUCKET=butts-dataset R2_PUBLIC_BASE_URL=https://pub-XXXX.r2.dev
python scripts/upload_r2.py path/to/image.jpg aerial/high_altitude/image.jpg
# → copy the printed row into external_storage/manifest.csv
```

### Option 3 – Google Drive

```bash
# Download OAuth credentials from Google Cloud Console → credentials.json
export GDRIVE_FOLDER_ID=your_folder_id
python scripts/upload_gdrive.py path/to/image.jpg
# → copy the printed row into external_storage/manifest.csv
```

### Downloading images from the manifest

Any contributor can reproduce the full dataset locally:

```bash
python scripts/download_images.py --dest images/
```

See [`scripts/README.md`](scripts/README.md) and [`external_storage/README.md`](external_storage/README.md) for full details.

## Contributing Images

1. Place images in the most specific matching subcategory folder.
2. Follow the naming convention: `<category>_<subcategory>_<YYYYMMDD>_<sequence>.jpg`
3. For **small/preview images** — commit directly (Git LFS handles the file automatically).
4. For **high-resolution originals** — upload to R2 or Google Drive using `scripts/upload_r2.py` or `scripts/upload_gdrive.py`, then add the printed row to `external_storage/manifest.csv`.
5. Add a corresponding annotation file in `annotations/` using YOLO `.txt` format.
6. Open a pull request with a brief description of the capture conditions.

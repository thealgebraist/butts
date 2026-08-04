# Scripts

Helper scripts for uploading high-resolution images to external storage and downloading them locally.

## Setup

```bash
pip install -r scripts/requirements.txt
```

## Uploading to Cloudflare R2

```bash
export R2_ACCOUNT_ID=your_account_id
export R2_ACCESS_KEY_ID=your_access_key
export R2_SECRET_ACCESS_KEY=your_secret_key
export R2_BUCKET=butts-dataset
export R2_PUBLIC_BASE_URL=https://pub-XXXX.r2.dev

python scripts/upload_r2.py path/to/image.jpg aerial/high_altitude/image.jpg
```

The script prints the public URL, file size, and SHA-256 hash — copy the output row straight into `external_storage/manifest.csv`.

## Uploading to Google Drive

```bash
# First time: download OAuth credentials from Google Cloud Console → save as credentials.json
export GDRIVE_FOLDER_ID=your_drive_folder_id

python scripts/upload_gdrive.py path/to/image.jpg
```

A browser window will open on the first run to grant permissions. The script makes the uploaded file publicly readable and prints the manifest row to copy.

## Downloading images from the manifest

```bash
# Download all externally stored images into images/
python scripts/download_images.py --dest images/

# Download only from Cloudflare R2
python scripts/download_images.py --dest images/ --backend r2

# See what would be downloaded without actually downloading
python scripts/download_images.py --dest images/ --dry-run
```

Downloaded files are placed in the correct `images/<category>/<subcategory>/` folder automatically. SHA-256 integrity is verified after each download unless `--no-verify` is passed.

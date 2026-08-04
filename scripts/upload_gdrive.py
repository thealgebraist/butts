#!/usr/bin/env python3
"""
upload_gdrive.py – Upload a local image to a Google Drive folder and print the shareable URL.

Requirements:
    pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib

Setup (one-time):
    1. Create a project in Google Cloud Console.
    2. Enable the Drive API.
    3. Download OAuth2 credentials as 'credentials.json'.
    4. Run this script once; it will open a browser to authenticate and save a token.

Environment variables:
    GDRIVE_FOLDER_ID – ID of the Drive folder to upload into (from the folder's URL).
    GDRIVE_CREDENTIALS – Path to credentials.json (default: credentials.json)
    GDRIVE_TOKEN      – Path to store/read the OAuth token (default: token.json)

Usage:
    python scripts/upload_gdrive.py path/to/image.jpg
"""

import argparse
import hashlib
import os

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_credentials() -> Credentials:
    creds_path = os.environ.get("GDRIVE_CREDENTIALS", "credentials.json")
    token_path = os.environ.get("GDRIVE_TOKEN", "token.json")
    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as f:
            f.write(creds.to_json())
    return creds


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload an image to Google Drive.")
    parser.add_argument("local_path", help="Path to the local image file.")
    args = parser.parse_args()

    folder_id = os.environ.get("GDRIVE_FOLDER_ID")
    creds = get_credentials()
    service = build("drive", "v3", credentials=creds)

    filename = os.path.basename(args.local_path)
    file_size = os.path.getsize(args.local_path)
    checksum = sha256_of_file(args.local_path)

    metadata = {"name": filename}
    if folder_id:
        metadata["parents"] = [folder_id]

    media = MediaFileUpload(args.local_path, resumable=True)
    print(f"Uploading {args.local_path} to Google Drive …")
    result = service.files().create(body=metadata, media_body=media, fields="id,webContentLink").execute()

    file_id = result.get("id")
    # Make the file publicly readable
    service.permissions().create(fileId=file_id, body={"role": "reader", "type": "anyone"}).execute()
    public_url = f"https://drive.google.com/uc?id={file_id}"

    print(f"Done.")
    print(f"  public_url : {public_url}")
    print(f"  size_bytes : {file_size}")
    print(f"  sha256     : {checksum}")
    print()
    print("Add this row to external_storage/manifest.csv:")
    print(f"{filename},,,gdrive,{public_url},{file_size},{checksum},,")


if __name__ == "__main__":
    main()

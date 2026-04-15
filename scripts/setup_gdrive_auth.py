#!/usr/bin/env python3
"""
One-time Google Drive OAuth2 setup.

1. Download OAuth2 Desktop credentials from Google Cloud Console:
   https://console.cloud.google.com → APIs & Services → Credentials
   → Create Credentials → OAuth client ID → Desktop app → Download JSON

2. Save the file to:  <project_root>/secrets/gdrive_credentials.json

3. Run this script once:
   PYTHONPATH=. .venv/bin/python scripts/setup_gdrive_auth.py

4. A browser window will open. Authorize with your Google account.
   token.json will be saved to secrets/gdrive_token.json automatically.

After that, archive_to_gdrive.py and the collector will use the cached token.
"""
from __future__ import annotations
from pathlib import Path
from _bootstrap import ensure_project_root

ROOT = ensure_project_root()
SECRETS_DIR = ROOT / "secrets"
CREDENTIALS_PATH = SECRETS_DIR / "gdrive_credentials.json"
TOKEN_PATH = SECRETS_DIR / "gdrive_token.json"
SCOPES = ["https://www.googleapis.com/auth/drive.file"]

from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

def main():
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)

    if not CREDENTIALS_PATH.exists():
        print(f"\n[ERROR] credentials.json not found at:\n  {CREDENTIALS_PATH}")
        return

    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), SCOPES)
            # run_console: prints URL → user pastes auth code back (works without browser on server)
            try:
                creds = flow.run_console()
            except Exception:
                # Fallback: try local server (works if running locally)
                creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_text(creds.to_json())

    print(f"\n[OK] Authentication successful. Token saved to:\n  {TOKEN_PATH}")
    print("archive_to_gdrive.py is now ready to use.")

if __name__ == "__main__":
    main()

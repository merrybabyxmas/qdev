#!/usr/bin/env python3
"""
Data archival script: local data → Google Drive, then prune local copies.

Thresholds (configurable via CLI args):
  --raw-rows      : rows per raw CSV before archiving (default 500)
  --dataset-rows  : rows in panel.csv before archiving dataset snapshot (default 2000)
  --max-datasets  : max number of dataset snapshots to keep locally (default 3)

Google Drive target folder: 1fl_PGsbO1tbdI0an9T1nnAfvhgda4X3f

Auth: OAuth2 via credentials.json (one-time browser flow, then token.json cached).
Place credentials.json at <project_root>/secrets/gdrive_credentials.json
or set GDRIVE_CREDENTIALS env var to the path.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/archive_to_gdrive.py [--dry-run] [--raw-rows 500]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import ensure_project_root

ROOT = ensure_project_root()

# ── Google Drive API ──────────────────────────────────────────────────────────
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from src.utils.logger import logger

# ── Constants ─────────────────────────────────────────────────────────────────
GDRIVE_FOLDER_ID = "1fl_PGsbO1tbdI0an9T1nnAfvhgda4X3f"
SCOPES = ["https://www.googleapis.com/auth/drive.file"]

SECRETS_DIR = ROOT / "secrets"
TOKEN_PATH = SECRETS_DIR / "gdrive_token.json"
CREDENTIALS_PATH = Path(os.environ.get("GDRIVE_CREDENTIALS", str(SECRETS_DIR / "gdrive_credentials.json")))

DATA_RAW_DIR = ROOT / "artifacts" / "data" / "raw"
DATA_FEATURE_DIR = ROOT / "artifacts" / "data" / "feature_ready"
DATASETS_DIR = ROOT / "artifacts" / "experiments" / "datasets"
MODELS_DIR = ROOT / "artifacts" / "models"
ARCHIVE_LOG_PATH = ROOT / "artifacts" / "control_plane" / "archive_log.jsonl"


# ── Auth ──────────────────────────────────────────────────────────────────────
def _get_drive_service():
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    creds = None

    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_PATH.exists():
                raise FileNotFoundError(
                    f"Google Drive credentials not found at {CREDENTIALS_PATH}.\n"
                    "Download OAuth2 credentials from Google Cloud Console and save as:\n"
                    f"  {CREDENTIALS_PATH}\n"
                    "Or set GDRIVE_CREDENTIALS env var to the correct path."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), SCOPES)
            creds = flow.run_local_server(port=0)

        TOKEN_PATH.write_text(creds.to_json())

    return build("drive", "v3", credentials=creds)


# ── Drive helpers ─────────────────────────────────────────────────────────────
def _ensure_subfolder(service, name: str, parent_id: str) -> str:
    """Return folder ID, creating it if it doesn't exist."""
    q = (
        f"name='{name}' and '{parent_id}' in parents "
        f"and mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    results = service.files().list(q=q, fields="files(id,name)").execute()
    files = results.get("files", [])
    if files:
        return files[0]["id"]

    meta = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    folder = service.files().create(body=meta, fields="id").execute()
    return folder["id"]


def _upload_file(service, local_path: Path, folder_id: str, dry_run: bool) -> str | None:
    if dry_run:
        logger.info(f"[DRY-RUN] Would upload {local_path.name} → GDrive folder {folder_id}")
        return "dry-run"

    mime = "text/csv" if local_path.suffix == ".csv" else "application/octet-stream"
    media = MediaFileUpload(str(local_path), mimetype=mime, resumable=True)
    meta = {"name": local_path.name, "parents": [folder_id]}
    uploaded = service.files().create(body=meta, media_body=media, fields="id").execute()
    file_id = uploaded.get("id")
    logger.info(f"Uploaded {local_path.name} → GDrive id={file_id}")
    return file_id


def _log_archive(entry: dict):
    ARCHIVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ARCHIVE_LOG_PATH, "a") as f:
        f.write(json.dumps({**entry, "archived_at": datetime.now(timezone.utc).isoformat()}) + "\n")


# ── Archive logic ─────────────────────────────────────────────────────────────
def archive_raw_csvs(service, threshold_rows: int, dry_run: bool):
    """Archive raw CSVs that exceed threshold_rows, then truncate to keep last 180 rows."""
    if not DATA_RAW_DIR.exists():
        return

    folder_id = _ensure_subfolder(service, "raw", GDRIVE_FOLDER_ID)
    date_folder = _ensure_subfolder(service, datetime.now(timezone.utc).strftime("%Y-%m-%d"), folder_id)

    for csv in sorted(DATA_RAW_DIR.glob("*.csv")):
        lines = csv.read_text(encoding="utf-8").splitlines()
        row_count = len(lines) - 1  # subtract header
        if row_count < threshold_rows:
            logger.info(f"{csv.name}: {row_count} rows < threshold {threshold_rows}, skipping")
            continue

        logger.info(f"{csv.name}: {row_count} rows ≥ threshold {threshold_rows}, archiving")
        file_id = _upload_file(service, csv, date_folder, dry_run)

        if not dry_run and file_id:
            # Keep only header + last 180 rows
            header = lines[0]
            keep = lines[-180:]
            csv.write_text("\n".join([header] + keep) + "\n", encoding="utf-8")
            logger.info(f"Truncated {csv.name} to {len(keep)} rows locally")
            _log_archive({"type": "raw_csv", "file": csv.name, "archived_rows": row_count,
                          "kept_rows": len(keep), "gdrive_id": file_id})

    # Mirror for feature_ready
    feat_folder_id = _ensure_subfolder(service, "feature_ready", GDRIVE_FOLDER_ID)
    feat_date_folder = _ensure_subfolder(service, datetime.now(timezone.utc).strftime("%Y-%m-%d"), feat_folder_id)

    for csv in sorted(DATA_FEATURE_DIR.glob("*.csv")):
        lines = csv.read_text(encoding="utf-8").splitlines()
        row_count = len(lines) - 1
        if row_count < threshold_rows:
            continue
        file_id = _upload_file(service, csv, feat_date_folder, dry_run)
        if not dry_run and file_id:
            header = lines[0]
            keep = lines[-180:]
            csv.write_text("\n".join([header] + keep) + "\n", encoding="utf-8")
            _log_archive({"type": "feature_csv", "file": csv.name, "archived_rows": row_count,
                          "kept_rows": len(keep), "gdrive_id": file_id})


def archive_dataset_snapshots(service, threshold_rows: int, max_local: int, dry_run: bool):
    """Archive old dataset panel.csv snapshots, keep only `max_local` newest locally."""
    if not DATASETS_DIR.exists():
        return

    snapshots = sorted(DATASETS_DIR.iterdir(), key=lambda p: p.stat().st_mtime)
    panels = [(d, d / "panel.csv") for d in snapshots if (d / "panel.csv").exists()]

    # Check if any panel exceeds threshold
    large_panels = [(d, p) for d, p in panels if sum(1 for _ in p.open()) - 1 >= threshold_rows]
    if not large_panels and len(panels) <= max_local:
        logger.info(f"Dataset snapshots: {len(panels)} local, none exceed threshold. Skipping.")
        return

    folder_id = _ensure_subfolder(service, "datasets", GDRIVE_FOLDER_ID)

    # Archive oldest ones beyond max_local
    to_archive = panels[:-max_local] if len(panels) > max_local else []
    # Also archive any that are large
    to_archive_dirs = set(d for d, _ in to_archive) | set(d for d, _ in large_panels)

    for dataset_dir in to_archive_dirs:
        version_folder_id = _ensure_subfolder(service, dataset_dir.name, folder_id)
        for f in sorted(dataset_dir.iterdir()):
            if f.is_file():
                file_id = _upload_file(service, f, version_folder_id, dry_run)
                if not dry_run and file_id:
                    _log_archive({"type": "dataset", "version": dataset_dir.name,
                                  "file": f.name, "gdrive_id": file_id})
        if not dry_run:
            shutil.rmtree(dataset_dir)
            logger.info(f"Removed local dataset snapshot: {dataset_dir.name}")


# ── Model archive ─────────────────────────────────────────────────────────────
def archive_models(service, max_local_models: int = 20, dry_run: bool = False) -> int:
    """
    Scan artifacts/models/ for .pkl and .pt files sorted by mtime.
    If total count > max_local_models, upload the oldest ones to GDrive
    subfolder 'qdev_models', then delete local copies.
    Returns the count of archived files.
    """
    if not MODELS_DIR.exists():
        logger.info("No models directory found, skipping model archive.")
        return 0

    # Collect all model files across all pipeline subdirectories
    all_model_files = sorted(
        [p for p in MODELS_DIR.rglob("*") if p.is_file() and p.suffix in (".pkl", ".pt")],
        key=lambda p: p.stat().st_mtime,
    )

    if len(all_model_files) <= max_local_models:
        logger.info(f"Model files: {len(all_model_files)} ≤ threshold {max_local_models}, skipping.")
        return 0

    to_archive = all_model_files[: len(all_model_files) - max_local_models]
    logger.info(f"Archiving {len(to_archive)} old model file(s) to GDrive (keeping {max_local_models}).")

    gdrive_models_folder_id = _ensure_subfolder(service, "qdev_models", GDRIVE_FOLDER_ID)
    archived_count = 0

    for model_path in to_archive:
        # Preserve pipeline_id as a subfolder on GDrive
        pipeline_id = model_path.parent.name
        pipeline_folder_id = _ensure_subfolder(service, pipeline_id, gdrive_models_folder_id)
        file_id = _upload_file(service, model_path, pipeline_folder_id, dry_run)
        if file_id:
            if not dry_run:
                model_path.unlink()
                logger.info(f"Deleted local model: {model_path}")
            _log_archive({
                "type": "model",
                "pipeline_id": pipeline_id,
                "file": model_path.name,
                "gdrive_id": file_id,
            })
            archived_count += 1

    return archived_count


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Archive data to Google Drive")
    parser.add_argument("--raw-rows", type=int, default=500,
                        help="Row threshold for raw/feature CSVs (default: 500)")
    parser.add_argument("--dataset-rows", type=int, default=2000,
                        help="Row threshold for dataset panel.csv (default: 2000)")
    parser.add_argument("--max-datasets", type=int, default=3,
                        help="Max local dataset snapshots to keep (default: 3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be archived without doing it")
    parser.add_argument("--models-only", action="store_true",
                        help="Skip data/dataset archival; only archive model files")
    parser.add_argument("--max-models", type=int, default=20,
                        help="Max local model files to keep before archiving (default: 20)")
    args = parser.parse_args()

    logger.info("Starting data archival", dry_run=args.dry_run,
                raw_threshold=args.raw_rows, dataset_threshold=args.dataset_rows,
                models_only=args.models_only, max_models=args.max_models)

    service = _get_drive_service()

    if not args.models_only:
        archive_raw_csvs(service, threshold_rows=args.raw_rows, dry_run=args.dry_run)
        archive_dataset_snapshots(service, threshold_rows=args.dataset_rows,
                                  max_local=args.max_datasets, dry_run=args.dry_run)

    archived = archive_models(service, max_local_models=args.max_models, dry_run=args.dry_run)
    logger.info(f"Model archival complete: {archived} file(s) archived.")

    logger.info("Archival complete.")


if __name__ == "__main__":
    main()

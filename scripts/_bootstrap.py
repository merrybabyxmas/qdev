from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_root() -> Path:
    """Make the repository root importable when scripts are executed directly."""
    project_root = Path(__file__).resolve().parents[1]
    root = str(project_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from src.utils.env import load_repo_env

        load_repo_env()
    except Exception:
        # Environment loading must not block script startup; explicit env vars still win.
        pass
    return project_root

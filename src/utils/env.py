from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()
    if "=" not in stripped:
        return None

    key, raw_value = stripped.split("=", 1)
    key = key.strip()
    if not key:
        return None

    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return key, value


def _load_env_file(path: Path, *, override: bool = False) -> bool:
    if not path.exists():
        return False

    loaded = False
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if override or key not in os.environ:
            os.environ[key] = value
            loaded = True
    return loaded


def load_repo_env(paths: Iterable[Path] | None = None) -> list[Path]:
    """
    Load repository-local environment variables from .env-style files.

    The loader keeps explicit process environment values authoritative, then
    optionally layers repo-local config for development and paper validation.
    """
    project_root = get_project_root()
    env_paths = list(paths or (project_root / ".env", project_root / ".env.local"))
    loaded_paths: list[Path] = []

    for path in env_paths:
        if path.name == ".env.local":
            loaded = _load_env_file(path, override=True)
        else:
            loaded = _load_env_file(path, override=False)
        if loaded:
            loaded_paths.append(path)
    return loaded_paths

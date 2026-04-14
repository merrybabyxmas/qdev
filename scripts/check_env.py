from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

from _bootstrap import ensure_project_root


def main() -> int:
    project_root = ensure_project_root()
    print(f"Project root: {project_root}")
    print(f"Python: {platform.python_version()} ({sys.executable})")

    required_modules = [
        "numpy",
        "pandas",
        "structlog",
        "hmmlearn",
        "lightgbm",
        "vectorbt",
        "alpaca",
        "pytest",
        "src.utils.config",
        "src.utils.logger",
        "src.features.builder",
        "src.models.hmm",
        "src.models.lgbm",
        "src.models.linear",
        "src.risk.manager",
        "src.backtest.engine",
        "src.backtest.matching_engine",
        "src.ingestion.loader",
        "src.ingestion.websocket_client",
        "src.brokers.mock",
        "src.monitoring.soak",
        "src.evaluation.dataset",
        "src.evaluation.registry",
        "src.evaluation.runner",
    ]

    failures: list[tuple[str, str]] = []
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
            print(f"OK  {module_name}")
        except Exception as exc:  # pragma: no cover - defensive path
            failures.append((module_name, repr(exc)))
            print(f"ERR {module_name}: {exc}")

    requirements_file = project_root / "requirements.txt"
    print(f"requirements.txt: {'found' if requirements_file.exists() else 'missing'}")

    if failures:
        print("\nEnvironment check failed:")
        for module_name, error in failures:
            print(f"- {module_name}: {error}")
        return 1

    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

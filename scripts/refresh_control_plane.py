#!/usr/bin/env python3
from __future__ import annotations

import json

from _bootstrap import ensure_project_root

ensure_project_root()

from src.controlplane.snapshot import build_dashboard_snapshot  # noqa: E402


def main() -> int:
    snapshot = build_dashboard_snapshot()
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

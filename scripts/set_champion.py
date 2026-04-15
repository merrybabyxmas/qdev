#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from _bootstrap import ensure_project_root

ensure_project_root()

from src.controlplane.snapshot import REGISTRY_JSON, build_dashboard_snapshot  # noqa: E402


def _load_registry() -> dict[str, object]:
    if not REGISTRY_JSON.exists():
        return {}
    return json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Set or clear a manual champion override.")
    parser.add_argument("--pipeline-id", type=str, default=None)
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    registry = _load_registry()
    if args.clear:
        registry["override_enabled"] = False
        registry["manual_champion_pipeline_id"] = None
    elif args.pipeline_id:
        registry["override_enabled"] = True
        registry["manual_champion_pipeline_id"] = args.pipeline_id
    else:
        raise SystemExit("Either --pipeline-id or --clear is required.")

    REGISTRY_JSON.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
    snapshot = build_dashboard_snapshot()
    print(json.dumps(snapshot.get("registry", {}), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

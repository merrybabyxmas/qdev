#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"

if [[ ! -x "$VENV/bin/python" ]]; then
  if python3 -m venv "$VENV" >/dev/null 2>&1; then
    :
  elif python3 -m virtualenv "$VENV" >/dev/null 2>&1; then
    :
  else
    echo "Could not create virtual environment. Install python3-venv or virtualenv first." >&2
    exit 1
  fi
fi

cd "$ROOT"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r requirements.txt

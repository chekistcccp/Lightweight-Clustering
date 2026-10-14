#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
if [[ "${1:-}" != "--help" && "${1:-}" != "-h" ]]; then
  python -m lightweight_clustering.preflight
fi
python -m lightweight_clustering.experiment "$@"

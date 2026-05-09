#!/usr/bin/env bash
# Thin wrapper (old path). Prefer: vln_choice/scripts/processing/run_build_progress_labels_4z.sh
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${_SCRIPT_DIR}/processing/run_build_progress_labels_4z.sh" "$@"

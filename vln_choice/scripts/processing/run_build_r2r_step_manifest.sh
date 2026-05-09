#!/usr/bin/env bash
# Rebuild R2R step manifest JSONL (build_r2r_step_manifest.py).
# Run from NaviLLM repo root.
#
# Required on host: Matterport scans (--scan_dir), rendered mp3d_views, connectivity, t2t_landmarks.
#
# Env overrides:
#   R2R_JSON              default: data/R2R/FGR2R_train.json
#   SCAN_DIR              default: /root/mount/Matterport3DSimulator/data/v1/scans
#   CONNECTIVITY_DIR      default: data/connectivity
#   RENDERED_VIEW_DIR     default: vln_choice/data_processed/mp3d_views
#   T2T_DIR               default: data/t2t_landmarks
#   OUTPUT_JSONL          default: vln_choice/data_processed/r2r_step_manifest.jsonl
#   LIMIT                 if non-empty, passes --limit
#   PROGRESS_LABELS_JSONL optional; advanced only — re-merge an existing labels JSONL into manifest
#                         rows (normal flow: build manifest first, then build_progress_labels.py).
#   INCLUDE_STOP_STEPS    set to 1 to add --include_stop_steps
#   SEED                  default 0
#   MAX_CANDIDATES        default 0 (keep all); set e.g. 4 for smaller choice sets
#
# Example (full manifest, no limit):
#   bash vln_choice/scripts/processing/run_build_r2r_step_manifest.sh
#
# Optional (advanced): merge an existing progress_labels JSONL back into manifest rows:
#   PROGRESS_LABELS_JSONL=... OUTPUT_JSONL=.../step_manifest_with_progress.jsonl \
#   bash vln_choice/scripts/processing/run_build_r2r_step_manifest.sh

set -euo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAVILLM_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${NAVILLM_ROOT}"

export PYTHONPATH="${NAVILLM_ROOT}/vln_choice:${PYTHONPATH:-}"

R2R_JSON="${R2R_JSON:-data/R2R/FGR2R_train.json}"
SCAN_DIR="${SCAN_DIR:-/root/mount/Matterport3DSimulator/data/v1/scans}"
CONNECTIVITY_DIR="${CONNECTIVITY_DIR:-data/connectivity}"
RENDERED_VIEW_DIR="${RENDERED_VIEW_DIR:-vln_choice/data_processed/mp3d_views}"
T2T_DIR="${T2T_DIR:-data/t2t_landmarks}"
OUTPUT_JSONL="${OUTPUT_JSONL:-vln_choice/data_processed/r2r_step_manifest.jsonl}"
SEED="${SEED:-0}"
MAX_CANDIDATES="${MAX_CANDIDATES:-0}"

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

STOP_ARGS=()
if [[ "${INCLUDE_STOP_STEPS:-0}" != "0" ]]; then
  STOP_ARGS=(--include_stop_steps)
fi

PROGRESS_ARGS=()
if [[ -n "${PROGRESS_LABELS_JSONL:-}" ]]; then
  PROGRESS_ARGS=(--progress_labels_jsonl "${PROGRESS_LABELS_JSONL}")
fi

EXTRA_CLI=()
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA_CLI <<< "${EXTRA_ARGS}"
fi

exec python vln_choice/scripts/processing/build_r2r_step_manifest.py \
  --r2r_json "${R2R_JSON}" \
  --connectivity_dir "${CONNECTIVITY_DIR}" \
  --scan_dir "${SCAN_DIR}" \
  --rendered_view_dir "${RENDERED_VIEW_DIR}" \
  --t2t_landmark_dir "${T2T_DIR}" \
  --output_jsonl "${OUTPUT_JSONL}" \
  --max_candidates "${MAX_CANDIDATES}" \
  --seed "${SEED}" \
  "${PROGRESS_ARGS[@]}" \
  "${STOP_ARGS[@]}" \
  "${LIMIT_ARGS[@]}" \
  "${EXTRA_CLI[@]}"

#!/usr/bin/env bash
# =============================================================================
# Call 4Z API to build R2R-style progress labels (see build_progress_labels.py).
# Run from NaviLLM repo root: bash vln_choice/scripts/processing/run_build_progress_labels_4z.sh
# Requires bash (arrays, pipefail). Env: FOURZ_API_BASE, FOURZ_API_KEY; optional FOURZ_MODEL, LIMIT, RESUME.
# =============================================================================

if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAVILLM_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${NAVILLM_ROOT}"

# --- 4Z credentials ---
: "${FOURZ_API_BASE:=}"
: "${FOURZ_API_KEY:=}"
: "${FOURZ_MODEL:=gpt-5.4-nano}"

if [[ -z "${FOURZ_API_BASE}" || -z "${FOURZ_API_KEY}" ]]; then
  echo "Missing FOURZ_API_BASE or FOURZ_API_KEY. Example:" >&2
  echo "  export FOURZ_API_BASE='https://4zapi.com/v1'" >&2
  echo "  export FOURZ_API_KEY='sk-...'" >&2
  echo "Then: bash vln_choice/scripts/processing/run_build_progress_labels_4z.sh" >&2
  exit 1
fi

export FOURZ_API_BASE FOURZ_API_KEY FOURZ_MODEL

# --- Paths (override via env) ---
R2R_JSON="${R2R_JSON:-data/R2R/R2R_val_seen_enc.json}"
STEP_MANIFEST_JSONL="${STEP_MANIFEST_JSONL:-vln_choice/data_processed/r2r_step_manifest.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-vln_choice/data_processed/r2r_progress_labels.jsonl}"
T2T_DIR="${T2T_DIR:-data/t2t_landmarks}"

# --- CLI knobs ---
LIMIT="${LIMIT:-0}"
PROGRESS_MODE="${PROGRESS_MODE:-candidate}"

LIMIT_ARGS=()
if [[ "${LIMIT}" != "0" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

RESUME_ARGS=()
if [[ "${RESUME:-0}" != "0" ]]; then
  RESUME_ARGS=(--resume)
fi

GEO_ARGS=()
if [[ -n "${CONNECTIVITY_DIR:-}" && -n "${SCAN_DIR:-}" ]]; then
  GEO_ARGS=(--connectivity_dir "${CONNECTIVITY_DIR}" --scan_dir "${SCAN_DIR}")
fi

SOURCE_ARGS=(--r2r_json "${R2R_JSON}")
if [[ -n "${STEP_MANIFEST_JSONL}" ]]; then
  SOURCE_ARGS=(--step_manifest_jsonl "${STEP_MANIFEST_JSONL}")
fi

# Space-separated extra flags, e.g. EXTRA_ARGS="--resume --limit 5"
EXTRA_CLI=()
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA_CLI <<< "${EXTRA_ARGS}"
fi

python vln_choice/scripts/processing/build_progress_labels.py \
  "${SOURCE_ARGS[@]}" \
  --output_jsonl "${OUTPUT_JSONL}" \
  --t2t_landmark_dir "${T2T_DIR}" \
  --splitter api_4z \
  --api_model "${FOURZ_MODEL}" \
  --progress_mode "${PROGRESS_MODE}" \
  "${GEO_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  "${LIMIT_ARGS[@]}" \
  "${EXTRA_CLI[@]}"

echo "Done. Output: ${OUTPUT_JSONL}"

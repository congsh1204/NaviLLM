#!/usr/bin/env bash
# One-click smoke test: default local Qwen3-VL-4B-Instruct + Qwen3-VL-8B-Instruct (same spirit as NaviLLM/scripts/evaluation/*.sh).
# Run from NaviLLM repo root:
#   bash vln_choice/scripts/evaluation/smoke_qwen_vl.sh
#
# Optional env:
#   IMAGE=/path/to.jpg
#   PROMPT="..."
#   MAX_IMAGE_SIDE=768  MAX_NEW_TOKENS=256
#   EXTRA_ARGS="--json_out"   # appended to the python command

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAVILLM_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${NAVILLM_ROOT}"

export PYTHONPATH="${NAVILLM_ROOT}/vln_choice:${PYTHONPATH:-}"

EXTRA=(
  --both
  --max_image_side "${MAX_IMAGE_SIDE:-1024}"
  --max_new_tokens "${MAX_NEW_TOKENS:-128}"
)

if [[ -n "${IMAGE:-}" ]]; then
  EXTRA+=( --image "${IMAGE}" )
fi
if [[ -n "${PROMPT:-}" ]]; then
  EXTRA+=( --prompt "${PROMPT}" )
fi

# shellcheck disable=SC2086
exec python vln_choice/scripts/evaluation/smoke_qwen_vl.py "${EXTRA[@]}" ${EXTRA_ARGS:-}

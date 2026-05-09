#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

cd "$(dirname "$0")/../../.."

MODEL_PATH="${MODEL_PATH:-data/models/Qwen3-VL-8B-Instruct}"
IMAGE_PATH="${1:-vln_choice/composite_images/r2r_48_0_step_000.jpg}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_IMAGE_SIDE="${MAX_IMAGE_SIDE:-1024}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

if [ ! -d "$MODEL_PATH" ]; then
  echo "Model path does not exist: $MODEL_PATH" >&2
  exit 1
fi

if [ ! -f "$IMAGE_PATH" ]; then
  echo "Image path does not exist: $IMAGE_PATH" >&2
  exit 1
fi

METADATA_JSON="$(mktemp)"
trap 'rm -f "$METADATA_JSON"' EXIT

python - "$IMAGE_PATH" "$METADATA_JSON" <<'PY'
import json
import sys
from pathlib import Path

image_path = Path(sys.argv[1])
metadata_path = Path(sys.argv[2])
sample_id = image_path.stem
jsonl_paths = [
    Path("vln_choice/data_processed/choice_samples.jsonl"),
    Path("vln_choice/data_processed/step_manifest_dedup_point_ids.jsonl"),
    Path("vln_choice/data_processed/step_manifest.jsonl"),
]


def load_t2t_landmarks(scan, viewpoint):
    if not scan or not viewpoint:
        return {}
    path = Path("data/t2t_landmarks") / scan / viewpoint / f"{scan}_{viewpoint}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


for jsonl_path in jsonl_paths:
    if not jsonl_path.exists():
        continue
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("sample_id") != sample_id:
                continue
            instruction = row.get("instruction", "")
            point_ids = row.get("candidate_point_ids", {})
            candidate_views = []
            for label, point_id in sorted(point_ids.items()):
                candidate_views.append(f"{label} -> view_{int(point_id):02d}")
            scan = row.get("scan")
            current_viewpoint = row.get("current_viewpoint")
            current_landmarks = load_t2t_landmarks(scan, current_viewpoint)
            metadata_path.write_text(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "scan": scan,
                        "current_viewpoint": current_viewpoint,
                        "instruction": instruction,
                        "candidate_view_mapping": candidate_views,
                        "current_t2t_landmarks": current_landmarks,
                        "target": row.get("target", {}),
                        "next_viewpoint": row.get("next_viewpoint"),
                        "option_mapping": row.get("option_mapping", {}),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            raise SystemExit(0)

metadata_path.write_text(
    json.dumps(
        {
            "sample_id": sample_id,
            "scan": None,
            "current_viewpoint": None,
            "instruction": "Not found. Set INSTRUCTION='...' manually if needed.",
            "candidate_view_mapping": [],
            "current_t2t_landmarks": {},
            "target": {},
            "next_viewpoint": None,
            "option_mapping": {},
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
PY

SAMPLE_METADATA="$(python - "$METADATA_JSON" <<'PY'
import json
import sys
from pathlib import Path

metadata = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
candidate_views = metadata.get("candidate_view_mapping") or []
current_landmarks = metadata.get("current_t2t_landmarks") or {}
print("Instruction:")
print(metadata.get("instruction") or "Not found.")
print()
print("Current viewpoint:")
print("scan={}".format(metadata.get("scan")))
print("viewpoint={}".format(metadata.get("current_viewpoint")))
print()
print("Current viewpoint t2t landmarks:")
lines = []
for view_id, landmarks in sorted(current_landmarks.items(), key=lambda item: int(item[0])):
    if landmarks:
        lines.append("view_{}: {}".format(int(view_id), ", ".join(landmarks)))
print("\n".join(lines) if lines else "No t2t landmarks found.")
print()
print("Candidate view mapping:")
print("\n".join(candidate_views) if candidate_views else "No non-STOP candidate views found.")
PY
)"

echo "========== Ground Truth =========="
python - "$METADATA_JSON" <<'PY'
import json
import sys
from pathlib import Path

metadata = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("sample_id={}".format(metadata.get("sample_id")))
print("scan={}".format(metadata.get("scan")))
print("current_viewpoint={}".format(metadata.get("current_viewpoint")))
print()
print("instruction:")
print(metadata.get("instruction") or "Not found.")
print()
print("current_t2t_landmarks:")
current_landmarks = metadata.get("current_t2t_landmarks") or {}
lines = []
for view_id, landmarks in sorted(current_landmarks.items(), key=lambda item: int(item[0])):
    if landmarks:
        lines.append("view_{}: {}".format(int(view_id), ", ".join(landmarks)))
print("\n".join(lines) if lines else "No t2t landmarks found.")
print()
print("candidate_view_mapping:")
candidate_views = metadata.get("candidate_view_mapping") or []
print("\n".join(candidate_views) if candidate_views else "No non-STOP candidate views found.")
print()
print("target_final_choice={}".format(metadata.get("target", {}).get("final_choice")))
print("next_viewpoint={}".format(metadata.get("next_viewpoint")))
PY
echo "========== Model Inference =========="

python vln_choice/scripts/evaluation/infer_step.py \
  --model_name_or_path "$MODEL_PATH" \
  --image "$IMAGE_PATH" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --max_image_side "$MAX_IMAGE_SIDE" \
  --metadata_json "$METADATA_JSON" \
  --choices A,B,C,D,E,F,G,H,I,J,K,L,STOP \
  --prompt "$(cat <<EOF
You are analyzing a full panoramic image grid for indoor navigation.

The image is a 3 x 12 panorama grid showing view_00 through view_35.
Candidate labels are not drawn on the image. Use the candidate-to-view mapping below to inspect the corresponding view tiles.

${SAMPLE_METADATA}

Choose the best next action according to the instruction. First summarize the useful scene evidence, then compare candidate views, and finally output one final_choice.

For each candidate in the mapping, describe:
1. the nearby room type, object, doorway, hallway, stairs, or landmark,
2. whether the candidate appears to lead toward a different area,
3. whether it supports or conflicts with the instruction,
4. any uncertainty caused by image quality, occlusion, or view layout.

final_choice must be exactly one of the candidate labels in the mapping, or STOP if the instruction goal has already been reached.

Return JSON only with this schema:
{
  "scene_summary": "Brief summary of the current indoor scene.",
  "instruction_relevant_evidence": "Evidence in the panorama that matters for the instruction.",
  "candidate_evidence": {
    "A": "Evidence from A's mapped view, if present.",
    "B": "Evidence from B's mapped view, if present."
  },
  "navigation_relevant_observations": [
    "Important visual cue 1",
    "Important visual cue 2"
  ],
  "final_choice": "A"
}
EOF
)"

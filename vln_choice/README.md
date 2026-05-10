# VLN Choice Prototype

Choice-style VLM navigation prototype that **complements** (does not replace) the main `NaviLLM/train.py` / `NavModel` path. Composite image + instruction → Qwen2.5-VL / Qwen3-VL → structured EGAC target.

```text
                  composite image + instruction + history
                                  │
                                  ▼
                    Qwen2.5-VL / Qwen3-VL  (LoRA SFT)
                                  │
                                  ▼
{
  "target_action":      "descend the stairs",
  "required_evidence":  ["stair", "step"],
  "observed_evidence":  ["dining room", "chandelier"],
  "commitment_state":   "defer | prepare | commit | stop",
  "next_action":        "move toward stair",
  "final_choice":       "A | B | C | D | STOP"
}
```

The first goal is to validate **step-level choice accuracy + commitment-state prediction** offline, before wiring online rollout in [`scripts/evaluation/rollout_eval.py`](scripts/evaluation/rollout_eval.py) (currently a placeholder).

Scripts are grouped by role under [`scripts/processing/`](scripts/processing/), [`scripts/train/`](scripts/train/), [`scripts/evaluation/`](scripts/evaluation/) — see [`scripts/README.md`](scripts/README.md).

---

## 0. Pipeline at a glance

```text
data/R2R/*.json                 ──┐
data/connectivity/                │   raw inputs
data/t2t_landmarks/{scan}/{vp}/   │
MP3D v1 scans                   ──┘
                                  │
                                  ▼
   1. render_mp3d_views.py            → mp3d_views/{scan}/{vp}/view_{00..35}.jpg
                                  │
                                  ▼
   2. build_r2r_step_manifest.py      → r2r_step_manifest.jsonl
                                        per row: instruction, history,
                                                 current.viewpoint_landmarks (full panorama),
                                                 candidates[].view_landmark (directional),
                                                 action.label
                                  │
                                  ▼
   3. build_progress_labels.py        → r2r_progress_labels.jsonl
                                        per instruction: new_instructions, chunk_view,
                                                         step_progress
                                  │
                                  ▼
   4. attach_progress_to_manifest.py  → step_manifest_with_progress.jsonl
                                        manifest rows + progress_reasoning
                                  │
                                  ▼
   5. build_composite_images.py       → choice_samples.jsonl
                                        + composite_images/{sample_id}.jpg
                                  │
                                  ▼
   6. prepare_sft_jsonl.py [--egac]   → qwen_sft[_egac].jsonl
                                  │
                                  ▼
   7. train_sft.py / smoke_qwen_vl / infer_step.py
```

All commands below assume **CWD = `NaviLLM/`** (repo root).

---

## 1. Data pipeline

### 1.1 Render MP3D 36-view RGB

```bash
python vln_choice/scripts/processing/render_mp3d_views.py \
  --connectivity_dir data/connectivity \
  --scan_dir /root/mount/Matterport3DSimulator/data/v1/scans \
  --output_dir vln_choice/data_processed/mp3d_views \
  --limit 100   # smoke first; remove for full run
```

Produces `mp3d_views/{scan}/{viewpoint}/view_{00..35}.jpg` — 36 panorama tiles per node, used both for composite training images and as the visual basis of t2t landmark phrases.

### 1.2 Build step manifest

```bash
python vln_choice/scripts/processing/build_r2r_step_manifest.py \
  --r2r_json data/R2R/FGR2R_train.json \
  --connectivity_dir data/connectivity \
  --scan_dir /root/mount/Matterport3DSimulator/data/v1/scans \
  --rendered_view_dir vln_choice/data_processed/mp3d_views \
  --t2t_landmark_dir data/t2t_landmarks \
  --output_jsonl vln_choice/data_processed/r2r_step_manifest.jsonl
```

One row per step on the expert path; rows are skipped when the current or candidate viewpoint images are not yet rendered.

```json
{
  "sample_id": "r2r_6250_0_step_001",
  "instruction": "Walk down one flight of stairs and stop on the landing.",
  "history": ["vp_start", "vp_001", "vp_002"],
  "current": {
    "viewpoint_id": "5be145994f97...",
    "view_dir": "vln_choice/data_processed/mp3d_views/17DRP5sb8fy/5be145994f97...",
    "viewpoint_landmarks": ["bathroom", "sink", "mirror", "stair", "..."]
  },
  "candidates": [
    {
      "label": "A",
      "viewpoint_id": "...",
      "view_index": 20,
      "view_landmark": "stair; step",
      "nav_order": 1,
      "pose_rad": {"relative": [-0.42, 0.03], "bearing": [0.81, 0.11], "misalignment": 0.31},
      "distance_m": 1.8
    },
    "...",
    {"label": "STOP", "viewpoint_id": "STOP", "...": "..."}
  ],
  "action": {"label": "B", "viewpoint_id": "..."}
}
```

Key fields:

| Field | Meaning |
|---|---|
| `current.viewpoint_landmarks` | **Full 360° t2t panorama** at the current node, deduplicated and normalized via [`viewpoint_landmark_phrases(..., normalize=True)`](vln_choice/progress/landmarks.py). Used as the **observed evidence** signal for EGAC. Cached intra-process so each `(scan, viewpoint)` JSON is read once. |
| `candidates[].view_landmark` | Directional landmarks toward the candidate's `view_index`, semicolon-joined, via `landmark_phrases_for_direction(..., single_view_only=True)`. |
| `candidates[].pose_rad` | `relative` / `bearing` / `misalignment` (rad). |
| `action.label` | Gold supervision: A/B/.../STOP. |

Useful flags:
- `--max_candidates 4` — limit to A/B/C/D-style MVP (default keeps **all** navigable candidates).
- `--keep_duplicate_point_ids` — disable same-`point_id` deduplication (default: dedup, expert next-viewpoint wins to avoid ambiguous labels like `A/C` on the same panorama tile).
- `--include_stop_steps` — also emit one STOP sample at each trajectory endpoint.
- `--limit N` — smoke.

### 1.3 Build progress labels

```bash
python vln_choice/scripts/processing/build_progress_labels.py \
  --step_manifest_jsonl vln_choice/data_processed/r2r_step_manifest.jsonl \
  --output_jsonl vln_choice/data_processed/r2r_progress_labels.jsonl \
  --t2t_landmark_dir data/t2t_landmarks \
  --splitter api_4z \
  --progress_mode candidate
```

One row per `instr_id`. Key fields:

| Field | Shape | Meaning |
|---|---|---|
| `new_instructions` | `[{text, entities, action}]` | Ordered subgoal list. |
| `chunk_view` | `[[start, end], ...]` | Closed `[start_path_pos, last_path_pos]` intervals (1-based) on the expert path, one per subgoal. |
| `step_progress` | `[{path_pos, viewpoint, expert_next_viewpoint, landmarks, landmark_source, subgoal_index, matter_sim_point_id, score, ...}]` | Per-step records aligning the path to subgoals. |
| `source_detail` | dict | Which splitter/aligner ran, `step_manifest_jsonl` used, monotonic-clip flag, etc. |

See [§4 Progress labels — detail](#4-progress-labels--detail) for splitter modes, prompt design, 4Z API config, fallback behavior, and resume.

### 1.4 Attach progress labels to manifest

A **pure JSONL transform** — no MatterSim, no candidate re-render. Replaces the older "rerun `build_r2r_step_manifest.py` with `--progress_labels_jsonl`" pattern (which was correct but slow).

```bash
python vln_choice/scripts/processing/attach_progress_to_manifest.py \
  --manifest_jsonl vln_choice/data_processed/r2r_step_manifest.jsonl \
  --progress_labels_jsonl vln_choice/data_processed/r2r_progress_labels.jsonl \
  --output_jsonl vln_choice/data_processed/step_manifest_with_progress.jsonl
```

Adds one new field per matchable row:

```json
"progress_reasoning": {
  "subgoal_index": 1,
  "subgoal": {"text": "descend the stairs", "entities": ["stair", "step"], "action": "descend"},
  "chunk_view": [3, 5],
  "alignment_score": 0.9,
  "confidence": null,
  "source": "v1_dp"
}
```

Lookup is by `instr_id` (parsed from `sample_id`); the chunk that contains `path_pos = step_idx + 1` wins. The script prints `attached / no_progress_for_step / unparseable_sample_id` counters at the end.

- `--strict` — drop rows missing progress instead of passing them through unchanged.

### 1.5 Build composite images

Stitches the 36 panorama tiles at the current node into a single grid image.

```bash
python vln_choice/scripts/processing/build_composite_images.py \
  --input_jsonl vln_choice/data_processed/step_manifest_with_progress.jsonl \
  --output_jsonl vln_choice/data_processed/choice_samples.jsonl \
  --image_dir vln_choice/composite_images \
  --cols 12 \
  --tile_size 224
```

Behavior:
- Output image: `{image_dir}/{sample_id}.jpg`.
- Adds `image` (relative path) and `image_mode` (`view_grid`) fields; preserves all other fields including `progress_reasoning`.
- `--cols 12` → 3×12 grid for 36 tiles. Per-tile `view_XX` overlay can be turned off with `--no_labels`.
- `--view_indices 0,8,16,24,32` to use a subset of view buckets.
- Candidate / next-step layouts are **not** drawn on the image — candidate metadata stays in the JSONL.

### 1.6 Prepare SFT samples

**Plain mode** (legacy single-key target — kept for backward comparison):

```bash
python vln_choice/scripts/processing/prepare_sft_jsonl.py \
  --input_jsonl vln_choice/data_processed/choice_samples.jsonl \
  --output_jsonl vln_choice/data_processed/qwen_sft.jsonl
```

Each row's `response` becomes `{"final_choice": "B"}`.

**EGAC mode** (structured target — see §2):

```bash
python vln_choice/scripts/processing/prepare_sft_jsonl.py \
  --input_jsonl vln_choice/data_processed/choice_samples.jsonl \
  --output_jsonl vln_choice/data_processed/qwen_sft_egac.jsonl \
  --egac
```

`--egac` requires both `current.viewpoint_landmarks` (from §1.2) and `progress_reasoning` (from §1.4). Rows missing either are skipped, with a `skipped_egac_missing=N` line in the summary. The script also prints the `commitment_state` distribution — see §2.4.

`--egac` and `--with_cot` are mutually exclusive.

### 1.7 Validate

```bash
python vln_choice/scripts/processing/validate_samples.py \
  --jsonl vln_choice/data_processed/qwen_sft_egac.jsonl
```

---

## 2. Evidence-Gated Action Commitment (EGAC)

EGAC extends the SFT target so the model learns **whether visual evidence supports committing to the next target action**, instead of just emitting a candidate label. The intent is to reduce *premature commitment* failures — picking the action of an upcoming subgoal before its required evidence is visible.

### 2.1 SFT target schema

Produced by `egac_response()` in [`vln_choice/prompts.py`](vln_choice/prompts.py); field order is fixed so JSON diffs and tokenization stay deterministic.

```json
{
  "target_action": "descend the stairs",
  "required_evidence": ["stair", "step"],
  "observed_evidence": ["dining room", "chandelier"],
  "commitment_state": "defer | prepare | commit | stop",
  "next_action": "move toward stair",
  "final_choice": "B"
}
```

The matching prompt is built by `build_egac_prompt()` and explicitly tells the model: *"if evidence is insufficient, prefer approaching/preparing rather than executing the action prematurely."*

### 2.2 Four-state derivation rule

In [`prepare_sft_jsonl.py::_derive_commitment_state`](scripts/processing/prepare_sft_jsonl.py):

| Condition | State |
|---|---|
| `final_choice == "STOP"` | **`stop`** |
| `required_evidence` non-empty AND `normalize(observed) ∩ normalize(required) == ∅` | **`defer`** |
| step is the last in the subgoal's `chunk_view` (`path_pos == chunk_view[1]`) | **`commit`** |
| otherwise | **`prepare`** |

Normalization for the overlap test uses `normalize_phrase` ([`vln_choice/progress/normalize.py`](vln_choice/progress/normalize.py)) — lowercase, drop articles/stopwords, collapse aliases (`stair` / `step` / `staircase` → `stairs`).

`next_action` is derived as:
- `target_action` for `commit` / `stop`
- `"move toward " + required_evidence[0]` for `defer` / `prepare`

### 2.3 EGAC field sources

| Field | Source |
|---|---|
| `target_action` | `progress_reasoning.subgoal.text` |
| `required_evidence` | `progress_reasoning.subgoal.entities` |
| `observed_evidence` | `current.viewpoint_landmarks` (§1.2) |
| `commitment_state` | derived (table above) |
| `next_action` | derived |
| `final_choice` | existing `target.final_choice` |

### 2.4 Output sanity check

`prepare_sft_jsonl.py --egac` prints, e.g.:

```text
wrote 12345 rows to vln_choice/data_processed/qwen_sft_egac.jsonl
skipped_egac_missing=12 (rows lacking progress_reasoning or viewpoint_landmarks)
commitment_state distribution: {'defer': 1234, 'prepare': 5678, 'commit': 4321, 'stop': 89}
```

If any state is near-zero, the data is unbalanced. Common causes:

- Most subgoals have empty `entities` → `defer` rule never fires (it requires non-empty `required_evidence`). Fix at the splitter prompt (§4.3) or post-process entities.
- `chunk_view` covers only 1 step per subgoal → every step is the chunk's last step → all `commit`, no `prepare`. Inspect a few `progress_reasoning.chunk_view` values.
- `--limit` was too small — distribution is just noisy, not skewed.
- Many rows missed progress (`skipped_egac_missing` is high) → check that step manifest and progress labels were built from the same `--r2r_json`.

---

## 3. Training & inference

### 3.1 SFT training

```bash
python vln_choice/scripts/train/train_sft.py \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --train_jsonl vln_choice/data_processed/qwen_sft_egac.jsonl \
  --output_dir vln_choice/outputs/checkpoints/qwen_vl_egac_lora
```

LoRA via TRL `SFTTrainer`. Start with **3B on V100**; move to 7B / 8B only after the data loop and EGAC distribution look healthy.

### 3.2 Smoke test (no training needed)

Verifies the model loaders and inference path on a default local `Qwen3-VL-{4B,8B}-Instruct` under `data/models/`:

```bash
bash vln_choice/scripts/evaluation/smoke_qwen_vl.sh
# Optional overrides:
#   IMAGE=/path/to.jpg PROMPT="..." MAX_IMAGE_SIDE=768 bash ...
```

Equivalent Python: `python vln_choice/scripts/evaluation/smoke_qwen_vl.py --both` (uses the first `vln_choice/composite_images/*.jpg` if present, else a synthetic RGB).

### 3.3 Single-step inference

```bash
python vln_choice/scripts/evaluation/infer_step.py [...]
```

One composite image + prompt → JSON parse via `parse_final_choice`.

### 3.4 Rollout evaluation (placeholder)

[`scripts/evaluation/rollout_eval.py`](scripts/evaluation/rollout_eval.py) is intentionally a **stage-2 placeholder**. Wire it only after step-level choice accuracy and STOP behavior are stable.

### 3.5 Qwen-VL shared library

[`vln_choice/qwen_vl/`](vln_choice/qwen_vl/) — single source of truth for model handling so Qwen2.5-VL / Qwen3-VL stay consistent across training and inference:

| Module | Purpose |
|---|---|
| `loaders.py` | `load_processor`, `load_vlm` |
| `inference.py` | `generate_from_image_prompt`, `resize_for_inference` |
| `sft_format.py` | `format_example_for_sft` (HF messages shape) |

Training and inference scripts only parse CLI args and call into this package.

---

## 4. Progress labels — detail

`build_progress_labels.py` builds v1 progress labels with deterministic landmark alignment after LLM subgoal split (default: [4Z API](https://4zapi.com)).

`chunk_view` entries are **`[start_path_pos, last_path_pos]` (closed, inclusive)** on the expert path, 1-based.

### 4.1 Two input modes

| Source flag | Behavior |
|---|---|
| `--step_manifest_jsonl` (recommended) | Group manifest rows by `sample_id` (`<instr_id>_step_<NNN>`) into one expert path per instruction; `scan` is inferred from `current.view_dir`. Per-step landmarks come **only** from the manifest's expert-action candidate `view_landmark`, split by `;` / `,`. Missing manifest rows yield empty `landmarks` with a tagged `landmark_source` (no t2t fallback). |
| `--r2r_json` (no manifest) | One row per `instructions × path` in the R2R JSON. Per-step landmarks come from `landmark_phrases_for_direction(t2t, current_vp → expert_next_vp, view_index)`. Terminal step usually has empty landmarks tagged `terminal_node_full`. |

### 4.2 Two `--progress_mode` variants

- **`candidate`** (default) — per-step granularity. Build `step_evidence` from per-step landmarks (above), then call the splitter to get ordered `subgoals` and `step_subgoal_index`; `chunk_view` is derived via `chunk_view_for_subgoals(...)`. Monotonic-clip on `subgoal_index` is on by default; disable with `--no_candidate_monotonic_clip`.
- **`dp`** (legacy) — gold-path **window + DP** on full-node landmarks (`align_subgoals_topk`). No per-step expert-edge semantics, no LLM `step_evidence` alignment branch.

After changing mode or interval semantics, **regenerate** `r2r_progress_labels.jsonl`.

### 4.3 LLM splitter prompt

For `--splitter api_4z`, the joint `split_and_align` call (single round-trip) outputs both `subgoals` and `step_subgoal_index`. The user prompt (`build_joint_split_and_alignment_user_prompt` in [`vln_choice/progress/splitter.py`](vln_choice/progress/splitter.py)) constrains the model to:

> Split the original instruction into fine-grained sub-instructions. Do not assume the first step already satisfies the main action. For each action, check whether its required target object is visible or plausible from the landmarks at the assigned path positions. If an action requires object **O** but **O** does not appear in early landmarks and appears later, assign earlier positions to sub-instructions that describe **approaching/reaching O** rather than performing the core action there. Avoid unsupported search actions such as "find O" unless the original instruction explicitly says to search/find/look for **O**. Return `subgoals` (concise text, entities, action types) and `step_subgoal_index`; `chunk_view` is derived downstream.

The standalone split-only prompt (`build_subgoal_split_user_prompt`, used when no per-step landmarks are available) reinforces: don't fabricate search/find, don't assume the first step already completes the main goal, "approach before stairs" first.

### 4.4 4Z API config

Default model id: `gpt-5.4-mini` (override with `FOURZ_MODEL` / `--api_model`). Read settings from env so keys stay out of `ps`:

```bash
export FOURZ_API_BASE="https://<console-base>/v1"
export FOURZ_API_KEY="..."
# optional: export FOURZ_MODEL="gpt-5.4-mini"

python vln_choice/scripts/processing/build_progress_labels.py \
  --step_manifest_jsonl vln_choice/data_processed/r2r_step_manifest.jsonl \
  --output_jsonl vln_choice/data_processed/r2r_progress_labels.jsonl \
  --t2t_landmark_dir data/t2t_landmarks \
  --splitter api_4z \
  --progress_mode candidate
```

Equivalent env aliases: `API_4Z_BASE`, `API_4Z_KEY`, `API_4Z_MODEL`. Optional auth header via `FOURZ_AUTH_HEADER` / `API_4Z_AUTH_HEADER`. CLI overrides: `--api_base`, `--api_key`, `--api_model` / `--model_name_or_path`.

Wrapper script: `bash vln_choice/scripts/processing/run_build_progress_labels_4z.sh`.

Fast no-model smoke (heuristic splitter, no LLM):

```bash
python vln_choice/scripts/processing/build_progress_labels.py \
  --r2r_json data/R2R/R2R_val_seen_enc.json \
  --output_jsonl vln_choice/data_processed/r2r_progress_labels.jsonl \
  --splitter heuristic \
  --limit 10
```

### 4.5 Resume / errors / logging

- **Resume**: `--resume` skips `instr_id` already in `output_jsonl` and **appends** new rows. First (non-resume) run truncates the output file. Each completed row is `append_jsonl_record`'d, so a Ctrl-C does not lose finished rows.
- **Read timeouts**: per-call HTTP timeout default **120 s** (`FOURZ_TIMEOUT_SEC` / `--api_timeout_sec`); transient timeouts retried up to **3** times with short backoff (`FOURZ_CHAT_RETRIES`).
- **Long outputs**: bump `--split_max_new_tokens`. Joint response uses `_extract_json_object` to handle markdown-wrapped JSON.
- **Fallback**: `source_detail.llm_alignment_source == "llm_split_and_alignment_fallback"` indicates the joint API failed (network, read timeout, invalid JSON, etc.). Heuristic split + first-subgoal-for-all-steps is used. Check `llm_error`.
- **Stderr**: `INFO ... LLM ok` per success, `WARNING ... LLM fallback` per fallback, end-of-run `INFO ... summary` (`written / llm_ok / llm_fallback / resume_skipped`).

---

## 5. Optional: geometry-only candidate dump

Export per-neighbor geometry (angular alignment, view pose, relative / bearing angles, optional Euclidean distance) without t2t landmarks. One line per navigable edge; uses the same R2R path vocabulary as `build_expert_viewpoint_candidate_landmarks.py` to decide which `(scan, viewpoint)` nodes to visit.

```bash
python vln_choice/scripts/dump_collect_candidates.py \
  --connectivity_dir data/connectivity \
  --scan_dir /root/mount/Matterport3DSimulator/data/v1/scans \
  --r2r_json data/R2R/FGR2R_train.json \
  --output_jsonl vln_choice/data_processed/collect_candidates_dump.jsonl
```

Merge with step-level instruction context:

```bash
python vln_choice/scripts/dump_collect_candidates.py \
  --connectivity_dir data/connectivity \
  --scan_dir /root/mount/Matterport3DSimulator/data/v1/scans \
  --step_manifest_jsonl vln_choice/data_processed/step_manifest.jsonl \
  --output_jsonl vln_choice/data_processed/collect_candidates_with_step_context.jsonl
```

When `--step_manifest_jsonl` is set, output is **one line per step** (not per edge), with a `candidates` array and `action` label. Use this merged file as candidate-edge source for `--candidate_context_jsonl`:

```bash
python vln_choice/scripts/processing/build_progress_labels.py \
  --r2r_json data/R2R/R2R_val_seen_enc.json \
  --output_jsonl vln_choice/data_processed/r2r_progress_labels.jsonl \
  --t2t_landmark_dir data/t2t_landmarks \
  --candidate_context_jsonl vln_choice/data_processed/collect_candidates_with_step_context.jsonl
```

Example merged step record:

| Group | JSON path | Meaning |
|---|---|---|
| Sample key | `id`, `scene_id` | step sample id and scan id |
| Instruction | `instruction.id`, `instruction.text` | id + text |
| Current state | `state.*` | current step, history, expert next viewpoint |
| Candidates | `candidates[]` | per-candidate geometry |
| Supervision | `action` | gold option letter (`A` / `B` / ... / `STOP`) |

```json
{
  "id": "r2r_6250_0_step_001",
  "scene_id": "17DRP5sb8fy",
  "instruction": {"id": "r2r_6250_0", "text": "Walk down one flight of stairs and stop on the landing."},
  "state": {
    "step": 1,
    "current_viewpoint": "5be145994f97...",
    "history": ["af3af33b...", "5be145994f97..."],
    "expert_next_viewpoint": "79aedad..."
  },
  "candidates": [
    {"label": "A", "viewpoint_id": "...", "view_index": 20, "nav_order": 1,
     "pose_rad": {"relative": [-0.42, 0.03], "bearing": [0.81, 0.11], "misalignment": 0.31},
     "distance_m": 1.8},
    {"label": "B", "viewpoint_id": "...", "view_index": 22, "nav_order": 3,
     "pose_rad": {"relative": [-0.05, 0.11], "bearing": [1.18, 0.19], "misalignment": 0.12},
     "distance_m": 2.4}
  ],
  "action": "B"
}
```

---

## 6. Notes & known pitfalls

- **Start small**: `Qwen2.5-VL-3B-Instruct` on V100 first; 7B/8B only after the data loop is validated.
- **First training run**: try plain `{"final_choice": "..."}` (no `--egac`) to confirm step-level accuracy beats random. Then enable `--egac` and check that `commitment_state` accuracy improves alongside `final_choice` accuracy.
- **Imbalanced commitment_state**: see §2.4 — most often a splitter / `entities` issue.
- **Re-rendering candidates is slow**; for "I just want progress merged into the manifest", always use [`attach_progress_to_manifest.py`](scripts/processing/attach_progress_to_manifest.py) (§1.4), not `build_r2r_step_manifest.py --progress_labels_jsonl`.
- After changing landmark policy or splitter/aligner code, **regenerate** `r2r_progress_labels.jsonl` to avoid stale alignments.
- `vln_choice/data_processed/` and `vln_choice/composite_images/` are gitignored — they are large derived artifacts; only the code under `vln_choice/` (and `vln_choice/configs/`, `vln_choice/scripts/`) is tracked.

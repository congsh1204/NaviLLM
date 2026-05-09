# VLN Choice Prototype

This is an isolated prototype for choice-style VLM navigation:

```text
Composite image + instruction + optional history
-> Qwen2.5-VL
-> {"final_choice": "A/B/C/D/STOP"}
```

It intentionally does not replace the existing NaviLLM `train.py` / `NavModel` path. The first goal is to validate step-level choice accuracy before wiring online rollout.

**Scripts are grouped by role:** data **`processing/`**, **`train/`**, **`evaluation/`** — see [`scripts/README.md`](scripts/README.md).

## MVP Flow

1. Render MP3D 36-view RGB images with MatterSim:

```bash
python vln_choice/scripts/processing/render_mp3d_views.py \
  --connectivity_dir data/connectivity \
  --scan_dir /root/mount/Matterport3DSimulator/data/v1/scans  \
  --output_dir vln_choice/data_processed/mp3d_views \
  --limit 100
```

2. Build a step manifest JSONL.

For an R2R MVP, build it from expert trajectories:

```bash
python vln_choice/scripts/processing/build_r2r_step_manifest.py \
  --r2r_json data/R2R/FGR2R_train.json \
  --connectivity_dir data/connectivity \
  --scan_dir /root/mount/Matterport3DSimulator/data/v1/scans \
  --rendered_view_dir vln_choice/data_processed/mp3d_views \
  --t2t_landmark_dir data/t2t_landmarks \
  --output_jsonl vln_choice/data_processed/r2r_step_manifest.jsonl\
  --limit 10
```

Each row should contain:

```json
{
  "sample_id": "traj_001_step_000",
  "instruction": "Walk to the hallway.",
  "history": ["vp_start", "vp_001", "vp_002"],
  "current_view_dir": "vln_choice/data_processed/mp3d_views/17DRP5sb8fy/00ebbf...",
  "candidate_image_paths": {
    "A": "vln_choice/data_processed/mp3d_views/17DRP5sb8fy/cand_vp_a/view_00.jpg",
    "B": "vln_choice/data_processed/mp3d_views/17DRP5sb8fy/cand_vp_b/view_00.jpg",
    "STOP": null
  },
  "option_mapping": {
    "A": "candidate_viewpoint_a",
    "B": "candidate_viewpoint_b",
    "STOP": "STOP"
  },
  "target": {
    "final_choice": "B"
  }
}
```

Remove `--limit` after the first smoke test. The builder skips samples whose current or candidate viewpoint images have not been rendered yet.
By default, all navigable candidates are kept. Add `--max_candidates 4` only if you want a smaller A/B/C/D-style MVP.
Candidates that map to the same 36-view `point_id` are deduplicated by default, with the expert next viewpoint taking priority. This avoids ambiguous labels such as `A/C` on the same marked panorama tile. Add `--keep_duplicate_point_ids` only if you need the original full candidate set for analysis.

3. Build composite images:

```bash
python vln_choice/scripts/processing/build_composite_images.py \
  --input_jsonl vln_choice/data_processed/step_manifest_dedup_point_ids.jsonl \
  --output_jsonl vln_choice/data_processed/choice_samples.jsonl \
  --image_dir vln_choice/composite_images \
  --cols 12 \
  --tile_size 224
```

**中文说明（合成训练/推理用的大图）**

- **目的**：把 manifest 里当前节点的 **多视角 RGB** 按行列 **拼成一张 JPEG**，并在输出的 JSONL 里写入该图的相对路径字段 `image`，以及 `image_mode`（固定为 `view_grid`），供下一步 `prepare_sft_jsonl.py` 组 SFT 样本。
- **输入 `--input_jsonl`**：step manifest；每行需能解析出一组视角图（默认 `current_view_dir/view_00.jpg` … `view_35.jpg`，共 36 张；也可用 `panorama_views` 给出路径列表）。
- **输出 `--image_dir`**：合成图目录，默认 `vln_choice/composite_images`；文件名为 `{sample_id}.jpg`。**输出 `--output_jsonl`**：在原始字段基础上增加 `image`、`image_mode`。
- **网格**：`--cols` 控制每行列数（默认 `12`，36 张图为 **3×12**）。每格默认带 **`view_XX` 角标**；纯拼图不加字可用 **`--no_labels`**。
- **子集**：`--view_indices` 为逗号分隔索引，只加载并拼接这些视角（顺序与列表一致）。
- **其它**：`--tile_size` 默认 `224`；`--limit` 只处理前 N 条便于冒烟。

By default this script **only** stitches viewpoint tiles into one grid; candidate/next-step layouts are **not** drawn on the image (candidate metadata stays in the JSONL).

4. Add prompts and assistant responses:

```bash
python vln_choice/scripts/processing/prepare_sft_jsonl.py \
  --input_jsonl vln_choice/data_processed/choice_samples.jsonl \
  --output_jsonl vln_choice/data_processed/qwen_sft.jsonl
```

5. Validate:

```bash
python vln_choice/scripts/processing/validate_samples.py \
  --jsonl vln_choice/data_processed/qwen_sft.jsonl
```

6. Train Qwen2.5-VL with LoRA:

```bash
python vln_choice/scripts/train/train_sft.py \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --train_jsonl vln_choice/data_processed/qwen_sft.jsonl \
  --output_dir vln_choice/outputs/checkpoints/qwen_vl_choice_lora
```

### Qwen-VL code layout & one-click smoke (aligned with NaviLLM `scripts/evaluation/`)

- **Shared library** (`vln_choice/vln_choice/qwen_vl/`): single place for **loading** (`load_processor`, `load_vlm`), **inference** (`generate_from_image_prompt`, `resize_for_inference`), and **SFT message shape** (`format_example_for_sft`). Training and inference scripts only parse CLI args and call into this package so Qwen2.5-VL / Qwen3-VL stay consistent.
- **Training entry**: `vln_choice/scripts/train/train_sft.py` — dataset + `SFTTrainer` only; base weights via the same loaders as inference.
- **Inference entry**: `vln_choice/scripts/evaluation/infer_step.py` — one composite image + prompt + optional `parse_final_choice`.
- **Evaluation-style wrappers** (same spirit as `NaviLLM/scripts/evaluation/eval_*.sh`): `vln_choice/scripts/evaluation/smoke_qwen_vl.sh` runs a **default local** `Qwen3-VL-4B-Instruct` and `Qwen3-VL-8B-Instruct` under `data/models/` (skip if a folder is missing). From **NaviLLM repo root**:

```bash
bash vln_choice/scripts/evaluation/smoke_qwen_vl.sh
# Optional: IMAGE=/path/to.jpg PROMPT="..." MAX_IMAGE_SIDE=768 bash vln_choice/scripts/evaluation/smoke_qwen_vl.sh
```

Equivalent Python: `python vln_choice/scripts/evaluation/smoke_qwen_vl.py --both` (uses first `vln_choice/composite_images/*.jpg` if present, else a tiny synthetic RGB image).

## Notes

- Start with `Qwen/Qwen2.5-VL-3B-Instruct` on V100. Move to 7B only after the data loop is validated.
- First train only `{"final_choice": "..."}`. Add candidate-wise evidence after single-step accuracy is above random.
- `vln_choice/scripts/evaluation/rollout_eval.py` is intentionally a second-stage placeholder. It should be wired only after step-level choice accuracy and STOP behavior are stable.

## Optional: geometry-only candidate dump

Export **MatterSim** per-neighbor geometry (angular alignment, view pose, relative / bearing angles, optional edge Euclidean distance) as JSONL—**one line per navigable edge**, no t2t landmarks. Uses the same R2R path vocabulary as `build_expert_viewpoint_candidate_landmarks.py` to decide which `(scan, viewpoint)` nodes to visit.

```bash
python vln_choice/scripts/dump_collect_candidates.py \
  --connectivity_dir data/connectivity \
  --scan_dir /root/mount/Matterport3DSimulator/data/v1/scans \
  --r2r_json data/R2R/FGR2R_train.json \
  --output_jsonl vln_choice/data_processed/collect_candidates_dump.jsonl
```

Merge with step-level instruction context (from `step_manifest.jsonl`):

```bash
python vln_choice/scripts/dump_collect_candidates.py \
  --connectivity_dir data/connectivity \
  --scan_dir /root/mount/Matterport3DSimulator/data/v1/scans \
  --step_manifest_jsonl vln_choice/data_processed/step_manifest.jsonl \
  --output_jsonl vln_choice/data_processed/collect_candidates_with_step_context.jsonl
```

When `--step_manifest_jsonl` is set, output is **one line per step** (not per edge), with a `candidates` array and `action` label.

Use this merged file as the candidate edge source for progress-label `candidate` mode:

```bash
python vln_choice/scripts/processing/build_progress_labels.py   --r2r_json data/R2R/R2R_val_seen_enc.json   --output_jsonl vln_choice/data_processed/r2r_progress_labels.jsonl   --t2t_landmark_dir data/t2t_landmarks   --candidate_context_jsonl vln_choice/data_processed/collect_candidates_with_step_context.jsonl
```

**Example: one merged step record**

| 分组 | JSON 路径 | 含义 |
|------|-------------|------|
| 样本主键 | `id`, `scene_id` | step 样本 id 与 scan id |
| 指令 | `instruction.id`, `instruction.text` | 指令标识与文本 |
| 当前状态 | `state.*` | 当前 step、历史、专家下一跳 |
| 候选集合 | `candidates[]` | 当前步每个候选动作（含几何） |
| 监督动作 | `action` | gold 选项字母（如 `A/B/.../STOP`） |

```json
{
  "id": "r2r_6250_0_step_001",
  "scene_id": "17DRP5sb8fy",
  "instruction": {
    "id": "r2r_6250_0",
    "text": "Walk down one flight of stairs and stop on the landing."
  },
  "state": {
    "step": 1,
    "current_viewpoint": "5be145994f974347850a48cecd04cdcd",
    "history": [
      "af3af33b0120469c9a00daa0d0b36799",
      "5be145994f974347850a48cecd04cdcd"
    ],
    "expert_next_viewpoint": "79aedad1206b4eea9c4b639ea2182eb7"
  },
  "candidates": [
    {
      "label": "A",
      "viewpoint_id": "293a1e64b85e4e84a3a44fe8ee9d9bd4",
      "view_index": 20,
      "nav_order": 1,
      "pose_rad": {
        "relative": [-0.42, 0.03],
        "bearing": [0.81, 0.11],
        "misalignment": 0.31
      },
      "distance_m": 1.8
    },
    {
      "label": "B",
      "viewpoint_id": "79aedad1206b4eea9c4b639ea2182eb7",
      "view_index": 22,
      "nav_order": 3,
      "pose_rad": {
        "relative": [-0.05, 0.11],
        "bearing": [1.18, 0.19],
        "misalignment": 0.12
      },
      "distance_m": 2.4
    }
  ],
  "action": "B"
}
```

## Progress Labels

Build v1 progress labels with deterministic landmark alignment after LLM subgoal split (**[4Z API](https://4zapi.com)** by default). **`chunk_view`** pairs are **`[start_path_pos, last_path_pos]` inclusive** on the expert path.

### 过程标签建立逻辑（`build_progress_labels.py`）

脚本按 **`--progress_mode`** 走两条主干；推荐流程用 **`candidate`** + **`--step_manifest_jsonl`**（与当前 step manifest 管线一致）。

1. **读入与轨迹**
   - **`--step_manifest_jsonl`**：按 `sample_id`（`<instr_id>_step_<NNN>`）把同一指令的多步合并成一条 expert **path**（取最长 `history` 作为轨迹）；`scan` 从 `current.view_dir` 路径推断。
   - **`--r2r_json`（未指定 manifest 时）**：按 R2R 条目展开 `instructions` × `path`，一条导航指令对应一行。

2. **逐步证据 `step_evidence`（喂给 LLM 的 steps）**
   - **有 `--step_manifest_jsonl`**：每一步的地标 **只**来自 manifest 里 **expert 动作对应候选**的 `view_landmark`（按 `;`/`,` 拆成短语写入 `landmarks`）；同步写入该候选的 `view_index` → `matter_sim_point_id`。缺失 manifest 行时用空地标并标记 `landmark_source`（**不再**回退到 t2t 文件）。
   - **仅有 `--r2r_json`**：每一步调用 `landmark_phrases_for_direction`，从 **`--t2t_landmark_dir`** 按当前视点 → expert 下一视点（及可选 `view_index`）取短语；终点一步通常为 `terminal_node_full`、空 `landmarks`。

3. **子目标拆分 + 步—子目标对齐（candidate + `--splitter api_4z`，默认）**
   - **`FourZApiSplitter.split_and_align`**：**单次**远程调用，模型同时输出有序 **`subgoals`** 与 **`step_subgoal_index`**（长度等于 steps）。用户消息里会约束：在 **原指令 + expert 路径上每一步的 `landmarks` 序列** 前提下做细粒度拆分；**不要默认第一步已完成主任务**；若某动作依赖的目标物体 **O** 在前期地标中从未出现、在后期才出现，则早期步应对齐到「接近 / 抵达 O」类子目标，而非在早期步就当作已完成该动作；**不要随意添加「寻找 O」**（除非原指令明确写了 search/find/look for）。详见下文「LLM prompt 要点」。
   - 默认开启沿路径的 **`subgoal_index` 单调不减**裁剪（可用 **`--no_candidate_monotonic_clip`** 关闭）。
   - **`chunk_view`** 由 **`chunk_view_for_subgoals(step_subgoal_index, …)`** 从步级赋值 **推导**，不要求模型单独输出 `chunk_view` JSON 字段（保持与实现一致）。
   - 若 splitter 不支持联合调用：可能退化为 **先拆分子目标再逐步对齐**（`align_steps`）；若仅用 **`--splitter heuristic`**：不走 LLM 对齐，而用 **`align_expert_candidate_landmarks`** 在 t2t 上做确定性打分对齐（此时 `step_progress` 的地标语义与 manifest 模式不同，见 `source_detail.landmark_resolution`）。

### LLM prompt 要点（`split_and_align` / `align_steps`，`vln_choice/progress/splitter.py`）

联合调用时的英文指令核心如下（实现已写入 `build_joint_split_and_alignment_user_prompt`）：

> Given the original navigation instruction and the landmark sequence along the expert path, split the instruction into fine-grained sub-instructions. Do not assume the first step already satisfies the main action. For each action, check whether its required target object is visible or plausible from the landmarks at the path positions assigned to that phase. If an action requires object **O** but **O** does not appear in early landmarks and appears later, assign earlier positions to sub-instructions that describe **approaching/reaching O** rather than performing the core action there. Avoid unsupported search actions such as "find O" unless the original instruction explicitly says to search/find/look for **O**. Return **`subgoals`** (concise text, entities, action types) and **`step_subgoal_index`**; **`chunk_view`** is derived downstream from those indices.

仅在拆分、不带逐步地标时（`build_subgoal_split_user_prompt`）也会弱化写入：**勿臆造 search/find**、**勿假设首步已完成主目标**、楼梯类先接近再上下。

若 `source_detail.llm_alignment_source` 为 **`llm_split_and_alignment_fallback`**，说明 **联合 API 未成功**（常见：网络、**读超时**、JSON 不合法、模型包 markdown）。脚本会回退到启发式拆分 + 全路径第一步子目标。请查看 **`llm_error`**。单次 HTTP 读超时默认 **120s**（`FOURZ_TIMEOUT_SEC` / `--api_timeout_sec` 可加大）；**`_chat`** 对瞬时可恢复的读超时默认 **重试 3 次**（`FOURZ_CHAT_RETRIES`），带短退避。输出过长可增大 **`--split_max_new_tokens`**。代码已对联合响应使用 **`_extract_json_object`**。

4. **输出 JSONL 各字段含义（概要）**
   - **`new_instructions`**：子目标列表（`text` / `entities` / `action`）。
   - **`chunk_view`**：与子目标顺序一一对应的区间列表 `[[start, end], ...]`。
   - **`step_progress`**：每步的专家边、地标、`subgoal_index`、`matter_sim_point_id` 等（manifest 模式下地标与 `r2r_step_manifest.jsonl` 中 expert 候选一致）。
   - **`source_detail`**：记录所用 splitter、对齐器、`step_manifest_jsonl` 路径、是否单调裁剪等。

5. **`dp` 模式（`--progress_mode dp`）**  
   不用逐步 expert 边语义；对整条 path 的节点级地标做 **滑动窗口 + DP**（`align_subgoals_topk`），不再填充上面的 LLM `step_evidence` 对齐分支。

修改地标策略或拆分/对齐实现后，应 **重新生成** `r2r_progress_labels.jsonl`，避免旧文件与代码不一致。

- **断点续跑**：`--resume` 会读取已有 `output_jsonl` 里出现过的 `instr_id` 并 **跳过**，新结果 **追加** 到文件末尾。每写好一行即 **落盘**（`append_jsonl_record`），中断后不会丢已完成的行。非 resume 的首次跑会 **清空** 输出文件再写。
- **日志**：stderr 上每条 **LLM 成功** 会打印 `INFO ... LLM ok`；**fallback** 仍打印 `WARNING ... LLM fallback`。结束前有一条 **`INFO ... summary`**（written / llm_ok / llm_fallback / resume_skipped）。

**Progress modes (`--progress_mode`)**

- **`candidate` (default)** — 以 expert path 上每一步为粒度：构建 **`step_evidence`**（地标来源见上文「逐步证据」），再由 LLM 或启发式对齐器给出 **`step_subgoal_index`** → **`chunk_view`**。可选 **`--no_candidate_monotonic_clip`**。详情见 **`step_progress`** 与 **`source_detail`**。
- **`dp`** — Legacy gold-path **window + DP** over node landmarks (`align_subgoals_topk`), no per-step expert-edge semantics.

Regenerate JSONL after changing mode or interval semantics.

Fast no-model smoke test:

```bash
python vln_choice/scripts/processing/build_progress_labels.py \
  --r2r_json data/R2R/R2R_val_seen_enc.json \
  --output_jsonl vln_choice/data_processed/r2r_progress_labels.jsonl \
  --splitter heuristic \
  --limit 10
```

Default: remote **[4Z API](https://4zapi.com)** (`--splitter api_4z` is the default). Set Base URL / key from the console; prefer env so the key does not show up in `ps`.

推荐（`candidate` 模式，且 prompt 证据优先使用 `step_manifest` 的候选地标）：

```bash
python vln_choice/scripts/processing/build_progress_labels.py \
  --step_manifest_jsonl vln_choice/data_processed/r2r_step_manifest.jsonl \
  --output_jsonl vln_choice/data_processed/r2r_progress_labels.jsonl \
  --t2t_landmark_dir data/t2t_landmarks \
  --splitter api_4z \
  --api_base "$FOURZ_API_BASE" \
  --api_key "$FOURZ_API_KEY" \
  --api_model "${FOURZ_MODEL:-gpt-5.4-mini}" \
  --progress_mode candidate
```

也可直接运行封装脚本：

```bash
bash vln_choice/scripts/processing/run_build_progress_labels_4z.sh
```

**初步尝试（子目标拆分）**：控制台若有 `gpt-5.4-mini`，可先只用 Base URL + Key；脚本默认 `--model_name_or_path gpt-5.4-mini`，等价于在未设置 `FOURZ_MODEL` / `--api_model` 时用该模型。名称以你控制台里的模型 id 为准（若有别名请改成一致）。

```bash
export FOURZ_API_BASE="https://<copy-from-4zapi-console>/v1"
export FOURZ_API_KEY="..."
# 可选；不写则用内置默认模型 id（见下）
# export FOURZ_MODEL="gpt-5.4-mini"

python vln_choice/scripts/processing/build_progress_labels.py \
  --r2r_json data/R2R/R2R_val_seen_enc.json \
  --output_jsonl vln_choice/data_processed/r2r_progress_labels.jsonl \
  --t2t_landmark_dir data/t2t_landmarks \
  --limit 10
```

显式指定模型（与默认相同时也推荐写在 env 里便于切换）：

```bash
export FOURZ_MODEL="gpt-5.4-mini"
python vln_choice/scripts/processing/build_progress_labels.py \
  --r2r_json data/R2R/R2R_val_seen_enc.json \
  --output_jsonl vln_choice/data_processed/r2r_progress_labels.jsonl \
  --t2t_landmark_dir data/t2t_landmarks \
  --limit 10
```

或仅用命令行：`--api_model gpt-5.4-mini`。

其它模型：设置 `FOURZ_MODEL`（或 `API_4Z_MODEL`）为控制台中的模型 id，也可用 `--api_model` / `--model_name_or_path` 覆盖默认的 `gpt-5.4-mini`。

等价 env 别名：`API_4Z_BASE`、`API_4Z_KEY`、`API_4Z_MODEL`。可选：`FOURZ_AUTH_HEADER` / `API_4Z_AUTH_HEADER`，或 `--api_base`、`--api_key`。

Attach progress labels to step-manifest rows:

```bash
python vln_choice/scripts/processing/build_r2r_step_manifest.py \
  --r2r_json data/R2R/FGR2R_train.json \
  --connectivity_dir data/connectivity \
  --scan_dir /root/mount/Matterport3DSimulator/data/v1/scans \
  --rendered_view_dir vln_choice/data_processed/mp3d_views \
  --progress_labels_jsonl vln_choice/data_processed/r2r_progress_labels.jsonl \
  --output_jsonl vln_choice/data_processed/step_manifest_with_progress.jsonl
```


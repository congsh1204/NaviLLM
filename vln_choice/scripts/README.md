# `vln_choice/scripts` layout

Scripts are grouped by role (aligned with NaviLLM’s separation of **data / train / eval**):

| Directory | Purpose |
|-----------|---------|
| **`processing/`** | **数据处理**：从仿真视图、R2R/manifest、合成图到 SFT JSONL、进度标签等；不含模型训练或评测指标。 |
| **`train/`** | **训练**：Qwen-VL LoRA SFT 入口。 |
| **`evaluation/`** | **测试 / 推理 / 冒烟**：单步推理、本地 Qwen3-VL 一键冒烟、占位 rollout；与 `NaviLLM/scripts/evaluation/` 风格一致（薄壳脚本 + 明确入口）。 |

## `processing/` — data pipeline

| Script | Role |
|--------|------|
| `render_mp3d_views.py` | MatterSim / MP3D 渲染多视角 RGB |
| `build_r2r_step_manifest.py` | 从 R2R 等构造 step manifest JSONL |
| `run_build_r2r_step_manifest.sh` | 一键重跑 step manifest（env 覆盖 R2R 路径、scan、输出、limit 等）；默认 Docker 下 scan 路径。**进度标签**在 manifest 之后由 `build_progress_labels.py`（读 manifest）生成；本脚本只写 manifest，不包含进度标签步骤。 |
| `build_expert_viewpoint_candidate_landmarks.py` | Expert 候选视点与地标 |
| `build_composite_images.py` | 多视角合成大图 |
| `prepare_sft_jsonl.py` | 从 choice_samples 写训练 JSONL |
| `validate_samples.py` | **校验**中间 JSONL（路径、字段），非模型精度评测 |
| `build_progress_labels.py` | Progress labels（子目标 + chunk_view；API/heuristic） |
| `count_progress_labels_gap.py` | 对比 manifest 与 `r2r_progress_labels.jsonl`，统计未生成的 **instr_id** 数量，可选写出缺失列表 |
| `run_build_progress_labels_4z.sh` | 调用 4Z API 构建 progress labels 的 bash 封装 |

Run from **NaviLLM repo root** (paths in commands assume `python vln_choice/scripts/processing/...`).

**Compatibility:** `vln_choice/scripts/run_build_progress_labels_4z.sh` forwards to `processing/run_build_progress_labels_4z.sh` (use if older docs/scripts reference the short path).

## `train/` — training

| Script | Role |
|--------|------|
| `train_sft.py` | LoRA SFT on `qwen_sft.jsonl` style data |

## `evaluation/` — inference & smoke tests

| Script | Role |
|--------|------|
| `infer_step.py` | 单张 composite + prompt → 模型输出 + `parse_final_choice` |
| `infer_qwen3_scene.sh` | 带 manifest/t2t 元数据的场景推理示例（调用 `infer_step.py`） |
| `smoke_qwen_vl.py` / `smoke_qwen_vl.sh` | 本地 **Qwen3-VL-4B / 8B** 一键冒烟 |
| `rollout_eval.py` | 在线 rollout **占位**（第二阶段再接 MatterSim） |

---

**迁移说明**：若你仍有旧文档里的路径（如 `vln_choice/scripts/build_progress_labels.py`），请改为 `vln_choice/scripts/processing/build_progress_labels.py`，推理改为 `vln_choice/scripts/evaluation/infer_step.py`，训练改为 `vln_choice/scripts/train/train_sft.py`。

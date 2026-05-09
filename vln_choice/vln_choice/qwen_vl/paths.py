"""Paths relative to the NaviLLM repository root (parent of the `vln_choice/` subfolder)."""

from pathlib import Path


def navillm_repo_root() -> Path:
    # vln_choice/vln_choice/qwen_vl/paths.py -> …/NaviLLM
    return Path(__file__).resolve().parents[3]


def default_qwen3_vl_4b_path() -> Path:
    return navillm_repo_root() / "data/models/Qwen3-VL-4B-Instruct"


def default_qwen3_vl_8b_path() -> Path:
    return navillm_repo_root() / "data/models/Qwen3-VL-8B-Instruct"


def default_composite_images_dir() -> Path:
    return navillm_repo_root() / "vln_choice/composite_images"

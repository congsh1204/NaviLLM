"""Shared Qwen-VL / Qwen3-VL loaders, inference, and SFT message formatting for vln_choice."""

from vln_choice.qwen_vl.loaders import load_processor, load_vlm, qwen_vl_upgrade_hint
from vln_choice.qwen_vl.inference import generate_from_image_prompt, resize_for_inference
from vln_choice.qwen_vl.sft_format import format_example_for_sft
from vln_choice.qwen_vl.paths import (
    navillm_repo_root,
    default_qwen3_vl_4b_path,
    default_qwen3_vl_8b_path,
)

__all__ = [
    "load_processor",
    "load_vlm",
    "qwen_vl_upgrade_hint",
    "generate_from_image_prompt",
    "resize_for_inference",
    "format_example_for_sft",
    "navillm_repo_root",
    "default_qwen3_vl_4b_path",
    "default_qwen3_vl_8b_path",
]

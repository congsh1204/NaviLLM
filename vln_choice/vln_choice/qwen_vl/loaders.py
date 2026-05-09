"""Load processor and VLM weights — shared by training and inference entrypoints."""

from __future__ import annotations

import transformers


def qwen_vl_upgrade_hint() -> str:
    return (
        "If this is Qwen3-VL, upgrade transformers in the container, for example:\n"
        "pip install -U transformers accelerate qwen-vl-utils"
    )


def load_processor(model_name_or_path: str):
    from transformers import AutoProcessor

    errors = []
    try:
        return AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    except Exception as exc:
        errors.append("AutoProcessor: {}".format(repr(exc)))

    processor_classes = [
        "Qwen3VLProcessor",
        "Qwen2_5_VLProcessor",
        "Qwen2VLProcessor",
    ]
    for class_name in processor_classes:
        processor_cls = getattr(transformers, class_name, None)
        if processor_cls is None:
            errors.append("{}: not available in transformers {}".format(class_name, transformers.__version__))
            continue
        try:
            return processor_cls.from_pretrained(model_name_or_path, trust_remote_code=True)
        except Exception as exc:
            errors.append("{}: {}".format(class_name, repr(exc)))

    raise RuntimeError(
        "Could not load processor from {} with transformers {}.\n{}\n{}".format(
            model_name_or_path,
            transformers.__version__,
            "\n".join(errors),
            qwen_vl_upgrade_hint(),
        )
    )


def load_vlm(model_name_or_path: str):
    """Prefer explicit Qwen3 class when available, then Auto* wrappers (same order as infer_step)."""
    errors = []
    priority = []
    qwen3_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    if qwen3_cls is not None:
        priority.append(("Qwen3VLForConditionalGeneration", qwen3_cls))
    for class_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"):
        model_cls = getattr(transformers, class_name, None)
        if model_cls is not None:
            priority.append((class_name, model_cls))

    for class_name, model_cls in priority:
        try:
            return model_cls.from_pretrained(
                model_name_or_path,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception as exc:
            errors.append("{}: {}".format(class_name, repr(exc)))

    raise RuntimeError(
        "Could not load VLM from {} with transformers {}.\n{}\n{}".format(
            model_name_or_path,
            transformers.__version__,
            "\n".join(errors),
            qwen_vl_upgrade_hint(),
        )
    )

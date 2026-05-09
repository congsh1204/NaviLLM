"""Vision-language generation used by `infer_step` and smoke tests."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from PIL import Image


def resize_for_inference(image: Image.Image, max_image_side: int) -> Image.Image:
    if max_image_side <= 0:
        return image
    width, height = image.size
    longest = max(width, height)
    if longest <= max_image_side:
        return image
    scale = max_image_side / float(longest)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.BICUBIC)


def generate_from_image_prompt(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    *,
    max_new_tokens: int = 128,
) -> str:
    """
    One-turn user message: image + text (same structure as SFT user turn in `format_example_for_sft`).
    """
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    new_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(new_ids, skip_special_tokens=True)[0]


def load_image_for_smoke(
    image_path: Optional[str],
    composite_dir,
) -> Tuple[Image.Image, str]:
    """Return (PIL image, description of source)."""
    from pathlib import Path

    if image_path:
        p = Path(image_path)
        im = Image.open(p).convert("RGB")
        return im, str(p)

    d = Path(composite_dir)
    if d.is_dir():
        for p in sorted(d.glob("*.jpg")):
            im = Image.open(p).convert("RGB")
            return im, str(p)
        for p in sorted(d.glob("*.png")):
            im = Image.open(p).convert("RGB")
            return im, str(p)

    im = Image.new("RGB", (448, 448), color=(64, 120, 200))
    return im, "synthetic_placeholder"

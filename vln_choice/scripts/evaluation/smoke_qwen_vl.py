#!/usr/bin/env python
"""One-shot or sequential smoke test for local Qwen3-VL checkpoints (mirrors `scripts/evaluation/*` style)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_pkg_path() -> Path:
    """`vln_choice/` folder that contains the `vln_choice` Python package (same as other scripts)."""
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    return root


_PKG_ROOT = _ensure_pkg_path()

from vln_choice.qwen_vl.inference import (  # noqa: E402
    generate_from_image_prompt,
    load_image_for_smoke,
    resize_for_inference,
)
from vln_choice.qwen_vl.loaders import load_processor, load_vlm  # noqa: E402
from vln_choice.qwen_vl.paths import (  # noqa: E402
    default_composite_images_dir,
    default_qwen3_vl_4b_path,
    default_qwen3_vl_8b_path,
)

try:
    from vln_choice.parser import parse_final_choice
except Exception:  # pragma: no cover
    parse_final_choice = None


def _run_one(
    model_path: Path,
    image,
    image_source: str,
    args: argparse.Namespace,
) -> dict:
    if not model_path.is_dir():
        return {
            "model": str(model_path),
            "ok": False,
            "error": "path not found or not a directory",
        }
    out = {
        "model": str(model_path),
        "image_source": image_source,
        "ok": True,
    }
    try:
        processor = load_processor(str(model_path))
        model = load_vlm(str(model_path))
        orig = image.size
        image = resize_for_inference(image, args.max_image_side)
        if image.size != orig:
            out["resized_from"] = "{}x{}".format(orig[0], orig[1])
            out["resized_to"] = "{}x{}".format(image.size[0], image.size[1])
        text = generate_from_image_prompt(
            model,
            processor,
            image,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
        )
        out["output"] = text
        if parse_final_choice is not None and args.choices:
            out["parsed_final_choice"] = parse_final_choice(text, args.choices.split(","))
    except Exception as exc:
        out["ok"] = False
        out["error"] = repr(exc)
    return out


def main():
    parser = argparse.ArgumentParser(description="Smoke-test local Qwen-VL / Qwen3-VL weights (4B+8B one-click).")
    parser.add_argument(
        "--both",
        action="store_true",
        help="Run default 4B then 8B under data/models/ (skip missing dirs).",
    )
    parser.add_argument("--model_name_or_path", default=None, help="Single checkpoint directory.")
    parser.add_argument("--image", default=None, help="Composite image; default first *.jpg under composite_images.")
    parser.add_argument("--prompt", default="Briefly describe the scene and any navigation cues you see.")
    parser.add_argument("--choices", default="A,B,C,D,STOP", help="For optional parse_final_choice.")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_image_side", type=int, default=1024)
    parser.add_argument("--json_out", action="store_true", help="Print one JSON object per model line.")
    args = parser.parse_args()

    comp_dir = default_composite_images_dir()
    image, image_src = load_image_for_smoke(args.image, comp_dir)

    if args.both:
        paths = [default_qwen3_vl_4b_path(), default_qwen3_vl_8b_path()]
    elif args.model_name_or_path:
        paths = [Path(args.model_name_or_path).expanduser()]
        if not paths[0].is_absolute():
            paths[0] = (_PKG_ROOT.parent / paths[0]).resolve()
    else:
        parser.error("Provide --both or --model_name_or_path")

    results = []
    for mp in paths:
        if not mp.is_dir():
            row = {"model": str(mp), "ok": False, "error": "skipped (missing)"}
            results.append(row)
            continue
        results.append(_run_one(mp, image, image_src, args))

    def _all_runs_ok(rows: list) -> bool:
        ran = [r for r in rows if r.get("error") != "skipped (missing)"]
        if not ran:
            return False
        return all(r.get("ok") for r in ran)

    if args.json_out:
        for row in results:
            print(json.dumps(row, ensure_ascii=False))
        sys.exit(0 if _all_runs_ok(results) else 1)

    ok_all = True
    for row in results:
        print("=== {} ===".format(row.get("model")))
        if row.get("error") == "skipped (missing)":
            print("skipped (checkpoint directory missing)")
            continue
        if not row.get("ok"):
            print("FAILED:", row.get("error", row))
            ok_all = False
            continue
        print(row.get("output", ""))
        if row.get("parsed_final_choice") is not None:
            print("parsed_final_choice={}".format(row["parsed_final_choice"]))
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()

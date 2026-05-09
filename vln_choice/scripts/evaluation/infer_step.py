#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vln_choice.parser import parse_final_choice
from vln_choice.qwen_vl.inference import generate_from_image_prompt, resize_for_inference
from vln_choice.qwen_vl.loaders import load_processor, load_vlm


def main():
    parser = argparse.ArgumentParser(description="Run one composite-image navigation choice inference.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--choices", default="A,B,C,D,STOP")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_image_side", type=int, default=0, help="Resize image so its longest side is at most this value. Use 0 to disable.")
    parser.add_argument("--metadata_json", default=None, help="Optional JSON file with target metadata for reporting correctness.")
    args = parser.parse_args()

    processor = load_processor(args.model_name_or_path)
    model = load_vlm(args.model_name_or_path)

    from PIL import Image

    image = Image.open(args.image).convert("RGB")
    original_size = image.size
    image = resize_for_inference(image, args.max_image_side)
    if image.size != original_size:
        print("resized_image={}x{} -> {}x{}".format(original_size[0], original_size[1], image.size[0], image.size[1]))

    output = generate_from_image_prompt(
        model,
        processor,
        image,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
    )
    print(output)
    parsed_final_choice = parse_final_choice(output, args.choices.split(","))
    print("parsed_final_choice={}".format(parsed_final_choice))
    if args.metadata_json:
        with open(args.metadata_json, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        target = metadata.get("target", {}).get("final_choice")
        if target is not None:
            print("target_final_choice={}".format(target))
            print("is_correct={}".format(parsed_final_choice == target))


if __name__ == "__main__":
    main()

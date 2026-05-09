#!/usr/bin/env python
"""Stitch multiple viewpoint RGB images into one JPEG grid (no candidate tiles / STOP bands)."""

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vln_choice.composite import stitch_views_to_grid
from vln_choice.io import read_jsonl, write_jsonl


def _load_views(sample, view_indices=None):
    """Load RGB images in order: optional subset of indices into the default 36-view naming."""
    if "panorama_views" in sample:
        paths = sample["panorama_views"]
    else:
        view_dir = Path(sample["current_view_dir"])
        paths = [str(view_dir / "view_{:02d}.jpg".format(i)) for i in range(36)]
    if view_indices is not None:
        paths = [paths[i] for i in view_indices if 0 <= i < len(paths)]
    return [Image.open(path).convert("RGB") for path in paths]


def main():
    parser = argparse.ArgumentParser(
        description="Stitch panorama viewpoint images into one grid JPEG per JSONL row.",
    )
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", default="vln_choice/data_processed/choice_samples.jsonl")
    parser.add_argument("--image_dir", default="vln_choice/composite_images")
    parser.add_argument("--tile_size", type=int, default=224)
    parser.add_argument(
        "--cols",
        type=int,
        default=12,
        help="Number of columns in the grid (36 views -> 12 cols => 3 rows).",
    )
    parser.add_argument(
        "--view_indices",
        default=None,
        help="Comma-separated indices into panorama_views / view_XX.jpg order (default: all available).",
    )
    parser.add_argument(
        "--no_labels",
        action="store_true",
        help="Do not draw view_XX corner labels on tiles.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    view_indices = None
    if args.view_indices:
        view_indices = [int(x) for x in args.view_indices.split(",") if x.strip()]

    rows = []
    for idx, sample in enumerate(read_jsonl(args.input_jsonl)):
        if args.limit is not None and idx >= args.limit:
            break
        sample_id = sample.get("sample_id", "sample_{:08d}".format(idx))
        output_path = Path(args.image_dir) / "{}.jpg".format(sample_id)
        views = _load_views(sample, view_indices=view_indices)
        image_path = stitch_views_to_grid(
            views,
            str(output_path),
            cols=args.cols,
            tile_size=args.tile_size,
            label_each=not args.no_labels,
        )
        sample["image"] = image_path
        sample["image_mode"] = "view_grid"
        rows.append(sample)
    write_jsonl(args.output_jsonl, rows)
    print("wrote {} samples to {}".format(len(rows), args.output_jsonl))


if __name__ == "__main__":
    main()

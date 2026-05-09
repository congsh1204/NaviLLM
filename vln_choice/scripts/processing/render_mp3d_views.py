#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vln_choice.render import build_simulator, render_36_views, save_views


def load_scan_viewpoints(connectivity_dir: str, limit: int = None):
    pairs = []
    with open(os.path.join(connectivity_dir, "scans.txt"), "r", encoding="utf-8") as f:
        scans = [line.strip() for line in f if line.strip()]
    for scan in scans:
        path = os.path.join(connectivity_dir, "{}_connectivity.json".format(scan))
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            if item.get("included"):
                pairs.append((scan, item["image_id"]))
                if limit is not None and len(pairs) >= limit:
                    return pairs
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Render canonical MP3D 36-view RGB images for VLM choice training.")
    parser.add_argument("--connectivity_dir", default="data/connectivity")
    parser.add_argument("--scan_dir", default="/home/pcl/Matterport3DData/v1/scans")
    parser.add_argument("--output_dir", default="vln_choice/data_processed/mp3d_views")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    pairs = load_scan_viewpoints(args.connectivity_dir, args.limit)
    sim = build_simulator(args.connectivity_dir, args.scan_dir)
    for idx, (scan, viewpoint) in enumerate(pairs, 1):
        out_dir = Path(args.output_dir) / scan / viewpoint
        if (out_dir / "view_35.jpg").exists():
            continue
        images = render_36_views(sim, scan, viewpoint)
        save_views(images, str(out_dir))
        if idx % 20 == 0:
            print("rendered {}/{} viewpoints".format(idx, len(pairs)))
    print("done: {}".format(args.output_dir))


if __name__ == "__main__":
    main()


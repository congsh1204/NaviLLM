#!/usr/bin/env python
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Placeholder for online VLN rollout using the Qwen-VL choice policy."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--max_steps", type=int, default=15)
    args = parser.parse_args()
    raise SystemExit(
        "rollout_eval is intentionally a second-stage integration script. "
        "First validate step-level samples and single-step choice accuracy. "
        "Then wire this script to MP3DAgent/MatterSim env with max_steps={} split={}.".format(
            args.max_steps, args.split
        )
    )


if __name__ == "__main__":
    main()


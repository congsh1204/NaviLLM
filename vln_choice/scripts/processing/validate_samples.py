#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vln_choice.io import read_jsonl


def validate(sample):
    errors = []
    if not Path(sample.get("image", "")).exists():
        errors.append("missing image")
    mapping = sample.get("option_mapping", {})
    valid = set(mapping.keys())
    if "STOP" not in valid:
        errors.append("missing STOP option")
    choice = sample.get("target", {}).get("final_choice")
    if choice not in valid:
        errors.append("invalid final_choice")
    prompt = sample.get("prompt")
    if prompt is not None and not isinstance(prompt, str):
        errors.append("prompt must be string")
    response = sample.get("response")
    if response:
        try:
            obj = json.loads(response)
            if obj.get("final_choice") not in valid:
                errors.append("response final_choice invalid")
        except json.JSONDecodeError:
            errors.append("response is not JSON")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    args = parser.parse_args()

    total = 0
    bad = 0
    for sample in read_jsonl(args.jsonl):
        total += 1
        errors = validate(sample)
        if errors:
            bad += 1
            print("{}: {}".format(sample.get("sample_id", total), "; ".join(errors)))
    print("validated={} bad={}".format(total, bad))
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()


#!/usr/bin/env python
"""Compare step-manifest instr_ids to progress-label JSONL rows (one label row per instr_id expected)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vln_choice.io import read_jsonl


def _instr_from_manifest_row(row: dict) -> str | None:
    sid = row.get("sample_id") or ""
    if "_step_" in sid:
        return sid.split("_step_", 1)[0]
    return row.get("instr_id")


def main():
    parser = argparse.ArgumentParser(
        description="Count instructions present in manifest but missing from progress_labels JSONL.",
    )
    parser.add_argument(
        "--step_manifest_jsonl",
        default="vln_choice/data_processed/r2r_step_manifest.jsonl",
        help="Manifest with sample_id like <instr_id>_step_NNN",
    )
    parser.add_argument(
        "--progress_labels_jsonl",
        default="vln_choice/data_processed/r2r_progress_labels.jsonl",
        help="Output of build_progress_labels.py (one row per instr_id)",
    )
    parser.add_argument(
        "--write_missing",
        default=None,
        help="Optional path: write one instr_id per line (missing only)",
    )
    parser.add_argument(
        "--limit_manifest_lines",
        type=int,
        default=None,
        help="Only scan first N lines of manifest (debug)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.step_manifest_jsonl)
    labels_path = Path(args.progress_labels_jsonl)

    expected: set[str] = set()
    for i, row in enumerate(read_jsonl(str(manifest_path))):
        if args.limit_manifest_lines is not None and i >= args.limit_manifest_lines:
            break
        rid = _instr_from_manifest_row(row)
        if rid:
            expected.add(str(rid))

    done: set[str] = set()
    if labels_path.is_file():
        for row in read_jsonl(str(labels_path)):
            rid = row.get("instr_id")
            if rid is not None:
                done.add(str(rid))

    missing = sorted(expected - done)
    n_exp = len(expected)
    n_done = len(done & expected)
    n_miss = len(missing)

    print(
        "manifest_instr_ids={} progress_labels_rows_matching_manifest={} missing={}".format(
            n_exp, n_done, n_miss
        )
    )
    print(
        "progress_labels_file_total_unique_instr_id={} (may include ids not in this manifest)".format(
            len(done)
        )
    )

    if args.write_missing and missing:
        Path(args.write_missing).parent.mkdir(parents=True, exist_ok=True)
        Path(args.write_missing).write_text("\n".join(missing) + "\n", encoding="utf-8")
        print("wrote {} missing instr_id(s) to {}".format(len(missing), args.write_missing))

    return 0 if n_miss == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Attach v1 progress labels onto an existing step manifest as a pure JSONL transform.

This replaces re-running ``build_r2r_step_manifest.py --progress_labels_jsonl`` when you
already have:

- ``r2r_step_manifest.jsonl`` (e.g. with the new ``current.viewpoint_landmarks`` field)
- ``r2r_progress_labels.jsonl`` (subgoals + step_progress + chunk_view)

It does **not** load MatterSim and does **not** re-render candidates, so it runs
anywhere and is dramatically faster. The output adds a ``progress_reasoning`` field to
each row, identical in shape to what ``build_r2r_step_manifest.progress_for_step``
produces.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vln_choice.io import read_jsonl, write_jsonl


_STEP_RE = re.compile(r"^(?P<instr>.+)_step_(?P<step>\d+)")


def _parse_sample_id(sample_id: str) -> Optional[Tuple[str, int]]:
    """Extract ``(instr_id, step_idx_0_based)`` from sample_id like ``r2r_6250_0_step_005[_stop]``."""
    m = _STEP_RE.match(sample_id or "")
    if not m:
        return None
    return m.group("instr"), int(m.group("step"))


def _index_progress_by_instr_id(progress_jsonl: str) -> Dict[str, Dict]:
    index: Dict[str, Dict] = {}
    for row in read_jsonl(progress_jsonl):
        instr_id = row.get("instr_id")
        if instr_id:
            index[instr_id] = row
    return index


def _progress_for_step(progress_row: Optional[Dict], step_idx: int) -> Optional[Dict]:
    """Find the chunk_view interval covering this 1-based step and return progress_reasoning."""
    if not progress_row:
        return None
    step_pos = step_idx + 1
    chunks = progress_row.get("chunk_view", []) or []
    subgoals = progress_row.get("new_instructions", []) or []
    for chunk_idx, chunk in enumerate(chunks):
        if not (isinstance(chunk, list) and len(chunk) == 2):
            continue
        start, end = int(chunk[0]), int(chunk[1])
        if start <= step_pos <= end:
            subgoal = subgoals[chunk_idx] if chunk_idx < len(subgoals) else None
            return {
                "subgoal_index": chunk_idx,
                "subgoal": subgoal,
                "chunk_view": chunk,
                "alignment_score": progress_row.get("alignment_score"),
                "confidence": progress_row.get("confidence"),
                "source": progress_row.get("source", "v1_dp"),
            }
    return None


def main():
    parser = argparse.ArgumentParser(description="Merge progress labels into step manifest (no MatterSim, no re-render).")
    parser.add_argument("--manifest_jsonl", required=True, help="Existing r2r_step_manifest.jsonl")
    parser.add_argument("--progress_labels_jsonl", required=True, help="r2r_progress_labels.jsonl")
    parser.add_argument("--output_jsonl", required=True, help="Output step_manifest_with_progress.jsonl")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Drop rows whose progress could not be attached, instead of passing them through unchanged.",
    )
    args = parser.parse_args()

    progress_index = _index_progress_by_instr_id(args.progress_labels_jsonl)
    print("loaded {} progress label entries from {}".format(len(progress_index), args.progress_labels_jsonl))

    n_rows = n_attached = n_no_match = n_unparseable = n_dropped = 0
    out_rows = []
    for sample in read_jsonl(args.manifest_jsonl):
        n_rows += 1
        parsed = _parse_sample_id(sample.get("sample_id", ""))
        if parsed is None:
            n_unparseable += 1
            if args.strict:
                n_dropped += 1
                continue
            out_rows.append(sample)
            continue
        instr_id, step_idx = parsed
        progress = _progress_for_step(progress_index.get(instr_id), step_idx)
        if progress is None:
            n_no_match += 1
            if args.strict:
                n_dropped += 1
                continue
        else:
            sample["progress_reasoning"] = progress
            n_attached += 1
        out_rows.append(sample)

    write_jsonl(args.output_jsonl, out_rows)
    print("wrote {} rows to {}".format(len(out_rows), args.output_jsonl))
    print("  attached={}  no_progress_for_step={}  unparseable_sample_id={}{}".format(
        n_attached,
        n_no_match,
        n_unparseable,
        "  dropped={}".format(n_dropped) if args.strict else "",
    ))


if __name__ == "__main__":
    main()

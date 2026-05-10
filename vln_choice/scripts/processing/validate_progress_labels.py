#!/usr/bin/env python
"""Offline validator for ``r2r_progress_labels.jsonl`` (Layer 1 + Layer 3 checks).

**Layer 1 — structural integrity** (each row must satisfy):
    L1_INSTR_ID_MISSING            instr_id absent or empty
    L1_NEW_INSTR_CHUNK_LEN_MISMATCH  len(new_instructions) != len(chunk_view)
    L1_CHUNK_INVALID               chunk_view[i] not [start, end] with start<=end and start>=1
    L1_CHUNKS_OVERLAP              two chunk_view intervals share a path_pos
    L1_CHUNKS_GAP                  a path_pos covered by step_progress is not inside any chunk
    L1_SUBGOAL_INDEX_OOR           step_progress[i].subgoal_index out of range
    L1_PATH_POS_DISORDER           step_progress not strictly increasing by path_pos
    L1_CHUNK_SUBGOAL_INCONSISTENT  chunk_view[sg].start..end != path_pos set where subgoal_index == sg

**Layer 3 — semantic heuristics** (signals of *probably wrong* labels):
    L3_SUBGOAL_NO_EVIDENCE_HIT     subgoal has non-empty entities, but no step in its chunk
                                   has landmarks overlapping those entities (after normalize_phrase)
    L3_EVIDENCE_PEAK_BEFORE_END    chunk has multi steps; the step with maximum entity overlap
                                   is strictly earlier than the chunk end — agent could/should
                                   commit earlier
    L3_ALL_LANDMARKS_EMPTY         no step_progress entry has any landmarks
    L3_ALL_STEPS_ONE_SUBGOAL       only one unique subgoal_index across the whole path
    L3_LLM_FALLBACK                source_detail.llm_alignment_source signals fallback

The script prints aggregate counts + a few example instr_ids per issue, and optionally
writes per-record diagnostics to ``--output_jsonl``.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vln_choice.io import read_jsonl, write_jsonl
from vln_choice.progress.normalize import normalize_phrase


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------


def _normalize_set(values) -> Set[str]:
    out: Set[str] = set()
    for v in values or []:
        if not isinstance(v, str):
            continue
        norm = normalize_phrase(v)
        if norm:
            out.add(norm)
    return out


def _entity_overlap_count(entities: List[str], landmarks: List[str]) -> int:
    return len(_normalize_set(entities) & _normalize_set(landmarks))


def _is_fallback_source(source_detail: Dict) -> bool:
    src = (source_detail or {}).get("llm_alignment_source")
    return isinstance(src, str) and "fallback" in src.lower()


# -----------------------------------------------------------------------------
# per-record checks
# -----------------------------------------------------------------------------


def _check_layer1(row: Dict, issues: List[str], details: Dict) -> None:
    instr_id = row.get("instr_id")
    if not instr_id:
        issues.append("L1_INSTR_ID_MISSING")

    new_instructions = row.get("new_instructions") or []
    chunk_view = row.get("chunk_view") or []
    step_progress = row.get("step_progress") or []

    if len(new_instructions) != len(chunk_view):
        issues.append("L1_NEW_INSTR_CHUNK_LEN_MISMATCH")
        details["len_mismatch"] = (len(new_instructions), len(chunk_view))

    # Validate chunk shapes
    valid_chunks: List[Tuple[int, int, int]] = []  # (sg_idx, start, end)
    for i, ch in enumerate(chunk_view):
        if not (isinstance(ch, list) and len(ch) == 2):
            issues.append("L1_CHUNK_INVALID")
            details.setdefault("invalid_chunks", []).append((i, ch))
            continue
        try:
            s, e = int(ch[0]), int(ch[1])
        except (TypeError, ValueError):
            issues.append("L1_CHUNK_INVALID")
            details.setdefault("invalid_chunks", []).append((i, ch))
            continue
        if not (s >= 1 and e >= s):
            issues.append("L1_CHUNK_INVALID")
            details.setdefault("invalid_chunks", []).append((i, [s, e]))
            continue
        valid_chunks.append((i, s, e))

    # Detect overlap between chunks (same path_pos in two intervals)
    pos_to_chunk: Dict[int, int] = {}
    overlapped = False
    for sg, s, e in valid_chunks:
        for p in range(s, e + 1):
            if p in pos_to_chunk:
                overlapped = True
                break
            pos_to_chunk[p] = sg
        if overlapped:
            break
    if overlapped:
        issues.append("L1_CHUNKS_OVERLAP")

    # step_progress order + subgoal_index range + chunk-vs-step consistency
    last_pos = 0
    disorder = False
    sg_for_pos: Dict[int, int] = {}
    for entry in step_progress:
        pp = entry.get("path_pos")
        if not isinstance(pp, int) or pp <= last_pos:
            disorder = True
            break
        last_pos = pp
        sg_idx = entry.get("subgoal_index")
        if isinstance(sg_idx, int):
            if not (0 <= sg_idx < max(1, len(new_instructions))):
                issues.append("L1_SUBGOAL_INDEX_OOR")
                details.setdefault("oor_subgoal_indices", []).append((pp, sg_idx))
            sg_for_pos[pp] = sg_idx
    if disorder:
        issues.append("L1_PATH_POS_DISORDER")

    # Gap: a step path_pos not inside any chunk
    if valid_chunks and sg_for_pos:
        for pp in sg_for_pos:
            if pp not in pos_to_chunk:
                issues.append("L1_CHUNKS_GAP")
                details.setdefault("gap_positions", []).append(pp)
                break

    # chunk_view interval == positions where subgoal_index == sg
    if valid_chunks and sg_for_pos and "L1_PATH_POS_DISORDER" not in issues:
        positions_by_sg: Dict[int, Set[int]] = defaultdict(set)
        for pp, sg in sg_for_pos.items():
            positions_by_sg[sg].add(pp)
        for sg, s, e in valid_chunks:
            expected = set(range(s, e + 1))
            actual = positions_by_sg.get(sg, set())
            if expected != actual:
                issues.append("L1_CHUNK_SUBGOAL_INCONSISTENT")
                details.setdefault("chunk_step_mismatch", []).append((sg, sorted(expected), sorted(actual)))
                break


def _check_layer3(row: Dict, issues: List[str], details: Dict) -> None:
    new_instructions = row.get("new_instructions") or []
    chunk_view = row.get("chunk_view") or []
    step_progress = row.get("step_progress") or []
    source_detail = row.get("source_detail") or {}

    if _is_fallback_source(source_detail):
        issues.append("L3_LLM_FALLBACK")

    # All landmarks empty
    if step_progress and not any((entry.get("landmarks") or []) for entry in step_progress):
        issues.append("L3_ALL_LANDMARKS_EMPTY")

    # All steps one subgoal
    sg_indices = {entry.get("subgoal_index") for entry in step_progress if isinstance(entry.get("subgoal_index"), int)}
    if len(sg_indices) <= 1 and len(new_instructions) > 1:
        issues.append("L3_ALL_STEPS_ONE_SUBGOAL")

    # Per-subgoal evidence checks
    pos_to_landmarks: Dict[int, List[str]] = {}
    for entry in step_progress:
        pp = entry.get("path_pos")
        if isinstance(pp, int):
            pos_to_landmarks[pp] = entry.get("landmarks") or []

    for sg, ch in enumerate(chunk_view):
        if not (isinstance(ch, list) and len(ch) == 2):
            continue
        try:
            s, e = int(ch[0]), int(ch[1])
        except (TypeError, ValueError):
            continue
        if sg >= len(new_instructions):
            continue
        entities = (new_instructions[sg] or {}).get("entities") or []
        if not entities:
            continue  # rule requires non-empty entities

        overlaps_per_pos = []
        for p in range(s, e + 1):
            lms = pos_to_landmarks.get(p, [])
            overlaps_per_pos.append((p, _entity_overlap_count(entities, lms)))
        if not overlaps_per_pos:
            continue

        if all(c == 0 for _, c in overlaps_per_pos):
            issues.append("L3_SUBGOAL_NO_EVIDENCE_HIT")
            details.setdefault("subgoal_no_hit", []).append({
                "subgoal_index": sg,
                "entities": entities,
                "chunk": [s, e],
                "landmarks_in_chunk": [pos_to_landmarks.get(p, []) for p in range(s, e + 1)],
            })
            continue

        # Evidence peak strictly before chunk end (only meaningful when chunk has >=2 positions)
        if e - s >= 1:
            peak_pos, peak_count = max(overlaps_per_pos, key=lambda kv: kv[1])
            end_count = overlaps_per_pos[-1][1]
            if peak_count > 0 and peak_pos < e and peak_count > end_count + 0:
                issues.append("L3_EVIDENCE_PEAK_BEFORE_END")
                details.setdefault("evidence_peak_before_end", []).append({
                    "subgoal_index": sg,
                    "chunk": [s, e],
                    "peak_pos": peak_pos,
                    "peak_count": peak_count,
                    "end_count": end_count,
                })


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Offline validator for r2r_progress_labels.jsonl (Layer 1 + Layer 3).")
    parser.add_argument("--progress_labels_jsonl", required=True)
    parser.add_argument(
        "--output_jsonl",
        default=None,
        help="Optional per-record diagnostic dump: {instr_id, issues, details}.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=5,
        help="How many example instr_ids to print per issue category.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only validate the first N rows (for smoke).",
    )
    args = parser.parse_args()

    issue_counter: Counter = Counter()
    issue_examples: Dict[str, List[str]] = defaultdict(list)
    subgoal_count_dist: Counter = Counter()
    path_len_dist: Counter = Counter()
    n_total = 0
    n_with_any_issue = 0
    n_fallback = 0

    out_rows: List[Dict] = []

    for row in read_jsonl(args.progress_labels_jsonl):
        if args.limit is not None and n_total >= args.limit:
            break
        n_total += 1

        instr_id = row.get("instr_id") or "<missing>"
        issues: List[str] = []
        details: Dict = {}
        _check_layer1(row, issues, details)
        _check_layer3(row, issues, details)

        if issues:
            n_with_any_issue += 1
        for code in set(issues):
            issue_counter[code] += 1
            if len(issue_examples[code]) < args.max_examples:
                issue_examples[code].append(instr_id)

        new_instructions = row.get("new_instructions") or []
        step_progress = row.get("step_progress") or []
        subgoal_count_dist[len(new_instructions)] += 1
        path_len_dist[len(step_progress)] += 1
        if "L3_LLM_FALLBACK" in issues:
            n_fallback += 1

        if args.output_jsonl is not None:
            out_rows.append({
                "instr_id": instr_id,
                "issues": sorted(set(issues)),
                "details": details,
            })

    if args.output_jsonl is not None:
        write_jsonl(args.output_jsonl, out_rows)

    # ---------- console summary ----------
    print()
    print("validate_progress_labels — summary")
    print("=" * 60)
    print("input:        {}".format(args.progress_labels_jsonl))
    print("records:      {}".format(n_total))
    print("with issue:   {}  ({:.1f}%)".format(n_with_any_issue, 100.0 * n_with_any_issue / max(1, n_total)))
    print("LLM fallback: {}  ({:.1f}%)".format(n_fallback, 100.0 * n_fallback / max(1, n_total)))
    print()

    print("subgoal count distribution:")
    for k in sorted(subgoal_count_dist):
        print("  subgoals={}: {} records".format(k, subgoal_count_dist[k]))
    print()

    print("path length distribution (top 10):")
    for k, v in path_len_dist.most_common(10):
        print("  steps={}: {} records".format(k, v))
    print()

    if not issue_counter:
        print("No issues detected.")
        return

    print("issue counts:")
    for code, n in sorted(issue_counter.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * n / max(1, n_total)
        examples = ", ".join(issue_examples[code])
        print("  {:<32} {:>6}  ({:>5.1f}%)  e.g. {}".format(code, n, pct, examples))

    if args.output_jsonl is not None:
        print("\nwrote per-record diagnostics to {}".format(args.output_jsonl))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vln_choice.io import read_jsonl, write_jsonl
from vln_choice.progress.normalize import normalize_phrase
from vln_choice.prompts import (
    build_choice_prompt,
    build_egac_prompt,
    egac_response,
    final_choice_response,
)


_STEP_IDX_RE = re.compile(r"_step_(\d+)")


def _parse_step_idx(sample_id: str) -> Optional[int]:
    """Extract the 0-based step index from a manifest sample_id like ``r2r_6250_0_step_001``."""
    m = _STEP_IDX_RE.search(sample_id or "")
    return int(m.group(1)) if m else None


def _normalize_set(values: List[str]) -> set:
    out = set()
    for v in values or []:
        norm = normalize_phrase(v)
        if norm:
            out.add(norm)
    return out


def _derive_commitment_state(
    observed_norm: set,
    required_norm: set,
    is_last_in_chunk: bool,
    final_choice: str,
) -> str:
    """4-state EGAC commitment: defer / prepare / commit / stop.

    - ``stop`` when the action label is STOP.
    - ``defer`` when there *is* required evidence but none of it is currently observed
      (the gating signal — agent should not yet commit to the action).
    - ``commit`` when this step is the last in the subgoal's chunk_view (the agent is about
      to execute / transition), and either there is evidence overlap or the subgoal carries
      no specific evidence requirements (e.g. generic "walk forward").
    - ``prepare`` otherwise — in-subgoal but more steps remain before commit.
    """
    if (final_choice or "").upper() == "STOP":
        return "stop"
    if required_norm and not (observed_norm & required_norm):
        return "defer"
    return "commit" if is_last_in_chunk else "prepare"


def _derive_next_action(
    commitment_state: str,
    target_action: str,
    required_evidence: List[str],
) -> str:
    if commitment_state == "stop":
        return "stop"
    if commitment_state == "commit":
        return target_action
    # defer / prepare: keep approaching the first piece of required evidence we still need.
    if required_evidence:
        return "move toward " + required_evidence[0]
    return target_action


def _extract_egac_fields(sample: Dict) -> Optional[Dict]:
    """Pull the 5 EGAC fields out of a manifest sample. Returns ``None`` if data is missing."""
    progress = sample.get("progress_reasoning")
    current = sample.get("current") or {}
    target = sample.get("target") or {}
    if not progress or "viewpoint_landmarks" not in current or "final_choice" not in target:
        return None

    subgoal = progress.get("subgoal") or {}
    target_action = subgoal.get("text") or ""
    required_evidence = list(subgoal.get("entities") or [])
    observed_evidence = list(current.get("viewpoint_landmarks") or [])
    chunk_view = progress.get("chunk_view") or []
    final_choice = target["final_choice"]

    step_idx = _parse_step_idx(sample.get("sample_id", ""))
    path_pos = (step_idx + 1) if step_idx is not None else None
    is_last_in_chunk = (
        path_pos is not None
        and isinstance(chunk_view, list)
        and len(chunk_view) == 2
        and int(chunk_view[1]) == path_pos
    )

    commitment_state = _derive_commitment_state(
        _normalize_set(observed_evidence),
        _normalize_set(required_evidence),
        is_last_in_chunk,
        final_choice,
    )
    next_action = _derive_next_action(commitment_state, target_action, required_evidence)

    return {
        "target_action": target_action,
        "required_evidence": required_evidence,
        "observed_evidence": observed_evidence,
        "commitment_state": commitment_state,
        "next_action": next_action,
        "final_choice": final_choice,
    }


def main():
    parser = argparse.ArgumentParser(description="Add Qwen-VL prompt/response fields to choice samples.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", default="vln_choice/data_processed/qwen_sft.jsonl")
    parser.add_argument("--with_cot", action="store_true")
    parser.add_argument(
        "--egac",
        action="store_true",
        help="Emit EGAC-style structured target instead of the plain final_choice. "
             "Requires manifest rows to carry both ``progress_reasoning`` and ``current.viewpoint_landmarks``.",
    )
    args = parser.parse_args()

    if args.egac and args.with_cot:
        parser.error("--egac and --with_cot are mutually exclusive (different output schemas)")

    rows = []
    skipped_egac_missing = 0
    state_counts: Dict[str, int] = {}

    for sample in read_jsonl(args.input_jsonl):
        labels = list(sample["option_mapping"].keys())

        if args.egac:
            fields = _extract_egac_fields(sample)
            if fields is None:
                skipped_egac_missing += 1
                continue
            sample["prompt"] = build_egac_prompt(
                sample["instruction"],
                labels,
                history=sample.get("history", []),
            )
            sample["response"] = egac_response(**fields)
            sample["egac_target"] = fields  # keep the parsed fields handy for downstream eval
            state_counts[fields["commitment_state"]] = state_counts.get(fields["commitment_state"], 0) + 1
        else:
            sample["prompt"] = build_choice_prompt(
                sample["instruction"],
                labels,
                history=sample.get("history", []),
                with_cot=args.with_cot,
            )
            if "response" not in sample:
                sample["response"] = final_choice_response(sample["target"]["final_choice"])

        rows.append(sample)

    write_jsonl(args.output_jsonl, rows)
    print("wrote {} rows to {}".format(len(rows), args.output_jsonl))
    if args.egac:
        print("skipped_egac_missing={} (rows lacking progress_reasoning or viewpoint_landmarks)".format(skipped_egac_missing))
        print("commitment_state distribution: {}".format(state_counts))


if __name__ == "__main__":
    main()


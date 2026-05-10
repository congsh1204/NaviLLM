#!/usr/bin/env python
"""LLM-as-judge for ``r2r_progress_labels.jsonl`` via the same 4Z API client used by
``build_progress_labels.py``.

For each record, ask the remote model to score the subgoal split + step alignment on
four 0-5 axes and write a one-line free-text critique. Aggregates mean/median/histogram
across all judged records and surfaces the worst N for inspection.

Output JSONL row shape (one per judged record):
    {"instr_id": "...", "scores": {"split_quality": 4, "alignment_quality": 5,
     "entity_quality": 4, "overall_quality": 4}, "notes": "...",
     "_judge_model": "<model id>"}

Failures (HTTP, timeout, malformed JSON) write a row with ``_llm_error`` instead of scores
so progress is preserved and you can re-run with ``--resume``.

Env vars (same as build_progress_labels.py):
    FOURZ_API_BASE / API_4Z_BASE
    FOURZ_API_KEY  / API_4Z_KEY
    FOURZ_MODEL    / API_4Z_MODEL  (recommend a stronger model than the splitter)

Cost note: this is one API call per record. Use ``--sample_n 100`` for a first pass.
"""

import argparse
import json
import random
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vln_choice.io import append_jsonl_record, read_jsonl
from vln_choice.progress.splitter import FourZApiSplitter, _extract_json_object


# -----------------------------------------------------------------------------
# prompt
# -----------------------------------------------------------------------------


JUDGE_SYSTEM_HINT = (
    "You evaluate navigation-instruction subgoal splits and step alignments."
)


def build_judge_user_prompt(row: Dict) -> str:
    instr = row.get("instruction") or ""
    new_instructions = row.get("new_instructions") or []
    chunk_view = row.get("chunk_view") or []
    step_progress = row.get("step_progress") or []

    sg_lines: List[str] = []
    for i, sg in enumerate(new_instructions):
        ch = chunk_view[i] if i < len(chunk_view) else None
        sg_lines.append(
            "  [{i}] steps={ch} text={text!r} entities={ents} action={act!r}".format(
                i=i,
                ch=ch,
                text=sg.get("text", ""),
                ents=sg.get("entities") or [],
                act=sg.get("action", ""),
            )
        )

    sp_lines: List[str] = []
    for entry in step_progress:
        sp_lines.append(
            "  pos={pp} sg={sg} landmarks={lms}".format(
                pp=entry.get("path_pos"),
                sg=entry.get("subgoal_index"),
                lms=entry.get("landmarks") or [],
            )
        )

    return (
        "Evaluate a navigation-instruction subgoal split and step alignment.\n"
        "\n"
        "Original instruction:\n"
        "  {instr}\n"
        "\n"
        "Proposed subgoals (chunk = inclusive path-position interval [start, end]):\n"
        "{sgs}\n"
        "\n"
        "Per-step expert path landmarks (what the agent observed at each path position):\n"
        "{steps}\n"
        "\n"
        "Score each axis 0-5 (5 = ideal, 0 = unusable; be strict — 5 means flawless, "
        "3 means acceptable but flawed):\n"
        "  - split_quality: the subgoal list faithfully covers the instruction without "
        "inventing or omitting steps; granularity is consistent with the original sentence.\n"
        "  - alignment_quality: each chunk's step range is reasonable given the per-step landmarks. "
        "Penalize when entities for subgoal X never appear in landmarks of its assigned steps, "
        "or when entities for subgoal X+1 already appear well before that subgoal's chunk starts.\n"
        "  - entity_quality: entities are concrete and specific (objects/places, not vague verbs); "
        "not invented; not omitted; matched at the right granularity for landmark matching.\n"
        "  - overall_quality: holistic judgment.\n"
        "\n"
        "Return ONLY a single valid JSON object, no markdown fences, no extra text:\n"
        '{{"split_quality": <int 0-5>, "alignment_quality": <int 0-5>, '
        '"entity_quality": <int 0-5>, "overall_quality": <int 0-5>, '
        '"notes": "<concise critique <= 30 words>"}}\n'
    ).format(instr=instr, sgs="\n".join(sg_lines), steps="\n".join(sp_lines))


# -----------------------------------------------------------------------------
# parsing + validation
# -----------------------------------------------------------------------------


_SCORE_KEYS = ("split_quality", "alignment_quality", "entity_quality", "overall_quality")


def _coerce_score(v) -> Optional[int]:
    try:
        n = int(round(float(v)))
    except (TypeError, ValueError):
        return None
    if not (0 <= n <= 5):
        return None
    return n


def parse_judge_response(content: str) -> Dict:
    """Return a dict with valid 'scores' + 'notes', else dict with '_llm_error'."""
    try:
        obj = _extract_json_object(content)
    except Exception as exc:  # noqa: BLE001
        return {"_llm_error": "json_parse: {}".format(exc)[:300]}
    if not isinstance(obj, dict):
        return {"_llm_error": "non-dict response"}

    scores: Dict[str, int] = {}
    for key in _SCORE_KEYS:
        coerced = _coerce_score(obj.get(key))
        if coerced is None:
            return {"_llm_error": "invalid score for {!r}: {!r}".format(key, obj.get(key))[:300]}
        scores[key] = coerced
    notes = obj.get("notes", "")
    if not isinstance(notes, str):
        notes = str(notes)
    return {"scores": scores, "notes": notes[:500]}


# -----------------------------------------------------------------------------
# resume helpers
# -----------------------------------------------------------------------------


def _load_existing_judged_ids(output_path: str) -> Set[str]:
    p = Path(output_path)
    if not p.exists():
        return set()
    out: Set[str] = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            instr_id = row.get("instr_id")
            if isinstance(instr_id, str):
                out.add(instr_id)
    return out


def _select_records(rows: List[Dict], args) -> List[Dict]:
    if args.instr_ids:
        wanted = {x.strip() for x in args.instr_ids.split(",") if x.strip()}
        return [r for r in rows if r.get("instr_id") in wanted]
    if args.sample_n is not None:
        rng = random.Random(args.seed)
        if args.sample_n >= len(rows):
            return list(rows)
        return rng.sample(rows, args.sample_n)
    if args.limit is not None:
        return rows[: args.limit]
    return rows


# -----------------------------------------------------------------------------
# aggregation
# -----------------------------------------------------------------------------


def _print_aggregate(judged: List[Dict], failed: List[Dict], bottom_n: int) -> None:
    print()
    print("judge_progress_labels — summary")
    print("=" * 60)
    print("judged ok:   {}".format(len(judged)))
    print("failed:      {}".format(len(failed)))
    if failed:
        err_kinds: Counter = Counter()
        for r in failed:
            err = r.get("_llm_error", "unknown")
            kind = re.split(r"[: ]", err, 1)[0] if err else "unknown"
            err_kinds[kind] += 1
        print("  error kinds:")
        for k, n in err_kinds.most_common():
            print("    {:<24} {}".format(k, n))

    if not judged:
        return

    for key in _SCORE_KEYS:
        vs = [r["scores"][key] for r in judged]
        print()
        print("{}:".format(key))
        print("  mean={:.2f}  median={:.1f}  min={}  max={}".format(
            statistics.fmean(vs), statistics.median(vs), min(vs), max(vs),
        ))
        hist = Counter(vs)
        bar_total = max(1, len(vs))
        for score in range(6):
            n = hist.get(score, 0)
            bar = "#" * int(round(40.0 * n / bar_total))
            print("    {}: {:>5}  {}".format(score, n, bar))

    if bottom_n > 0:
        worst = sorted(judged, key=lambda r: (r["scores"]["overall_quality"], r["scores"]["alignment_quality"]))
        print()
        print("worst {} records by overall_quality (for inspection):".format(min(bottom_n, len(worst))))
        for r in worst[:bottom_n]:
            scores = r["scores"]
            print("  {iid}  overall={oq} split={sq} align={aq} entity={eq}  notes={notes!r}".format(
                iid=r.get("instr_id", "?"),
                oq=scores["overall_quality"],
                sq=scores["split_quality"],
                aq=scores["alignment_quality"],
                eq=scores["entity_quality"],
                notes=r.get("notes", "")[:80],
            ))


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="LLM-as-judge for r2r_progress_labels.jsonl via 4Z API.")
    parser.add_argument("--progress_labels_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True, help="Per-record judge output (append-mode; resume-friendly).")
    parser.add_argument("--resume", action="store_true", help="Skip instr_ids already present in --output_jsonl.")

    parser.add_argument("--sample_n", type=int, default=None, help="Random-sample N records (mutually exclusive with --limit/--instr_ids).")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N records.")
    parser.add_argument("--instr_ids", default=None, help="Comma-separated instr_ids to judge (overrides sample/limit).")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--api_base", default=None, help="Override FOURZ_API_BASE / API_4Z_BASE.")
    parser.add_argument("--api_key", default=None, help="Override FOURZ_API_KEY / API_4Z_KEY.")
    parser.add_argument("--api_model", default=None, help="Override FOURZ_MODEL / API_4Z_MODEL.")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--api_timeout_sec", type=float, default=None)
    parser.add_argument("--chat_retries", type=int, default=None)

    parser.add_argument("--bottom_n", type=int, default=10, help="How many worst records to print at the end.")
    parser.add_argument("--print_every", type=int, default=20, help="Heartbeat: print progress every N judged records.")
    parser.add_argument("--sleep_between_calls", type=float, default=0.0, help="Optional pacing in seconds.")

    args = parser.parse_args()

    if sum(x is not None for x in (args.sample_n, args.limit, args.instr_ids)) > 1:
        parser.error("--sample_n / --limit / --instr_ids are mutually exclusive")

    # load + select
    all_rows = list(read_jsonl(args.progress_labels_jsonl))
    selected = _select_records(all_rows, args)
    if args.resume:
        already = _load_existing_judged_ids(args.output_jsonl)
        before = len(selected)
        selected = [r for r in selected if r.get("instr_id") not in already]
        print("resume: skipping {} already-judged records, {} remaining".format(before - len(selected), len(selected)))

    if not selected:
        print("nothing to judge.")
        return

    splitter = FourZApiSplitter(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.api_model,
        max_new_tokens=args.max_new_tokens,
        timeout_sec=args.api_timeout_sec,
        chat_retries=args.chat_retries,
    )

    print("judging {} records via {} ...".format(len(selected), splitter.model))

    judged: List[Dict] = []
    failed: List[Dict] = []

    for idx, row in enumerate(selected, start=1):
        instr_id = row.get("instr_id") or "<missing>"
        prompt = build_judge_user_prompt(row)
        out_row: Dict = {"instr_id": instr_id, "_judge_model": splitter.model}
        try:
            content = splitter._chat(prompt)
            parsed = parse_judge_response(content)
            out_row.update(parsed)
        except Exception as exc:  # noqa: BLE001
            out_row["_llm_error"] = "{}: {}".format(type(exc).__name__, exc)[:300]

        if "_llm_error" in out_row:
            failed.append(out_row)
            print("  WARN [{i}/{n}] {iid} failed: {err}".format(
                i=idx, n=len(selected), iid=instr_id, err=out_row["_llm_error"][:160],
            ), file=sys.stderr, flush=True)
        else:
            judged.append(out_row)

        append_jsonl_record(args.output_jsonl, out_row)

        if args.print_every and idx % args.print_every == 0:
            ok = len(judged)
            ko = len(failed)
            mean_overall = (
                statistics.fmean(r["scores"]["overall_quality"] for r in judged)
                if judged else float("nan")
            )
            print("  [{}/{}] ok={} fail={}  running mean overall_quality={:.2f}".format(
                idx, len(selected), ok, ko, mean_overall,
            ), flush=True)

        if args.sleep_between_calls > 0:
            time.sleep(args.sleep_between_calls)

    _print_aggregate(judged, failed, args.bottom_n)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vln_choice.io import append_jsonl_record
from vln_choice.progress.align_dp import align_subgoals_topk
from vln_choice.progress.candidate_progress import (
    align_expert_candidate_landmarks,
    chunk_view_for_subgoals,
    compact_subgoals_and_assignments,
)
from vln_choice.progress.landmarks import build_path_landmarks, landmark_phrases_for_direction
from vln_choice.progress.splitter import get_splitter, heuristic_split


def _warn_llm_fallback(instr_id: str, source: str, detail: Optional[str] = None) -> None:
    """Emit a visible stderr warning when the remote LLM path falls back to heuristics."""
    msg = "WARNING [build_progress_labels] LLM fallback instr_id={} source={}".format(instr_id, source)
    if detail:
        msg += " detail={}".format(detail[:800])
    print(msg, file=sys.stderr, flush=True)


def _info_llm_ok(instr_id: str, source: str) -> None:
    print(
        "INFO [build_progress_labels] LLM ok instr_id={} source={}".format(instr_id, source),
        file=sys.stderr,
        flush=True,
    )


def load_json_or_jsonl(path: str) -> List[Dict]:
    input_path = Path(path)
    if input_path.suffix == ".jsonl":
        rows = []
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    return json.loads(input_path.read_text(encoding="utf-8"))


def _classify_llm_error(detail: Optional[str]) -> str:
    """Bucket stderr/API errors for summary stats (502, timeout, …)."""
    if not detail:
        return "unknown"
    d = str(detail).lower()
    if "502" in d or "bad gateway" in d:
        return "http_502"
    if "503" in detail or "service unavailable" in d:
        return "http_503"
    if "504" in detail or "gateway timeout" in d:
        return "http_504"
    if re.search(r"\b5\d\d\b", detail) or "http 5" in d:
        return "http_5xx"
    if "timeout" in d or "timed out" in d:
        return "timeout"
    if "connection" in d or "refused" in d or "reset" in d:
        return "connection"
    if "json" in d or "parse" in d or "expected json" in d:
        return "parse"
    return "other"


def _load_instr_ids_file(path: str) -> Set[str]:
    """One instr_id per line, optional JSONL rows with \"instr_id\", lines starting with # skipped."""
    out: Set[str] = set()
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            try:
                rid = json.loads(line).get("instr_id")
                if rid is not None:
                    out.add(str(rid))
            except Exception:
                continue
        else:
            out.add(line)
    return out


def _rewrite_jsonl_drop_instr_ids(output_jsonl: str, drop_ids: Set[str]) -> int:
    """Remove rows whose instr_id is in drop_ids; returns number of lines removed."""
    p = Path(output_jsonl)
    if not p.is_file() or not drop_ids:
        return 0
    kept = []
    removed = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line
            line = line.strip()
            if not line:
                continue
            try:
                rid = json.loads(line).get("instr_id")
                if rid is not None and str(rid) in drop_ids:
                    removed += 1
                    continue
            except Exception:
                kept.append(raw)
                continue
            kept.append(raw)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.writelines(kept)
    tmp.replace(p)
    return removed


def _note_llm_fallback(
    instr_id: str,
    source: str,
    llm_error: Optional[str],
    manifest_path: Optional[Path],
    by_class: Counter,
) -> None:
    """Append one line to the fallback manifest and bump error-class counters."""
    cls = _classify_llm_error(llm_error)
    by_class[cls] += 1
    if manifest_path is not None:
        append_jsonl_record(
            str(manifest_path),
            {
                "instr_id": instr_id,
                "llm_alignment_source": source,
                "error_class": cls,
                "llm_error": (llm_error or "")[:2000],
            },
        )


def _load_done_instr_ids(output_jsonl: str) -> set:
    """Collect instr_id values already written to a JSONL file (for --resume)."""
    done = set()
    p = Path(output_jsonl)
    if not p.is_file():
        return done
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rid = json.loads(line).get("instr_id")
                if rid is not None:
                    done.add(rid)
            except Exception:
                continue
    return done


def iter_instruction_items(rows: Iterable[Dict]) -> Iterable[Dict]:
    for item in rows:
        instructions = item.get("instructions")
        if isinstance(instructions, str):
            instructions = [instructions]
        for instr_idx, instruction in enumerate(instructions or []):
            yield {
                "path_id": item.get("path_id", item.get("id", item.get("instr_id"))),
                "instr_idx": instr_idx,
                "instr_id": item.get("instr_id", "r2r_{}_{}".format(item.get("path_id", "unknown"), instr_idx)),
                "scan": item["scan"],
                "path": item["path"],
                "instruction": instruction,
                "raw_item": item,
            }


def _infer_scan_from_view_dir(view_dir: str):
    if not view_dir:
        return None
    parts = str(view_dir).split("/")
    if "mp3d_views" in parts:
        i = parts.index("mp3d_views")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def _parse_sample_id(sample_id: str):
    # expected: <instr_id>_step_<NNN> or <instr_id>_step_<NNN>_stop
    if not sample_id or "_step_" not in sample_id:
        return None, None
    instr_id, tail = sample_id.split("_step_", 1)
    step_str = tail.split("_", 1)[0]
    try:
        step_idx = int(step_str)
    except Exception:
        step_idx = None
    return instr_id, step_idx


def _build_precomputed_edge_index_from_step_manifest_rows(step_rows: List[Dict]) -> Dict[str, Dict[int, Dict[str, object]]]:
    out: Dict[str, Dict[int, Dict[str, object]]] = {}
    for row in step_rows:
        sample_id = row.get("sample_id")
        instr_id, step_idx = _parse_sample_id(sample_id)
        if not instr_id or step_idx is None:
            continue
        action = row.get("action") or {}
        if not isinstance(action, dict):
            continue
        action_label = action.get("label")
        if action_label == "STOP":
            continue
        chosen = None
        for cand in row.get("candidates", []) or []:
            if cand.get("label") == action_label:
                chosen = cand
                break
        if chosen is None:
            continue
        out.setdefault(instr_id, {})[step_idx] = {
            "expert_next_viewpoint": action.get("viewpoint_id"),
            "view_index": chosen.get("view_index"),
        }
    return out


def _split_landmark_text(text: str) -> List[str]:
    if not text:
        return []
    parts = []
    for piece in str(text).replace(";", ",").split(","):
        p = piece.strip()
        if p:
            parts.append(p)
    return parts


def _build_step_manifest_landmark_index(step_rows: List[Dict]) -> Dict[str, Dict[int, Dict[str, object]]]:
    out: Dict[str, Dict[int, Dict[str, object]]] = {}
    for row in step_rows:
        sample_id = row.get("sample_id")
        instr_id, step_idx = _parse_sample_id(sample_id)
        if not instr_id or step_idx is None:
            continue

        current = row.get("current") or {}
        action = row.get("action") or {}
        candidates = row.get("candidates") or []
        action_label = action.get("label") if isinstance(action, dict) else None
        chosen = None
        if action_label and action_label != "STOP":
            for cand in candidates:
                if cand.get("label") == action_label:
                    chosen = cand
                    break

        landmarks = _split_landmark_text((chosen or {}).get("view_landmark", ""))
        source = "step_manifest_action_candidate"
        if action_label == "STOP":
            source = "step_manifest_terminal"
        elif not landmarks:
            source = "step_manifest_action_candidate_empty"

        out.setdefault(instr_id, {})[step_idx] = {
            "viewpoint": current.get("viewpoint_id"),
            "expert_next_viewpoint": action.get("viewpoint_id") if isinstance(action, dict) else None,
            "matter_sim_point_id": (chosen or {}).get("view_index"),
            "landmark_source": source,
            "landmarks": landmarks,
        }
    return out


def iter_instruction_items_from_step_manifest(rows: Iterable[Dict]) -> Iterable[Dict]:
    grouped = {}
    for row in rows:
        sample_id = row.get("sample_id")
        instr_id, step_idx = _parse_sample_id(sample_id)
        if not instr_id or step_idx is None:
            continue
        grouped.setdefault(instr_id, []).append((step_idx, row))

    for instr_id, seq in grouped.items():
        seq.sort(key=lambda x: x[0])
        first = seq[0][1]
        # Use longest history as reconstructed trajectory path.
        best_row = max((r for _, r in seq), key=lambda r: len(r.get("history", []) or []))
        path = best_row.get("history", []) or []
        view_dir = ((first.get("current") or {}).get("view_dir"))
        scan = _infer_scan_from_view_dir(view_dir)
        if not scan or not path:
            continue
        yield {
            "path_id": instr_id,
            "instr_idx": 0,
            "instr_id": instr_id,
            "scan": scan,
            "path": path,
            "instruction": first.get("instruction", ""),
            "raw_item": first,
        }


def reference_fgr2r_progress(raw_item: Dict, instr_idx: int):
    if "new_instructions" not in raw_item or "chunk_view" not in raw_item:
        return None
    try:
        chunks = ast.literal_eval(raw_item["new_instructions"])[instr_idx]
        chunk_view = raw_item["chunk_view"][instr_idx]
    except Exception:
        return None
    return {
        "new_instructions": [" ".join(chunk) for chunk in chunks],
        "chunk_view": chunk_view,
    }


def confidence_from_score(score: float, high_threshold: float, medium_threshold: float) -> str:
    if score >= high_threshold:
        return "high"
    if score >= medium_threshold:
        return "medium"
    return "low"


def _enforce_monotonic(assignments: List[int]) -> List[int]:
    if not assignments:
        return assignments
    out = assignments[:]
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def _build_step_evidence(
    scan: str,
    path: List[str],
    t2t_landmark_dir: str,
    precomputed_step_edges: Dict[int, Dict[str, object]],
    manifest_step_landmarks: Dict[int, Dict[str, object]],
    *,
    from_step_manifest: bool,
):
    """Step evidence for LLM prompts.

    With ``from_step_manifest=True`` (``--step_manifest_jsonl``): landmarks come **only**
    from the manifest index — **no** ``landmark_phrases_for_direction`` / t2t lookup.

    With ``from_step_manifest=False``: landmarks come **only** from t2t files via
    ``landmark_phrases_for_direction``.
    """
    rows = []
    n = len(path)
    for t in range(max(0, n - 1)):
        cur_vp = path[t]
        next_vp = path[t + 1]
        view_idx = None
        if precomputed_step_edges and t in precomputed_step_edges:
            edge = precomputed_step_edges[t]
            next_vp = edge.get("expert_next_viewpoint") or next_vp
            view_idx = edge.get("view_index")

        if from_step_manifest:
            if t in manifest_step_landmarks:
                st = manifest_step_landmarks[t]
                if st.get("viewpoint"):
                    cur_vp = st["viewpoint"]
                if st.get("expert_next_viewpoint") is not None:
                    next_vp = st.get("expert_next_viewpoint")
                if st.get("matter_sim_point_id") is not None:
                    view_idx = st.get("matter_sim_point_id")
                phrases = st.get("landmarks", []) or []
                lm_src = st.get("landmark_source", "step_manifest_action_candidate")
            else:
                phrases, lm_src = [], "missing_step_manifest_landmark"
        else:
            phrases, lm_src = landmark_phrases_for_direction(
                t2t_landmark_dir,
                scan,
                cur_vp,
                next_vp,
                view_index=view_idx,
                single_view_only=True,
            )
        rows.append(
            {
                "path_pos": t + 1,
                "viewpoint": cur_vp,
                "expert_next_viewpoint": next_vp,
                "matter_sim_point_id": view_idx,
                "landmark_source": lm_src,
                "landmarks": phrases,
            }
        )
    if n > 0:
        if from_step_manifest:
            tail_landmarks = manifest_step_landmarks.get(n - 1, {}) or {}
            rows.append(
                {
                    "path_pos": n,
                    "viewpoint": tail_landmarks.get("viewpoint", path[-1]),
                    "expert_next_viewpoint": tail_landmarks.get("expert_next_viewpoint"),
                    "matter_sim_point_id": tail_landmarks.get("matter_sim_point_id"),
                    "landmark_source": tail_landmarks.get("landmark_source", "step_manifest_terminal"),
                    "landmarks": tail_landmarks.get("landmarks", []),
                }
            )
        else:
            rows.append(
                {
                    "path_pos": n,
                    "viewpoint": path[-1],
                    "expert_next_viewpoint": None,
                    "matter_sim_point_id": None,
                    "landmark_source": "terminal_node_full",
                    "landmarks": [],
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build pseudo progress labels for R2R-style instruction/path data.")
    parser.add_argument("--r2r_json", default="data/R2R/R2R_val_seen_enc.json")
    parser.add_argument("--step_manifest_jsonl", default=None, help="Optional new step-manifest JSONL as primary source.")
    parser.add_argument("--output_jsonl", default="vln_choice/data_processed/r2r_progress_labels.jsonl")
    parser.add_argument("--t2t_landmark_dir", default="data/t2t_landmarks")
    parser.add_argument(
        "--splitter",
        choices=["heuristic", "api_4z"],
        default="api_4z",
        help="Default api_4z: FourZApiSplitter (OpenAI-compatible, https://4zapi.com). "
        "Use heuristic for no-network smoke tests.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="gpt-5.4-mini",
        help="api_4z default remote model id for first trials (4Z: often listed as gpt-5.4-mini). "
        "Override with FOURZ_MODEL / API_4Z_MODEL or --api_model; must match console model id.",
    )
    parser.add_argument(
        "--api_base",
        default=None,
        help="api_4z: console Base URL ending in /v1 (see https://4zapi.com; or FOURZ_API_BASE).",
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help="api_4z only: Bearer token (or FOURZ_API_KEY). Prefer env so it does not appear in ps.",
    )
    parser.add_argument(
        "--api_model",
        default=None,
        help="api_4z: remote model id (or FOURZ_MODEL). Example trial model on 4Z: gpt-5.4-mini.",
    )
    parser.add_argument(
        "--api_timeout_sec",
        type=float,
        default=None,
        help="api_4z: HTTP read timeout seconds (or FOURZ_TIMEOUT_SEC / API_4Z_TIMEOUT_SEC). "
        "Default 120 in splitter if unset. On read timeouts, each request is retried (see FOURZ_CHAT_RETRIES). "
        "Increase via FOURZ_TIMEOUT_SEC or this flag for slow joint prompts.",
    )
    parser.add_argument(
        "--api_chat_retries",
        type=int,
        default=None,
        help="api_4z: attempts on transient read timeouts (or FOURZ_CHAT_RETRIES). Default 3.",
    )
    parser.add_argument(
        "--split_max_new_tokens",
        type=int,
        default=512,
        help="api_4z only: max completion tokens for subgoal split.",
    )
    parser.add_argument(
        "--connectivity_dir",
        default=None,
        help="candidate mode only: MatterSim connectivity dir → resolve point_id (view angle) toward expert next node.",
    )
    parser.add_argument(
        "--scan_dir",
        default=None,
        help="candidate mode only: MatterSim MP3D scan root (paired with --connectivity_dir).",
    )
    parser.add_argument(
        "--candidate_context_jsonl",
        default=None,
        help="candidate mode optional: merged candidate context JSONL providing action-selected view_index.",
    )
    parser.add_argument(
        "--progress_mode",
        choices=["candidate", "dp"],
        default="candidate",
        help="candidate: assign each path step via expert-next-node t2t landmarks vs subgoals (choice-centric proxy). "
        "dp: gold-path window+DP intervals (legacy).",
    )
    parser.add_argument(
        "--no_candidate_monotonic_clip",
        action="store_true",
        help="candidate mode only: disable greedy non-decreasing subgoal_index along path.",
    )
    parser.add_argument(
        "--force_cover_all_subgoals",
        action="store_true",
        help="candidate mode only: force each subgoal to own at least one step (may produce less natural chunk_view).",
    )
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--window_radius", type=int, default=1)
    parser.add_argument("--high_threshold", type=float, default=0.55)
    parser.add_argument("--medium_threshold", type=float, default=0.25)
    parser.add_argument("--min_alignment_score", type=float, default=0.0, help="Drop rows below this alignment score (candidate or dp). Use 0 to keep all.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip instr_id already present in --output_jsonl and append new rows only (safe to continue after interrupt).",
    )
    parser.add_argument(
        "--instr_ids_file",
        default=None,
        help="Only process these instr_id values (one per line, # comments, or JSONL rows with instr_id). "
        "Use to retry rows listed in <output_stem>_llm_fallback.jsonl.",
    )
    parser.add_argument(
        "--drop_instr_ids_from_output",
        default=None,
        help="With --resume: before processing, delete rows for these instr_ids from --output_jsonl so they can be rebuilt.",
    )
    parser.add_argument(
        "--no_fallback_manifest",
        action="store_true",
        help="Do not write <output_stem>_llm_fallback.jsonl for each LLM fallback.",
    )
    args = parser.parse_args()
    if args.drop_instr_ids_from_output and not args.resume:
        parser.error("--drop_instr_ids_from_output requires --resume")

    source_rows = load_json_or_jsonl(args.step_manifest_jsonl) if args.step_manifest_jsonl else load_json_or_jsonl(args.r2r_json)
    if args.step_manifest_jsonl:
        rows = list(iter_instruction_items_from_step_manifest(source_rows))
        precomputed_edge_index = _build_precomputed_edge_index_from_step_manifest_rows(source_rows)
        step_manifest_landmark_index = _build_step_manifest_landmark_index(source_rows)
    else:
        rows = list(iter_instruction_items(source_rows))
        precomputed_edge_index = {}
        step_manifest_landmark_index = {}

    if args.progress_mode == "candidate" and args.candidate_context_jsonl:
        # External context overrides / supplements.
        ext_rows = load_json_or_jsonl(args.candidate_context_jsonl)
        ext_index = _build_precomputed_edge_index_from_step_manifest_rows(ext_rows)
        precomputed_edge_index.update(ext_index)

    if args.instr_ids_file:
        allow = _load_instr_ids_file(args.instr_ids_file)
        before_ct = len(rows)
        rows = [r for r in rows if str(r.get("instr_id")) in allow]
        print(
            "instr_ids_file {}: kept {} of {} row(s)".format(args.instr_ids_file, len(rows), before_ct),
            flush=True,
        )

    sim = None
    if args.progress_mode == "candidate" and args.connectivity_dir and args.scan_dir and not precomputed_edge_index:
        from vln_choice.render import build_simulator

        sim = build_simulator(args.connectivity_dir, args.scan_dir)

    splitter = get_splitter(
        args.splitter,
        args.model_name_or_path,
        args.split_max_new_tokens,
        api_base=args.api_base,
        api_key=args.api_key,
        api_model=args.api_model,
        timeout_sec=args.api_timeout_sec,
        chat_retries=args.api_chat_retries,
    )

    output_path = Path(args.output_jsonl)
    fallback_manifest_path = None if args.no_fallback_manifest else output_path.with_name(output_path.stem + "_llm_fallback.jsonl")

    if not args.resume:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
        if fallback_manifest_path is not None:
            fallback_manifest_path.write_text("", encoding="utf-8")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if args.drop_instr_ids_from_output:
            drop_ids = _load_instr_ids_file(args.drop_instr_ids_from_output)
            nrm = _rewrite_jsonl_drop_instr_ids(args.output_jsonl, drop_ids)
            print(
                "drop_instr_ids_from_output: removed {} line(s) from {} ({} instr_id(s))".format(
                    nrm, args.output_jsonl, len(drop_ids)
                ),
                flush=True,
            )

    done_instr_ids = _load_done_instr_ids(args.output_jsonl) if args.resume else set()
    if args.resume and done_instr_ids:
        print(
            "resume: will skip {} instr_id(s) already in {}".format(len(done_instr_ids), args.output_jsonl),
            flush=True,
        )
    if fallback_manifest_path is not None:
        print(
            "LLM fallback manifest: {}".format(fallback_manifest_path),
            flush=True,
        )

    rows_written = 0
    skipped_resume = 0
    llm_fallback_count = 0
    llm_ok_count = 0
    llm_fallback_by_class: Counter = Counter()

    for idx, item in enumerate(rows):
        if args.limit is not None and idx >= args.limit:
            break
        if args.resume and item.get("instr_id") in done_instr_ids:
            skipped_resume += 1
            continue
        if args.log_every > 0 and idx % args.log_every == 0:
            print(
                "processing idx={} rows_written={}".format(idx, rows_written),
                flush=True,
            )
        if args.progress_mode == "candidate":
            pre_edges = precomputed_edge_index.get(item["instr_id"], {})
            step_evidence = _build_step_evidence(
                item["scan"],
                item["path"],
                args.t2t_landmark_dir,
                pre_edges,
                step_manifest_landmark_index.get(item["instr_id"], {}),
                from_step_manifest=bool(args.step_manifest_jsonl),
            )
            if splitter is not None and hasattr(splitter, "split_and_align"):
                llm_joint = splitter.split_and_align(item["instruction"], step_evidence)
                if str(llm_joint.get("source", "")).endswith("_fallback"):
                    llm_fallback_count += 1
                    _note_llm_fallback(
                        str(item.get("instr_id", "")),
                        str(llm_joint.get("source")),
                        llm_joint.get("llm_error"),
                        fallback_manifest_path,
                        llm_fallback_by_class,
                    )
                    _warn_llm_fallback(
                        item.get("instr_id", ""),
                        str(llm_joint.get("source")),
                        llm_joint.get("llm_error"),
                    )
                else:
                    llm_ok_count += 1
                    _info_llm_ok(item.get("instr_id", ""), str(llm_joint.get("source")))
                subgoals = llm_joint.get("subgoals", []) or heuristic_split(item["instruction"])
                split_source = getattr(splitter, "model", None) or args.api_model or args.model_name_or_path or ""
                assignments = llm_joint.get("step_subgoal_index", [0] * len(step_evidence))
                if not args.no_candidate_monotonic_clip:
                    assignments = _enforce_monotonic(assignments)
                subgoals, assignments, dropped_subgoal_indices = compact_subgoals_and_assignments(subgoals, assignments)
                for i, row in enumerate(step_evidence):
                    row["subgoal_index"] = assignments[i] if i < len(assignments) else assignments[-1]
                    row["score"] = 1.0
                    row["scores_per_subgoal"] = []
                best_chunk_view, hull_warnings = chunk_view_for_subgoals(assignments, len(subgoals))
                best_score = 1.0
                alignments = [{"chunk_view": best_chunk_view, "alignment_score": best_score, "score": best_score}]
                source_tag = "v1_candidate_llm_joint"
                aligner_name = "llm_split_then_step_assignment_single_call"
                source_detail_extra = {
                    "aligner": aligner_name,
                    "landmark_resolution": "step_manifest_action_landmark",
                    "matter_sim_for_point_id": bool(sim),
                    "step_manifest_jsonl": args.step_manifest_jsonl,
                    "candidate_context_jsonl": args.candidate_context_jsonl,
                    "candidate_monotonic_clip": not args.no_candidate_monotonic_clip,
                    "tie_break_detail": {"hull_warnings": hull_warnings, "monotonic_clip": not args.no_candidate_monotonic_clip},
                    "llm_alignment_confidence": llm_joint.get("confidence"),
                    "llm_alignment_source": llm_joint.get("source"),
                }
                if llm_joint.get("llm_error"):
                    source_detail_extra["llm_error"] = llm_joint["llm_error"]
                if dropped_subgoal_indices:
                    source_detail_extra["dropped_empty_subgoal_indices"] = dropped_subgoal_indices
                step_progress_field = step_evidence
            elif splitter is not None and hasattr(splitter, "align_steps"):
                subgoals = splitter.split(item["instruction"])
                split_source = getattr(splitter, "model", None) or args.api_model or args.model_name_or_path or ""
                llm_align = splitter.align_steps(item["instruction"], subgoals, step_evidence)
                if str(llm_align.get("source", "")).endswith("_fallback"):
                    llm_fallback_count += 1
                    _note_llm_fallback(
                        str(item.get("instr_id", "")),
                        str(llm_align.get("source")),
                        llm_align.get("llm_error"),
                        fallback_manifest_path,
                        llm_fallback_by_class,
                    )
                    _warn_llm_fallback(
                        item.get("instr_id", ""),
                        str(llm_align.get("source")),
                        llm_align.get("llm_error"),
                    )
                else:
                    llm_ok_count += 1
                    _info_llm_ok(item.get("instr_id", ""), str(llm_align.get("source")))
                assignments = llm_align.get("step_subgoal_index", [0] * len(step_evidence))
                if not args.no_candidate_monotonic_clip:
                    assignments = _enforce_monotonic(assignments)
                subgoals, assignments, dropped_subgoal_indices = compact_subgoals_and_assignments(subgoals, assignments)
                for i, row in enumerate(step_evidence):
                    row["subgoal_index"] = assignments[i] if i < len(assignments) else assignments[-1]
                    row["score"] = 1.0
                    row["scores_per_subgoal"] = []
                best_chunk_view, hull_warnings = chunk_view_for_subgoals(assignments, len(subgoals))
                best_score = 1.0
                alignments = [{"chunk_view": best_chunk_view, "alignment_score": best_score, "score": best_score}]
                source_tag = "v1_candidate_llm"
                aligner_name = "llm_step_to_subgoal_assignment"
                source_detail_extra = {
                    "aligner": aligner_name,
                    "landmark_resolution": "step_manifest_action_landmark",
                    "matter_sim_for_point_id": bool(sim),
                    "step_manifest_jsonl": args.step_manifest_jsonl,
                    "candidate_context_jsonl": args.candidate_context_jsonl,
                    "candidate_monotonic_clip": not args.no_candidate_monotonic_clip,
                    "tie_break_detail": {"hull_warnings": hull_warnings, "monotonic_clip": not args.no_candidate_monotonic_clip},
                    "llm_alignment_confidence": llm_align.get("confidence"),
                    "llm_alignment_source": llm_align.get("source"),
                }
                if llm_align.get("llm_error"):
                    source_detail_extra["llm_error"] = llm_align["llm_error"]
                if dropped_subgoal_indices:
                    source_detail_extra["dropped_empty_subgoal_indices"] = dropped_subgoal_indices
                step_progress_field = step_evidence
            else:
                if splitter is None:
                    subgoals = heuristic_split(item["instruction"])
                    split_source = "heuristic"
                else:
                    subgoals = splitter.split(item["instruction"])
                    split_source = getattr(splitter, "model", None) or args.api_model or args.model_name_or_path or ""
                cand = align_expert_candidate_landmarks(
                    subgoals,
                    item["scan"],
                    item["path"],
                    args.t2t_landmark_dir,
                    monotonic_clip=not args.no_candidate_monotonic_clip,
                    sim=sim,
                    precomputed_step_edges=pre_edges,
                    force_cover_all_subgoals=args.force_cover_all_subgoals,
                )
                best_chunk_view = cand.get("chunk_view") or []
                best_score = float(cand.get("alignment_score", 0.0))
                alignments = [{"chunk_view": best_chunk_view, "alignment_score": best_score, "score": best_score}]
                source_tag = "v1_candidate_next"
                aligner_name = "expert_destination_t2t_vs_subgoals"
                source_detail_extra = {
                    "aligner": aligner_name,
                    "landmark_resolution": "current_node_t2t_direction_or_view_index",
                    "matter_sim_for_point_id": bool(sim),
                    "step_manifest_jsonl": args.step_manifest_jsonl,
                    "candidate_context_jsonl": args.candidate_context_jsonl,
                    "candidate_monotonic_clip": not args.no_candidate_monotonic_clip,
                    "tie_break_detail": cand.get("tie_break_detail"),
                }
                step_progress_field = cand.get("step_progress", [])
        else:
            path_landmarks = build_path_landmarks(args.t2t_landmark_dir, item["scan"], item["path"])
            if splitter is None:
                subgoals = heuristic_split(item["instruction"])
                split_source = "heuristic"
            else:
                subgoals = splitter.split(item["instruction"])
                split_source = getattr(splitter, "model", None) or args.api_model or args.model_name_or_path or ""
            alignments = align_subgoals_topk(
                subgoals,
                path_landmarks,
                top_k=args.top_k,
                window_radius=args.window_radius,
            )
            best = alignments[0] if alignments else {"chunk_view": [], "alignment_score": 0.0}
            best_chunk_view = best["chunk_view"]
            best_score = float(best["alignment_score"])
            source_tag = "v1_dp"
            aligner_name = "landmark_window_dp"
            source_detail_extra = {
                "aligner": aligner_name,
                "window_radius": args.window_radius,
                "top_k": args.top_k,
            }
            step_progress_field = []

        confidence = confidence_from_score(best_score, args.high_threshold, args.medium_threshold)
        if best_score < args.min_alignment_score:
            continue

        output = {
            "path_id": item["path_id"],
            "instr_idx": item["instr_idx"],
            "instr_id": item["instr_id"],
            "scan": item["scan"],
            "path": item["path"],
            "instruction": item["instruction"],
            "new_instructions": subgoals,
            "chunk_view": best_chunk_view,
            "alignment_score": best_score,
            "confidence": confidence,
            "source": source_tag,
            "candidates": [
                {
                    "chunk_view": row["chunk_view"],
                    "score": row["alignment_score"],
                }
                for row in alignments
            ],
            "topk_chunk_view": alignments,
            "step_progress": step_progress_field,
            "source_detail": {
                "splitter": args.splitter,
                "split_model": split_source,
                "progress_mode": args.progress_mode,
                "min_alignment_score": args.min_alignment_score,
                **source_detail_extra,
            },
        }
        ref = reference_fgr2r_progress(item["raw_item"], item["instr_idx"])
        if ref is not None:
            output["reference_progress"] = ref
        append_jsonl_record(args.output_jsonl, output)
        rows_written += 1

    print("wrote {} rows to {}".format(rows_written, args.output_jsonl))
    fb_detail = ""
    if llm_fallback_count:
        fb_detail = " | llm_fallback_by_class: " + " ".join(
            "{}={}".format(k, llm_fallback_by_class[k]) for k in sorted(llm_fallback_by_class.keys())
        )
    print(
        "INFO [build_progress_labels] summary written={} llm_ok={} llm_fallback={} resume_skipped={}{}".format(
            rows_written, llm_ok_count, llm_fallback_count, skipped_resume, fb_detail
        ),
        file=sys.stderr,
        flush=True,
    )
    if llm_fallback_count:
        print(
            "WARNING [build_progress_labels] {} row(s) used LLM fallback (manifest {}; rows still contain labels via heuristic fallback).".format(
                llm_fallback_count,
                fallback_manifest_path if fallback_manifest_path is not None else "disabled",
            ),
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    main()


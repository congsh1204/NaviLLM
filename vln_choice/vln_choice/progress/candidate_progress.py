"""Progress labels from expert step direction using **current-node** t2t landmarks.

At ``path[t]``, use **only** the t2t bucket for the MatterSim ``point_id`` view aimed at
``path[t+1]`` (see ``landmark_phrases_for_direction(..., single_view_only=True)``). Neighbor-id
keys are a fallback when ``point_id`` is unavailable.
"""

from typing import Any, Dict, List, Optional, Tuple

from vln_choice.candidates import expert_edge_point_id

from .landmarks import viewpoint_term_counts, viewpoint_term_counts_for_direction
from .schema import Subgoal, coerce_subgoals


def score_subgoal_vs_term_counts(subgoal: Subgoal, term_counts: Dict[str, int]) -> float:
    """Same weighting spirit as ``align_dp._subgoal_position_score`` but fixed counts (no window)."""
    terms = subgoal.match_terms
    if not terms:
        return 0.0
    counts_float = {k: float(v) for k, v in term_counts.items()}
    hits = sum(1.0 for term in terms if term in counts_float)
    weighted_hits = sum(min(1.0, counts_float.get(term, 0.0)) for term in terms)
    return (0.7 * weighted_hits + 0.3 * hits) / max(1, len(terms))


def _argmax_scores(scores: List[float]) -> int:
    best_i = 0
    best_v = scores[0]
    for i, v in enumerate(scores[1:], start=1):
        if v > best_v:
            best_v = v
            best_i = i
    return best_i


def _enforce_monotonic(assignments: List[int]) -> List[int]:
    """Greedy clip so subgoal index never decreases along the path (reduces oscillation noise)."""
    if not assignments:
        return assignments
    out = assignments[:]
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def chunk_view_for_subgoals(assignments: List[int], num_subgoals: int) -> Tuple[List[List[int]], List[str]]:
    """Build ``chunk_view[k]`` = closed hull of path_pos where assignment equals ``k`` (1-based).

    Returns (chunk_view, warnings). Empty subgoals keep empty interval ``[]`` (no forced terminal fallback).
    """
    warnings = []
    n = len(assignments)
    if num_subgoals <= 0:
        return [], ["num_subgoals=0"]
    chunks = []
    for k in range(num_subgoals):
        positions = sorted({i + 1 for i, sg in enumerate(assignments) if sg == k})
        if not positions:
            warnings.append("subgoal_{}_no_steps".format(k))
            chunks.append([])
            continue
        lo, hi = positions[0], positions[-1]
        if positions != list(range(lo, hi + 1)):
            warnings.append("subgoal_{}_noncontiguous_hull".format(k))
        chunks.append([lo, hi])
    return chunks, warnings


def _fill_missing_subgoals(assignments: List[int], step_progress: List[dict], num_subgoals: int) -> None:
    """Ensure each subgoal index appears at least once so chunk_view stays aligned with new_instructions."""
    present = set(assignments)
    for k in range(num_subgoals):
        if k in present:
            continue
        best_t = 0
        best_s = -1.0
        for row in step_progress:
            vec = row.get("scores_per_subgoal") or []
            if k < len(vec) and vec[k] > best_s:
                best_s = vec[k]
                best_t = int(row["path_pos"]) - 1
        assignments[best_t] = k
        present.add(k)


def align_expert_candidate_landmarks(
    subgoals,
    scan: str,
    path: List[str],
    t2t_landmark_dir: str,
    monotonic_clip: bool = True,
    sim: Optional[Any] = None,
    precomputed_step_edges: Optional[Dict[int, Dict[str, Any]]] = None,
    force_cover_all_subgoals: bool = False,
) -> Dict:
    """Assign each **current** path node to a subgoal using directional t2t landmarks **at path[t]**.

    For step ``t`` (current viewpoint ``path[t]``, expert next ``path[t+1]``):

    - Resolve MatterSim ``point_id`` (0–35) toward ``path[t+1]`` when ``sim`` is set; read **only**
      that view bucket in ``path[t]``'s t2t JSON (``single_view_only=True``).
    - If ``point_id`` is unavailable, fall back to neighbor-id keyed phrases under ``path[t]`` only.
    - Never merge all 36 views in candidate mode.

    Terminal ``path[-1]``: score against full-node landmarks at the terminal (no outgoing edge).

    Returns dict with chunk_view (parallel to subgoals), step_progress, alignment_score, etc.
    """
    sg_list = coerce_subgoals(subgoals)
    m = len(sg_list)
    n = len(path)
    if m == 0 or n == 0:
        return {
            "chunk_view": [],
            "alignment_score": 0.0,
            "step_progress": [],
            "tie_break_detail": {"reason": "empty_subgoals_or_path"},
        }

    step_progress = []
    per_step_best_scores = []

    if n == 1:
        counts = viewpoint_term_counts(t2t_landmark_dir, scan, path[0])
        scores = [score_subgoal_vs_term_counts(sg, counts) for sg in sg_list]
        k = _argmax_scores(scores)
        assignments = [k]
        step_progress.append(
            {
                "path_pos": 1,
                "viewpoint": path[0],
                "expert_next_viewpoint": None,
                "landmark_source": "current_node_full",
                "matter_sim_point_id": None,
                "subgoal_index": k,
                "score": scores[k],
                "scores_per_subgoal": scores,
            }
        )
        per_step_best_scores.append(scores[k])
    else:
        assignments = [0] * n
        for t in range(n - 1):
            cur_vp = path[t]
            next_vp = path[t + 1]
            view_idx = None

            if precomputed_step_edges is not None and t in precomputed_step_edges:
                edge = precomputed_step_edges[t]
                next_vp = edge.get("expert_next_viewpoint") or next_vp
                view_idx = edge.get("view_index")
            elif sim is not None:
                view_idx = expert_edge_point_id(sim, scan, cur_vp, next_vp)

            counts, lm_src = viewpoint_term_counts_for_direction(
                t2t_landmark_dir,
                scan,
                cur_vp,
                next_vp,
                view_idx,
                single_view_only=True,
            )
            scores = [score_subgoal_vs_term_counts(sg, counts) for sg in sg_list]
            k = _argmax_scores(scores)
            assignments[t] = k
            per_step_best_scores.append(scores[k])
            step_progress.append(
                {
                    "path_pos": t + 1,
                    "viewpoint": cur_vp,
                    "expert_next_viewpoint": next_vp,
                    "landmark_source": lm_src,
                    "matter_sim_point_id": view_idx,
                    "subgoal_index": k,
                    "score": scores[k],
                    "scores_per_subgoal": scores,
                }
            )
        counts_last = viewpoint_term_counts(t2t_landmark_dir, scan, path[-1])
        scores_last = [score_subgoal_vs_term_counts(sg, counts_last) for sg in sg_list]
        k_last = _argmax_scores(scores_last)
        assignments[-1] = k_last
        per_step_best_scores.append(scores_last[k_last])
        step_progress.append(
            {
                "path_pos": n,
                "viewpoint": path[-1],
                "expert_next_viewpoint": None,
                "landmark_source": "terminal_node_full",
                "matter_sim_point_id": None,
                "subgoal_index": k_last,
                "score": scores_last[k_last],
                "scores_per_subgoal": scores_last,
            }
        )

    if force_cover_all_subgoals:
        _fill_missing_subgoals(assignments, step_progress, m)

    raw_assignments = assignments[:]
    if monotonic_clip:
        assignments = _enforce_monotonic(assignments)
    for row in step_progress:
        p = int(row["path_pos"])
        row["raw_subgoal_index"] = raw_assignments[p - 1]
        row["subgoal_index"] = assignments[p - 1]

    chunk_view, hull_warnings = chunk_view_for_subgoals(assignments, m)
    alignment_score = max(
        0.0, min(1.0, sum(per_step_best_scores) / max(1, len(per_step_best_scores)))
    )

    return {
        "chunk_view": chunk_view,
        "alignment_score": alignment_score,
        "step_progress": step_progress,
        "assignments": assignments,
        "tie_break_detail": {"hull_warnings": hull_warnings, "monotonic_clip": monotonic_clip},
    }

"""Gold-path interval labeling — not choice-centric candidate scoring.

Partitions the **expert trajectory** into contiguous segments and assigns each segment to one
instruction subgoal by matching subgoal terms to **node-level** t2t landmark phrases on visited
viewpoints (see ``landmarks.build_path_landmarks``).

This does **not** evaluate, at a decision step, each **navigable candidate next viewpoint** by
landmarks visible when facing / selecting that option (“choosing B serves subgoal *k*”). That
would require per-candidate or per-edge (heading) landmark evidence; v1 does not consume it.
"""

from typing import Dict, List, Tuple

from .schema import Subgoal, coerce_subgoals


def _position_terms(path_landmarks: List[Dict], pos_idx: int, window_radius: int) -> Dict[str, int]:
    counts = {}
    lo = max(0, pos_idx - window_radius)
    hi = min(len(path_landmarks), pos_idx + window_radius + 1)
    for idx in range(lo, hi):
        weight = 1.0 / (1.0 + abs(idx - pos_idx))
        for term, count in path_landmarks[idx].get("term_counts", {}).items():
            counts[term] = counts.get(term, 0.0) + weight * count
    return counts


def _subgoal_position_score(subgoal: Subgoal, path_landmarks: List[Dict], pos_idx: int, window_radius: int) -> float:
    terms = subgoal.match_terms
    if not terms or not path_landmarks:
        return 0.0
    counts = _position_terms(path_landmarks, pos_idx, window_radius)
    hits = sum(1.0 for term in terms if term in counts)
    weighted_hits = sum(min(1.0, counts.get(term, 0.0)) for term in terms)
    return (0.7 * weighted_hits + 0.3 * hits) / max(1, len(terms))


def _prefix_scores(subgoals: List[Subgoal], path_landmarks: List[Dict], window_radius: int) -> List[List[float]]:
    n = len(path_landmarks)
    prefix = []
    for subgoal in subgoals:
        vals = [0.0]
        for pos_idx in range(n):
            vals.append(vals[-1] + _subgoal_position_score(subgoal, path_landmarks, pos_idx, window_radius))
        prefix.append(vals)
    return prefix


def _interval_score(prefix: List[List[float]], subgoal_idx: int, start_pos: int, end_pos: int) -> float:
    # start_pos/end_pos are 1-based, end exclusive.
    if end_pos <= start_pos:
        return -1e9
    return prefix[subgoal_idx][end_pos - 1] - prefix[subgoal_idx][start_pos - 1]


def _state_push(states: List[Tuple[float, List[List[int]]]], candidate, top_k: int) -> List[Tuple[float, List[List[int]]]]:
    states.append(candidate)
    states.sort(key=lambda item: item[0], reverse=True)
    deduped = []
    seen = set()
    for score, chunks in states:
        key = tuple(tuple(chunk) for chunk in chunks)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((score, chunks))
        if len(deduped) >= top_k:
            break
    return deduped


def _chunk_view_half_open_to_inclusive(chunks: List[List[int]]) -> List[List[int]]:
    """DP uses half-open [start, end); exported chunk_view uses closed [start, last] (both ends kept)."""
    out = []
    for pair in chunks:
        if len(pair) != 2:
            continue
        start_pos, end_pos = int(pair[0]), int(pair[1])
        if end_pos <= start_pos:
            continue
        out.append([start_pos, end_pos - 1])
    return out


def align_subgoals_topk(subgoals, path_landmarks: List[Dict], top_k: int = 3, window_radius: int = 1) -> List[Dict]:
    """Align subgoals to contiguous **expert-path** intervals (monotonic top-k DP).

    ``path_landmarks[i]`` is landmarks at the *i*-th node on the gold path — not landmarks per
    outgoing candidate edge from a fixed current node.

    Internally intervals are half-open [start, end) over 1-based ``path_pos``. Exported
    ``chunk_view`` pairs are **closed** [start, last] with ``last == end - 1`` so both the first
    and last viewpoint of each segment are included (membership: ``start <= path_pos <= last``).
    """
    subgoals = coerce_subgoals(subgoals)
    n = len(path_landmarks)
    m = len(subgoals)
    if not subgoals or not path_landmarks:
        return []

    top_k = max(1, top_k)
    prefix = _prefix_scores(subgoals, path_landmarks, window_radius)

    # dp[i][e] stores top alignments for first i subgoals ending at exclusive path position e.
    # e ranges from 2..n+1 because intervals are [start, end) over 1-based path positions.
    dp = [[[] for _ in range(n + 2)] for _ in range(m + 1)]
    dp[0][1] = [(0.0, [])]

    for i in range(1, m + 1):
        min_end = i + 1
        for end_pos in range(min_end, n + 2):
            states = []
            for start_pos in range(i, end_pos):
                prev_states = dp[i - 1][start_pos]
                if not prev_states:
                    continue
                interval = [start_pos, end_pos]
                interval_score = _interval_score(prefix, i - 1, start_pos, end_pos)
                for prev_score, prev_chunks in prev_states:
                    states = _state_push(states, (prev_score + interval_score, prev_chunks + [interval]), top_k)
            dp[i][end_pos] = states

    final_states = []
    for end_pos in range(m + 1, n + 2):
        for state in dp[m][end_pos]:
            final_states = _state_push(final_states, state, top_k)

    max_possible = float(max(1, m))
    results = []
    for raw_score, chunks in final_states:
        avg_score = max(0.0, min(1.0, raw_score / max_possible))
        chunk_view = _chunk_view_half_open_to_inclusive(chunks)
        results.append(
            {
                "chunk_view": chunk_view,
                "score": avg_score,
                "alignment_score": avg_score,
            }
        )
    return results


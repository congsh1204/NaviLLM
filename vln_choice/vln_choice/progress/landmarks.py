import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .normalize import normalize_phrase, tokenize, unique_normalized_terms


def _landmark_file(root: str, scan: str, viewpoint: str) -> Path:
    return Path(root) / scan / viewpoint / "{}_{}.json".format(scan, viewpoint)


def _flatten_to_phrases(values) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    if isinstance(values, list):
        out = []
        for x in values:
            out.extend(_flatten_to_phrases(x))
        return out
    if isinstance(values, dict):
        out = []
        for v in values.values():
            out.extend(_flatten_to_phrases(v))
        return out
    return [str(values)]


def _lookup_view_bucket_phrases(data, view_index: int) -> Tuple[List[str], Optional[str]]:
    """Return phrases for MatterSim view bucket ``view_index`` (0–35) if present."""
    keys_to_try = (
        str(view_index),
        "view_{:02d}".format(view_index),
        "{:02d}".format(view_index),
    )
    if isinstance(data, dict):
        for key in keys_to_try:
            if key in data:
                return _flatten_to_phrases(data[key]), "t2t_view_index:{}".format(view_index)
        for sub_key in ("views", "by_view", "view_landmarks", "landmarks_by_view", "nav", "navigable"):
            sub = data.get(sub_key)
            if not isinstance(sub, dict):
                continue
            for vk in keys_to_try:
                if vk in sub:
                    return _flatten_to_phrases(sub[vk]), "t2t_nested_view:{}:{}".format(sub_key, view_index)
    return [], None


def landmark_phrases_for_direction(
    t2t_landmark_dir: str,
    scan: str,
    current_viewpoint: str,
    next_viewpoint: str,
    view_index: Optional[int] = None,
    single_view_only: bool = False,
) -> Tuple[List[str], str]:
    """Landmark phrases when standing at ``current_viewpoint`` and aligning edge ``current→next``.

    **Primary (your data layout):** JSON under ``current_viewpoint`` keyed by MatterSim
    discretized view ``0``–``35``. Use ``view_index`` = ``point_id`` from ``path[t]`` toward
    ``path[t+1]`` — **only that bucket**, no merging other views.

    **Alternate:** top-level / nested key equals ``next_viewpoint`` (neighbor-id dumps).

    ``single_view_only=True`` (candidate progress): never merge all 36 views; if the aligned
    bucket is missing, return empty phrases and a diagnostic tag.

    File: ``{t2t_landmark_dir}/{scan}/{current_viewpoint}/{scan}_{current_viewpoint}.json``.
    """
    path = _landmark_file(t2t_landmark_dir, scan, current_viewpoint)
    if not path.exists():
        return [], "missing_file"

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 1) MatterSim view toward path[t+1] — takes precedence when provided
    if view_index is not None:
        phrases, vtag = _lookup_view_bucket_phrases(data, int(view_index))
        if vtag is not None:
            return phrases, vtag
        if single_view_only:
            return [], "t2t_view_bucket_missing:{}".format(view_index)

    # 2) Neighbor viewpoint id (edge-aligned text dumps without 36-view split)
    if isinstance(data, dict) and next_viewpoint in data:
        phrases = _flatten_to_phrases(data[next_viewpoint])
        return phrases, "t2t_neighbor_key"

    if isinstance(data, dict):
        for sub_key in ("views", "by_view", "view_landmarks", "landmarks_by_view", "nav", "navigable"):
            sub = data.get(sub_key)
            if isinstance(sub, dict) and next_viewpoint in sub:
                return _flatten_to_phrases(sub[next_viewpoint]), "t2t_nested_neighbor:{}".format(sub_key)

    if single_view_only:
        if view_index is None:
            return [], "t2t_need_sim_for_point_id_or_neighbor_key"
        return [], "t2t_no_directional_landmarks_after_neighbor_miss"

    # Legacy: merge all buckets (weak direction)
    if isinstance(data, dict):
        phrases = []
        for values in data.values():
            phrases.extend(_flatten_to_phrases(values))
        return phrases, "t2t_fallback_all_keys_at_current_node"
    if isinstance(data, list):
        return _flatten_to_phrases(data), "t2t_flat_list"
    return [], "t2t_empty"


def viewpoint_term_counts_for_direction(
    t2t_landmark_dir: str,
    scan: str,
    current_viewpoint: str,
    next_viewpoint: str,
    view_index: Optional[int] = None,
    single_view_only: bool = False,
) -> Tuple[Dict[str, int], str]:
    phrases, tag = landmark_phrases_for_direction(
        t2t_landmark_dir,
        scan,
        current_viewpoint,
        next_viewpoint,
        view_index,
        single_view_only=single_view_only,
    )
    return _term_counts(phrases), tag


def _read_viewpoint_landmarks(root: str, scan: str, viewpoint: str) -> List[str]:
    path = _landmark_file(root, scan, viewpoint)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    phrases = []
    if isinstance(data, dict):
        for values in data.values():
            if isinstance(values, list):
                phrases.extend(str(value) for value in values)
            elif values is not None:
                phrases.append(str(values))
    elif isinstance(data, list):
        phrases.extend(str(value) for value in data)
    return phrases


def _term_counts(phrases: Iterable[str]) -> Dict[str, int]:
    counts = Counter()
    for phrase in phrases:
        counts.update(tokenize(phrase))
    return dict(counts)


def viewpoint_term_counts(t2t_landmark_dir: str, scan: str, viewpoint: str) -> Dict[str, int]:
    """Token counts for one node's t2t landmark phrases (used for candidate / edge-side proxies)."""
    phrases = _read_viewpoint_landmarks(t2t_landmark_dir, scan, viewpoint)
    return _term_counts(phrases)


def viewpoint_landmark_phrases(
    t2t_landmark_dir: str,
    scan: str,
    viewpoint: str,
    *,
    normalize: bool = False,
) -> Tuple[List[str], str]:
    """Unique landmark phrases visible from a viewpoint's full 360° panorama (all 36 views merged).

    Used as the ``observed_evidence`` signal in EGAC (Evidence-Gated Action Commitment): what
    the agent can plausibly see somewhere at this node, regardless of heading. For a single
    view-bucket / directional query, use ``landmark_phrases_for_direction`` instead.

    Default dedup is on the **surface form** (case-insensitive, whitespace-stripped) — preserves
    natural phrases like ``"a bathroom"`` and keeps ``"step"`` / ``"stairs"`` distinct, matching
    label examples like ``["stair", "step"]``. Set ``normalize=True`` for tighter collapse via
    ``normalize_phrase`` (drops articles/stopwords + applies ``_ALIASES``: ``"step"``/``"stair"``
    → ``"stairs"``), useful when comparing against ``required_evidence``.

    Order of returned phrases follows first occurrence as the JSON's view buckets are walked.

    Returns ``([phrases], tag)`` where ``tag`` is ``"missing_file"`` when the JSON does not exist,
    ``"t2t_full_node_normalized_dedup"`` when ``normalize=True``, else ``"t2t_full_node_dedup"``.
    """
    path = _landmark_file(t2t_landmark_dir, scan, viewpoint)
    if not path.exists():
        return [], "missing_file"

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    raw_phrases = _flatten_to_phrases(data)

    seen: set = set()
    out: List[str] = []
    for phrase in raw_phrases:
        if not phrase:
            continue
        if normalize:
            key = normalize_phrase(phrase)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(key)
        else:
            stripped = phrase.strip()
            key = stripped.lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(stripped)

    tag = "t2t_full_node_normalized_dedup" if normalize else "t2t_full_node_dedup"
    return out, tag


def build_path_landmarks(t2t_landmark_dir: str, scan: str, path: List[str]) -> List[Dict]:
    """Collect t2t landmark phrases for each **graph viewpoint on the expert path** (full-node merge).

    For **directional** (candidate-view) phrases from ``path[t]`` toward ``path[t+1]``, use
    ``landmark_phrases_for_direction`` / ``viewpoint_term_counts_for_direction`` instead.

    ``path_pos`` is 1-based. ``chunk_view`` from DP uses **closed** intervals ``[start, last]``.
    """
    rows = []
    for idx, viewpoint in enumerate(path):
        phrases = _read_viewpoint_landmarks(t2t_landmark_dir, scan, viewpoint)
        rows.append(
            {
                "path_pos": idx + 1,
                "viewpoint": viewpoint,
                "landmarks": phrases,
                "terms": unique_normalized_terms(phrases),
                "term_counts": _term_counts(phrases),
                "missing_landmarks": len(phrases) == 0,
            }
        )
    return rows


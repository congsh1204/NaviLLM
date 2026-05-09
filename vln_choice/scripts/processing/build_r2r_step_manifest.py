#!/usr/bin/env python
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vln_choice.candidates import collect_candidates
from vln_choice.io import read_jsonl
from vln_choice.io import write_jsonl
from vln_choice.progress.landmarks import landmark_phrases_for_direction, viewpoint_landmark_phrases


_VIEWPOINT_LM_CACHE: dict = {}


def _cached_viewpoint_landmarks(t2t_landmark_dir: str, scan: str, viewpoint: str) -> list:
    key = (t2t_landmark_dir, scan, viewpoint)
    cached = _VIEWPOINT_LM_CACHE.get(key)
    if cached is not None:
        return cached
    phrases, _ = viewpoint_landmark_phrases(t2t_landmark_dir, scan, viewpoint, normalize=True)
    _VIEWPOINT_LM_CACHE[key] = phrases
    return phrases
from vln_choice.render import build_simulator


def load_r2r_items(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for item in data:
        for instr_idx, instruction in enumerate(item["instructions"]):
            rows.append(
                {
                    "path_id": item["path_id"],
                    "instr_idx": instr_idx,
                    "instr_id": "r2r_{}_{}".format(item["path_id"], instr_idx),
                    "scan": item["scan"],
                    "path": item["path"],
                    "instruction": instruction,
                }
            )
    return rows


def load_progress_labels(path: str):
    if not path:
        return {}
    labels = {}
    for row in read_jsonl(path):
        keys = []
        if row.get("instr_id"):
            keys.append(("instr_id", row["instr_id"]))
        if row.get("path_id") is not None and row.get("instr_idx") is not None:
            keys.append(("path_instr", str(row["path_id"]), int(row["instr_idx"])))
        for key in keys:
            labels[key] = row
    return labels


def progress_for_step(progress_labels, item, step_idx: int):
    if not progress_labels:
        return None
    row = progress_labels.get(("instr_id", item["instr_id"]))
    if row is None:
        row = progress_labels.get(("path_instr", str(item["path_id"]), int(item.get("instr_idx", 0))))
    if row is None:
        return None

    step_pos = step_idx + 1
    for chunk_idx, chunk in enumerate(row.get("chunk_view", [])):
        if len(chunk) != 2:
            continue
        start, end = int(chunk[0]), int(chunk[1])
        # chunk_view pairs are **closed** [start, last]: both endpoints included (path_pos 1-based).
        if start <= step_pos <= end:
            subgoals = row.get("new_instructions", [])
            subgoal = subgoals[chunk_idx] if chunk_idx < len(subgoals) else None
            return {
                "subgoal_index": chunk_idx,
                "subgoal": subgoal,
                "chunk_view": chunk,
                "alignment_score": row.get("alignment_score"),
                "confidence": row.get("confidence"),
                "source": row.get("source", "v1_dp"),
            }
    return None


def view_path(rendered_view_dir: str, scan: str, viewpoint: str, point_id: int) -> str:
    return str(Path(rendered_view_dir) / scan / viewpoint / "view_{:02d}.jpg".format(point_id))


def current_view_dir(rendered_view_dir: str, scan: str, viewpoint: str) -> str:
    return str(Path(rendered_view_dir) / scan / viewpoint)


def _view_path_from_label(candidate_image_paths, label: str):
    return (candidate_image_paths or {}).get(label)


def _stop_candidate_row():
    return {
        "label": "STOP",
        "viewpoint_id": "STOP",
        "view_path": None,
        "view_index": None,
        "view_angle": [0.0, 0.0],
        "view_landmark": "",
        "nav_order": 0,
        "pose_rad": {
            "relative": [0.0, 0.0],
            "bearing": [0.0, 0.0],
            "misalignment": 0.0,
        },
    }


def build_options(candidates, expert_next, max_candidates: int, rng: random.Random):
    candidate_by_vp = {c["viewpoint"]: c for c in candidates}
    if expert_next not in candidate_by_vp:
        return None

    expert_candidate = candidate_by_vp[expert_next]
    negatives = [c for c in candidates if c["viewpoint"] != expert_next]
    rng.shuffle(negatives)
    if max_candidates is None or max_candidates <= 0:
        selected = [expert_candidate] + negatives
    else:
        selected = [expert_candidate] + negatives[: max(0, max_candidates - 1)]
    rng.shuffle(selected)

    labels = [chr(ord("A") + i) for i in range(len(selected))]
    label_to_candidate = dict(zip(labels, selected))
    expert_label = next(label for label, cand in label_to_candidate.items() if cand["viewpoint"] == expert_next)
    return label_to_candidate, expert_label


def main():
    parser = argparse.ArgumentParser(description="Build R2R step-level choice manifest from expert paths.")
    parser.add_argument("--r2r_json", default="data/R2R/FGR2R_train.json")
    parser.add_argument("--connectivity_dir", default="data/connectivity")
    parser.add_argument("--scan_dir", default="/home/pcl/Matterport3DData/v1/scans")
    parser.add_argument("--rendered_view_dir", default="vln_choice/data_processed/mp3d_views")
    parser.add_argument("--t2t_landmark_dir", default="data/t2t_landmarks")
    parser.add_argument("--output_jsonl", default="vln_choice/data_processed/step_manifest.jsonl")
    parser.add_argument("--progress_labels_jsonl", default=None, help="Optional v1 progress labels to attach to each step.")
    parser.add_argument(
        "--max_candidates",
        type=int,
        default=0,
        help="Maximum navigable candidates before STOP. Use 0 or a negative value to keep all candidates.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include_stop_steps", action="store_true", help="Also add one STOP sample at each trajectory endpoint.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    sim = build_simulator(args.connectivity_dir, args.scan_dir)
    progress_labels = load_progress_labels(args.progress_labels_jsonl)
    rows = []
    skipped_missing_expert = 0
    skipped_missing_images = 0

    for item in load_r2r_items(args.r2r_json):
        scan = item["scan"]
        path = item["path"]
        for step_idx, cur_vp in enumerate(path[:-1]):
            expert_next = path[step_idx + 1]
            candidates = collect_candidates(sim, scan, cur_vp)
            options = build_options(candidates, expert_next, args.max_candidates, rng)
            if options is None:
                skipped_missing_expert += 1
                continue
            label_to_candidate, expert_label = options

            candidate_image_paths = {}
            candidate_point_ids = {}
            option_mapping = {}
            missing = False
            for label, cand in label_to_candidate.items():
                image_path = view_path(args.rendered_view_dir, scan, cand["viewpoint"], cand["point_id"])
                if not Path(image_path).exists():
                    missing = True
                    break
                candidate_image_paths[label] = image_path
                candidate_point_ids[label] = cand["point_id"]
                option_mapping[label] = cand["viewpoint"]
            if missing or not (Path(current_view_dir(args.rendered_view_dir, scan, cur_vp)) / "view_35.jpg").exists():
                skipped_missing_images += 1
                continue

            candidate_image_paths["STOP"] = None
            option_mapping["STOP"] = "STOP"

            candidate_rows = []
            for label, cand in sorted(label_to_candidate.items(), key=lambda kv: kv[0]):
                view_index = int(cand["point_id"])
                cand_view_path = _view_path_from_label(candidate_image_paths, label)
                landmarks, _ = landmark_phrases_for_direction(
                    args.t2t_landmark_dir,
                    scan,
                    cur_vp,
                    cand["viewpoint"],
                    view_index=view_index,
                    single_view_only=True,
                )
                candidate_rows.append(
                    {
                        "label": label,
                        "viewpoint_id": cand["viewpoint"],
                        "view_path": cand_view_path,
                        "view_index": view_index,
                        "view_angle": [float(cand["view_heading_rad"]), float(cand["view_elevation_rad"])],
                        "view_landmark": "" if not landmarks else "; ".join(landmarks),
                        "nav_order": int(cand["idx"]),
                        "pose_rad": {
                            "relative": [float(cand["neighbor_rel_heading_rad"]), float(cand["neighbor_rel_elevation_rad"])],
                            "bearing": [float(cand["neighbor_bearing_heading_rad"]), float(cand["neighbor_bearing_elevation_rad"])],
                            "misalignment": float(cand["distance"]),
                        },
                        **({"distance_m": float(cand["edge_euclidean_m"])} if cand.get("edge_euclidean_m") is not None else {}),
                    }
                )
            candidate_rows.append(_stop_candidate_row())

            rows.append(
                {
                    "sample_id": "{}_step_{:03d}".format(item["instr_id"], step_idx),
                    "instruction": item["instruction"],
                    "history": path[:step_idx + 1],
                    "current": {
                        "viewpoint_id": cur_vp,
                        "view_dir": current_view_dir(args.rendered_view_dir, scan, cur_vp),
                        "viewpoint_landmarks": _cached_viewpoint_landmarks(args.t2t_landmark_dir, scan, cur_vp),
                    },
                    "candidates": candidate_rows,
                    "action": {
                        "label": expert_label,
                        "viewpoint_id": expert_next,
                    },
                }
            )
            progress = progress_for_step(progress_labels, item, step_idx)
            if progress is not None:
                rows[-1]["progress_reasoning"] = progress
            if args.limit is not None and len(rows) >= args.limit:
                write_jsonl(args.output_jsonl, rows)
                print("wrote {} rows to {}".format(len(rows), args.output_jsonl))
                print("skipped_missing_expert={} skipped_missing_images={}".format(skipped_missing_expert, skipped_missing_images))
                return

        if path:
            cur_vp = path[-1]
            cur_dir = Path(current_view_dir(args.rendered_view_dir, scan, cur_vp))
            if (cur_dir / "view_35.jpg").exists():
                terminal_candidates = collect_candidates(sim, scan, cur_vp)
                terminal_rows = []
                for idx, cand in enumerate(sorted(terminal_candidates, key=lambda x: (x["viewpoint"], x["point_id"]))):
                    image_path = view_path(args.rendered_view_dir, scan, cand["viewpoint"], cand["point_id"])
                    if not Path(image_path).exists():
                        continue
                    view_index = int(cand["point_id"])
                    landmarks, _ = landmark_phrases_for_direction(
                        args.t2t_landmark_dir,
                        scan,
                        cur_vp,
                        cand["viewpoint"],
                        view_index=view_index,
                        single_view_only=True,
                    )
                    terminal_rows.append(
                        {
                            "label": chr(ord("A") + idx),
                            "viewpoint_id": cand["viewpoint"],
                            "view_path": image_path,
                            "view_index": view_index,
                            "view_angle": [float(cand["view_heading_rad"]), float(cand["view_elevation_rad"])],
                            "view_landmark": "" if not landmarks else "; ".join(landmarks),
                            "nav_order": int(cand["idx"]),
                            "pose_rad": {
                                "relative": [float(cand["neighbor_rel_heading_rad"]), float(cand["neighbor_rel_elevation_rad"])],
                                "bearing": [float(cand["neighbor_bearing_heading_rad"]), float(cand["neighbor_bearing_elevation_rad"])],
                                "misalignment": float(cand["distance"]),
                            },
                            **({"distance_m": float(cand["edge_euclidean_m"])} if cand.get("edge_euclidean_m") is not None else {}),
                        }
                    )
                terminal_rows.append(_stop_candidate_row())
                rows.append(
                    {
                        "sample_id": "{}_step_{:03d}_stop".format(item["instr_id"], len(path) - 1),
                        "instruction": item["instruction"],
                        "history": path,
                        "current": {
                            "viewpoint_id": cur_vp,
                            "view_dir": str(cur_dir),
                            "viewpoint_landmarks": _cached_viewpoint_landmarks(args.t2t_landmark_dir, scan, cur_vp),
                        },
                        "candidates": terminal_rows,
                        "action": {
                            "label": "STOP",
                            "viewpoint_id": "STOP",
                        },
                    }
                )
                progress = progress_for_step(progress_labels, item, len(path) - 1)
                if progress is not None:
                    rows[-1]["progress_reasoning"] = progress

    write_jsonl(args.output_jsonl, rows)
    print("wrote {} rows to {}".format(len(rows), args.output_jsonl))
    print("skipped_missing_expert={} skipped_missing_images={}".format(skipped_missing_expert, skipped_missing_images))


if __name__ == "__main__":
    main()


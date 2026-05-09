#!/usr/bin/env python
"""Export t2t landmarks per navigable candidate at each expert-path viewpoint.

For every viewpoint that appears on expert trajectories in R2R-style JSON, uses MatterSim to
enumerate navigable **next** viewpoints from that node and the **discretized view index**
(``point_id`` in 0–35) that best faces each neighbor. Reads **only that view bucket** from the
t2t JSON at the current node (same semantics as candidate progress, ``single_view_only=True``).

Requires ``--connectivity_dir`` and ``--scan_dir`` (same as MatterSim elsewhere in this repo).
常见本机路径示例：``~/mount/Matterport3DSimulator/data/v1/scans``（须含各 scan 下的
``matterport_skybox_images/<vp>_skybox_small.jpg``；勿使用文档占位路径如 ``/你的/Matterport/...``）。

Output: JSONL, one object per ``(scan, viewpoint)`` with a ``candidates`` array.

输出 JSON 字段说明（每行一个顶层对象）::

    顶层
        scan — Matterport 场景 / scan id（与 R2R、connectivity 一致）。
        viewpoint — 当前导航图视点 id（在此点枚举可导航邻居）。
        num_candidates — candidates 数组长度。
        candidates — 每个可导航邻居一条：邻居、对齐用的离散视角、t2t 短语。
        error — 仅异常时出现；失败说明字符串；此时 candidates 为空、num_candidates 为 0。

    candidates[] 每条
        next_viewpoint — 从当前 viewpoint 一步可达的邻居视点 id。
        matter_sim_point_id — MatterSim 在当前视点的 36 向离散视角编号 0–35；与 t2t JSON
            中同一编号下的 bucket（如键 \"14\"）对应，landmarks 只从该 bucket 读取。
        matter_sim_nav_order — 在该 point_id 对应帧里，navigableLocations 中该邻居的序号
            （从可导航列表 [1:] 起算、从 1 开始）；调试用。
        angular_distance_rad — 与 ``distance`` 相同：sqrt(rel_heading^2+rel_elevation^2)（弧度）。
        distance — 角对齐指标，与 ``angular_distance_rad`` 数值相同（便于与 dump 脚本字段对齐）。
        view_heading_rad / view_elevation_rad — point_id 对应帧下 MatterSim 相机朝向（弧度）。
        neighbor_rel_heading_rad / neighbor_rel_elevation_rad — 该帧 MatterSim 给出的相对邻居角度。
        neighbor_bearing_heading_rad / neighbor_bearing_elevation_rad — 相机姿态加相对角后的 bearing。
        edge_euclidean_m — 可选，当前点到邻居 navigable 位置的欧氏距离（米）。
        landmarks — 当前视点 t2t 文件中、仅按 matter_sim_point_id 对应视角 bucket 解析的短语列表。
        landmark_source — landmarks 来源或为空原因；与 vln_choice.progress.landmarks 一致。

    landmark_source 常见取值
        t2t_view_index:k — 顶层（或兼容键名）下找到视角 k 的 bucket。
        t2t_nested_view:<sub_key>:k — 嵌套表 sub_key（如 views、navigable）下找到视角 k。
        t2t_neighbor_key — 以 next_viewpoint 为键的邻居对齐文本（非 36 视角分桶）。
        t2t_nested_neighbor:<sub_key> — 邻居键出现在嵌套 dict sub_key 下。
        t2t_view_bucket_missing:k — 指定视角 k 但 JSON 中无该 bucket；single_view_only 不合并
            其它视角，故 landmarks 为空。
        t2t_need_sim_for_point_id_or_neighbor_key — 无法用语义解析方向时的占位（本脚本通常
            已有 point_id，一般少见）。
        missing_file — 缺少 t2t 文件：{t2t_landmark_dir}/{scan}/{viewpoint}/{scan}_{viewpoint}.json

磁盘上的 t2t 路径与视角键：文件见上；JSON 内视角键由 landmarks._lookup_view_bucket_phrases
尝试 \"0\"–\"35\"、view_00–view_35 等；matter_sim_point_id 即当前视点的离散视角索引。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vln_choice.candidates import collect_candidates
from vln_choice.progress.landmarks import landmark_phrases_for_direction
from vln_choice.render import build_simulator


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


def ordered_unique_path_viewpoints(rows: Iterable[Dict]) -> List[Tuple[str, str]]:
    """Preserve first-seen order of (scan, viewpoint) along expert paths."""
    seen: Set[Tuple[str, str]] = set()
    out: List[Tuple[str, str]] = []
    for item in rows:
        scan = item.get("scan")
        path = item.get("path")
        if not scan or not isinstance(path, list):
            continue
        for vp in path:
            if not isinstance(vp, str):
                continue
            key = (scan, vp)
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


def build_row(
    sim,
    t2t_landmark_dir: str,
    scan: str,
    viewpoint: str,
) -> Dict:
    raw = collect_candidates(sim, scan, viewpoint)
    raw.sort(key=lambda c: (c["viewpoint"], c["point_id"]))
    candidates = []
    for c in raw:
        next_vp = c["viewpoint"]
        pid = int(c["point_id"])
        phrases, tag = landmark_phrases_for_direction(
            t2t_landmark_dir,
            scan,
            viewpoint,
            next_vp,
            view_index=pid,
            single_view_only=True,
        )
        candidates.append(
            {
                "next_viewpoint": next_vp,
                "matter_sim_point_id": pid,
                "matter_sim_nav_order": int(c["idx"]),
                "distance": float(c["distance"]),
                "angular_distance_rad": float(c["distance"]),
                "view_heading_rad": float(c["view_heading_rad"]),
                "view_elevation_rad": float(c["view_elevation_rad"]),
                "neighbor_rel_heading_rad": float(c["neighbor_rel_heading_rad"]),
                "neighbor_rel_elevation_rad": float(c["neighbor_rel_elevation_rad"]),
                "neighbor_bearing_heading_rad": float(c["neighbor_bearing_heading_rad"]),
                "neighbor_bearing_elevation_rad": float(c["neighbor_bearing_elevation_rad"]),
                **({"edge_euclidean_m": float(c["edge_euclidean_m"])} if c.get("edge_euclidean_m") is not None else {}),
                "landmarks": phrases,
                "landmark_source": tag,
            }
        )
    return {
        "scan": scan,
        "viewpoint": viewpoint,
        "num_candidates": len(candidates),
        "candidates": candidates,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Dump per-candidate directional t2t landmarks at each expert-path viewpoint."
    )
    parser.add_argument("--r2r_json", default="data/R2R/R2R_val_seen_enc.json", help="R2R-style JSON array.")
    parser.add_argument(
        "--connectivity_dir",
        required=True,
        help="MatterSim connectivity dir (contains *_connectivity.json and scans.txt). ~ is expanded.",
    )
    parser.add_argument(
        "--scan_dir",
        required=True,
        help=(
            "Matterport scan root for MatterSim.setDatasetPath, e.g. "
            "~/mount/Matterport3DSimulator/data/v1/scans (must contain skybox images per scan). "
            "~ is expanded to $HOME."
        ),
    )
    parser.add_argument("--t2t_landmark_dir", default="data/t2t_landmarks")
    parser.add_argument(
        "--output_jsonl",
        default="vln_choice/data_processed/expert_viewpoint_candidate_landmarks.jsonl",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many unique (scan, viewpoint) pairs (after skip).",
    )
    parser.add_argument("--skip", type=int, default=0, help="Skip this many unique pairs from the start.")
    parser.add_argument(
        "--path_instr_limit",
        type=int,
        default=None,
        help="Optional: only read the first N top-level JSON rows when collecting viewpoints (debug).",
    )
    args = parser.parse_args()

    connectivity_dir = str(Path(args.connectivity_dir).expanduser().resolve())
    scan_dir = str(Path(args.scan_dir).expanduser().resolve())

    rows = load_json_or_jsonl(args.r2r_json)
    if args.path_instr_limit is not None:
        rows = rows[: args.path_instr_limit]

    pairs = ordered_unique_path_viewpoints(rows)
    if args.skip:
        pairs = pairs[args.skip :]
    if args.limit is not None:
        pairs = pairs[: args.limit]

    sim = build_simulator(connectivity_dir, scan_dir)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    with out_path.open("w", encoding="utf-8") as f:
        for scan, viewpoint in pairs:
            try:
                row = build_row(sim, args.t2t_landmark_dir, scan, viewpoint)
            except Exception as ex:
                row = {
                    "scan": scan,
                    "viewpoint": viewpoint,
                    "error": str(ex),
                    "num_candidates": 0,
                    "candidates": [],
                }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_ok += 1
            if n_ok % 500 == 0:
                print("wrote {} viewpoints".format(n_ok), flush=True)

    print("wrote {} rows to {}".format(n_ok, args.output_jsonl))


if __name__ == "__main__":
    main()

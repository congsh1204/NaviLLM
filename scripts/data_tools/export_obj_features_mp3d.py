#!/usr/bin/env python
"""Export object LMDB features into viewpoint-centric metadata + HDF5 features.

The original object features are stored as LMDB entries keyed by
`scan_viewpoint`.  This script makes them easier to inspect and consume:

  data/obj_features/mp3d_obj_feat/
    metadata.jsonl
    features.hdf5

Each line in metadata.jsonl describes one viewpoint and the objects visible
there.  The matching feature matrix is stored in features.hdf5 at key
`scan_viewpoint`, with rows aligned to the `objects` list in metadata.jsonl.

Use `--source soon` or `--source reverie` to choose which original LMDB to
normalize into this MP3D viewpoint-centric format.  Running the script again
for a different source overwrites the output unless a different `--output_dir`
is provided.

python scripts/data_tools/export_obj_features_mp3d.py --source soon
python scripts/data_tools/export_obj_features_mp3d.py --source reverie
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import h5py
import lmdb
import msgpack
import msgpack_numpy
import numpy as np
from tqdm import tqdm

msgpack_numpy.patch()


def _to_builtin(value: Any) -> Any:
    """Convert numpy/msgpack values into JSON-serializable Python objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(_to_builtin(k)): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


def _split_scanvp(scanvp: str) -> Tuple[str, str]:
    scan, viewpoint = scanvp.split("_", 1)
    return scan, viewpoint


def _iter_lmdb_entries(lmdb_dir: Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
    env = lmdb.open(str(lmdb_dir), readonly=True, lock=False, readahead=False)
    try:
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                scanvp = key.decode("ascii")
                obj_data = msgpack.unpackb(value, raw=False)
                yield scanvp, obj_data
    finally:
        env.close()


def _reverie_objects(obj_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    obj_ids = obj_data.get("obj_ids", [])
    view_ids = obj_data.get("view_ids", [])
    obj_names = obj_data.get("obj_names", [])
    bboxes = obj_data.get("bboxes", [])
    centers = obj_data.get("centers", [])

    objects = []
    for i, obj_id in enumerate(obj_ids):
        objects.append(
            {
                "row": i,
                "obj_id": _to_builtin(obj_id),
                "name": _to_builtin(obj_names[i]) if i < len(obj_names) else None,
                "view_id": _to_builtin(view_ids[i]) if i < len(view_ids) else None,
                "bbox_xywh": _to_builtin(bboxes[i]) if i < len(bboxes) else None,
                "center_heading_elevation": _to_builtin(centers[i]) if i < len(centers) else None,
            }
        )
    return objects


def _soon_objects(obj_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    obj_ids = obj_data.get("obj_ids", [])
    names = obj_data.get("names", [])
    view_ids = obj_data.get("view_ids", [])
    scores = obj_data.get("scores", [])
    mask_sizes = obj_data.get("mask_sizes", [])
    xyxy_bboxes = obj_data.get("xyxy_bboxes", [])
    centers_2d = obj_data.get("2d_centers", [])
    sizes_2d = obj_data.get("2d_sizes", [])
    centers_3d = obj_data.get("3d_centers", [])
    sizes_3d = obj_data.get("3d_sizes", [])

    objects = []
    for i, obj_id in enumerate(obj_ids):
        objects.append(
            {
                "row": i,
                "obj_id": _to_builtin(obj_id),
                "name": _to_builtin(names[i]) if i < len(names) else None,
                "view_id": _to_builtin(view_ids[i]) if i < len(view_ids) else None,
                "score": _to_builtin(scores[i]) if i < len(scores) else None,
                "mask_size": _to_builtin(mask_sizes[i]) if i < len(mask_sizes) else None,
                "bbox_xyxy": _to_builtin(xyxy_bboxes[i]) if i < len(xyxy_bboxes) else None,
                "center_2d": _to_builtin(centers_2d[i]) if i < len(centers_2d) else None,
                "size_2d": _to_builtin(sizes_2d[i]) if i < len(sizes_2d) else None,
                "center_3d": _to_builtin(centers_3d[i]) if i < len(centers_3d) else None,
                "size_3d": _to_builtin(sizes_3d[i]) if i < len(sizes_3d) else None,
            }
        )
    return objects


def export_source(source: str, lmdb_dir: Path, output_dir: Path, feature_dim: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"
    features_path = output_dir / "features.hdf5"

    object_builder = _reverie_objects if source == "reverie" else _soon_objects

    entries = list(_iter_lmdb_entries(lmdb_dir))
    with metadata_path.open("w") as meta_f, h5py.File(features_path, "w") as feat_f:
        for scanvp, obj_data in tqdm(entries, desc=f"export {source}"):
            scan, viewpoint = _split_scanvp(scanvp)
            fts = np.asarray(obj_data["fts"], dtype=np.float32)
            if feature_dim > 0:
                fts = fts[:, :feature_dim]

            feat_f.create_dataset(scanvp, data=fts, compression="gzip", compression_opts=4)

            record = {
                "source": source,
                "scanvp": scanvp,
                "scan": scan,
                "viewpoint": viewpoint,
                "feature_hdf5": str(features_path.name),
                "feature_key": scanvp,
                "feature_shape": list(fts.shape),
                "objects": object_builder(obj_data),
            }
            if source == "soon" and "world_xyz" in obj_data:
                record["world_xyz"] = _to_builtin(obj_data["world_xyz"])

            meta_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {metadata_path}")
    print(f"Wrote {features_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/obj_features/mp3d_obj_feat"))
    parser.add_argument(
        "--source",
        choices=("reverie", "soon"),
        default="soon",
        help="Which original object feature database to export into mp3d_obj_feat.",
    )
    parser.add_argument(
        "--input_lmdb",
        type=Path,
        default=None,
        help="Optional explicit LMDB directory. Defaults to data/obj_features/{source}_obj_feat.",
    )
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=768,
        help="Feature dimensions to keep from `fts`. Use <=0 to keep all original dimensions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lmdb_dirs = {
        "reverie": args.data_dir / "obj_features" / "reverie_obj_feat",
        "soon": args.data_dir / "obj_features" / "soon_obj_feat",
    }

    lmdb_dir = args.input_lmdb if args.input_lmdb is not None else lmdb_dirs[args.source]
    if not lmdb_dir.exists():
        raise FileNotFoundError(f"Missing LMDB directory: {lmdb_dir}")
    export_source(args.source, lmdb_dir, args.output_dir, args.feature_dim)


if __name__ == "__main__":
    main()

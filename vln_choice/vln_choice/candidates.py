import math
from typing import Dict, List, Optional


def _edge_euclidean_m(state, loc) -> Optional[float]:
    """Straight-line distance current viewpoint → navigable neighbor pose, if coordinates exist."""
    sl = getattr(state, "location", None)
    if sl is None or not all(hasattr(loc, a) for a in ("x", "y", "z")):
        return None
    try:
        ox, oy, oz = float(sl.x), float(sl.y), float(sl.z)
        nx, ny, nz = float(loc.x), float(loc.y), float(loc.z)
        return math.sqrt((nx - ox) ** 2 + (ny - oy) ** 2 + (nz - oz) ** 2)
    except Exception:
        return None


def expert_edge_point_id(sim, scan_id: str, viewpoint_id: str, next_viewpoint_id: str):
    """Return MatterSim discretized view index (0–35) best aligned with navigable ``next_viewpoint_id``."""
    for cand in collect_candidates(sim, scan_id, viewpoint_id):
        if cand["viewpoint"] == next_viewpoint_id:
            return int(cand["point_id"])
    return None


def collect_candidates(sim, scan_id: str, viewpoint_id: str) -> List[Dict]:
    """Collect navigable candidates with the best MatterSim view index for each next viewpoint.

    Each dict includes (beyond legacy keys):

    - ``distance``: angular alignment metric sqrt(rel_heading^2 + rel_elevation^2) at the chosen view.
    - ``view_heading_rad`` / ``view_elevation_rad``: agent camera pose at the winning ``point_id``.
    - ``neighbor_rel_heading_rad`` / ``neighbor_rel_elevation_rad``: MatterSim relative angles to the neighbor.
    - ``neighbor_bearing_heading_rad`` / ``neighbor_bearing_elevation_rad``: heading+rel, elevation+rel.
    - ``edge_euclidean_m``: optional 3D straight-line distance to neighbor pose when coords exist.
    """
    by_viewpoint = {}
    for ix in range(36):
        if ix == 0:
            sim.newEpisode([scan_id], [viewpoint_id], [0], [math.radians(-30)])
        elif ix % 12 == 0:
            sim.makeAction([0], [1.0], [1.0])
        else:
            sim.makeAction([0], [1.0], [0])

        state = sim.getState()[0]
        if state.viewIndex != ix:
            raise RuntimeError("MatterSim viewIndex mismatch: expected {}, got {}".format(ix, state.viewIndex))

        for loc_idx, loc in enumerate(state.navigableLocations[1:], start=1):
            distance = math.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
            existing = by_viewpoint.get(loc.viewpointId)
            if existing is None or distance < existing["distance"]:
                edge_m = _edge_euclidean_m(state, loc)
                row = {
                    "scan": scan_id,
                    "viewpoint": loc.viewpointId,
                    "point_id": ix,
                    "idx": loc_idx,
                    "distance": distance,
                    "view_heading_rad": float(state.heading),
                    "view_elevation_rad": float(state.elevation),
                    "neighbor_rel_heading_rad": float(loc.rel_heading),
                    "neighbor_rel_elevation_rad": float(loc.rel_elevation),
                    "neighbor_bearing_heading_rad": float(state.heading + loc.rel_heading),
                    "neighbor_bearing_elevation_rad": float(state.elevation + loc.rel_elevation),
                }
                if edge_m is not None:
                    row["edge_euclidean_m"] = float(edge_m)
                by_viewpoint[loc.viewpointId] = row
    return list(by_viewpoint.values())


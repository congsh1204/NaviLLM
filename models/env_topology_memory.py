class EnvironmentTopologyMemory(object):
    """Memory module for Environment Topology Graph (ETG)."""

    def __init__(self):
        self.nodes = {}
        self.edges = set()

    def _ensure_node(self, vp):
        if vp not in self.nodes:
            self.nodes[vp] = {
                "s_env": None,  # scene-level feature
                "d_env": None,  # object-level feature
                "m_env": {
                    "visited": False,
                    "traj_order": None,
                    "step_id": None,
                },
            }

    def add_edge(self, src_vp, dst_vp):
        self._ensure_node(src_vp)
        self._ensure_node(dst_vp)
        self.edges.add((src_vp, dst_vp))
        self.edges.add((dst_vp, src_vp))

    def update_scene_feature(self, vp, feat):
        self._ensure_node(vp)
        # Avoid keeping GPU tensors inside memory; store compact CPU data.
        if hasattr(feat, "detach"):
            feat = feat.detach()
        if hasattr(feat, "to"):
            feat = feat.to(dtype="float16")
        if hasattr(feat, "cpu"):
            feat = feat.cpu()
        if hasattr(feat, "numpy"):
            feat = feat.numpy()
        self.nodes[vp]["s_env"] = feat

    def update_object_feature(self, vp, feat):
        self._ensure_node(vp)
        # Avoid keeping GPU tensors inside memory; store compact CPU data.
        if hasattr(feat, "detach"):
            feat = feat.detach()
        if hasattr(feat, "to"):
            feat = feat.to(dtype="float16")
        if hasattr(feat, "cpu"):
            feat = feat.cpu()
        if hasattr(feat, "numpy"):
            feat = feat.numpy()
        self.nodes[vp]["d_env"] = feat

    def update_status(self, vp, visited=None, traj_order=None, step_id=None):
        self._ensure_node(vp)
        if visited is not None:
            self.nodes[vp]["m_env"]["visited"] = bool(visited)
        if traj_order is not None:
            self.nodes[vp]["m_env"]["traj_order"] = int(traj_order)
        if step_id is not None:
            self.nodes[vp]["m_env"]["step_id"] = int(step_id)

    def to_json(self):
        return {
            "nodes": self.nodes,
            "edges": list(self.edges),
        }

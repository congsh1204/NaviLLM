import math
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image


def build_simulator(connectivity_dir: str, scan_dir: str, width: int = 640, height: int = 480, vfov: int = 60):
    """Build a MatterSim renderer matching the existing feature extraction view order."""
    import MatterSim

    sim = MatterSim.Simulator()
    sim.setNavGraphPath(str(connectivity_dir))
    sim.setDatasetPath(str(scan_dir))
    sim.setCameraResolution(width, height)
    sim.setCameraVFOV(math.radians(vfov))
    sim.setDiscretizedViewingAngles(True)
    sim.setDepthEnabled(False)
    sim.setPreloadingEnabled(False)
    sim.setBatchSize(1)
    sim.initialize()
    return sim


def render_36_views(sim, scan_id: str, viewpoint_id: str) -> List[Image.Image]:
    """Render the canonical 36 MP3D views for a scan viewpoint as RGB PIL images."""
    images = []
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
        # MatterSim returns BGR; convert to RGB for PIL/Qwen processors.
        rgb = np.array(state.rgb, copy=True)[:, :, ::-1]
        images.append(Image.fromarray(rgb))
    return images


def save_views(images: List[Image.Image], output_dir: str, prefix: str = "view") -> List[str]:
    """Save rendered views and return relative file names."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, image in enumerate(images):
        path = out_dir / "{}_{:02d}.jpg".format(prefix, idx)
        image.save(path, quality=92)
        paths.append(str(path))
    return paths


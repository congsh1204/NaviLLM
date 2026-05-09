import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML or JSON config file."""
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        if cfg_path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(f)
        if cfg_path.suffix.lower() == ".json":
            return json.load(f)
    raise ValueError("Unsupported config format: {}".format(cfg_path))


def ensure_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


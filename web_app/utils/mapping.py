import json
from pathlib import Path

_CFG = None

def load_mapping(path: str = None):
    global _CFG
    if _CFG is not None:
        return _CFG
    if path:
        p = Path(path)
    else:
        p = Path(__file__).resolve().parents[2] / "config" / "mapping.json"
    if not p.exists():
        raise FileNotFoundError(f"mapping.json not found at {p}")
    _CFG = json.loads(p.read_text())
    return _CFG

def detector_index_to_project(detector_class_index):
    cfg = load_mapping()
    mapping = cfg.get("detector_to_project", {})
    # Accept int or str keys
    key = str(detector_class_index)
    return mapping.get(key, None)

def get_thresholds():
    cfg = load_mapping()
    return cfg.get("confidence_threshold", 0.3), cfg.get("nms_iou", 0.45)

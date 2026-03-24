import json
from pathlib import Path
from datetime import datetime, timezone

RUNS_DIR = Path(__file__).resolve().parents[2] / "results" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

def generate_run_id():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

def create_run_folder(run_id: str = None, metadata: dict = None):
    run_id = run_id or generate_run_id()
    tmp = RUNS_DIR / f"{run_id}_tmp"
    final = RUNS_DIR / run_id
    if final.exists():
        # ensure uniqueness
        run_id = f"{run_id}_{datetime.now(timezone.utc).strftime('%f')}"
        tmp = RUNS_DIR / f"{run_id}_tmp"
        final = RUNS_DIR / run_id
    tmp.mkdir(parents=True, exist_ok=False)
    meta = metadata or {}
    meta.update({"run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat()})
    (tmp / "metadata.json").write_text(json.dumps(meta, indent=2))
    # atomic rename
    tmp.rename(final)
    return final

import shutil
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

RUNS_DIR = Path(__file__).resolve().parents[1] / "results" / "runs"
ARCHIVE_DIR = Path(__file__).resolve().parents[1] / "results" / "archive"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

AGE_LIMIT = timedelta(days=30)
KEEP_LAST = 10

def parse_run_id(name: str):
    return datetime.strptime(name, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)

def archive_old_runs(dry_run: bool = False):
    now = datetime.now(timezone.utc)
    runs = sorted([d.name for d in RUNS_DIR.iterdir() if d.is_dir()])
    for run in runs:
        try:
            dt = parse_run_id(run)
        except Exception:
            continue
        if now - dt > AGE_LIMIT:
            src = RUNS_DIR / run
            tgt = ARCHIVE_DIR / f"{run}.tar.gz"
            print(f"Archiving {src} -> {tgt}")
            if not dry_run:
                shutil.make_archive(str(ARCHIVE_DIR / run), 'gztar', str(src))
                shutil.rmtree(src)
                idx = ARCHIVE_DIR / 'archive-index.json'
                data = json.loads(idx.read_text()) if idx.exists() else {}
                data[run] = {"archived_utc": now.isoformat(), "archive_path": str(tgt)}
                idx.write_text(json.dumps(data, indent=2))

    remaining = sorted([d.name for d in RUNS_DIR.iterdir() if d.is_dir()])
    if len(remaining) > KEEP_LAST:
        to_del = remaining[:-KEEP_LAST]
        for run in to_del:
            print(f"Deleting run {run} to enforce keep-last-{KEEP_LAST}")
            if not dry_run:
                shutil.rmtree(RUNS_DIR / run)

if __name__ == '__main__':
    # default: dry-run print only. Use dry_run=False to actually delete/archive
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--apply', action='store_true', help='Apply archives and deletions')
    args = p.parse_args()
    archive_old_runs(dry_run=not args.apply)

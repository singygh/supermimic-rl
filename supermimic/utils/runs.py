"""Per-launch run directories shared by the toy and Isaac Lab entry points."""
from datetime import datetime
import json
from pathlib import Path
import sys


def create_run_directory(log_root, run_name=None, *, resume_from=None, started_at=None):
    """Create a unique local-time directory, including when resuming training."""
    if run_name is not None and (not run_name or run_name in (".", "..")
                                 or "/" in run_name or "\\" in run_name):
        raise ValueError("run_name must be a nonempty directory suffix without path separators")
    started_at = started_at or datetime.now().astimezone()
    root = Path(log_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    base = started_at.strftime("%Y-%m-%d_%H-%M-%S_%f")
    if run_name:
        base += "_" + run_name
    run = root / base
    suffix = 0
    while True:
        try:
            run.mkdir()
            break
        except FileExistsError:
            suffix += 1
            run = root / f"{base}_{suffix}"
    (run / "run_info.json").write_text(json.dumps({
        "started_at": started_at.isoformat(), "run_name": run_name,
        "log_root": str(root), "resume_from": str(resume_from) if resume_from else None,
        "command": sys.argv,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Training run directory: {run}", flush=True)
    return run


def resolve_resume_checkpoint(resume, log_root):
    """Resolve before creating this launch's directory, so 'latest' is unambiguous."""
    from .checkpoints import latest_checkpoint

    if not resume:
        return None
    if resume == "latest":
        return latest_checkpoint(log_root)
    path = Path(resume).expanduser().resolve()
    if path.is_dir():
        return latest_checkpoint(path)
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    return path

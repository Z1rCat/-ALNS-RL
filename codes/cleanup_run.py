import argparse
import os
import shutil
import stat
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional


DEFAULT_DELETE_DIRS = ("data", "alns_outputs")


def _on_rm_error(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        raise


def _safe_rmtree(path: Path, *, attempts: int = 10, sleep_s: float = 1.0) -> None:
    last_exc: Optional[Exception] = None
    for i in range(attempts):
        try:
            shutil.rmtree(path, onerror=_on_rm_error)
            return
        except Exception as exc:
            last_exc = exc
            if i + 1 >= attempts:
                break
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed to delete {path}: {last_exc}")


def cleanup_run_dir(
    run_dir: Path,
    *,
    delete_dirs: Iterable[str] = DEFAULT_DELETE_DIRS,
    require_files: Iterable[str] = ("meta.json", "rl_trace.csv", "rl_training.csv", "baseline_wait.csv", "baseline_reroute.csv", "metrics.json"),
) -> List[str]:
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir not found or not a directory: {run_dir}")

    missing = [name for name in require_files if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Refuse to cleanup {run_dir}: missing required files: {missing}")

    deleted: List[str] = []
    for name in delete_dirs:
        target = run_dir / name
        if not target.exists():
            continue
        if target.is_dir():
            _safe_rmtree(target)
            deleted.append(name)
        else:
            raise RuntimeError(f"Refuse to cleanup {run_dir}: expected directory but got file: {target}")
    return deleted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cleanup a completed run_dir to save disk space.")
    parser.add_argument("--run-dir", required=True, help="Run directory to cleanup.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be deleted, without deleting.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    to_delete = [str(run_dir / name) for name in DEFAULT_DELETE_DIRS if (run_dir / name).exists()]
    if args.dry_run:
        print("[cleanup] dry-run delete:", to_delete)
        return 0
    try:
        deleted = cleanup_run_dir(run_dir)
        print(f"[cleanup] deleted: {deleted}")
        return 0
    except Exception as exc:
        print(f"[cleanup] failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
# coding: utf-8
"""
Batch re-draw paper figures for multiple run_* directories.

Example:
  python codes/redraw_paper_figures.py
  python codes/redraw_paper_figures.py --runs run_20260120_123246_371223_R30_V1_3_DQN_S42
  python codes/redraw_paper_figures.py --clean --window 30
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = REPO_ROOT / "codes" / "logs"
DEFAULT_PLOT_SCRIPT = REPO_ROOT / "codes" / "plotting" / "plot_paper_figure.py"

DEFAULT_RUNS = [

"A:/MYpython/34959_RL/codes/logs/run_20260125_152317_780783_R30_S5_1_A2C_HAT_SNA"
]


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _resolve_run_dir(run_root: Path, run: str) -> Path:
    candidate = Path(run)
    if candidate.is_absolute() or candidate.exists() or any(sep in run for sep in ("/", "\\")):
        return candidate.resolve()
    return (run_root / run).resolve()


def parse_args():
    parser = argparse.ArgumentParser(description="Batch re-draw paper figures for run directories.")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT), help="root folder that contains run_* dirs")
    parser.add_argument(
        "--plot-script",
        default=str(DEFAULT_PLOT_SCRIPT),
        help="path to plot_paper_figure.py",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="run directory names or paths; defaults to the preconfigured list",
    )
    parser.add_argument("--window", type=int, default=None, help="override smoothing window passed to plot script")
    parser.add_argument("--clean", action="store_true", help="delete run_dir/paper_figures before plotting")
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    parser.add_argument("--continue-on-error", action="store_true", help="continue after failures")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    plot_script = Path(args.plot_script).resolve()
    runs = DEFAULT_RUNS if not args.runs else args.runs
    runs = _dedupe_keep_order(runs)

    if not plot_script.exists():
        print(f"Error: plot script not found: {plot_script}", file=sys.stderr)
        return 2

    ok = 0
    failed = 0
    missing = 0
    for run in runs:
        run_dir = _resolve_run_dir(run_root, run)
        if not run_dir.exists():
            print(f"[skip] run dir not found: {run_dir}", file=sys.stderr)
            missing += 1
            if not args.continue_on_error:
                return 2
            continue

        out_dir = run_dir / "paper_figures"
        if args.clean and out_dir.exists():
            try:
                shutil.rmtree(out_dir)
            except Exception as exc:
                print(f"[warn] failed to remove {out_dir}: {exc}", file=sys.stderr)

        cmd: List[str] = [sys.executable, str(plot_script), "--run-dir", str(run_dir)]
        if args.window is not None:
            cmd += ["--window", str(args.window)]

        print(f"[plot] {run_dir.name}")
        if args.dry_run:
            print("  ", " ".join(cmd))
            ok += 1
            continue

        proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if proc.returncode == 0:
            ok += 1
        else:
            failed += 1
            print(f"[fail] {run_dir.name} (exit={proc.returncode})", file=sys.stderr)
            if not args.continue_on_error:
                return 1

    print(f"Done. ok={ok}, failed={failed}, missing={missing}")
    return 0 if failed == 0 and missing == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

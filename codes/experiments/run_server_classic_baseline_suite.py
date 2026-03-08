from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


THIS_DIR = Path(__file__).resolve().parent
CODES_DIR = THIS_DIR.parent
ROOT_DIR = CODES_DIR.parent
NEXUS_DIR = CODES_DIR / "nexus"
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from experiments.classic_baseline_suite_config import (
    DEFAULT_REQUEST_NUMBERS,
    DEFAULT_SEEDS,
    MAIN_TABLE_DISTS,
    TRAINABLE_VARIANTS,
    validate_distribution_subset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Server-grade orchestrator for the classic comparison suite: "
            "A2C/PPO/PPO_LSTM/RARL/PLR_UED plus random/always1/always0 baselines, "
            "with metrics, plots, and final aggregate reporting."
        )
    )
    parser.add_argument("--run-folder", type=str, default="classic_baseline_suite_runs")
    parser.add_argument("--report-folder", type=str, default="classic_baseline_suite_report")
    parser.add_argument("--max-workers", type=int, default=8, help="concurrent stream tasks")
    parser.add_argument("--generator-workers", type=int, default=1, help="generator workers per task")
    parser.add_argument("--skip-run", action="store_true", default=False)
    parser.add_argument("--skip-report", action="store_true", default=False)
    parser.add_argument("--recursive-report", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--notify-success", action="store_true", default=False)
    parser.add_argument("--no-notify-failure", action="store_false", dest="notify_failure", default=True)
    parser.add_argument(
        "--relax-watchdog",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="disable output/progress stall watchdogs to reduce false kills on powerful servers",
    )
    return parser.parse_args()


def _resolve_nexus_path(raw: str) -> Path:
    path = Path(str(raw or "").strip())
    if not path:
        raise ValueError("empty path")
    if path.is_absolute():
        return path.resolve()
    return (NEXUS_DIR / path).resolve()


def _run_cmd(cmd: List[str], cwd: Path, env: Dict[str, str], *, dry_run: bool) -> int:
    printable = " ".join(shlex.quote(str(x)) for x in cmd)
    print(f"[classic-suite] run: {printable}")
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(cwd), env=env).returncode


def _build_stream_cmd(args: argparse.Namespace, run_root: Path) -> List[str]:
    cmd = [
        sys.executable,
        str(CODES_DIR / "experiments" / "run_experiments_server_stream.py"),
        "--run-folder",
        str(run_root),
        "--max-workers",
        str(int(args.max_workers)),
        "--generator-workers",
        str(int(args.generator_workers)),
        "--run-baseline",
        "--baseline-include-random",
        "--run-plots",
        "--run-metrics",
        "--resume-existing",
        "--skip-completed",
        "--no-precheck",
    ]
    for variant in TRAINABLE_VARIANTS:
        cmd.extend(["--variant", variant])
    for dist_name in MAIN_TABLE_DISTS:
        cmd.extend(["--dist-name", dist_name])
    for request_number in DEFAULT_REQUEST_NUMBERS:
        cmd.extend(["--request-number", str(int(request_number))])
    for seed in DEFAULT_SEEDS:
        cmd.extend(["--seed", str(int(seed))])
    if args.notify_success:
        cmd.append("--notify-success")
    if not bool(args.notify_failure):
        cmd.append("--no-notify-failure")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def _build_report_cmd(args: argparse.Namespace, run_root: Path, report_root: Path) -> List[str]:
    cmd = [
        sys.executable,
        str(CODES_DIR / "analysis" / "build_classic_baseline_suite_report.py"),
        "--root",
        str(run_root),
        "--out-dir",
        str(report_root),
        "--summary-prefix",
        "classic_baseline_suite",
    ]
    if args.recursive_report:
        cmd.append("--recursive")
    return cmd


def _watchdog_env_overrides(enabled: bool) -> Dict[str, str]:
    if not enabled:
        return {}
    return {
        "MASTER_STALL_S": "0",
        "MASTER_PROGRESS_STALL_S": "0",
        "MASTER_MAX_WALL_S": "0",
        "BASELINE_STALL_S": "0",
        "BASELINE_PROGRESS_STALL_S": "0",
        "BASELINE_MAX_WALL_S": "0",
        "MASTER_POLL_S": "10",
        "BASELINE_POLL_S": "10",
        "RUN_HEARTBEAT_S": "20",
        "RUN_LOCK_LEASE_S": "7200",
    }


def main() -> int:
    args = parse_args()
    validate_distribution_subset(MAIN_TABLE_DISTS)

    run_root = _resolve_nexus_path(args.run_folder)
    report_root = _resolve_nexus_path(args.report_folder)
    run_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env_overrides = _watchdog_env_overrides(bool(args.relax_watchdog))
    env.update(env_overrides)

    manifest = {
        "run_root": str(run_root),
        "report_root": str(report_root),
        "variants": TRAINABLE_VARIANTS,
        "baseline_columns": ["random", "always1", "always0"],
        "seeds": DEFAULT_SEEDS,
        "request_numbers": DEFAULT_REQUEST_NUMBERS,
        "distributions": MAIN_TABLE_DISTS,
        "relax_watchdog": bool(args.relax_watchdog),
        "env_overrides": env_overrides,
        "commands": {},
    }

    stream_cmd = _build_stream_cmd(args=args, run_root=run_root)
    report_cmd = _build_report_cmd(args=args, run_root=run_root, report_root=report_root)
    manifest["commands"]["stream"] = stream_cmd
    manifest["commands"]["report"] = report_cmd

    manifest_path = report_root / "classic_baseline_suite_pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[classic-suite] wrote manifest: {manifest_path}")

    run_code = 0
    if not args.skip_run:
        run_code = _run_cmd(stream_cmd, cwd=ROOT_DIR, env=env, dry_run=bool(args.dry_run))
        if run_code != 0:
            print(f"[classic-suite][WARN] run stage returned exit={run_code}; continue to reporting with available runs")

    report_code = 0
    if not args.skip_report:
        report_code = _run_cmd(report_cmd, cwd=ROOT_DIR, env=env, dry_run=bool(args.dry_run))

    if run_code != 0:
        return int(run_code)
    return int(report_code)


if __name__ == "__main__":
    raise SystemExit(main())

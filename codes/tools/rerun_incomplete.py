#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


ROOT_DIR = Path(__file__).resolve().parents[2]
CODES_DIR = ROOT_DIR / "codes"
LOG_ROOT = CODES_DIR / "logs"


@dataclass
class RunInfo:
    run_dir: Path
    request_number: int
    dist_name: str
    algorithm: str
    seed: str


def parse_run_name(name: str) -> Optional[RunInfo]:
    if not name.startswith("run_") or "_S" not in name:
        return None
    base, seed = name.rsplit("_S", 1)
    if "_R" not in base:
        return None
    _, rest = base.split("_R", 1)
    parts = rest.split("_")
    if len(parts) < 2:
        return None
    req_str = parts[0]
    try:
        request_number = int(req_str)
    except ValueError:
        return None
    tail = parts[1:]
    known_algos = [
        "PPO_HAT",
        "A2C_HAT",
        "LBKLAC",
        "DRCB",
        "PPO",
        "A2C",
        "DQN",
        "HAT",
    ]
    algorithm = None
    algo_len = 0
    for cand in known_algos:
        cand_parts = cand.split("_")
        if len(cand_parts) <= len(tail) and tail[-len(cand_parts) :] == cand_parts:
            algorithm = cand
            algo_len = len(cand_parts)
            break
    if algorithm is None:
        algorithm = tail[-1]
        algo_len = 1
    dist_tokens = tail[:-algo_len]
    if not dist_tokens:
        return None
    dist_name = "_".join(dist_tokens)
    return RunInfo(
        run_dir=Path(name),
        request_number=request_number,
        dist_name=dist_name,
        algorithm=algorithm,
        seed=seed,
    )


def is_hat_algorithm(algorithm: str) -> bool:
    return "HAT" in algorithm.upper()


def run_command(cmd: List[str], cwd: Optional[Path] = None, dry_run: bool = False) -> int:
    if dry_run:
        print("[dry-run]", " ".join(cmd))
        return 0
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None).returncode


def run_master(info: RunInfo, dry_run: bool) -> int:
    cmd = [
        sys.executable,
        str(CODES_DIR / "Dynamic_master34959.py"),
        "--dist_name",
        info.dist_name,
        "--request_number",
        str(info.request_number),
        "--algorithm",
        info.algorithm,
        "--workers",
        "1",
        "--seed",
        str(info.seed).replace("NA", "42"),
        "--run-name",
        info.run_dir.name,
    ]
    return run_command(cmd, cwd=ROOT_DIR, dry_run=dry_run)


def run_baseline(run_dir: Path, policy: str, include_random: bool, dry_run: bool) -> int:
    cmd = [
        sys.executable,
        str(CODES_DIR / "experiments" / "run_benchmark_replay.py"),
        "--run-dir",
        str(run_dir),
        "--policy",
        policy,
    ]
    if include_random and policy == "all":
        cmd.append("--include-random")
    return run_command(cmd, cwd=ROOT_DIR, dry_run=dry_run)


def run_metrics(run_dir: Path, dry_run: bool) -> int:
    cmd = [
        sys.executable,
        str(CODES_DIR / "analysis" / "compute_metrics.py"),
        "--run-dir",
        str(run_dir),
    ]
    code = run_command(cmd, cwd=ROOT_DIR, dry_run=dry_run)
    if code != 0:
        print(f"[fail] metrics exit={code} {run_dir.name}")
    return code


def run_plots(run_dir: Path, dry_run: bool) -> int:
    cmd = [
        sys.executable,
        str(CODES_DIR / "plotting" / "plot_paper_figure.py"),
        "--run-dir",
        str(run_dir),
    ]
    code = run_command(cmd, cwd=ROOT_DIR, dry_run=dry_run)
    if code != 0:
        print(f"[fail] plots exit={code} {run_dir.name}")
    return code


def resolve_target_runs(logs_root: Path, run_dir: Optional[Path]) -> Iterable[Path]:
    if run_dir is not None:
        return [run_dir]
    if not logs_root.exists():
        return []
    return sorted([p for p in logs_root.iterdir() if p.is_dir() and p.name.startswith("run_")])


def baseline_state(run_dir: Path) -> Tuple[bool, bool, bool]:
    return (
        (run_dir / "baseline_wait.csv").exists(),
        (run_dir / "baseline_reroute.csv").exists(),
        (run_dir / "baseline_random.csv").exists(),
    )


def _failed_marker_path(run_dir: Path) -> Path:
    return run_dir / "FAILED.json"


def _write_failed_marker(run_dir: Path, stage: str, code: int) -> None:
    payload = {
        "stage": stage,
        "status": "failed",
        "exit_code": int(code),
    }
    try:
        payload["ts"] = __import__("time").time()
        _failed_marker_path(run_dir).write_text(
            __import__("json").dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _clear_failed_marker(run_dir: Path) -> None:
    _safe_unlink(_failed_marker_path(run_dir), dry_run=False)


def _clean_attempt_files(run_dir: Path, dry_run: bool) -> None:
    for path in run_dir.glob("baseline_*.csv.attempt*"):
        _safe_unlink(path, dry_run)


def _safe_unlink(path: Path, dry_run: bool) -> None:
    if not path.exists():
        return
    if dry_run:
        print(f"[dry-run] delete {path}")
        return
    try:
        path.unlink()
    except Exception:
        pass


def clean_before_rerun(run_dir: Path, *, policy: str, full: bool, dry_run: bool) -> None:
    if full:
        for name in ["rl_trace.csv", "rl_training.csv", "rl_summary.csv", "console_output.txt"]:
            _safe_unlink(run_dir / name, dry_run)
        for name in ["baseline_wait.csv", "baseline_reroute.csv", "baseline_random.csv"]:
            _safe_unlink(run_dir / name, dry_run)
    else:
        if policy == "wait":
            _safe_unlink(run_dir / "baseline_wait.csv", dry_run)
        elif policy == "reroute":
            _safe_unlink(run_dir / "baseline_reroute.csv", dry_run)
        elif policy == "random":
            _safe_unlink(run_dir / "baseline_random.csv", dry_run)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="", help="single run directory to process")
    parser.add_argument("--logs-root", default=str(LOG_ROOT), help="logs root (default: codes/logs)")
    parser.add_argument("--workers", type=int, default=0, help="parallel workers for rerun (0=auto)")
    parser.add_argument(
        "--clean",
        action="store_true",
        default=True,
        help="delete existing outputs before rerun (default: on)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_false",
        dest="clean",
        help="disable deleting existing outputs before rerun",
    )
    parser.add_argument(
        "--include-random",
        action="store_true",
        default=True,
        help="include random baseline when running all (default: on)",
    )
    parser.add_argument(
        "--no-include-random",
        action="store_false",
        dest="include_random",
        help="disable random baseline when running all",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    logs_root = Path(args.logs_root).resolve()
    include_random = bool(args.include_random)
    dry_run = bool(args.dry_run)
    clean_before = bool(args.clean)
    worker_count = int(args.workers) if args.workers is not None else 0
    if worker_count <= 0:
        worker_count = max(1, (os.cpu_count() or 2) - 1)

    targets = list(resolve_target_runs(logs_root, run_dir))
    if not targets:
        print("No runs found.")
        return 1

    def _process_run(run_path: Path) -> None:
        info = parse_run_name(run_path.name)
        if info is None:
            print(f"[skip] unrecognized run name: {run_path.name}")
            return
        info.run_dir = run_path
        failed_marker = _failed_marker_path(run_path)
        force_full = failed_marker.exists()

        try:
            if is_hat_algorithm(info.algorithm) or force_full:
                tag = "HAT" if is_hat_algorithm(info.algorithm) else "FAILED"
                print(f"[rerun] {tag} detected -> full rerun: {run_path.name}")
                if clean_before:
                    clean_before_rerun(run_path, policy="all", full=True, dry_run=dry_run)
                code = run_master(info, dry_run=dry_run)
                if code != 0:
                    print(f"[fail] master exit={code} {run_path.name}")
                    _write_failed_marker(run_path, "master", code)
                    return
                code = run_baseline(run_path, "all", include_random, dry_run=dry_run)
                if code != 0:
                    print(f"[fail] baseline exit={code} {run_path.name}")
                    _write_failed_marker(run_path, "baseline", code)
                    return
                if run_metrics(run_path, dry_run=dry_run) != 0:
                    _write_failed_marker(run_path, "metrics", 1)
                    return
                if run_plots(run_path, dry_run=dry_run) != 0:
                    _write_failed_marker(run_path, "plots", 1)
                    return
                _clear_failed_marker(run_path)
                return

            if (run_path / "paper_figures").exists():
                print(f"[ok] paper_figures exists -> skip: {run_path.name}")
                if run_metrics(run_path, dry_run=dry_run) != 0:
                    _write_failed_marker(run_path, "metrics", 1)
                else:
                    _clear_failed_marker(run_path)
                return

            has_wait, has_reroute, has_random = baseline_state(run_path)
            if has_random:
                policy = "random"
            elif has_reroute:
                policy = "reroute"
            elif has_wait:
                policy = "wait"
            else:
                policy = "all"

            if policy == "all":
                print(f"[rerun] no baseline -> full rerun: {run_path.name}")
                if clean_before:
                    clean_before_rerun(run_path, policy=policy, full=True, dry_run=dry_run)
                code = run_master(info, dry_run=dry_run)
                if code != 0:
                    print(f"[fail] master exit={code} {run_path.name}")
                    _write_failed_marker(run_path, "master", code)
                    return
            else:
                print(f"[rerun] baseline policy={policy}: {run_path.name}")
                if clean_before:
                    clean_before_rerun(run_path, policy=policy, full=False, dry_run=dry_run)

            code = run_baseline(run_path, policy, include_random, dry_run=dry_run)
            if code != 0:
                print(f"[fail] baseline exit={code} {run_path.name}")
                _write_failed_marker(run_path, "baseline", code)
                return
            if run_metrics(run_path, dry_run=dry_run) != 0:
                _write_failed_marker(run_path, "metrics", 1)
                return
            if run_plots(run_path, dry_run=dry_run) != 0:
                _write_failed_marker(run_path, "plots", 1)
                return
            _clear_failed_marker(run_path)
        finally:
            _clean_attempt_files(run_path, dry_run=dry_run)

    max_workers = min(len(targets), worker_count)
    if max_workers <= 1:
        for run_path in targets:
            _process_run(run_path)
    else:
        print(f"[parallel] workers={max_workers} targets={len(targets)}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_run, run_path) for run_path in targets]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

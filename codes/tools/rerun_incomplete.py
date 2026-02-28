#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations

import argparse
import csv
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


ROOT_DIR = Path(__file__).resolve().parents[2]
CODES_DIR = ROOT_DIR / "codes"
LOG_ROOT = CODES_DIR / "logs"
MASTER_OUTPUT_FILES = ("rl_trace.csv", "rl_training.csv", "rl_summary.csv", "console_output.txt")
BASELINE_OUTPUT_FILES = ("baseline_wait.csv", "baseline_reroute.csv", "baseline_random.csv")


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
        "PPO_HAT_LSTM",
        "PPO_HAT_PDI",
        "PPO_PROTOMEM",
        "PPO_HAT_MOE",
        "A2C_HAT_MOE",
        "QRDQN_CVAR",
        "BE_CVAR_DQN",
        "PPO_LSTM",
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


def run_command(
    cmd: List[str],
    cwd: Optional[Path] = None,
    dry_run: bool = False,
    env: Optional[dict] = None,
) -> int:
    if dry_run:
        print("[dry-run]", " ".join(cmd))
        return 0
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env).returncode


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
    env = dict(os.environ)
    env["RL_LOG_ROOT"] = str(info.run_dir.parent)
    return run_command(cmd, cwd=ROOT_DIR, dry_run=dry_run, env=env)


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


def baseline_required_policies(include_random: bool) -> List[str]:
    policies = ["wait", "reroute"]
    if include_random:
        policies.append("random")
    return policies


def baseline_policy_filename(policy: str) -> str:
    value = str(policy).strip().lower()
    if value == "wait":
        return "baseline_wait.csv"
    if value == "reroute":
        return "baseline_reroute.csv"
    if value == "random":
        return "baseline_random.csv"
    raise ValueError(f"unknown baseline policy: {policy}")


def baseline_policy_path(run_dir: Path, policy: str) -> Path:
    return run_dir / baseline_policy_filename(policy)


def baseline_presence_flags(run_dir: Path) -> Dict[str, bool]:
    wait_path = baseline_policy_path(run_dir, "wait")
    reroute_path = baseline_policy_path(run_dir, "reroute")
    random_path = baseline_policy_path(run_dir, "random")
    wait_file = _csv_has_data_row(wait_path, min_rows=1)
    reroute_file = _csv_has_data_row(reroute_path, min_rows=1)
    random_file = _csv_has_data_row(random_path, min_rows=1)
    return {
        "wait": wait_file,
        "reroute": reroute_file,
        "random": random_file,
        "wait_file": wait_file,
        "reroute_file": reroute_file,
        "random_file": random_file,
        "wait_impl": _csv_has_phase_row(wait_path, phase="implement", min_rows=1),
        "reroute_impl": _csv_has_phase_row(reroute_path, phase="implement", min_rows=1),
        "random_impl": _csv_has_phase_row(random_path, phase="implement", min_rows=1),
        "paper": _dir_has_content(run_dir / "paper_figures"),
    }


def baseline_success_flags(run_dir: Path) -> Dict[str, bool]:
    flags = baseline_presence_flags(run_dir)
    # Avoid baseline/plot cyclic dependency: baseline success should not require paper.
    wait_success = bool(flags["reroute"] or flags["wait_impl"] or (flags["paper"] and flags["wait"]))
    reroute_success = bool(
        flags["random"] or flags["reroute_impl"] or (flags["paper"] and flags["wait"] and flags["reroute"])
    )
    random_success = bool(flags["random_impl"] or (flags["paper"] and flags["wait"] and flags["reroute"] and flags["random"]))
    return {
        "wait": wait_success,
        "reroute": reroute_success,
        "random": random_success,
        "paper": bool(flags["paper"]),
        "wait_file": bool(flags["wait_file"]),
        "reroute_file": bool(flags["reroute_file"]),
        "random_file": bool(flags["random_file"]),
        "wait_impl": bool(flags["wait_impl"]),
        "reroute_impl": bool(flags["reroute_impl"]),
        "random_impl": bool(flags["random_impl"]),
    }


def missing_baseline_policies(run_dir: Path, include_random: bool) -> List[str]:
    success = baseline_success_flags(run_dir)
    missing: List[str] = []
    for policy in baseline_required_policies(include_random):
        if not bool(success.get(policy, False)):
            missing.append(policy)
    return missing


def run_cleanup(run_dir: Path, dry_run: bool) -> int:
    cmd = [
        sys.executable,
        str(CODES_DIR / "tools" / "cleanup_run.py"),
        "--run-dir",
        str(run_dir),
    ]
    code = run_command(cmd, cwd=ROOT_DIR, dry_run=dry_run)
    if code != 0:
        print(f"[fail] cleanup exit={code} {run_dir.name}")
    return code


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


def _file_has_content(path: Path, min_bytes: int = 1) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size >= int(min_bytes)
    except Exception:
        return False


def _csv_has_data_row(path: Path, min_rows: int = 1) -> bool:
    if not _file_has_content(path, min_bytes=8):
        return False
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                next(reader)
            except StopIteration:
                return False
            count = 0
            for row in reader:
                if not row:
                    continue
                if all((str(cell).strip() == "") for cell in row):
                    continue
                count += 1
                if count >= int(min_rows):
                    return True
        return False
    except Exception:
        return False


def _csv_has_phase_row(path: Path, phase: str, min_rows: int = 1) -> bool:
    if not _file_has_content(path, min_bytes=8):
        return False
    target = str(phase).strip().lower()
    if not target:
        return False
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return False
            field_map = {str(name).strip().lower(): name for name in reader.fieldnames if name}
            phase_key = field_map.get("phase")
            if not phase_key:
                return False
            count = 0
            for row in reader:
                value = str(row.get(phase_key, "")).strip().lower()
                if value != target:
                    continue
                count += 1
                if count >= int(min_rows):
                    return True
        return False
    except Exception:
        return False


def _dir_has_content(path: Path) -> bool:
    try:
        return path.exists() and path.is_dir() and any(path.iterdir())
    except Exception:
        return False


def collect_stage_state(run_dir: Path, include_random: bool) -> Dict[str, bool]:
    trace_ok = _csv_has_data_row(run_dir / "rl_trace.csv", min_rows=5)
    train_ok = _csv_has_data_row(run_dir / "rl_training.csv", min_rows=5)
    summary_ok = _csv_has_data_row(run_dir / "rl_summary.csv", min_rows=1)
    baseline_any = any(
        _csv_has_data_row(run_dir / name, min_rows=1)
        for name in ("baseline_wait.csv", "baseline_reroute.csv", "baseline_random.csv")
    )
    master = trace_ok and train_ok and (summary_ok or baseline_any)
    baseline = len(missing_baseline_policies(run_dir, include_random)) == 0
    metrics = _file_has_content(run_dir / "metrics.json", min_bytes=16)
    plots = _dir_has_content(run_dir / "paper_figures")
    cleanup = not (run_dir / "data").exists() and not (run_dir / "alns_outputs").exists()
    return {
        "master": master,
        "baseline": baseline,
        "metrics": metrics,
        "plots": plots,
        "cleanup": cleanup,
        "complete": master and baseline and metrics and plots,
    }


def _failed_marker_path(run_dir: Path) -> Path:
    return run_dir / "FAILED.json"


def _load_failed_stage(run_dir: Path) -> str:
    path = _failed_marker_path(run_dir)
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if not isinstance(data, dict):
        return ""
    return str(data.get("stage", "")).strip().lower()


def _apply_failed_stage_hint(state: Dict[str, bool], failed_stage: str) -> Dict[str, bool]:
    stage = str(failed_stage or "").strip().lower()
    if not stage:
        return state
    patched = dict(state)
    if stage == "master":
        patched["master"] = False
        patched["baseline"] = False
        patched["metrics"] = False
        patched["plots"] = False
        patched["cleanup"] = False
    elif stage == "baseline":
        patched["baseline"] = False
        patched["metrics"] = False
        patched["plots"] = False
        patched["cleanup"] = False
    elif stage == "metrics":
        patched["metrics"] = False
        patched["plots"] = False
        patched["cleanup"] = False
    elif stage in {"plot", "plots"}:
        patched["plots"] = False
        patched["cleanup"] = False
    elif stage == "cleanup":
        patched["cleanup"] = False
    patched["complete"] = patched["master"] and patched["baseline"] and patched["metrics"] and patched["plots"]
    return patched


def _write_failed_marker(run_dir: Path, stage: str, code: int, extra: Optional[Dict[str, object]] = None) -> None:
    payload = {
        "stage": stage,
        "status": "failed",
        "exit_code": int(code),
        "ts": time.time(),
    }
    if extra:
        payload.update(extra)
    try:
        _failed_marker_path(run_dir).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _clear_failed_marker(run_dir: Path) -> None:
    _safe_unlink(_failed_marker_path(run_dir), dry_run=False)


def _clean_attempt_files(run_dir: Path, dry_run: bool) -> None:
    for path in run_dir.glob("baseline_*.csv.attempt*"):
        _safe_unlink(path, dry_run)


def _rotate_master_files_precheck(run_dir: Path, dry_run: bool) -> None:
    for name in MASTER_OUTPUT_FILES:
        path = run_dir / name
        _safe_unlink(path, dry_run)
    stop_flag = run_dir / "34959.txt"
    _safe_unlink(stop_flag, dry_run)


def _delete_baseline_policy_file(run_dir: Path, policy: str, dry_run: bool) -> None:
    path = baseline_policy_path(run_dir, policy)
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


def _safe_rmtree(path: Path, dry_run: bool) -> None:
    if not path.exists() or not path.is_dir():
        return
    if dry_run:
        print(f"[dry-run] delete dir {path}")
        return
    try:
        __import__("shutil").rmtree(path)
    except Exception:
        pass


def clean_before_stage(run_dir: Path, *, stage: str, dry_run: bool) -> None:
    stage_name = stage.lower().strip()
    if stage_name == "master":
        for name in MASTER_OUTPUT_FILES:
            _safe_unlink(run_dir / name, dry_run)
        for name in BASELINE_OUTPUT_FILES:
            _safe_unlink(run_dir / name, dry_run)
        _safe_unlink(run_dir / "metrics.json", dry_run)
        _safe_rmtree(run_dir / "paper_figures", dry_run)
        _safe_unlink(run_dir / "DONE.json", dry_run)
        return
    if stage_name == "baseline":
        for name in BASELINE_OUTPUT_FILES:
            _safe_unlink(run_dir / name, dry_run)
        _safe_unlink(run_dir / "metrics.json", dry_run)
        _safe_rmtree(run_dir / "paper_figures", dry_run)
        _safe_unlink(run_dir / "DONE.json", dry_run)
        return
    if stage_name == "metrics":
        _safe_unlink(run_dir / "metrics.json", dry_run)
        return
    if stage_name == "plots":
        _safe_rmtree(run_dir / "paper_figures", dry_run)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="", help="single run directory to process")
    parser.add_argument("--logs-root", default=str(LOG_ROOT), help="logs root (default: codes/logs)")
    parser.add_argument("--workers", type=int, default=0, help="parallel workers for rerun (0=auto)")
    parser.add_argument(
        "--algorithm",
        action="append",
        default=[],
        help="only process matching algorithm(s); repeatable",
    )
    parser.add_argument(
        "--dist-name",
        action="append",
        default=[],
        help="only process matching distribution name(s); repeatable",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="delete existing outputs for a stage before rerun (default: off)",
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
    allowed_algorithms: Set[str] = {str(v).strip().upper() for v in list(args.algorithm or []) if str(v).strip()}
    allowed_dists: Set[str] = {str(v).strip() for v in list(args.dist_name or []) if str(v).strip()}
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
        if allowed_algorithms and info.algorithm.upper() not in allowed_algorithms:
            print(f"[skip] algorithm filtered: {run_path.name}")
            return
        if allowed_dists and info.dist_name not in allowed_dists:
            print(f"[skip] dist filtered: {run_path.name}")
            return
        info.run_dir = run_path
        force_full = is_hat_algorithm(info.algorithm)
        state = collect_stage_state(run_path, include_random=include_random)
        baseline_detail = baseline_success_flags(run_path)
        failed_stage = _load_failed_stage(run_path)
        state = _apply_failed_stage_hint(state, failed_stage)
        master_reran = False
        print(
            f"[state] {run_path.name} "
            f"master={int(state['master'])} baseline={int(state['baseline'])} "
            f"metrics={int(state['metrics'])} plots={int(state['plots'])} cleanup={int(state['cleanup'])}"
            + (f" failed_stage={failed_stage}" if failed_stage else "")
            + " | "
            + (
                f"W={int(baseline_detail.get('wait_file', False))} "
                f"R={int(baseline_detail.get('reroute_file', False))} "
                f"N={int(baseline_detail.get('random_file', False))} "
                f"WI={int(baseline_detail.get('wait_impl', False))} "
                f"RI={int(baseline_detail.get('reroute_impl', False))} "
                f"NI={int(baseline_detail.get('random_impl', False))} "
                f"P={int(baseline_detail.get('paper', False))} "
                f"wait_ok={int(baseline_detail.get('wait', False))} "
                f"reroute_ok={int(baseline_detail.get('reroute', False))} "
                f"random_ok={int(baseline_detail.get('random', False))}"
            )
        )

        try:
            if state["complete"] and state["cleanup"]:
                print(f"[ok] completed + cleaned -> skip: {run_path.name}")
                if not dry_run:
                    _clear_failed_marker(run_path)
                return

            if force_full:
                print(f"[rerun] HAT detected -> rerun from master: {run_path.name}")
                if clean_before:
                    clean_before_stage(run_path, stage="master", dry_run=dry_run)
                else:
                    _rotate_master_files_precheck(run_path, dry_run=dry_run)
                code = run_master(info, dry_run=dry_run)
                if code != 0:
                    print(f"[fail] master exit={code} {run_path.name}")
                    if not dry_run:
                        _write_failed_marker(run_path, "master", code)
                    return
                state["master"] = True
                master_reran = True
                state["baseline"] = False
                state["metrics"] = False
                state["plots"] = False
                state["cleanup"] = False

            if not state["master"]:
                print(f"[rerun] master incomplete -> rerun master: {run_path.name}")
                if clean_before:
                    clean_before_stage(run_path, stage="master", dry_run=dry_run)
                else:
                    _rotate_master_files_precheck(run_path, dry_run=dry_run)
                code = run_master(info, dry_run=dry_run)
                if code != 0:
                    print(f"[fail] master exit={code} {run_path.name}")
                    if not dry_run:
                        _write_failed_marker(run_path, "master", code)
                    return
                state["master"] = True
                master_reran = True
                state["baseline"] = False
                state["metrics"] = False
                state["plots"] = False
                state["cleanup"] = False

            if not state["baseline"]:
                if master_reran:
                    policies = baseline_required_policies(include_random)
                else:
                    policies = missing_baseline_policies(run_path, include_random)
                print(f"[rerun] baseline incomplete -> run missing={policies}: {run_path.name}")
                if clean_before:
                    clean_before_stage(run_path, stage="baseline", dry_run=dry_run)
                    policies = baseline_required_policies(include_random)
                for policy in policies:
                    if not clean_before:
                        _delete_baseline_policy_file(run_path, policy, dry_run=dry_run)
                    code = run_baseline(run_path, policy, include_random, dry_run=dry_run)
                    if code != 0:
                        print(f"[fail] baseline policy={policy} exit={code} {run_path.name}")
                        if not dry_run:
                            _write_failed_marker(run_path, "baseline", code, extra={"policy": policy})
                        return
                if len(missing_baseline_policies(run_path, include_random)) != 0 and not dry_run:
                    _write_failed_marker(
                        run_path,
                        "baseline",
                        1,
                        extra={"reason": "incomplete_after_policy_runs", "missing": missing_baseline_policies(run_path, include_random)},
                    )
                    return
                state["baseline"] = True
                state["metrics"] = False
                state["plots"] = False
                state["cleanup"] = False

            if not state["metrics"]:
                if clean_before:
                    clean_before_stage(run_path, stage="metrics", dry_run=dry_run)
                if run_metrics(run_path, dry_run=dry_run) != 0:
                    if not dry_run:
                        _write_failed_marker(run_path, "metrics", 1)
                    return
                state["metrics"] = True

            if not state["plots"]:
                if clean_before:
                    clean_before_stage(run_path, stage="plots", dry_run=dry_run)
                if run_plots(run_path, dry_run=dry_run) != 0:
                    if not dry_run:
                        _write_failed_marker(run_path, "plots", 1)
                    return
                state["plots"] = True

            if state["master"] and state["baseline"] and state["metrics"] and state["plots"] and not state["cleanup"]:
                if run_cleanup(run_path, dry_run=dry_run) != 0:
                    if not dry_run:
                        _write_failed_marker(run_path, "cleanup", 1)
                    return
                state["cleanup"] = True

            if dry_run:
                return

            final_state = collect_stage_state(run_path, include_random=include_random)
            if final_state["complete"]:
                _clear_failed_marker(run_path)
                print(f"[ok] resume complete: {run_path.name}")
                return

            print(f"[fail] postcheck incomplete: {run_path.name} state={final_state}")
            _write_failed_marker(run_path, "postcheck", 1, extra={"state": final_state})
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

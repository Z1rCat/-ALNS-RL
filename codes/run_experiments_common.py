import concurrent.futures
import datetime
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


ROOT_DIR = Path(__file__).resolve().parent.parent
CODES_DIR = ROOT_DIR / "codes"
LOG_ROOT = CODES_DIR / "logs"


@dataclass
class ExperimentConfig:
    name: str
    distributions: List[str]
    request_numbers: List[int]
    algorithms: List[str]
    seeds: List[int]
    generator_workers: int = 1
    max_workers: Optional[int] = None
    run_baseline: bool = True
    run_plots: bool = True
    # Explicitly exclude specific (dist, request_number, algorithm, seed) tasks.
    # Useful for resuming partial runs without rerunning completed combinations.
    exclude_tasks: Optional[List[Tuple[str, int, str, int]]] = None


def detect_physical_cores() -> int:
    try:
        import psutil

        count = psutil.cpu_count(logical=False)
        if count:
            return count
    except Exception:
        pass
    return os.cpu_count() or 1


def resolve_max_workers(config: ExperimentConfig, override: Optional[int]) -> int:
    if override is not None and override > 0:
        return override
    if config.max_workers is not None and config.max_workers > 0:
        return config.max_workers
    return max(1, detect_physical_cores() - 2)


def build_run_name(dist_name: str, request_number: int, algorithm: str, seed: Optional[int]) -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    seed_tag = f"S{seed}" if seed is not None else "SNA"
    return f"run_{timestamp}_R{request_number}_{dist_name}_{algorithm}_{seed_tag}"


def run_command(cmd: List[str], cwd: Optional[Path] = None) -> int:
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    return proc.returncode


def run_task(task: Tuple[str, int, str, int], config: ExperimentConfig, dry_run: bool) -> Tuple[str, str]:
    dist_name, request_number, algorithm, seed = task
    run_name = build_run_name(dist_name, request_number, algorithm, seed)
    run_dir = LOG_ROOT / run_name

    master_cmd = [
        sys.executable,
        str(CODES_DIR / "Dynamic_master34959.py"),
        "--dist_name",
        dist_name,
        "--request_number",
        str(request_number),
        "--algorithm",
        algorithm,
        "--workers",
        str(config.generator_workers),
        "--seed",
        str(seed),
        "--run-name",
        run_name,
    ]

    baseline_cmd = [
        sys.executable,
        str(CODES_DIR / "run_benchmark_replay.py"),
        "--run-dir",
        str(run_dir),
        "--policy",
        "all",
    ]

    plot_cmd = [
        sys.executable,
        str(CODES_DIR / "plot_paper_figure.py"),
        "--run-dir",
        str(run_dir),
    ]

    print(f"[{config.name}] start {run_name}")
    if dry_run:
        print("  master:", " ".join(master_cmd))
        if config.run_baseline:
            print("  baseline:", " ".join(baseline_cmd))
        if config.run_plots:
            print("  plot:", " ".join(plot_cmd))
        return run_name, "dry_run"

    code = run_command(master_cmd, cwd=ROOT_DIR)
    if code != 0:
        return run_name, "failed_master"

    if config.run_baseline:
        code = run_command(baseline_cmd, cwd=ROOT_DIR)
        if code != 0:
            return run_name, "failed_baseline"

    if config.run_plots:
        code = run_command(plot_cmd, cwd=ROOT_DIR)
        if code != 0:
            return run_name, "failed_plot"

    return run_name, "ok"


def build_tasks(config: ExperimentConfig) -> List[Tuple[str, int, str, int]]:
    tasks: List[Tuple[str, int, str, int]] = []
    excluded = set()
    if config.exclude_tasks:
        for dist_name, request_number, algorithm, seed in config.exclude_tasks:
            excluded.add((str(dist_name), int(request_number), str(algorithm), int(seed)))
    for dist_name in config.distributions:
        for request_number in config.request_numbers:
            for algorithm in config.algorithms:
                for seed in config.seeds:
                    key = (str(dist_name), int(request_number), str(algorithm), int(seed))
                    if key in excluded:
                        continue
                    tasks.append((dist_name, request_number, algorithm, seed))
    return tasks


def run_experiments(config: ExperimentConfig, max_workers: Optional[int], dry_run: bool) -> int:
    tasks = build_tasks(config)
    if not tasks:
        print("No tasks to run.")
        return 1

    worker_count = resolve_max_workers(config, max_workers)
    print(f"[{config.name}] total tasks: {len(tasks)} | workers: {worker_count}")

    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(run_task, task, config, dry_run) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            run_name, status = future.result()
            print(f"[{config.name}] {run_name} -> {status}")
            if status not in {"ok", "dry_run"}:
                failed += 1
    return failed

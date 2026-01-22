import concurrent.futures
import datetime
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT_DIR = Path(__file__).resolve().parent.parent
CODES_DIR = ROOT_DIR / "codes"
LOG_ROOT = CODES_DIR / "logs"


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


class ResourceMonitor:
    def __init__(self, interval_s: float = 1.0) -> None:
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._samples: List[Dict[str, Any]] = []
        self._notes: List[str] = []

        self._psutil = None
        try:
            import psutil  # type: ignore

            self._psutil = psutil
        except Exception:
            self._psutil = None

        self._has_proc = Path("/proc/stat").exists() and Path("/proc/meminfo").exists()
        self._prev_cpu_total: Optional[int] = None
        self._prev_cpu_idle: Optional[int] = None

        self._has_nvidia_smi = shutil.which("nvidia-smi") is not None

    def start(self) -> None:
        if self._thread is not None:
            return
        if self._psutil is None and not self._has_proc and not self._has_nvidia_smi:
            self._notes.append("No psutil/procfs/nvidia-smi detected; resource sampling disabled.")
            return

        if self._psutil is not None:
            try:
                self._psutil.cpu_percent(interval=None)
            except Exception:
                pass
        if self._has_proc:
            try:
                total, idle = self._read_proc_cpu_times()
                self._prev_cpu_total = total
                self._prev_cpu_idle = idle
            except Exception as exc:
                self._notes.append(f"Failed to init /proc CPU sampling: {exc}")
                self._has_proc = False

        self._thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is None:
            return
        self._thread.join(timeout=max(5.0, self.interval_s * 2.0))

    def summary(self) -> Dict[str, Any]:
        cpu_vals = [s["cpu_percent"] for s in self._samples if isinstance(s.get("cpu_percent"), (int, float))]
        ram_vals = [s["ram_used_gb"] for s in self._samples if isinstance(s.get("ram_used_gb"), (int, float))]
        gpu_vals = [s["gpu_util_percent"] for s in self._samples if isinstance(s.get("gpu_util_percent"), (int, float))]
        gpu_mem_vals = [
            s["gpu_mem_used_mb"] for s in self._samples if isinstance(s.get("gpu_mem_used_mb"), (int, float))
        ]

        return {
            "sample_interval_sec": self.interval_s,
            "sample_count": len(self._samples),
            "cpu_percent_avg": _mean([float(v) for v in cpu_vals]),
            "cpu_percent_peak": float(max(cpu_vals)) if cpu_vals else None,
            "ram_used_gb_avg": _mean([float(v) for v in ram_vals]),
            "ram_used_gb_peak": float(max(ram_vals)) if ram_vals else None,
            "gpu_util_percent_avg": _mean([float(v) for v in gpu_vals]),
            "gpu_util_percent_peak": float(max(gpu_vals)) if gpu_vals else None,
            "gpu_mem_used_mb_avg": _mean([float(v) for v in gpu_mem_vals]),
            "gpu_mem_used_mb_peak": float(max(gpu_mem_vals)) if gpu_mem_vals else None,
            "notes": self._notes,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            sample: Dict[str, Any] = {"ts": time.time()}

            cpu_percent = self._sample_cpu_percent()
            if cpu_percent is not None:
                sample["cpu_percent"] = cpu_percent

            ram_used_gb = self._sample_ram_used_gb()
            if ram_used_gb is not None:
                sample["ram_used_gb"] = ram_used_gb

            gpu = self._sample_gpu()
            if gpu is not None:
                sample.update(gpu)

            self._samples.append(sample)
            self._stop.wait(self.interval_s)

    def _sample_cpu_percent(self) -> Optional[float]:
        if self._psutil is not None:
            try:
                return float(self._psutil.cpu_percent(interval=None))
            except Exception:
                return None
        if self._has_proc:
            try:
                total, idle = self._read_proc_cpu_times()
                if self._prev_cpu_total is None or self._prev_cpu_idle is None:
                    self._prev_cpu_total = total
                    self._prev_cpu_idle = idle
                    return None
                delta_total = total - self._prev_cpu_total
                delta_idle = idle - self._prev_cpu_idle
                self._prev_cpu_total = total
                self._prev_cpu_idle = idle
                if delta_total <= 0:
                    return None
                busy = max(0, delta_total - delta_idle)
                return float(busy / delta_total * 100.0)
            except Exception:
                return None
        return None

    def _sample_ram_used_gb(self) -> Optional[float]:
        if self._psutil is not None:
            try:
                used = float(self._psutil.virtual_memory().used)
                return used / (1024.0**3)
            except Exception:
                return None
        if self._has_proc:
            try:
                used_bytes = self._read_proc_mem_used_bytes()
                return float(used_bytes) / (1024.0**3)
            except Exception:
                return None
        return None

    def _sample_gpu(self) -> Optional[Dict[str, Any]]:
        if not self._has_nvidia_smi:
            return None
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=2).strip()
        except Exception as exc:
            self._notes.append(f"nvidia-smi failed: {exc}")
            self._has_nvidia_smi = False
            return None

        if not out:
            return None

        utils: List[float] = []
        mems: List[float] = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                utils.append(float(parts[0]))
                mems.append(float(parts[1]))
            except ValueError:
                continue

        if not utils and not mems:
            return None

        return {
            "gpu_util_percent": _mean(utils) if utils else None,
            "gpu_mem_used_mb": float(sum(mems)) if mems else None,
        }

    def _read_proc_cpu_times(self) -> Tuple[int, int]:
        text = Path("/proc/stat").read_text(encoding="utf-8")
        first = text.splitlines()[0].split()
        if not first or first[0] != "cpu":
            raise ValueError("unexpected /proc/stat format")
        nums = [int(v) for v in first[1:]]
        if len(nums) < 4:
            raise ValueError("unexpected /proc/stat cpu fields")
        idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
        total = int(sum(nums))
        return total, idle

    def _read_proc_mem_used_bytes(self) -> int:
        mem_total_kb = None
        mem_available_kb = None
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                mem_total_kb = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                mem_available_kb = int(line.split()[1])
            if mem_total_kb is not None and mem_available_kb is not None:
                break
        if mem_total_kb is None or mem_available_kb is None:
            raise ValueError("missing MemTotal/MemAvailable in /proc/meminfo")
        used_kb = mem_total_kb - mem_available_kb
        return int(used_kb * 1024)


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
    run_metrics: bool = True
    cleanup_after_run: bool = False
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
    run_dir.mkdir(parents=True, exist_ok=True)

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

    metrics_cmd = [
        sys.executable,
        str(CODES_DIR / "compute_metrics.py"),
        "--run-dir",
        str(run_dir),
    ]

    cleanup_cmd = [
        sys.executable,
        str(CODES_DIR / "cleanup_run.py"),
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
        if config.run_metrics:
            print("  metrics:", " ".join(metrics_cmd))
        if config.cleanup_after_run:
            print("  cleanup:", " ".join(cleanup_cmd))
        return run_name, "dry_run"

    monitor = ResourceMonitor(interval_s=1.0)
    monitor.start()
    stage_times: Dict[str, float] = {}
    started_at = time.monotonic()
    status = "ok"

    try:
        t0 = time.monotonic()
        code = run_command(master_cmd, cwd=ROOT_DIR)
        stage_times["master_sec"] = time.monotonic() - t0
        if code != 0:
            status = "failed_master"
            return run_name, status

        if config.run_baseline:
            t0 = time.monotonic()
            code = run_command(baseline_cmd, cwd=ROOT_DIR)
            stage_times["baseline_sec"] = time.monotonic() - t0
            if code != 0:
                status = "failed_baseline"
                return run_name, status

        if config.run_plots:
            t0 = time.monotonic()
            code = run_command(plot_cmd, cwd=ROOT_DIR)
            stage_times["plot_sec"] = time.monotonic() - t0
            if code != 0:
                status = "failed_plot"
                return run_name, status

        monitor.stop()
        usage: Dict[str, Any] = monitor.summary()
        usage["stage_wall_time_sec"] = stage_times
        usage["wall_time_sec"] = time.monotonic() - started_at
        (run_dir / "resource_usage.json").write_text(json.dumps(usage, ensure_ascii=False, indent=2), encoding="utf-8")

        parts = [f"wall={usage.get('wall_time_sec'):.1f}s" if usage.get("wall_time_sec") is not None else "wall=?"]
        if usage.get("cpu_percent_avg") is not None:
            parts.append(f"cpu_avg={usage['cpu_percent_avg']:.1f}%")
        if usage.get("gpu_util_percent_avg") is not None:
            parts.append(f"gpu_avg={usage['gpu_util_percent_avg']:.1f}%")
        if stage_times.get("master_sec") is not None:
            parts.append(f"master={stage_times['master_sec']:.1f}s")
        print(f"[{config.name}] {run_name} resource: " + " | ".join(parts))

        if config.run_metrics:
            t0 = time.monotonic()
            code = run_command(metrics_cmd, cwd=ROOT_DIR)
            stage_times["metrics_sec"] = time.monotonic() - t0
            if code != 0:
                status = "failed_metrics"
                return run_name, status

        if config.cleanup_after_run:
            t0 = time.monotonic()
            code = run_command(cleanup_cmd, cwd=ROOT_DIR)
            stage_times["cleanup_sec"] = time.monotonic() - t0
            if code != 0:
                status = "failed_cleanup"
                return run_name, status

        return run_name, status
    finally:
        monitor.stop()


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

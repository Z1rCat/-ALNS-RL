from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime
import json
import math
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


THIS_DIR = Path(__file__).resolve().parent
CODES_DIR = THIS_DIR.parent
SERVER_OUTPUT_ROOT = CODES_DIR / "nexus"
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from experiments.run_experiments_common import (  # noqa: E402
    ExperimentConfig,
    NotificationManager,
    TaskPlan,
    build_execution_plan,
    build_tasks,
    resolve_max_workers,
    run_task,
)


DEFAULT_DISTS = [
    "O_10_90",
    "O_90_10",
    "O_30_80",
    "O_60_20",
    "O_10_120",
    "O_120_10",
    "G_10_90_50",
    "G_10_40_90",
    "G_40_80_10",
    "G_30_60_90",
    "F1_10_90",
    "F1_90_10",
    "F2_10_90",
    "F2_30_80",
    "R_10_90",
    "R_30_80",
]


@dataclass(frozen=True)
class VariantSpec:
    raw: str
    algorithm: str
    algo_version: str
    ppo_new_window: Optional[int]


@dataclass
class ScheduledJob:
    plan: TaskPlan
    config: ExperimentConfig
    variant: VariantSpec
    dist_name: str
    request_number: int
    seed: int
    algorithm_key: str

    @property
    def job_key(self) -> str:
        return (
            f"{self.variant.raw}|{self.dist_name}|R{self.request_number}|S{self.seed}"
            f"|{self.plan.run_name}"
        )


@dataclass
class RunningDispatch:
    slot_id: int
    predicted_seconds: float
    started_at: float
    attempt: int
    job: ScheduledJob
    timeout_limit_s: float = 0.0
    kill_sent: bool = False
    kill_detail: str = ""


@dataclass
class TaskResult:
    run_name: str
    status: str
    elapsed_seconds: float
    error: str = ""


def _dedupe_keep_order(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in values:
        key = str(item).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _parse_reason_tokens(raw: str) -> List[str]:
    return [str(x).strip().lower() for x in str(raw or "").split(",") if str(x).strip()]


def _norm_algo_version(version: str) -> str:
    value = str(version or "v1").strip().lower()
    aliases = {
        "v31": "v3.1",
        "v3_1": "v3.1",
        "v32": "v3.2",
        "v3_2": "v3.2",
        "v41": "v4.1",
        "v4_1": "v4.1",
        "v61_cvarppo": "v6.1_cvarppo",
        "v6_1_cvarppo": "v6.1_cvarppo",
        "v62_v3cvar": "v6.2_v3cvar",
        "v6_2_v3cvar": "v6.2_v3cvar",
        "v63_cadm": "v6.3_cadm",
        "v6_3_cadm": "v6.3_cadm",
        "v71_poolppo": "v7.1_poolppo",
        "v7_1_poolppo": "v7.1_poolppo",
        "v72_poolv3": "v7.2_poolv3",
        "v7_2_poolv3": "v7.2_poolv3",
    }
    return aliases.get(value, value)


def _default_window_for_version(version: str) -> int:
    v = _norm_algo_version(version)
    if v == "v3.1":
        return 8
    if v in (
        "v2",
        "v3",
        "v3.2",
        "v4",
        "v4.1",
        "v5.1_abppo",
        "v5.2_qcritic",
        "v5.3_auxweak",
        "v6.2_v3cvar",
        "v7.2_poolv3",
    ):
        return 4
    return 1


def _parse_variant(spec: str, n_stack_override: Optional[int]) -> VariantSpec:
    raw = str(spec or "").strip()
    if not raw:
        raise ValueError("empty variant spec")
    if ":" in raw:
        algo_part, version_part = raw.split(":", 1)
    elif "@" in raw:
        algo_part, version_part = raw.split("@", 1)
    else:
        algo_part, version_part = raw, "v1"
    algorithm = str(algo_part or "").strip().upper()
    if not algorithm:
        raise ValueError(f"invalid variant spec: {raw}")
    algo_version = _norm_algo_version(version_part or "v1")
    ppo_window: Optional[int] = None
    if algorithm == "PPO_NEW":
        if n_stack_override is not None:
            ppo_window = max(1, int(n_stack_override))
        elif algo_version == "v1":
            ppo_window = 1
        else:
            ppo_window = _default_window_for_version(algo_version)
    return VariantSpec(
        raw=raw,
        algorithm=algorithm,
        algo_version=algo_version,
        ppo_new_window=ppo_window,
    )


def _parse_variants(args: argparse.Namespace) -> List[VariantSpec]:
    specs = [str(v).strip() for v in (args.variant or []) if str(v).strip()]
    if not specs:
        specs = [f"{str(args.algo).strip()}:{str(args.algo_version).strip()}"]
    out: List[VariantSpec] = []
    seen = set()
    for spec in specs:
        item = _parse_variant(spec=spec, n_stack_override=args.n_stack)
        key = (item.algorithm, item.algo_version, item.ppo_new_window)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _resolve_target_run_root(run_folder: str) -> Path:
    raw = str(run_folder or "").strip()
    if not raw:
        raise ValueError("--run-folder is required")
    p = Path(raw)
    if p.is_absolute():
        return p.resolve()
    return (SERVER_OUTPUT_ROOT / p).resolve()


def _parse_kv_float_map(raw: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    text = str(raw or "").strip()
    if not text:
        return out
    for part in [x.strip() for x in text.split(",") if x.strip()]:
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        key = str(k).strip().upper()
        if not key:
            continue
        try:
            val = float(str(v).strip())
        except Exception:
            continue
        if val > 0:
            out[key] = float(val)
    return out


def _append_csv_row(path: Path, fieldnames: List[str], row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception:
        return {}
    if isinstance(data, dict):
        return data
    return {}


def _save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _try_import_psutil():
    try:
        import psutil  # type: ignore
        return psutil
    except Exception:
        return None


def _detect_cpu_counts() -> Tuple[int, int]:
    logical = max(1, int(os.cpu_count() or 1))
    physical = logical
    psutil = _try_import_psutil()
    if psutil is not None:
        try:
            logical = max(1, int(psutil.cpu_count(logical=True) or logical))
        except Exception:
            pass
        try:
            physical = int(psutil.cpu_count(logical=False) or 0)
            if physical <= 0:
                physical = max(1, logical // 2)
        except Exception:
            physical = max(1, logical // 2)
    return logical, max(1, physical)


def _sample_system_pressure() -> Dict[str, float]:
    psutil = _try_import_psutil()
    cpu_percent = float("nan")
    mem_percent = float("nan")
    swap_percent = float("nan")
    avail_gb = float("nan")
    load_per_core = float("nan")
    logical, _ = _detect_cpu_counts()

    if psutil is not None:
        try:
            cpu_percent = float(psutil.cpu_percent(interval=None))
        except Exception:
            pass
        try:
            vm = psutil.virtual_memory()
            mem_percent = float(vm.percent)
            avail_gb = float(vm.available) / (1024.0**3)
        except Exception:
            pass
        try:
            swap_percent = float(psutil.swap_memory().percent)
        except Exception:
            pass

    if hasattr(os, "getloadavg"):
        try:
            load1, _, _ = os.getloadavg()
            load_per_core = float(load1) / float(max(1, logical))
        except Exception:
            pass

    return {
        "cpu_percent": float(cpu_percent),
        "mem_percent": float(mem_percent),
        "swap_percent": float(swap_percent),
        "avail_gb": float(avail_gb),
        "load_per_core": float(load_per_core),
        "logical_cores": float(logical),
    }


def _is_finite(x: float) -> bool:
    return (x == x) and math.isfinite(x)


def _calc_initial_active_limit(
    *,
    worker_cap: int,
    total_jobs: int,
    min_workers: int,
    per_task_mem_gb: float,
) -> Tuple[int, Dict[str, float]]:
    logical, physical = _detect_cpu_counts()
    pressure = _sample_system_pressure()
    avail_gb = float(pressure.get("avail_gb", float("nan")))
    cpu_now = float(pressure.get("cpu_percent", float("nan")))

    if physical >= 96:
        core_factor = 0.50
    elif physical >= 64:
        core_factor = 0.60
    elif physical >= 24:
        core_factor = 0.75
    else:
        core_factor = 0.90
    by_core = max(1, int(math.floor(float(physical) * float(core_factor))))

    if _is_finite(avail_gb) and avail_gb > 0 and per_task_mem_gb > 0:
        by_mem = max(1, int(math.floor(float(avail_gb) / float(per_task_mem_gb))))
    else:
        by_mem = worker_cap

    # Aggressive-by-default startup:
    # In auto mode we prefer to saturate available slots first, and let
    # runtime watchdog/reschedule/autoscale backoff handle overload events.
    initial = min(int(worker_cap), int(total_jobs))
    initial = max(int(min_workers), int(initial))
    initial = min(int(worker_cap), int(initial))
    return int(initial), {
        "logical_cores": float(logical),
        "physical_cores": float(physical),
        "core_factor": float(core_factor),
        "by_core": float(by_core),
        "by_mem": float(by_mem),
        "startup_mode": "aggressive_full_cap",
        "cpu_now": float(cpu_now),
        "avail_gb": float(avail_gb),
    }


def _read_failed_reason(run_dir: Path) -> str:
    failed = run_dir / "FAILED.json"
    if not failed.exists():
        return ""
    data = _load_json(failed)
    reason = str(data.get("reason", "")).strip().lower()
    stage = str(data.get("stage", "")).strip().lower()
    code = str(data.get("exit_code", "")).strip().lower()
    text = f"{reason}|{stage}|{code}"
    return text


def _read_watchdog_tail_reason(run_dir: Path, max_lines: int = 20) -> str:
    path = run_dir / "watchdog_events.jsonl"
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return ""
    tail = lines[-max(1, int(max_lines)) :]
    for raw in reversed(tail):
        try:
            data = json.loads(raw)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        event = str(data.get("event", "")).strip().lower()
        if event in {"stall_timeout", "wall_timeout", "attempt_failed", "lock_busy_skip"}:
            reason = str(data.get("reason", "")).strip().lower()
            stage = str(data.get("stage", "")).strip().lower()
            return f"{reason}|{stage}|watchdog"
    return ""


def _compose_failure_reason(run_dir: Path, status: str, error: str) -> str:
    parts = []
    failed_reason = _read_failed_reason(run_dir)
    if failed_reason:
        parts.append(failed_reason)
    wd_reason = _read_watchdog_tail_reason(run_dir)
    if wd_reason:
        parts.append(wd_reason)
    status_text = str(status or "").strip().lower()
    if status_text:
        parts.append(status_text)
    err = str(error or "").strip().lower()
    if err:
        parts.append(err[:400])
    return "|".join([p for p in parts if p])


def _calc_dispatch_timeout_seconds(predicted_seconds: float, args: argparse.Namespace) -> float:
    pred = max(1.0, float(predicted_seconds))
    factor = max(1.0, float(args.dispatch_timeout_factor))
    lower = max(30.0, float(args.dispatch_timeout_min_sec))
    upper = max(lower, float(args.dispatch_timeout_max_sec))
    return min(upper, max(lower, pred * factor))


def _kill_run_process_tree(run_dir: Path) -> Tuple[bool, str]:
    status_path = run_dir / "run_status.json"
    data = _load_json(status_path)
    if not data:
        return False, "missing_run_status"
    stage_status = str(data.get("status", "")).strip().lower()
    if stage_status not in {"running", "killed"}:
        return False, f"status_not_running:{stage_status}"
    pid_raw = data.get("pid", 0)
    try:
        pid = int(pid_raw)
    except Exception:
        return False, f"invalid_pid:{pid_raw}"
    if pid <= 0:
        return False, f"invalid_pid:{pid}"
    try:
        if os.name == "nt":
            cmd = ["taskkill", "/PID", str(pid), "/T", "/F"]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            ok = int(proc.returncode) == 0
            msg = (proc.stdout or proc.stderr or "").strip()[:300]
            return ok, f"taskkill_rc={proc.returncode}|{msg}"
        os.kill(pid, 15)
        return True, "sigterm_sent"
    except Exception as exc:
        return False, f"kill_exception:{exc}"


class AdaptiveDurationModel:
    def __init__(
        self,
        *,
        algo_init: Dict[str, float],
        dist_init: Dict[str, float],
        lr: float,
        global_lr: float,
        min_pred_seconds: float,
    ) -> None:
        self.algo_coef: Dict[str, float] = {str(k).upper(): float(v) for k, v in algo_init.items() if float(v) > 0}
        self.dist_coef: Dict[str, float] = {str(k).upper(): float(v) for k, v in dist_init.items() if float(v) > 0}
        self.algo_count: Dict[str, int] = {str(k).upper(): 0 for k in self.algo_coef.keys()}
        self.dist_count: Dict[str, int] = {str(k).upper(): 0 for k in self.dist_coef.keys()}
        self.global_scale: float = 1.0
        self.lr = max(0.0, float(lr))
        self.global_lr = max(0.0, float(global_lr))
        self.min_pred_seconds = max(1.0, float(min_pred_seconds))

    def ensure_key(self, algo_key: str, dist_name: str) -> None:
        a = str(algo_key).upper()
        d = str(dist_name).upper()
        self.algo_coef.setdefault(a, 1.0)
        self.dist_coef.setdefault(d, 1.0)
        self.algo_count.setdefault(a, 0)
        self.dist_count.setdefault(d, 0)

    def predict(self, *, algo_key: str, dist_name: str, request_number: int) -> float:
        self.ensure_key(algo_key=algo_key, dist_name=dist_name)
        a = self.algo_coef[str(algo_key).upper()]
        d = self.dist_coef[str(dist_name).upper()]
        req_scale = max(1.0, float(request_number)) / 30.0
        pred = float(self.global_scale) * float(a) * float(d) * float(req_scale)
        return max(self.min_pred_seconds, float(pred))

    def update(
        self,
        *,
        algo_key: str,
        dist_name: str,
        predicted_seconds: float,
        actual_seconds: float,
    ) -> Dict[str, float]:
        self.ensure_key(algo_key=algo_key, dist_name=dist_name)
        if actual_seconds <= 0 or predicted_seconds <= 0:
            return {
                "ratio": float("nan"),
                "algo_coef": float(self.algo_coef[str(algo_key).upper()]),
                "dist_coef": float(self.dist_coef[str(dist_name).upper()]),
                "global_scale": float(self.global_scale),
            }

        ratio = max(0.05, min(20.0, float(actual_seconds) / float(predicted_seconds)))
        akey = str(algo_key).upper()
        dkey = str(dist_name).upper()
        self.algo_count[akey] = int(self.algo_count.get(akey, 0)) + 1
        self.dist_count[dkey] = int(self.dist_count.get(dkey, 0)) + 1

        a_lr = float(self.lr) / math.sqrt(float(self.algo_count[akey]))
        d_lr = float(self.lr) / math.sqrt(float(self.dist_count[dkey]))
        g_lr = float(self.global_lr)

        self.algo_coef[akey] = float(self.algo_coef[akey]) * (float(ratio) ** float(a_lr))
        self.dist_coef[dkey] = float(self.dist_coef[dkey]) * (float(ratio) ** float(d_lr))
        self.global_scale = float(self.global_scale) * (float(ratio) ** float(g_lr))

        self._renormalize()
        return {
            "ratio": float(ratio),
            "algo_coef": float(self.algo_coef[akey]),
            "dist_coef": float(self.dist_coef[dkey]),
            "global_scale": float(self.global_scale),
        }

    def _renormalize(self) -> None:
        def _geo_mean(vals: List[float]) -> float:
            clean = [float(v) for v in vals if float(v) > 0]
            if not clean:
                return 1.0
            return math.exp(sum(math.log(v) for v in clean) / float(len(clean)))

        ga = _geo_mean(list(self.algo_coef.values()))
        gd = _geo_mean(list(self.dist_coef.values()))
        if ga > 0:
            for k in list(self.algo_coef.keys()):
                self.algo_coef[k] = float(self.algo_coef[k]) / float(ga)
            self.global_scale *= float(ga)
        if gd > 0:
            for k in list(self.dist_coef.keys()):
                self.dist_coef[k] = float(self.dist_coef[k]) / float(gd)
            self.global_scale *= float(gd)
        self.global_scale = max(1e-3, min(1e5, float(self.global_scale)))

    def to_dict(self) -> Dict[str, object]:
        return {
            "algo_coef": {k: float(v) for k, v in sorted(self.algo_coef.items())},
            "dist_coef": {k: float(v) for k, v in sorted(self.dist_coef.items())},
            "algo_count": {k: int(v) for k, v in sorted(self.algo_count.items())},
            "dist_count": {k: int(v) for k, v in sorted(self.dist_count.items())},
            "global_scale": float(self.global_scale),
            "lr": float(self.lr),
            "global_lr": float(self.global_lr),
            "min_pred_seconds": float(self.min_pred_seconds),
            "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, object],
        *,
        lr: float,
        global_lr: float,
        min_pred_seconds: float,
        algo_init: Dict[str, float],
        dist_init: Dict[str, float],
    ) -> "AdaptiveDurationModel":
        obj = cls(
            algo_init=algo_init,
            dist_init=dist_init,
            lr=lr,
            global_lr=global_lr,
            min_pred_seconds=min_pred_seconds,
        )
        for key, val in (data.get("algo_coef", {}) or {}).items():
            try:
                fv = float(val)
            except Exception:
                continue
            if fv > 0:
                obj.algo_coef[str(key).upper()] = fv
        for key, val in (data.get("dist_coef", {}) or {}).items():
            try:
                fv = float(val)
            except Exception:
                continue
            if fv > 0:
                obj.dist_coef[str(key).upper()] = fv
        for key, val in (data.get("algo_count", {}) or {}).items():
            obj.algo_count[str(key).upper()] = max(0, int(val))
        for key, val in (data.get("dist_count", {}) or {}).items():
            obj.dist_count[str(key).upper()] = max(0, int(val))
        try:
            gs = float(data.get("global_scale", 1.0))
            if gs > 0:
                obj.global_scale = gs
        except Exception:
            pass
        obj._renormalize()
        return obj


def _run_precheck(
    *,
    run_root: Path,
    algorithms: List[str],
    dists: List[str],
    args: argparse.Namespace,
) -> int:
    has_existing_runs = any(run_root.glob("run_*"))
    if not bool(args.precheck):
        return 0
    if not has_existing_runs:
        print(f"[adaptive] precheck skipped (no existing run_* under {run_root})")
        return 0
    cmd = [
        sys.executable,
        str(CODES_DIR / "tools" / "rerun_incomplete.py"),
        "--logs-root",
        str(run_root),
        "--no-clean",
    ]
    for algo in sorted(set(algorithms)):
        cmd.extend(["--algorithm", str(algo)])
    for dist_name in sorted(set(dists)):
        cmd.extend(["--dist-name", str(dist_name)])
    if int(args.precheck_workers or 0) > 0:
        cmd.extend(["--workers", str(int(args.precheck_workers))])
    if bool(args.dry_run):
        cmd.append("--dry-run")
    return subprocess.run(cmd, cwd=str(CODES_DIR)).returncode


def _execute_one_job(
    job: ScheduledJob,
    *,
    dry_run: bool,
    notifier: Optional[NotificationManager],
) -> TaskResult:
    t0 = time.monotonic()
    try:
        run_name, status = run_task(job.plan, job.config, bool(dry_run), notifier)
        elapsed = time.monotonic() - t0
        return TaskResult(run_name=str(run_name), status=str(status), elapsed_seconds=float(elapsed))
    except Exception:
        elapsed = time.monotonic() - t0
        return TaskResult(
            run_name=str(job.plan.run_name),
            status="failed_internal_exception",
            elapsed_seconds=float(elapsed),
            error=traceback.format_exc(),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adaptive parallel scheduler over (variant, dist, R, seed). "
            "Reuses run_experiments_common watchdog/resume/lock, and adds online duration-model dispatch."
        )
    )
    parser.add_argument("--run-folder", type=str, required=True, help="target folder under codes/nexus or absolute path")
    parser.add_argument("--variant", action="append", default=None, help="repeatable, e.g. PPO_NEW:v3.1, NOVA_EDRL:v1")
    parser.add_argument("--algo", type=str, default="PPO_NEW", help="fallback algorithm when --variant not set")
    parser.add_argument("--algo-version", type=str, default="v3", help="fallback version when --variant not set")
    parser.add_argument("--n-stack", type=int, default=None, help="optional global n_stack override for PPO_NEW")
    parser.add_argument("--dist-name", action="append", default=None, help="distribution name (repeatable)")
    parser.add_argument("--request-number", type=int, action="append", default=None, help="request number R (repeatable)")
    parser.add_argument("--seed", type=int, action="append", default=None, help="seed (repeatable)")
    parser.add_argument("--max-workers", type=int, default=None, help="parallel worker slots")
    parser.add_argument("--generator-workers", type=int, default=1, help="generator workers per task")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stage-mode", type=str, default="train_eval", help="train_eval/train_only/eval_only")
    parser.add_argument("--init-model-path", type=str, default="", help="optional checkpoint to load")
    parser.add_argument("--save-model-path", type=str, default="", help="optional checkpoint to save")

    parser.add_argument("--run-baseline", action="store_true", default=True)
    parser.add_argument("--no-run-baseline", action="store_false", dest="run_baseline")
    parser.add_argument("--run-plots", action="store_true", default=True)
    parser.add_argument("--no-run-plots", action="store_false", dest="run_plots")
    parser.add_argument("--run-metrics", action="store_true", default=True)
    parser.add_argument("--no-run-metrics", action="store_false", dest="run_metrics")
    parser.add_argument("--cleanup-after-run", action="store_true", default=True)

    parser.add_argument("--resume-existing", action="store_true", default=True)
    parser.add_argument("--no-resume-existing", action="store_false", dest="resume_existing")
    parser.add_argument("--skip-completed", action="store_true", default=True)
    parser.add_argument("--no-skip-completed", action="store_false", dest="skip_completed")

    parser.add_argument("--precheck", action="store_true", default=True)
    parser.add_argument("--no-precheck", action="store_false", dest="precheck")
    parser.add_argument("--precheck-workers", type=int, default=0)
    parser.add_argument("--notify-success", action="store_true", default=False)
    parser.add_argument("--no-notify-failure", action="store_false", dest="notify_failure", default=True)

    parser.add_argument("--scheduler-policy", type=str, default="adaptive_lpt", choices=["adaptive_lpt", "fifo"])
    parser.add_argument("--algo-coef-init", type=str, default="", help="e.g. PPO=1.0,PPO_NEW=1.2,NOVA_EDRL=2.0")
    parser.add_argument("--dist-coef-init", type=str, default="", help="e.g. O_10_90=1.3,F2_10_60=1.1")
    parser.add_argument("--model-lr", type=float, default=0.35)
    parser.add_argument("--model-global-lr", type=float, default=0.10)
    parser.add_argument("--model-min-pred-sec", type=float, default=60.0)
    parser.add_argument("--coef-state-path", type=str, default="")
    parser.add_argument("--reset-coef-state", action="store_true")
    parser.add_argument("--events-csv", type=str, default="")
    parser.add_argument("--summary-json", type=str, default="")
    parser.add_argument("--auto-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-workers", type=int, default=1)
    parser.add_argument("--per-task-mem-gb", type=float, default=2.0)
    parser.add_argument("--adjust-interval-sec", type=float, default=15.0)
    parser.add_argument("--adjust-cooldown-sec", type=float, default=45.0)
    parser.add_argument("--up-step", type=int, default=1)
    parser.add_argument("--down-step", type=int, default=1)
    parser.add_argument("--high-cpu", type=float, default=92.0)
    parser.add_argument("--low-cpu", type=float, default=65.0)
    parser.add_argument("--high-mem", type=float, default=92.0)
    parser.add_argument("--low-mem", type=float, default=80.0)
    parser.add_argument("--high-swap", type=float, default=50.0)
    parser.add_argument("--low-swap", type=float, default=15.0)
    parser.add_argument("--high-load-per-core", type=float, default=1.20)
    parser.add_argument("--low-load-per-core", type=float, default=0.75)
    parser.add_argument("--high-streak", type=int, default=2)
    parser.add_argument("--low-streak", type=int, default=3)
    parser.add_argument("--timeout-downstep", type=int, default=2)
    parser.add_argument(
        "--reschedule-timeout-jobs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="requeue timeout/stall failed jobs back to pending pool",
    )
    parser.add_argument(
        "--reschedule-max-attempts",
        type=int,
        default=3,
        help="max dispatch attempts per job when timeout/stall happens",
    )
    parser.add_argument(
        "--reschedule-reasons",
        type=str,
        default="timeout,stall,|124",
        help="comma-separated substrings to trigger reschedule",
    )
    parser.add_argument(
        "--reschedule-unknown-once",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="allow one retry for failed jobs with unknown failure reason",
    )
    parser.add_argument(
        "--reschedule-on-locked",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="requeue skipped_locked jobs instead of counting as completed",
    )
    parser.add_argument("--requeue-delay-base-sec", type=float, default=20.0)
    parser.add_argument("--requeue-delay-max-sec", type=float, default=300.0)
    parser.add_argument(
        "--dispatch-kill-on-timeout",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="kill run subprocess tree when dispatch elapsed exceeds adaptive timeout",
    )
    parser.add_argument("--dispatch-timeout-factor", type=float, default=8.0)
    parser.add_argument("--dispatch-timeout-min-sec", type=float, default=1800.0)
    parser.add_argument("--dispatch-timeout-max-sec", type=float, default=21600.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = _resolve_target_run_root(args.run_folder)
    run_root.mkdir(parents=True, exist_ok=True)

    variants = _parse_variants(args)
    dists = _dedupe_keep_order([str(x).strip() for x in (args.dist_name or DEFAULT_DISTS) if str(x).strip()])
    requests = [int(x) for x in (args.request_number or [30])]
    seeds = [int(x) for x in (args.seed or [42])]
    if not dists:
        raise ValueError("empty dist list")

    precheck_rc = _run_precheck(
        run_root=run_root,
        algorithms=[v.algorithm for v in variants],
        dists=dists,
        args=args,
    )
    if precheck_rc != 0:
        print(f"[adaptive] precheck failed (exit={precheck_rc})")
        return 1

    all_jobs: List[ScheduledJob] = []
    skipped_completed_total = 0
    for variant in variants:
        cfg = ExperimentConfig(
            name=f"adaptive_{variant.algorithm}_{variant.algo_version}",
            distributions=list(dists),
            request_numbers=[int(x) for x in requests],
            algorithms=[variant.algorithm],
            seeds=[int(x) for x in seeds],
            generator_workers=max(1, int(args.generator_workers)),
            max_workers=args.max_workers,
            run_baseline=bool(args.run_baseline),
            baseline_include_random=True,
            run_plots=bool(args.run_plots),
            run_metrics=bool(args.run_metrics),
            cleanup_after_run=bool(args.cleanup_after_run),
            resume_existing=bool(args.resume_existing),
            skip_completed=bool(args.skip_completed),
            notify_on_failure=bool(args.notify_failure),
            notify_on_success=bool(args.notify_success),
            log_subdir=str(run_root),
            algo_version=str(variant.algo_version),
            ppo_new_window=variant.ppo_new_window,
            stage_mode=str(args.stage_mode),
            init_model_path=str(args.init_model_path).strip() or None,
            save_model_path=str(args.save_model_path).strip() or None,
        )
        tasks = build_tasks(cfg)
        plans, skipped = build_execution_plan(cfg, tasks, run_root)
        skipped_completed_total += int(skipped)
        for plan in plans:
            all_jobs.append(
                ScheduledJob(
                    plan=plan,
                    config=cfg,
                    variant=variant,
                    dist_name=str(plan.dist_name),
                    request_number=int(plan.request_number),
                    seed=int(plan.seed),
                    algorithm_key=str(variant.algorithm).upper(),
                )
            )

    if not all_jobs:
        print(f"[adaptive] all tasks already completed. skipped_completed={skipped_completed_total}")
        return 0

    pending_by_dist: Dict[str, List[ScheduledJob]] = {str(d): [] for d in dists}
    for job in all_jobs:
        pending_by_dist.setdefault(str(job.dist_name), []).append(job)
    for key in list(pending_by_dist.keys()):
        pending_by_dist[key].sort(key=lambda j: (str(j.algorithm_key), int(j.seed), int(j.request_number), str(j.plan.run_name)))
    deferred_jobs: List[Tuple[float, ScheduledJob]] = []

    reschedule_reason_tokens = _parse_reason_tokens(args.reschedule_reasons)
    reschedule_max_attempts = max(1, int(args.reschedule_max_attempts))

    algo_init = _parse_kv_float_map(args.algo_coef_init)
    dist_init = _parse_kv_float_map(args.dist_coef_init)
    state_path = (
        Path(str(args.coef_state_path)).resolve()
        if str(args.coef_state_path).strip()
        else (run_root / "adaptive_scheduler_coef_state.json").resolve()
    )
    if bool(args.reset_coef_state) or (not state_path.exists()):
        model = AdaptiveDurationModel(
            algo_init=algo_init,
            dist_init=dist_init,
            lr=float(args.model_lr),
            global_lr=float(args.model_global_lr),
            min_pred_seconds=float(args.model_min_pred_sec),
        )
    else:
        model = AdaptiveDurationModel.from_dict(
            _load_json(state_path),
            lr=float(args.model_lr),
            global_lr=float(args.model_global_lr),
            min_pred_seconds=float(args.model_min_pred_sec),
            algo_init=algo_init,
            dist_init=dist_init,
        )

    for job in all_jobs:
        model.ensure_key(algo_key=job.algorithm_key, dist_name=job.dist_name)

    worker_cap = min(resolve_max_workers(all_jobs[0].config, args.max_workers), len(all_jobs))
    min_workers = max(1, min(int(args.min_workers), int(worker_cap)))
    if bool(args.auto_workers):
        active_limit, init_diag = _calc_initial_active_limit(
            worker_cap=int(worker_cap),
            total_jobs=len(all_jobs),
            min_workers=int(min_workers),
            per_task_mem_gb=max(0.1, float(args.per_task_mem_gb)),
        )
    else:
        active_limit = int(worker_cap)
        init_diag = {}

    notifier = NotificationManager(run_root=run_root)
    events_csv = (
        Path(str(args.events_csv)).resolve()
        if str(args.events_csv).strip()
        else (run_root / "adaptive_scheduler_events.csv").resolve()
    )
    summary_json = (
        Path(str(args.summary_json)).resolve()
        if str(args.summary_json).strip()
        else (run_root / "adaptive_scheduler_summary.json").resolve()
    )

    print(f"[adaptive] run_root={run_root}")
    print(f"[adaptive] variants={[v.raw for v in variants]}")
    print(f"[adaptive] distributions={dists}")
    print(f"[adaptive] requests={requests} seeds={seeds}")
    print(
        f"[adaptive] jobs={len(all_jobs)} skipped_completed={skipped_completed_total} "
        f"worker_cap={worker_cap} active_init={active_limit} min_workers={min_workers} "
        f"auto_workers={int(bool(args.auto_workers))}"
    )
    print(
        f"[adaptive] stages: baseline={int(bool(args.run_baseline))} "
        f"plots={int(bool(args.run_plots))} metrics={int(bool(args.run_metrics))} "
        f"cleanup={int(bool(args.cleanup_after_run))} "
        f"baseline_random={int(True)}"
    )
    print(
        f"[adaptive] reschedule_timeout_jobs={int(bool(args.reschedule_timeout_jobs))} "
        f"max_attempts={int(reschedule_max_attempts)} reason_tokens={reschedule_reason_tokens}"
    )
    print(
        f"[adaptive] requeue_delay base={float(args.requeue_delay_base_sec):.1f}s "
        f"max={float(args.requeue_delay_max_sec):.1f}s "
        f"unknown_once={int(bool(args.reschedule_unknown_once))} "
        f"on_locked={int(bool(args.reschedule_on_locked))}"
    )
    print(
        f"[adaptive] dispatch_timeout kill={int(bool(args.dispatch_kill_on_timeout))} "
        f"factor={float(args.dispatch_timeout_factor):.2f} "
        f"min={float(args.dispatch_timeout_min_sec):.1f}s "
        f"max={float(args.dispatch_timeout_max_sec):.1f}s"
    )
    if init_diag:
        print(
            "[adaptive] auto_init "
            f"physical={init_diag.get('physical_cores')} logical={init_diag.get('logical_cores')} "
            f"core_factor={init_diag.get('core_factor')} by_core={init_diag.get('by_core')} "
            f"by_mem={init_diag.get('by_mem')} cpu_now={init_diag.get('cpu_now')} "
            f"avail_gb={init_diag.get('avail_gb')}"
        )

    event_fields = [
        "ts", "event", "slot_id", "dist_name", "algorithm", "variant",
        "request_number", "seed", "run_name", "attempt", "predicted_seconds", "elapsed_seconds",
        "status", "ratio", "algo_coef", "dist_coef", "global_scale",
        "active_limit", "cpu_percent", "mem_percent", "swap_percent", "load_per_core",
        "remaining_jobs", "timeout_like", "requeue_reason", "dispatch_timeout_s", "error",
    ]

    slot_pred_loads = [0.0 for _ in range(worker_cap)]
    slot_actual_loads = [0.0 for _ in range(worker_cap)]
    free_slots = list(range(worker_cap))
    failures: Dict[str, str] = {}
    status_counter: Dict[str, int] = {}
    completed_attempts = 0
    completed_jobs_final = 0
    job_attempts: Dict[str, int] = {}
    run_started = time.monotonic()
    pressure_last = _sample_system_pressure()
    high_streak = 0
    low_streak = 0
    last_adjust_check = 0.0
    last_adjust_event = 0.0

    def _remaining_jobs_count() -> int:
        return sum(len(v) for v in pending_by_dist.values()) + len(deferred_jobs)

    def _release_deferred_jobs(now_ts: float) -> int:
        if not deferred_jobs:
            return 0
        ready: List[Tuple[float, ScheduledJob]] = []
        waiting: List[Tuple[float, ScheduledJob]] = []
        for release_ts, job in deferred_jobs:
            if float(release_ts) <= float(now_ts):
                ready.append((release_ts, job))
            else:
                waiting.append((release_ts, job))
        deferred_jobs.clear()
        deferred_jobs.extend(waiting)
        for _, job in sorted(ready, key=lambda x: x[0]):
            pending_by_dist.setdefault(str(job.dist_name), []).append(job)
        return len(ready)

    def _pick_next_dist() -> Optional[str]:
        for d in dists:
            if pending_by_dist.get(d):
                return d
        for d, items in pending_by_dist.items():
            if items:
                return d
        return None

    def _pick_next_job() -> Optional[ScheduledJob]:
        dist = _pick_next_dist()
        if not dist:
            return None
        queue = pending_by_dist.get(dist, [])
        if not queue:
            return None
        if str(args.scheduler_policy).strip().lower() == "fifo":
            return queue.pop(0)
        best_idx = 0
        best_pred = -1.0
        for idx, job in enumerate(queue):
            pred = model.predict(
                algo_key=job.algorithm_key,
                dist_name=job.dist_name,
                request_number=int(job.request_number),
            )
            if pred > best_pred:
                best_pred = pred
                best_idx = idx
        return queue.pop(best_idx)

    def _pressure_is_high(p: Dict[str, float], current_limit: int) -> bool:
        cpu = float(p.get("cpu_percent", float("nan")))
        mem = float(p.get("mem_percent", float("nan")))
        swap = float(p.get("swap_percent", float("nan")))
        load_pc = float(p.get("load_per_core", float("nan")))
        avail_gb = float(p.get("avail_gb", float("nan")))
        high = False
        if _is_finite(cpu) and cpu >= float(args.high_cpu):
            high = True
        if _is_finite(mem) and mem >= float(args.high_mem):
            high = True
        if _is_finite(swap) and swap >= float(args.high_swap):
            high = True
        if _is_finite(load_pc) and load_pc >= float(args.high_load_per_core):
            high = True
        if _is_finite(avail_gb):
            # Keep autoscale aggressive: only treat memory as high-pressure when
            # available memory is critically low, instead of strict per-slot reservation.
            mem_hard_floor = max(0.2, float(args.per_task_mem_gb) * 0.25)
            if avail_gb < mem_hard_floor:
                high = True
        return bool(high)

    def _pressure_is_low(p: Dict[str, float], current_limit: int) -> bool:
        cpu = float(p.get("cpu_percent", float("nan")))
        mem = float(p.get("mem_percent", float("nan")))
        swap = float(p.get("swap_percent", float("nan")))
        load_pc = float(p.get("load_per_core", float("nan")))
        avail_gb = float(p.get("avail_gb", float("nan")))
        low = True
        if _is_finite(cpu) and cpu > float(args.low_cpu):
            low = False
        if _is_finite(mem) and mem > float(args.low_mem):
            low = False
        if _is_finite(swap) and swap > float(args.low_swap):
            low = False
        if _is_finite(load_pc) and load_pc > float(args.low_load_per_core):
            low = False
        if _is_finite(avail_gb):
            mem_hard_floor = max(0.2, float(args.per_task_mem_gb) * 0.25)
            if avail_gb < mem_hard_floor:
                low = False
        return bool(low)

    running: Dict[concurrent.futures.Future, RunningDispatch] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_cap) as pool:
        while _remaining_jobs_count() > 0 or running:
            now = time.monotonic()
            released = _release_deferred_jobs(now)
            if released > 0:
                print(f"[adaptive][requeue] released={released} pending={_remaining_jobs_count()}")
            if bool(args.auto_workers) and (now - last_adjust_check >= max(5.0, float(args.adjust_interval_sec))):
                last_adjust_check = now
                pressure_last = _sample_system_pressure()
                is_high = _pressure_is_high(pressure_last, active_limit)
                is_low = _pressure_is_low(pressure_last, active_limit)
                if is_high:
                    high_streak += 1
                    low_streak = 0
                elif is_low:
                    low_streak += 1
                    high_streak = 0
                else:
                    high_streak = 0
                    low_streak = 0

                if now - last_adjust_event >= max(5.0, float(args.adjust_cooldown_sec)):
                    if high_streak >= max(1, int(args.high_streak)) and active_limit > min_workers:
                        step = max(1, int(args.down_step))
                        prev = int(active_limit)
                        active_limit = max(min_workers, int(active_limit) - step)
                        high_streak = 0
                        low_streak = 0
                        last_adjust_event = now
                        print(
                            f"[adaptive][autoscale] pressure-high: active_limit {prev}->{active_limit} "
                            f"cpu={pressure_last.get('cpu_percent')} mem={pressure_last.get('mem_percent')} "
                            f"load={pressure_last.get('load_per_core')}"
                        )
                    elif (
                        low_streak >= max(1, int(args.low_streak))
                        and active_limit < int(worker_cap)
                        and _remaining_jobs_count() > 0
                    ):
                        step = max(1, int(args.up_step))
                        prev = int(active_limit)
                        active_limit = min(int(worker_cap), int(active_limit) + step)
                        high_streak = 0
                        low_streak = 0
                        last_adjust_event = now
                        print(
                            f"[adaptive][autoscale] pressure-low: active_limit {prev}->{active_limit} "
                            f"cpu={pressure_last.get('cpu_percent')} mem={pressure_last.get('mem_percent')} "
                            f"load={pressure_last.get('load_per_core')}"
                        )

            while free_slots and _remaining_jobs_count() > 0 and len(running) < int(active_limit):
                slot_id = min(free_slots, key=lambda s: slot_pred_loads[s])
                free_slots.remove(slot_id)
                job = _pick_next_job()
                if job is None:
                    free_slots.append(slot_id)
                    break
                attempt_no = int(job_attempts.get(job.job_key, 0) + 1)
                job_attempts[job.job_key] = attempt_no
                pred = model.predict(
                    algo_key=job.algorithm_key,
                    dist_name=job.dist_name,
                    request_number=int(job.request_number),
                )
                slot_pred_loads[slot_id] += float(pred)
                _append_csv_row(
                    events_csv, event_fields,
                    {
                        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                        "event": "dispatch",
                        "slot_id": int(slot_id),
                        "dist_name": str(job.dist_name),
                        "algorithm": str(job.algorithm_key),
                        "variant": str(job.variant.raw),
                        "request_number": int(job.request_number),
                        "seed": int(job.seed),
                        "run_name": str(job.plan.run_name),
                        "attempt": int(attempt_no),
                        "predicted_seconds": float(pred),
                        "elapsed_seconds": "",
                        "status": "",
                        "ratio": "",
                        "algo_coef": float(model.algo_coef.get(str(job.algorithm_key).upper(), 1.0)),
                        "dist_coef": float(model.dist_coef.get(str(job.dist_name).upper(), 1.0)),
                        "global_scale": float(model.global_scale),
                        "active_limit": int(active_limit),
                        "cpu_percent": pressure_last.get("cpu_percent", ""),
                        "mem_percent": pressure_last.get("mem_percent", ""),
                        "swap_percent": pressure_last.get("swap_percent", ""),
                        "load_per_core": pressure_last.get("load_per_core", ""),
                        "remaining_jobs": int(_remaining_jobs_count()),
                        "timeout_like": "",
                        "requeue_reason": "",
                        "dispatch_timeout_s": float(_calc_dispatch_timeout_seconds(pred, args)),
                        "error": "",
                    },
                )
                fut = pool.submit(_execute_one_job, job, dry_run=bool(args.dry_run), notifier=notifier)
                running[fut] = RunningDispatch(
                    slot_id=int(slot_id),
                    predicted_seconds=float(pred),
                    started_at=time.monotonic(),
                    attempt=int(attempt_no),
                    job=job,
                    timeout_limit_s=float(_calc_dispatch_timeout_seconds(pred, args)),
                )

            if bool(args.dispatch_kill_on_timeout):
                for fut, dispatch in list(running.items()):
                    if dispatch.kill_sent:
                        continue
                    elapsed_running = float(time.monotonic() - dispatch.started_at)
                    timeout_limit = float(dispatch.timeout_limit_s)
                    if timeout_limit <= 0 or elapsed_running < timeout_limit:
                        continue
                    ok, kill_detail = _kill_run_process_tree(dispatch.job.plan.run_dir)
                    dispatch.kill_sent = True
                    dispatch.kill_detail = str(kill_detail)
                    _append_csv_row(
                        events_csv, event_fields,
                        {
                            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                            "event": "dispatch_timeout_kill",
                            "slot_id": int(dispatch.slot_id),
                            "dist_name": str(dispatch.job.dist_name),
                            "algorithm": str(dispatch.job.algorithm_key),
                            "variant": str(dispatch.job.variant.raw),
                            "request_number": int(dispatch.job.request_number),
                            "seed": int(dispatch.job.seed),
                            "run_name": str(dispatch.job.plan.run_name),
                            "attempt": int(dispatch.attempt),
                            "predicted_seconds": float(dispatch.predicted_seconds),
                            "elapsed_seconds": float(elapsed_running),
                            "status": "kill_sent" if ok else "kill_failed",
                            "ratio": "",
                            "algo_coef": float(model.algo_coef.get(str(dispatch.job.algorithm_key).upper(), 1.0)),
                            "dist_coef": float(model.dist_coef.get(str(dispatch.job.dist_name).upper(), 1.0)),
                            "global_scale": float(model.global_scale),
                            "active_limit": int(active_limit),
                            "cpu_percent": pressure_last.get("cpu_percent", ""),
                            "mem_percent": pressure_last.get("mem_percent", ""),
                            "swap_percent": pressure_last.get("swap_percent", ""),
                            "load_per_core": pressure_last.get("load_per_core", ""),
                            "remaining_jobs": int(_remaining_jobs_count()),
                            "timeout_like": 1,
                            "requeue_reason": "dispatch_timeout",
                            "dispatch_timeout_s": float(timeout_limit),
                            "error": str(kill_detail)[:1000],
                        },
                    )
                    if bool(args.auto_workers):
                        prev = int(active_limit)
                        drop = max(1, int(args.timeout_downstep))
                        active_limit = max(min_workers, int(active_limit) - drop)
                        if prev != int(active_limit):
                            print(
                                f"[adaptive][autoscale] dispatch-timeout-backoff: active_limit {prev}->{active_limit} "
                                f"run={dispatch.job.plan.run_name} detail={kill_detail}"
                            )

            if not running:
                if _remaining_jobs_count() > 0:
                    time.sleep(0.2)
                continue

            done, _ = concurrent.futures.wait(
                list(running.keys()),
                timeout=1.0,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if not done:
                continue

            for fut in done:
                dispatch = running.pop(fut)
                slot_id = int(dispatch.slot_id)
                free_slots.append(slot_id)
                completed_attempts += 1
                job = dispatch.job
                attempt_no = int(dispatch.attempt)
                try:
                    result = fut.result()
                except Exception:
                    result = TaskResult(
                        run_name=str(job.plan.run_name),
                        status="failed_internal_exception",
                        elapsed_seconds=float(time.monotonic() - dispatch.started_at),
                        error=traceback.format_exc(),
                    )

                slot_pred_loads[slot_id] = max(
                    0.0,
                    float(slot_pred_loads[slot_id]) - float(dispatch.predicted_seconds) + float(result.elapsed_seconds),
                )
                slot_actual_loads[slot_id] += float(result.elapsed_seconds)
                status_counter[result.status] = status_counter.get(result.status, 0) + 1

                ratio = float("nan")
                algo_coef_val = float(model.algo_coef.get(str(job.algorithm_key).upper(), 1.0))
                dist_coef_val = float(model.dist_coef.get(str(job.dist_name).upper(), 1.0))
                reason_text = ""
                should_reschedule = False
                timeout_like_flag = 0
                requeue_reason = ""
                status_out = str(result.status)
                if result.status == "ok":
                    upd = model.update(
                        algo_key=job.algorithm_key,
                        dist_name=job.dist_name,
                        predicted_seconds=float(dispatch.predicted_seconds),
                        actual_seconds=float(result.elapsed_seconds),
                    )
                    ratio = float(upd.get("ratio", float("nan")))
                    algo_coef_val = float(upd.get("algo_coef", algo_coef_val))
                    dist_coef_val = float(upd.get("dist_coef", dist_coef_val))
                    failures.pop(job.job_key, None)
                    completed_jobs_final += 1
                elif result.status in {"dry_run", "skipped_completed"}:
                    failures.pop(job.job_key, None)
                    completed_jobs_final += 1
                else:
                    reason_text = _compose_failure_reason(job.plan.run_dir, result.status, result.error)
                    reason_text_lower = str(reason_text).lower()
                    timeout_like = any(tok in reason_text_lower for tok in reschedule_reason_tokens)
                    lock_like = ("lock" in reason_text_lower and "busy" in reason_text_lower) or (
                        str(result.status).strip().lower() == "skipped_locked"
                    )
                    interrupted_like = "interrupted" in reason_text_lower
                    transient_like = timeout_like or lock_like or interrupted_like
                    timeout_like_flag = int(timeout_like or dispatch.kill_sent)
                    if bool(args.auto_workers) and timeout_like:
                        prev = int(active_limit)
                        drop = max(1, int(args.timeout_downstep))
                        active_limit = max(min_workers, int(active_limit) - drop)
                        last_adjust_event = time.monotonic()
                        print(
                            f"[adaptive][autoscale] timeout-backoff: active_limit {prev}->{active_limit} "
                            f"run={job.plan.run_name} reason={reason_text}"
                        )
                    unknown_retry = (
                        bool(args.reschedule_unknown_once)
                        and str(result.status).startswith("failed_")
                        and attempt_no == 1
                        and not transient_like
                    )
                    if (
                        bool(args.reschedule_timeout_jobs)
                        and (
                            transient_like
                            or unknown_retry
                            or (lock_like and bool(args.reschedule_on_locked))
                        )
                        and attempt_no < int(reschedule_max_attempts)
                    ):
                        should_reschedule = True
                        status_out = f"{result.status}_requeued"
                        if timeout_like:
                            requeue_reason = "timeout_like"
                        elif lock_like:
                            requeue_reason = "lock_busy"
                        elif unknown_retry:
                            requeue_reason = "unknown_retry_once"
                        else:
                            requeue_reason = "transient"
                        delay = min(
                            float(args.requeue_delay_max_sec),
                            float(args.requeue_delay_base_sec) * (2 ** max(0, attempt_no - 1)),
                        )
                        deferred_jobs.append((time.monotonic() + float(delay), job))
                        print(
                            f"[adaptive][reschedule] {job.job_key} attempt={attempt_no}/"
                            f"{int(reschedule_max_attempts)} reason={reason_text} delay={delay:.1f}s"
                        )
                    else:
                        failures[job.job_key] = f"{result.status}: {result.error[:400]}"
                        completed_jobs_final += 1

                _append_csv_row(
                    events_csv, event_fields,
                    {
                        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                        "event": "complete",
                        "slot_id": int(slot_id),
                        "dist_name": str(job.dist_name),
                        "algorithm": str(job.algorithm_key),
                        "variant": str(job.variant.raw),
                        "request_number": int(job.request_number),
                        "seed": int(job.seed),
                        "run_name": str(result.run_name),
                        "attempt": int(attempt_no),
                        "predicted_seconds": float(dispatch.predicted_seconds),
                        "elapsed_seconds": float(result.elapsed_seconds),
                        "status": status_out,
                        "ratio": "" if (ratio != ratio) else float(ratio),
                        "algo_coef": float(algo_coef_val),
                        "dist_coef": float(dist_coef_val),
                        "global_scale": float(model.global_scale),
                        "active_limit": int(active_limit),
                        "cpu_percent": pressure_last.get("cpu_percent", ""),
                        "mem_percent": pressure_last.get("mem_percent", ""),
                        "swap_percent": pressure_last.get("swap_percent", ""),
                        "load_per_core": pressure_last.get("load_per_core", ""),
                        "remaining_jobs": int(_remaining_jobs_count()),
                        "timeout_like": int(timeout_like_flag),
                        "requeue_reason": str(requeue_reason),
                        "dispatch_timeout_s": float(dispatch.timeout_limit_s),
                        "error": str(reason_text or result.error)[:1000],
                    },
                )
                _save_json(state_path, model.to_dict())
                print(
                    f"[adaptive] done attempt={completed_attempts:03d} final={completed_jobs_final:03d}/{len(all_jobs)} "
                    f"{job.variant.raw}|{job.dist_name}|R{job.request_number}|S{job.seed} "
                    f"attempt={attempt_no} status={status_out} "
                    f"elapsed={result.elapsed_seconds:.1f}s pred={dispatch.predicted_seconds:.1f}s"
                )

    total_elapsed = time.monotonic() - run_started
    makespan = max(slot_actual_loads) if slot_actual_loads else 0.0
    summary = {
        "run_root": str(run_root),
        "total_jobs": len(all_jobs),
        "skipped_completed": int(skipped_completed_total),
        "completed_attempts": int(completed_attempts),
        "completed_jobs": int(completed_jobs_final),
        "failed_jobs": int(len(failures)),
        "deferred_jobs_left": int(len(deferred_jobs)),
        "status_counter": status_counter,
        "worker_cap": int(worker_cap),
        "active_limit_final": int(active_limit),
        "min_workers": int(min_workers),
        "auto_workers": int(bool(args.auto_workers)),
        "total_elapsed_seconds": float(total_elapsed),
        "slot_actual_loads": [float(x) for x in slot_actual_loads],
        "slot_pred_loads": [float(x) for x in slot_pred_loads],
        "actual_makespan_seconds": float(makespan),
        "pressure_last": pressure_last,
        "model": model.to_dict(),
        "failures": failures,
    }
    _save_json(summary_json, summary)
    _save_json(state_path, model.to_dict())
    print(f"[adaptive] summary_json={summary_json}")
    print(
        f"[adaptive] finished total={len(all_jobs)} completed={completed_jobs_final} "
        f"attempts={completed_attempts} failed={len(failures)} "
        f"makespan={makespan:.1f}s elapsed={total_elapsed:.1f}s"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

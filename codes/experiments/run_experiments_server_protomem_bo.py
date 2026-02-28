import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

THIS_DIR = Path(__file__).resolve().parent
CODES_DIR = THIS_DIR.parent
SERVER_OUTPUT_ROOT = CODES_DIR / "nexus"
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from experiments.run_experiments_common import ExperimentConfig, run_experiments


TARGET_DISTRIBUTIONS = [
    "G_10_30_60",
    "O_10_60",
    "O_30_60",
    "F2_10_60",
]


@dataclass
class TrialSummary:
    score: float
    core_mean_reward: float
    core_std_reward: float
    summary_mean_reward: float
    implement_mean_reward: float
    implement_action1_ratio_mean: float
    proto_top1_entropy_mean: float
    action_bias_penalty: float
    collapse_penalty: float
    missing: int
    failed_tasks: int
    runtime_sec: float
    rewards: Dict[str, float]
    implement_rewards: Dict[str, float]
    implement_action1_ratios: Dict[str, float]
    proto_top1_entropy: Dict[str, float]
    task_count: int


def _c(text: str, color: str = "", bold: bool = False) -> str:
    if not (sys.stdout and sys.stdout.isatty()):
        return text
    palette = {
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
        "white": "37",
    }
    codes: List[str] = []
    if bold:
        codes.append("1")
    if color and color in palette:
        codes.append(palette[color])
    if not codes:
        return text
    return f"\033[{';'.join(codes)}m{text}\033[0m"


def resolve_target_run_root(run_folder: str) -> Path:
    raw = str(run_folder or "").strip()
    if not raw:
        raise ValueError("--run-folder is required")
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    return (SERVER_OUTPUT_ROOT / candidate).resolve()


def parse_int_csv(raw: str) -> List[int]:
    out: List[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    if not out:
        raise ValueError("seed list cannot be empty")
    return out


def build_config(seeds: Sequence[int], generator_workers: int) -> ExperimentConfig:
    return ExperimentConfig(
        name="protomem_bo",
        distributions=list(TARGET_DISTRIBUTIONS),
        request_numbers=[30],
        algorithms=["PPO_PROTOMEM"],
        seeds=[int(x) for x in seeds],
        generator_workers=max(1, int(generator_workers)),
        run_baseline=False,
        baseline_include_random=False,
        run_metrics=False,
        run_plots=False,
        cleanup_after_run=False,
        log_subdir="server_runs",
    )


@contextmanager
def temporary_env(overrides: Dict[str, str]):
    backup: Dict[str, str] = {}
    missing: List[str] = []
    for key, value in overrides.items():
        if key in os.environ:
            backup[key] = os.environ[key]
        else:
            missing.append(key)
        os.environ[key] = str(value)
    try:
        yield
    finally:
        for key in missing:
            os.environ.pop(key, None)
        for key, value in backup.items():
            os.environ[key] = value


def _atomic_write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _master_time_from_resource_usage(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    st = data.get("stage_wall_time_sec")
    if not isinstance(st, dict):
        return None
    total = 0.0
    has = False
    for key, value in st.items():
        if not str(key).startswith("master_sec_attempt_"):
            continue
        try:
            total += float(value)
            has = True
        except Exception:
            continue
    return total if has else None


def _estimate_run_seconds_from_history(root: Path, algorithm: str) -> Optional[float]:
    pattern = re.compile(rf"^run_.*_R30_(.+)_{re.escape(algorithm)}_S\d+$")
    samples: List[float] = []
    for run_dir in root.rglob("run_*"):
        if not run_dir.is_dir():
            continue
        m = pattern.match(run_dir.name)
        if not m:
            continue
        dist_name = m.group(1)
        if dist_name not in TARGET_DISTRIBUTIONS:
            continue
        t = _master_time_from_resource_usage(run_dir / "resource_usage.json")
        if t is not None and t > 30:
            samples.append(t)
    if not samples:
        return None
    return float(statistics.median(samples))


def estimate_runtime_seconds(
    n_trials: int,
    task_count_per_trial: int,
    max_workers_per_trial: int,
    bo_jobs: int,
    per_run_seconds: Optional[float],
) -> Optional[float]:
    if per_run_seconds is None:
        return None
    effective_workers = max(1, min(task_count_per_trial, int(max_workers_per_trial)))
    batches_per_trial = math.ceil(task_count_per_trial / effective_workers)
    one_trial = per_run_seconds * batches_per_trial
    total = one_trial * max(1, int(n_trials)) / max(1, int(bo_jobs))
    return float(total)


def _read_average_reward(summary_csv: Path) -> Optional[float]:
    if not summary_csv.exists():
        return None
    try:
        with summary_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = row.get("average_reward")
                if raw is None:
                    continue
                return float(raw)
    except Exception:
        return None
    return None


def _read_implement_metrics(trace_csv: Path) -> Tuple[Optional[float], Optional[float], int]:
    if not trace_csv.exists():
        return None, None, 0
    rewards: List[float] = []
    action01_total = 0
    action1 = 0
    try:
        with trace_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                phase = str(row.get("phase", "")).strip().lower()
                if phase != "implement":
                    continue
                try:
                    reward = float(str(row.get("reward", "")).strip())
                except Exception:
                    continue
                # Keep only RL step-like rows and exclude sentinel values.
                if reward not in (0.0, 1.0):
                    continue
                rewards.append(reward)
                action = str(row.get("action", "")).strip()
                if action in {"0", "1"}:
                    action01_total += 1
                    if action == "1":
                        action1 += 1
    except Exception:
        return None, None, 0
    mean_reward = (sum(rewards) / len(rewards)) if rewards else None
    action1_ratio = (float(action1) / float(action01_total)) if action01_total > 0 else None
    return (float(mean_reward) if mean_reward is not None else None, action1_ratio, len(rewards))


def _read_proto_top1_entropy(training_csv: Path) -> Optional[float]:
    if not training_csv.exists():
        return None
    values: List[float] = []
    try:
        with training_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = str(row.get("pm_proto_top1_entropy", "")).strip()
                if raw == "":
                    continue
                try:
                    values.append(float(raw))
                except Exception:
                    continue
    except Exception:
        return None
    if not values:
        return None
    return float(sum(values) / len(values))


def _find_task_run_dir(trial_root: Path, dist_name: str, seed: int) -> Optional[Path]:
    suffix = f"_R30_{dist_name}_PPO_PROTOMEM_S{seed}"
    matches = [d for d in trial_root.glob(f"run_*{suffix}") if d.is_dir()]
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def evaluate_trial_outputs(trial_root: Path, distributions: Sequence[str], seeds: Sequence[int], failed: int, runtime_sec: float) -> TrialSummary:
    rewards: Dict[str, float] = {}
    implement_rewards: Dict[str, float] = {}
    implement_action1_ratios: Dict[str, float] = {}
    proto_top1_entropy: Dict[str, float] = {}
    missing = 0
    for dist_name in distributions:
        for seed in seeds:
            run_dir = _find_task_run_dir(trial_root, dist_name, int(seed))
            key = f"{dist_name}|S{seed}"
            if run_dir is None:
                missing += 1
                continue

            reward = _read_average_reward(run_dir / "rl_summary.csv")
            if reward is None:
                missing += 1
            else:
                rewards[key] = reward

            impl_reward, impl_action1_ratio, impl_count = _read_implement_metrics(run_dir / "rl_trace.csv")
            if impl_reward is not None and impl_count > 0:
                implement_rewards[key] = float(impl_reward)
            if impl_action1_ratio is not None:
                implement_action1_ratios[key] = float(impl_action1_ratio)

            proto_entropy = _read_proto_top1_entropy(run_dir / "rl_training.csv")
            if proto_entropy is not None:
                proto_top1_entropy[key] = float(proto_entropy)

    summary_vals = list(rewards.values())
    implement_vals = list(implement_rewards.values())
    action_ratio_vals = list(implement_action1_ratios.values())
    proto_entropy_vals = list(proto_top1_entropy.values())

    summary_mean = float(sum(summary_vals) / len(summary_vals)) if summary_vals else 0.0
    implement_mean = float(sum(implement_vals) / len(implement_vals)) if implement_vals else 0.0
    core_vals = implement_vals if implement_vals else summary_vals
    core_mean = float(sum(core_vals) / len(core_vals)) if core_vals else 0.0
    core_std = float(statistics.pstdev(core_vals)) if len(core_vals) > 1 else 0.0
    action1_ratio_mean = float(sum(action_ratio_vals) / len(action_ratio_vals)) if action_ratio_vals else 0.0
    proto_entropy_mean = float(sum(proto_entropy_vals) / len(proto_entropy_vals)) if proto_entropy_vals else 0.0

    # Penalize extreme action bias in implement phase (especially action=1 overuse).
    action_bias_penalty = 0.0
    if action_ratio_vals:
        action_bias_penalty = max(0.0, action1_ratio_mean - 0.35) * 0.8

    # Penalize prototype-collapse tendency when top1 entropy is too low.
    collapse_penalty = 0.0
    if proto_entropy_vals:
        collapse_penalty = max(0.0, 0.05 - proto_entropy_mean) * 1.5

    # Main objective: implement reward first, then stability and anti-collapse constraints.
    score = core_mean
    score -= 0.12 * core_std
    score -= action_bias_penalty
    score -= collapse_penalty
    score -= 0.10 * float(missing)
    score -= 0.05 * float(failed)

    return TrialSummary(
        score=float(score),
        core_mean_reward=float(core_mean),
        core_std_reward=float(core_std),
        summary_mean_reward=float(summary_mean),
        implement_mean_reward=float(implement_mean),
        implement_action1_ratio_mean=float(action1_ratio_mean),
        proto_top1_entropy_mean=float(proto_entropy_mean),
        action_bias_penalty=float(action_bias_penalty),
        collapse_penalty=float(collapse_penalty),
        missing=int(missing),
        failed_tasks=int(failed),
        runtime_sec=float(runtime_sec),
        rewards=rewards,
        implement_rewards=implement_rewards,
        implement_action1_ratios=implement_action1_ratios,
        proto_top1_entropy=proto_top1_entropy,
        task_count=len(distributions) * len(seeds),
    )


def suggest_params(trial) -> Dict[str, str]:
    # Stop-bleeding search space: softer routing + stronger anti-collapse + gentle memory updates.
    tau = trial.suggest_float("PM_TAU", 0.72, 0.98)
    lambda_sparse = trial.suggest_float("PM_LAMBDA_SPARSE", 1e-6, 2e-4, log=True)
    lambda_div = trial.suggest_float("PM_LAMBDA_DIV", 2e-3, 2e-2, log=True)
    mem_lr_scale = trial.suggest_float("PM_MEM_LR_SCALE", 0.15, 0.80)
    use_smooth = trial.suggest_categorical("PM_USE_SMOOTH", [0, 1])
    smooth_alpha = trial.suggest_float("PM_SMOOTH_ALPHA", 0.08, 0.22) if use_smooth == 1 else 0.10
    lambda_stable = trial.suggest_float("PM_LAMBDA_STABLE", 0.005, 0.12)
    stable_warmup = trial.suggest_int("PM_STABLE_WARMUP_UPDATES", 0, 6)
    n_epochs = trial.suggest_int("PM_N_EPOCHS", 4, 7)
    lr = trial.suggest_float("PM_LR", 1e-4, 4e-4, log=True)

    return {
        "PM_INPUT_MODE": "full",
        "PM_NUM_PROTOTYPES": "32",
        "PM_MEM_DIM": "64",
        "PM_HIDDEN_DIM": "64",
        "PM_TAU": f"{tau:.6f}",
        "PM_LAMBDA_SPARSE": f"{lambda_sparse:.8g}",
        "PM_LAMBDA_DIV": f"{lambda_div:.8g}",
        "PM_LAMBDA_STABLE": f"{lambda_stable:.8g}",
        "PM_LAMBDA_AUX": "0.0",
        "PM_MEM_LR_SCALE": f"{mem_lr_scale:.6f}",
        "PM_USE_SMOOTH": str(int(use_smooth)),
        "PM_SMOOTH_ALPHA": f"{smooth_alpha:.6f}",
        "PM_SMOOTH_TRAIN_TEST_CONSISTENT": "1",
        "PM_STABLE_BUF_PER_PHASE": "400",
        "PM_STABLE_BATCH_RATIO": "0.30",
        "PM_STABLE_WARMUP_UPDATES": str(int(stable_warmup)),
        "PM_N_STEPS": "10",
        "PM_BATCH_SIZE": "10",
        "PM_N_EPOCHS": str(int(n_epochs)),
        "PM_LR": f"{lr:.8g}",
    }


def append_trial_csv(csv_path: Path, trial_number: int, summary: TrialSummary, params: Dict[str, str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    fieldnames = [
        "trial",
        "score",
        "core_mean_reward",
        "core_std_reward",
        "summary_mean_reward",
        "implement_mean_reward",
        "implement_action1_ratio_mean",
        "proto_top1_entropy_mean",
        "action_bias_penalty",
        "collapse_penalty",
        "missing",
        "failed_tasks",
        "runtime_sec",
        "task_count",
        "params_json",
        "rewards_json",
        "implement_rewards_json",
        "implement_action1_ratios_json",
        "proto_top1_entropy_json",
    ]
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(
            {
                "trial": int(trial_number),
                "score": f"{summary.score:.6f}",
                "core_mean_reward": f"{summary.core_mean_reward:.6f}",
                "core_std_reward": f"{summary.core_std_reward:.6f}",
                "summary_mean_reward": f"{summary.summary_mean_reward:.6f}",
                "implement_mean_reward": f"{summary.implement_mean_reward:.6f}",
                "implement_action1_ratio_mean": f"{summary.implement_action1_ratio_mean:.6f}",
                "proto_top1_entropy_mean": f"{summary.proto_top1_entropy_mean:.6f}",
                "action_bias_penalty": f"{summary.action_bias_penalty:.6f}",
                "collapse_penalty": f"{summary.collapse_penalty:.6f}",
                "missing": int(summary.missing),
                "failed_tasks": int(summary.failed_tasks),
                "runtime_sec": f"{summary.runtime_sec:.3f}",
                "task_count": int(summary.task_count),
                "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
                "rewards_json": json.dumps(summary.rewards, ensure_ascii=False, sort_keys=True),
                "implement_rewards_json": json.dumps(summary.implement_rewards, ensure_ascii=False, sort_keys=True),
                "implement_action1_ratios_json": json.dumps(summary.implement_action1_ratios, ensure_ascii=False, sort_keys=True),
                "proto_top1_entropy_json": json.dumps(summary.proto_top1_entropy, ensure_ascii=False, sort_keys=True),
            }
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-folder",
        type=str,
        required=True,
        help="target folder under codes/nexus (or absolute path)",
    )
    parser.add_argument("--n-trials", type=int, default=24, help="number of Bayesian optimization trials")
    parser.add_argument("--bo-jobs", type=int, default=1, help="parallel Optuna trials")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="parallel workers inside each trial (across distributions/seeds)",
    )
    parser.add_argument("--generator-workers", type=int, default=1, help="workers passed into Dynamic_master34959.py")
    parser.add_argument("--seeds", type=str, default="42", help="comma-separated seeds, e.g. 42,43")
    parser.add_argument("--study-name", type=str, default="protomem_bo_r30", help="Optuna study name")
    parser.add_argument("--sampler-seed", type=int, default=42, help="seed for Optuna TPE sampler")
    parser.add_argument("--timeout-hours", type=float, default=0.0, help="stop optimization after N hours (0=disable)")
    parser.add_argument(
        "--estimate-per-run-sec",
        type=float,
        default=0.0,
        help="manual seconds per single run (if 0, auto-estimate from historical logs)",
    )
    parser.add_argument(
        "--protomem-slowdown",
        type=float,
        default=1.12,
        help="multiplier applied when estimated from PPO history",
    )
    parser.add_argument("--dry-run-trial", action="store_true", help="run objective once without Optuna for validation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = parse_int_csv(args.seeds)
    run_root = resolve_target_run_root(args.run_folder)
    run_root.mkdir(parents=True, exist_ok=True)

    try:
        import optuna
    except Exception:
        print(_c("Missing dependency: optuna", "red", True))
        print("Install with:")
        print("  python -m pip install optuna")
        return 2

    history_root = CODES_DIR / "nexus"
    if args.estimate_per_run_sec > 0:
        per_run_sec = float(args.estimate_per_run_sec)
        estimate_source = "manual"
    else:
        per_run_protomem = _estimate_run_seconds_from_history(history_root, algorithm="PPO_PROTOMEM")
        if per_run_protomem is not None:
            per_run_sec = per_run_protomem
            estimate_source = "history:PPO_PROTOMEM"
        else:
            per_run_ppo = _estimate_run_seconds_from_history(history_root, algorithm="PPO")
            if per_run_ppo is not None:
                per_run_sec = float(per_run_ppo) * float(args.protomem_slowdown)
                estimate_source = f"history:PPO x {args.protomem_slowdown:.2f}"
            else:
                per_run_sec = None
                estimate_source = "unavailable"

    task_count = len(TARGET_DISTRIBUTIONS) * len(seeds)
    effective_workers = max(1, min(int(args.max_workers), task_count))
    max_parallel_cores = int(args.bo_jobs) * effective_workers
    est_total_sec = estimate_runtime_seconds(
        n_trials=int(args.n_trials),
        task_count_per_trial=task_count,
        max_workers_per_trial=effective_workers,
        bo_jobs=max(1, int(args.bo_jobs)),
        per_run_seconds=per_run_sec,
    )
    est_one_trial_sec = estimate_runtime_seconds(
        n_trials=1,
        task_count_per_trial=task_count,
        max_workers_per_trial=effective_workers,
        bo_jobs=1,
        per_run_seconds=per_run_sec,
    )

    print(f"{_c('[protomem_bo]', 'cyan', True)} run_root={run_root}")
    print(
        f"{_c('[protomem_bo]', 'cyan', True)} "
        f"tasks_per_trial={task_count} workers_per_trial={effective_workers} bo_jobs={args.bo_jobs} "
        f"max_parallel_cores~{max_parallel_cores}"
    )
    if est_total_sec is not None and est_one_trial_sec is not None:
        print(
            f"{_c('[protomem_bo]', 'cyan', True)} "
            f"estimate_source={estimate_source} "
            f"single_trial~{est_one_trial_sec/60:.1f}min total~{est_total_sec/3600:.2f}h"
        )
    else:
        print(f"{_c('[protomem_bo]', 'cyan', True)} estimate_source={estimate_source} (no runtime estimate)")

    db_path = run_root / "optuna_study.db"
    trials_csv = run_root / "bo_trials.csv"
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_distributions": TARGET_DISTRIBUTIONS,
        "request_numbers": [30],
        "algorithm": "PPO_PROTOMEM",
        "seeds": seeds,
        "n_trials": int(args.n_trials),
        "bo_jobs": int(args.bo_jobs),
        "workers_per_trial": int(effective_workers),
        "max_parallel_cores_est": int(max_parallel_cores),
        "estimate_source": estimate_source,
        "estimate_per_run_sec": per_run_sec,
        "estimate_one_trial_sec": est_one_trial_sec,
        "estimate_total_sec": est_total_sec,
        "objective": {
            "primary": "implement_mean_reward (fallback to summary_mean_reward if implement unavailable)",
            "penalties": {
                "core_std_weight": 0.12,
                "action_bias_threshold": 0.35,
                "action_bias_scale": 0.8,
                "collapse_entropy_floor": 0.05,
                "collapse_scale": 1.5,
                "missing_weight": 0.10,
                "failed_weight": 0.05,
            },
        },
    }
    _atomic_write_json(run_root / "bo_plan.json", metadata)

    timeout_sec = None
    if args.timeout_hours and float(args.timeout_hours) > 0:
        timeout_sec = int(float(args.timeout_hours) * 3600)

    io_lock = threading.Lock()

    def objective(trial):
        params = suggest_params(trial)
        trial_root = run_root / f"trial_{trial.number:04d}"
        trial_root.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(trial_root / "trial_params.json", {"trial": trial.number, "params": params})

        config = build_config(seeds=seeds, generator_workers=args.generator_workers)
        config.name = f"protomem_bo:t{trial.number:04d}"
        config.log_subdir = str(trial_root)
        # For BO each trial must be independent.
        config.resume_existing = False
        config.skip_completed = False
        config.notify_on_failure = False
        config.notify_on_success = False

        t0 = time.monotonic()
        with temporary_env(params):
            failed = run_experiments(config, max_workers=effective_workers, dry_run=False)
        runtime = time.monotonic() - t0

        summary = evaluate_trial_outputs(
            trial_root=trial_root,
            distributions=TARGET_DISTRIBUTIONS,
            seeds=seeds,
            failed=int(failed),
            runtime_sec=runtime,
        )
        trial_payload = {
            "trial": int(trial.number),
            "status": "done",
            "score": summary.score,
            "core_mean_reward": summary.core_mean_reward,
            "core_std_reward": summary.core_std_reward,
            "summary_mean_reward": summary.summary_mean_reward,
            "implement_mean_reward": summary.implement_mean_reward,
            "implement_action1_ratio_mean": summary.implement_action1_ratio_mean,
            "proto_top1_entropy_mean": summary.proto_top1_entropy_mean,
            "action_bias_penalty": summary.action_bias_penalty,
            "collapse_penalty": summary.collapse_penalty,
            "missing": summary.missing,
            "failed_tasks": summary.failed_tasks,
            "runtime_sec": summary.runtime_sec,
            "params": params,
            "summary_rewards": summary.rewards,
            "implement_rewards": summary.implement_rewards,
            "implement_action1_ratios": summary.implement_action1_ratios,
            "proto_top1_entropy": summary.proto_top1_entropy,
        }
        with io_lock:
            append_trial_csv(trials_csv, trial_number=trial.number, summary=summary, params=params)
            _atomic_write_json(trial_root / "trial_result.json", trial_payload)

        trial.set_user_attr("run_dir", str(trial_root))
        trial.set_user_attr("core_mean_reward", summary.core_mean_reward)
        trial.set_user_attr("core_std_reward", summary.core_std_reward)
        trial.set_user_attr("summary_mean_reward", summary.summary_mean_reward)
        trial.set_user_attr("implement_mean_reward", summary.implement_mean_reward)
        trial.set_user_attr("implement_action1_ratio_mean", summary.implement_action1_ratio_mean)
        trial.set_user_attr("proto_top1_entropy_mean", summary.proto_top1_entropy_mean)
        trial.set_user_attr("action_bias_penalty", summary.action_bias_penalty)
        trial.set_user_attr("collapse_penalty", summary.collapse_penalty)
        trial.set_user_attr("missing", summary.missing)
        trial.set_user_attr("failed_tasks", summary.failed_tasks)
        trial.set_user_attr("runtime_sec", summary.runtime_sec)
        trial.set_user_attr("summary_rewards", summary.rewards)
        trial.set_user_attr("implement_rewards", summary.implement_rewards)
        trial.set_user_attr("implement_action1_ratios", summary.implement_action1_ratios)
        trial.set_user_attr("proto_top1_entropy", summary.proto_top1_entropy)

        print(
            f"{_c('[protomem_bo]', 'cyan', True)} "
            f"{_c(f'TRIAL {trial.number}', 'magenta', True)} "
            f"score={summary.score:.4f} impl={summary.implement_mean_reward:.4f} "
            f"core={summary.core_mean_reward:.4f} std={summary.core_std_reward:.4f} "
            f"a1={summary.implement_action1_ratio_mean:.3f} "
            f"ent={summary.proto_top1_entropy_mean:.4f} "
            f"missing={summary.missing} failed={summary.failed_tasks} "
            f"time={summary.runtime_sec/60:.1f}min"
        )
        return summary.score

    if args.dry_run_trial:
        class DummyTrial:
            number = 0

            @staticmethod
            def suggest_float(name, low, high, log=False):
                if log:
                    return math.sqrt(low * high)
                return (low + high) / 2

            @staticmethod
            def suggest_int(name, low, high):
                return int((low + high) // 2)

            @staticmethod
            def suggest_categorical(name, choices):
                return choices[0]

            @staticmethod
            def set_user_attr(name, value):
                return None

        _ = objective(DummyTrial())
        print(f"{_c('[protomem_bo]', 'cyan', True)} {_c('DRY-RUN', 'green', True)} completed one trial")
        return 0

    sampler = optuna.samplers.TPESampler(seed=int(args.sampler_seed), multivariate=True)
    pruner = optuna.pruners.NopPruner()
    study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{db_path}",
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    print(
        f"{_c('[protomem_bo]', 'cyan', True)} "
        f"{_c('OPTUNA', 'green', True)} study={args.study_name} n_trials={args.n_trials} timeout={timeout_sec}"
    )
    def _on_trial_done(study_obj, frozen_trial):
        if study_obj.best_trial is None:
            return
        payload = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "best_value": float(study_obj.best_value),
            "best_trial": int(study_obj.best_trial.number),
            "best_params": study_obj.best_trial.params,
            "best_user_attrs": study_obj.best_trial.user_attrs,
            "n_trials_total": len(study_obj.trials),
            "last_finished_trial": int(frozen_trial.number),
        }
        with io_lock:
            _atomic_write_json(run_root / "best_result_live.json", payload)

    study.optimize(
        objective,
        n_trials=int(args.n_trials),
        timeout=timeout_sec,
        n_jobs=int(args.bo_jobs),
        gc_after_trial=True,
        callbacks=[_on_trial_done],
    )

    best = {
        "best_value": float(study.best_value),
        "best_trial": int(study.best_trial.number),
        "best_params": study.best_trial.params,
        "best_user_attrs": study.best_trial.user_attrs,
        "n_trials_total": len(study.trials),
    }
    _atomic_write_json(run_root / "best_result.json", best)
    print(
        f"{_c('[protomem_bo]', 'cyan', True)} {_c('BEST', 'green', True)} "
        f"trial={best['best_trial']} value={best['best_value']:.6f}"
    )
    print(json.dumps(best["best_params"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

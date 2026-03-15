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
DEFAULT_SMOKE_WARMSTART_STATE = (THIS_DIR / "transport_scheduler_warmstart_smoke_v1.json").resolve()
DEFAULT_MAIN36_WARMSTART_STATE = (THIS_DIR / "transport_scheduler_warmstart_main36_v1.json").resolve()
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from experiments.transport_main_suite_config import (
    DEFAULT_BASELINE_VARIANTS,
    DEFAULT_REQUEST_NUMBERS,
    DEFAULT_SEEDS,
    FAMILY_REGISTRY,
    WAVE_REGISTRY,
    family_of_dist,
    resolve_distributions_for_waves,
    resolve_wave_names,
    validate_distribution_subset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paper-grade transport main suite runner. "
            "Builds a family-structured benchmark plan on top of the unified/common server execution stack."
        )
    )
    parser.add_argument("--run-folder", type=str, default="transport_main_suite_runs")
    parser.add_argument(
        "--wave",
        action="append",
        default=None,
        help="repeatable wave name, e.g. smoke, core_shift, memory_generalization, full_main",
    )
    parser.add_argument(
        "--incremental-waves",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="when multiple waves are selected, later waves only run distributions not covered by earlier waves",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["unified", "stream", "adaptive"],
        default="adaptive",
        help="execution backend; adaptive is the transport-suite default",
    )

    parser.add_argument(
        "--include-default-baselines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include the current baseline comparison algorithms by default",
    )
    parser.add_argument(
        "--main-variant",
        action="append",
        default=None,
        help="optional future main method variant(s); left empty by default",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        help="additional variant(s) to include, repeatable",
    )

    parser.add_argument("--request-number", type=int, action="append", default=None)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--generator-workers", type=int, default=1)
    parser.add_argument("--n-stack", type=int, default=None)

    parser.add_argument("--stage-mode", type=str, default="train_eval", help="train_eval/train_only/eval_only")
    parser.add_argument("--init-model-path", type=str, default="", help="optional checkpoint to load")
    parser.add_argument("--save-model-path", type=str, default="", help="optional checkpoint to save")

    parser.add_argument(
        "--run-baseline",
        action="store_true",
        default=False,
        help="optional legacy replay baseline stage; disabled by default in transport suite",
    )
    parser.add_argument("--no-run-baseline", action="store_false", dest="run_baseline")
    parser.add_argument("--baseline-include-random", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run-plots", action="store_true", default=False)
    parser.add_argument("--no-run-plots", action="store_false", dest="run_plots")
    parser.add_argument("--run-metrics", action="store_true", default=True)
    parser.add_argument("--no-run-metrics", action="store_false", dest="run_metrics")
    parser.add_argument("--cleanup-after-run", action="store_true", default=False)

    parser.add_argument("--resume-existing", action="store_true", default=True)
    parser.add_argument("--no-resume-existing", action="store_false", dest="resume_existing")
    parser.add_argument("--skip-completed", action="store_true", default=True)
    parser.add_argument("--no-skip-completed", action="store_false", dest="skip_completed")
    parser.add_argument("--precheck", action="store_true", default=True)
    parser.add_argument("--no-precheck", action="store_false", dest="precheck")
    parser.add_argument("--precheck-workers", type=int, default=0)

    parser.add_argument("--skip-run", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--notify-success", action="store_true", default=False)
    parser.add_argument("--no-notify-failure", action="store_false", dest="notify_failure", default=True)
    parser.add_argument("--notify-scheduler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--notify-schedule-times", type=str, default="08:00,12:00,16:00,20:00")
    parser.add_argument("--notify-batch-size", type=int, default=5)
    parser.add_argument("--notify-on-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--notify-on-requeue", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--notify-on-finish", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--notify-live-status-interval-s", type=float, default=30.0)
    parser.add_argument("--notify-state-path", type=str, default="")
    parser.add_argument("--live-status-path", type=str, default="")
    parser.add_argument(
        "--relax-watchdog",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="disable output/progress stall watchdogs to reduce false kills on strong servers",
    )

    parser.add_argument(
        "--scheduler-policy",
        type=str,
        default="optimal_hybrid",
        choices=["optimal_hybrid", "optimal_exact", "adaptive_lpt", "fifo"],
        help="adaptive backend policy; optimal_hybrid is the default exact-frontier scheduler",
    )
    parser.add_argument("--algo-coef-init", type=str, default="", help="optional adaptive algo coef init map")
    parser.add_argument("--dist-coef-init", type=str, default="", help="optional adaptive dist coef init map")
    parser.add_argument("--model-lr", type=float, default=0.35)
    parser.add_argument("--model-global-lr", type=float, default=0.10)
    parser.add_argument("--model-min-pred-sec", type=float, default=60.0)
    parser.add_argument("--scheduler-opt-max-exact-jobs", type=int, default=18)
    parser.add_argument("--scheduler-opt-frontier-jobs", type=int, default=14)
    parser.add_argument("--scheduler-opt-time-limit-sec", type=float, default=3.0)
    parser.add_argument("--scheduler-opt-load-round-sec", type=float, default=1.0)
    parser.add_argument(
        "--scheduler-opt-solver",
        type=str,
        default="auto",
        choices=["auto", "gurobi", "mp_bnb", "serial_bnb"],
        help="exact scheduler backend for adaptive dispatch",
    )
    parser.add_argument("--scheduler-opt-max-solver-workers", type=int, default=2)
    parser.add_argument("--scheduler-opt-gurobi-mip-gap", type=float, default=0.0)
    parser.add_argument("--coef-state-path", type=str, default="", help="optional path to adaptive_scheduler_coef_state.json")
    parser.add_argument(
        "--template-state-path",
        type=str,
        default="",
        help="optional shared warm-start template to refresh during execution; defaults to transport warm-start template",
    )
    parser.add_argument("--reset-coef-state", action="store_true", default=False)
    parser.add_argument("--auto-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-workers", type=int, default=1)
    parser.add_argument("--per-task-mem-gb", type=float, default=2.0)
    parser.add_argument("--adjust-interval-sec", type=float, default=15.0)
    parser.add_argument("--adjust-cooldown-sec", type=float, default=45.0)
    parser.add_argument("--up-step", type=int, default=1)
    parser.add_argument("--down-step", type=int, default=1)
    parser.add_argument("--reschedule-timeout-jobs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reschedule-max-attempts", type=int, default=3)
    parser.add_argument("--reschedule-reasons", type=str, default="timeout,stall,|124")
    parser.add_argument("--reschedule-unknown-once", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reschedule-on-locked", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--requeue-delay-base-sec", type=float, default=20.0)
    parser.add_argument("--requeue-delay-max-sec", type=float, default=300.0)
    parser.add_argument("--dispatch-kill-on-timeout", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dispatch-timeout-factor", type=float, default=8.0)
    parser.add_argument("--dispatch-timeout-min-sec", type=float, default=1800.0)
    parser.add_argument("--dispatch-timeout-max-sec", type=float, default=21600.0)
    return parser.parse_args()


def _resolve_nexus_path(raw: str) -> Path:
    path = Path(str(raw or "").strip())
    if not path:
        raise ValueError("empty path")
    if path.is_absolute():
        return path.resolve()
    return (NEXUS_DIR / path).resolve()


def _dedupe_keep_order(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in values:
        item = str(raw or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _resolve_variants(args: argparse.Namespace) -> Dict[str, List[str]]:
    default_baselines = list(DEFAULT_BASELINE_VARIANTS) if bool(args.include_default_baselines) else []
    main_variants = _dedupe_keep_order([str(v).strip() for v in (args.main_variant or []) if str(v).strip()])
    extra_variants = _dedupe_keep_order([str(v).strip() for v in (args.variant or []) if str(v).strip()])
    all_variants = _dedupe_keep_order(default_baselines + main_variants + extra_variants)
    if not all_variants:
        raise ValueError("no variants selected; enable default baselines or provide --variant / --main-variant")
    return {
        "default_baselines": default_baselines,
        "main_variants": main_variants,
        "extra_variants": extra_variants,
        "all_variants": all_variants,
    }


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


def _validate_backend_compatibility(args: argparse.Namespace) -> None:
    backend = str(args.backend).strip().lower()
    if backend == "stream":
        if str(args.stage_mode).strip().lower() != "train_eval":
            raise ValueError("backend=stream only supports stage_mode=train_eval in the current codebase")
        if str(args.init_model_path).strip():
            raise ValueError("backend=stream does not forward --init-model-path; use backend=unified/adaptive")
        if str(args.save_model_path).strip():
            raise ValueError("backend=stream does not forward --save-model-path; use backend=unified/adaptive")
        return
    if backend == "adaptive":
        return


def _build_unified_cmd(
    *,
    args: argparse.Namespace,
    run_root: Path,
    variants: List[str],
    dist_names: List[str],
    requests: List[int],
    seeds: List[int],
) -> List[str]:
    cmd = [
        sys.executable,
        str(CODES_DIR / "experiments" / "run_experiments_server_unified.py"),
        "--run-folder",
        str(run_root),
        "--max-workers",
        str(int(args.max_workers)),
        "--generator-workers",
        str(int(args.generator_workers)),
        "--stage-mode",
        str(args.stage_mode),
    ]
    for item in variants:
        cmd.extend(["--variant", item])
    for item in dist_names:
        cmd.extend(["--dist-name", item])
    for item in requests:
        cmd.extend(["--request-number", str(int(item))])
    for item in seeds:
        cmd.extend(["--seed", str(int(item))])
    if args.n_stack is not None:
        cmd.extend(["--n-stack", str(int(args.n_stack))])
    if str(args.init_model_path).strip():
        cmd.extend(["--init-model-path", str(args.init_model_path).strip()])
    if str(args.save_model_path).strip():
        cmd.extend(["--save-model-path", str(args.save_model_path).strip()])
    if args.run_baseline:
        cmd.append("--run-baseline")
    else:
        cmd.append("--no-run-baseline")
    if args.baseline_include_random:
        cmd.append("--baseline-include-random")
    if args.run_plots:
        cmd.append("--run-plots")
    else:
        cmd.append("--no-run-plots")
    if args.run_metrics:
        cmd.append("--run-metrics")
    else:
        cmd.append("--no-run-metrics")
    if args.cleanup_after_run:
        cmd.append("--cleanup-after-run")
    if args.resume_existing:
        cmd.append("--resume-existing")
    else:
        cmd.append("--no-resume-existing")
    if args.skip_completed:
        cmd.append("--skip-completed")
    else:
        cmd.append("--no-skip-completed")
    if args.precheck:
        cmd.append("--precheck")
    else:
        cmd.append("--no-precheck")
    cmd.extend(["--precheck-workers", str(int(args.precheck_workers))])
    if args.notify_success:
        cmd.append("--notify-success")
    if not bool(args.notify_failure):
        cmd.append("--no-notify-failure")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def _build_stream_cmd(
    *,
    args: argparse.Namespace,
    run_root: Path,
    variants: List[str],
    dist_names: List[str],
    requests: List[int],
    seeds: List[int],
) -> List[str]:
    cmd = [
        sys.executable,
        str(CODES_DIR / "experiments" / "run_experiments_server_stream.py"),
        "--run-folder",
        str(run_root),
        "--max-workers",
        str(int(args.max_workers)),
        "--generator-workers",
        str(int(args.generator_workers)),
    ]
    for item in variants:
        cmd.extend(["--variant", item])
    for item in dist_names:
        cmd.extend(["--dist-name", item])
    for item in requests:
        cmd.extend(["--request-number", str(int(item))])
    for item in seeds:
        cmd.extend(["--seed", str(int(item))])
    if args.n_stack is not None:
        cmd.extend(["--n-stack", str(int(args.n_stack))])
    if args.run_baseline:
        cmd.append("--run-baseline")
    else:
        cmd.append("--no-run-baseline")
    if args.baseline_include_random:
        cmd.append("--baseline-include-random")
    if args.run_plots:
        cmd.append("--run-plots")
    else:
        cmd.append("--no-run-plots")
    if args.run_metrics:
        cmd.append("--run-metrics")
    else:
        cmd.append("--no-run-metrics")
    if args.cleanup_after_run:
        cmd.append("--cleanup-after-run")
    if args.resume_existing:
        cmd.append("--resume-existing")
    else:
        cmd.append("--no-resume-existing")
    if args.skip_completed:
        cmd.append("--skip-completed")
    else:
        cmd.append("--no-skip-completed")
    if args.precheck:
        cmd.append("--precheck")
    else:
        cmd.append("--no-precheck")
    cmd.extend(["--precheck-workers", str(int(args.precheck_workers))])
    if args.notify_success:
        cmd.append("--notify-success")
    if not bool(args.notify_failure):
        cmd.append("--no-notify-failure")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def _build_adaptive_cmd(
    *,
    args: argparse.Namespace,
    run_root: Path,
    variants: List[str],
    dist_names: List[str],
    requests: List[int],
    seeds: List[int],
    default_warmstart_state: Path,
) -> List[str]:
    cmd = [
        sys.executable,
        str(CODES_DIR / "experiments" / "run_experiments_server_adaptive.py"),
        "--run-folder",
        str(run_root),
        "--max-workers",
        str(int(args.max_workers)),
        "--generator-workers",
        str(int(args.generator_workers)),
        "--stage-mode",
        str(args.stage_mode),
        "--scheduler-policy",
        str(args.scheduler_policy),
        "--model-lr",
        str(float(args.model_lr)),
        "--model-global-lr",
        str(float(args.model_global_lr)),
        "--model-min-pred-sec",
        str(float(args.model_min_pred_sec)),
        "--scheduler-opt-max-exact-jobs",
        str(int(args.scheduler_opt_max_exact_jobs)),
        "--scheduler-opt-frontier-jobs",
        str(int(args.scheduler_opt_frontier_jobs)),
        "--scheduler-opt-time-limit-sec",
        str(float(args.scheduler_opt_time_limit_sec)),
        "--scheduler-opt-load-round-sec",
        str(float(args.scheduler_opt_load_round_sec)),
        "--scheduler-opt-solver",
        str(args.scheduler_opt_solver),
        "--scheduler-opt-max-solver-workers",
        str(int(args.scheduler_opt_max_solver_workers)),
        "--scheduler-opt-gurobi-mip-gap",
        str(float(args.scheduler_opt_gurobi_mip_gap)),
        "--min-workers",
        str(int(args.min_workers)),
        "--per-task-mem-gb",
        str(float(args.per_task_mem_gb)),
        "--adjust-interval-sec",
        str(float(args.adjust_interval_sec)),
        "--adjust-cooldown-sec",
        str(float(args.adjust_cooldown_sec)),
        "--up-step",
        str(int(args.up_step)),
        "--down-step",
        str(int(args.down_step)),
        "--reschedule-max-attempts",
        str(int(args.reschedule_max_attempts)),
        "--reschedule-reasons",
        str(args.reschedule_reasons),
        "--requeue-delay-base-sec",
        str(float(args.requeue_delay_base_sec)),
        "--requeue-delay-max-sec",
        str(float(args.requeue_delay_max_sec)),
        "--dispatch-timeout-factor",
        str(float(args.dispatch_timeout_factor)),
        "--dispatch-timeout-min-sec",
        str(float(args.dispatch_timeout_min_sec)),
        "--dispatch-timeout-max-sec",
        str(float(args.dispatch_timeout_max_sec)),
    ]
    for item in variants:
        cmd.extend(["--variant", item])
    for item in dist_names:
        cmd.extend(["--dist-name", item])
    for item in requests:
        cmd.extend(["--request-number", str(int(item))])
    for item in seeds:
        cmd.extend(["--seed", str(int(item))])
    if args.n_stack is not None:
        cmd.extend(["--n-stack", str(int(args.n_stack))])
    if str(args.init_model_path).strip():
        cmd.extend(["--init-model-path", str(args.init_model_path).strip()])
    if str(args.save_model_path).strip():
        cmd.extend(["--save-model-path", str(args.save_model_path).strip()])
    if args.run_baseline:
        cmd.append("--run-baseline")
    else:
        cmd.append("--no-run-baseline")
    if args.run_plots:
        cmd.append("--run-plots")
    else:
        cmd.append("--no-run-plots")
    if args.run_metrics:
        cmd.append("--run-metrics")
    else:
        cmd.append("--no-run-metrics")
    if args.cleanup_after_run:
        cmd.append("--cleanup-after-run")
    if args.resume_existing:
        cmd.append("--resume-existing")
    else:
        cmd.append("--no-resume-existing")
    if args.skip_completed:
        cmd.append("--skip-completed")
    else:
        cmd.append("--no-skip-completed")
    if args.precheck:
        cmd.append("--precheck")
    else:
        cmd.append("--no-precheck")
    cmd.extend(["--precheck-workers", str(int(args.precheck_workers))])
    if args.notify_success:
        cmd.append("--notify-success")
    if not bool(args.notify_failure):
        cmd.append("--no-notify-failure")
    if args.notify_scheduler:
        cmd.append("--notify-scheduler")
    else:
        cmd.append("--no-notify-scheduler")
    cmd.extend(["--notify-schedule-times", str(args.notify_schedule_times)])
    cmd.extend(["--notify-batch-size", str(int(args.notify_batch_size))])
    if args.notify_on_start:
        cmd.append("--notify-on-start")
    else:
        cmd.append("--no-notify-on-start")
    if args.notify_on_requeue:
        cmd.append("--notify-on-requeue")
    else:
        cmd.append("--no-notify-on-requeue")
    if args.notify_on_finish:
        cmd.append("--notify-on-finish")
    else:
        cmd.append("--no-notify-on-finish")
    cmd.extend(["--notify-live-status-interval-s", str(float(args.notify_live_status_interval_s))])
    if str(args.notify_state_path).strip():
        cmd.extend(["--notify-state-path", str(args.notify_state_path).strip()])
    if str(args.live_status_path).strip():
        cmd.extend(["--live-status-path", str(args.live_status_path).strip()])
    if str(args.algo_coef_init).strip():
        cmd.extend(["--algo-coef-init", str(args.algo_coef_init).strip()])
    if str(args.dist_coef_init).strip():
        cmd.extend(["--dist-coef-init", str(args.dist_coef_init).strip()])
    if str(args.coef_state_path).strip():
        cmd.extend(["--coef-state-path", str(args.coef_state_path).strip()])
    template_state_path = str(args.template_state_path).strip() or str(default_warmstart_state)
    if template_state_path:
        cmd.extend(["--template-state-path", template_state_path])
    if args.reset_coef_state:
        cmd.append("--reset-coef-state")
    if args.auto_workers:
        cmd.append("--auto-workers")
    else:
        cmd.append("--no-auto-workers")
    if args.reschedule_timeout_jobs:
        cmd.append("--reschedule-timeout-jobs")
    else:
        cmd.append("--no-reschedule-timeout-jobs")
    if args.reschedule_unknown_once:
        cmd.append("--reschedule-unknown-once")
    else:
        cmd.append("--no-reschedule-unknown-once")
    if args.reschedule_on_locked:
        cmd.append("--reschedule-on-locked")
    else:
        cmd.append("--no-reschedule-on-locked")
    if args.dispatch_kill_on_timeout:
        cmd.append("--dispatch-kill-on-timeout")
    else:
        cmd.append("--no-dispatch-kill-on-timeout")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def _build_backend_cmd(
    *,
    args: argparse.Namespace,
    run_root: Path,
    variants: List[str],
    dist_names: List[str],
    requests: List[int],
    seeds: List[int],
    default_warmstart_state: Path,
) -> List[str]:
    backend = str(args.backend).strip().lower()
    if backend == "stream":
        return _build_stream_cmd(
            args=args,
            run_root=run_root,
            variants=variants,
            dist_names=dist_names,
            requests=requests,
            seeds=seeds,
        )
    if backend == "adaptive":
        return _build_adaptive_cmd(
            args=args,
            run_root=run_root,
            variants=variants,
            dist_names=dist_names,
            requests=requests,
            seeds=seeds,
            default_warmstart_state=default_warmstart_state,
        )
    return _build_unified_cmd(
        args=args,
        run_root=run_root,
        variants=variants,
        dist_names=dist_names,
        requests=requests,
        seeds=seeds,
    )


def _run_cmd(cmd: List[str], cwd: Path, env: Dict[str, str], *, dry_run: bool) -> int:
    printable = " ".join(shlex.quote(str(x)) for x in cmd)
    print(f"[transport-suite] run: {printable}")
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(cwd), env=env).returncode


def _family_counts(dist_names: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for name in dist_names:
        family = family_of_dist(name)
        counts[family] = int(counts.get(family, 0)) + 1
    return counts


def _write_manifest(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_default_warmstart_state(wave_names: List[str]) -> Path:
    selected = {str(name or "").strip().lower() for name in (wave_names or []) if str(name or "").strip()}
    if "main_36" in selected:
        return DEFAULT_MAIN36_WARMSTART_STATE
    return DEFAULT_SMOKE_WARMSTART_STATE


def _maybe_seed_default_coef_state(
    run_root: Path,
    args: argparse.Namespace,
    wave_names: List[str],
) -> Dict[str, object]:
    explicit_path = str(getattr(args, "coef_state_path", "") or "").strip()
    if explicit_path or bool(getattr(args, "reset_coef_state", False)):
        return {
            "seeded": False,
            "source": "",
            "target": explicit_path,
            "reason": "explicit_or_reset",
        }
    source = _resolve_default_warmstart_state(wave_names)
    target = (run_root / "adaptive_scheduler_coef_state.json").resolve()
    if not source.exists():
        return {
            "seeded": False,
            "source": str(source),
            "target": str(target),
            "reason": "missing_default_seed",
        }
    if target.exists():
        return {
            "seeded": False,
            "source": str(source),
            "target": str(target),
            "reason": "target_exists",
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return {
        "seeded": True,
        "source": str(source),
        "target": str(target),
        "reason": "auto_seeded",
    }


def main() -> int:
    args = parse_args()
    _validate_backend_compatibility(args)

    run_root = _resolve_nexus_path(args.run_folder)
    run_root.mkdir(parents=True, exist_ok=True)

    wave_names = resolve_wave_names(args.wave)
    default_warmstart_state = _resolve_default_warmstart_state(wave_names)
    seeded_coef_state = (
        _maybe_seed_default_coef_state(run_root, args, wave_names)
        if str(args.backend).strip().lower() == "adaptive"
        else {
            "seeded": False,
            "source": "",
            "target": str((run_root / "adaptive_scheduler_coef_state.json").resolve()),
            "reason": "backend_not_adaptive",
        }
    )
    wave_dist_map = resolve_distributions_for_waves(
        wave_names,
        incremental=bool(args.incremental_waves),
    )
    all_requested_dists = _dedupe_keep_order(
        [dist for _, items in wave_dist_map.items() for dist in items]
    )
    validate_distribution_subset(all_requested_dists)

    requests = [int(x) for x in (args.request_number or DEFAULT_REQUEST_NUMBERS)]
    seeds = [int(x) for x in (args.seed or DEFAULT_SEEDS)]
    variant_info = _resolve_variants(args)

    env = dict(os.environ)
    env_overrides = _watchdog_env_overrides(bool(args.relax_watchdog))
    env.update(env_overrides)

    commands: Dict[str, List[str]] = {}
    commands_pretty: Dict[str, str] = {}
    wave_payload: Dict[str, object] = {}
    for wave_name in wave_names:
        dist_names = list(wave_dist_map.get(wave_name, []))
        if dist_names:
            cmd = _build_backend_cmd(
                args=args,
                run_root=run_root,
                variants=variant_info["all_variants"],
                dist_names=dist_names,
                requests=requests,
                seeds=seeds,
                default_warmstart_state=default_warmstart_state,
            )
            commands[wave_name] = cmd
            commands_pretty[wave_name] = " ".join(shlex.quote(str(x)) for x in cmd)
        else:
            commands[wave_name] = []
            commands_pretty[wave_name] = ""
        wave_payload[wave_name] = {
            "raw_dist_names": list(WAVE_REGISTRY[wave_name]),
            "effective_dist_names": dist_names,
            "effective_dist_count": len(dist_names),
            "family_counts": _family_counts(dist_names),
        }

    manifest = {
        "run_root": str(run_root),
        "backend": str(args.backend),
        "waves": wave_names,
        "incremental_waves": bool(args.incremental_waves),
        "request_numbers": requests,
        "seeds": seeds,
        "default_baseline_variants": variant_info["default_baselines"],
        "main_variants": variant_info["main_variants"],
        "extra_variants": variant_info["extra_variants"],
        "all_variants": variant_info["all_variants"],
        "main_variant_slot_empty": len(variant_info["main_variants"]) == 0,
        "run_baseline": bool(args.run_baseline),
        "baseline_include_random": bool(args.baseline_include_random),
        "run_plots": bool(args.run_plots),
        "run_metrics": bool(args.run_metrics),
        "cleanup_after_run": bool(args.cleanup_after_run),
        "resume_existing": bool(args.resume_existing),
        "skip_completed": bool(args.skip_completed),
        "precheck": bool(args.precheck),
        "precheck_workers": int(args.precheck_workers),
        "max_workers": int(args.max_workers),
        "generator_workers": int(args.generator_workers),
        "stage_mode": str(args.stage_mode),
        "init_model_path": str(args.init_model_path).strip(),
        "save_model_path": str(args.save_model_path).strip(),
        "relax_watchdog": bool(args.relax_watchdog),
        "notification": {
            "notify_success": bool(args.notify_success),
            "notify_failure": bool(args.notify_failure),
            "notify_scheduler": bool(args.notify_scheduler),
            "notify_schedule_times": str(args.notify_schedule_times),
            "notify_batch_size": int(args.notify_batch_size),
            "notify_on_start": bool(args.notify_on_start),
            "notify_on_requeue": bool(args.notify_on_requeue),
            "notify_on_finish": bool(args.notify_on_finish),
            "notify_live_status_interval_s": float(args.notify_live_status_interval_s),
            "notify_state_path": str(args.notify_state_path).strip(),
            "live_status_path": str(args.live_status_path).strip(),
        },
        "env_overrides": env_overrides,
        "scheduler": {
            "policy": str(args.scheduler_policy),
            "algo_coef_init": str(args.algo_coef_init).strip(),
            "dist_coef_init": str(args.dist_coef_init).strip(),
            "coef_state_path": str(args.coef_state_path).strip(),
            "reset_coef_state": bool(args.reset_coef_state),
            "default_warmstart_state": str(default_warmstart_state),
            "seeded_coef_state": seeded_coef_state,
            "model_lr": float(args.model_lr),
            "model_global_lr": float(args.model_global_lr),
            "model_min_pred_sec": float(args.model_min_pred_sec),
            "scheduler_opt_solver": str(args.scheduler_opt_solver),
            "scheduler_opt_max_solver_workers": int(args.scheduler_opt_max_solver_workers),
            "scheduler_opt_gurobi_mip_gap": float(args.scheduler_opt_gurobi_mip_gap),
            "scheduler_opt_max_exact_jobs": int(args.scheduler_opt_max_exact_jobs),
            "scheduler_opt_frontier_jobs": int(args.scheduler_opt_frontier_jobs),
            "scheduler_opt_time_limit_sec": float(args.scheduler_opt_time_limit_sec),
            "scheduler_opt_load_round_sec": float(args.scheduler_opt_load_round_sec),
            "template_state_path": str(args.template_state_path).strip() or str(default_warmstart_state),
            "auto_workers": bool(args.auto_workers),
            "min_workers": int(args.min_workers),
            "per_task_mem_gb": float(args.per_task_mem_gb),
            "adjust_interval_sec": float(args.adjust_interval_sec),
            "adjust_cooldown_sec": float(args.adjust_cooldown_sec),
            "up_step": int(args.up_step),
            "down_step": int(args.down_step),
            "reschedule_timeout_jobs": bool(args.reschedule_timeout_jobs),
            "reschedule_max_attempts": int(args.reschedule_max_attempts),
            "reschedule_reasons": str(args.reschedule_reasons),
            "reschedule_unknown_once": bool(args.reschedule_unknown_once),
            "reschedule_on_locked": bool(args.reschedule_on_locked),
            "requeue_delay_base_sec": float(args.requeue_delay_base_sec),
            "requeue_delay_max_sec": float(args.requeue_delay_max_sec),
            "dispatch_kill_on_timeout": bool(args.dispatch_kill_on_timeout),
            "dispatch_timeout_factor": float(args.dispatch_timeout_factor),
            "dispatch_timeout_min_sec": float(args.dispatch_timeout_min_sec),
            "dispatch_timeout_max_sec": float(args.dispatch_timeout_max_sec),
        },
        "family_registry": FAMILY_REGISTRY,
        "wave_registry": wave_payload,
        "commands": commands,
        "commands_pretty": commands_pretty,
    }

    manifest_path = run_root / "transport_main_suite_manifest.json"
    _write_manifest(manifest_path, manifest)
    print(f"[transport-suite] wrote manifest: {manifest_path}")

    if args.skip_run:
        print("[transport-suite] skip-run enabled; manifest only")
        return 0

    for wave_name in wave_names:
        dist_names = list(wave_dist_map.get(wave_name, []))
        if not dist_names:
            print(f"[transport-suite] skip empty wave: {wave_name}")
            continue
        print(
            f"[transport-suite] wave={wave_name} "
            f"dists={len(dist_names)} variants={len(variant_info['all_variants'])} "
            f"seeds={len(seeds)} requests={len(requests)}"
        )
        rc = _run_cmd(commands[wave_name], cwd=ROOT_DIR, env=env, dry_run=bool(args.dry_run))
        if rc != 0:
            print(f"[transport-suite] wave failed: {wave_name} exit={rc}")
            return int(rc)

    print("[transport-suite] all selected waves completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

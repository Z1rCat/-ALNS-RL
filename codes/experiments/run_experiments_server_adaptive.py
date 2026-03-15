from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime
import html
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


@dataclass(frozen=True)
class PredictedPendingJob:
    job: ScheduledJob
    predicted_seconds: float


@dataclass
class SchedulerPlan:
    policy: str
    mode: str
    total_pending_jobs: int
    optimized_jobs: int
    slot_sequences: Dict[int, List[ScheduledJob]]
    best_makespan_seconds: float
    lower_bound_seconds: float
    upper_bound_seconds: float
    exact_nodes: int = 0
    exact_time_seconds: float = 0.0
    time_limit_hit: bool = False
    used_fallback: bool = False

    def summary_dict(self) -> Dict[str, object]:
        return {
            "policy": str(self.policy),
            "mode": str(self.mode),
            "total_pending_jobs": int(self.total_pending_jobs),
            "optimized_jobs": int(self.optimized_jobs),
            "best_makespan_seconds": float(self.best_makespan_seconds),
            "best_makespan_human": _format_eta(float(self.best_makespan_seconds)),
            "lower_bound_seconds": float(self.lower_bound_seconds),
            "lower_bound_human": _format_eta(float(self.lower_bound_seconds)),
            "upper_bound_seconds": float(self.upper_bound_seconds),
            "upper_bound_human": _format_eta(float(self.upper_bound_seconds)),
            "exact_nodes": int(self.exact_nodes),
            "exact_time_seconds": float(self.exact_time_seconds),
            "exact_time_human": _format_eta(float(self.exact_time_seconds)),
            "time_limit_hit": bool(self.time_limit_hit),
            "used_fallback": bool(self.used_fallback),
            "dispatchable_slots": [
                int(slot_id)
                for slot_id, seq in sorted(self.slot_sequences.items(), key=lambda kv: kv[0])
                if seq
            ],
        }


def _predict_pending_jobs(
    pending_by_dist: Dict[str, List[ScheduledJob]],
    model: "AdaptiveDurationModel",
) -> List[PredictedPendingJob]:
    out: List[PredictedPendingJob] = []
    for items in pending_by_dist.values():
        for job in items:
            out.append(
                PredictedPendingJob(
                    job=job,
                    predicted_seconds=float(
                        model.predict(
                            algo_key=job.algorithm_key,
                            dist_name=job.dist_name,
                            request_number=int(job.request_number),
                        )
                    ),
                )
            )
    out.sort(key=lambda item: (-float(item.predicted_seconds), str(item.job.job_key)))
    return out


def _greedy_lpt_schedule(
    *,
    base_loads: List[float],
    jobs: List[PredictedPendingJob],
) -> Tuple[Dict[int, List[ScheduledJob]], List[float], float]:
    loads = [float(x) for x in base_loads]
    sequences: Dict[int, List[ScheduledJob]] = {idx: [] for idx in range(len(loads))}
    ordered_jobs = sorted(jobs, key=lambda item: (-float(item.predicted_seconds), str(item.job.job_key)))
    for item in ordered_jobs:
        slot_id = min(range(len(loads)), key=lambda idx: (loads[idx], idx))
        sequences[slot_id].append(item.job)
        loads[slot_id] += float(item.predicted_seconds)
    makespan = max(loads) if loads else 0.0
    return sequences, loads, float(makespan)


def _pop_specific_pending_job(
    pending_by_dist: Dict[str, List[ScheduledJob]],
    target: ScheduledJob,
) -> Optional[ScheduledJob]:
    dist_key = str(target.dist_name)
    queue = pending_by_dist.get(dist_key, [])
    for idx, job in enumerate(queue):
        if job is target or str(job.job_key) == str(target.job_key):
            return queue.pop(idx)
    return None


def _exact_branch_and_bound_schedule(
    *,
    base_loads: List[float],
    jobs: List[PredictedPendingJob],
    time_limit_s: float,
    load_round_seconds: float,
) -> Tuple[Dict[int, List[ScheduledJob]], List[float], float, float, int, bool, float]:
    if not jobs:
        empty = {idx: [] for idx in range(len(base_loads))}
        makespan = max(base_loads) if base_loads else 0.0
        return empty, [float(x) for x in base_loads], float(makespan), float(makespan), 0, False, 0.0

    seed_sequences, seed_loads, seed_makespan = _greedy_lpt_schedule(base_loads=base_loads, jobs=jobs)
    best_sequences = {idx: list(items) for idx, items in seed_sequences.items()}
    best_loads = [float(x) for x in seed_loads]
    best_makespan = float(seed_makespan)

    ordered_jobs = sorted(jobs, key=lambda item: (-float(item.predicted_seconds), str(item.job.job_key)))
    m = len(base_loads)
    suffix_sum: List[float] = [0.0 for _ in range(len(ordered_jobs) + 1)]
    for idx in range(len(ordered_jobs) - 1, -1, -1):
        suffix_sum[idx] = suffix_sum[idx + 1] + float(ordered_jobs[idx].predicted_seconds)

    round_unit = max(1.0, float(load_round_seconds))
    visited = set()
    start_ts = time.monotonic()
    exact_nodes = 0
    time_limit_hit = False
    work_sequences: Dict[int, List[ScheduledJob]] = {idx: [] for idx in range(m)}

    def _round_load(x: float) -> int:
        return int(round(float(x) / round_unit))

    def _lower_bound(idx: int, loads: List[float], assigned_sum: float) -> float:
        rem_sum = float(suffix_sum[idx])
        lb = max(max(loads), (float(sum(base_loads)) + float(assigned_sum) + rem_sum) / float(max(1, m)))
        if idx < len(ordered_jobs):
            lb = max(lb, min(loads) + float(ordered_jobs[idx].predicted_seconds))
        return float(lb)

    def _dfs(idx: int, loads: List[float], assigned_sum: float) -> None:
        nonlocal best_sequences, best_loads, best_makespan, exact_nodes, time_limit_hit
        if time.monotonic() - start_ts > float(time_limit_s):
            time_limit_hit = True
            return
        exact_nodes += 1
        lb = _lower_bound(idx, loads, assigned_sum)
        if lb >= float(best_makespan) - 1e-9:
            return
        if idx >= len(ordered_jobs):
            candidate = max(loads) if loads else 0.0
            if candidate < float(best_makespan) - 1e-9:
                best_makespan = float(candidate)
                best_loads = [float(x) for x in loads]
                best_sequences = {slot_id: list(items) for slot_id, items in work_sequences.items()}
            return
        key = (idx, tuple(sorted(_round_load(x) for x in loads)))
        if key in visited:
            return
        visited.add(key)

        item = ordered_jobs[idx]
        duration = float(item.predicted_seconds)
        seen_loads = set()
        for slot_id in sorted(range(m), key=lambda s: (loads[s], s)):
            rounded = _round_load(loads[slot_id])
            if rounded in seen_loads:
                continue
            seen_loads.add(rounded)
            new_load = float(loads[slot_id]) + duration
            if new_load >= float(best_makespan) - 1e-9:
                continue
            loads[slot_id] = new_load
            work_sequences[slot_id].append(item.job)
            _dfs(idx + 1, loads, float(assigned_sum) + duration)
            work_sequences[slot_id].pop()
            loads[slot_id] = float(loads[slot_id]) - duration
            if time_limit_hit:
                return

    start_loads = [float(x) for x in base_loads]
    _dfs(0, start_loads, 0.0)
    lower_bound = _lower_bound(0, [float(x) for x in base_loads], 0.0)
    elapsed = max(0.0, float(time.monotonic() - start_ts))
    return (
        best_sequences,
        best_loads,
        float(best_makespan),
        float(lower_bound),
        int(exact_nodes),
        bool(time_limit_hit),
        float(elapsed),
    )


def _try_import_gurobi():
    try:
        import gurobipy as gp  # type: ignore
        return gp
    except Exception:
        return None


def _calc_solver_parallel_workers(args: argparse.Namespace, *, running_count: int) -> int:
    max_cap = max(1, int(getattr(args, "scheduler_opt_max_solver_workers", 2)))
    logical, physical = _detect_cpu_counts()
    pressure = _sample_system_pressure()
    cpu_now = float(pressure.get("cpu_percent", float("nan")))
    if _is_finite(cpu_now) and cpu_now >= 88.0:
        return 1
    # Be conservative: active experiment runs already consume the machine.
    assumed_busy_cores = max(2, int(running_count) * 2)
    headroom = max(1, int(physical) - int(assumed_busy_cores) - 1)
    return max(1, min(int(max_cap), int(headroom), int(logical)))


def _gurobi_assignment_schedule(
    *,
    base_loads: List[float],
    jobs: List[PredictedPendingJob],
    time_limit_s: float,
    threads: int,
    mip_gap: float,
) -> Optional[Tuple[Dict[int, List[ScheduledJob]], List[float], float, float, bool, float]]:
    gp = _try_import_gurobi()
    if gp is None or not jobs:
        return None
    try:
        ordered_jobs = sorted(jobs, key=lambda item: (-float(item.predicted_seconds), str(item.job.job_key)))
        m = len(base_loads)
        n = len(ordered_jobs)
        model = gp.Model("transport_scheduler_assignment")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = max(0.1, float(time_limit_s))
        model.Params.Threads = max(1, int(threads))
        model.Params.MIPGap = max(0.0, float(mip_gap))

        x = model.addVars(n, m, vtype=gp.GRB.BINARY, name="x")
        z = model.addVar(lb=max(base_loads) if base_loads else 0.0, vtype=gp.GRB.CONTINUOUS, name="z")
        for j in range(n):
            model.addConstr(gp.quicksum(x[j, s] for s in range(m)) == 1, name=f"assign_{j}")
        for s in range(m):
            model.addConstr(
                float(base_loads[s]) + gp.quicksum(float(ordered_jobs[j].predicted_seconds) * x[j, s] for j in range(n)) <= z,
                name=f"load_{s}",
            )
        model.setObjective(z, gp.GRB.MINIMIZE)
        start_ts = time.monotonic()
        model.optimize()
        elapsed = max(0.0, float(time.monotonic() - start_ts))
        status = int(model.Status)
        if status not in {
            int(gp.GRB.OPTIMAL),
            int(gp.GRB.TIME_LIMIT),
            int(gp.GRB.SUBOPTIMAL),
        }:
            model.dispose()
            return None

        sequences: Dict[int, List[ScheduledJob]] = {idx: [] for idx in range(m)}
        final_loads = [float(x) for x in base_loads]
        for j, item in enumerate(ordered_jobs):
            assigned_slot = None
            for s in range(m):
                try:
                    val = float(x[j, s].X)
                except Exception:
                    val = 0.0
                if val >= 0.5:
                    assigned_slot = s
                    break
            if assigned_slot is None:
                assigned_slot = min(range(m), key=lambda idx: (final_loads[idx], idx))
            sequences[int(assigned_slot)].append(item.job)
            final_loads[int(assigned_slot)] += float(item.predicted_seconds)
        for slot_id in list(sequences.keys()):
            sequences[slot_id].sort(
                key=lambda job: (
                    -float(next((it.predicted_seconds for it in ordered_jobs if it.job is job or it.job.job_key == job.job_key), 0.0)),
                    str(job.job_key),
                )
            )
        makespan = float(model.ObjVal) if model.SolCount > 0 else float(max(final_loads) if final_loads else 0.0)
        lower_bound = float(model.ObjBound) if hasattr(model, "ObjBound") else makespan
        time_limit_hit = status == int(gp.GRB.TIME_LIMIT)
        model.dispose()
        return sequences, final_loads, makespan, lower_bound, bool(time_limit_hit), float(elapsed)
    except Exception:
        return None


def _exact_branch_worker(payload: Tuple[List[float], List[PredictedPendingJob], float, float, int]) -> Tuple[Dict[int, List[ScheduledJob]], List[float], float, float, int, bool, float]:
    base_loads, jobs, time_limit_s, load_round_seconds, fixed_slot = payload
    if not jobs:
        empty = {idx: [] for idx in range(len(base_loads))}
        makespan = max(base_loads) if base_loads else 0.0
        return empty, list(base_loads), float(makespan), float(makespan), 0, False, 0.0
    first = jobs[0]
    if len(jobs) == 1:
        sequences = {idx: [] for idx in range(len(base_loads))}
        loads = [float(x) for x in base_loads]
        sequences[int(fixed_slot)] = [first.job]
        loads[int(fixed_slot)] += float(first.predicted_seconds)
        makespan = max(loads) if loads else 0.0
        return sequences, loads, float(makespan), float(makespan), 1, False, 0.0
    branch_loads = [float(x) for x in base_loads]
    branch_loads[int(fixed_slot)] += float(first.predicted_seconds)
    sequences, loads, makespan, lower_bound, nodes, timed_out, elapsed = _exact_branch_and_bound_schedule(
        base_loads=branch_loads,
        jobs=jobs[1:],
        time_limit_s=float(time_limit_s),
        load_round_seconds=float(load_round_seconds),
    )
    sequences = {idx: list(items) for idx, items in sequences.items()}
    sequences.setdefault(int(fixed_slot), [])
    sequences[int(fixed_slot)] = [first.job] + list(sequences[int(fixed_slot)])
    return sequences, loads, float(makespan), float(lower_bound), int(nodes), bool(timed_out), float(elapsed)


def _parallel_exact_branch_and_bound_schedule(
    *,
    base_loads: List[float],
    jobs: List[PredictedPendingJob],
    time_limit_s: float,
    load_round_seconds: float,
    max_workers: int,
) -> Tuple[Dict[int, List[ScheduledJob]], List[float], float, float, int, bool, float]:
    if max_workers <= 1 or len(jobs) <= 1:
        return _exact_branch_and_bound_schedule(
            base_loads=base_loads,
            jobs=jobs,
            time_limit_s=time_limit_s,
            load_round_seconds=load_round_seconds,
        )
    unique_slots: List[int] = []
    seen = set()
    for slot_id in sorted(range(len(base_loads)), key=lambda idx: (base_loads[idx], idx)):
        rounded = int(round(float(base_loads[slot_id]) / max(1.0, float(load_round_seconds))))
        if rounded in seen:
            continue
        seen.add(rounded)
        unique_slots.append(int(slot_id))
    if len(unique_slots) <= 1:
        return _exact_branch_and_bound_schedule(
            base_loads=base_loads,
            jobs=jobs,
            time_limit_s=time_limit_s,
            load_round_seconds=load_round_seconds,
        )

    effective_workers = max(1, min(int(max_workers), len(unique_slots)))
    batches = max(1, int(math.ceil(len(unique_slots) / float(effective_workers))))
    per_branch_limit = max(0.25, float(time_limit_s) / float(batches))
    payloads = [
        ([float(x) for x in base_loads], list(jobs), float(per_branch_limit), float(load_round_seconds), int(slot_id))
        for slot_id in unique_slots
    ]
    best_result = _exact_branch_and_bound_schedule(
        base_loads=base_loads,
        jobs=jobs,
        time_limit_s=min(float(time_limit_s), 0.75),
        load_round_seconds=load_round_seconds,
    )
    best_makespan = float(best_result[2])
    best_lower = float(best_result[3])
    total_nodes = int(best_result[4])
    any_timeout = bool(best_result[5])
    wall_start = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=effective_workers) as pool:
        futures = [pool.submit(_exact_branch_worker, payload) for payload in payloads]
        for fut in concurrent.futures.as_completed(futures, timeout=max(1.0, float(time_limit_s) * 1.25)):
            try:
                result = fut.result()
            except Exception:
                continue
            total_nodes += int(result[4])
            any_timeout = bool(any_timeout or result[5])
            best_lower = min(float(best_lower), float(result[3]))
            if float(result[2]) < float(best_makespan) - 1e-9:
                best_result = result
                best_makespan = float(result[2])
    elapsed = max(0.0, float(time.monotonic() - wall_start))
    return (
        best_result[0],
        best_result[1],
        float(best_result[2]),
        float(best_lower),
        int(total_nodes),
        bool(any_timeout),
        float(elapsed),
    )


def _build_optimal_scheduler_plan(
    *,
    policy: str,
    pending_by_dist: Dict[str, List[ScheduledJob]],
    slot_pred_loads: List[float],
    model: "AdaptiveDurationModel",
    args: argparse.Namespace,
    running_count: int = 0,
) -> SchedulerPlan:
    pending_jobs = _predict_pending_jobs(pending_by_dist, model)
    base_loads = [float(x) for x in slot_pred_loads]
    empty_sequences = {idx: [] for idx in range(len(base_loads))}
    if not pending_jobs:
        makespan = max(base_loads) if base_loads else 0.0
        return SchedulerPlan(
            policy=str(policy),
            mode="empty",
            total_pending_jobs=0,
            optimized_jobs=0,
            slot_sequences=empty_sequences,
            best_makespan_seconds=float(makespan),
            lower_bound_seconds=float(makespan),
            upper_bound_seconds=float(makespan),
        )

    total_pending = len(pending_jobs)
    exact_cap = max(1, int(getattr(args, "scheduler_opt_max_exact_jobs", 18)))
    frontier_jobs = max(1, int(getattr(args, "scheduler_opt_frontier_jobs", 14)))
    frontier_jobs = min(frontier_jobs, total_pending)
    time_limit_s = max(0.1, float(getattr(args, "scheduler_opt_time_limit_sec", 3.0)))
    round_seconds = max(1.0, float(getattr(args, "scheduler_opt_load_round_sec", 1.0)))
    solver_pref = str(getattr(args, "scheduler_opt_solver", "auto")).strip().lower() or "auto"
    solver_workers = _calc_solver_parallel_workers(args, running_count=running_count)
    gurobi_gap = max(0.0, float(getattr(args, "scheduler_opt_gurobi_mip_gap", 0.0)))

    def _solve_exact_jobs(
        jobs_for_exact: List[PredictedPendingJob],
        *,
        mode_suffix: str,
    ) -> Tuple[Dict[int, List[ScheduledJob]], List[float], float, float, int, bool, float, str, bool]:
        fallback_used = False
        gurobi_res = None
        if solver_pref in {"auto", "gurobi"}:
            try:
                gurobi_res = _gurobi_assignment_schedule(
                    base_loads=base_loads,
                    jobs=jobs_for_exact,
                    time_limit_s=time_limit_s,
                    threads=max(1, int(solver_workers)),
                    mip_gap=gurobi_gap,
                )
            except Exception:
                gurobi_res = None
                fallback_used = True
            if gurobi_res is not None:
                sequences, loads, makespan, lower_bound, timed_out, elapsed = gurobi_res
                return (
                    sequences,
                    loads,
                    float(makespan),
                    float(lower_bound),
                    0,
                    bool(timed_out),
                    float(elapsed),
                    f"gurobi_{mode_suffix}",
                    bool(fallback_used),
                )
            if solver_pref == "gurobi":
                fallback_used = True

        if solver_pref in {"auto", "mp_bnb"} and int(solver_workers) > 1:
            try:
                sequences, loads, makespan, lower_bound, exact_nodes, timed_out, elapsed = _parallel_exact_branch_and_bound_schedule(
                    base_loads=base_loads,
                    jobs=jobs_for_exact,
                    time_limit_s=time_limit_s,
                    load_round_seconds=round_seconds,
                    max_workers=int(solver_workers),
                )
                return (
                    sequences,
                    loads,
                    float(makespan),
                    float(lower_bound),
                    int(exact_nodes),
                    bool(timed_out),
                    float(elapsed),
                    f"mp_{mode_suffix}",
                    bool(fallback_used),
                )
            except Exception:
                fallback_used = True
        elif solver_pref == "mp_bnb":
            fallback_used = True

        sequences, loads, makespan, lower_bound, exact_nodes, timed_out, elapsed = _exact_branch_and_bound_schedule(
            base_loads=base_loads,
            jobs=jobs_for_exact,
            time_limit_s=time_limit_s,
            load_round_seconds=round_seconds,
        )
        return (
            sequences,
            loads,
            float(makespan),
            float(lower_bound),
            int(exact_nodes),
            bool(timed_out),
            float(elapsed),
            f"serial_{mode_suffix}",
            True if fallback_used else solver_pref not in {"serial_bnb", "auto"},
        )

    force_exact = str(policy).strip().lower() == "optimal_exact"
    if total_pending <= exact_cap or force_exact:
        exact_sequences, exact_loads, exact_makespan, lower_bound, exact_nodes, timed_out, exact_elapsed, solve_mode, solve_fallback = _solve_exact_jobs(
            pending_jobs,
            mode_suffix="full",
        )
        return SchedulerPlan(
            policy=str(policy),
            mode=str(solve_mode),
            total_pending_jobs=int(total_pending),
            optimized_jobs=int(total_pending),
            slot_sequences=exact_sequences,
            best_makespan_seconds=float(exact_makespan),
            lower_bound_seconds=float(lower_bound),
            upper_bound_seconds=float(exact_makespan),
            exact_nodes=int(exact_nodes),
            exact_time_seconds=float(exact_elapsed),
            time_limit_hit=bool(timed_out),
            used_fallback=bool(solve_fallback),
        )

    prefix_jobs = pending_jobs[:frontier_jobs]
    tail_jobs = pending_jobs[frontier_jobs:]
    prefix_sequences, prefix_loads, prefix_makespan, lower_bound, exact_nodes, timed_out, exact_elapsed, solve_mode, solve_fallback = _solve_exact_jobs(
        prefix_jobs,
        mode_suffix="frontier",
    )
    tail_sequences, final_loads, final_makespan = _greedy_lpt_schedule(
        base_loads=prefix_loads,
        jobs=tail_jobs,
    )
    merged_sequences = {
        slot_id: list(prefix_sequences.get(slot_id, [])) + list(tail_sequences.get(slot_id, []))
        for slot_id in range(len(base_loads))
    }
    return SchedulerPlan(
        policy=str(policy),
        mode=f"{solve_mode}_greedy_tail",
        total_pending_jobs=int(total_pending),
        optimized_jobs=int(frontier_jobs),
        slot_sequences=merged_sequences,
        best_makespan_seconds=float(final_makespan),
        lower_bound_seconds=float(lower_bound),
        upper_bound_seconds=float(final_makespan),
        exact_nodes=int(exact_nodes),
        exact_time_seconds=float(exact_elapsed),
        time_limit_hit=bool(timed_out),
        used_fallback=bool(tail_jobs) or bool(solve_fallback),
    )


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


def _parse_clock_slots(raw: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for part in [str(x).strip() for x in str(raw or "").split(",") if str(x).strip()]:
        if ":" not in part:
            continue
        hh_raw, mm_raw = part.split(":", 1)
        try:
            hh = int(hh_raw)
            mm = int(mm_raw)
        except Exception:
            continue
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            continue
        key = f"{hh:02d}:{mm:02d}"
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return sorted(out)


def _format_float(x: object, digits: int = 1, suffix: str = "") -> str:
    try:
        val = float(x)
    except Exception:
        return "n/a"
    if not math.isfinite(val):
        return "n/a"
    return f"{val:.{int(digits)}f}{suffix}"


def _format_eta(seconds: float) -> str:
    try:
        sec = float(seconds)
    except Exception:
        return "n/a"
    if not math.isfinite(sec) or sec < 0:
        return "n/a"
    total = int(round(sec))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _trim_mapping(mapping: Dict[str, object], limit: int = 12) -> Dict[str, object]:
    items = list(mapping.items())
    items.sort(key=lambda kv: str(kv[0]))
    return {str(k): v for k, v in items[: max(1, int(limit))]}


def _severity_cn(severity: str) -> str:
    mapping = {
        "INFO": "信息",
        "WARN": "告警",
        "ERROR": "错误",
        "DONE": "完成",
    }
    return mapping.get(str(severity or "INFO").upper(), "信息")


def _planner_mode_cn(mode: str) -> str:
    raw = str(mode or "").strip().lower()
    mapping = {
        "startup": "启动前状态",
        "empty": "无待调度任务",
        "legacy_greedy": "传统贪心派单",
        "serial_full": "串行精确分支定界",
        "mp_full": "多进程精确分支定界",
        "gurobi_full": "Gurobi 精确派单",
        "serial_frontier_greedy_tail": "串行前沿精确 + 贪心尾部",
        "mp_frontier_greedy_tail": "多进程前沿精确 + 贪心尾部",
        "gurobi_frontier_greedy_tail": "Gurobi 前沿精确 + 贪心尾部",
    }
    return mapping.get(raw, raw or "未知")


def _build_scheduler_diagnosis(snapshot: Dict[str, object]) -> List[str]:
    pressure = snapshot.get("pressure", {}) if isinstance(snapshot.get("pressure"), dict) else {}
    planner = snapshot.get("planner", {}) if isinstance(snapshot.get("planner"), dict) else {}
    failed_jobs = int(snapshot.get("failed_jobs", 0) or 0)
    remaining_jobs = int(snapshot.get("remaining_jobs", 0) or 0)
    running_jobs_count = int(snapshot.get("running_jobs_count", 0) or 0)
    requeued_total = int(snapshot.get("requeued_total", 0) or 0)
    cpu = float(pressure.get("cpu_percent", float("nan")))
    mem = float(pressure.get("mem_percent", float("nan")))
    load_pc = float(pressure.get("load_per_core", float("nan")))
    mode = str(planner.get("mode", "")).strip()
    lines: List[str] = []
    if failed_jobs > 0:
        lines.append("存在失败任务，建议优先查看“最终失败任务”和“最近异常”两节。")
    elif remaining_jobs == 0 and running_jobs_count == 0:
        lines.append("当前批次已全部完成，没有剩余任务。")
    else:
        lines.append("当前批次仍在推进，建议结合运行中任务和剩余任务分布判断是否存在长尾。")
    if requeued_total > 0:
        lines.append(f"累计发生 {requeued_total} 次重排/回队，说明部分任务曾触发超时、停滞或锁冲突。")
    if _is_finite(cpu) and cpu >= 88.0:
        lines.append(f"CPU 使用率约 {_format_float(cpu, 1, '%')}，资源压力偏高，可考虑降低 max_workers 或 solver worker 数。")
    elif _is_finite(cpu):
        lines.append(f"CPU 使用率约 {_format_float(cpu, 1, '%')}，当前资源压力总体可控。")
    if _is_finite(mem) and mem >= 88.0:
        lines.append(f"内存占用约 {_format_float(mem, 1, '%')}，建议关注内存抖动和交换区使用。")
    if _is_finite(load_pc) and load_pc >= 1.1:
        lines.append(f"每核负载约 {_format_float(load_pc, 2)}，说明机器已接近满载。")
    if mode:
        lines.append(f"当前调度求解模式为“{_planner_mode_cn(mode)}”。")
    return lines


def _format_scheduler_message(snapshot: Dict[str, object]) -> str:
    pressure = snapshot.get("pressure", {}) if isinstance(snapshot.get("pressure"), dict) else {}
    running_jobs = snapshot.get("running_jobs", []) if isinstance(snapshot.get("running_jobs"), list) else []
    recent_completions = snapshot.get("recent_completions", []) if isinstance(snapshot.get("recent_completions"), list) else []
    recent_issues = snapshot.get("recent_issues", []) if isinstance(snapshot.get("recent_issues"), list) else []
    pending_by_dist = snapshot.get("pending_by_dist", {}) if isinstance(snapshot.get("pending_by_dist"), dict) else {}
    failures = snapshot.get("failures", {}) if isinstance(snapshot.get("failures"), dict) else {}
    status_counter = snapshot.get("status_counter", {}) if isinstance(snapshot.get("status_counter"), dict) else {}
    variants = snapshot.get("variants", []) if isinstance(snapshot.get("variants"), list) else []
    distributions = snapshot.get("distributions", []) if isinstance(snapshot.get("distributions"), list) else []
    request_numbers = snapshot.get("request_numbers", []) if isinstance(snapshot.get("request_numbers"), list) else []
    seeds = snapshot.get("seeds", []) if isinstance(snapshot.get("seeds"), list) else []
    settings = snapshot.get("settings", {}) if isinstance(snapshot.get("settings"), dict) else {}
    paths = snapshot.get("paths", {}) if isinstance(snapshot.get("paths"), dict) else {}
    by_variant = snapshot.get("by_variant", []) if isinstance(snapshot.get("by_variant"), list) else []
    by_dist = snapshot.get("by_dist", []) if isinstance(snapshot.get("by_dist"), list) else []
    slot_pred_loads = snapshot.get("slot_pred_loads", []) if isinstance(snapshot.get("slot_pred_loads"), list) else []
    slot_actual_loads = snapshot.get("slot_actual_loads", []) if isinstance(snapshot.get("slot_actual_loads"), list) else []
    planner = snapshot.get("planner", {}) if isinstance(snapshot.get("planner"), dict) else {}

    started_at_iso = str(snapshot.get("started_at_iso", "")).strip()
    timestamp = str(snapshot.get("timestamp", "")).strip()
    diagnosis = _build_scheduler_diagnosis(snapshot)

    lines = [
        "调度运行简报",
        f"运行目录：{snapshot.get('run_root', '')}",
        f"当前阶段：{snapshot.get('phase', '')}",
        f"当前时间：{timestamp}",
        f"启动时间：{started_at_iso}",
        f"累计运行：{snapshot.get('elapsed_total_human', 'n/a')}",
        "",
        "一、总体进展",
        (
            f"总任务={snapshot.get('total_jobs', 0)}，"
            f"已完成={snapshot.get('completed_jobs', 0)}，"
            f"运行中={snapshot.get('running_jobs_count', 0)}，"
            f"剩余={snapshot.get('remaining_jobs', 0)}，"
            f"延后队列={snapshot.get('deferred_jobs', 0)}，"
            f"失败={snapshot.get('failed_jobs', 0)}，"
            f"成功率={snapshot.get('success_rate', 'n/a')}"
        ),
        (
            f"累计尝试={snapshot.get('completed_attempts', 0)}，"
            f"累计重排={snapshot.get('requeued_total', 0)}，"
            f"当前并发上限={snapshot.get('active_limit', 0)}/{snapshot.get('worker_cap', 0)}，"
            f"自动调度并发={'开启' if int(snapshot.get('auto_workers', 0) or 0) else '关闭'}"
        ),
        "",
        "二、资源与 ETA",
        (
            f"CPU={_format_float(pressure.get('cpu_percent'), 1, '%')}，"
            f"内存={_format_float(pressure.get('mem_percent'), 1, '%')}，"
            f"交换区={_format_float(pressure.get('swap_percent'), 1, '%')}，"
            f"可用内存GB={_format_float(pressure.get('avail_gb'), 2)}，"
            f"每核负载={_format_float(pressure.get('load_per_core'), 2)}"
        ),
        (
            f"剩余 ETA={snapshot.get('eta_remaining', 'n/a')}，"
            f"预计完成时间={snapshot.get('eta_finish_at', 'n/a')}，"
            f"已完成任务平均耗时={snapshot.get('avg_completed_runtime_human', 'n/a')}，"
            f"当前最长运行任务={snapshot.get('max_running_elapsed_human', 'n/a')}"
        ),
        "",
        "三、任务空间",
        (
            f"算法({len(variants)})={json.dumps(variants[:10], ensure_ascii=False)}；"
            f"分布({len(distributions)})={json.dumps(distributions[:12], ensure_ascii=False)}；"
            f"R={json.dumps(request_numbers, ensure_ascii=False)}；"
            f"seed={json.dumps(seeds, ensure_ascii=False)}"
        ),
        "",
        "四、调度器状态",
        f"调度配置：{json.dumps(settings, ensure_ascii=False)}",
        f"当前求解器：{json.dumps(planner, ensure_ascii=False)}",
        f"状态计数：{json.dumps(status_counter, ensure_ascii=False)}",
        f"各分布待运行数：{json.dumps(pending_by_dist, ensure_ascii=False)}",
        f"预测负载数组：{json.dumps(slot_pred_loads, ensure_ascii=False)}",
        f"实际负载数组：{json.dumps(slot_actual_loads, ensure_ascii=False)}",
    ]
    if diagnosis:
        lines.extend(["", "五、诊断提示"])
        lines.extend([f"- {item}" for item in diagnosis])
    if paths:
        lines.extend(["", f"结果路径：{json.dumps(paths, ensure_ascii=False)}"])
    lines.extend(_format_count_table_text("按算法统计", by_variant))
    lines.extend(_format_count_table_text("按分布统计", by_dist))
    if running_jobs:
        lines.append("")
        lines.append("当前运行中的任务：")
        for item in running_jobs[:8]:
            if not isinstance(item, dict):
                continue
            lines.append(
                "  - "
                f"槽位={item.get('slot_id')}，"
                f"任务={item.get('variant')}|{item.get('dist_name')}|R{item.get('request_number')}|S{item.get('seed')}，"
                f"尝试={item.get('attempt')}，"
                f"已运行={item.get('elapsed_human', 'n/a')}，"
                f"预测={item.get('predicted_human', 'n/a')}，"
                f"超时阈值={item.get('timeout_human', 'n/a')}"
            )
    if recent_completions:
        lines.append("")
        lines.append("最近完成任务：")
        for item in recent_completions[:8]:
            if not isinstance(item, dict):
                continue
            lines.append(
                "  - "
                f"{item.get('ts', '')}，"
                f"任务={item.get('variant')}|{item.get('dist_name')}|R{item.get('request_number')}|S{item.get('seed')}，"
                f"尝试={item.get('attempt')}，"
                f"状态={item.get('status')}，"
                f"耗时={item.get('elapsed_human', 'n/a')}，"
                f"预测={item.get('predicted_human', 'n/a')}，"
                f"实际/预测比={item.get('ratio', 'n/a')}"
            )
    if recent_issues:
        lines.append("")
        lines.append("最近异常：")
        for item in recent_issues[:8]:
            if not isinstance(item, dict):
                continue
            lines.append(
                "  - "
                f"{item.get('ts', '')}，类型={item.get('kind', '')}，任务={item.get('job_key', '')}，"
                f"尝试={item.get('attempt', '')}，详情={item.get('detail', '')}"
            )
    if failures:
        lines.append("")
        lines.append("最终失败任务：")
        for key, detail in list(failures.items())[:8]:
            lines.append(f"  - 任务={key}，失败详情={detail}")
    return "\n".join(lines)


def _format_count_table_text(title: str, rows: List[Dict[str, object]], *, label_key: str = "name") -> List[str]:
    if not rows:
        return []
    lines = ["", f"{title}:"]
    for row in rows[:12]:
        lines.append(
            "  - "
            f"{row.get(label_key, '')}: "
            f"总数={row.get('total', 0)} "
            f"成功={row.get('ok', 0)} "
            f"失败={row.get('failed', 0)} "
            f"运行中={row.get('running', 0)} "
            f"待运行={row.get('pending', 0)} "
            f"延后={row.get('deferred', 0)} "
            f"重排={row.get('requeued', 0)}"
        )
    return lines


def _html_table(headers: List[str], rows: List[List[object]]) -> str:
    if not rows:
        return "<p><em>n/a</em></p>"
    head = "".join(
        f"<th style='text-align:left;padding:6px 8px;border-bottom:1px solid #d8d8d8;background:#f5f5f5'>{html.escape(str(h))}</th>"
        for h in headers
    )
    body_rows = []
    for row in rows:
        cols = "".join(
            f"<td style='padding:6px 8px;border-bottom:1px solid #eeeeee;vertical-align:top'>{html.escape(str(v))}</td>"
            for v in row
        )
        body_rows.append(f"<tr>{cols}</tr>")
    return (
        "<table style='border-collapse:collapse;font-family:Segoe UI,Arial,sans-serif;font-size:13px;margin:6px 0 16px 0;min-width:680px'>"
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
    )


def _format_scheduler_html(snapshot: Dict[str, object]) -> str:
    severity = str(snapshot.get("notify_severity", "INFO")).upper()
    if severity == "ERROR":
        badge_bg = "#fce8e6"
        badge_fg = "#b3261e"
    elif severity == "WARN":
        badge_bg = "#fef7e0"
        badge_fg = "#8d5f00"
    elif severity == "DONE":
        badge_bg = "#e6f4ea"
        badge_fg = "#188038"
    else:
        badge_bg = "#e8f0fe"
        badge_fg = "#1967d2"
    pressure = snapshot.get("pressure", {}) if isinstance(snapshot.get("pressure"), dict) else {}
    settings = snapshot.get("settings", {}) if isinstance(snapshot.get("settings"), dict) else {}
    paths = snapshot.get("paths", {}) if isinstance(snapshot.get("paths"), dict) else {}
    running_jobs = snapshot.get("running_jobs", []) if isinstance(snapshot.get("running_jobs"), list) else []
    recent_completions = snapshot.get("recent_completions", []) if isinstance(snapshot.get("recent_completions"), list) else []
    recent_issues = snapshot.get("recent_issues", []) if isinstance(snapshot.get("recent_issues"), list) else []
    failures = snapshot.get("failures", {}) if isinstance(snapshot.get("failures"), dict) else {}
    by_variant = snapshot.get("by_variant", []) if isinstance(snapshot.get("by_variant"), list) else []
    by_dist = snapshot.get("by_dist", []) if isinstance(snapshot.get("by_dist"), list) else []
    planner = snapshot.get("planner", {}) if isinstance(snapshot.get("planner"), dict) else {}
    diagnosis = _build_scheduler_diagnosis(snapshot)

    kpi_rows = [
        ["运行目录", snapshot.get("run_root", "")],
        ["当前阶段", snapshot.get("phase", "")],
        ["当前时间", snapshot.get("timestamp", "")],
        ["启动时间", snapshot.get("started_at_iso", "")],
        ["累计运行", snapshot.get("elapsed_total_human", "n/a")],
        ["总任务数", snapshot.get("total_jobs", 0)],
        ["已完成", snapshot.get("completed_jobs", 0)],
        ["运行中", snapshot.get("running_jobs_count", 0)],
        ["剩余任务", snapshot.get("remaining_jobs", 0)],
        ["延后队列", snapshot.get("deferred_jobs", 0)],
        ["失败任务", snapshot.get("failed_jobs", 0)],
        ["成功率", snapshot.get("success_rate", "n/a")],
        ["累计尝试", snapshot.get("completed_attempts", 0)],
        ["累计重排", snapshot.get("requeued_total", 0)],
        ["当前并发", f"{snapshot.get('active_limit', 0)}/{snapshot.get('worker_cap', 0)}"],
        ["剩余 ETA", snapshot.get("eta_remaining", "n/a")],
        ["预计完成时间", snapshot.get("eta_finish_at", "n/a")],
        ["平均完成耗时", snapshot.get("avg_completed_runtime_human", "n/a")],
        ["当前最长运行", snapshot.get("max_running_elapsed_human", "n/a")],
        ["CPU", _format_float(pressure.get("cpu_percent"), 1, "%")],
        ["内存", _format_float(pressure.get("mem_percent"), 1, "%")],
        ["交换区", _format_float(pressure.get("swap_percent"), 1, "%")],
        ["可用内存GB", _format_float(pressure.get("avail_gb"), 2)],
        ["每核负载", _format_float(pressure.get("load_per_core"), 2)],
    ]
    settings_rows = [[k, v] for k, v in settings.items()]
    planner_rows = [[k, v] for k, v in planner.items()]
    paths_rows = [[k, v] for k, v in paths.items()]
    running_rows = [
        [
            item.get("slot_id", ""),
            f"{item.get('variant', '')}|{item.get('dist_name', '')}|R{item.get('request_number', '')}|S{item.get('seed', '')}",
            item.get("attempt", ""),
            item.get("elapsed_human", "n/a"),
            item.get("predicted_human", "n/a"),
            item.get("timeout_human", "n/a"),
        ]
        for item in running_jobs[:10]
        if isinstance(item, dict)
    ]
    completion_rows = [
        [
            item.get("ts", ""),
            f"{item.get('variant', '')}|{item.get('dist_name', '')}|R{item.get('request_number', '')}|S{item.get('seed', '')}",
            item.get("status", ""),
            item.get("attempt", ""),
            item.get("elapsed_human", "n/a"),
            item.get("predicted_human", "n/a"),
            item.get("ratio", "n/a"),
        ]
        for item in recent_completions[:10]
        if isinstance(item, dict)
    ]
    issue_rows = [
        [item.get("ts", ""), item.get("kind", ""), item.get("job_key", ""), item.get("attempt", ""), item.get("detail", "")]
        for item in recent_issues[:10]
        if isinstance(item, dict)
    ]
    failure_rows = [[k, v] for k, v in list(failures.items())[:10]]
    by_variant_rows = [
        [row.get("name", ""), row.get("total", 0), row.get("ok", 0), row.get("failed", 0), row.get("running", 0), row.get("pending", 0), row.get("deferred", 0), row.get("requeued", 0)]
        for row in by_variant[:12]
        if isinstance(row, dict)
    ]
    by_dist_rows = [
        [row.get("name", ""), row.get("total", 0), row.get("ok", 0), row.get("failed", 0), row.get("running", 0), row.get("pending", 0), row.get("deferred", 0), row.get("requeued", 0)]
        for row in by_dist[:16]
        if isinstance(row, dict)
    ]

    return (
        "<html><body style='font-family:Segoe UI,Arial,sans-serif;color:#202124;line-height:1.45'>"
        f"<div style='margin:0 0 10px 0'><span style='display:inline-block;padding:4px 10px;border-radius:999px;"
        f"background:{badge_bg};color:{badge_fg};font-weight:700;font-size:12px;letter-spacing:0.2px'>{html.escape(_severity_cn(severity))}</span></div>"
        f"<h2 style='margin:0 0 8px 0'>服务器调度运行简报</h2>"
        f"<p style='margin:0 0 14px 0;color:#5f6368'>本邮件汇总了 <strong>{html.escape(str(snapshot.get('run_root', '')))}</strong> 的当前状态，便于远程判断运行是否正常。</p>"
        + (
            "<div style='margin:0 0 16px 0;padding:10px 14px;border-left:4px solid #8ab4f8;background:#f8fbff'>"
            "<div style='font-weight:700;margin:0 0 6px 0'>诊断提示</div>"
            + "".join(f"<div style='margin:3px 0'>- {html.escape(str(item))}</div>" for item in diagnosis)
            + "</div>"
            if diagnosis
            else ""
        )
        + "<h3 style='margin:14px 0 6px 0'>一、总体概览</h3>"
        + _html_table(["项目", "值"], kpi_rows)
        + "<h3 style='margin:14px 0 6px 0'>二、调度配置</h3>"
        + _html_table(["配置项", "值"], settings_rows)
        + "<h3 style='margin:14px 0 6px 0'>三、求解器状态</h3>"
        + _html_table(["求解项", "值"], planner_rows)
        + "<h3 style='margin:14px 0 6px 0'>四、按算法统计</h3>"
        + _html_table(["算法", "总数", "成功", "失败", "运行中", "待运行", "延后", "重排"], by_variant_rows)
        + "<h3 style='margin:14px 0 6px 0'>五、按分布统计</h3>"
        + _html_table(["分布", "总数", "成功", "失败", "运行中", "待运行", "延后", "重排"], by_dist_rows)
        + "<h3 style='margin:14px 0 6px 0'>六、当前运行任务</h3>"
        + _html_table(["槽位", "任务", "尝试", "已运行", "预测", "超时阈值"], running_rows)
        + "<h3 style='margin:14px 0 6px 0'>七、最近完成任务</h3>"
        + _html_table(["时间", "任务", "状态", "尝试", "耗时", "预测", "实际/预测比"], completion_rows)
        + "<h3 style='margin:14px 0 6px 0'>八、最近异常</h3>"
        + _html_table(["时间", "类型", "任务", "尝试", "详情"], issue_rows)
        + "<h3 style='margin:14px 0 6px 0'>九、最终失败任务</h3>"
        + _html_table(["任务", "详情"], failure_rows)
        + "<h3 style='margin:14px 0 6px 0'>十、结果路径</h3>"
        + _html_table(["路径键", "值"], paths_rows)
        + "</body></html>"
    )


class SchedulerNotifier:
    def __init__(self, *, notifier: NotificationManager, run_root: Path, args: argparse.Namespace) -> None:
        self.notifier = notifier
        self.run_root = run_root
        self.enabled = bool(notifier.enabled) and bool(getattr(args, "notify_scheduler", True))
        self.clock_slots = _parse_clock_slots(getattr(args, "notify_schedule_times", ""))
        self.batch_size = max(0, int(getattr(args, "notify_batch_size", 0) or 0))
        self.notify_on_start = bool(getattr(args, "notify_on_start", True))
        self.notify_on_requeue = bool(getattr(args, "notify_on_requeue", True))
        self.notify_on_finish = bool(getattr(args, "notify_on_finish", True))
        self.live_status_interval_s = max(5.0, float(getattr(args, "notify_live_status_interval_s", 30.0) or 30.0))
        self.state_path = (
            Path(str(getattr(args, "notify_state_path", "")).strip()).resolve()
            if str(getattr(args, "notify_state_path", "")).strip()
            else (run_root / "adaptive_scheduler_notify_state.json").resolve()
        )
        self.live_status_path = (
            Path(str(getattr(args, "live_status_path", "")).strip()).resolve()
            if str(getattr(args, "live_status_path", "")).strip()
            else (run_root / "adaptive_scheduler_live_status.json").resolve()
        )
        raw_state = _load_json(self.state_path)
        sent_slots = raw_state.get("sent_clock_slots", [])
        self.sent_clock_slots = set(str(x).strip() for x in sent_slots if str(x).strip())
        self.last_batch_bucket = int(raw_state.get("last_batch_bucket", 0) or 0)
        self.start_sent = bool(raw_state.get("start_sent", False))
        self.finish_sent = bool(raw_state.get("finish_sent", False))
        self.last_live_write_ts = 0.0

    def _decorate_title(self, severity: str, title: str) -> str:
        return f"[{_severity_cn(severity)}] {title}"

    def _save_state(self) -> None:
        try:
            keep_slots = sorted(self.sent_clock_slots)[-64:]
            _save_json(
                self.state_path,
                {
                    "sent_clock_slots": keep_slots,
                    "last_batch_bucket": int(self.last_batch_bucket),
                    "start_sent": bool(self.start_sent),
                    "finish_sent": bool(self.finish_sent),
                    "ts": time.time(),
                },
            )
        except Exception:
            pass

    def update_live_status(self, snapshot: Dict[str, object], *, force: bool = False) -> None:
        now = time.monotonic()
        if (not force) and (now - self.last_live_write_ts < self.live_status_interval_s):
            return
        self.last_live_write_ts = now
        try:
            _save_json(self.live_status_path, snapshot)
        except Exception:
            pass

    def notify_start(self, snapshot: Dict[str, object]) -> None:
        if not self.enabled or not self.notify_on_start or self.start_sent:
            return
        payload = dict(snapshot)
        payload["notify_severity"] = "INFO"
        title = self._decorate_title(
            "INFO",
            f"调度启动：{Path(str(snapshot.get('run_root', self.run_root))).name}，"
            f"任务数={snapshot.get('total_jobs', 0)}，并发槽位={snapshot.get('worker_cap', 0)}",
        )
        self.notifier.send(
            "scheduler_started",
            title,
            _format_scheduler_message(payload),
            payload=payload,
            html_message=_format_scheduler_html(payload),
        )
        self.start_sent = True
        self._save_state()

    def notify_clock_ticks(self, snapshot: Dict[str, object]) -> None:
        if not self.enabled or not self.clock_slots:
            return
        now = datetime.datetime.now()
        today = now.date().isoformat()
        for slot in self.clock_slots:
            hh, mm = slot.split(":", 1)
            due = now.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
            key = f"{today}T{slot}"
            if now >= due and key not in self.sent_clock_slots:
                payload = dict(snapshot)
                payload["scheduled_slot"] = slot
                payload["notify_severity"] = "WARN" if int(payload.get("failed_jobs", 0) or 0) > 0 else "INFO"
                title = self._decorate_title(
                    payload["notify_severity"],
                    f"定时状态播报 {slot}："
                    f"已完成={snapshot.get('completed_jobs', 0)}/{snapshot.get('total_jobs', 0)}，"
                    f"失败={snapshot.get('failed_jobs', 0)}",
                )
                self.notifier.send(
                    "scheduler_scheduled_status",
                    title,
                    _format_scheduler_message(payload),
                    payload=payload,
                    html_message=_format_scheduler_html(payload),
                )
                self.sent_clock_slots.add(key)
                self._save_state()

    def notify_batch(self, snapshot: Dict[str, object]) -> None:
        if not self.enabled or self.batch_size <= 0:
            return
        completed = int(snapshot.get("completed_jobs", 0) or 0)
        bucket = completed // self.batch_size
        if bucket <= 0 or bucket <= int(self.last_batch_bucket):
            return
        payload = dict(snapshot)
        payload["batch_bucket"] = int(bucket)
        payload["batch_size"] = int(self.batch_size)
        payload["notify_severity"] = "WARN" if int(payload.get("failed_jobs", 0) or 0) > 0 else "INFO"
        title = self._decorate_title(
            payload["notify_severity"],
            f"批次进度更新：已完成={completed}/{snapshot.get('total_jobs', 0)}，"
            f"失败={snapshot.get('failed_jobs', 0)}",
        )
        self.notifier.send(
            "scheduler_batch_progress",
            title,
            _format_scheduler_message(payload),
            payload=payload,
            html_message=_format_scheduler_html(payload),
        )
        self.last_batch_bucket = int(bucket)
        self._save_state()

    def notify_requeue(
        self,
        *,
        snapshot: Dict[str, object],
        job: ScheduledJob,
        attempt: int,
        max_attempts: int,
        detail: str,
        delay_s: float,
    ) -> None:
        if not self.enabled or not self.notify_on_requeue:
            return
        payload = dict(snapshot)
        payload.update(
            {
                "job_key": job.job_key,
                "attempt": int(attempt),
                "max_attempts": int(max_attempts),
                "detail": str(detail),
                "requeue_delay_s": float(delay_s),
                "variant": str(job.variant.raw),
                "dist_name": str(job.dist_name),
                "request_number": int(job.request_number),
                "seed": int(job.seed),
                "notify_severity": "WARN",
            }
        )
        title = self._decorate_title(
            "WARN",
            f"任务重排：{job.variant.raw}|{job.dist_name}|R{job.request_number}|S{job.seed}，"
            f"尝试={attempt}/{max_attempts}",
        )
        message = _format_scheduler_message(payload) + "\n\n" + f"重排原因={detail}\n延迟重试(秒)={delay_s:.1f}"
        self.notifier.send(
            "scheduler_job_requeued",
            title,
            message,
            payload=payload,
            html_message=_format_scheduler_html(payload),
        )

    def notify_finish(self, snapshot: Dict[str, object]) -> None:
        if not self.enabled or not self.notify_on_finish or self.finish_sent:
            return
        payload = dict(snapshot)
        severity = "ERROR" if int(payload.get("failed_jobs", 0) or 0) > 0 else "DONE"
        payload["notify_severity"] = severity
        title = self._decorate_title(
            severity,
            f"调度结束：{Path(str(snapshot.get('run_root', self.run_root))).name}，"
            f"已完成={snapshot.get('completed_jobs', 0)}/{snapshot.get('total_jobs', 0)}，"
            f"失败={snapshot.get('failed_jobs', 0)}",
        )
        self.notifier.send(
            "scheduler_finished",
            title,
            _format_scheduler_message(payload),
            payload=payload,
            html_message=_format_scheduler_html(payload),
        )
        self.finish_sent = True
        self._save_state()


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
        "--scheduler-policy",
        type=str,
        default="optimal_hybrid",
        choices=["optimal_hybrid", "optimal_exact", "adaptive_lpt", "fifo"],
    )
    parser.add_argument("--algo-coef-init", type=str, default="", help="e.g. PPO=1.0,PPO_NEW=1.2,NOVA_EDRL=2.0")
    parser.add_argument("--dist-coef-init", type=str, default="", help="e.g. O_10_90=1.3,F2_10_60=1.1")
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
        help="exact planner backend: auto prefers Gurobi, then multiprocessing B&B, then serial B&B",
    )
    parser.add_argument(
        "--scheduler-opt-max-solver-workers",
        type=int,
        default=2,
        help="upper bound on solver-side parallel workers when mp_bnb/Gurobi is used",
    )
    parser.add_argument(
        "--scheduler-opt-gurobi-mip-gap",
        type=float,
        default=0.0,
        help="optional Gurobi MIP gap tolerance for assignment scheduler",
    )
    parser.add_argument("--coef-state-path", type=str, default="")
    parser.add_argument("--template-state-path", type=str, default="")
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
    template_state_path = Path(str(args.template_state_path)).resolve() if str(args.template_state_path).strip() else None
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

    persist_paths: List[Path] = [state_path]
    if template_state_path is not None and template_state_path.resolve() != state_path.resolve():
        persist_paths.append(template_state_path)

    def _persist_model_state() -> None:
        payload = model.to_dict()
        for target_path in persist_paths:
            _save_json(target_path, payload)

    _persist_model_state()

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
    scheduler_notifier = SchedulerNotifier(notifier=notifier, run_root=run_root, args=args)
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
    print(
        f"[adaptive] scheduler policy={str(args.scheduler_policy)} "
        f"solver={str(args.scheduler_opt_solver)} "
        f"solver_workers_cap={int(args.scheduler_opt_max_solver_workers)} "
        f"opt_max_exact_jobs={int(args.scheduler_opt_max_exact_jobs)} "
        f"opt_frontier_jobs={int(args.scheduler_opt_frontier_jobs)} "
        f"opt_time_limit={float(args.scheduler_opt_time_limit_sec):.2f}s "
        f"opt_round={float(args.scheduler_opt_load_round_sec):.1f}s "
        f"gurobi_gap={float(args.scheduler_opt_gurobi_mip_gap):.4f}"
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
    requeued_total = 0
    job_attempts: Dict[str, int] = {}
    recent_completions: List[Dict[str, object]] = []
    recent_issues: List[Dict[str, object]] = []
    run_started = time.monotonic()
    run_started_wall = datetime.datetime.now()
    pressure_last = _sample_system_pressure()
    total_by_variant: Dict[str, int] = {}
    total_by_dist: Dict[str, int] = {}
    ok_by_variant: Dict[str, int] = {}
    ok_by_dist: Dict[str, int] = {}
    failed_by_variant: Dict[str, int] = {}
    failed_by_dist: Dict[str, int] = {}
    requeued_by_variant: Dict[str, int] = {}
    requeued_by_dist: Dict[str, int] = {}
    for job in all_jobs:
        total_by_variant[str(job.variant.raw)] = int(total_by_variant.get(str(job.variant.raw), 0)) + 1
        total_by_dist[str(job.dist_name)] = int(total_by_dist.get(str(job.dist_name), 0)) + 1
    high_streak = 0
    low_streak = 0
    last_adjust_check = 0.0
    last_adjust_event = 0.0
    latest_plan = SchedulerPlan(
        policy=str(args.scheduler_policy),
        mode="startup",
        total_pending_jobs=int(len(all_jobs)),
        optimized_jobs=0,
        slot_sequences={idx: [] for idx in range(worker_cap)},
        best_makespan_seconds=max(slot_pred_loads) if slot_pred_loads else 0.0,
        lower_bound_seconds=max(slot_pred_loads) if slot_pred_loads else 0.0,
        upper_bound_seconds=max(slot_pred_loads) if slot_pred_loads else 0.0,
    )

    def _remaining_jobs_count() -> int:
        return sum(len(v) for v in pending_by_dist.values()) + len(deferred_jobs)

    def _record_issue(kind: str, *, job: ScheduledJob, attempt: int, detail: str) -> None:
        recent_issues.append(
            {
                "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                "kind": str(kind),
                "job_key": str(job.job_key),
                "attempt": int(attempt),
                "detail": str(detail)[:500],
            }
        )
        if len(recent_issues) > 20:
            del recent_issues[:-20]

    def _estimate_remaining_seconds() -> float:
        total = 0.0
        now_local = time.monotonic()
        for dispatch in running.values():
            elapsed_running = max(0.0, float(now_local - dispatch.started_at))
            total += max(0.0, float(dispatch.predicted_seconds) - elapsed_running)
        for items in pending_by_dist.values():
            for job in items:
                total += float(
                    model.predict(
                        algo_key=job.algorithm_key,
                        dist_name=job.dist_name,
                        request_number=int(job.request_number),
                    )
                )
        for _, job in deferred_jobs:
            total += float(
                model.predict(
                    algo_key=job.algorithm_key,
                    dist_name=job.dist_name,
                    request_number=int(job.request_number),
                )
            )
        return float(total / max(1, int(active_limit)))

    def _build_snapshot(phase: str) -> Dict[str, object]:
        now_local = time.monotonic()
        running_jobs: List[Dict[str, object]] = []
        for dispatch in sorted(running.values(), key=lambda item: (item.slot_id, item.started_at)):
            elapsed_running = max(0.0, float(now_local - dispatch.started_at))
            running_jobs.append(
                {
                    "slot_id": int(dispatch.slot_id),
                    "variant": str(dispatch.job.variant.raw),
                    "dist_name": str(dispatch.job.dist_name),
                    "request_number": int(dispatch.job.request_number),
                    "seed": int(dispatch.job.seed),
                    "attempt": int(dispatch.attempt),
                    "predicted_seconds": float(dispatch.predicted_seconds),
                    "predicted_human": _format_eta(float(dispatch.predicted_seconds)),
                    "elapsed_seconds": float(elapsed_running),
                    "elapsed_human": _format_eta(float(elapsed_running)),
                    "timeout_limit_s": float(dispatch.timeout_limit_s),
                    "timeout_human": _format_eta(float(dispatch.timeout_limit_s)),
                }
            )
        pending_counts = {str(k): int(len(v)) for k, v in pending_by_dist.items() if len(v) > 0}
        pending_by_variant: Dict[str, int] = {}
        deferred_by_variant: Dict[str, int] = {}
        running_by_variant: Dict[str, int] = {}
        running_by_dist: Dict[str, int] = {}
        for items in pending_by_dist.values():
            for job in items:
                pending_by_variant[str(job.variant.raw)] = int(pending_by_variant.get(str(job.variant.raw), 0)) + 1
        deferred_by_dist: Dict[str, int] = {}
        for _, job in deferred_jobs:
            deferred_by_variant[str(job.variant.raw)] = int(deferred_by_variant.get(str(job.variant.raw), 0)) + 1
            deferred_by_dist[str(job.dist_name)] = int(deferred_by_dist.get(str(job.dist_name), 0)) + 1
        for dispatch in running.values():
            running_by_variant[str(dispatch.job.variant.raw)] = int(running_by_variant.get(str(dispatch.job.variant.raw), 0)) + 1
            running_by_dist[str(dispatch.job.dist_name)] = int(running_by_dist.get(str(dispatch.job.dist_name), 0)) + 1
        by_variant_rows: List[Dict[str, object]] = []
        for name in sorted(total_by_variant.keys()):
            by_variant_rows.append(
                {
                    "name": name,
                    "total": int(total_by_variant.get(name, 0)),
                    "ok": int(ok_by_variant.get(name, 0)),
                    "failed": int(failed_by_variant.get(name, 0)),
                    "running": int(running_by_variant.get(name, 0)),
                    "pending": int(pending_by_variant.get(name, 0)),
                    "deferred": int(deferred_by_variant.get(name, 0)),
                    "requeued": int(requeued_by_variant.get(name, 0)),
                }
            )
        by_dist_rows: List[Dict[str, object]] = []
        for name in sorted(total_by_dist.keys()):
            by_dist_rows.append(
                {
                    "name": name,
                    "total": int(total_by_dist.get(name, 0)),
                    "ok": int(ok_by_dist.get(name, 0)),
                    "failed": int(failed_by_dist.get(name, 0)),
                    "running": int(running_by_dist.get(name, 0)),
                    "pending": int(pending_counts.get(name, 0)),
                    "deferred": int(deferred_by_dist.get(name, 0)),
                    "requeued": int(requeued_by_dist.get(name, 0)),
                }
            )
        eta_remaining_s = _estimate_remaining_seconds()
        completed_elapsed = [float(item.get("elapsed_seconds", 0.0)) for item in recent_completions if isinstance(item, dict)]
        avg_completed_runtime = (sum(completed_elapsed) / len(completed_elapsed)) if completed_elapsed else float("nan")
        max_running_elapsed = max([float(item.get("elapsed_seconds", 0.0)) for item in running_jobs], default=float("nan"))
        total_elapsed_s = max(0.0, float(now_local - run_started))
        eta_finish_at = "n/a"
        if math.isfinite(eta_remaining_s):
            eta_finish_at = (datetime.datetime.now() + datetime.timedelta(seconds=float(eta_remaining_s))).isoformat(timespec="seconds")
        success_rate = "n/a"
        denom = int(completed_jobs_final + len(failures))
        if denom > 0:
            success_rate = f"{(100.0 * float(completed_jobs_final) / float(denom)):.1f}%"
        return {
            "run_root": str(run_root),
            "phase": str(phase),
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "started_at_iso": run_started_wall.isoformat(timespec="seconds"),
            "elapsed_total_seconds": float(total_elapsed_s),
            "elapsed_total_human": _format_eta(float(total_elapsed_s)),
            "total_jobs": int(len(all_jobs)),
            "completed_jobs": int(completed_jobs_final),
            "completed_attempts": int(completed_attempts),
            "running_jobs_count": int(len(running)),
            "remaining_jobs": int(_remaining_jobs_count()),
            "deferred_jobs": int(len(deferred_jobs)),
            "failed_jobs": int(len(failures)),
            "success_rate": success_rate,
            "status_counter": dict(status_counter),
            "requeued_total": int(requeued_total),
            "worker_cap": int(worker_cap),
            "active_limit": int(active_limit),
            "min_workers": int(min_workers),
            "auto_workers": int(bool(args.auto_workers)),
            "pressure": dict(pressure_last),
            "slot_pred_loads": [float(x) for x in slot_pred_loads],
            "slot_actual_loads": [float(x) for x in slot_actual_loads],
            "pending_by_dist": _trim_mapping(pending_counts, limit=16),
            "running_jobs": running_jobs[:12],
            "recent_completions": list(recent_completions[-10:]),
            "recent_issues": list(recent_issues[-10:]),
            "failures": _trim_mapping(failures, limit=10),
            "eta_remaining_seconds": float(eta_remaining_s),
            "eta_remaining": _format_eta(float(eta_remaining_s)),
            "eta_finish_at": eta_finish_at,
            "avg_completed_runtime_seconds": float(avg_completed_runtime) if math.isfinite(avg_completed_runtime) else float("nan"),
            "avg_completed_runtime_human": _format_eta(float(avg_completed_runtime)) if math.isfinite(avg_completed_runtime) else "n/a",
            "max_running_elapsed_seconds": float(max_running_elapsed) if math.isfinite(max_running_elapsed) else float("nan"),
            "max_running_elapsed_human": _format_eta(float(max_running_elapsed)) if math.isfinite(max_running_elapsed) else "n/a",
            "variants": [str(v.raw) for v in variants],
            "distributions": list(dists),
            "request_numbers": [int(x) for x in requests],
            "seeds": [int(x) for x in seeds],
            "by_variant": by_variant_rows,
            "by_dist": by_dist_rows,
            "planner": latest_plan.summary_dict(),
            "settings": {
                "scheduler_policy": str(args.scheduler_policy),
                "scheduler_opt_solver": str(args.scheduler_opt_solver),
                "scheduler_opt_max_solver_workers": int(args.scheduler_opt_max_solver_workers),
                "scheduler_opt_gurobi_mip_gap": float(args.scheduler_opt_gurobi_mip_gap),
                "run_baseline": bool(args.run_baseline),
                "run_metrics": bool(args.run_metrics),
                "run_plots": bool(args.run_plots),
                "cleanup_after_run": bool(args.cleanup_after_run),
                "resume_existing": bool(args.resume_existing),
                "skip_completed": bool(args.skip_completed),
                "reschedule_timeout_jobs": bool(args.reschedule_timeout_jobs),
                "reschedule_max_attempts": int(reschedule_max_attempts),
                "dispatch_kill_on_timeout": bool(args.dispatch_kill_on_timeout),
                "dispatch_timeout_factor": float(args.dispatch_timeout_factor),
            },
            "paths": {
                "events_csv": str(events_csv),
                "summary_json": str(summary_json),
                "coef_state_path": str(state_path),
                "template_state_path": str(template_state_path) if template_state_path is not None else "",
                "live_status_path": str(scheduler_notifier.live_status_path),
            },
        }

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
    scheduler_notifier.update_live_status(_build_snapshot("startup"), force=True)
    scheduler_notifier.notify_start(_build_snapshot("startup"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_cap) as pool:
        while _remaining_jobs_count() > 0 or running:
            scheduler_notifier.update_live_status(_build_snapshot("loop"))
            scheduler_notifier.notify_clock_ticks(_build_snapshot("loop"))
            scheduler_notifier.notify_batch(_build_snapshot("loop"))
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

            current_policy = str(args.scheduler_policy).strip().lower()
            if free_slots and _remaining_jobs_count() > 0 and len(running) < int(active_limit):
                if current_policy in {"optimal_hybrid", "optimal_exact"}:
                    latest_plan = _build_optimal_scheduler_plan(
                        policy=current_policy,
                        pending_by_dist=pending_by_dist,
                        slot_pred_loads=slot_pred_loads,
                        model=model,
                        args=args,
                        running_count=len(running),
                    )
                else:
                    latest_plan = SchedulerPlan(
                        policy=str(args.scheduler_policy),
                        mode="legacy_greedy",
                        total_pending_jobs=int(_remaining_jobs_count()),
                        optimized_jobs=0,
                        slot_sequences={idx: [] for idx in range(worker_cap)},
                        best_makespan_seconds=max(slot_pred_loads) if slot_pred_loads else 0.0,
                        lower_bound_seconds=max(slot_pred_loads) if slot_pred_loads else 0.0,
                        upper_bound_seconds=max(slot_pred_loads) if slot_pred_loads else 0.0,
                    )

            while free_slots and _remaining_jobs_count() > 0 and len(running) < int(active_limit):
                if current_policy in {"optimal_hybrid", "optimal_exact"}:
                    dispatchable_slots = [slot_id for slot_id in free_slots if latest_plan.slot_sequences.get(slot_id)]
                    if not dispatchable_slots:
                        break
                    slot_id = min(dispatchable_slots, key=lambda s: slot_pred_loads[s])
                    planned_job = latest_plan.slot_sequences[slot_id].pop(0)
                    job = _pop_specific_pending_job(pending_by_dist, planned_job)
                    if job is None:
                        continue
                else:
                    slot_id = min(free_slots, key=lambda s: slot_pred_loads[s])
                    job = _pick_next_job()
                    if job is None:
                        break
                free_slots.remove(slot_id)
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
                    ok_by_variant[str(job.variant.raw)] = int(ok_by_variant.get(str(job.variant.raw), 0)) + 1
                    ok_by_dist[str(job.dist_name)] = int(ok_by_dist.get(str(job.dist_name), 0)) + 1
                elif result.status in {"dry_run", "skipped_completed"}:
                    failures.pop(job.job_key, None)
                    completed_jobs_final += 1
                    ok_by_variant[str(job.variant.raw)] = int(ok_by_variant.get(str(job.variant.raw), 0)) + 1
                    ok_by_dist[str(job.dist_name)] = int(ok_by_dist.get(str(job.dist_name), 0)) + 1
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
                        requeued_total += 1
                        requeued_by_variant[str(job.variant.raw)] = int(requeued_by_variant.get(str(job.variant.raw), 0)) + 1
                        requeued_by_dist[str(job.dist_name)] = int(requeued_by_dist.get(str(job.dist_name), 0)) + 1
                        deferred_jobs.append((time.monotonic() + float(delay), job))
                        _record_issue(
                            "requeued",
                            job=job,
                            attempt=attempt_no,
                            detail=f"{reason_text} -> {requeue_reason} delay={delay:.1f}s",
                        )
                        print(
                            f"[adaptive][reschedule] {job.job_key} attempt={attempt_no}/"
                            f"{int(reschedule_max_attempts)} reason={reason_text} delay={delay:.1f}s"
                        )
                        scheduler_notifier.notify_requeue(
                            snapshot=_build_snapshot("requeue"),
                            job=job,
                            attempt=attempt_no,
                            max_attempts=int(reschedule_max_attempts),
                            detail=str(reason_text),
                            delay_s=float(delay),
                        )
                    else:
                        failures[job.job_key] = f"{result.status}: {result.error[:400]}"
                        completed_jobs_final += 1
                        failed_by_variant[str(job.variant.raw)] = int(failed_by_variant.get(str(job.variant.raw), 0)) + 1
                        failed_by_dist[str(job.dist_name)] = int(failed_by_dist.get(str(job.dist_name), 0)) + 1
                        _record_issue("final_failure", job=job, attempt=attempt_no, detail=str(reason_text or result.error))

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
                recent_completions.append(
                    {
                        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                        "variant": str(job.variant.raw),
                        "dist_name": str(job.dist_name),
                        "request_number": int(job.request_number),
                        "seed": int(job.seed),
                        "attempt": int(attempt_no),
                        "status": str(status_out),
                        "elapsed_seconds": float(result.elapsed_seconds),
                        "elapsed_human": _format_eta(float(result.elapsed_seconds)),
                        "predicted_seconds": float(dispatch.predicted_seconds),
                        "predicted_human": _format_eta(float(dispatch.predicted_seconds)),
                        "ratio": "" if (ratio != ratio) else f"{float(ratio):.2f}",
                    }
                )
                if len(recent_completions) > 20:
                    del recent_completions[:-20]
                _persist_model_state()
                scheduler_notifier.update_live_status(_build_snapshot("post_complete"), force=True)
                scheduler_notifier.notify_clock_ticks(_build_snapshot("post_complete"))
                scheduler_notifier.notify_batch(_build_snapshot("post_complete"))
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
        "requeued_total": int(requeued_total),
        "total_elapsed_seconds": float(total_elapsed),
        "slot_actual_loads": [float(x) for x in slot_actual_loads],
        "slot_pred_loads": [float(x) for x in slot_pred_loads],
        "actual_makespan_seconds": float(makespan),
        "pressure_last": pressure_last,
        "model": model.to_dict(),
        "planner": latest_plan.summary_dict(),
        "state_path": str(state_path),
        "template_state_path": str(template_state_path) if template_state_path is not None else "",
        "events_csv": str(events_csv),
        "summary_json": str(summary_json),
        "failures": failures,
        "recent_issues": list(recent_issues[-20:]),
    }
    _save_json(summary_json, summary)
    _persist_model_state()
    final_snapshot = _build_snapshot("finished")
    final_snapshot.update(summary)
    scheduler_notifier.update_live_status(final_snapshot, force=True)
    scheduler_notifier.notify_finish(final_snapshot)
    print(f"[adaptive] summary_json={summary_json}")
    print(
        f"[adaptive] finished total={len(all_jobs)} completed={completed_jobs_final} "
        f"attempts={completed_attempts} failed={len(failures)} "
        f"makespan={makespan:.1f}s elapsed={total_elapsed:.1f}s"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

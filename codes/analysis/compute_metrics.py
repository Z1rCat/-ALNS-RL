import argparse
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


INVALID_REWARD = -10000000
SUMMARY_SCHEMA_VERSION = "transport_run_summary_v1"

ROOT_DIR = Path(__file__).resolve().parents[2]
CODES_DIR = ROOT_DIR / "codes"
LOG_ROOT = CODES_DIR / "logs"
SUMMARY_DIR = LOG_ROOT / "summary"
SUMMARY_CSV = SUMMARY_DIR / "metrics_summary.csv"
DISTRIBUTION_CONFIG_PATH = ROOT_DIR / "distribution_config.json"

TRANSPORT_FIELD_ORDER = [
    "overall_distance",
    "overall_cost",
    "overall_time",
    "overall_profit",
    "overall_emission",
    "served_requests",
    "overall_request_cost",
    "overall_vehicle_cost",
    "overall_wait_cost",
    "overall_transshipment_cost",
    "overall_un_load_cost",
    "overall_emission_cost",
    "overall_storage_cost",
    "overall_delay_penalty",
    "iteration_time",
    "barge_served_requests",
    "train_served_requests",
    "truck_served_requests",
]

DEFAULT_SUMMARY_FIELDS = [
    "run_id",
    "scenario",
    "scenario_family",
    "scenario_pattern",
    "algorithm",
    "algo_version",
    "seed",
    "request_number",
    "stage_mode",
    "mean_min",
    "mean_max",
    "mean_span",
    "ab_gap",
    "n_test",
    "n_reward_rl",
    "n_reward_random",
    "n_reward_wait",
    "n_reward_reroute",
    "NPS",
    "J_rl_avg",
    "J_rand_avg",
    "J_a0_avg",
    "J_a1_avg",
    "J_best_static_avg",
    "delta_rl_vs_wait",
    "delta_rl_vs_reroute",
    "delta_rl_vs_best_static",
    "rl_reward_std",
    "rl_reward_p10",
    "rl_reward_p90",
    "implement_decision_count",
    "implement_matched_count",
    "implement_unmatched_count",
    "implement_matched_rate",
    "implement_action1_rate",
    "removal_count",
    "insertion_count",
    "removal_action1_rate",
    "insertion_action1_rate",
    "wait_share",
    "insert_reject_share",
    "p_action1_mean",
    "reward_given_action0",
    "reward_given_action1",
    "training_time_last_sec",
    "implementation_time_last_sec",
    "decision_latency_ms_mean",
    "decision_latency_ms_p90",
    "transport_train_available",
    "transport_train_served_requests",
    "transport_train_overall_cost",
    "transport_train_overall_time",
    "transport_train_overall_emission",
    "transport_train_overall_wait_cost",
    "transport_train_overall_storage_cost",
    "transport_train_overall_delay_penalty",
    "transport_implement_available",
    "transport_implement_served_requests",
    "transport_implement_overall_cost",
    "transport_implement_overall_time",
    "transport_implement_overall_emission",
    "transport_implement_overall_wait_cost",
    "transport_implement_overall_storage_cost",
    "transport_implement_overall_delay_penalty",
    "wall_time_sec",
    "cpu_percent_avg",
    "cpu_percent_peak",
    "ram_used_gb_peak",
    "gpu_util_percent_avg",
    "gpu_util_percent_peak",
    "gpu_mem_used_mb_peak",
    "warning_count",
]

_DIST_CONFIG_CACHE: Optional[Dict[str, Dict[str, Any]]] = None
_OBJ_DYNAMIC_RE = re.compile(r"dynamic(\d+)", re.IGNORECASE)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_csv_robust(path: Path) -> pd.DataFrame:
    errors: List[str] = []
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            with path.open("r", encoding=enc, errors="replace", newline="") as f:
                reader = csv.DictReader(f, restkey="__extra__")
                rows: List[Dict[str, Any]] = []
                for row in reader:
                    if row is None:
                        continue
                    rows.append(row)
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            if "__extra__" in df.columns:
                df = df.drop(columns=["__extra__"])
            return df
        except Exception as exc:
            errors.append(f"{enc}:{type(exc).__name__}:{exc}")

    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc, engine="python", on_bad_lines="skip")
        except Exception as exc:
            errors.append(f"pandas-{enc}:{type(exc).__name__}:{exc}")
    raise RuntimeError(f"failed to read csv={path}; {' | '.join(errors)}")


def _load_optional_csv(path: Path, *, warnings: List[str], label: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return _read_csv_robust(path)
    except Exception as exc:
        warnings.append(f"failed to read {label}: {exc}")
        return pd.DataFrame()


def _coerce_lower(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value in ("", None):
            return None
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    except Exception:
        return None


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, str, bool)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        if abs(value - round(value)) < 1e-9:
            return int(round(value))
        return float(value)
    return value


def _series_stats(values: Sequence[float]) -> Dict[str, Any]:
    series = pd.to_numeric(pd.Series(list(values), dtype="object"), errors="coerce").dropna()
    series = series[series != INVALID_REWARD]
    if series.empty:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p10": None,
            "median": None,
            "p90": None,
            "max": None,
        }
    return {
        "count": int(series.shape[0]),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=0)) if series.shape[0] > 1 else 0.0,
        "min": float(series.min()),
        "p10": float(series.quantile(0.10)),
        "median": float(series.quantile(0.50)),
        "p90": float(series.quantile(0.90)),
        "max": float(series.max()),
    }


def _sort_group_values(values: Sequence[Any]) -> List[Any]:
    def _key(item: Any) -> Tuple[int, float, str]:
        number = _safe_float(item)
        if number is not None:
            return (0, number, str(item))
        return (1, 0.0, str(item))

    return sorted(values, key=_key)


def _group_reward_stats(df: pd.DataFrame, group_col: str) -> Dict[str, Dict[str, Any]]:
    if df.empty or group_col not in df.columns or "reward" not in df.columns:
        return {}
    work = df.copy()
    work["reward"] = pd.to_numeric(work["reward"], errors="coerce")
    work = work[work["reward"].notna()]
    work = work[work["reward"] != INVALID_REWARD]
    if work.empty:
        return {}
    grouped: Dict[str, Dict[str, Any]] = {}
    unique_values = _sort_group_values(work[group_col].dropna().unique().tolist())
    for key in unique_values:
        sub = work[work[group_col] == key]
        if sub.empty:
            continue
        grouped[str(key)] = _series_stats(sub["reward"].tolist())
    return grouped


def _load_distribution_map() -> Dict[str, Dict[str, Any]]:
    global _DIST_CONFIG_CACHE
    if _DIST_CONFIG_CACHE is not None:
        return _DIST_CONFIG_CACHE
    mapping: Dict[str, Dict[str, Any]] = {}
    if DISTRIBUTION_CONFIG_PATH.exists():
        try:
            data = _read_json(DISTRIBUTION_CONFIG_PATH)
            for item in data.get("distributions", []):
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                if name:
                    mapping[name] = item
        except Exception:
            mapping = {}
    _DIST_CONFIG_CACHE = mapping
    return mapping


def _family_of_distribution(name: str) -> str:
    raw = str(name or "").strip()
    if raw.startswith("F1_"):
        return "F1"
    if raw.startswith("F2_"):
        return "F2"
    return raw.split("_", 1)[0] if raw else ""


def _build_scenario_summary(meta: Dict[str, Any]) -> Dict[str, Any]:
    scenario_name = str(meta.get("distribution", "") or "")
    spec = dict(_load_distribution_map().get(scenario_name, {}))
    means_raw = spec.get("means", {}) if isinstance(spec.get("means", {}), dict) else {}
    means: Dict[str, Any] = {}
    mean_values: List[float] = []
    for key, value in means_raw.items():
        numeric = _safe_float(value)
        means[str(key)] = _json_value(numeric if numeric is not None else value)
        if numeric is not None:
            mean_values.append(float(numeric))

    mean_min = min(mean_values) if mean_values else None
    mean_max = max(mean_values) if mean_values else None
    ab_gap = None
    if "A" in means_raw and "B" in means_raw:
        a_val = _safe_float(means_raw.get("A"))
        b_val = _safe_float(means_raw.get("B"))
        if a_val is not None and b_val is not None:
            ab_gap = float(b_val - a_val)

    return {
        "name": scenario_name,
        "family": _family_of_distribution(scenario_name),
        "pattern": str(spec.get("pattern", "") or ""),
        "display": str(spec.get("display", "") or scenario_name),
        "means": means,
        "mean_min": _json_value(mean_min),
        "mean_max": _json_value(mean_max),
        "mean_span": _json_value((mean_max - mean_min) if mean_min is not None and mean_max is not None else None),
        "ab_gap": _json_value(ab_gap),
        "num_regimes": len(mean_values),
    }


def _filter_reward_events_from_trace(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "reward" not in df.columns or "phase" not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    work["phase"] = _coerce_lower(work["phase"])
    if "stage" in work.columns:
        work["stage"] = _coerce_lower(work["stage"])
        receive = work[work["stage"] == "receive_reward"].copy()
        if not receive.empty:
            work = receive
    work = work[work["phase"] == "implement"].copy()
    if "source" in work.columns:
        src = work["source"].astype(str).str.strip().str.upper()
        rl_only = work[src == "RL"].copy()
        if not rl_only.empty:
            work = rl_only
    work["reward"] = pd.to_numeric(work["reward"], errors="coerce")
    work = work[work["reward"].notna()]
    work = work[work["reward"] != INVALID_REWARD]
    return work.reset_index(drop=True)


def _filter_reward_events_from_training(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "reward" not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    if "phase" in work.columns:
        work["phase"] = _coerce_lower(work["phase"])
        work = work[work["phase"] == "implement"].copy()
    else:
        work["phase"] = "implement"
    work["reward"] = pd.to_numeric(work["reward"], errors="coerce")
    work = work[work["reward"].notna()]
    work = work[work["reward"] != INVALID_REWARD]
    return work.reset_index(drop=True)


def _filter_reward_events_from_baseline(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "reward" not in df.columns or "phase" not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    work["phase"] = _coerce_lower(work["phase"])
    work = work[work["phase"] == "implement"].copy()
    if "source" in work.columns:
        src = work["source"].astype(str).str.strip().str.upper()
        base_only = work[src == "BASELINE"].copy()
        if not base_only.empty:
            work = base_only
    if "stage" in work.columns:
        work["stage"] = _coerce_lower(work["stage"])
        receive = work[work["stage"] == "receive_reward"].copy()
        if not receive.empty:
            work = receive
        else:
            finish_rows = work[work["stage"].isin(["finish_removal", "finish_insertion"])].copy()
            if not finish_rows.empty:
                work = finish_rows
            else:
                begin_insert = work[work["stage"] == "begin_insertion"].copy()
                if not begin_insert.empty:
                    work = begin_insert
    work["reward"] = pd.to_numeric(work["reward"], errors="coerce")
    work = work[work["reward"].notna()]
    work = work[work["reward"] != INVALID_REWARD]
    return work.reset_index(drop=True)


def _extract_rl_events(
    trace_df: pd.DataFrame,
    training_df: pd.DataFrame,
    *,
    warnings: List[str],
) -> Tuple[pd.DataFrame, str]:
    trace_events = _filter_reward_events_from_trace(trace_df)
    if not trace_events.empty:
        return trace_events, "rl_trace.csv (implement/receive_reward)"
    training_events = _filter_reward_events_from_training(training_df)
    if not training_events.empty:
        warnings.append(
            "RL rewards fallback to rl_training.csv because no valid implement receive_reward rows found in rl_trace.csv"
        )
        return training_events, "rl_training.csv (phase=implement, reward)"
    raise FileNotFoundError("no valid RL reward events found in rl_trace.csv or rl_training.csv")


def _align_reward_frames(named_frames: Dict[str, pd.DataFrame]) -> Tuple[int, Dict[str, pd.DataFrame]]:
    usable = {
        name: frame.reset_index(drop=True)
        for name, frame in named_frames.items()
        if isinstance(frame, pd.DataFrame) and not frame.empty and "reward" in frame.columns
    }
    if not usable:
        return 0, {}
    n_common = int(min(int(frame.shape[0]) for frame in usable.values()))
    if n_common <= 0:
        return 0, {}
    trimmed = {name: frame.iloc[:n_common].reset_index(drop=True) for name, frame in usable.items()}
    return n_common, trimmed


def _mean_reward(frame: pd.DataFrame) -> Optional[float]:
    if frame.empty or "reward" not in frame.columns:
        return None
    series = pd.to_numeric(frame["reward"], errors="coerce").dropna()
    series = series[series != INVALID_REWARD]
    if series.empty:
        return None
    return float(series.mean())


def _compute_reward_summary(
    rl_events: pd.DataFrame,
    baseline_events: Dict[str, pd.DataFrame],
    *,
    warnings: List[str],
) -> Dict[str, Any]:
    phase_groups = {
        "by_phase_label": {
            "rl": _group_reward_stats(rl_events, "phase_label"),
            "wait": _group_reward_stats(baseline_events.get("wait", pd.DataFrame()), "phase_label"),
            "reroute": _group_reward_stats(baseline_events.get("reroute", pd.DataFrame()), "phase_label"),
            "random": _group_reward_stats(baseline_events.get("random", pd.DataFrame()), "phase_label"),
        },
        "by_gt_mean": {
            "rl": _group_reward_stats(rl_events, "gt_mean"),
            "wait": _group_reward_stats(baseline_events.get("wait", pd.DataFrame()), "gt_mean"),
            "reroute": _group_reward_stats(baseline_events.get("reroute", pd.DataFrame()), "gt_mean"),
            "random": _group_reward_stats(baseline_events.get("random", pd.DataFrame()), "gt_mean"),
        },
    }

    aligned_n, aligned = _align_reward_frames(
        {
            "rl": rl_events,
            "wait": baseline_events.get("wait", pd.DataFrame()),
            "reroute": baseline_events.get("reroute", pd.DataFrame()),
            "random": baseline_events.get("random", pd.DataFrame()),
        }
    )
    rl_aligned = aligned.get("rl", rl_events)
    wait_aligned = aligned.get("wait", pd.DataFrame())
    reroute_aligned = aligned.get("reroute", pd.DataFrame())
    random_aligned = aligned.get("random", pd.DataFrame())

    j_rl_avg = _mean_reward(rl_aligned if not rl_aligned.empty else rl_events)
    j_a0_avg = _mean_reward(wait_aligned if not wait_aligned.empty else baseline_events.get("wait", pd.DataFrame()))
    j_a1_avg = _mean_reward(
        reroute_aligned if not reroute_aligned.empty else baseline_events.get("reroute", pd.DataFrame())
    )
    j_rand_avg = _mean_reward(
        random_aligned if not random_aligned.empty else baseline_events.get("random", pd.DataFrame())
    )

    static_candidates = [value for value in [j_a0_avg, j_a1_avg] if value is not None]
    j_best_static_avg = float(max(static_candidates)) if static_candidates else None
    nps_denominator = (
        float(j_best_static_avg - j_rand_avg)
        if j_best_static_avg is not None and j_rand_avg is not None
        else None
    )
    nps = None
    if nps_denominator is not None and j_rl_avg is not None and nps_denominator > 1e-6:
        nps = float((j_rl_avg - j_rand_avg) / nps_denominator)
    elif nps_denominator is not None and nps_denominator <= 1e-6:
        warnings.append(
            f"NPS denominator is too small or non-positive; set NPS to null (den={nps_denominator:.8f})"
        )

    rl_stats = _series_stats(pd.to_numeric(rl_events["reward"], errors="coerce").dropna().tolist())
    return {
        "overall": {
            "rl": rl_stats,
            "wait": _series_stats(
                pd.to_numeric(baseline_events.get("wait", pd.DataFrame()).get("reward"), errors="coerce")
                .dropna()
                .tolist()
                if not baseline_events.get("wait", pd.DataFrame()).empty
                else []
            ),
            "reroute": _series_stats(
                pd.to_numeric(baseline_events.get("reroute", pd.DataFrame()).get("reward"), errors="coerce")
                .dropna()
                .tolist()
                if not baseline_events.get("reroute", pd.DataFrame()).empty
                else []
            ),
            "random": _series_stats(
                pd.to_numeric(baseline_events.get("random", pd.DataFrame()).get("reward"), errors="coerce")
                .dropna()
                .tolist()
                if not baseline_events.get("random", pd.DataFrame()).empty
                else []
            ),
        },
        "comparison": {
            "n_common": int(aligned_n if aligned_n > 0 else rl_stats["count"]),
            "J_rl_avg": j_rl_avg,
            "J_a0_avg": j_a0_avg,
            "J_a1_avg": j_a1_avg,
            "J_rand_avg": j_rand_avg,
            "J_best_static_avg": j_best_static_avg,
            "NPS_denominator": nps_denominator,
            "NPS": nps,
            "delta_rl_vs_wait": float(j_rl_avg - j_a0_avg) if j_rl_avg is not None and j_a0_avg is not None else None,
            "delta_rl_vs_reroute": float(j_rl_avg - j_a1_avg)
            if j_rl_avg is not None and j_a1_avg is not None
            else None,
            "delta_rl_vs_best_static": float(j_rl_avg - j_best_static_avg)
            if j_rl_avg is not None and j_best_static_avg is not None
            else None,
        },
        "availability": {
            "wait_available": not baseline_events.get("wait", pd.DataFrame()).empty,
            "reroute_available": not baseline_events.get("reroute", pd.DataFrame()).empty,
            "random_available": not baseline_events.get("random", pd.DataFrame()).empty,
        },
        **phase_groups,
    }


def _select_action_rows_from_trace(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if data.empty or "stage" not in data.columns or "action" not in data.columns:
        return pd.DataFrame(), pd.DataFrame()
    valid = data.copy()
    valid["stage"] = _coerce_lower(valid["stage"])
    valid["action"] = pd.to_numeric(valid["action"], errors="coerce")
    valid = valid[valid["action"].isin([0, 1])].copy()
    if valid.empty:
        return pd.DataFrame(), pd.DataFrame()

    removal = valid[valid["stage"].isin(["send_action", "begin_removal"])].copy()
    if removal.empty:
        removal = valid[valid["stage"] == "finish_removal"].copy()

    insertion = valid[valid["stage"] == "finish_insertion"].copy()
    if insertion.empty:
        insertion = valid[valid["stage"] == "begin_insertion"].copy()
    return removal, insertion


def _compute_trace_action_profile(trace_df: pd.DataFrame) -> Dict[str, Any]:
    if trace_df.empty or "phase" not in trace_df.columns:
        return {}
    work = trace_df.copy()
    work["phase"] = _coerce_lower(work["phase"])
    implement = work[work["phase"] == "implement"].copy()
    removal, insertion = _select_action_rows_from_trace(implement)

    removal_total = int(removal.shape[0])
    insertion_total = int(insertion.shape[0])
    removal_wait = int((pd.to_numeric(removal.get("action"), errors="coerce") == 0).sum()) if removal_total else 0
    removal_reroute = int((pd.to_numeric(removal.get("action"), errors="coerce") == 1).sum()) if removal_total else 0
    insert_accept = int((pd.to_numeric(insertion.get("action"), errors="coerce") == 0).sum()) if insertion_total else 0
    insert_reject = int((pd.to_numeric(insertion.get("action"), errors="coerce") == 1).sum()) if insertion_total else 0
    decision_count = removal_total + insertion_total

    return {
        "decision_count": int(decision_count),
        "removal_count": int(removal_total),
        "insertion_count": int(insertion_total),
        "removal_wait_count": int(removal_wait),
        "removal_reroute_count": int(removal_reroute),
        "insert_accept_count": int(insert_accept),
        "insert_reject_count": int(insert_reject),
        "wait_share": float(removal_wait / removal_total) if removal_total > 0 else None,
        "removal_action1_rate": float(removal_reroute / removal_total) if removal_total > 0 else None,
        "insertion_action1_rate": float(insert_reject / insertion_total) if insertion_total > 0 else None,
        "insert_reject_share": float(insert_reject / insertion_total) if insertion_total > 0 else None,
        "action1_rate": float((removal_reroute + insert_reject) / decision_count) if decision_count > 0 else None,
    }


def _compute_decision_metrics(decision_df: pd.DataFrame) -> Dict[str, Any]:
    if decision_df.empty or "phase" not in decision_df.columns:
        return {}
    work = decision_df.copy()
    work["phase"] = _coerce_lower(work["phase"])
    implement = work[work["phase"] == "implement"].copy()
    if implement.empty:
        return {}

    for col in ("action", "reward", "matched", "p_action1", "ts_decision", "ts_reward", "gt_mean"):
        if col in implement.columns:
            implement[col] = pd.to_numeric(implement[col], errors="coerce")

    matched_total = int((implement["matched"] == 1).sum()) if "matched" in implement.columns else int(implement.shape[0])
    decision_total = int(implement.shape[0])
    eval_scope = implement.copy()
    if "matched" in eval_scope.columns:
        matched_scope = eval_scope[eval_scope["matched"] == 1].copy()
        if not matched_scope.empty:
            eval_scope = matched_scope

    eval_scope = eval_scope[pd.to_numeric(eval_scope["action"], errors="coerce").isin([0, 1])].copy()
    latency_ms_mean = None
    latency_ms_p90 = None
    if {"ts_decision", "ts_reward"} <= set(implement.columns):
        latency = (
            pd.to_numeric(implement["ts_reward"], errors="coerce")
            - pd.to_numeric(implement["ts_decision"], errors="coerce")
        ) * 1000.0
        latency = latency.dropna()
        if not latency.empty:
            latency_ms_mean = float(latency.mean())
            latency_ms_p90 = float(latency.quantile(0.90))

    by_phase_label: Dict[str, Dict[str, Any]] = {}
    if "phase_label" in eval_scope.columns and not eval_scope.empty:
        for key in _sort_group_values(eval_scope["phase_label"].dropna().unique().tolist()):
            sub = eval_scope[eval_scope["phase_label"] == key].copy()
            if sub.empty:
                continue
            actions = pd.to_numeric(sub["action"], errors="coerce")
            rewards = pd.to_numeric(sub["reward"], errors="coerce")
            p_action1 = pd.to_numeric(sub["p_action1"], errors="coerce") if "p_action1" in sub.columns else pd.Series(dtype=float)
            by_phase_label[str(key)] = {
                "count": int(sub.shape[0]),
                "action1_rate": float((actions == 1).mean()) if not actions.empty else None,
                "reward_mean": float(rewards.dropna().mean()) if rewards.notna().any() else None,
                "p_action1_mean": float(p_action1.dropna().mean()) if not p_action1.empty and p_action1.notna().any() else None,
            }

    actions = pd.to_numeric(eval_scope["action"], errors="coerce")
    rewards = pd.to_numeric(eval_scope["reward"], errors="coerce")
    action0_rewards = rewards[actions == 0].dropna()
    action1_rewards = rewards[actions == 1].dropna()
    p_action1_series = pd.to_numeric(eval_scope["p_action1"], errors="coerce") if "p_action1" in eval_scope.columns else pd.Series(dtype=float)

    return {
        "decision_count": int(decision_total),
        "matched_count": int(matched_total),
        "unmatched_count": int(max(0, decision_total - matched_total)),
        "matched_rate": float(matched_total / decision_total) if decision_total > 0 else None,
        "action1_rate": float((actions == 1).mean()) if not eval_scope.empty else None,
        "reward_given_action0": float(action0_rewards.mean()) if not action0_rewards.empty else None,
        "reward_given_action1": float(action1_rewards.mean()) if not action1_rewards.empty else None,
        "p_action1_mean": float(p_action1_series.dropna().mean())
        if not p_action1_series.empty and p_action1_series.notna().any()
        else None,
        "decision_latency_ms_mean": latency_ms_mean,
        "decision_latency_ms_p90": latency_ms_p90,
        "by_phase_label": by_phase_label,
    }


def _last_numeric_value(df: pd.DataFrame, column: str, *, phase: Optional[str] = None) -> Optional[float]:
    if df.empty or column not in df.columns:
        return None
    work = df.copy()
    if phase is not None and "phase" in work.columns:
        work["phase"] = _coerce_lower(work["phase"])
        work = work[work["phase"] == str(phase).strip().lower()].copy()
    if work.empty:
        return None
    values = pd.to_numeric(work[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[-1])


def _compute_training_metrics(training_df: pd.DataFrame) -> Dict[str, Any]:
    if training_df.empty:
        return {}
    work = training_df.copy()
    if "phase" in work.columns:
        work["phase"] = _coerce_lower(work["phase"])
    return {
        "train": {
            "rows": int(work[work["phase"] == "train"].shape[0]) if "phase" in work.columns else 0,
            "last_reward": _last_numeric_value(work, "reward", phase="train"),
            "last_avg_reward": _last_numeric_value(work, "avg_reward", phase="train"),
            "last_rolling_avg": _last_numeric_value(work, "rolling_avg", phase="train"),
            "last_action1_rate": _last_numeric_value(work, "action1_rate", phase="train"),
            "last_p_action1": _last_numeric_value(work, "p_action1", phase="train"),
            "training_time_last_sec": _last_numeric_value(work, "training_time", phase="train"),
        },
        "implement": {
            "rows": int(work[work["phase"] == "implement"].shape[0]) if "phase" in work.columns else 0,
            "last_reward": _last_numeric_value(work, "reward", phase="implement"),
            "last_avg_reward": _last_numeric_value(work, "avg_reward", phase="implement"),
            "last_rolling_avg": _last_numeric_value(work, "rolling_avg", phase="implement"),
            "last_action1_rate": _last_numeric_value(work, "action1_rate", phase="implement"),
            "last_p_action1": _last_numeric_value(work, "p_action1", phase="implement"),
            "last_reward_given_action0": _last_numeric_value(work, "reward_given_action0", phase="implement"),
            "last_reward_given_action1": _last_numeric_value(work, "reward_given_action1", phase="implement"),
            "implementation_time_last_sec": _last_numeric_value(work, "implementation_time", phase="implement"),
        },
    }


def _pick_transport_obj_file(alns_root: Path, *, implement: bool) -> Optional[Path]:
    if not alns_root.exists():
        return None
    candidates: List[Tuple[int, float, Path]] = []
    for path in alns_root.rglob("obj_record*.xlsx"):
        lower = str(path).lower()
        is_implement = "implement" in lower
        if is_implement != implement:
            continue
        match = _OBJ_DYNAMIC_RE.search(lower)
        dynamic_idx = int(match.group(1)) if match else -1
        try:
            mtime = float(path.stat().st_mtime)
        except Exception:
            mtime = 0.0
        candidates.append((dynamic_idx, mtime, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], str(item[2]).lower()))
    return candidates[-1][2]


def _load_transport_metrics_from_obj(path: Path, *, run_dir: Path, warnings: List[str]) -> Dict[str, Any]:
    try:
        workbook = pd.ExcelFile(path)
    except Exception as exc:
        warnings.append(f"failed to open transport objective workbook {path}: {exc}")
        return {}

    for sheet_name in ("final_obj", "obj_record_best", "obj_record"):
        if sheet_name not in workbook.sheet_names:
            continue
        try:
            df = pd.read_excel(path, sheet_name=sheet_name)
        except Exception as exc:
            warnings.append(f"failed to read sheet {sheet_name} from {path}: {exc}")
            continue
        if df.empty:
            continue
        row = df.iloc[-1].to_dict()
        payload: Dict[str, Any] = {}
        for key in TRANSPORT_FIELD_ORDER:
            if key in row:
                payload[key] = _json_value(_safe_float(row.get(key)))
        payload["source_file"] = str(path.relative_to(run_dir))
        payload["source_sheet"] = sheet_name
        match = _OBJ_DYNAMIC_RE.search(str(path))
        payload["dynamic_index"] = int(match.group(1)) if match else None
        return payload
    warnings.append(f"no usable objective sheet found in {path}")
    return {}


def _compute_transport_metrics(run_dir: Path, *, warnings: List[str]) -> Dict[str, Any]:
    alns_root = run_dir / "alns_outputs"
    train_path = _pick_transport_obj_file(alns_root, implement=False)
    implement_path = _pick_transport_obj_file(alns_root, implement=True)
    return {
        "train": _load_transport_metrics_from_obj(train_path, run_dir=run_dir, warnings=warnings) if train_path else {},
        "implement": _load_transport_metrics_from_obj(implement_path, run_dir=run_dir, warnings=warnings)
        if implement_path
        else {},
    }


def _load_meta(run_dir: Path) -> Dict[str, Any]:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return _read_json(meta_path)
    except Exception:
        return {}


def _load_resource_usage(run_dir: Path, *, warnings: List[str]) -> Optional[Dict[str, Any]]:
    path = run_dir / "resource_usage.json"
    if not path.exists():
        return None
    try:
        data = _read_json(path)
        return data if isinstance(data, dict) else None
    except Exception as exc:
        warnings.append(f"failed to read resource_usage.json: {exc}")
        return None


def _acquire_lock(lock_path: Path, timeout_s: float = 120.0, poll_s: float = 0.2) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + timeout_s
    while True:
        try:
            fd = _os_open_exclusive(lock_path)
            _os_close_fd(fd)
            return
        except FileExistsError:
            if time.time() >= deadline:
                raise TimeoutError(f"Timeout waiting for lock: {lock_path}")
            time.sleep(poll_s)


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return


def _os_open_exclusive(path: Path) -> int:
    import os

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    return os.open(str(path), flags)


def _os_close_fd(fd: int) -> None:
    import os

    os.close(fd)


def _write_summary_row(summary_csv: Path, row: Dict[str, Any]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    lock_path = summary_csv.with_suffix(summary_csv.suffix + ".lock")

    _acquire_lock(lock_path)
    try:
        existing_fieldnames: Optional[List[str]] = None
        existing_rows: List[Dict[str, Any]] = []
        if summary_csv.exists():
            with summary_csv.open("r", newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                existing_fieldnames = reader.fieldnames
                for item in reader:
                    existing_rows.append(item)

        if existing_fieldnames:
            fieldnames = list(existing_fieldnames)
            for name in DEFAULT_SUMMARY_FIELDS:
                if name not in fieldnames:
                    fieldnames.append(name)
        else:
            fieldnames = list(DEFAULT_SUMMARY_FIELDS)

        needs_rewrite = bool(existing_fieldnames) and list(existing_fieldnames) != fieldnames
        if needs_rewrite:
            tmp_path = summary_csv.with_suffix(summary_csv.suffix + ".tmp")
            with tmp_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for existing in existing_rows:
                    writer.writerow({key: existing.get(key) for key in fieldnames})
                writer.writerow({key: row.get(key) for key in fieldnames})
            os.replace(tmp_path, summary_csv)
        else:
            needs_header = (not summary_csv.exists()) or summary_csv.stat().st_size == 0
            with summary_csv.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if needs_header:
                    writer.writeheader()
                writer.writerow({key: row.get(key) for key in fieldnames})
    finally:
        _release_lock(lock_path)


def _write_single_row_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(DEFAULT_SUMMARY_FIELDS))
        writer.writeheader()
        writer.writerow({key: row.get(key) for key in DEFAULT_SUMMARY_FIELDS})


def _flatten_summary_row(summary: Dict[str, Any]) -> Dict[str, Any]:
    scenario = summary.get("scenario_info", {})
    reward = summary.get("reward_metrics", {})
    reward_overall = reward.get("overall", {})
    reward_cmp = reward.get("comparison", {})
    action = summary.get("action_metrics", {})
    training = summary.get("training_metrics", {})
    train_diag = training.get("train", {})
    impl_diag = training.get("implement", {})
    transport = summary.get("transport_metrics", {})
    transport_train = transport.get("train", {})
    transport_impl = transport.get("implement", {})
    resource = summary.get("resource_usage", {}) if isinstance(summary.get("resource_usage"), dict) else {}

    return {
        "run_id": summary.get("run_id"),
        "scenario": scenario.get("name"),
        "scenario_family": scenario.get("family"),
        "scenario_pattern": scenario.get("pattern"),
        "algorithm": summary.get("algorithm"),
        "algo_version": summary.get("algo_version"),
        "seed": summary.get("seed"),
        "request_number": summary.get("request_number"),
        "stage_mode": summary.get("stage_mode"),
        "mean_min": scenario.get("mean_min"),
        "mean_max": scenario.get("mean_max"),
        "mean_span": scenario.get("mean_span"),
        "ab_gap": scenario.get("ab_gap"),
        "n_test": summary.get("n_test"),
        "n_reward_rl": reward_overall.get("rl", {}).get("count"),
        "n_reward_random": reward_overall.get("random", {}).get("count"),
        "n_reward_wait": reward_overall.get("wait", {}).get("count"),
        "n_reward_reroute": reward_overall.get("reroute", {}).get("count"),
        "NPS": reward_cmp.get("NPS"),
        "J_rl_avg": reward_cmp.get("J_rl_avg"),
        "J_rand_avg": reward_cmp.get("J_rand_avg"),
        "J_a0_avg": reward_cmp.get("J_a0_avg"),
        "J_a1_avg": reward_cmp.get("J_a1_avg"),
        "J_best_static_avg": reward_cmp.get("J_best_static_avg"),
        "delta_rl_vs_wait": reward_cmp.get("delta_rl_vs_wait"),
        "delta_rl_vs_reroute": reward_cmp.get("delta_rl_vs_reroute"),
        "delta_rl_vs_best_static": reward_cmp.get("delta_rl_vs_best_static"),
        "rl_reward_std": reward_overall.get("rl", {}).get("std"),
        "rl_reward_p10": reward_overall.get("rl", {}).get("p10"),
        "rl_reward_p90": reward_overall.get("rl", {}).get("p90"),
        "implement_decision_count": action.get("implement_decision_count"),
        "implement_matched_count": action.get("implement_matched_count"),
        "implement_unmatched_count": action.get("implement_unmatched_count"),
        "implement_matched_rate": action.get("implement_matched_rate"),
        "implement_action1_rate": action.get("implement_action1_rate"),
        "removal_count": action.get("removal_count"),
        "insertion_count": action.get("insertion_count"),
        "removal_action1_rate": action.get("removal_action1_rate"),
        "insertion_action1_rate": action.get("insertion_action1_rate"),
        "wait_share": action.get("wait_share"),
        "insert_reject_share": action.get("insert_reject_share"),
        "p_action1_mean": action.get("p_action1_mean"),
        "reward_given_action0": action.get("reward_given_action0"),
        "reward_given_action1": action.get("reward_given_action1"),
        "training_time_last_sec": train_diag.get("training_time_last_sec"),
        "implementation_time_last_sec": impl_diag.get("implementation_time_last_sec"),
        "decision_latency_ms_mean": action.get("decision_latency_ms_mean"),
        "decision_latency_ms_p90": action.get("decision_latency_ms_p90"),
        "transport_train_available": int(bool(transport_train)),
        "transport_train_served_requests": transport_train.get("served_requests"),
        "transport_train_overall_cost": transport_train.get("overall_cost"),
        "transport_train_overall_time": transport_train.get("overall_time"),
        "transport_train_overall_emission": transport_train.get("overall_emission"),
        "transport_train_overall_wait_cost": transport_train.get("overall_wait_cost"),
        "transport_train_overall_storage_cost": transport_train.get("overall_storage_cost"),
        "transport_train_overall_delay_penalty": transport_train.get("overall_delay_penalty"),
        "transport_implement_available": int(bool(transport_impl)),
        "transport_implement_served_requests": transport_impl.get("served_requests"),
        "transport_implement_overall_cost": transport_impl.get("overall_cost"),
        "transport_implement_overall_time": transport_impl.get("overall_time"),
        "transport_implement_overall_emission": transport_impl.get("overall_emission"),
        "transport_implement_overall_wait_cost": transport_impl.get("overall_wait_cost"),
        "transport_implement_overall_storage_cost": transport_impl.get("overall_storage_cost"),
        "transport_implement_overall_delay_penalty": transport_impl.get("overall_delay_penalty"),
        "wall_time_sec": resource.get("wall_time_sec"),
        "cpu_percent_avg": resource.get("cpu_percent_avg"),
        "cpu_percent_peak": resource.get("cpu_percent_peak"),
        "ram_used_gb_peak": resource.get("ram_used_gb_peak"),
        "gpu_util_percent_avg": resource.get("gpu_util_percent_avg"),
        "gpu_util_percent_peak": resource.get("gpu_util_percent_peak"),
        "gpu_mem_used_mb_peak": resource.get("gpu_mem_used_mb_peak"),
        "warning_count": len(summary.get("warnings", [])),
    }


def compute_metrics(run_dir: Path, *, summary_csv: Path = SUMMARY_CSV) -> Dict[str, Any]:
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    warnings: List[str] = []
    meta = _load_meta(run_dir)
    run_id = str(meta.get("run_name") or run_dir.name)
    scenario_name = str(meta.get("distribution", "") or "")
    algorithm = meta.get("algorithm")
    algo_version = meta.get("algo_version")
    seed = meta.get("seed")
    request_number = meta.get("request_number")
    stage_mode = meta.get("stage_mode")

    trace_df = _load_optional_csv(run_dir / "rl_trace.csv", warnings=warnings, label="rl_trace.csv")
    training_df = _load_optional_csv(run_dir / "rl_training.csv", warnings=warnings, label="rl_training.csv")
    decision_df = _load_optional_csv(run_dir / "rl_decision.csv", warnings=warnings, label="rl_decision.csv")
    if trace_df.empty and training_df.empty:
        raise FileNotFoundError("neither rl_trace.csv nor rl_training.csv could be loaded")

    rl_events, reward_source_rl = _extract_rl_events(trace_df, training_df, warnings=warnings)
    baseline_events = {
        "wait": _filter_reward_events_from_baseline(
            _load_optional_csv(run_dir / "baseline_wait.csv", warnings=warnings, label="baseline_wait.csv")
        ),
        "reroute": _filter_reward_events_from_baseline(
            _load_optional_csv(run_dir / "baseline_reroute.csv", warnings=warnings, label="baseline_reroute.csv")
        ),
        "random": _filter_reward_events_from_baseline(
            _load_optional_csv(run_dir / "baseline_random.csv", warnings=warnings, label="baseline_random.csv")
        ),
    }

    reward_summary = _compute_reward_summary(rl_events, baseline_events, warnings=warnings)
    trace_action = _compute_trace_action_profile(trace_df)
    decision_action = _compute_decision_metrics(decision_df)
    action_summary = {
        "implement_decision_count": trace_action.get("decision_count", decision_action.get("decision_count")),
        "implement_matched_count": decision_action.get("matched_count"),
        "implement_unmatched_count": decision_action.get("unmatched_count"),
        "implement_matched_rate": decision_action.get("matched_rate"),
        "implement_action1_rate": trace_action.get("action1_rate", decision_action.get("action1_rate")),
        "removal_count": trace_action.get("removal_count"),
        "insertion_count": trace_action.get("insertion_count"),
        "removal_action1_rate": trace_action.get("removal_action1_rate"),
        "insertion_action1_rate": trace_action.get("insertion_action1_rate"),
        "wait_share": trace_action.get("wait_share"),
        "insert_reject_share": trace_action.get("insert_reject_share"),
        "p_action1_mean": decision_action.get("p_action1_mean"),
        "reward_given_action0": decision_action.get("reward_given_action0"),
        "reward_given_action1": decision_action.get("reward_given_action1"),
        "decision_latency_ms_mean": decision_action.get("decision_latency_ms_mean"),
        "decision_latency_ms_p90": decision_action.get("decision_latency_ms_p90"),
        "by_phase_label": decision_action.get("by_phase_label", {}),
    }
    training_summary = _compute_training_metrics(training_df)
    transport_summary = _compute_transport_metrics(run_dir, warnings=warnings)
    resource_usage = _load_resource_usage(run_dir, warnings=warnings)

    reward_cmp = reward_summary.get("comparison", {})
    metrics = {
        "run_summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "run_id": run_id,
        "scenario": scenario_name,
        "algorithm": algorithm,
        "algo_version": algo_version,
        "seed": seed,
        "request_number": request_number,
        "stage_mode": stage_mode,
        "n_test": reward_cmp.get("n_common"),
        "reward_source_rl": reward_source_rl,
        "J_rl_avg": reward_cmp.get("J_rl_avg"),
        "J_rand_avg": reward_cmp.get("J_rand_avg"),
        "J_a0_avg": reward_cmp.get("J_a0_avg"),
        "J_a1_avg": reward_cmp.get("J_a1_avg"),
        "J_best_static_avg": reward_cmp.get("J_best_static_avg"),
        "NPS_denominator": reward_cmp.get("NPS_denominator"),
        "NPS": reward_cmp.get("NPS"),
        "scenario_info": _build_scenario_summary(meta),
        "reward_metrics": reward_summary,
        "action_metrics": action_summary,
        "training_metrics": training_summary,
        "transport_metrics": transport_summary,
        "meta": meta,
        "warnings": warnings,
        "resource_usage": resource_usage,
    }

    payload = json.dumps(metrics, ensure_ascii=False, indent=2)
    (run_dir / "metrics.json").write_text(payload, encoding="utf-8")
    (run_dir / "run_summary.json").write_text(payload, encoding="utf-8")

    flat_row = _flatten_summary_row(metrics)
    _write_single_row_csv(run_dir / "run_summary_flat.csv", flat_row)
    _write_summary_row(summary_csv, flat_row)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute run-level transport/RL summary for a completed run_dir.")
    parser.add_argument("--run-dir", required=True, help="Run directory (contains rl_trace.csv / baseline_*.csv).")
    parser.add_argument(
        "--summary-csv",
        default=str(SUMMARY_CSV),
        help="Global summary CSV to append (default: codes/logs/summary/metrics_summary.csv).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        compute_metrics(Path(args.run_dir), summary_csv=Path(args.summary_csv))
    except Exception as exc:
        print(f"[metrics] failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

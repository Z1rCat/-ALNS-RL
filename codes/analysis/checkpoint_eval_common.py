from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from compare_trace_slices import (
    _build_decision_frame,
    _normalize_decision_frame,
    _read_json,
    _severity_bucket,
)


def safe_mean(series: pd.Series | list[float] | None) -> float:
    if series is None:
        return math.nan
    clean = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    if clean.empty:
        return math.nan
    return float(clean.mean())


def safe_last_finite(series: pd.Series | None) -> float:
    if series is None:
        return math.nan
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return math.nan
    return float(clean.iloc[-1])


def load_meta(run_dir: Path) -> dict[str, Any]:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return _read_json(meta_path)
    except Exception:
        return {}


def load_pipeline_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "post_stage" / "pipeline_summary.json"
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_run_frame(run_dir: Path, run_label: str | None = None) -> pd.DataFrame:
    label = str(run_label or run_dir.name)
    if not (run_dir / "rl_trace.csv").exists():
        return pd.DataFrame()
    if not (run_dir / "rl_decision.csv").exists():
        return pd.DataFrame()
    frame = _build_decision_frame(run_dir)
    if frame.empty:
        return pd.DataFrame()
    normalized = _normalize_decision_frame(frame, run_label=label, run_dir=run_dir, meta=load_meta(run_dir))
    normalized["severity_bucket"] = _severity_bucket(normalized["severity"])
    return normalized


def load_training_frame(run_dir: Path) -> pd.DataFrame:
    training_path = run_dir / "rl_training.csv"
    if not training_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(training_path)
    except Exception:
        return pd.DataFrame()


def summarize_run_dir(
    run_dir: Path,
    *,
    run_label: str | None = None,
    phase_name: str = "implement",
    hard_stage_family: str = "removal",
    hard_min_severity: int = 5,
    easy_max_severity: int = 3,
    lambda_fp: float = 0.20,
) -> dict[str, Any]:
    label = str(run_label or run_dir.name)
    meta = load_meta(run_dir)
    summary = load_pipeline_summary(run_dir)
    frame = build_run_frame(run_dir, run_label=label)
    training_df = load_training_frame(run_dir)

    row: dict[str, Any] = {
        "run_label": label,
        "run_dir": str(run_dir),
        "status": "ok",
        "distribution": str(meta.get("distribution", "")),
        "algorithm": str(meta.get("algorithm", "")),
        "algo_version": str(meta.get("algo_version", "")),
        "seed": str(meta.get("seed", "")),
        "stage_mode_meta": str(meta.get("stage_mode", "")),
        "done_marker": int((run_dir / "DONE.json").exists()),
        "has_rl_trace": int((run_dir / "rl_trace.csv").exists()),
        "has_rl_decision": int((run_dir / "rl_decision.csv").exists()),
        "has_rl_training": int((run_dir / "rl_training.csv").exists()),
        "has_rl_summary": int((run_dir / "rl_summary.csv").exists()),
        "has_pipeline_summary": int(bool(summary)),
        "hard_stage_family": str(hard_stage_family),
        "hard_min_severity": int(hard_min_severity),
        "easy_max_severity": int(easy_max_severity),
        "lambda_fp": float(lambda_fp),
        "total_rows": 0,
        "phase_rows": 0,
        "avg_reward": math.nan,
        "pos_rate": math.nan,
        "action1_rate": math.nan,
        "wait_share": math.nan,
        "engage_share": math.nan,
        "mean_p_action1": math.nan,
        "hard_rows": 0,
        "Q_hard_slice": math.nan,
        "R_hard_action1": math.nan,
        "hard_action1_rate": math.nan,
        "hard_wait_share": math.nan,
        "hard_mean_p_action1": math.nan,
        "easy_rows": 0,
        "P_easy_action0": math.nan,
        "P_easy_waitlike": math.nan,
        "easy_action1_rate": math.nan,
        "easy_mean_p_action1": math.nan,
        "Q_hard_rem": math.nan,
        "R_hard_rem": math.nan,
        "rl_training_last_avg_reward": math.nan,
        "rl_training_last_rolling_avg": math.nan,
        "rl_training_last_action1_rate": math.nan,
        "phase4_ckpt_policy": summary.get("phase4_ckpt_policy", ""),
        "phase4_ckpt_source": summary.get("phase4_ckpt_source", ""),
        "phase4_ckpt_iter_id": summary.get("phase4_ckpt_iter_id", ""),
        "phase4_ckpt_objective_score": summary.get("phase4_ckpt_objective_score", ""),
    }

    if frame.empty:
        row["status"] = "no_decision_frame"
        return row

    phase_df = frame[frame["phase"].astype(str) == str(phase_name)].copy()
    row["total_rows"] = int(len(frame))
    row["phase_rows"] = int(len(phase_df))
    if phase_df.empty:
        row["status"] = "no_phase_rows"
    else:
        row["avg_reward"] = safe_mean(phase_df["reward"])
        row["pos_rate"] = safe_mean((pd.to_numeric(phase_df["reward"], errors="coerce") > 0).astype(float))
        row["action1_rate"] = safe_mean(phase_df["action1"])
        row["wait_share"] = safe_mean(phase_df["wait_like"])
        row["engage_share"] = safe_mean(phase_df["engage_like"])
        row["mean_p_action1"] = safe_mean(phase_df["p_action1"])

        hard_df = phase_df[
            (phase_df["stage_family"].astype(str) == str(hard_stage_family))
            & (pd.to_numeric(phase_df["severity"], errors="coerce") >= int(hard_min_severity))
        ].copy()
        easy_df = phase_df[
            pd.to_numeric(phase_df["severity"], errors="coerce") <= int(easy_max_severity)
        ].copy()

        row["hard_rows"] = int(len(hard_df))
        row["easy_rows"] = int(len(easy_df))
        row["Q_hard_slice"] = safe_mean(hard_df["reward"])
        row["R_hard_action1"] = safe_mean(hard_df["action1"] * hard_df["reward"]) if not hard_df.empty else math.nan
        row["hard_action1_rate"] = safe_mean(hard_df["action1"])
        row["hard_wait_share"] = safe_mean(hard_df["wait_like"])
        row["hard_mean_p_action1"] = safe_mean(hard_df["p_action1"])
        row["P_easy_action0"] = safe_mean(easy_df["action0"] * easy_df["reward"]) if not easy_df.empty else math.nan
        row["P_easy_waitlike"] = safe_mean(easy_df["wait_like"] * easy_df["reward"]) if not easy_df.empty else math.nan
        row["easy_action1_rate"] = safe_mean(easy_df["action1"])
        row["easy_mean_p_action1"] = safe_mean(easy_df["p_action1"])

        if str(hard_stage_family) == "removal":
            row["Q_hard_rem"] = row["Q_hard_slice"]
            row["R_hard_rem"] = row["R_hard_action1"]

        easy_action1_rate = row["easy_action1_rate"]
        if not math.isnan(row["R_hard_action1"]) and not math.isnan(float(easy_action1_rate)):
            row["C_sel_tilde"] = float(row["R_hard_action1"]) - float(lambda_fp) * float(easy_action1_rate)
        else:
            row["C_sel_tilde"] = math.nan

    if not training_df.empty:
        row["rl_training_last_avg_reward"] = safe_last_finite(training_df.get("avg_reward"))
        row["rl_training_last_rolling_avg"] = safe_last_finite(training_df.get("rolling_avg"))
        row["rl_training_last_action1_rate"] = safe_last_finite(training_df.get("action1_rate"))

    return row

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _safe_float(value, default=0.0) -> float:
    try:
        if value is None:
            return float(default)
        if isinstance(value, str) and value.strip() == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value, default=0) -> int:
    try:
        if value is None:
            return int(default)
        if isinstance(value, str) and value.strip() == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _pair_impl_actions_rewards(trace_df: pd.DataFrame) -> pd.DataFrame:
    impl = trace_df[trace_df.get("phase", "").astype(str) == "implement"].copy()
    send = impl[impl.get("stage", "").astype(str) == "send_action"].copy().reset_index(drop=True)
    recv = impl[impl.get("stage", "").astype(str) == "receive_reward"].copy().reset_index(drop=True)
    n = int(min(len(send), len(recv)))
    if n <= 0:
        return pd.DataFrame(columns=["phase_label", "action", "reward", "p_action1"])
    send = send.iloc[:n].copy()
    recv = recv.iloc[:n].copy()
    paired = pd.DataFrame(
        {
            "phase_label": send.get("phase_label", "").astype(str),
            "action": pd.to_numeric(send.get("action"), errors="coerce"),
            "reward": pd.to_numeric(recv.get("reward"), errors="coerce"),
            "p_action1": pd.to_numeric(send.get("p_action1"), errors="coerce"),
        }
    )
    paired["phase_label"] = paired["phase_label"].replace("", "unknown")
    paired = paired.dropna(subset=["action", "reward"])
    paired["action"] = paired["action"].astype(int)
    return paired


def _summarize_group(df: pd.DataFrame, group_name: str) -> Dict[str, float]:
    if df.empty:
        return {
            "group": group_name,
            "n_total": 0,
            "n_action0": 0,
            "n_action1": 0,
            "action1_rate": 0.0,
            "reward_mean": 0.0,
            "reward_given_action0": 0.0,
            "reward_given_action1": 0.0,
            "p_action1_mean": 0.0,
            "reward_gap_a1_a0": 0.0,
        }
    a0 = df[df["action"] == 0]["reward"]
    a1 = df[df["action"] == 1]["reward"]
    r0 = float(a0.mean()) if not a0.empty else 0.0
    r1 = float(a1.mean()) if not a1.empty else 0.0
    return {
        "group": group_name,
        "n_total": int(df.shape[0]),
        "n_action0": int(a0.shape[0]),
        "n_action1": int(a1.shape[0]),
        "action1_rate": float((df["action"] == 1).mean()),
        "reward_mean": float(df["reward"].mean()),
        "reward_given_action0": r0,
        "reward_given_action1": r1,
        "p_action1_mean": float(df["p_action1"].dropna().mean()) if "p_action1" in df.columns else 0.0,
        "reward_gap_a1_a0": float(r1 - r0),
    }


def summarize_run(run_dir: Path) -> Dict[str, pd.DataFrame]:
    trace_path = run_dir / "rl_trace.csv"
    train_path = run_dir / "rl_training.csv"
    if not trace_path.exists():
        raise FileNotFoundError(f"missing rl_trace.csv: {trace_path}")
    if not train_path.exists():
        raise FileNotFoundError(f"missing rl_training.csv: {train_path}")

    trace_df = pd.read_csv(trace_path)
    train_df = pd.read_csv(train_path)

    paired = _pair_impl_actions_rewards(trace_df)
    rows: List[Dict[str, float]] = []
    rows.append(_summarize_group(paired, "ALL"))
    for group, sub_df in paired.groupby("phase_label"):
        rows.append(_summarize_group(sub_df, str(group)))
    metrics_df = pd.DataFrame(rows)

    t_train = train_df[train_df.get("phase", "").astype(str) == "train"].copy()
    tcr_cols = [
        "tcr_enabled",
        "tcr_rollout_groups",
        "tcr_trigger_events",
        "tcr_triggered_groups",
        "tcr_new_samples",
        "tcr_buffer_size",
        "tcr_action1_rate_trigger_mean",
        "tcr_reward_gap_trigger_mean",
        "tcr_aux_loss",
        "tcr_aux_applied_batches",
    ]
    tcr_summary = {}
    for col in tcr_cols:
        if col not in t_train.columns:
            continue
        series = pd.to_numeric(t_train[col], errors="coerce").dropna()
        if series.empty:
            continue
        if col in {"tcr_trigger_events", "tcr_new_samples", "tcr_aux_applied_batches"}:
            tcr_summary[f"{col}_sum"] = float(series.sum())
        else:
            tcr_summary[f"{col}_mean"] = float(series.mean())

    if tcr_summary:
        tcr_summary["tcr_metrics_available"] = 1
        tcr_df = pd.DataFrame([tcr_summary])
    else:
        tcr_df = pd.DataFrame([{"tcr_metrics_available": 0}])
    return {"group_metrics": metrics_df, "tcr_metrics": tcr_df}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize TCR-related run metrics.")
    parser.add_argument("--run_dir", required=True, help="Run directory containing rl_trace.csv and rl_training.csv")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    outputs = summarize_run(run_dir)

    group_path = run_dir / "tcr_group_metrics.csv"
    tcr_path = run_dir / "tcr_training_metrics.csv"
    outputs["group_metrics"].to_csv(group_path, index=False)
    outputs["tcr_metrics"].to_csv(tcr_path, index=False)

    print(f"[OK] wrote: {group_path}")
    print(f"[OK] wrote: {tcr_path}")


if __name__ == "__main__":
    main()

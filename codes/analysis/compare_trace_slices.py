from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


NEXUS_ROOT = Path(r"a:\MYpython\34959_RL\codes\nexus")


@dataclass
class RunSpec:
    label: str
    run_name_token: str
    algorithm: str
    algo_version: str | None


RUN_SPECS = [
    RunSpec("PPO", "_PPO_S", "PPO", "v1"),
    RunSpec("PPO_NEW_v3", "_PPO_NEW_S", "PPO_NEW", "v3"),
    RunSpec("PPO_NEW_v3_1", "_PPO_NEW_S", "PPO_NEW", "v3.1"),
    RunSpec("PLR_UED", "_PLR_UED_S", "PPO_NEW", "v3"),
]


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return math.nan


def _safe_int(value) -> int | None:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def _trace_row_iter(trace_path: Path) -> Iterable[dict]:
    with trace_path.open("r", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split(",")
        for line in f:
            parts = line.rstrip("\n").split(",")
            if not parts or len(parts) < 15:
                continue
            pre = parts[:15]
            idx = 15
            passed_tokens: list[str] = []
            while idx < len(parts):
                passed_tokens.append(parts[idx])
                token = parts[idx].rstrip()
                if token.endswith(']"') or token.endswith("]"):
                    idx += 1
                    break
                idx += 1
            rest = parts[idx:]
            row = pre + [",".join(passed_tokens)] + rest
            if len(row) < len(header):
                row += [""] * (len(header) - len(row))
            elif len(row) > len(header):
                row = row[: len(header)]
            yield dict(zip(header, row))


def _build_decision_frame(run_dir: Path) -> pd.DataFrame:
    pending: deque[dict] = deque()
    stage_family = ""
    rows: list[dict] = []
    for row in _trace_row_iter(run_dir / "rl_trace.csv"):
        stage = str(row.get("stage", ""))
        if stage.startswith("begin_"):
            stage_family = stage[len("begin_") :]
        if str(row.get("source", "")) != "RL":
            continue
        if stage == "send_action":
            payload = dict(row)
            payload["stage_family"] = stage_family
            pending.append(payload)
        elif stage == "receive_reward":
            base = pending.popleft() if pending else dict(row)
            merged = dict(base)
            for key in (
                "ts",
                "phase",
                "phase_label",
                "severity",
                "gt_mean",
                "duration_type",
                "current_time",
                "context_id",
                "regime_id",
                "reward",
                "request",
                "vehicle",
                "uncertainty_index",
                "action",
                "p_action1",
            ):
                merged[key] = row.get(key, merged.get(key, ""))
            merged["stage"] = "receive_reward"
            merged["matched"] = 1
            if "stage_family" not in merged:
                merged["stage_family"] = stage_family
            rows.append(merged)
    trace_df = pd.DataFrame(rows)
    decision_csv = run_dir / "rl_decision.csv"
    if not decision_csv.exists() or trace_df.empty:
        return trace_df

    dec_df = pd.read_csv(decision_csv)
    dec_df["_phase_seq"] = dec_df.groupby(dec_df.get("phase", pd.Series([""] * len(dec_df))).fillna("").astype(str)).cumcount()
    trace_df["_phase_seq"] = trace_df.groupby(trace_df.get("phase", pd.Series([""] * len(trace_df))).fillna("").astype(str)).cumcount()
    enrich_cols = [
        "_phase_seq",
        "phase",
        "severity",
        "phase_label",
        "gt_mean",
        "duration_type",
        "current_time",
        "context_id",
        "regime_id",
        "p_action1",
        "stage_family",
    ]
    merged = dec_df.merge(
        trace_df[enrich_cols],
        on=["phase", "_phase_seq"],
        how="left",
        suffixes=("", "_trace"),
    )
    for col in ["severity", "phase_label", "gt_mean", "duration_type", "current_time", "context_id", "regime_id", "p_action1"]:
        trace_col = f"{col}_trace"
        if trace_col in merged.columns:
            merged[col] = merged[col] if col in merged.columns else np.nan
            merged[col] = merged[col].where(merged[col].notna() & (merged[col].astype(str) != ""), merged[trace_col])
            merged.drop(columns=[trace_col], inplace=True)
    if "stage_family_trace" in merged.columns:
        stage_series = merged.get("stage_family", pd.Series([""] * len(merged), index=merged.index))
        merged["stage_family"] = stage_series.where(stage_series.notna() & (stage_series.astype(str) != ""), merged["stage_family_trace"])
        merged.drop(columns=["stage_family_trace"], inplace=True)
    if "impl_stream" in merged.columns:
        merged["stage_family"] = merged["impl_stream"].where(
            merged["impl_stream"].fillna("").astype(str) != "",
            merged.get("stage_family", pd.Series([""] * len(merged), index=merged.index)),
        )
    return merged.drop(columns=["_phase_seq"])


def _normalize_decision_frame(df: pd.DataFrame, run_label: str, run_dir: Path, meta: dict) -> pd.DataFrame:
    out = df.copy()
    out["run_label"] = run_label
    out["run_dir"] = str(run_dir)
    out["distribution"] = meta.get("distribution", "")
    out["seed"] = meta.get("seed", "")
    out["algorithm"] = meta.get("algorithm", "")
    out["algo_version"] = meta.get("algo_version", "")

    for col in ("action", "reward", "severity", "gt_mean", "p_action1"):
        out[col] = pd.to_numeric(out.get(col), errors="coerce")

    def _series(name: str) -> pd.Series:
        if name in out.columns:
            return out[name]
        return pd.Series([""] * len(out), index=out.index)

    out["phase"] = _series("phase").fillna("").astype(str)
    out["phase_label"] = _series("phase_label").fillna("").astype(str)
    out["duration_type"] = _series("duration_type").fillna("").astype(str)
    out["context_id"] = _series("context_id").fillna("").astype(str)
    out["regime_id"] = _series("regime_id").fillna("").astype(str)
    out["stage_family"] = _series("stage_family").fillna("").astype(str)
    out["stage_family"] = out["stage_family"].replace({"": "unknown"})

    out["reward_pos"] = (out["reward"] > 0).astype(int)
    out["action1"] = (out["action"] == 1).astype(int)
    out["action0"] = (out["action"] == 0).astype(int)

    out["semantic_action"] = "unknown"
    rem = out["stage_family"] == "removal"
    ins = out["stage_family"] == "insertion"
    out.loc[rem & (out["action"] == 0), "semantic_action"] = "wait"
    out.loc[rem & (out["action"] == 1), "semantic_action"] = "remove"
    out.loc[ins & (out["action"] == 0), "semantic_action"] = "insert"
    out.loc[ins & (out["action"] == 1), "semantic_action"] = "non_insert"
    out["wait_like"] = out["semantic_action"].isin(["wait", "non_insert"]).astype(int)
    out["engage_like"] = out["semantic_action"].isin(["remove", "insert"]).astype(int)
    return out


def _severity_bucket(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    unique_vals = sorted(v for v in numeric.dropna().unique())
    if not unique_vals:
        return pd.Series(["unknown"] * len(series), index=series.index)
    if len(unique_vals) <= 6:
        def mapper(v):
            if math.isnan(v):
                return "unknown"
            if v <= 3:
                return "low(2-3)"
            if v <= 5:
                return "mid(4-5)"
            return "high(6+)"

        return numeric.apply(mapper)
    q = numeric.quantile([0.33, 0.67]).to_list()
    q1, q2 = q[0], q[1]
    def mapper(v):
        if math.isnan(v):
            return "unknown"
        if v <= q1:
            return f"low(<= {q1:.2f})"
        if v <= q2:
            return f"mid(<= {q2:.2f})"
        return f"high(> {q2:.2f})"
    return numeric.apply(mapper)


def _slice_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    group_cols = [c for c in group_cols if c in df.columns]
    rows: list[dict] = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        row["n"] = int(len(g))
        row["avg_reward"] = float(g["reward"].mean()) if len(g) else math.nan
        row["pos_rate"] = float(g["reward_pos"].mean()) if len(g) else math.nan
        row["action1_rate"] = float(g["action1"].mean()) if len(g) else math.nan
        p_nonnull = g["p_action1"].dropna()
        row["mean_p_action1"] = float(p_nonnull.mean()) if len(p_nonnull) else math.nan
        row["wait_share"] = float(g["wait_like"].mean()) if len(g) else math.nan
        row["engage_share"] = float(g["engage_like"].mean()) if len(g) else math.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(group_cols).reset_index(drop=True)
    return out


def _run_choice_score(run_dir: Path) -> tuple:
    done = (run_dir / "DONE.json").exists()
    trace_ok = (run_dir / "rl_trace.csv").exists()
    summary_ok = (run_dir / "rl_summary.csv").exists()
    trace_mtime = (run_dir / "rl_trace.csv").stat().st_mtime if trace_ok else 0.0
    return (int(done), int(summary_ok and trace_ok), trace_mtime)


def _select_run(distribution: str, seed: int, spec: RunSpec) -> Path | None:
    candidates: list[Path] = []
    for meta_path in NEXUS_ROOT.glob("**/meta.json"):
        run_dir = meta_path.parent
        name = run_dir.name
        if spec.run_name_token not in name:
            continue
        try:
            meta = _read_json(meta_path)
        except Exception:
            continue
        if str(meta.get("distribution", "")) != distribution:
            continue
        if _safe_int(meta.get("seed")) != seed:
            continue
        if str(meta.get("algorithm", "")) != spec.algorithm:
            continue
        version = str(meta.get("algo_version", ""))
        if spec.algo_version is not None and version != spec.algo_version:
            continue
        if not ((run_dir / "rl_trace.csv").exists() and (run_dir / "rl_summary.csv").exists()):
            continue
        candidates.append(run_dir)
    if not candidates:
        return None
    candidates.sort(key=_run_choice_score, reverse=True)
    return candidates[0]


def _load_selected_runs(distribution: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows = []
    decision_frames = []
    for spec in RUN_SPECS:
        run_dir = _select_run(distribution, seed, spec)
        if run_dir is None:
            run_rows.append(
                {
                    "run_label": spec.label,
                    "distribution": distribution,
                    "seed": seed,
                    "selected": 0,
                    "run_dir": "",
                    "done": 0,
                    "summary_ok": 0,
                    "trace_ok": 0,
                    "credibility": "missing",
                    "why_selected": "no matching run with rl_trace.csv + rl_summary.csv",
                }
            )
            continue
        meta = _read_json(run_dir / "meta.json")
        done = int((run_dir / "DONE.json").exists())
        credibility = "complete" if done else "main_result_usable"
        why = "DONE.json present" if done else "DONE.json missing, but rl_summary.csv + rl_trace.csv exist"
        decision_df = _normalize_decision_frame(_build_decision_frame(run_dir), spec.label, run_dir, meta)
        decision_frames.append(decision_df)
        run_rows.append(
            {
                "run_label": spec.label,
                "distribution": distribution,
                "seed": seed,
                "selected": 1,
                "run_dir": str(run_dir),
                "done": done,
                "summary_ok": int((run_dir / "rl_summary.csv").exists()),
                "trace_ok": int((run_dir / "rl_trace.csv").exists()),
                "credibility": credibility,
                "why_selected": why,
                "algorithm": meta.get("algorithm", ""),
                "algo_version": meta.get("algo_version", ""),
            }
        )
    run_df = pd.DataFrame(run_rows)
    decision_df = pd.concat(decision_frames, ignore_index=True) if decision_frames else pd.DataFrame()
    return run_df, decision_df


def _summary_alignment(run_df: pd.DataFrame, decision_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in run_df.itertuples(index=False):
        if getattr(row, "selected", 0) != 1:
            continue
        run_dir = Path(row.run_dir)
        summary = list(csv.DictReader((run_dir / "rl_summary.csv").open("r", encoding="utf-8")))[0]
        sub = decision_df[(decision_df["run_label"] == row.run_label) & (decision_df["phase"] == "implement")].copy()
        rem = sub[sub["stage_family"] == "removal"]
        ins = sub[sub["stage_family"] == "insertion"]
        rows.append(
            {
                "run_label": row.run_label,
                "impl_trace_n": int(len(sub)),
                "summary_reward_count": _safe_int(summary.get("reward_count")),
                "trace_removal_remove": int(((rem["action"] == 1)).sum()),
                "summary_removal_action": _safe_int(summary.get("removal_action")),
                "trace_removal_wait": int(((rem["action"] == 0)).sum()),
                "summary_removal_wait_action": _safe_int(summary.get("removal_wait_action")),
                "trace_insertion_insert": int(((ins["action"] == 0)).sum()),
                "summary_insertion_action": _safe_int(summary.get("insertion_action")),
                "trace_insertion_non": int(((ins["action"] == 1)).sum()),
                "summary_insertion_non_action": _safe_int(summary.get("insertion_non_action")),
            }
        )
    return pd.DataFrame(rows)


def _build_outputs(decision_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    core = decision_df[decision_df["phase"] == "implement"].copy()
    if core.empty:
        return outputs
    core["severity_bucket"] = _severity_bucket(core["severity"])

    outputs["o10_90_all_implement_decisions"] = core.sort_values(["run_label"]).reset_index(drop=True)
    outputs["o10_90_by_stage_family"] = _slice_metrics(core, ["run_label", "stage_family"])
    outputs["o10_90_by_severity"] = _slice_metrics(core, ["run_label", "severity"])
    outputs["o10_90_by_severity_bucket"] = _slice_metrics(core, ["run_label", "severity_bucket"])
    outputs["o10_90_by_stage_x_severity"] = _slice_metrics(core, ["run_label", "stage_family", "severity"])
    outputs["o10_90_by_stage_x_severity_bucket"] = _slice_metrics(core, ["run_label", "stage_family", "severity_bucket"])
    outputs["o10_90_by_raw_action"] = _slice_metrics(core, ["run_label", "action"])
    outputs["o10_90_by_semantic_action"] = _slice_metrics(core, ["run_label", "semantic_action"])
    outputs["o10_90_by_stage_x_raw_action"] = _slice_metrics(core, ["run_label", "stage_family", "action"])
    outputs["o10_90_by_stage_x_semantic_action"] = _slice_metrics(core, ["run_label", "stage_family", "semantic_action"])
    outputs["o10_90_by_phase_label"] = _slice_metrics(core, ["run_label", "phase_label"])

    for candidate in ["duration_type", "gt_mean", "context_id", "regime_id"]:
        if candidate in core.columns and core[candidate].nunique(dropna=False) > 1:
            outputs[f"o10_90_by_{candidate}"] = _slice_metrics(core, ["run_label", candidate])
    return outputs


def _load_optional_reference(distribution: str, seed: int, keep_labels: list[str]) -> pd.DataFrame:
    run_rows, decision_df = _load_selected_runs(distribution, seed)
    if decision_df.empty:
        return pd.DataFrame()
    keep = decision_df["run_label"].isin(keep_labels)
    ref = decision_df[keep & (decision_df["phase"] == "implement")].copy()
    if ref.empty:
        return pd.DataFrame()
    ref["severity_bucket"] = _severity_bucket(ref["severity"])
    return pd.concat(
        [
            _slice_metrics(ref, ["run_label", "stage_family"]).assign(distribution=distribution, slice_type="stage_family"),
            _slice_metrics(ref, ["run_label", "severity_bucket"]).assign(distribution=distribution, slice_type="severity_bucket"),
            _slice_metrics(ref, ["run_label", "semantic_action"]).assign(distribution=distribution, slice_type="semantic_action"),
        ],
        ignore_index=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare trace-level slices across PPO / PPO_NEW / PLR_UED.")
    parser.add_argument("--distribution", default="O_10_90")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        default=str(Path(r"a:\MYpython\34959_RL\codes\analysis\outputs\trace_slice_compare")),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_df, decision_df = _load_selected_runs(args.distribution, args.seed)
    align_df = _summary_alignment(run_df, decision_df) if not decision_df.empty else pd.DataFrame()
    outputs = _build_outputs(decision_df)

    ref_frames = []
    for ref_dist in ["F1_10_90", "G_10_90_60"]:
        ref = _load_optional_reference(ref_dist, args.seed, ["PPO", "PPO_NEW_v3_1"])
        if not ref.empty:
            ref_frames.append(ref)
    ref_df = pd.concat(ref_frames, ignore_index=True) if ref_frames else pd.DataFrame()

    run_df.to_csv(out_dir / "selected_runs.csv", index=False, encoding="utf-8-sig")
    if not align_df.empty:
        align_df.to_csv(out_dir / "trace_summary_alignment.csv", index=False, encoding="utf-8-sig")
    for name, df in outputs.items():
        df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    if not ref_df.empty:
        ref_df.to_csv(out_dir / "reference_f1_g_slices.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(out_dir / "trace_slice_compare.xlsx", engine="openpyxl") as writer:
        run_df.to_excel(writer, sheet_name="selected_runs", index=False)
        if not align_df.empty:
            align_df.to_excel(writer, sheet_name="trace_vs_summary", index=False)
        for name, df in outputs.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
        if not ref_df.empty:
            ref_df.to_excel(writer, sheet_name="ref_f1_g", index=False)

    print(f"wrote outputs to: {out_dir}")
    print(run_df.to_string(index=False))
    if not align_df.empty:
        print("\n=== trace_vs_summary ===")
        print(align_df.to_string(index=False))


if __name__ == "__main__":
    main()

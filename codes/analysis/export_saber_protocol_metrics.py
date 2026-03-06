from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import pandas as pd

from compare_trace_slices import (
    _build_decision_frame,
    _normalize_decision_frame,
    _read_json,
    _severity_bucket,
    _slice_metrics,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT_DIR / "codes" / "analysis" / "outputs" / "saber_protocol"


def _safe_mean(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return math.nan
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return math.nan
    return float(clean.mean())


def _load_meta(run_dir: Path) -> dict:
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        try:
            return _read_json(meta_path)
        except Exception:
            return {}
    return {}


def _build_run_frame(run_dir: Path, run_label: str) -> pd.DataFrame:
    meta = _load_meta(run_dir)
    frame = _build_decision_frame(run_dir)
    if frame.empty:
        return pd.DataFrame()
    out = _normalize_decision_frame(frame, run_label=run_label, run_dir=run_dir, meta=meta)
    out["severity_bucket"] = _severity_bucket(out["severity"])
    return out


def _iter_rows(grouped: Iterable[tuple], group_cols: list[str]) -> list[dict]:
    rows: list[dict] = []
    for keys, g in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        row["n"] = int(len(g))
        row["avg_reward"] = _safe_mean(g["reward"])
        row["pos_rate"] = _safe_mean((g["reward"] > 0).astype(float))
        row["action1_rate"] = _safe_mean(g["action1"])
        row["wait_share"] = _safe_mean(g["wait_like"])
        row["engage_share"] = _safe_mean(g["engage_like"])
        p_nonnull = pd.to_numeric(g.get("p_action1", pd.Series(dtype=float)), errors="coerce").dropna()
        row["mean_p_action1"] = _safe_mean(p_nonnull)
        rows.append(row)
    return rows


def _primary_metrics(df: pd.DataFrame, hard_threshold: int, easy_threshold: int, lambda_fp: float) -> dict:
    impl = df[df["phase"].astype(str) == "implement"].copy()
    hard_rem = impl[(impl["stage_family"] == "removal") & (impl["severity"] >= hard_threshold)]
    hard_ins = impl[(impl["stage_family"] == "insertion") & (impl["severity"] >= hard_threshold)]
    easy = impl[impl["severity"] <= easy_threshold]

    action1_reward_hard = (hard_rem["action1"] * hard_rem["reward"]) if not hard_rem.empty else pd.Series(dtype=float)
    action0_reward_easy = (easy["action0"] * easy["reward"]) if not easy.empty else pd.Series(dtype=float)
    waitlike_reward_easy = (easy["wait_like"] * easy["reward"]) if not easy.empty else pd.Series(dtype=float)

    pipeline_summary_path = Path(str(df["run_dir"].iloc[0])) / "post_stage" / "pipeline_summary.json"
    pipeline_summary = {}
    if pipeline_summary_path.exists():
        try:
            pipeline_summary = json.loads(pipeline_summary_path.read_text(encoding="utf-8"))
        except Exception:
            pipeline_summary = {}

    return {
        "run_label": str(df["run_label"].iloc[0]),
        "run_dir": str(df["run_dir"].iloc[0]),
        "distribution": str(df["distribution"].iloc[0]),
        "algorithm": str(df["algorithm"].iloc[0]),
        "algo_version": str(df["algo_version"].iloc[0]),
        "seed": str(df["seed"].iloc[0]),
        "done_marker": int((Path(str(df["run_dir"].iloc[0])) / "DONE.json").exists()),
        "has_rl_trace": int((Path(str(df["run_dir"].iloc[0])) / "rl_trace.csv").exists()),
        "has_rl_decision": int((Path(str(df["run_dir"].iloc[0])) / "rl_decision.csv").exists()),
        "has_rl_summary": int((Path(str(df["run_dir"].iloc[0])) / "rl_summary.csv").exists()),
        "n_implement": int(len(impl)),
        "avg_reward_implement": _safe_mean(impl["reward"]),
        "Q_hard_rem": _safe_mean(hard_rem["reward"]),
        "R_hard_rem": _safe_mean(action1_reward_hard),
        "hard_rem_action1_rate": _safe_mean(hard_rem["action1"]),
        "hard_rem_wait_share": _safe_mean(hard_rem["wait_like"]),
        "hard_rem_mean_p_action1": _safe_mean(hard_rem["p_action1"]),
        "P_easy_action0": _safe_mean(action0_reward_easy),
        "P_easy_waitlike": _safe_mean(waitlike_reward_easy),
        "easy_action1_rate": _safe_mean(easy["action1"]),
        "M_ins": _safe_mean(hard_ins["reward"]),
        "hard_ins_action1_rate": _safe_mean(hard_ins["action1"]),
        "C_sel_tilde": _safe_mean(action1_reward_hard) - float(lambda_fp) * _safe_mean(easy["action1"]),
        "phase4_ckpt_policy": pipeline_summary.get("phase4_ckpt_policy", ""),
        "phase4_ckpt_source": pipeline_summary.get("phase4_ckpt_source", ""),
        "phase4_ckpt_iter_id": pipeline_summary.get("phase4_ckpt_iter_id", ""),
        "phase4_ckpt_objective_score": pipeline_summary.get("phase4_ckpt_objective_score", ""),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export frozen SABER phase-0/1 metrics from run directories."
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="absolute path to a run_* directory; pass multiple times for multiple runs",
    )
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--hard-threshold", type=int, default=5)
    parser.add_argument("--easy-threshold", type=int, default=3)
    parser.add_argument("--lambda-fp", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(str(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    selected_rows: list[dict] = []
    primary_rows: list[dict] = []

    for raw_run_dir in args.run_dir:
        run_dir = Path(str(raw_run_dir)).resolve()
        run_label = run_dir.name
        frame = _build_run_frame(run_dir=run_dir, run_label=run_label)
        selected_rows.append(
            {
                "run_label": run_label,
                "run_dir": str(run_dir),
                "done_marker": int((run_dir / "DONE.json").exists()),
                "has_rl_trace": int((run_dir / "rl_trace.csv").exists()),
                "has_rl_decision": int((run_dir / "rl_decision.csv").exists()),
                "has_rl_summary": int((run_dir / "rl_summary.csv").exists()),
                "rows_loaded": int(len(frame)),
            }
        )
        if frame.empty:
            continue
        frames.append(frame)
        primary_rows.append(
            _primary_metrics(
                frame,
                hard_threshold=int(args.hard_threshold),
                easy_threshold=int(args.easy_threshold),
                lambda_fp=float(args.lambda_fp),
            )
        )

    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(out_dir / "selected_runs.csv", index=False, encoding="utf-8-sig")

    if not frames:
        manifest = {
            "status": "no_frames_loaded",
            "hard_threshold": int(args.hard_threshold),
            "easy_threshold": int(args.easy_threshold),
            "lambda_fp": float(args.lambda_fp),
        }
        (out_dir / "protocol_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return

    full_df = pd.concat(frames, ignore_index=True)
    impl_df = full_df[full_df["phase"].astype(str) == "implement"].copy()

    primary_df = pd.DataFrame(primary_rows).sort_values(["distribution", "run_label"]).reset_index(drop=True)
    primary_df.to_csv(out_dir / "primary_metrics.csv", index=False, encoding="utf-8-sig")

    _slice_metrics(impl_df, ["run_label", "stage_family"]).to_csv(
        out_dir / "by_stage_family.csv", index=False, encoding="utf-8-sig"
    )
    _slice_metrics(impl_df, ["run_label", "severity"]).to_csv(
        out_dir / "by_severity.csv", index=False, encoding="utf-8-sig"
    )
    _slice_metrics(impl_df, ["run_label", "severity_bucket"]).to_csv(
        out_dir / "by_severity_bucket.csv", index=False, encoding="utf-8-sig"
    )
    _slice_metrics(impl_df, ["run_label", "stage_family", "severity"]).to_csv(
        out_dir / "by_stage_x_severity.csv", index=False, encoding="utf-8-sig"
    )
    _slice_metrics(impl_df, ["run_label", "stage_family", "semantic_action"]).to_csv(
        out_dir / "by_stage_x_semantic_action.csv", index=False, encoding="utf-8-sig"
    )

    hard_threshold = int(args.hard_threshold)
    easy_threshold = int(args.easy_threshold)
    protocol_rows = _iter_rows(
        impl_df.groupby(
            [
                "run_label",
                pd.Series(
                    [
                        "hard_rem"
                        if (sf == "removal" and sev >= hard_threshold)
                        else "hard_ins"
                        if (sf == "insertion" and sev >= hard_threshold)
                        else "easy"
                        if sev <= easy_threshold
                        else "other"
                        for sf, sev in zip(
                            impl_df["stage_family"].astype(str),
                            pd.to_numeric(impl_df["severity"], errors="coerce").fillna(-1),
                        )
                    ],
                    index=impl_df.index,
                    name="protocol_slice",
                ),
            ],
            dropna=False,
        ),
        ["run_label", "protocol_slice"],
    )
    pd.DataFrame(protocol_rows).to_csv(
        out_dir / "protocol_slices.csv", index=False, encoding="utf-8-sig"
    )

    manifest = {
        "status": "ok",
        "hard_threshold": hard_threshold,
        "easy_threshold": easy_threshold,
        "lambda_fp": float(args.lambda_fp),
        "primary_metrics": {
            "Q_hard_rem": "mean reward on implement rows where stage_family=removal and severity>=hard_threshold",
            "R_hard_rem": "mean of action1*reward on implement rows where stage_family=removal and severity>=hard_threshold",
            "P_easy_action0": "mean of action0*reward on implement rows where severity<=easy_threshold",
            "P_easy_waitlike": "mean of wait_like*reward on implement rows where severity<=easy_threshold",
            "M_ins": "mean reward on implement rows where stage_family=insertion and severity>=hard_threshold",
            "C_sel_tilde": "R_hard_rem - lambda_fp * easy_action1_rate",
        },
    }
    (out_dir / "protocol_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT_DIR / "codes" / "analysis" / "outputs" / "phase1_ckpt_ranking"


def _default_python_exe() -> str:
    candidates = [
        ROOT_DIR / "codes" / "env" / "python.exe",
        ROOT_DIR / ".venv" / "Scripts" / "python.exe",
        Path(sys.executable),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(sys.executable)


def _safe_float(value, default: float = float("nan")) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return float(default)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank phase1 historical checkpoints using phase1_ckpt_manifest.csv and checkpoint eval summary."
    )
    parser.add_argument("--manifest-path", type=str, required=True)
    parser.add_argument("--summary-csv", type=str, default="")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--run-eval", action="store_true", help="run eval_checkpoint_pool.py before ranking")
    parser.add_argument("--python-exe", type=str, default=_default_python_exe())
    parser.add_argument("--dist-name", type=str, default="")
    parser.add_argument("--request-number", type=int, default=0)
    parser.add_argument("--algorithm", type=str, default="PPO_NEW")
    parser.add_argument("--algo-version", type=str, default="v3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--external-data-root", type=str, default="")
    parser.add_argument("--timeout-sec", type=int, default=0)
    parser.add_argument("--allow-partial-on-timeout", action="store_true")
    parser.add_argument("--reward-floor-ratio", type=float, default=0.80)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def _load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    if "checkpoint_path" not in df.columns:
        raise ValueError(f"manifest missing checkpoint_path column: {path}")
    df["checkpoint_path"] = df["checkpoint_path"].astype(str)
    if "completed_train_tables" in df.columns:
        df["completed_train_tables"] = pd.to_numeric(df["completed_train_tables"], errors="coerce")
    if "table_number" in df.columns:
        df["table_number"] = pd.to_numeric(df["table_number"], errors="coerce")
    return df.drop_duplicates(subset=["checkpoint_path"], keep="last").reset_index(drop=True)


def _run_eval_from_manifest(args: argparse.Namespace, manifest_df: pd.DataFrame, out_dir: Path) -> Path:
    if not str(args.dist_name).strip() or int(args.request_number) <= 0:
        raise ValueError("--dist-name and --request-number are required when --run-eval is used")
    cmd = [
        str(args.python_exe),
        str(ROOT_DIR / "codes" / "analysis" / "eval_checkpoint_pool.py"),
        "--out-dir",
        str((out_dir / "eval").resolve()),
        "--dist-name",
        str(args.dist_name),
        "--request-number",
        str(int(args.request_number)),
        "--algorithm",
        str(args.algorithm),
        "--algo-version",
        str(args.algo_version),
        "--seed",
        str(int(args.seed)),
        "--workers",
        str(int(args.workers)),
    ]
    if str(args.external_data_root).strip():
        cmd.extend(["--external-data-root", str(Path(args.external_data_root).resolve())])
    if int(args.timeout_sec) > 0:
        cmd.extend(["--timeout-sec", str(int(args.timeout_sec))])
    if bool(args.allow_partial_on_timeout):
        cmd.append("--allow-partial-on-timeout")
    for checkpoint_path in manifest_df["checkpoint_path"].astype(str).tolist():
        cmd.extend(["--checkpoint", str(Path(checkpoint_path).resolve())])
    subprocess.run(cmd, cwd=str(ROOT_DIR), check=True)
    return (out_dir / "eval" / "checkpoint_eval_summary.csv").resolve()


def _load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    for col in ["avg_reward", "hard_action1_rate", "mean_p_action1", "phase_rows"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["checkpoint_path"] = df.get("checkpoint_path", pd.Series([""] * len(df))).astype(str)
    return df


def _rank(merged: pd.DataFrame, reward_floor_ratio: float) -> pd.DataFrame:
    df = merged.copy()
    df["status"] = df.get("status", pd.Series([""] * len(df))).astype(str)
    df["subprocess_status"] = df.get("subprocess_status", pd.Series([""] * len(df))).astype(str)
    df["eval_ok"] = (df["status"] == "ok").astype(int)
    ok_rewards = pd.to_numeric(df.loc[df["eval_ok"] == 1, "avg_reward"], errors="coerce").dropna()
    best_reward = float(ok_rewards.max()) if not ok_rewards.empty else math.nan
    if best_reward == best_reward:
        reward_floor = float(best_reward) * float(reward_floor_ratio)
        df["reward_floor"] = reward_floor
        df["reward_ok"] = (
            pd.to_numeric(df.get("avg_reward"), errors="coerce") >= float(reward_floor)
        ).astype(int)
    else:
        df["reward_floor"] = math.nan
        df["reward_ok"] = 0
    df["rank_candidate"] = ((df["eval_ok"] == 1) & (df["reward_ok"] == 1)).astype(int)
    sort_cols = [
        "rank_candidate",
        "hard_action1_rate",
        "mean_p_action1",
        "avg_reward",
        "completed_train_tables",
    ]
    ascending = [False, False, False, False, False]
    for col in sort_cols:
        if col not in df.columns:
            df[col] = math.nan
    df = df.sort_values(sort_cols, ascending=ascending, na_position="last").reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def main() -> None:
    args = _parse_args()
    out_dir = Path(str(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(str(args.manifest_path)).resolve()
    manifest_df = _load_manifest(manifest_path)
    manifest_df.to_csv(out_dir / "phase1_ckpt_candidates.csv", index=False, encoding="utf-8-sig")

    if bool(args.run_eval):
        summary_path = _run_eval_from_manifest(args, manifest_df, out_dir=out_dir)
    else:
        if not str(args.summary_csv).strip():
            raise ValueError("--summary-csv is required when --run-eval is not used")
        summary_path = Path(str(args.summary_csv)).resolve()

    summary_df = _load_summary(summary_path)
    merged = manifest_df.merge(summary_df, on="checkpoint_path", how="left", suffixes=("_manifest", ""))
    ranked = _rank(merged, reward_floor_ratio=float(args.reward_floor_ratio))
    ranked.to_csv(out_dir / "phase1_ckpt_ranked.csv", index=False, encoding="utf-8-sig")
    top_k = max(1, int(args.top_k))
    ranked.head(top_k).to_csv(out_dir / "phase1_ckpt_topk.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()

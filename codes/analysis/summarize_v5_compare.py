from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def _safe_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        txt = str(v).strip()
        if txt == "":
            return None
        return float(txt)
    except Exception:
        return None


def _read_rl_summary_avg(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        if df.empty or "average_reward" not in df.columns:
            return None
        return _safe_float(df.iloc[0]["average_reward"])
    except Exception:
        return None


def _read_trace_metrics(path: Path) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "action1_rate": None,
        "p_action1": None,
        "reward_given_action0": None,
        "reward_given_action1": None,
        "n_action0": None,
        "n_action1": None,
    }
    if not path.exists():
        return out
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    except Exception:
        return out
    if df.empty:
        return out

    for c in ("phase", "stage", "action", "reward", "p_action1"):
        if c not in df.columns:
            return out

    impl = df[df["phase"].astype(str).str.lower() == "implement"].copy()
    if impl.empty:
        return out

    send = impl[impl["stage"].astype(str) == "send_action"].copy()
    recv = impl[impl["stage"].astype(str) == "receive_reward"].copy()

    for frame in (send, recv):
        if not frame.empty:
            frame["action"] = pd.to_numeric(frame["action"], errors="coerce")
            frame["reward"] = pd.to_numeric(frame["reward"], errors="coerce")
            frame["p_action1"] = pd.to_numeric(frame["p_action1"], errors="coerce")

    if not send.empty and send["action"].notna().any():
        valid = send[send["action"].isin([0, 1])]
        if not valid.empty:
            out["action1_rate"] = _safe_float((valid["action"] == 1).mean())
            p1 = valid["p_action1"].dropna()
            if not p1.empty:
                out["p_action1"] = _safe_float(p1.mean())

    if not recv.empty and recv["action"].notna().any() and recv["reward"].notna().any():
        valid = recv[recv["action"].isin([0, 1])]
        if not valid.empty:
            a0 = valid[valid["action"] == 0]["reward"]
            a1 = valid[valid["action"] == 1]["reward"]
            out["n_action0"] = int(a0.shape[0])
            out["n_action1"] = int(a1.shape[0])
            if not a0.empty:
                out["reward_given_action0"] = _safe_float(a0.mean())
            if not a1.empty:
                out["reward_given_action1"] = _safe_float(a1.mean())
    return out


def collect_runs(root: Path, versions: List[str], dist_filter: Optional[List[str]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    wanted = {v.strip().lower() for v in versions if v.strip()}

    for run_dir in sorted(root.glob("run_*")):
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        algo = str(meta.get("algorithm", "")).strip().upper()
        if algo != "PPO_NEW":
            continue
        algo_version = str(meta.get("algo_version", "v1")).strip().lower()
        if wanted and algo_version not in wanted:
            continue
        dist = str(meta.get("distribution", "")).strip()
        if dist_filter and dist not in dist_filter:
            continue

        avg_reward = _read_rl_summary_avg(run_dir / "rl_summary.csv")
        trace_metrics = _read_trace_metrics(run_dir / "rl_trace.csv")
        rows.append(
            {
                "run_dir": run_dir.name,
                "dist": dist,
                "seed": meta.get("seed"),
                "algo_version": algo_version,
                "avg_reward": avg_reward,
                **trace_metrics,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize PPO_NEW v3 vs v5.x runs from run folders.")
    p.add_argument("--root", required=True, help="run root directory containing run_* folders")
    p.add_argument(
        "--versions",
        action="append",
        default=["v3", "v5.1_abppo", "v5.2_qcritic", "v5.3_auxweak"],
        help="algo_version to include (repeatable)",
    )
    p.add_argument("--dist", action="append", default=None, help="optional distribution filter (repeatable)")
    p.add_argument("--out-csv", default=None, help="output csv path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        print(f"[ERR] root not found: {root}")
        return 2

    df = collect_runs(root=root, versions=args.versions, dist_filter=args.dist)
    if df.empty:
        print("[WARN] no matched runs found.")
        return 1

    summary = (
        df.groupby(["algo_version", "dist"], as_index=False)
        .agg(
            n=("avg_reward", "count"),
            avg_reward_mean=("avg_reward", "mean"),
            avg_reward_std=("avg_reward", "std"),
            action1_rate_mean=("action1_rate", "mean"),
            p_action1_mean=("p_action1", "mean"),
            reward_a0_mean=("reward_given_action0", "mean"),
            reward_a1_mean=("reward_given_action1", "mean"),
            n_action0_sum=("n_action0", "sum"),
            n_action1_sum=("n_action1", "sum"),
        )
        .sort_values(["dist", "algo_version"])
    )
    print("\n=== V3 vs V5 Summary ===")
    print(summary.to_string(index=False))

    out_csv = Path(args.out_csv) if args.out_csv else (root / "v3_v5_compare_summary.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[OK] saved: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

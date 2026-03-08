from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


ROOT_DIR = Path(__file__).resolve().parents[2]
CODES_DIR = ROOT_DIR / "codes"


def _parse_float_list(raw: str) -> List[float]:
    out = []
    for x in str(raw).split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    return out


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _last_train_row(path: Path) -> Dict[str, str]:
    rows = _read_csv_rows(path)
    for row in reversed(rows):
        if str(row.get("phase", "")).strip().lower() != "train":
            continue
        cql_alpha = str(row.get("cql_alpha", "")).strip()
        cql_td = str(row.get("cql_td_loss", "")).strip()
        cql_cql = str(row.get("cql_cql_loss", "")).strip()
        if cql_alpha or cql_td or cql_cql:
            return row
    # Fallback: if no metric-bearing train row exists, return last train row.
    for row in reversed(rows):
        if str(row.get("phase", "")).strip().lower() == "train":
            return row
    return {}


def _last_eval_row(path: Path) -> Dict[str, str]:
    rows = _read_csv_rows(path)
    for row in reversed(rows):
        if str(row.get("phase", "")).strip().lower() == "eval":
            return row
    return {}


def _safe_float(v: object, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CQL alpha ablation and export summary metrics from rl_training.csv"
    )
    parser.add_argument("--alphas", type=str, default="0.1,1,5")
    parser.add_argument("--run-prefix", type=str, default="cql_alpha_ablation")
    parser.add_argument("--dist-name", type=str, default="O_10_90")
    parser.add_argument("--request-number", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed-n", type=int, default=6)
    parser.add_argument("--external-data-root", type=str, default="")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output-csv", type=str, default="")
    args = parser.parse_args()

    alphas = _parse_float_list(args.alphas)
    if not alphas:
        raise ValueError("alphas is empty")
    python_bin = str(Path(args.python_bin).resolve())
    summary_rows: List[Dict[str, object]] = []

    for alpha in alphas:
        alpha_tag = str(alpha).replace(".", "p")
        run_name = f"{args.run_prefix}_a{alpha_tag}_s{int(args.seed)}"
        run_root = (CODES_DIR / "logs" / run_name).resolve()
        env = os.environ.copy()
        env["RL_TRAIN_ONLY_STOP_MODE"] = "fixed_n"
        env["RL_TRAIN_ONLY_FIXED_TABLES"] = str(int(max(1, args.fixed_n)))
        env["RL_TRAIN_ONLY_EARLY_STOP"] = "0"
        env["CQL_ALPHA"] = str(float(alpha))

        cmd = [
            python_bin,
            str(CODES_DIR / "Dynamic_master34959.py"),
            "--dist_name",
            str(args.dist_name),
            "--request_number",
            str(int(args.request_number)),
            "--algorithm",
            "CQL_DQN",
            "--stage-mode",
            "train_only",
            "--run-name",
            run_name,
            "--seed",
            str(int(args.seed)),
            "--workers",
            str(int(max(1, args.workers))),
        ]
        if str(args.external_data_root).strip():
            cmd.extend(
                [
                    "--skip-generator",
                    "--external-data-root",
                    str(Path(args.external_data_root).resolve()),
                ]
            )

        print(f"[CQL-ABLATION] run={run_name} alpha={alpha}")
        subprocess.run(cmd, check=True, env=env)

        train_csv = run_root / "rl_training.csv"
        row = _last_train_row(train_csv)
        eval_row = _last_eval_row(train_csv)
        avg_reward = _safe_float(row.get("avg_reward", ""))
        rolling_avg = _safe_float(row.get("rolling_avg", ""))
        if avg_reward != avg_reward:  # NaN
            avg_reward = _safe_float(eval_row.get("avg_reward", ""))
        if rolling_avg != rolling_avg:  # NaN
            rolling_avg = _safe_float(eval_row.get("rolling_avg", ""))
        summary_rows.append(
            {
                "run_name": run_name,
                "alpha": float(alpha),
                "avg_reward": avg_reward,
                "rolling_avg": rolling_avg,
                "cql_td_loss": _safe_float(row.get("cql_td_loss", "")),
                "cql_cql_loss": _safe_float(row.get("cql_cql_loss", "")),
                "cql_updates": _safe_float(row.get("cql_updates", "")),
                "cql_q_mean": _safe_float(row.get("cql_q_mean", "")),
                "cql_q_std": _safe_float(row.get("cql_q_std", "")),
                "cql_q_max": _safe_float(row.get("cql_q_max", "")),
                "cql_q_taken": _safe_float(row.get("cql_q_taken", "")),
                "cql_lse_q": _safe_float(row.get("cql_lse_q", "")),
                "cql_ood_q_gap": _safe_float(row.get("cql_ood_q_gap", "")),
            }
        )

    out_csv = (
        Path(args.output_csv).resolve()
        if str(args.output_csv).strip()
        else (CODES_DIR / "logs" / f"{args.run_prefix}_summary.csv").resolve()
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summary_rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[CQL-ABLATION] summary={out_csv}")


if __name__ == "__main__":
    main()

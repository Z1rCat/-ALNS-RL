import json
from pathlib import Path

import pandas as pd


def collect_logs(output_path: Path) -> None:
    root = Path("codes/logs")
    runs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_")])
    rows = []
    for run in runs:
        meta_path = run / "meta.json"
        summary_path = run / "rl_summary.csv"
        training_path = run / "rl_training.csv"

        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except UnicodeDecodeError:
                meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))

        summary = {}
        if summary_path.exists():
            try:
                summary = pd.read_csv(summary_path).iloc[0].to_dict()
            except Exception:  # pragma: no cover
                summary = {}

        train_steps = 0
        implement_steps = 0
        train_mean = None
        implement_mean = None
        eval_last = None
        if training_path.exists():
            df = pd.read_csv(training_path, on_bad_lines="skip")
            if "phase" in df.columns:
                train_rewards = pd.to_numeric(df.loc[df["phase"] == "train", "reward"], errors="coerce").dropna()
                implement_rewards = pd.to_numeric(df.loc[df["phase"] == "implement", "reward"], errors="coerce").dropna()
                train_steps = int(train_rewards.shape[0])
                implement_steps = int(implement_rewards.shape[0])
                if train_steps:
                    train_mean = float(train_rewards.mean())
                if implement_steps:
                    implement_mean = float(implement_rewards.mean())
                eval_avg = pd.to_numeric(df.loc[df["phase"] == "eval", "avg_reward"], errors="coerce").dropna()
                if eval_avg.shape[0]:
                    eval_last = float(eval_avg.iloc[-1])

        rows.append(
            {
                "run": run.name,
                "R": meta.get("request_number"),
                "dist": meta.get("distribution"),
                "summary_avg_reward": summary.get("average_reward"),
                "summary_std_reward": summary.get("std_reward"),
                "reward_count": summary.get("reward_count"),
                "removal_wait": summary.get("removal_wait_action"),
                "removal_reroute": summary.get("removal_action"),
                "insertion_accept": summary.get("insertion_action"),
                "insertion_reject": summary.get("insertion_non_action"),
                "train_steps": train_steps,
                "train_mean_reward": train_mean,
                "implement_steps": implement_steps,
                "implement_mean_reward": implement_mean,
                "eval_last_avg_reward": eval_last,
            }
        )

    df = pd.DataFrame(rows)
    agg = (
        df.dropna(subset=["R", "dist"])
        .groupby(["R", "dist"])
        .agg(
            n=("run", "count"),
            avg_reward=("summary_avg_reward", "mean"),
            std_reward=("summary_avg_reward", "std"),
            train_steps=("train_steps", "mean"),
            removal_reroute=("removal_reroute", "mean"),
            removal_wait=("removal_wait", "mean"),
            insertion_accept=("insertion_accept", "mean"),
            insertion_reject=("insertion_reject", "mean"),
        )
        .reset_index()
        .sort_values(["R", "avg_reward"], ascending=[True, False])
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(output_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    out = Path("ALNS_Research_Documentation/rl_logs_aggregate.csv")
    collect_logs(out)
    print("写入", out)

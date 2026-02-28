from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None


RUN_NAME_RE = re.compile(
    r"^run_(?P<date>\d{8})_(?P<time>\d{6})_(?P<micros>\d+)_R(?P<r>\d+)_(?P<dist>.+)_(?P<algo>PPO_NEW|PPO|A2C|DQN)_S(?P<seed>\d+)$"
)


@dataclass
class RunRecord:
    run_dir: str
    algo: str
    algo_version: str
    dist: str
    seed: int
    request_number: int
    avg_reward: float
    reward_count: float
    action1_rate: float
    ts_key: Tuple[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build thesis report assets from thesis_gapfill_patch runs.")
    parser.add_argument("--run-root", default="codes/nexus/thesis_gapfill_patch", help="Directory with run_* folders.")
    parser.add_argument(
        "--out-dir",
        default="ALNS_Research_Documentation/latex/reports/rl_nonstationary_ood_study/assets",
        help="Output folder for csv and figures.",
    )
    return parser.parse_args()


def _safe_float(value: object) -> float:
    if value is None:
        return float("nan")
    txt = str(value).strip()
    if txt == "":
        return float("nan")
    try:
        return float(txt)
    except Exception:
        return float("nan")


def _parse_ts_key(run_name: str) -> Tuple[str, int]:
    m = RUN_NAME_RE.match(run_name)
    if not m:
        return ("", -1)
    return (f"{m.group('date')}_{m.group('time')}", int(m.group("micros")))


def _read_first_row_csv(csv_path: Path) -> Dict[str, str]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
    return row or {}


def collect_completed_runs(run_root: Path) -> pd.DataFrame:
    rows: List[RunRecord] = []
    for run_dir in run_root.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        meta_path = run_dir / "meta.json"
        summary_path = run_dir / "rl_summary.csv"
        if not (meta_path.exists() and summary_path.exists()):
            continue

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        try:
            summary = _read_first_row_csv(summary_path)
        except Exception:
            continue

        algo = str(meta.get("algorithm", "")).strip()
        dist = str(meta.get("distribution", "")).strip()
        seed = int(meta.get("seed"))
        request_number = int(meta.get("request_number"))
        algo_version = str(meta.get("algo_version", "v1")).strip() or "v1"

        avg_reward = _safe_float(summary.get("average_reward"))
        reward_count = _safe_float(summary.get("reward_count"))
        removal_action = _safe_float(summary.get("removal_action"))
        insertion_action = _safe_float(summary.get("insertion_action"))
        action1_rate = (
            (removal_action + insertion_action) / reward_count
            if reward_count == reward_count and reward_count > 0
            else float("nan")
        )

        rows.append(
            RunRecord(
                run_dir=run_dir.name,
                algo=algo,
                algo_version=algo_version,
                dist=dist,
                seed=seed,
                request_number=request_number,
                avg_reward=avg_reward,
                reward_count=reward_count,
                action1_rate=action1_rate,
                ts_key=_parse_ts_key(run_dir.name),
            )
        )

    df = pd.DataFrame([r.__dict__ for r in rows])
    if df.empty:
        return df

    # Keep newest run for each key to avoid duplicated retries.
    df = df.sort_values(["algo", "dist", "seed", "algo_version", "ts_key"])
    latest = df.groupby(["algo", "dist", "seed", "algo_version"], as_index=False).tail(1).copy()
    latest = latest.drop(columns=["ts_key"])
    return latest


def collect_dqn_partial(run_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for run_dir in run_root.iterdir():
        if not run_dir.is_dir() or "_DQN_" not in run_dir.name:
            continue
        meta_path = run_dir / "meta.json"
        training_path = run_dir / "rl_training.csv"
        if not (meta_path.exists() and training_path.exists()):
            continue

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        max_step = np.nan
        last_training_time = np.nan
        last_rolling_avg = np.nan
        with training_path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                step = _safe_float(row.get("step_idx"))
                if step == step:
                    max_step = max(max_step, step) if max_step == max_step else step
                training_time = _safe_float(row.get("training_time"))
                if training_time == training_time:
                    last_training_time = training_time
                rolling = _safe_float(row.get("rolling_avg"))
                if rolling == rolling and rolling >= 0:
                    last_rolling_avg = rolling

        rows.append(
            {
                "run_dir": run_dir.name,
                "dist": meta.get("distribution"),
                "seed": meta.get("seed"),
                "max_step": max_step,
                "training_time_s": last_training_time,
                "last_rolling_avg": last_rolling_avg,
            }
        )
    return pd.DataFrame(rows)


def apply_plot_style() -> None:
    if sns is not None:
        sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_fig(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def fig_baseline_global(df: pd.DataFrame, out_base: Path) -> None:
    base = df[df["algo"].isin(["A2C", "PPO"])].copy()
    order = ["F1_10_90", "G_10_90_60", "G_10_30_60", "O_10_30", "O_10_60", "O_10_90"]
    base["dist"] = pd.Categorical(base["dist"], categories=order, ordered=True)
    base = base.sort_values("dist")

    fig, ax = plt.subplots(figsize=(11.6, 4.6))
    if sns is not None:
        sns.barplot(
            data=base,
            x="dist",
            y="avg_reward",
            hue="algo",
            estimator=np.mean,
            errorbar=("ci", 95),
            capsize=0.06,
            palette={"PPO": "#1f77b4", "A2C": "#ff7f0e"},
            ax=ax,
        )
        sns.stripplot(
            data=base,
            x="dist",
            y="avg_reward",
            hue="algo",
            dodge=True,
            alpha=0.8,
            size=4,
            marker="o",
            linewidth=0.5,
            edgecolor="white",
            palette={"PPO": "#174f82", "A2C": "#b3570a"},
            ax=ax,
        )
        handles, labels = ax.get_legend_handles_labels()
        keep = []
        seen = set()
        for h, l in zip(handles, labels):
            if l in ("PPO", "A2C") and l not in seen:
                keep.append((h, l))
                seen.add(l)
        ax.legend([h for h, _ in keep], [l for _, l in keep], title="Algorithm", ncol=2, frameon=True)
    else:
        # fallback minimal plot
        means = base.groupby(["dist", "algo"])["avg_reward"].mean().unstack()
        means.plot(kind="bar", ax=ax, color=["#ff7f0e", "#1f77b4"])
        ax.legend(title="Algorithm")

    ax.set_ylim(0.45, 1.01)
    ax.set_ylabel("Average Reward")
    ax.set_xlabel("Distribution")
    ax.set_title("Global Baseline Landscape on thesis_gapfill_patch (3 seeds per setting)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    # Mark OOD region
    for idx, dist in enumerate(order):
        if dist.startswith("O_"):
            ax.axvspan(idx - 0.5, idx + 0.5, color="#f5f5f5", zorder=0)
    ax.text(4.1, 0.98, "OOD region", color="#555555", fontsize=10)
    save_fig(fig, out_base)


def fig_ppo_boundary_heatmap(df: pd.DataFrame, out_base: Path) -> None:
    ppo = df[(df["algo"] == "PPO")].copy()
    order = ["F1_10_90", "G_10_90_60", "G_10_30_60", "O_10_30", "O_10_60", "O_10_90"]
    pivot = ppo.pivot_table(index="dist", columns="seed", values="avg_reward", aggfunc="mean")
    pivot = pivot.reindex(order)
    seeds = sorted([c for c in pivot.columns if pd.notna(c)])
    pivot = pivot[seeds]

    fig, ax = plt.subplots(figsize=(6.8, 4.9))
    im = ax.imshow(pivot.values, cmap="YlGnBu", vmin=0.45, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([str(int(s)) for s in seeds])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Distribution")
    ax.set_title("PPO Baseline Boundary on New Distributions")
    cbar = fig.colorbar(im, ax=ax, fraction=0.048, pad=0.03)
    cbar.set_label("Average Reward")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if v == v:
                color = "white" if v < 0.7 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", color=color, fontsize=9)

    save_fig(fig, out_base)


def fig_ppo_id_ood_gap(df: pd.DataFrame, out_base: Path) -> None:
    ppo = df[df["algo"] == "PPO"].copy()
    ood_set = {"O_10_30", "O_10_60", "O_10_90"}
    ppo["group"] = ppo["dist"].map(lambda d: "OOD" if d in ood_set else "Non-OOD")
    grouped = (
        ppo.groupby(["seed", "group"], as_index=False)["avg_reward"]
        .mean()
        .pivot(index="seed", columns="group", values="avg_reward")
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6.9, 4.3))
    x = [0, 1]
    for _, row in grouped.iterrows():
        y0 = row.get("Non-OOD", np.nan)
        y1 = row.get("OOD", np.nan)
        if y0 == y0 and y1 == y1:
            ax.plot(x, [y0, y1], marker="o", linewidth=1.8, alpha=0.85, label=f"seed={int(row['seed'])}")
            ax.text(1.03, y1, f"{int(row['seed'])}", fontsize=9, va="center")

    # mean line
    mean_non = grouped["Non-OOD"].mean()
    mean_ood = grouped["OOD"].mean()
    ax.plot(x, [mean_non, mean_ood], color="black", linewidth=3, marker="D", markersize=6, label="mean")
    ax.text(0.03, mean_non + 0.01, f"mean={mean_non:.3f}", fontsize=10, color="black")
    ax.text(1.03, mean_ood, f"mean={mean_ood:.3f}", fontsize=10, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(["Non-OOD", "OOD"])
    ax.set_ylim(0.45, 0.98)
    ax.set_ylabel("Average Reward")
    ax.set_title("PPO Generalization Gap (same seeds, grouped by distribution)")
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, out_base)


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_plot_style()

    latest = collect_completed_runs(run_root)
    if latest.empty:
        raise SystemExit(f"No completed runs found under: {run_root}")

    dqn_partial = collect_dqn_partial(run_root)

    latest_csv = out_dir / "thesis_gapfill_patch_latest_metrics.csv"
    latest.to_csv(latest_csv, index=False, encoding="utf-8-sig")

    if not dqn_partial.empty:
        dqn_csv = out_dir / "thesis_gapfill_patch_dqn_partial_snapshot.csv"
        dqn_partial.to_csv(dqn_csv, index=False, encoding="utf-8-sig")

    # Aggregated tables for direct LaTeX lookup.
    agg_algo_dist = (
        latest.groupby(["algo", "algo_version", "dist"], as_index=False)
        .agg(
            mean_avg_reward=("avg_reward", "mean"),
            std_avg_reward=("avg_reward", "std"),
            mean_action1_rate=("action1_rate", "mean"),
            count=("avg_reward", "count"),
        )
        .sort_values(["algo", "algo_version", "dist"])
    )
    agg_algo_dist.to_csv(out_dir / "thesis_gapfill_patch_agg_algo_dist.csv", index=False, encoding="utf-8-sig")

    # Main figures for paper section.
    fig_baseline_global(latest, out_dir / "fig_baseline_global_landscape")
    fig_ppo_boundary_heatmap(latest, out_dir / "fig_ppo_boundary_heatmap")
    fig_ppo_id_ood_gap(latest, out_dir / "fig_ppo_generalization_gap")

    print(f"[OK] latest rows: {len(latest)}")
    print(f"[OK] wrote: {latest_csv}")
    print(f"[OK] wrote: {out_dir / 'thesis_gapfill_patch_agg_algo_dist.csv'}")
    if not dqn_partial.empty:
        print(f"[OK] wrote: {out_dir / 'thesis_gapfill_patch_dqn_partial_snapshot.csv'}")
    print(f"[OK] figures in: {out_dir}")


if __name__ == "__main__":
    main()

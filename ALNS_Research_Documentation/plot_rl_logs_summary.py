from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_summary(agg_path: Path, out_dir: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    agg = pd.read_csv(agg_path)
    agg = agg.dropna(subset=["avg_reward"])

    pivot = agg.pivot_table(index="dist", columns="R", values="avg_reward", aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(1.2 * max(3, pivot.shape[1]), 0.6 * max(6, pivot.shape[0])))
    ax = sns.heatmap(
        pivot,
        vmin=0,
        vmax=1,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "平均奖励（=决策正确率）"},
    )
    ax.set_title("不同请求规模 R 与分布下的实施阶段平均奖励", pad=14)
    ax.set_xlabel("请求规模 R")
    ax.set_ylabel("不确定性分布")
    plt.tight_layout()
    heatmap_path = out_dir / "overall_implement_avg_reward_heatmap.png"
    plt.savefig(heatmap_path, dpi=200)
    plt.close()

    rows = []
    for _, row in agg.iterrows():
        if pd.isna(row["avg_reward"]):
            continue
        removal_ratio = _safe_ratio(row.get("removal_reroute"), row.get("removal_wait"))
        insertion_ratio = _safe_ratio(row.get("insertion_accept"), row.get("insertion_reject"))
        rows.append(
            {
                "R": row["R"],
                "dist": row["dist"],
                "avg_reward": row["avg_reward"],
                "removal_ratio": removal_ratio,
                "insertion_ratio": insertion_ratio,
            }
        )

    plot_df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
    Rs = sorted(plot_df["R"].unique())
    colors = {R: plt.cm.tab10(i % 10) for i, R in enumerate(Rs)}

    ax = axes[0]
    for R in Rs:
        sub = plot_df[plot_df["R"] == R]
        ax.scatter(sub["removal_ratio"], sub["avg_reward"], s=60, alpha=0.7, label=f"R={int(R)}", color=colors[R])
    ax.set_title("规划阶段：重新规划比例 vs 平均奖励")
    ax.set_xlabel("规划阶段 Reroute(1) 比例")
    ax.set_ylabel("实施阶段平均奖励")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    for R in Rs:
        sub = plot_df[plot_df["R"] == R]
        ax.scatter(sub["insertion_ratio"], sub["avg_reward"], s=60, alpha=0.7, label=f"R={int(R)}", color=colors[R])
    ax.set_title("插入阶段：Accept(0) 比例 vs 平均奖励")
    ax.set_xlabel("插入阶段 Accept(0) 比例")
    ax.set_ylabel("实施阶段平均奖励")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)), frameon=False)
    fig.suptitle("动作偏好（塌缩）与实施效果的关系", y=1.02)
    fig.tight_layout()
    scatter_path = out_dir / "overall_reward_vs_action_ratio.png"
    fig.savefig(scatter_path)
    plt.close(fig)

    print("heatmap:", heatmap_path)
    print("scatter:", scatter_path)


def _safe_ratio(a, b):
    denom = (a or 0) + (b or 0)
    if denom == 0:
        return None
    return (a or 0) / denom


if __name__ == "__main__":
    agg_path = Path("ALNS_Research_Documentation/rl_logs_aggregate.csv")
    out_dir = Path("ALNS_Research_Documentation/figures_rl_logs")
    plot_summary(agg_path, out_dir)

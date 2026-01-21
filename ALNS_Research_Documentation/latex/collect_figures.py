import shutil
from pathlib import Path

# ========= 1. 路径配置 =========

LOG_ROOT = Path(r"A:\MYpython\34959_RL\codes\logs")
LATEX_FIG_ROOT = Path(
    r"A:\MYpython\34959_RL\ALNS_Research_Documentation\latex\figures"
)

# 只处理你指定的 6 个 run
RUNS = {
    "run_20260120_175128_674609_R30_V1_3_A2C_S42": "run_V1_3_A2C",
    "run_20260120_123246_371223_R30_V1_3_DQN_S42": "run_V1_3_DQN",
    "run_20260120_113326_895986_R30_S3_1_DQN_S42": "run_S3_1_DQN",
    "run_20260120_113326_897492_R30_S3_1_A2C_S42": "run_S3_1_A2C",
    "run_20260119_233229_651769_R30_S5_1_DQN_S42": "run_S5_1_DQN",
    "run_20260119_233229_651769_R30_S5_1_A2C_S42": "run_S5_1_A2C",
}

FIG_NAMES = [
    "fig1_environment.pdf",
    "fig2_adaptation.pdf",
    "fig3_policy_heatmap.pdf",
    "fig4_cumulative_advantage.pdf",
]

# ========= 2. 执行复制 =========

LATEX_FIG_ROOT.mkdir(parents=True, exist_ok=True)

for run_dir_name, short_name in RUNS.items():
    src_dir = LOG_ROOT / run_dir_name / "paper_figures"
    dst_dir = LATEX_FIG_ROOT / short_name

    if not src_dir.exists():
        print(f"[跳过] 未找到源目录: {src_dir}")
        continue

    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n处理 {run_dir_name} -> {short_name}")

    for fig in FIG_NAMES:
        src = src_dir / fig
        dst = dst_dir / fig

        if not src.exists():
            print(f"  [缺失] {fig}")
            continue

        shutil.copy2(src, dst)
        print(f"  [复制] {fig}")

print("\n完成：所有指定 run 的图片已整理到 latex/figures/")

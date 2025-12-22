import json
import math
from pathlib import Path
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# 尝试导入 seaborn 进行美化
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# 忽略不必要的警告
warnings.filterwarnings("ignore")

LOG_ROOT = Path(__file__).resolve().parent.parent / "logs"
SMOOTH_WINDOW = 20  # 增大平滑窗口，曲线更好看

# --- SCI 风格配置 ---
SCI_COLORS = {
    "blue": "#0072B2",   # 训练色
    "red": "#D55E00",    # 测试色 (突出显示)
    "green": "#009E73",  # 辅助色
    "grey": "#7F7F7F",   # 背景色
    "lightblue": "#56B4E9"
}

def set_sci_style():
    """配置全局绘图风格，符合 SCI 顶刊标准 (支持中文)"""
    # 尝试加载中文字体，防止方块乱码
    import platform
    if platform.system() == "Windows":
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
    else:
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False # 解决负号问题
    
    # 字号与线宽
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['lines.linewidth'] = 2.0
    
    # 边框与刻度
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    if HAS_SEABORN:
        sns.set_context("paper", font_scale=1.2)
        sns.set_style("ticks", {"axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.3})

def _format_axis(ax, title=None, xlabel=None, ylabel=None, grid=True):
    """统一美化坐标轴"""
    if title: ax.set_title(title, pad=15, fontweight='bold')
    if xlabel: ax.set_xlabel(xlabel, labelpad=8)
    if ylabel: ax.set_ylabel(ylabel, labelpad=8)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    if grid:
        ax.grid(True, linestyle='--', alpha=0.3)
    else:
        ax.grid(False)

def _read_meta(run_dir: Path) -> dict:
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except: return {}
    return {}

def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists(): return pd.DataFrame()
    try: return pd.read_csv(path)
    except: return pd.DataFrame()

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _ensure_numeric(df: pd.DataFrame, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

def _save_or_note(fig, out_path: Path, note: str):
    if note:
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, note, ha="center", va="center", color='gray')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def _smooth_series(series: pd.Series) -> pd.Series:
    if series is None or series.empty: return series
    window = min(SMOOTH_WINDOW, len(series))
    if window <= 1: return series
    return series.rolling(window=window, min_periods=1, center=True).mean()

# =========================================================
# 新增核心功能：训练与测试对比曲线 (Analyze Overfitting/Collapse)
# =========================================================
def plot_comparison_curve(df_train: pd.DataFrame, df_impl: pd.DataFrame, title: str, out_path: Path):
    """
    绘制 Training vs Implementation 对比图
    这是分析 '过敏反应' (Overreaction) 和 '分布漂移' (Distribution Shift) 的关键图表
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    has_data = False
    
    # 1. 绘制训练曲线 (蓝色背景)
    if not df_train.empty:
        x_train = df_train["step_idx"]
        y_train_smooth = _smooth_series(df_train["reward"])
        ax.plot(x_train, y_train_smooth, color=SCI_COLORS['blue'], 
                label='Training Phase (Smoothed)', alpha=0.7)
        # 绘制原始噪点 (可选，淡化)
        # ax.scatter(x_train, df_train["reward"], s=1, color=SCI_COLORS['blue'], alpha=0.1)
        has_data = True
        
    # 2. 绘制实施曲线 (红色突出)
    if not df_impl.empty:
        # 注意：实施阶段的 step 通常是接着训练阶段的，这里直接画在同一X轴上
        x_impl = df_impl["step_idx"]
        y_impl_smooth = _smooth_series(df_impl["reward"])
        
        ax.plot(x_impl, y_impl_smooth, color=SCI_COLORS['red'], linewidth=2.5,
                label='Implementation Phase')
        
        # 标出分界线
        split_point = x_impl.min()
        ax.axvline(x=split_point, color='black', linestyle='--', alpha=0.5)
        ax.text(split_point, 0.5, ' Phase Switch', rotation=90, va='center', ha='right', fontsize=10, color='gray')
        has_data = True

    if not has_data:
        _save_or_note(fig, out_path, "No Data for Comparison")
        return

    _format_axis(ax, title, "Global Interaction Steps", "Average Reward (Smoothed)")
    
    # 强制 Y 轴范围，方便对比 0 和 1
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc='lower right', frameon=True, framealpha=0.9)
    
    _save_or_note(fig, out_path, None)


# ----------------- 原有绘图函数保留并优化 -----------------

def plot_train_reward(df_train: pd.DataFrame, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if df_train.empty:
        _save_or_note(fig, out_path, "No train data")
        return
    
    df_train = df_train.sort_values("step_idx")
    smoothed = _smooth_series(df_train["reward"])
    
    ax.plot(df_train["step_idx"], df_train["reward"], color=SCI_COLORS['lightblue'], alpha=0.2, label='Raw')
    ax.plot(df_train["step_idx"], smoothed, color=SCI_COLORS['blue'], linewidth=2.0, label='Smoothed')
    
    _format_axis(ax, title, "Training Steps", "Reward")
    ax.legend(loc='best', frameon=False)
    _save_or_note(fig, out_path, None)

def plot_action_distribution(trace: pd.DataFrame, stage_set, title: str, labels, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    phases = ["train", "implement"]
    colors = [SCI_COLORS["blue"], SCI_COLORS["red"]] # 区分训练和测试颜色
    
    for idx, phase in enumerate(phases):
        ax = axes[idx]
        sub = trace[(trace["phase"] == phase) & (trace["stage"].isin(stage_set)) & trace["action"].isin([0, 1])]
        counts = [((sub["action"]==0).sum()), ((sub["action"]==1).sum())]
        
        bars = ax.bar(labels, counts, color=colors[idx], edgecolor='black', alpha=0.8, width=0.6)
        for bar in bars:
            h = bar.get_height()
            if h > 0: ax.text(bar.get_x()+bar.get_width()/2, h, f'{int(h)}', ha='center', va='bottom', fontsize=9)

        ax.set_title(phase.capitalize())
        if idx == 0: _format_axis(ax, ylabel="Count", grid=False)
        else: 
            _format_axis(ax, grid=False)
            ax.spines['left'].set_visible(False)
            ax.tick_params(left=False)

    fig.suptitle(title, fontweight='bold', y=1.02)
    _save_or_note(fig, out_path, None)

def plot_reward_hist(trace: pd.DataFrame, train: pd.DataFrame, title: str, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    phases = ["train", "implement"]
    colors = [SCI_COLORS["blue"], SCI_COLORS["red"]]
    
    for idx, phase in enumerate(phases):
        ax = axes[idx]
        # 合并来源查找数据
        sub = pd.DataFrame()
        if not train.empty: sub = train[(train["phase"] == phase)]
        if sub.empty and not trace.empty: sub = trace[(trace["phase"] == phase)]
        
        if not sub.empty: sub = sub[sub["reward"].isin([0, 1])]
        
        if sub.empty:
            ax.text(0.5, 0.5, "No Data", ha="center")
        else:
            counts = sub["reward"].value_counts().sort_index()
            vals = [counts.get(0, 0), counts.get(1, 0)]
            bars = ax.bar([0, 1], vals, color=colors[idx], edgecolor='black', alpha=0.8, width=0.5)
            for bar in bars:
                h = bar.get_height()
                if h > 0: ax.text(bar.get_x()+bar.get_width()/2, h, f'{int(h)}', ha='center', va='bottom')
            
            ax.set_title(phase.capitalize())
            ax.set_xticks([0, 1])
            if idx == 0: _format_axis(ax, xlabel="Reward", ylabel="Count", grid=False)
            else: 
                _format_axis(ax, xlabel="Reward", grid=False)
                ax.spines['left'].set_visible(False)
                ax.tick_params(left=False)
                
    fig.suptitle(title, fontweight='bold', y=1.02)
    _save_or_note(fig, out_path, None)

# ----------------- 主流程 -----------------

def process_run(run_dir: Path):
    meta = _read_meta(run_dir)
    title_prefix = f"R={meta.get('request_number','?')}, {meta.get('distribution','?')}"

    train_path = run_dir / "rl_training.csv"
    trace_path = run_dir / "rl_trace.csv"

    df_train = _normalize_columns(_safe_read_csv(train_path))
    df_trace = _normalize_columns(_safe_read_csv(trace_path))

    if not df_train.empty: _ensure_numeric(df_train, ["step_idx", "reward"])
    if not df_trace.empty: _ensure_numeric(df_trace, ["action", "reward"])

    # 分离数据
    df_train_phase = pd.DataFrame()
    df_impl_phase = pd.DataFrame()
    if not df_train.empty:
        df_train_phase = df_train[(df_train["phase"] == "train")].sort_values("step_idx")
        df_impl_phase = df_train[(df_train["phase"] == "implement")].sort_values("step_idx")

    set_sci_style()
    print(f"Processing: {run_dir.name}")

    # 1. [新增] 训练 vs 测试 对比曲线 (分析失败案例的神器)
    plot_comparison_curve(
        df_train_phase, df_impl_phase, 
        f"Reward Evolution: Training vs Implementation\n({title_prefix})", 
        run_dir / "comparison_reward_curve.png"
    )

    # 2. 原有的单一训练曲线
    plot_train_reward(df_train_phase, f"Training Reward\n({title_prefix})", run_dir / "train_reward_curve.png")
    
    # 3. 动作分布
    plot_action_distribution(df_trace, {"send_action"}, f"Action Dist: Removal\n({title_prefix})", ["Wait", "Reroute"], run_dir / "action_dist_removal.png")
    plot_action_distribution(df_trace, {"finish_insertion"}, f"Action Dist: Insertion\n({title_prefix})", ["Accept", "Reject"], run_dir / "action_dist_insertion.png")

    # 4. Reward 直方图
    plot_reward_hist(df_trace, df_train, f"Reward Distribution\n({title_prefix})", run_dir / "reward_hist.png")

def main():
    if not LOG_ROOT.exists(): return
    run_dirs = [p for p in LOG_ROOT.iterdir() if p.is_dir() and p.name.startswith("run_")]
    for run_dir in sorted(run_dirs):
        process_run(run_dir)
        print(f"Saved: {run_dir.name}")

if __name__ == "__main__":
    main()
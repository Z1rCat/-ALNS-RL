import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


THIS_DIR = Path(__file__).resolve().parent
CODES_DIR = THIS_DIR.parent
ROOT_DIR = CODES_DIR.parent
NEXUS_DIR = CODES_DIR / "nexus"

DEFAULT_DISTS = [
    "O_10_90",
    "O_10_60",
    "G_10_30_60",
    "G_10_60_90",
]

DEFAULT_VARIANTS = [
    "PPO",
    "PPO_NEW:v1",
    "PPO_NEW:v2",
    "PPO_NEW:v3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command pipeline: run unified experiments, build probe summary, build phase summary, and draw summary plots."
    )
    parser.add_argument(
        "--run-folder",
        type=str,
        default="main_pipeline_run",
        help="run output folder under codes/nexus (or absolute path)",
    )
    parser.add_argument(
        "--report-folder",
        type=str,
        default="main_pipeline_report",
        help="report output folder under codes/nexus (or absolute path)",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        help="variant spec, repeatable. e.g. PPO, PPO_NEW:v3, PPO_NEW:v4.1",
    )
    parser.add_argument(
        "--dist-name",
        action="append",
        default=None,
        help="distribution name, repeatable",
    )
    parser.add_argument(
        "--request-number",
        type=int,
        action="append",
        default=None,
        help="request number R, repeatable",
    )
    parser.add_argument("--seed", type=int, action="append", default=None, help="seed, repeatable")
    parser.add_argument("--max-workers", type=int, default=6, help="parallel workers across scenarios")
    parser.add_argument("--generator-workers", type=int, default=1, help="generator workers for master")
    parser.add_argument("--n-stack", type=int, default=None, help="override PPO_NEW stack size")

    parser.add_argument("--run-baseline", action="store_true", default=False, help="run baseline stage")
    parser.add_argument("--run-paper-plots", action="store_true", default=False, help="run per-run paper_figures stage")
    parser.add_argument("--run-metrics", action="store_true", default=True, help="run metrics stage (default: on)")
    parser.add_argument("--no-run-metrics", action="store_false", dest="run_metrics")

    parser.add_argument("--skip-run", action="store_true", default=False, help="skip training stage")
    parser.add_argument("--skip-probe", action="store_true", default=False, help="skip probe_batch stage")
    parser.add_argument("--skip-summary", action="store_true", default=False, help="skip phase summary stage")
    parser.add_argument(
        "--skip-summary-plots",
        action="store_true",
        default=False,
        help="skip drawing summary plots from phase summary csv",
    )

    parser.add_argument("--probe-n-stack", type=int, default=4, help="n_stack passed to probe_batch")
    parser.add_argument("--probe-modes", type=str, default="A,B", help="modes passed to probe_batch")
    parser.add_argument("--probe-feature-kind", type=str, default="Xt", choices=["xt", "Xt", "both"])
    parser.add_argument(
        "--probe-split-mode",
        type=str,
        default="phase_table",
        choices=["phase_table", "table_only", "phase_only"],
    )
    parser.add_argument("--probe-seed", type=int, default=42, help="seed passed to probe_batch")
    parser.add_argument("--force-reprobe", action="store_true", default=False, help="force regenerate probe report")
    parser.add_argument("--recursive", action="store_true", default=False, help="recursive scan run_* dirs")
    parser.add_argument("--summary-prefix", type=str, default="phase_main", help="summary file prefix")
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser.parse_args()


def _resolve_nexus_path(raw: str) -> Path:
    path = Path(str(raw or "").strip())
    if not path:
        raise ValueError("empty path")
    if path.is_absolute():
        return path.resolve()
    return (NEXUS_DIR / path).resolve()


def _run_cmd(cmd: List[str], cwd: Path) -> int:
    printable = " ".join(shlex.quote(x) for x in cmd)
    print(f"[pipeline] run: {printable}")
    return subprocess.run(cmd, cwd=str(cwd)).returncode


def _build_unified_cmd(args: argparse.Namespace, run_root: Path) -> List[str]:
    variants = [str(v).strip() for v in (args.variant or DEFAULT_VARIANTS) if str(v).strip()]
    dists = [str(d).strip() for d in (args.dist_name or DEFAULT_DISTS) if str(d).strip()]
    request_numbers = [int(x) for x in (args.request_number or [30])]
    seeds = [int(s) for s in (args.seed or [42])]

    cmd = [
        sys.executable,
        str(CODES_DIR / "experiments" / "run_experiments_server_unified.py"),
        "--run-folder",
        str(run_root),
        "--max-workers",
        str(int(args.max_workers)),
        "--generator-workers",
        str(int(args.generator_workers)),
    ]
    for item in variants:
        cmd.extend(["--variant", item])
    for item in dists:
        cmd.extend(["--dist-name", item])
    for item in request_numbers:
        cmd.extend(["--request-number", str(int(item))])
    for item in seeds:
        cmd.extend(["--seed", str(int(item))])
    if args.n_stack is not None:
        cmd.extend(["--n-stack", str(int(args.n_stack))])
    if args.run_baseline:
        cmd.append("--run-baseline")
    else:
        cmd.append("--no-run-baseline")
    if args.run_paper_plots:
        cmd.append("--run-plots")
    else:
        cmd.append("--no-run-plots")
    if args.run_metrics:
        cmd.append("--run-metrics")
    else:
        cmd.append("--no-run-metrics")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def _build_probe_cmd(args: argparse.Namespace, run_root: Path, report_root: Path) -> List[str]:
    cmd = [
        sys.executable,
        str(CODES_DIR / "analysis" / "probe_batch.py"),
        "--root",
        str(run_root),
        "--out-csv",
        str(report_root / "probe_summary.csv"),
        "--copy-reports-dir",
        str(report_root / "probe_reports"),
        "--n-stack",
        str(int(args.probe_n_stack)),
        "--modes",
        str(args.probe_modes),
        "--feature-kind",
        str(args.probe_feature_kind),
        "--split-mode",
        str(args.probe_split_mode),
        "--seed",
        str(int(args.probe_seed)),
    ]
    if args.recursive:
        cmd.append("--recursive")
    if args.force_reprobe:
        cmd.append("--force-reprobe")
    return cmd


def _build_summary_cmd(args: argparse.Namespace, run_root: Path, report_root: Path) -> List[str]:
    cmd = [
        sys.executable,
        str(CODES_DIR / "analysis" / "make_phase0_report.py"),
        "--root",
        str(run_root),
        "--probe_csv",
        str(report_root / "probe_summary.csv"),
        "--out-dir",
        str(report_root),
        "--summary-prefix",
        str(args.summary_prefix),
    ]
    if args.recursive:
        cmd.append("--recursive")
    return cmd


def _draw_summary_plots(summary_csv: Path, out_dir: Path) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as exc:
        print(f"[pipeline] skip summary plots: {type(exc).__name__}: {exc}")
        return []

    if not summary_csv.exists():
        print(f"[pipeline] skip summary plots: summary csv missing: {summary_csv}")
        return []

    try:
        df = pd.read_csv(summary_csv, encoding="utf-8-sig")
    except Exception as exc:
        print(f"[pipeline] skip summary plots: cannot read csv: {type(exc).__name__}: {exc}")
        return []

    if df.empty:
        print("[pipeline] skip summary plots: empty summary table")
        return []

    if "algo" not in df.columns:
        print("[pipeline] skip summary plots: missing column `algo`")
        return []

    algo_series = df["algo"].fillna("").astype(str)
    if "algo_version" in df.columns:
        version_series = df["algo_version"].fillna("").astype(str)
    else:
        version_series = algo_series.copy().map(lambda _: "")

    variant_labels = []
    for idx in range(len(df)):
        algo = algo_series.iloc[idx].strip()
        ver = version_series.iloc[idx].strip()
        if algo.upper() == "PPO_NEW" and ver:
            variant_labels.append(f"{algo}:{ver}")
        else:
            variant_labels.append(algo or "UNKNOWN")
    df["variant_label"] = variant_labels

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []

    if {"dist", "avg_reward", "variant_label"}.issubset(set(df.columns)):
        reward_df = df[["dist", "variant_label", "avg_reward"]].copy()
        reward_df["avg_reward"] = pd.to_numeric(reward_df["avg_reward"], errors="coerce")
        reward_df = reward_df.dropna(subset=["avg_reward"])
        if not reward_df.empty:
            pivot = reward_df.pivot_table(index="dist", columns="variant_label", values="avg_reward", aggfunc="mean")
            ax = pivot.plot(kind="bar", figsize=(12, 6))
            ax.set_title("Average Reward by Distribution and Variant")
            ax.set_xlabel("Distribution")
            ax.set_ylabel("Average Reward")
            ax.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            path = out_dir / "plot_avg_reward_by_dist_variant.png"
            plt.savefig(path, dpi=180)
            plt.close()
            outputs.append(path)

    if {"action1_rate", "avg_reward", "variant_label"}.issubset(set(df.columns)):
        scatter_df = df[["action1_rate", "avg_reward", "variant_label"]].copy()
        scatter_df["action1_rate"] = pd.to_numeric(scatter_df["action1_rate"], errors="coerce")
        scatter_df["avg_reward"] = pd.to_numeric(scatter_df["avg_reward"], errors="coerce")
        scatter_df = scatter_df.dropna(subset=["action1_rate", "avg_reward"])
        if not scatter_df.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            for label, sub in scatter_df.groupby("variant_label"):
                ax.scatter(sub["action1_rate"], sub["avg_reward"], alpha=0.8, s=48, label=label)
            ax.set_title("Action-1 Rate vs Average Reward")
            ax.set_xlabel("Action-1 Rate")
            ax.set_ylabel("Average Reward")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
            plt.tight_layout()
            path = out_dir / "plot_action1_rate_vs_avg_reward.png"
            plt.savefig(path, dpi=180)
            plt.close(fig)
            outputs.append(path)

    gain_cols = [c for c in ("A_delta_best_bacc", "B_delta_best_bacc") if c in df.columns]
    if gain_cols:
        gain_df = df[["variant_label"] + gain_cols].copy()
        for col in gain_cols:
            gain_df[col] = pd.to_numeric(gain_df[col], errors="coerce")
        gain_df = gain_df.groupby("variant_label", as_index=False).mean(numeric_only=True)
        if not gain_df.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            x = list(range(len(gain_df)))
            width = 0.36
            if "A_delta_best_bacc" in gain_df.columns:
                ax.bar(
                    [i - width / 2 for i in x],
                    gain_df["A_delta_best_bacc"].fillna(0.0).tolist(),
                    width=width,
                    label="Mode A",
                )
            if "B_delta_best_bacc" in gain_df.columns:
                ax.bar(
                    [i + width / 2 for i in x],
                    gain_df["B_delta_best_bacc"].fillna(0.0).tolist(),
                    width=width,
                    label="Mode B",
                )
            ax.set_xticks(x)
            ax.set_xticklabels(gain_df["variant_label"].tolist(), rotation=20, ha="right")
            ax.set_title("Probe Delta BACC by Variant")
            ax.set_ylabel("Delta BACC")
            ax.grid(axis="y", alpha=0.25)
            ax.legend()
            plt.tight_layout()
            path = out_dir / "plot_probe_delta_bacc_by_variant.png"
            plt.savefig(path, dpi=180)
            plt.close(fig)
            outputs.append(path)

    return outputs


def main() -> int:
    args = parse_args()
    run_root = _resolve_nexus_path(args.run_folder)
    report_root = _resolve_nexus_path(args.report_folder)
    report_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_root": str(run_root),
        "report_root": str(report_root),
        "summary_prefix": str(args.summary_prefix),
        "steps": [],
    }

    if not args.skip_run:
        run_cmd = _build_unified_cmd(args=args, run_root=run_root)
        manifest["steps"].append({"name": "run", "cmd": run_cmd})
        code = _run_cmd(run_cmd, cwd=ROOT_DIR)
        if code != 0:
            print(f"[pipeline] training stage failed: exit={code}")
            return code

    probe_csv = report_root / "probe_summary.csv"
    if not args.skip_probe:
        probe_cmd = _build_probe_cmd(args=args, run_root=run_root, report_root=report_root)
        manifest["steps"].append({"name": "probe_batch", "cmd": probe_cmd})
        code = _run_cmd(probe_cmd, cwd=ROOT_DIR)
        if code != 0:
            print(f"[pipeline] probe stage failed: exit={code}")
            return code

    summary_csv = report_root / f"{args.summary_prefix}_summary.csv"
    if not args.skip_summary:
        summary_cmd = _build_summary_cmd(args=args, run_root=run_root, report_root=report_root)
        manifest["steps"].append({"name": "phase_summary", "cmd": summary_cmd})
        code = _run_cmd(summary_cmd, cwd=ROOT_DIR)
        if code != 0:
            print(f"[pipeline] summary stage failed: exit={code}")
            return code

    plot_outputs: List[Path] = []
    if not args.skip_summary_plots:
        plot_outputs = _draw_summary_plots(summary_csv=summary_csv, out_dir=report_root)
        for path in plot_outputs:
            print(f"[pipeline] wrote plot: {path}")

    manifest["outputs"] = {
        "probe_summary_csv": str(probe_csv),
        "phase_summary_csv": str(summary_csv),
        "plots": [str(p) for p in plot_outputs],
    }
    manifest_path = report_root / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[pipeline] wrote manifest: {manifest_path}")
    print("[pipeline] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

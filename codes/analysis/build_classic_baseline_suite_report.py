from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
CODES_DIR = THIS_DIR.parent
ROOT_DIR = CODES_DIR.parent
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from experiments.classic_baseline_suite_config import (
    BASELINE_POLICY_LABELS,
    DEFAULT_REQUEST_NUMBERS,
    DEFAULT_SEEDS,
    MAIN_TABLE_DISTS,
    REPORT_ALGORITHM_ORDER,
    REPORT_FAMILY_ORDER,
    TRAINABLE_VARIANTS,
    algorithm_order_key,
    canonical_baseline_label,
    family_of_dist,
    family_order_key,
    validate_distribution_subset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a baseline-aware summary for the classic comparison suite. "
            "Outputs trainable algorithms plus random/always1/always0 as independent rows."
        )
    )
    parser.add_argument("--root", required=True, type=str, help="run root containing run_* folders")
    parser.add_argument("--out-dir", required=True, type=str, help="output directory for aggregated tables and plots")
    parser.add_argument("--recursive", action="store_true", help="recursively discover run_* folders")
    parser.add_argument("--summary-prefix", type=str, default="classic_baseline_suite")
    return parser.parse_args()


def _resolve_path(raw: str) -> Path:
    path = Path(str(raw or "").strip())
    if not path:
        raise ValueError("empty path")
    if path.is_absolute():
        return path.resolve()
    return (ROOT_DIR / path).resolve()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))


def _discover_run_dirs(root: Path, recursive: bool) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"run root not found: {root}")
    if recursive:
        candidates = [p for p in root.rglob("run_*") if p.is_dir()]
    else:
        candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    out: List[Path] = []
    for path in candidates:
        if (path / "meta.json").exists():
            out.append(path.resolve())
    return sorted(set(out))


def _pick_float(metrics: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        if key not in metrics:
            continue
        value = metrics.get(key)
        try:
            if value is None or value == "":
                continue
            return float(value)
        except Exception:
            continue
    return None


def _canonical_trainable_label(raw: str) -> str:
    label = str(raw or "").strip().upper()
    return label


def _collect_rows(run_dirs: Sequence[Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    trainable_rows: List[Dict[str, Any]] = []
    baseline_rows: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        meta = _load_json(run_dir / "meta.json")
        metrics = _load_json(run_dir / "metrics.json")
        if not metrics:
            continue

        dist_name = str(metrics.get("scenario") or meta.get("distribution") or "").strip()
        if not dist_name:
            continue
        request_number = metrics.get("request_number", meta.get("request_number"))
        seed = metrics.get("seed", meta.get("seed"))
        algorithm = _canonical_trainable_label(metrics.get("algorithm") or meta.get("algorithm"))
        if algorithm not in TRAINABLE_VARIANTS:
            continue

        try:
            request_number = int(request_number)
            seed = int(seed)
        except Exception:
            continue

        family = family_of_dist(dist_name)
        run_mtime = float(run_dir.stat().st_mtime)
        run_id = str(metrics.get("run_id") or meta.get("run_name") or run_dir.name)
        avg_reward = _pick_float(metrics, "J_rl_avg", "R_RL")
        nps = _pick_float(metrics, "NPS", "G")

        trainable_rows.append(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "run_mtime": run_mtime,
                "dist": dist_name,
                "family": family,
                "request_number": request_number,
                "seed": seed,
                "algorithm_label": algorithm,
                "avg_reward": avg_reward,
                "nps": nps,
                "kind": "trainable",
                "source_algorithm": algorithm,
                "source_count": 1,
            }
        )

        baseline_specs = [
            ("always0", _pick_float(metrics, "J_a0_avg", "R_wait")),
            ("always1", _pick_float(metrics, "J_a1_avg", "R_reroute")),
            ("random", _pick_float(metrics, "J_rand_avg", "R_random")),
        ]
        for label, reward in baseline_specs:
            if reward is None:
                continue
            baseline_rows.append(
                {
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "run_mtime": run_mtime,
                    "dist": dist_name,
                    "family": family,
                    "request_number": request_number,
                    "seed": seed,
                    "algorithm_label": canonical_baseline_label(label),
                    "avg_reward": reward,
                    "nps": None,
                    "kind": "baseline_raw",
                    "source_algorithm": algorithm,
                    "source_count": 1,
                }
            )

    return pd.DataFrame(trainable_rows), pd.DataFrame(baseline_rows)


def _dedupe_trainable_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = df.sort_values(["dist", "request_number", "seed", "algorithm_label", "run_mtime"])
    work = work.drop_duplicates(
        subset=["dist", "request_number", "seed", "algorithm_label"],
        keep="last",
    )
    return work.reset_index(drop=True)


def _collapse_baseline_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    grouped = (
        df.groupby(["dist", "family", "request_number", "seed", "algorithm_label"], as_index=False)
        .agg(
            avg_reward=("avg_reward", "mean"),
            source_count=("avg_reward", "size"),
            source_algorithms=("source_algorithm", lambda s: ",".join(sorted({str(x) for x in s if str(x).strip()}))),
            run_dirs=("run_dir", lambda s: ",".join(sorted({str(x) for x in s if str(x).strip()}))),
            run_ids=("run_id", lambda s: ",".join(sorted({str(x) for x in s if str(x).strip()}))),
        )
    )
    grouped["kind"] = "baseline"
    grouped["nps"] = None
    return grouped


def _build_seed_level_table(trainable_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    trainable = trainable_df.copy()
    if not trainable.empty:
        trainable["source_algorithms"] = trainable["source_algorithm"]
        trainable["run_dirs"] = trainable["run_dir"]
        trainable["run_ids"] = trainable["run_id"]
    cols = [
        "dist",
        "family",
        "request_number",
        "seed",
        "algorithm_label",
        "avg_reward",
        "nps",
        "kind",
        "source_count",
        "source_algorithms",
        "run_dirs",
        "run_ids",
    ]
    parts = []
    if not trainable.empty:
        parts.append(trainable[cols].copy())
    if not baseline_df.empty:
        parts.append(baseline_df[cols].copy())
    if not parts:
        return pd.DataFrame(columns=cols)
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(
        by=["dist", "seed", "algorithm_label"],
        key=lambda s: s.map(lambda x: algorithm_order_key(str(x))[0]) if s.name == "algorithm_label" else s,
    )
    return out.reset_index(drop=True)


def _aggregate_main_table(seed_level_df: pd.DataFrame) -> pd.DataFrame:
    if seed_level_df.empty:
        return pd.DataFrame(
            columns=["dist", "family", "algorithm_label", "mean_reward", "std_reward", "n_seeds", "mean_nps", "std_nps"]
        )
    grouped = (
        seed_level_df.groupby(["dist", "family", "algorithm_label"], as_index=False)
        .agg(
            mean_reward=("avg_reward", "mean"),
            std_reward=("avg_reward", "std"),
            n_seeds=("seed", "nunique"),
            mean_nps=("nps", "mean"),
            std_nps=("nps", "std"),
        )
    )
    grouped["expected_seeds"] = len(DEFAULT_SEEDS)
    grouped["coverage_ratio"] = grouped["n_seeds"] / float(len(DEFAULT_SEEDS))
    grouped = grouped.sort_values(
        by=["dist", "algorithm_label"],
        key=lambda s: s.map(lambda x: algorithm_order_key(str(x))[0]) if s.name == "algorithm_label" else s,
    )
    return grouped.reset_index(drop=True)


def _aggregate_family_table(seed_level_df: pd.DataFrame) -> pd.DataFrame:
    if seed_level_df.empty:
        return pd.DataFrame(
            columns=["family", "algorithm_label", "mean_reward", "std_reward", "n_rows", "mean_nps", "std_nps"]
        )
    grouped = (
        seed_level_df.groupby(["family", "algorithm_label"], as_index=False)
        .agg(
            mean_reward=("avg_reward", "mean"),
            std_reward=("avg_reward", "std"),
            n_rows=("avg_reward", "size"),
            mean_nps=("nps", "mean"),
            std_nps=("nps", "std"),
        )
    )
    grouped = grouped.sort_values(
        by=["family", "algorithm_label"],
        key=lambda s: s.map(lambda x: family_order_key(str(x))[0]) if s.name == "family" else s.map(lambda x: algorithm_order_key(str(x))[0]),
    )
    return grouped.reset_index(drop=True)


def _build_coverage_table(seed_level_df: pd.DataFrame) -> pd.DataFrame:
    if seed_level_df.empty:
        return pd.DataFrame(columns=["dist", "algorithm_label", "observed_seeds", "expected_seeds", "coverage_ratio"])
    grouped = (
        seed_level_df.groupby(["dist", "algorithm_label"], as_index=False)
        .agg(observed_seeds=("seed", "nunique"))
    )
    grouped["expected_seeds"] = len(DEFAULT_SEEDS)
    grouped["coverage_ratio"] = grouped["observed_seeds"] / float(len(DEFAULT_SEEDS))
    return grouped.sort_values(
        by=["dist", "algorithm_label"],
        key=lambda s: s.map(lambda x: algorithm_order_key(str(x))[0]) if s.name == "algorithm_label" else s,
    ).reset_index(drop=True)


def _build_missing_tables(seed_level_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    observed = {
        (
            str(row["dist"]),
            int(row["request_number"]),
            int(row["seed"]),
            str(row["algorithm_label"]),
        )
        for _, row in seed_level_df.iterrows()
    }
    trainable_missing: List[Dict[str, Any]] = []
    baseline_missing: List[Dict[str, Any]] = []

    for dist_name in MAIN_TABLE_DISTS:
        family = family_of_dist(dist_name)
        for request_number in DEFAULT_REQUEST_NUMBERS:
            for seed in DEFAULT_SEEDS:
                for algo in TRAINABLE_VARIANTS:
                    key = (dist_name, request_number, seed, algo)
                    if key not in observed:
                        trainable_missing.append(
                            {
                                "dist": dist_name,
                                "family": family,
                                "request_number": request_number,
                                "seed": seed,
                                "algorithm_label": algo,
                            }
                        )
                for algo in BASELINE_POLICY_LABELS:
                    key = (dist_name, request_number, seed, algo)
                    if key not in observed:
                        baseline_missing.append(
                            {
                                "dist": dist_name,
                                "family": family,
                                "request_number": request_number,
                                "seed": seed,
                                "algorithm_label": algo,
                            }
                        )
    return pd.DataFrame(trainable_missing), pd.DataFrame(baseline_missing)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _plot_heatmap(main_table_df: pd.DataFrame, out_path: Path) -> Optional[Path]:
    if main_table_df.empty:
        return None
    pivot = (
        main_table_df.pivot_table(index="dist", columns="algorithm_label", values="mean_reward", aggfunc="mean")
        .reindex(index=MAIN_TABLE_DISTS, columns=REPORT_ALGORITHM_ORDER)
    )
    if pivot.empty:
        return None
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(pivot.fillna(float("nan")).values, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns), rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    ax.set_title("Mean Reward Heatmap Across Main Benchmark Distributions")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Reward")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _plot_family_bar(family_df: pd.DataFrame, out_path: Path) -> Optional[Path]:
    if family_df.empty:
        return None
    families = [f for f in REPORT_FAMILY_ORDER if f in set(family_df["family"].astype(str))]
    algos = [a for a in REPORT_ALGORITHM_ORDER if a in set(family_df["algorithm_label"].astype(str))]
    if not families or not algos:
        return None

    x = list(range(len(families)))
    width = 0.9 / max(1, len(algos))
    fig, ax = plt.subplots(figsize=(14, 6))
    for idx, algo in enumerate(algos):
        sub = family_df[family_df["algorithm_label"] == algo].copy()
        sub = sub.set_index("family").reindex(families)
        offsets = [pos - 0.45 + width / 2 + idx * width for pos in x]
        ax.bar(
            offsets,
            sub["mean_reward"].fillna(0.0).tolist(),
            width=width,
            label=algo,
            yerr=sub["std_reward"].fillna(0.0).tolist(),
            capsize=2,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(families)
    ax.set_title("Family-Level Mean Reward Comparison")
    ax.set_ylabel("Mean Reward")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _plot_coverage(coverage_df: pd.DataFrame, out_path: Path) -> Optional[Path]:
    if coverage_df.empty:
        return None
    pivot = (
        coverage_df.pivot_table(index="dist", columns="algorithm_label", values="observed_seeds", aggfunc="max")
        .reindex(index=MAIN_TABLE_DISTS, columns=REPORT_ALGORITHM_ORDER)
    )
    if pivot.empty:
        return None
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(pivot.fillna(0.0).values, aspect="auto", vmin=0, vmax=len(DEFAULT_SEEDS))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns), rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    ax.set_title("Seed Coverage Heatmap")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Observed Seeds")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _write_summary_markdown(
    out_path: Path,
    *,
    run_count: int,
    trainable_rows: int,
    baseline_rows: int,
    seed_level_rows: int,
    trainable_missing: int,
    baseline_missing: int,
    coverage_df: pd.DataFrame,
) -> None:
    coverage_lines = []
    if not coverage_df.empty:
        cov = coverage_df.groupby("algorithm_label", as_index=False)["observed_seeds"].mean()
        cov = cov.sort_values("algorithm_label", key=lambda s: s.map(lambda x: algorithm_order_key(str(x))[0]))
        for _, row in cov.iterrows():
            coverage_lines.append(
                f"- `{row['algorithm_label']}`: average observed seeds per distribution = {float(row['observed_seeds']):.2f}"
            )

    text = "\n".join(
        [
            "# Classic Baseline Suite Summary",
            "",
            f"- discovered run directories: `{run_count}`",
            f"- trainable run rows: `{trainable_rows}`",
            f"- raw baseline rows: `{baseline_rows}`",
            f"- seed-level rows after dedupe/collapse: `{seed_level_rows}`",
            f"- missing trainable cells: `{trainable_missing}`",
            f"- missing baseline cells: `{baseline_missing}`",
            "",
            "## Coverage",
            *(coverage_lines or ["- no coverage rows available"]),
        ]
    )
    out_path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = _resolve_path(args.root)
    out_dir = _resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    validate_distribution_subset(MAIN_TABLE_DISTS)

    run_dirs = _discover_run_dirs(root=root, recursive=bool(args.recursive))
    trainable_df_raw, baseline_df_raw = _collect_rows(run_dirs)
    trainable_df = _dedupe_trainable_rows(trainable_df_raw)
    baseline_df = _collapse_baseline_rows(baseline_df_raw)
    seed_level_df = _build_seed_level_table(trainable_df=trainable_df, baseline_df=baseline_df)
    main_table_df = _aggregate_main_table(seed_level_df)
    family_table_df = _aggregate_family_table(seed_level_df)
    coverage_df = _build_coverage_table(seed_level_df)
    missing_trainable_df, missing_baseline_df = _build_missing_tables(seed_level_df)

    if main_table_df.empty:
        wide_mean_df = pd.DataFrame(columns=["dist"] + REPORT_ALGORITHM_ORDER)
        wide_std_df = pd.DataFrame(columns=["dist"] + REPORT_ALGORITHM_ORDER)
    else:
        wide_mean_df = (
            main_table_df[["dist", "algorithm_label", "mean_reward"]]
            .pivot_table(index="dist", columns="algorithm_label", values="mean_reward", aggfunc="mean")
            .reindex(index=MAIN_TABLE_DISTS, columns=REPORT_ALGORITHM_ORDER)
            .reset_index()
        )
        wide_std_df = (
            main_table_df[["dist", "algorithm_label", "std_reward"]]
            .pivot_table(index="dist", columns="algorithm_label", values="std_reward", aggfunc="mean")
            .reindex(index=MAIN_TABLE_DISTS, columns=REPORT_ALGORITHM_ORDER)
            .reset_index()
        )

    prefix = str(args.summary_prefix or "classic_baseline_suite").strip()
    _write_csv(trainable_df_raw, out_dir / f"{prefix}_trainable_run_rows.csv")
    _write_csv(baseline_df_raw, out_dir / f"{prefix}_baseline_raw_rows.csv")
    _write_csv(seed_level_df, out_dir / f"{prefix}_seed_level.csv")
    _write_csv(main_table_df, out_dir / f"{prefix}_main_table_long.csv")
    _write_csv(wide_mean_df, out_dir / f"{prefix}_main_table_wide_mean.csv")
    _write_csv(wide_std_df, out_dir / f"{prefix}_main_table_wide_std.csv")
    _write_csv(family_table_df, out_dir / f"{prefix}_family_table.csv")
    _write_csv(coverage_df, out_dir / f"{prefix}_coverage.csv")
    _write_csv(missing_trainable_df, out_dir / f"{prefix}_missing_trainable.csv")
    _write_csv(missing_baseline_df, out_dir / f"{prefix}_missing_baseline.csv")

    outputs = {
        "heatmap": str(out_dir / f"{prefix}_plot_reward_heatmap.png"),
        "family_bar": str(out_dir / f"{prefix}_plot_family_bar.png"),
        "coverage_heatmap": str(out_dir / f"{prefix}_plot_seed_coverage.png"),
    }
    _plot_heatmap(main_table_df, Path(outputs["heatmap"]))
    _plot_family_bar(family_table_df, Path(outputs["family_bar"]))
    _plot_coverage(coverage_df, Path(outputs["coverage_heatmap"]))

    summary_payload = {
        "run_root": str(root),
        "out_dir": str(out_dir),
        "summary_prefix": prefix,
        "expected_trainable_tasks": len(MAIN_TABLE_DISTS) * len(DEFAULT_REQUEST_NUMBERS) * len(DEFAULT_SEEDS) * len(TRAINABLE_VARIANTS),
        "expected_baseline_cells": len(MAIN_TABLE_DISTS) * len(DEFAULT_REQUEST_NUMBERS) * len(DEFAULT_SEEDS) * len(BASELINE_POLICY_LABELS),
        "discovered_runs": len(run_dirs),
        "trainable_rows": int(len(trainable_df)),
        "baseline_rows": int(len(baseline_df)),
        "seed_level_rows": int(len(seed_level_df)),
        "missing_trainable_rows": int(len(missing_trainable_df)),
        "missing_baseline_rows": int(len(missing_baseline_df)),
        "plots": outputs,
    }
    (out_dir / f"{prefix}_manifest.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_summary_markdown(
        out_path=out_dir / f"{prefix}_summary.md",
        run_count=len(run_dirs),
        trainable_rows=int(len(trainable_df)),
        baseline_rows=int(len(baseline_df_raw)),
        seed_level_rows=int(len(seed_level_df)),
        trainable_missing=int(len(missing_trainable_df)),
        baseline_missing=int(len(missing_baseline_df)),
        coverage_df=coverage_df,
    )
    print(f"[classic-suite-report] wrote outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

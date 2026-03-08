from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
CODES_DIR = THIS_DIR.parent
ROOT_DIR = CODES_DIR.parent
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from checkpoint_eval_common import summarize_run_dir


VARIANT_ORDER = ["main", "actor_rollback", "rollback"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a recovery A/B suite into one comparison table and a few simple plots."
    )
    parser.add_argument("--suite-root", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    return parser.parse_args()


def _resolve_path(raw: str, *, base: Path) -> Path:
    path = Path(str(raw or "").strip())
    if not path:
        raise ValueError("empty path")
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def _pick_compare_csv(summary: Dict[str, Any], variant: str) -> str:
    if variant == "actor_rollback":
        return str(summary.get("actor_rollback_compare_csv", "") or "")
    if variant == "rollback":
        return str(summary.get("rollback_compare_csv", "") or "")
    return ""


def _pick_selected_branch(summary: Dict[str, Any], variant: str) -> str:
    if variant == "actor_rollback":
        return str(summary.get("actor_rollback_selected_branch", "") or "")
    if variant == "rollback":
        return str(summary.get("rollback_selected_branch", "") or "")
    return "main"


def _pick_trigger_metric(summary: Dict[str, Any], variant: str) -> str:
    if variant == "actor_rollback":
        return str(summary.get("actor_rollback_trigger_metric", "") or "")
    if variant == "rollback":
        return str(summary.get("rollback_trigger_metric", "") or "")
    return ""


def _pick_trigger_value(summary: Dict[str, Any], variant: str) -> Any:
    if variant == "actor_rollback":
        return summary.get("actor_rollback_main_metric_value", "")
    if variant == "rollback":
        return summary.get("rollback_main_metric_value", "")
    return ""


def _pick_selected_ckpt(summary: Dict[str, Any], variant: str) -> str:
    if variant == "actor_rollback":
        return str(summary.get("actor_rollback_selected_ckpt", "") or "")
    if variant == "rollback":
        return str(summary.get("rollback_selected_ckpt", "") or "")
    return str(summary.get("phase4_init_ckpt", "") or "")


def _collect_compare_rows(compare_csv: Path, suite_variant: str) -> pd.DataFrame:
    if not compare_csv.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(compare_csv, encoding="utf-8-sig")
    except Exception:
        try:
            df = pd.read_csv(compare_csv)
        except Exception:
            return pd.DataFrame()
    if df.empty:
        return df
    df.insert(0, "suite_variant", str(suite_variant))
    return df


def _variant_run_dir(suite_root: Path, variant: str) -> Path:
    return (suite_root / variant).resolve()


def _build_summary_rows(suite_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, Any]] = []
    compare_frames: List[pd.DataFrame] = []

    for variant in VARIANT_ORDER:
        run_dir = _variant_run_dir(suite_root, variant)
        if not run_dir.exists():
            summary_rows.append(
                {
                    "suite_variant": variant,
                    "run_dir": str(run_dir),
                    "status": "missing_run_dir",
                }
            )
            continue

        pipeline_summary = _load_json(run_dir / "post_stage" / "pipeline_summary.json")
        run_metrics = summarize_run_dir(run_dir, run_label=variant)

        row: Dict[str, Any] = dict(run_metrics)
        row["suite_variant"] = variant
        row["pipeline_summary_exists"] = int(bool(pipeline_summary))
        row["phase4_init_ckpt"] = str(pipeline_summary.get("phase4_init_ckpt", "") or "")
        row["phase4_ckpt_source"] = str(pipeline_summary.get("phase4_ckpt_source", "") or "")
        row["phase4_ckpt_policy"] = str(pipeline_summary.get("phase4_ckpt_policy", "") or "")
        row["phase4_ckpt_iter_id"] = str(pipeline_summary.get("phase4_ckpt_iter_id", "") or "")
        row["phase4_ckpt_objective_score"] = pipeline_summary.get("phase4_ckpt_objective_score", "")
        row["selected_branch"] = _pick_selected_branch(pipeline_summary, variant)
        row["selected_ckpt"] = _pick_selected_ckpt(pipeline_summary, variant)
        row["compare_csv"] = _pick_compare_csv(pipeline_summary, variant)
        row["trigger_metric"] = _pick_trigger_metric(pipeline_summary, variant)
        row["trigger_value"] = _pick_trigger_value(pipeline_summary, variant)
        row["actor_rollback_enabled"] = pipeline_summary.get("actor_rollback_enabled", "")
        row["actor_rollback_triggered"] = pipeline_summary.get("actor_rollback_triggered", "")
        row["rollback_enabled"] = pipeline_summary.get("rollback_enabled", "")
        row["rollback_triggered"] = pipeline_summary.get("rollback_triggered", "")
        summary_rows.append(row)

        compare_csv_raw = str(row.get("compare_csv", "") or "").strip()
        if compare_csv_raw:
            compare_df = _collect_compare_rows(Path(compare_csv_raw), suite_variant=variant)
            if not compare_df.empty:
                compare_frames.append(compare_df)

    summary_df = pd.DataFrame(summary_rows)
    compare_df = pd.concat(compare_frames, ignore_index=True) if compare_frames else pd.DataFrame()
    return summary_df, compare_df


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _plot_metric_bars(summary_df: pd.DataFrame, out_path: Path) -> Optional[Path]:
    if summary_df.empty:
        return None
    metric_cols = [col for col in ["avg_reward", "action1_rate", "hard_action1_rate"] if col in summary_df.columns]
    clean = summary_df.copy()
    if "suite_variant" not in clean.columns:
        return None
    for col in metric_cols:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean = clean[clean["suite_variant"].astype(str).isin(VARIANT_ORDER)].copy()
    if clean.empty or not metric_cols:
        return None

    x = list(range(len(clean)))
    width = 0.24 if len(metric_cols) >= 3 else 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    offsets = {
        1: [0.0],
        2: [-width / 2, width / 2],
        3: [-width, 0.0, width],
    }[len(metric_cols)]

    for idx, col in enumerate(metric_cols):
        bars_x = [pos + offsets[idx] for pos in x]
        ax.bar(bars_x, clean[col].fillna(0.0).tolist(), width=width, label=col)
    ax.set_xticks(x)
    ax.set_xticklabels(clean["suite_variant"].astype(str).tolist())
    ax.set_title("Recovery Suite Comparison")
    ax.set_ylabel("Metric Value")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _write_markdown(summary_df: pd.DataFrame, out_path: Path) -> None:
    lines = ["# Recovery A/B Suite Summary", ""]
    if summary_df.empty:
        lines.append("- no runs discovered")
    else:
        for variant in VARIANT_ORDER:
            sub = summary_df[summary_df["suite_variant"].astype(str) == variant]
            if sub.empty:
                lines.append(f"- `{variant}`: missing")
                continue
            row = sub.iloc[0]
            lines.append(
                f"- `{variant}`: status=`{row.get('status', '')}`, "
                f"avg_reward=`{row.get('avg_reward', '')}`, "
                f"action1_rate=`{row.get('action1_rate', '')}`, "
                f"hard_action1_rate=`{row.get('hard_action1_rate', '')}`, "
                f"selected_branch=`{row.get('selected_branch', '')}`"
            )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    suite_root = _resolve_path(args.suite_root, base=ROOT_DIR)
    out_dir = _resolve_path(args.out_dir, base=ROOT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df, compare_df = _build_summary_rows(suite_root)
    if not summary_df.empty:
        summary_df["suite_variant"] = pd.Categorical(summary_df["suite_variant"], categories=VARIANT_ORDER, ordered=True)
        summary_df = summary_df.sort_values("suite_variant").reset_index(drop=True)

    _write_csv(summary_df, out_dir / "recovery_ab_suite_summary.csv")
    _write_csv(compare_df, out_dir / "recovery_ab_suite_branch_compare_rows.csv")
    plot_path = _plot_metric_bars(summary_df, out_dir / "recovery_ab_suite_plot_metrics.png")
    _write_markdown(summary_df, out_dir / "recovery_ab_suite_summary.md")

    manifest = {
        "suite_root": str(suite_root),
        "out_dir": str(out_dir),
        "summary_csv": str((out_dir / "recovery_ab_suite_summary.csv").resolve()),
        "branch_compare_csv": str((out_dir / "recovery_ab_suite_branch_compare_rows.csv").resolve()),
        "plot_metrics": "" if plot_path is None else str(plot_path.resolve()),
    }
    (out_dir / "recovery_ab_suite_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[recovery-ab-summary] wrote outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

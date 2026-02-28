#!/usr/bin/env python3
"""
Batch probe runner/collector for Phase 0 reporting.

For each run directory:
1) Reuse existing probe_report.json when available.
2) Otherwise call probe_one_run(...) from probe_upper_bound.py.
3) Extract key metrics into a centralized CSV.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    from probe_upper_bound import probe_one_run
except ImportError:
    from codes.analysis.probe_upper_bound import probe_one_run


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PHASE0_DIR = ROOT_DIR / "codes" / "nexus" / "phase0_report"
DEFAULT_SUMMARY_CSV = DEFAULT_PHASE0_DIR / "probe_summary.csv"
DEFAULT_REPORT_NAME = "probe_report.json"
EXPECTED_MODES = ("mode_A_Xt", "mode_B_Xt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch run/collect upper-bound probe reports.")
    parser.add_argument("--root", required=True, type=str, help="Directory containing run_* folders.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan run_* directories (default: non-recursive).",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default=DEFAULT_REPORT_NAME,
        help="Probe report file name inside each run dir.",
    )
    parser.add_argument(
        "--force-reprobe",
        action="store_true",
        help="Re-generate probe report even if it already exists.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="A,B",
        help="Modes passed to probe generator when report missing.",
    )
    parser.add_argument(
        "--feature-kind",
        type=str,
        default="Xt",
        choices=["xt", "Xt", "both"],
        help="Feature kind passed to probe generator when report missing.",
    )
    parser.add_argument("--n-stack", type=int, default=4, help="n_stack passed to probe generator.")
    parser.add_argument(
        "--split-mode",
        type=str,
        default="phase_table",
        choices=["phase_table", "table_only", "phase_only"],
        help="Split mode passed to probe generator.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed passed to probe generator.")
    parser.add_argument(
        "--min-action-samples",
        type=int,
        default=20,
        help="Minimum action samples for optimistic proxy in probe generator.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=str(DEFAULT_SUMMARY_CSV),
        help="Centralized probe summary CSV output path.",
    )
    parser.add_argument(
        "--copy-reports-dir",
        type=str,
        default=str(DEFAULT_PHASE0_DIR / "probe_reports"),
        help="Copy each per-run probe report JSON to this directory.",
    )
    return parser.parse_args()


def discover_run_dirs(root: Path, recursive: bool) -> List[Path]:
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"root not found: {root}")
    if root.is_dir() and (root / "rl_trace.csv").exists():
        return [root]

    if recursive:
        candidates = [p for p in root.rglob("run_*") if p.is_dir()]
    else:
        candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_")]

    run_dirs = [p.resolve() for p in candidates if (p / "rl_trace.csv").exists()]
    run_dirs = sorted(set(run_dirs))
    if not run_dirs:
        raise FileNotFoundError(f"no run_* directories with rl_trace.csv under: {root}")
    return run_dirs


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_meta(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "meta.json"
    if not path.exists():
        return {}
    try:
        return _read_json(path)
    except Exception:
        return {}


def _nested(obj: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict):
            return default
        if key not in cur:
            return default
        cur = cur[key]
    return cur


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _read_or_generate_report(
    run_dir: Path,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], str, Path]:
    report_path = run_dir / str(args.report_name)
    if report_path.exists() and not args.force_reprobe:
        return _read_json(report_path), "existing", report_path

    report = probe_one_run(
        run_dir=run_dir,
        feature_kind=str(args.feature_kind),
        n_stack=int(args.n_stack),
        split_mode=str(args.split_mode),
        seed=int(args.seed),
        min_action_samples=int(args.min_action_samples),
        modes=str(args.modes),
        report_name=str(args.report_name),
        write_report=True,
    )
    return report, "generated", report_path


def _extract_row(
    run_dir: Path,
    report: Dict[str, Any],
    mode_name: str,
    report_source: str,
) -> Dict[str, Any]:
    meta = _load_meta(run_dir)
    run_id = str(report.get("run_id") or run_dir.name)
    scenario_id = str(report.get("scenario_id") or meta.get("distribution") or "")
    algo = str(meta.get("algorithm") or "")
    seed = _safe_int(meta.get("seed"))

    entry = _nested(report, ("entries", mode_name), default={}) or {}
    build_stats = entry.get("build_stats", {}) if isinstance(entry, dict) else {}
    evaluation = entry.get("evaluation", {}) if isinstance(entry, dict) else {}
    split_summary = evaluation.get("split_summary", {}) if isinstance(evaluation, dict) else {}
    baselines = evaluation.get("baselines", {}) if isinstance(evaluation, dict) else {}
    models = evaluation.get("models", {}) if isinstance(evaluation, dict) else {}

    baseline_bacc = _safe_float(
        _nested(baselines, ("action_rate", "overall", "balanced_accuracy"))
    )
    if baseline_bacc is None:
        baseline_bacc = _safe_float(
            _nested(baselines, ("majority", "overall", "balanced_accuracy"))
        )

    logreg_bacc = _safe_float(_nested(models, ("logreg", "overall", "balanced_accuracy")))
    mlp_bacc = _safe_float(_nested(models, ("mlp", "overall", "balanced_accuracy")))
    logreg_roc_auc = _safe_float(_nested(models, ("logreg", "overall", "roc_auc")))
    mlp_roc_auc = _safe_float(_nested(models, ("mlp", "overall", "roc_auc")))
    n_train = _safe_int(split_summary.get("n_train"))
    n_test = _safe_int(split_summary.get("n_test"))

    delta_logreg = (
        (logreg_bacc - baseline_bacc) if (logreg_bacc is not None and baseline_bacc is not None) else None
    )
    delta_mlp = (mlp_bacc - baseline_bacc) if (mlp_bacc is not None and baseline_bacc is not None) else None

    notes: List[str] = [f"probe_source={report_source}"]
    dropped_keys = [
        "dropped_missing_stage_bit",
        "dropped_missing_obs_now",
        "dropped_invalid_action_reward",
        "fallback_prev_obs_count",
    ]
    for key in dropped_keys:
        value = _safe_int(build_stats.get(key))
        if value is not None and value > 0:
            notes.append(f"{key}={value}")

    if not entry:
        notes.append("missing_mode_entry")
    if n_train is not None and n_train <= 0:
        notes.append("empty_train")
    if n_test is not None and n_test <= 0:
        notes.append("empty_test")

    return {
        "run_dir": str(run_dir.resolve()),
        "run_id": run_id,
        "dist": scenario_id,
        "algo": algo,
        "seed": seed,
        "mode": mode_name,
        "baseline_bacc": baseline_bacc,
        "logreg_bacc": logreg_bacc,
        "mlp_bacc": mlp_bacc,
        "logreg_roc_auc": logreg_roc_auc,
        "mlp_roc_auc": mlp_roc_auc,
        "delta_logreg_bacc": delta_logreg,
        "delta_mlp_bacc": delta_mlp,
        "n_train": n_train,
        "n_test": n_test,
        "notes": ";".join(notes),
    }


def _copy_report_to_hub(report_path: Path, run_id: str, hub_dir: Path) -> Optional[Path]:
    if not report_path.exists():
        return None
    hub_dir.mkdir(parents=True, exist_ok=True)
    target = hub_dir / f"{run_id}.json"
    shutil.copy2(report_path, target)
    return target


def _print_brief(df: pd.DataFrame) -> None:
    cols = [
        "run_id",
        "mode",
        "baseline_bacc",
        "logreg_bacc",
        "mlp_bacc",
        "delta_logreg_bacc",
        "delta_mlp_bacc",
        "n_train",
        "n_test",
    ]
    view = df[cols].copy() if all(c in df.columns for c in cols) else df.copy()
    with pd.option_context("display.max_rows", 200, "display.max_columns", 50, "display.width", 180):
        print(view.to_string(index=False))


def main() -> int:
    args = parse_args()

    root = Path(args.root).resolve()
    out_csv = Path(args.out_csv).resolve()
    copy_reports_dir = Path(args.copy_reports_dir).resolve()

    run_dirs = discover_run_dirs(root=root, recursive=bool(args.recursive))
    rows: List[Dict[str, Any]] = []

    print(f"[probe_batch] root={root}")
    print(f"[probe_batch] run_count={len(run_dirs)}")

    for run_dir in run_dirs:
        try:
            report, source, report_path = _read_or_generate_report(run_dir=run_dir, args=args)
        except Exception as exc:
            print(f"[WARN] skip {run_dir.name}: probe load/generate failed: {type(exc).__name__}: {exc}")
            for mode in EXPECTED_MODES:
                rows.append(
                    {
                        "run_dir": str(run_dir.resolve()),
                        "run_id": run_dir.name,
                        "dist": "",
                        "algo": "",
                        "seed": None,
                        "mode": mode,
                        "baseline_bacc": None,
                        "logreg_bacc": None,
                        "mlp_bacc": None,
                        "logreg_roc_auc": None,
                        "mlp_roc_auc": None,
                        "delta_logreg_bacc": None,
                        "delta_mlp_bacc": None,
                        "n_train": None,
                        "n_test": None,
                        "notes": f"probe_failed:{type(exc).__name__}:{exc}",
                    }
                )
            continue

        run_id = str(report.get("run_id") or run_dir.name)
        copied = _copy_report_to_hub(report_path=report_path, run_id=run_id, hub_dir=copy_reports_dir)
        if copied is not None:
            print(f"[probe_batch] {run_id}: report={source}, copied->{copied}")
        else:
            print(f"[probe_batch] {run_id}: report={source}")

        for mode in EXPECTED_MODES:
            rows.append(_extract_row(run_dir=run_dir, report=report, mode_name=mode, report_source=source))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("no rows extracted for probe summary")

    df = df.sort_values(by=["run_id", "mode"], kind="stable")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("")
    print(f"[probe_batch] wrote: {out_csv}")
    _print_brief(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

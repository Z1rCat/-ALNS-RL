#!/usr/bin/env python3
"""
Build a centralized Phase-0 report:
- Merge RL metrics (avg_reward from rl_summary.csv, metrics.json if available)
- Merge action-structure metrics computed from rl_trace.csv
- Merge probe summary (mode_A_Xt / mode_B_Xt)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PHASE0_DIR = ROOT_DIR / "codes" / "nexus" / "phase0_report"
DEFAULT_PROBE_CSV = DEFAULT_PHASE0_DIR / "probe_summary.csv"
INVALID_SENTINEL = -10000000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate centralized Phase-0 summary report.")
    parser.add_argument("--root", required=True, type=str, help="Directory containing run_* folders.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan run_* directories (default: non-recursive).",
    )
    parser.add_argument(
        "--probe_csv",
        type=str,
        default=str(DEFAULT_PROBE_CSV),
        help="Probe summary CSV path produced by probe_batch.py",
    )
    parser.add_argument(
        "--action_csv",
        type=str,
        default="",
        help="Optional precomputed action metrics CSV; if empty, compute from rl_trace.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_PHASE0_DIR),
        help="Output directory for phase0_summary files.",
    )
    parser.add_argument(
        "--summary-prefix",
        type=str,
        default="phase0",
        help="Output prefix for summary files, e.g., phase0 -> phase0_summary.csv",
    )
    parser.add_argument(
        "--action-min-rows",
        type=int,
        default=20,
        help="Minimum action rows in implement scope; fallback to train+implement if below this value.",
    )
    parser.add_argument(
        "--a-low-threshold",
        type=float,
        default=0.01,
        help="Verdict threshold: A_delta_best_bacc < a_low_threshold.",
    )
    parser.add_argument(
        "--b-high-threshold",
        type=float,
        default=0.05,
        help="Verdict threshold: B_delta_best_bacc >= b_high_threshold.",
    )
    parser.add_argument(
        "--a-high-threshold",
        type=float,
        default=0.05,
        help="Verdict threshold: A_delta_best_bacc >= a_high_threshold.",
    )
    parser.add_argument(
        "--rl-stagnation-threshold",
        type=float,
        default=0.75,
        help="If avg_reward < this and A_delta_best_bacc is high, mark training/optimization bottleneck.",
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


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))


def load_meta(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "meta.json"
    if not path.exists():
        return {}
    try:
        return read_json(path)
    except Exception:
        return {}


def read_csv_robust(path: Path) -> Tuple[pd.DataFrame, int, str]:
    if not path.exists():
        raise FileNotFoundError(f"csv not found: {path}")

    errors: List[str] = []
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            with path.open("r", encoding=enc, errors="replace", newline="") as f:
                reader = csv.DictReader(f, restkey="__extra__")
                rows: List[Dict[str, Any]] = []
                extra_count = 0
                for row in reader:
                    if row is None:
                        continue
                    if row.get("__extra__"):
                        extra_count += 1
                    rows.append(row)
            if not rows:
                return pd.DataFrame(), extra_count, enc
            df = pd.DataFrame(rows)
            if "__extra__" in df.columns:
                df = df.drop(columns=["__extra__"])
            return df, extra_count, enc
        except Exception as exc:
            errors.append(f"{enc}:{type(exc).__name__}:{exc}")

    # Fallback path if DictReader fails for all encodings.
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            df = pd.read_csv(path, encoding=enc, engine="python", on_bad_lines="skip")
            return df, 0, enc
        except Exception as exc:
            errors.append(f"pandas-{enc}:{type(exc).__name__}:{exc}")
    raise RuntimeError(f"failed to read csv={path}; {' | '.join(errors)}")


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _to_lower_str(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def _select_action_rows(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    notes: List[str] = []
    stage = _to_lower_str(data["stage"]) if "stage" in data.columns else pd.Series(dtype=str)
    valid = data[data["action"].isin([0, 1])].copy()
    if valid.empty:
        return valid, valid, notes

    valid_stage = stage.loc[valid.index]

    removal = valid[valid_stage.isin(["send_action", "begin_removal"])].copy()
    if removal.empty:
        fallback = valid[valid_stage == "finish_removal"].copy()
        if not fallback.empty:
            removal = fallback
            notes.append("removal_stage_fallback=finish_removal")

    insertion = valid[valid_stage == "finish_insertion"].copy()
    if insertion.empty:
        fallback = valid[valid_stage == "begin_insertion"].copy()
        if not fallback.empty:
            insertion = fallback
            notes.append("insertion_stage_fallback=begin_insertion")

    return removal, insertion, notes


def compute_action_metrics(run_dir: Path, min_rows: int) -> Dict[str, Any]:
    trace_path = run_dir / "rl_trace.csv"
    df, skipped, _enc = read_csv_robust(trace_path)

    required = {"phase", "stage", "action"}
    missing = sorted(required - set(df.columns))
    notes: List[str] = []
    if missing:
        return {
            "run_id": run_dir.name,
            "wait_share": None,
            "action1_rate": None,
            "action_scope": "unavailable",
            "decision_count": 0,
            "removal_count": 0,
            "insertion_count": 0,
            "removal_wait_count": 0,
            "removal_reroute_count": 0,
            "insert_accept_count": 0,
            "insert_reject_count": 0,
            "trace_extra_columns_rows": skipped,
            "action_notes": f"missing_columns={','.join(missing)}",
        }

    work = df.copy()
    work["phase"] = _to_lower_str(work["phase"])
    work["stage"] = _to_lower_str(work["stage"])
    work["action"] = pd.to_numeric(work["action"], errors="coerce")
    work = work[work["phase"].isin(["train", "implement"])].copy()

    implement = work[work["phase"] == "implement"].copy()
    removal, insertion, stage_notes = _select_action_rows(implement)
    notes.extend(stage_notes)
    scope = "implement"
    decision_count = int(len(removal) + len(insertion))

    if decision_count < int(min_rows):
        combined = work[work["phase"].isin(["train", "implement"])].copy()
        removal, insertion, stage_notes = _select_action_rows(combined)
        notes.extend(stage_notes)
        scope = "train+implement"
        decision_count = int(len(removal) + len(insertion))
        notes.append(f"action_scope_fallback={scope}(min_rows={min_rows})")

    rw = int((removal["action"] == 0).sum()) if not removal.empty else 0
    rr = int((removal["action"] == 1).sum()) if not removal.empty else 0
    ia = int((insertion["action"] == 0).sum()) if not insertion.empty else 0
    ir = int((insertion["action"] == 1).sum()) if not insertion.empty else 0

    removal_total = rw + rr
    total = rw + rr + ia + ir
    wait_share = (rw / removal_total) if removal_total > 0 else None
    action1_rate = ((rr + ir) / total) if total > 0 else None

    return {
        "run_id": run_dir.name,
        "wait_share": wait_share,
        "action1_rate": action1_rate,
        "action_scope": scope,
        "decision_count": int(total),
        "removal_count": int(removal_total),
        "insertion_count": int(ia + ir),
        "removal_wait_count": rw,
        "removal_reroute_count": rr,
        "insert_accept_count": ia,
        "insert_reject_count": ir,
        "trace_extra_columns_rows": skipped,
        "action_notes": ";".join(dict.fromkeys(notes)),
    }


def load_avg_reward(run_dir: Path) -> Tuple[Optional[float], int, str]:
    summary_path = run_dir / "rl_summary.csv"
    if not summary_path.exists():
        return None, 0, "missing_rl_summary_csv"
    df, skipped, _enc = read_csv_robust(summary_path)
    if "average_reward" not in df.columns:
        return None, skipped, "missing_average_reward_column"
    series = pd.to_numeric(df["average_reward"], errors="coerce").dropna()
    if series.empty:
        return None, skipped, "average_reward_empty_after_coerce"
    return float(series.iloc[-1]), skipped, ""


def load_metrics_json(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "metrics.json"
    if not path.exists():
        return {}
    try:
        payload = read_json(path)
    except Exception:
        return {}
    return {
        "G": safe_float(payload.get("G")),
        "G_prime": safe_float(payload.get("G_prime")),
        "Adv0": safe_float(payload.get("Adv0")),
        "Adv1": safe_float(payload.get("Adv1")),
        "AdvRand": safe_float(payload.get("AdvRand")),
        "metrics_n_test": safe_float(payload.get("n_test")),
    }


def collect_base_records(run_dirs: Sequence[Path], min_action_rows: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        meta = load_meta(run_dir)
        run_id = str(meta.get("run_name") or run_dir.name)
        dist = str(meta.get("distribution") or "")
        algo = str(meta.get("algorithm") or "")
        seed = meta.get("seed")
        algo_version = str(meta.get("algo_version") or "")

        avg_reward, summary_extra_rows, summary_note = load_avg_reward(run_dir)
        action_metrics = compute_action_metrics(run_dir=run_dir, min_rows=min_action_rows)
        metrics = load_metrics_json(run_dir)

        notes = []
        if summary_note:
            notes.append(summary_note)
        if summary_extra_rows > 0:
            notes.append(f"rl_summary_rows_with_extra_columns={summary_extra_rows}")
        if int(action_metrics.get("trace_extra_columns_rows", 0)) > 0:
            notes.append(f"rl_trace_rows_with_extra_columns={action_metrics['trace_extra_columns_rows']}")
        if action_metrics.get("action_notes"):
            notes.append(str(action_metrics["action_notes"]))
        if not metrics:
            notes.append("metrics_json_missing_or_unreadable")

        row = {
            "run_dir": str(run_dir.resolve()),
            "run_id": run_id,
            "dist": dist,
            "algo": algo,
            "seed": seed,
            "algo_version": algo_version,
            "avg_reward": avg_reward,
            "notes": ";".join(dict.fromkeys(notes)),
        }
        row.update(action_metrics)
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def _mode_prefix(mode: str) -> Optional[str]:
    mode = str(mode)
    if mode == "mode_A_Xt":
        return "A"
    if mode == "mode_B_Xt":
        return "B"
    return None


def load_probe_wide(probe_csv: Path) -> pd.DataFrame:
    if not probe_csv.exists():
        raise FileNotFoundError(f"probe_csv not found: {probe_csv}")
    df, _skipped, _enc = read_csv_robust(probe_csv)
    if "run_id" not in df.columns and "run_dir" in df.columns:
        df["run_id"] = df["run_dir"].astype(str).apply(lambda x: Path(x).name)
    required = {"run_id", "mode"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"probe_csv missing required columns: {missing}")

    keep_cols = [
        "baseline_bacc",
        "logreg_bacc",
        "mlp_bacc",
        "logreg_roc_auc",
        "mlp_roc_auc",
        "delta_logreg_bacc",
        "delta_mlp_bacc",
        "n_train",
        "n_test",
    ]
    for col in keep_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    rows: List[pd.DataFrame] = []
    for mode_name in ("mode_A_Xt", "mode_B_Xt"):
        sub = df[df["mode"].astype(str) == mode_name].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(by=["run_id"], kind="stable").drop_duplicates(subset=["run_id"], keep="first")
        prefix = _mode_prefix(mode_name)
        rename_map = {}
        for col in keep_cols:
            if col in sub.columns:
                rename_map[col] = f"{prefix}_{col}"
        sub = sub[["run_id"] + [c for c in keep_cols if c in sub.columns]]
        sub = sub.rename(columns=rename_map)
        rows.append(sub)

    if not rows:
        return pd.DataFrame(columns=["run_id"])

    wide = rows[0]
    for other in rows[1:]:
        wide = wide.merge(other, on="run_id", how="outer")
    return wide


def _delta_best(row: pd.Series, prefix: str) -> Optional[float]:
    c1 = row.get(f"{prefix}_delta_logreg_bacc")
    c2 = row.get(f"{prefix}_delta_mlp_bacc")
    vals = [x for x in [c1, c2] if pd.notna(x)]
    if not vals:
        return None
    return float(max(vals))


def build_verdict(
    row: pd.Series,
    a_low_threshold: float,
    b_high_threshold: float,
    a_high_threshold: float,
    rl_stagnation_threshold: float,
) -> str:
    a = row.get("A_delta_best_bacc")
    b = row.get("B_delta_best_bacc")
    avg_reward = row.get("avg_reward")
    a_ok = pd.notna(a)
    b_ok = pd.notna(b)
    if a_ok and b_ok and float(a) < float(a_low_threshold) and float(b) >= float(b_high_threshold):
        return "信息瓶颈，历史可挖"
    if a_ok and float(a) >= float(a_high_threshold):
        if pd.isna(avg_reward) or float(avg_reward) < float(rl_stagnation_threshold):
            return "训练/优化瓶颈"
    return "待观察"


def maybe_load_action_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"action_csv not found: {path}")
    df, _skipped, _enc = read_csv_robust(path)
    if "run_id" not in df.columns and "run_dir" in df.columns:
        df["run_id"] = df["run_dir"].astype(str).apply(lambda x: Path(x).name)
    if "run_id" not in df.columns:
        raise ValueError("action_csv must contain run_id or run_dir")
    return df


def write_markdown(path: Path, df: pd.DataFrame, title: str) -> None:
    lines: List[str] = [f"# {title}", ""]
    try:
        lines.append(df.to_markdown(index=False))
    except Exception as exc:
        lines.append(f"to_markdown unavailable: {type(exc).__name__}: {exc}")
        lines.append("")
        lines.append(df.to_csv(index=False))
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()

    root = Path(args.root).resolve()
    probe_csv = Path(args.probe_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    prefix = str(args.summary_prefix or "phase0").strip() or "phase0"
    action_csv = Path(args.action_csv).resolve() if str(args.action_csv).strip() else None

    run_dirs = discover_run_dirs(root=root, recursive=bool(args.recursive))
    base_df = collect_base_records(run_dirs=run_dirs, min_action_rows=int(args.action_min_rows))
    probe_df = load_probe_wide(probe_csv=probe_csv)

    merged = base_df.merge(probe_df, on="run_id", how="left")

    if action_csv is not None:
        ext_action = maybe_load_action_csv(action_csv)
        ext_action = ext_action.drop_duplicates(subset=["run_id"], keep="first")
        merged = merged.merge(ext_action, on="run_id", how="left", suffixes=("", "_actioncsv"))

    merged["A_delta_best_bacc"] = merged.apply(lambda r: _delta_best(r, "A"), axis=1)
    merged["B_delta_best_bacc"] = merged.apply(lambda r: _delta_best(r, "B"), axis=1)
    merged["verdict"] = merged.apply(
        lambda r: build_verdict(
            r,
            a_low_threshold=float(args.a_low_threshold),
            b_high_threshold=float(args.b_high_threshold),
            a_high_threshold=float(args.a_high_threshold),
            rl_stagnation_threshold=float(args.rl_stagnation_threshold),
        ),
        axis=1,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{prefix}_summary.csv"
    md_path = out_dir / f"{prefix}_summary.md"
    html_path = out_dir / f"{prefix}_summary.html"

    merged = merged.sort_values(by=["dist", "algo", "run_id"], kind="stable")
    merged.to_csv(csv_path, index=False, encoding="utf-8-sig")

    display_cols = [
        "dist",
        "algo",
        "run_id",
        "avg_reward",
        "wait_share",
        "action1_rate",
        "A_baseline_bacc",
        "A_logreg_bacc",
        "A_mlp_bacc",
        "B_baseline_bacc",
        "B_logreg_bacc",
        "B_mlp_bacc",
        "A_delta_best_bacc",
        "B_delta_best_bacc",
        "verdict",
        "notes",
    ]
    display_cols = [c for c in display_cols if c in merged.columns]
    display_df = merged[display_cols].copy()

    write_markdown(md_path, display_df, title=f"{prefix.upper()} Summary")
    html_path.write_text(display_df.to_html(index=False), encoding="utf-8")

    print(f"[phase0] runs={len(run_dirs)}")
    print(f"[phase0] wrote: {csv_path}")
    print(f"[phase0] wrote: {md_path}")
    print(f"[phase0] wrote: {html_path}")
    with pd.option_context("display.max_rows", 200, "display.max_columns", 50, "display.width", 220):
        print(display_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

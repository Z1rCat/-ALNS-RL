from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT_DIR = Path(__file__).resolve().parents[2]
CODES_DIR = ROOT_DIR / "codes"
DEFAULT_LOGS_ROOT = CODES_DIR / "logs"


def _safe_int(v: object, default: int = 0) -> int:
    try:
        return int(str(v).strip())
    except Exception:
        return int(default)


def _safe_float(v: object, default: float = float("nan")) -> float:
    try:
        return float(str(v).strip())
    except Exception:
        return float(default)


def _is_finite(x: float) -> bool:
    return (x == x) and math.isfinite(x)


def _fmt_float(x: float) -> str:
    return "" if not _is_finite(float(x)) else f"{float(x):.10f}"


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    cleaned: List[Dict[str, str]] = []
    for row in rows:
        item: Dict[str, str] = {}
        for k, v in row.items():
            key = str(k).replace("\ufeff", "").strip()
            item[key] = "" if v is None else str(v)
        cleaned.append(item)
    return cleaned


def _write_csv_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _iter_sorted_unique(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    # Keep the last row for each iter_id.
    by_iter: Dict[int, Dict[str, str]] = {}
    for row in rows:
        iter_id = _safe_int(row.get("iter_id", ""), default=-1)
        if iter_id <= 0:
            continue
        by_iter[iter_id] = row
    return [by_iter[k] for k in sorted(by_iter.keys())]


def _mean(values: List[float]) -> float:
    clean = [float(v) for v in values if _is_finite(float(v))]
    if not clean:
        return float("nan")
    return float(sum(clean)) / float(len(clean))


def _std(values: List[float]) -> float:
    clean = [float(v) for v in values if _is_finite(float(v))]
    n = len(clean)
    if n <= 1:
        return 0.0 if n == 1 else float("nan")
    m = float(sum(clean)) / float(n)
    var = float(sum((x - m) ** 2 for x in clean)) / float(n - 1)
    return math.sqrt(max(0.0, var))


def _last_finite(values: List[float]) -> float:
    for x in reversed(values):
        if _is_finite(float(x)):
            return float(x)
    return float("nan")


def _resolve_run_roots(
    run_roots: List[str],
    logs_root: Path,
    run_glob: str,
) -> List[Path]:
    out: List[Path] = []
    for raw in run_roots:
        p = Path(str(raw)).resolve()
        out.append(p)
    if run_glob.strip():
        for p in sorted(logs_root.glob(run_glob.strip())):
            if p.is_dir():
                out.append(p.resolve())
    dedup: List[Path] = []
    seen = set()
    for p in out:
        key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(p)
    return dedup


def _load_buffer_entries(buffer_path: Path) -> List[Dict[str, object]]:
    if not buffer_path.exists():
        return []
    try:
        payload = json.loads(buffer_path.read_text(encoding="utf-8-sig"))
        entries = payload.get("entries", [])
        if isinstance(entries, list):
            return [x for x in entries if isinstance(x, dict)]
    except Exception:
        return []
    return []


def _priority_metrics(
    entries: List[Dict[str, object]],
    priority_topk: int,
) -> Tuple[float, float, int, int]:
    if not entries:
        return float("nan"), float("nan"), 0, 0
    sorted_entries = sorted(
        entries,
        key=lambda x: _safe_float(x.get("score_ema", 0.0), default=0.0),
        reverse=True,
    )
    k = max(1, min(int(priority_topk), len(sorted_entries)))
    topk = sorted_entries[:k]
    topk_cov = float(sum(1 for e in topk if _safe_int(e.get("n_sampled", 0), default=0) > 0)) / float(k)
    topk_n = float(sum(max(0, _safe_int(e.get("n_sampled", 0), default=0)) for e in topk))
    all_n = float(sum(max(0, _safe_int(e.get("n_sampled", 0), default=0)) for e in sorted_entries))
    topk_share = (topk_n / all_n) if all_n > 0 else float("nan")
    return topk_cov, topk_share, k, len(sorted_entries)


def _build_run_outputs(
    run_root: Path,
    priority_topk: int,
    top_levels: int,
) -> Optional[Dict[str, object]]:
    post_stage = run_root / "post_stage"
    plr_stats_csv = post_stage / "outer_plr_stats.csv"
    train_round_csv = post_stage / "outer_train_round.csv"
    if not plr_stats_csv.exists():
        return None

    plr_rows = _iter_sorted_unique(_read_csv_rows(plr_stats_csv))
    if not plr_rows:
        return None
    train_rows = _iter_sorted_unique(_read_csv_rows(train_round_csv))
    train_by_iter: Dict[int, Dict[str, str]] = {
        _safe_int(r.get("iter_id", ""), default=-1): r for r in train_rows
    }

    curve_rows: List[Dict[str, object]] = []
    replay_ratios: List[float] = []
    recent_replay_ratios: List[float] = []
    topk_coverages: List[float] = []
    topk_sample_shares: List[float] = []
    objective_replay: List[float] = []
    objective_new: List[float] = []
    replay_iters = 0
    new_iters = 0

    for row in plr_rows:
        iter_id = _safe_int(row.get("iter_id", ""), default=-1)
        source = str(row.get("source", "")).strip().lower()
        total_samples = max(0, _safe_int(row.get("total_samples", ""), default=0))
        replay_samples = max(0, _safe_int(row.get("replay_samples", ""), default=0))
        new_samples = max(0, _safe_int(row.get("new_samples", ""), default=0))
        replay_ratio = _safe_float(row.get("replay_ratio", ""), default=float("nan"))
        recent_rr = _safe_float(row.get("recent_replay_ratio_w20", ""), default=float("nan"))
        topk_cov = _safe_float(row.get("topk_coverage", ""), default=float("nan"))
        topk_share = _safe_float(row.get("topk_sample_share", ""), default=float("nan"))
        replay_ratios.append(replay_ratio)
        recent_replay_ratios.append(recent_rr)
        topk_coverages.append(topk_cov)
        topk_sample_shares.append(topk_share)
        if source == "replay":
            replay_iters += 1
        elif source == "new":
            new_iters += 1

        train = train_by_iter.get(iter_id, {})
        objective = _safe_float(train.get("objective_score", ""), default=float("nan"))
        if source == "replay" and _is_finite(objective):
            objective_replay.append(objective)
        elif source == "new" and _is_finite(objective):
            objective_new.append(objective)

        curve_rows.append(
            {
                "iter_id": iter_id,
                "phase": str(row.get("phase", "")),
                "source": source,
                "buffer_size": _safe_int(row.get("buffer_size", ""), default=0),
                "total_samples": total_samples,
                "new_samples": new_samples,
                "replay_samples": replay_samples,
                "new_share": _fmt_float(float(new_samples) / float(total_samples) if total_samples > 0 else float("nan")),
                "replay_share": _fmt_float(float(replay_samples) / float(total_samples) if total_samples > 0 else float("nan")),
                "replay_ratio": _fmt_float(replay_ratio),
                "recent_replay_ratio_w20": _fmt_float(recent_rr),
                "topk_coverage": _fmt_float(topk_cov),
                "topk_sample_share": _fmt_float(topk_share),
                "entry_index": row.get("entry_index", ""),
                "entry_score_ema": row.get("entry_score_ema", ""),
                "objective_score": _fmt_float(objective),
                "J": _fmt_float(_safe_float(train.get("J", ""), default=float("nan"))),
                "dJ": _fmt_float(_safe_float(train.get("dJ", ""), default=float("nan"))),
                "avg_reward": _fmt_float(_safe_float(train.get("avg_reward", ""), default=float("nan"))),
                "action_source_train_round": str(train.get("action_source", "")),
            }
        )

    analysis_dir = post_stage / "analysis" / "plr"
    curve_csv = analysis_dir / "outer_plr_curve.csv"
    _write_csv_rows(
        curve_csv,
        [
            "iter_id",
            "phase",
            "source",
            "buffer_size",
            "total_samples",
            "new_samples",
            "replay_samples",
            "new_share",
            "replay_share",
            "replay_ratio",
            "recent_replay_ratio_w20",
            "topk_coverage",
            "topk_sample_share",
            "entry_index",
            "entry_score_ema",
            "objective_score",
            "J",
            "dJ",
            "avg_reward",
            "action_source_train_round",
        ],
        curve_rows,
    )

    buffer_path = post_stage / "outer_policy_plr_buffer.json"
    entries = _load_buffer_entries(buffer_path)
    prio_cov, prio_share, k_used, n_levels = _priority_metrics(entries, priority_topk=priority_topk)

    top_levels_rows: List[Dict[str, object]] = []
    if entries:
        sorted_entries = sorted(
            entries,
            key=lambda x: _safe_float(x.get("score_ema", 0.0), default=0.0),
            reverse=True,
        )
        for idx, item in enumerate(sorted_entries[: max(1, int(top_levels))], start=1):
            action = item.get("action", {})
            top_levels_rows.append(
                {
                    "rank": idx,
                    "score_ema": _fmt_float(_safe_float(item.get("score_ema", ""), default=float("nan"))),
                    "n_sampled": _safe_int(item.get("n_sampled", ""), default=0),
                    "last_iter": _safe_int(item.get("last_iter", ""), default=-1),
                    "mu_a": _safe_int((action or {}).get("mu_a", ""), default=-1),
                    "mu_b": _safe_int((action or {}).get("mu_b", ""), default=-1),
                    "ratio_a": _fmt_float(_safe_float((action or {}).get("ratio_a", ""), default=float("nan"))),
                    "num_files": _safe_int((action or {}).get("num_files", ""), default=-1),
                    "pattern": str((action or {}).get("pattern", "")),
                }
            )
    top_levels_csv = analysis_dir / "outer_plr_top_levels.csv"
    _write_csv_rows(
        top_levels_csv,
        [
            "rank",
            "score_ema",
            "n_sampled",
            "last_iter",
            "mu_a",
            "mu_b",
            "ratio_a",
            "num_files",
            "pattern",
        ],
        top_levels_rows,
    )

    summary = {
        "run_root": str(run_root),
        "curve_csv": str(curve_csv),
        "top_levels_csv": str(top_levels_csv),
        "n_iters": len(plr_rows),
        "n_replay_iters": int(replay_iters),
        "n_new_iters": int(new_iters),
        "replay_iter_rate": _fmt_float(float(replay_iters) / float(len(plr_rows)) if plr_rows else float("nan")),
        "replay_ratio_final": _fmt_float(_last_finite(replay_ratios)),
        "replay_ratio_mean": _fmt_float(_mean(replay_ratios)),
        "replay_ratio_std": _fmt_float(_std(replay_ratios)),
        "recent_replay_ratio_w20_final": _fmt_float(_last_finite(recent_replay_ratios)),
        "topk_coverage_final": _fmt_float(_last_finite(topk_coverages)),
        "topk_coverage_mean": _fmt_float(_mean(topk_coverages)),
        "topk_sample_share_final": _fmt_float(_last_finite(topk_sample_shares)),
        "topk_sample_share_mean": _fmt_float(_mean(topk_sample_shares)),
        "objective_mean_replay_source": _fmt_float(_mean(objective_replay)),
        "objective_mean_new_source": _fmt_float(_mean(objective_new)),
        "priority_topk": int(k_used),
        "priority_n_levels": int(n_levels),
        "priority_topk_coverage_from_buffer": _fmt_float(prio_cov),
        "priority_topk_sample_share_from_buffer": _fmt_float(prio_share),
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize PLR level-replay stats into plot-ready CSVs (per-run and multi-run)."
    )
    parser.add_argument(
        "--run-root",
        action="append",
        default=[],
        help="run root path (repeatable), e.g., codes/logs/exp_xxx",
    )
    parser.add_argument(
        "--logs-root",
        type=str,
        default=str(DEFAULT_LOGS_ROOT),
        help="logs root for --run-glob discovery",
    )
    parser.add_argument(
        "--run-glob",
        type=str,
        default="",
        help="glob under --logs-root to discover run roots, e.g. 'plr_*_s42'",
    )
    parser.add_argument(
        "--priority-topk",
        type=int,
        default=10,
        help="top-k levels to evaluate high-priority coverage in replay buffer",
    )
    parser.add_argument(
        "--top-levels",
        type=int,
        default=30,
        help="how many top levels to export to outer_plr_top_levels.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="",
        help="optional output path for multi-run summary csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logs_root = Path(str(args.logs_root)).resolve()
    run_roots = _resolve_run_roots(
        run_roots=[str(x) for x in (args.run_root or [])],
        logs_root=logs_root,
        run_glob=str(args.run_glob or ""),
    )
    if not run_roots:
        raise ValueError("no run roots found; provide --run-root or --run-glob")

    summary_rows: List[Dict[str, object]] = []
    skipped: List[str] = []
    for run_root in run_roots:
        result = _build_run_outputs(
            run_root=run_root,
            priority_topk=max(1, int(args.priority_topk)),
            top_levels=max(1, int(args.top_levels)),
        )
        if result is None:
            skipped.append(str(run_root))
            continue
        summary_rows.append(result)
        print(
            f"[PLR-SUMMARY] run={run_root.name} iters={result['n_iters']} "
            f"replay_final={result['replay_ratio_final']} topk_cov={result['topk_coverage_final']}"
        )

    if not summary_rows:
        raise RuntimeError(f"no valid runs found (skipped={len(skipped)})")

    summary_csv = (
        Path(str(args.summary_csv)).resolve()
        if str(args.summary_csv or "").strip()
        else (logs_root / "plr_replay_experiment_summary.csv").resolve()
    )
    _write_csv_rows(
        summary_csv,
        [
            "run_root",
            "curve_csv",
            "top_levels_csv",
            "n_iters",
            "n_replay_iters",
            "n_new_iters",
            "replay_iter_rate",
            "replay_ratio_final",
            "replay_ratio_mean",
            "replay_ratio_std",
            "recent_replay_ratio_w20_final",
            "topk_coverage_final",
            "topk_coverage_mean",
            "topk_sample_share_final",
            "topk_sample_share_mean",
            "objective_mean_replay_source",
            "objective_mean_new_source",
            "priority_topk",
            "priority_n_levels",
            "priority_topk_coverage_from_buffer",
            "priority_topk_sample_share_from_buffer",
        ],
        summary_rows,
    )
    print(f"[PLR-SUMMARY] summary_csv={summary_csv}")
    if skipped:
        print(f"[PLR-SUMMARY] skipped={len(skipped)} runs without outer_plr_stats.csv")


if __name__ == "__main__":
    main()

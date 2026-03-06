from __future__ import annotations

import argparse
import json
import math
from itertools import product
from pathlib import Path
from typing import Iterable

import pandas as pd

from export_saber_protocol_metrics import _build_run_frame, _primary_metrics


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT_DIR / "codes" / "analysis" / "outputs" / "saber_offline_calibration"


DEFAULT_WEIGHTS = {
    "w_q": 0.28,
    "w_h": 0.18,
    "w_c": 0.14,
    "w_e": 0.15,
    "w_i": 0.05,
}


def _safe_float(value, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _spearman(a: pd.Series, b: pd.Series) -> float:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    mask = aa.notna() & bb.notna()
    if int(mask.sum()) < 2:
        return math.nan
    return float(aa[mask].rank().corr(bb[mask].rank()))


def _gaussian_gate(j_value: float, center: float, sigma: float) -> float:
    if math.isnan(j_value):
        return math.nan
    if sigma <= 0:
        return 1.0
    return float(math.exp(-((float(j_value) - float(center)) ** 2) / (2.0 * float(sigma) ** 2)))


def _load_primary_from_runs(
    run_dirs: Iterable[Path],
    hard_threshold: int,
    easy_threshold: int,
    lambda_fp: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    for run_dir in run_dirs:
        frame = _build_run_frame(run_dir=run_dir, run_label=run_dir.name)
        if frame.empty:
            rows.append(
                {
                    "run_label": run_dir.name,
                    "run_dir": str(run_dir),
                    "load_status": "empty_frame",
                }
            )
            continue
        metric_row = _primary_metrics(
            frame,
            hard_threshold=int(hard_threshold),
            easy_threshold=int(easy_threshold),
            lambda_fp=float(lambda_fp),
        )
        metric_row["load_status"] = "ok"
        rows.append(metric_row)
    return pd.DataFrame(rows)


def _score_row(row: pd.Series, weights: dict[str, float], use_gate: bool, gate_center: float, gate_sigma: float) -> float:
    def metric(name: str) -> float:
        value = _safe_float(row.get(name, math.nan), default=math.nan)
        return 0.0 if math.isnan(value) else float(value)

    base = (
        float(weights["w_q"]) * metric("Q_hard_rem")
        + float(weights["w_h"]) * metric("R_hard_rem")
        + float(weights["w_c"]) * metric("C_sel_tilde")
        + float(weights["w_e"]) * metric("P_easy_waitlike")
        + float(weights["w_i"]) * metric("M_ins")
    )
    if not use_gate:
        return float(base)
    gate = _gaussian_gate(
        j_value=_safe_float(row.get("avg_reward_implement", math.nan)),
        center=float(gate_center),
        sigma=float(gate_sigma),
    )
    if math.isnan(gate):
        return math.nan
    return float(gate * base)


def _score_frame(
    df: pd.DataFrame,
    weights: dict[str, float],
    use_gate: bool,
    gate_center: float,
    gate_sigma: float,
) -> pd.DataFrame:
    out = df.copy()
    out["score_raw"] = out.apply(
        lambda row: _score_row(row, weights=weights, use_gate=False, gate_center=gate_center, gate_sigma=gate_sigma),
        axis=1,
    )
    out["score_gated_proxy"] = out.apply(
        lambda row: _score_row(row, weights=weights, use_gate=use_gate, gate_center=gate_center, gate_sigma=gate_sigma),
        axis=1,
    )
    out["L_proxy_avg_reward"] = out["avg_reward_implement"].apply(
        lambda x: _gaussian_gate(_safe_float(x, math.nan), center=float(gate_center), sigma=float(gate_sigma))
    )
    return out


def _with_ablation_scores(df: pd.DataFrame, gate_center: float, gate_sigma: float) -> pd.DataFrame:
    presets = {
        "default": DEFAULT_WEIGHTS,
        "no_qhard": {**DEFAULT_WEIGHTS, "w_q": 0.0},
        "no_rhard": {**DEFAULT_WEIGHTS, "w_h": 0.0},
        "no_easy": {**DEFAULT_WEIGHTS, "w_e": 0.0},
        "no_ins": {**DEFAULT_WEIGHTS, "w_i": 0.0},
        "hard_only": {"w_q": 0.35, "w_h": 0.25, "w_c": 0.0, "w_e": 0.0, "w_i": 0.0},
    }
    out = df.copy()
    for name, weights in presets.items():
        out[f"score_{name}"] = out.apply(
            lambda row: _score_row(row, weights=weights, use_gate=False, gate_center=gate_center, gate_sigma=gate_sigma),
            axis=1,
        )
    return out


def _weight_search(df: pd.DataFrame, gate_center: float, gate_sigma: float) -> pd.DataFrame:
    rows: list[dict] = []
    values_q = [0.20, 0.24, 0.28, 0.32, 0.36]
    values_h = [0.10, 0.14, 0.18, 0.22]
    values_c = [0.00, 0.08, 0.14]
    values_e = [0.05, 0.10, 0.15, 0.20]
    values_i = [0.00, 0.03, 0.05, 0.08]
    for w_q, w_h, w_c, w_e, w_i in product(values_q, values_h, values_c, values_e, values_i):
        if not (w_q >= w_h >= w_i):
            continue
        if (w_q + w_h + w_c + w_e + w_i) <= 0:
            continue
        weights = {"w_q": w_q, "w_h": w_h, "w_c": w_c, "w_e": w_e, "w_i": w_i}
        scored = _score_frame(df, weights=weights, use_gate=False, gate_center=gate_center, gate_sigma=gate_sigma)
        spearman_avg = _spearman(scored["score_raw"], scored["avg_reward_implement"])
        spearman_qhard = _spearman(scored["score_raw"], scored["Q_hard_rem"])
        spearman_rhard = _spearman(scored["score_raw"], scored["R_hard_rem"])
        plr = scored[scored["run_label"].astype(str).str.contains("PLR_UED", case=False, na=False)]
        ppo = scored[
            scored["algorithm"].astype(str).str.upper().eq("PPO")
            & scored["distribution"].astype(str).eq("O_10_90")
        ]
        ppo_new = scored[
            scored["algorithm"].astype(str).str.upper().eq("PPO_NEW")
            & scored["distribution"].astype(str).eq("O_10_90")
        ]
        rows.append(
            {
                "w_q": w_q,
                "w_h": w_h,
                "w_c": w_c,
                "w_e": w_e,
                "w_i": w_i,
                "spearman_avg_reward": spearman_avg,
                "spearman_q_hard": spearman_qhard,
                "spearman_r_hard": spearman_rhard,
                "plr_score_mean": float(pd.to_numeric(plr["score_raw"], errors="coerce").mean()) if not plr.empty else math.nan,
                "ppo_score_mean": float(pd.to_numeric(ppo["score_raw"], errors="coerce").mean()) if not ppo.empty else math.nan,
                "ppo_new_score_mean": float(pd.to_numeric(ppo_new["score_raw"], errors="coerce").mean()) if not ppo_new.empty else math.nan,
                "plr_beats_ppo": int(
                    not plr.empty
                    and not ppo.empty
                    and float(pd.to_numeric(plr["score_raw"], errors="coerce").mean())
                    > float(pd.to_numeric(ppo["score_raw"], errors="coerce").mean())
                ),
                "plr_beats_ppo_new": int(
                    not plr.empty
                    and not ppo_new.empty
                    and float(pd.to_numeric(plr["score_raw"], errors="coerce").mean())
                    > float(pd.to_numeric(ppo_new["score_raw"], errors="coerce").mean())
                ),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["priority_score"] = (
        pd.to_numeric(out["plr_beats_ppo"], errors="coerce").fillna(0.0) * 2.0
        + pd.to_numeric(out["plr_beats_ppo_new"], errors="coerce").fillna(0.0) * 2.0
        + pd.to_numeric(out["spearman_q_hard"], errors="coerce").fillna(-1.0)
        + 0.5 * pd.to_numeric(out["spearman_avg_reward"], errors="coerce").fillna(-1.0)
    )
    return out.sort_values(
        ["priority_score", "spearman_q_hard", "spearman_avg_reward"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline calibration for SABER protocol scores.")
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="absolute path to run_* directory; pass multiple times",
    )
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--hard-threshold", type=int, default=5)
    parser.add_argument("--easy-threshold", type=int, default=3)
    parser.add_argument("--lambda-fp", type=float, default=0.20)
    parser.add_argument("--gate-center", type=float, default=0.55)
    parser.add_argument("--gate-sigma", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = [Path(str(p)).resolve() for p in args.run_dir]
    out_dir = Path(str(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    primary_df = _load_primary_from_runs(
        run_dirs=run_dirs,
        hard_threshold=int(args.hard_threshold),
        easy_threshold=int(args.easy_threshold),
        lambda_fp=float(args.lambda_fp),
    )
    primary_df.to_csv(out_dir / "primary_metrics.csv", index=False, encoding="utf-8-sig")

    scored_df = _score_frame(
        primary_df,
        weights=DEFAULT_WEIGHTS,
        use_gate=True,
        gate_center=float(args.gate_center),
        gate_sigma=float(args.gate_sigma),
    )
    scored_df = _with_ablation_scores(
        scored_df,
        gate_center=float(args.gate_center),
        gate_sigma=float(args.gate_sigma),
    )
    scored_df.to_csv(out_dir / "offline_scores.csv", index=False, encoding="utf-8-sig")

    weight_search_df = _weight_search(
        primary_df,
        gate_center=float(args.gate_center),
        gate_sigma=float(args.gate_sigma),
    )
    weight_search_df.to_csv(out_dir / "weight_search.csv", index=False, encoding="utf-8-sig")

    summary = {
        "status": "ok",
        "defaults": DEFAULT_WEIGHTS,
        "gate_proxy": {
            "enabled_for_score_gated_proxy": True,
            "center": float(args.gate_center),
            "sigma": float(args.gate_sigma),
            "note": "This uses avg_reward_implement as a run-level proxy only. It is not the true candidate-level learnability gate L(g).",
        },
        "top_weight_search_rows": (
            weight_search_df.head(10).to_dict(orient="records")
            if not weight_search_df.empty
            else []
        ),
    }
    (out_dir / "offline_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

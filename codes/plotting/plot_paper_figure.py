#!/usr/bin/env python3
"""
Generate publication-ready figures from an ALNS-RL run directory.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns


DEFAULT_DPI = 300
DEFAULT_WINDOW = 30
DEFAULT_RISK_QUANTILE = 0.99
CHINESE_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
]


def _has_chinese_font() -> bool:
    try:
        available = {font.name for font in font_manager.fontManager.ttflist}
    except Exception:
        return False
    return any(name in available for name in CHINESE_FONT_CANDIDATES)


USE_CHINESE = _has_chinese_font()

TEXT_CN = {
    "phase_train": "\u8bad\u7ec3",
    "phase_implement": "\u5b9e\u65bd",
    "phase_eval": "\u8bc4\u4f30",
    "phase_title": "\u9636\u6bb5",
    "phase_label_title": "\u73af\u5883\u9636\u6bb5",
    "env_title": "\u7406\u8bba\u73af\u5883\u793a\u610f",
    "env_dist_title": "\u5206\u5e03\u5bc6\u5ea6",
    "env_x": "\u5b9e\u4f8b\u5e8f\u5217",
    "env_y": "\u4e0d\u786e\u5b9a\u6027\u5f3a\u5ea6",
    "env_x_dist": "\u6570\u503c",
    "env_y_dist": "\u5bc6\u5ea6",
    "legend_phase": "\u9636\u6bb5",
    "phase_a": "\u9636\u6bb5A",
    "phase_b": "\u9636\u6bb5B",
    "phase_c": "\u9636\u6bb5C",
    "mean_label": "\u5747\u503c",
    "adapt_title": "\u9002\u5e94S\u66f2\u7ebf",
    "decision_step": "\u51b3\u7b56\u6b65",
    "smoothed_reward": "\u5e73\u6ed1\u5956\u52b1",
    "always_wait": "\u59cb\u7ec8\u7b49\u5f85",
    "always_reroute": "\u59cb\u7ec8\u6539\u9053",
    "random_policy": "\u968f\u673a\u7b56\u7565",
    "mean_fixed": "\u5747\u503c\u56fa\u5b9a",
    "heatmap_freq_title": "\u52a8\u4f5c\u9891\u7387",
    "heatmap_reward_title": "\u5e73\u5747\u5956\u52b1",
    "heatmap_y": "\u9636\u6bb5",
    "heatmap_x": "\u52a8\u4f5c\u7c7b\u578b",
    "removal_wait": "\u79fb\u9664-\u7b49\u5f85",
    "removal_reroute": "\u79fb\u9664-\u91cd\u89c4\u5212",
    "insert_accept": "\u63d2\u5165-\u63a5\u53d7",
    "insert_reject": "\u63d2\u5165-\u62d2\u7edd",
    "cum_adv_title": "\u7d2f\u8ba1\u4f18\u52bf",
    "cum_adv_x": "\u51b3\u7b56\u6b65",
    "cum_adv_y": "\u7d2f\u8ba1(Reward_RL - Reward_Baseline)",
    "missing_baseline": "\u7f3a\u5c11\u57fa\u7ebf\u6216\u5bf9\u9f50\u6570\u636e\uff0c\u65e0\u6cd5\u8ba1\u7b97\u7d2f\u8ba1\u4f18\u52bf\u3002",
    "missing_baseline_short": "\u65e0\u57fa\u7ebf\u6570\u636e\uff0c\u65e0\u6cd5\u8ba1\u7b97\u7d2f\u8ba1\u4f18\u52bf\u3002",
    "risk_quantile": "\u98ce\u9669\u5206\u4f4d\u6570",
}

TEXT_EN = {
    "phase_train": "Train",
    "phase_implement": "Implement",
    "phase_eval": "Eval",
    "phase_title": "Phase",
    "phase_label_title": "Environment Phase",
    "env_title": "Theoretical Environment Schematic",
    "env_dist_title": "Distribution Density",
    "env_x": "Instance Sequence",
    "env_y": "Uncertainty Intensity",
    "env_x_dist": "Value",
    "env_y_dist": "Density",
    "legend_phase": "Phase",
    "phase_a": "Phase A",
    "phase_b": "Phase B",
    "phase_c": "Phase C",
    "mean_label": "mean",
    "adapt_title": "Adaptation S-Curve",
    "decision_step": "Decision Step",
    "smoothed_reward": "Smoothed Reward",
    "always_wait": "Always Wait",
    "always_reroute": "Always Reroute",
    "random_policy": "Random",
    "mean_fixed": "Mean fixed",
    "heatmap_freq_title": "Action Frequency",
    "heatmap_reward_title": "Average Reward",
    "heatmap_y": "Phase",
    "heatmap_x": "Action Type",
    "removal_wait": "Removal-Wait",
    "removal_reroute": "Removal-Reroute",
    "insert_accept": "Insertion-Accept",
    "insert_reject": "Insertion-Reject",
    "cum_adv_title": "Cumulative Advantage",
    "cum_adv_x": "Decision Step",
    "cum_adv_y": "Cumulative (Reward_RL - Reward_Baseline)",
    "missing_baseline": "Missing baseline or aligned data; cannot compute cumulative advantage.",
    "missing_baseline_short": "No baseline data; cannot compute cumulative advantage.",
    "risk_quantile": "Risk Quantile",
}

TEXT = TEXT_CN if USE_CHINESE else TEXT_EN

METRIC_LABELS_CN = {
    "gt_mean": "\u73af\u5883\u5747\u503c(gt_mean)",
    "gt_std": "\u73af\u5883\u6807\u51c6\u5dee(gt_std)",
    "p_disaster": "\u707e\u5bb3\u6982\u7387(p_disaster)",
    "severity": "\u4e25\u91cd\u5ea6",
}

METRIC_LABELS_EN = {
    "gt_mean": "Environment Mean (gt_mean)",
    "gt_std": "Environment Std (gt_std)",
    "p_disaster": "Disaster Probability (p_disaster)",
    "severity": "Severity",
}

METRIC_LABELS = METRIC_LABELS_CN if USE_CHINESE else METRIC_LABELS_EN
FONT_SANS = CHINESE_FONT_CANDIDATES + ["DejaVu Sans"] if USE_CHINESE else ["DejaVu Sans", "Arial", "Liberation Sans"]
PHASE_ORDER = {"train": 0, "implement": 1, "eval": 2}
PHASE_LABELS = {
    "train": TEXT["phase_train"],
    "implement": TEXT["phase_implement"],
    "eval": TEXT["phase_eval"],
}
PHASE_LABEL_TITLES = {"phase": TEXT["phase_title"], "phase_label": TEXT["phase_label_title"]}
ROOT_DIR = Path(__file__).resolve().parent.parent
DIST_CONFIG_PATH = ROOT_DIR / "distribution_config.json"


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")


def _coerce_numeric(df: pd.DataFrame, columns) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _smooth_series(values: pd.Series, window: int) -> pd.Series:
    if values.empty:
        return values
    window = max(1, min(window, len(values)))
    if window == 1:
        return values
    min_periods = max(3, window // 3)
    return values.rolling(window=window, min_periods=min_periods).mean()


def _choose_group_key(df: pd.DataFrame) -> Optional[str]:
    if "phase_label" in df.columns:
        labels = df["phase_label"].dropna().unique()
        if len(labels) >= 2:
            return "phase_label"
    if "phase" in df.columns:
        labels = df["phase"].dropna().unique()
        if len(labels) >= 2:
            return "phase"
    return None


def _pick_env_metric(df: pd.DataFrame) -> str:
    if "gt_mean" in df.columns and df["gt_mean"].notna().any():
        return "gt_mean"
    if "severity" in df.columns and df["severity"].notna().any():
        return "severity"
    raise ValueError("No usable environment metric found (gt_mean or severity).")


def _metric_label(metric: str) -> str:
    if metric.startswith("risk_q"):
        pct = metric.replace("risk_q", "")
        try:
            pct_val = float(pct)
            return f"{TEXT['risk_quantile']} (q={pct_val:.0f}%)"
        except Exception:
            return TEXT["risk_quantile"]
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def _load_distribution_config() -> List[Dict[str, object]]:
    if not DIST_CONFIG_PATH.exists():
        return []
    try:
        data = json.loads(DIST_CONFIG_PATH.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        data = json.loads(DIST_CONFIG_PATH.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict) and isinstance(data.get("distributions"), list):
        return data["distributions"]
    return []


def _get_distribution_spec(dist_name: str) -> dict:
    for item in _load_distribution_config():
        if str(item.get("name")) == dist_name:
            return item
    return {"name": dist_name, "pattern": "ab", "means": {"A": 9, "B": 90}}


def _extract_mean_value(value: object) -> Optional[float]:
    if isinstance(value, dict):
        for key in ("mean", "mu"):
            if key in value:
                try:
                    return float(value[key])
                except Exception:
                    return None
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return None


def _is_mixture_distribution(spec: dict) -> bool:
    dist_type = str(spec.get("dist_type", "") or "").strip().lower()
    if dist_type == "mixture":
        return True

    means = spec.get("means")
    if isinstance(means, dict):
        keys = {str(k).lower() for k in means.keys()}
        if {"normal", "disaster"}.issubset(keys):
            return True

    extra = spec.get("extra")
    if isinstance(extra, dict) and "p_disaster" in extra:
        return True

    return False


def _resolve_p_disaster(extra: object, phase_label: object) -> float:
    if not isinstance(extra, dict):
        return 0.0
    spec = extra.get("p_disaster")
    if isinstance(spec, dict):
        key = str(phase_label) if phase_label is not None else ""
        value = spec.get(key)
        try:
            return float(value)
        except Exception:
            return 0.0
    try:
        return float(spec)
    except Exception:
        return 0.0


def _lognormal_cdf(x: float, mean_val: float, sigma: float = 0.5) -> float:
    if x <= 0 or mean_val <= 0:
        return 0.0
    mu = math.log(max(mean_val, 1e-6)) - 0.5 * sigma * sigma
    z = (math.log(x) - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def _mixture_risk_quantile(mean_normal: float, mean_disaster: float, p_disaster: float, q: float) -> float:
    q = min(0.9999, max(0.5, float(q)))
    p = min(1.0, max(0.0, float(p_disaster)))

    def cdf(val: float) -> float:
        return (1.0 - p) * _lognormal_cdf(val, mean_normal) + p * _lognormal_cdf(val, mean_disaster)

    lo = 1e-6
    hi = max(mean_normal, mean_disaster, 1.0) * 50.0
    for _ in range(60):
        if cdf(hi) >= q:
            break
        hi *= 2.0
        if hi > 1e7:
            break

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < q:
            lo = mid
        else:
            hi = mid
    return hi


def _build_mixture_risk_series(aligned: pd.DataFrame, spec: dict, risk_q: float) -> Tuple[pd.Series, str]:
    means = spec.get("means")
    extra = spec.get("extra")
    mean_normal = None
    mean_disaster = None
    if isinstance(means, dict):
        mean_normal = _extract_mean_value(means.get("normal", means.get("A")))
        mean_disaster = _extract_mean_value(means.get("disaster", means.get("B")))

    if mean_normal is None or mean_disaster is None:
        return pd.Series(dtype=float), ""

    note_lines = [f"normal={mean_normal:g}, disaster={mean_disaster:g}, q={risk_q:.2f}"]
    p_spec = extra.get("p_disaster") if isinstance(extra, dict) else None
    if isinstance(p_spec, dict):
        parts = []
        for k, v in p_spec.items():
            try:
                parts.append(f"{k}={float(v):.3g}")
            except Exception:
                continue
        if parts:
            note_lines.append("p_disaster: " + ", ".join(parts))
    else:
        try:
            p_val = float(p_spec) if p_spec is not None else 0.0
            note_lines.append(f"p_disaster={p_val:.3g}")
        except Exception:
            pass

    if "phase_label" in aligned.columns and aligned["phase_label"].notna().any():
        labels = aligned["phase_label"].dropna().astype(str).unique().tolist()
        risk_map: Dict[str, float] = {}
        for lbl in labels:
            p = _resolve_p_disaster(extra, lbl)
            risk_map[lbl] = _mixture_risk_quantile(mean_normal, mean_disaster, p, risk_q)
        series = aligned["phase_label"].map(lambda v: risk_map.get(str(v)) if pd.notna(v) else np.nan)
        return pd.to_numeric(series, errors="coerce"), "\n".join(note_lines)

    p = _resolve_p_disaster(extra, "A")
    const = _mixture_risk_quantile(mean_normal, mean_disaster, p, risk_q)
    return pd.Series([const] * len(aligned), dtype=float), "\n".join(note_lines)


def _build_phase_sequence(pattern: str, n_points: int) -> Tuple[List[Tuple[str, int]], List[int]]:
    if n_points <= 0:
        return [], []
    ratios = {
        "ab": [0.7, 0.3],
        "aba": [0.35, 0.35, 0.3],
        "recall": [0.85, 0.15],
        "adaptation": [0.2, 0.8],
        "abc": [0.35, 0.35, 0.3],
    }
    if pattern == "random_mix":
        return [("A", n_points)], []
    seq_ratios = ratios.get(pattern, ratios["ab"])
    labels = ["A", "B", "A"] if pattern == "aba" else ["A", "B", "C"] if pattern == "abc" else ["A", "B"]
    counts = [int(round(n_points * r)) for r in seq_ratios]
    counts[-1] = n_points - sum(counts[:-1])
    segments = list(zip(labels, counts))
    boundaries = []
    total = 0
    for _, count in segments[:-1]:
        total += count
        boundaries.append(total)
    return segments, boundaries


def _lognormal_samples(mean_val: float, std_val: Optional[float], size: int) -> np.ndarray:
    mean_val = max(1.0, float(mean_val))
    if std_val is None:
        sigma = 0.5
    else:
        std_val = max(1e-6, float(std_val))
        sigma = math.sqrt(math.log(1 + (std_val * std_val) / (mean_val * mean_val)))
    mu = math.log(mean_val) - 0.5 * sigma * sigma
    return np.random.lognormal(mean=mu, sigma=sigma, size=size)


def _gamma_samples(mean_val: float, var_val: Optional[float], size: int) -> np.ndarray:
    mean_val = max(1.0, float(mean_val))
    if var_val is None:
        shape = 2.0
        scale = mean_val / shape
    else:
        var_val = max(1e-6, float(var_val))
        shape = mean_val * mean_val / var_val
        scale = var_val / mean_val
    return np.random.gamma(shape, scale, size=size)


def _weibull_samples(mean_val: float, shape: float, size: int) -> np.ndarray:
    mean_val = max(1.0, float(mean_val))
    shape = max(0.1, float(shape))
    scale = mean_val / math.gamma(1 + 1 / shape)
    return np.random.weibull(shape, size=size) * scale


def _generate_mock_data(
    dist_name: str, n_points: int = 600, seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, float], List[int]]:
    np.random.seed(seed)
    spec = _get_distribution_spec(dist_name)
    pattern = str(spec.get("pattern", "ab"))
    means = spec.get("means", {})
    dist_type = spec.get("dist_type")
    variance = spec.get("variance")
    extra = spec.get("extra", {})

    segments, boundaries = _build_phase_sequence(pattern, n_points)
    if pattern == "random_mix":
        labels = np.random.choice(["A", "B"], size=n_points, p=[0.5, 0.5])
        segments = [("A", (labels == "A").sum()), ("B", (labels == "B").sum())]
    values = []
    phase_labels = []

    phase_means = {}
    for phase, _ in segments:
        if phase in means:
            phase_spec = means[phase]
            if isinstance(phase_spec, dict):
                phase_means[phase] = float(phase_spec.get("mean", phase_spec.get("mu", 1)))
            else:
                phase_means[phase] = float(phase_spec)

    if dist_type == "mixture":
        normal_mean = means.get("normal", 9)
        disaster_mean = means.get("disaster", 90)
        for phase, count in segments:
            if pattern == "random_mix":
                count = int((labels == phase).sum())
            p_disaster = extra.get("p_disaster", 0.05)
            if isinstance(p_disaster, dict):
                p_disaster = p_disaster.get(phase, 0.05)
            draw = np.random.rand(count) < float(p_disaster)
            vals = np.empty(count)
            n_disaster = int(draw.sum())
            vals[draw] = _lognormal_samples(disaster_mean, None, n_disaster)
            vals[~draw] = _lognormal_samples(normal_mean, None, count - n_disaster)
            values.append(vals)
            phase_labels.extend([phase] * count)
    else:
        for phase, count in segments:
            if pattern == "random_mix":
                count = int((labels == phase).sum())
            phase_spec = means.get(phase, list(means.values())[0] if means else 9)
            if isinstance(phase_spec, dict):
                mean_val = phase_spec.get("mean", phase_spec.get("mu", 9))
                std_val = phase_spec.get("std")
                dist = phase_spec.get("dist", "lognormal")
            else:
                mean_val = phase_spec
                std_val = None
                dist = "lognormal"
            if std_val is None and variance is not None:
                if isinstance(variance, dict):
                    var_val = variance.get(phase)
                else:
                    var_val = variance
                if var_val is not None:
                    std_val = math.sqrt(float(var_val))
            if dist_type == "gamma":
                vals = _gamma_samples(mean_val, variance.get(phase) if isinstance(variance, dict) else variance, count)
            elif dist_type == "weibull":
                shape = extra.get("shape", 1.5)
                vals = _weibull_samples(mean_val, shape, count)
            elif dist == "normal":
                sigma = std_val if std_val is not None else max(1.0, float(mean_val) * 0.2)
                vals = np.random.normal(mean_val, sigma, size=count)
            else:
                vals = _lognormal_samples(mean_val, std_val, count)
            vals = np.maximum(vals, 1)
            values.append(vals)
            phase_labels.extend([phase] * count)

    df = pd.DataFrame(
        {
            "instance_id": np.arange(len(phase_labels)),
            "value": np.concatenate(values) if values else np.array([]),
            "phase_label": phase_labels,
        }
    )
    return df, phase_means, boundaries


def _format_phase_label(value: object) -> str:
    mapping = {"A": TEXT["phase_a"], "B": TEXT["phase_b"], "C": TEXT["phase_c"]}
    return mapping.get(str(value), str(value))


def _group_display(env: pd.DataFrame, group_key: str) -> tuple[str, list, str]:
    display_col = "_group_display"
    if group_key == "phase":
        env[display_col] = env["phase"].astype(str).str.lower().map(PHASE_LABELS).fillna(env["phase"])
        order = [
            PHASE_LABELS.get(v, str(v))
            for v in sorted(env["phase"].dropna().unique(), key=lambda v: PHASE_ORDER.get(str(v).lower(), 99))
        ]
        title = PHASE_LABEL_TITLES.get(group_key, group_key)
        return display_col, order, title
    if group_key == "phase_label":
        env[display_col] = env["phase_label"].map(_format_phase_label)
        order = [_format_phase_label(v) for v in sorted(env["phase_label"].dropna().unique())]
        title = PHASE_LABEL_TITLES.get(group_key, group_key)
        return display_col, order, title
    env[display_col] = env[group_key].astype(str)
    order = sorted(env[group_key].dropna().unique())
    return display_col, order, group_key


def _load_rl_training(run_dir: Path) -> pd.DataFrame:
    training_path = run_dir / "rl_training.csv"
    if not training_path.exists():
        return pd.DataFrame()
    df = _read_csv(training_path)
    _coerce_numeric(df, ["reward", "step_idx"])
    df = df[df["reward"].notna()]
    df = df[df["reward"] != -10000000]
    if "phase" not in df.columns:
        df["phase"] = "train"
    return df


def _load_baseline_events(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        print(f"Warning: baseline file not found for {label}: {path}", file=sys.stderr)
        return pd.DataFrame()
    df = _read_csv(path)
    if "reward" not in df.columns or "phase" not in df.columns or "table_number" not in df.columns:
        print(f"Warning: baseline file missing required columns for {label}: {path}", file=sys.stderr)
        return pd.DataFrame()
    _coerce_numeric(df, ["reward", "table_number", "ts"])
    df = df[df["reward"].notna()]
    df = df[df["reward"] != -10000000]
    if df.empty:
        print(f"Warning: baseline file has no valid rewards for {label}: {path}", file=sys.stderr)
        return pd.DataFrame()
    if "source" in df.columns and (df["source"] == "BASELINE").any():
        df = df[df["source"] == "BASELINE"]
    elif "stage" in df.columns:
        df = df[df["stage"] == "finish_removal"]
    df["phase"] = df["phase"].astype(str).str.lower()
    return df.reset_index(drop=True)


def _sort_by_phase_table(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if "phase" in data.columns:
        data["phase"] = data["phase"].astype(str).str.lower()
    _coerce_numeric(data, ["table_number", "ts"])
    data = data[data["table_number"].notna()]
    data["phase_order"] = data["phase"].map(PHASE_ORDER).fillna(99)
    data = data.sort_values(["phase_order", "table_number", "ts"], na_position="last")
    return data


def _align_rewards_to_decisions(decisions: pd.DataFrame, training: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty or training.empty:
        return pd.DataFrame()
    decisions = decisions.copy()
    decisions["phase"] = decisions["phase"].astype(str).str.lower()

    training = training.copy()
    training["phase"] = training["phase"].astype(str).str.lower()
    sort_cols = []
    if "ts" in training.columns:
        sort_cols.append("ts")
    if "step_idx" in training.columns:
        sort_cols.append("step_idx")
    if sort_cols:
        training = training.sort_values(sort_cols)

    aligned_rows = []
    for phase in ["train", "implement", "eval"]:
        phase_decisions = decisions[decisions["phase"] == phase]
        if phase_decisions.empty:
            continue
        phase_rewards = training[training["phase"] == phase]["reward"].tolist()
        if not phase_rewards:
            continue
        n = min(len(phase_decisions), len(phase_rewards))
        if len(phase_decisions) != len(phase_rewards):
            print(
                f"Warning: reward count mismatch in {phase} "
                f"(decisions={len(phase_decisions)}, rewards={len(phase_rewards)}).",
                file=sys.stderr,
            )
        phase_decisions = phase_decisions.iloc[:n].copy()
        phase_decisions["reward"] = phase_rewards[:n]
        aligned_rows.append(phase_decisions)

    if not aligned_rows:
        return pd.DataFrame()
    return pd.concat(aligned_rows, ignore_index=True)


def _prepare_aligned_decisions(trace: pd.DataFrame, training: pd.DataFrame) -> pd.DataFrame:
    decisions = trace.copy()
    if "stage" in decisions.columns:
        decisions = decisions[decisions["stage"] == "send_action"]
    if decisions.empty or training.empty:
        return pd.DataFrame()
    _coerce_numeric(decisions, ["ts"])
    if "ts" in decisions.columns:
        decisions = decisions.sort_values("ts")
    aligned = _align_rewards_to_decisions(decisions, training)
    if aligned.empty:
        return pd.DataFrame()
    return _sort_by_phase_table(aligned)


def _plot_environment_shift(run_dir: Path, out_path: Path) -> None:
    meta_path = run_dir / "meta.json"
    dist_name = "S3_1"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        dist_name = meta.get("distribution", dist_name)

    env, phase_means, boundaries = _generate_mock_data(dist_name, n_points=600)
    if env.empty:
        raise ValueError("Mock data generation failed.")
    metric_label = TEXT["env_y"]

    group_key = "phase_label"
    group_values = sorted(env["phase_label"].dropna().unique())
    legend_title = TEXT["legend_phase"]
    hue_col = "phase_label"
    if len(group_values) == 2:
        palette = ["#1F4E79", "#C94F4F"]
    else:
        palette = sns.color_palette("deep", n_colors=max(1, len(group_values)))
    segment_col = hue_col
    segment_order = group_values

    def _mean_label(name: object, mean_val: float) -> str:
        if pd.isna(mean_val):
            return str(name)
        return f"{TEXT['phase_title']} {name} ({TEXT['mean_label']}={mean_val:.1f})"

    segment_means = {}
    segment_means.update({k: phase_means.get(k, np.nan) for k in group_values})
    env["_segment_label"] = env[hue_col].map(lambda v: _mean_label(v, segment_means.get(v, np.nan)))
    segment_order = [_mean_label(val, segment_means.get(val, np.nan)) for val in group_values]
    segment_col = "_segment_label"

    color_map = {label: palette[idx % len(palette)] for idx, label in enumerate(segment_order)}

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 4.2),
        dpi=DEFAULT_DPI,
        gridspec_kw={"width_ratios": [2.6, 1]},
    )
    ax_left, ax_right = axes

    if group_key:
        sns.scatterplot(
            data=env,
            x="instance_id",
            y="value",
            hue=segment_col,
            palette=color_map,
            hue_order=segment_order,
            s=45,
            linewidth=0,
            alpha=0.8,
            ax=ax_left,
        )
        ax_left.legend(title=legend_title, loc="upper right", frameon=False)
    else:
        ax_left.scatter(
            env["instance_id"],
            env["value"],
            s=45,
            alpha=0.8,
            color=palette[0],
        )
    ax_left.set_title(TEXT["env_title"])
    ax_left.set_xlabel(TEXT["env_x"])
    ax_left.set_ylabel(metric_label)
    ax_left.grid(True, linestyle="--", alpha=0.4)

    for x in boundaries:
        ax_left.axvline(x, color="gray", linestyle="--", linewidth=1, alpha=0.6)

    values = env["value"].dropna()
    if values.shape[0] >= 3:
        sns.kdeplot(
            values,
            ax=ax_right,
            fill=True,
            color="#7f7f7f",
            alpha=0.4,
            linewidth=1.5,
            warn_singular=False,
        )
    else:
        ax_right.hist(values, bins=10, color="#7f7f7f", alpha=0.4)

    for label, mean_val in segment_means.items():
        if pd.isna(mean_val):
            continue
        display_label = _mean_label(label, mean_val)
        color = color_map.get(display_label, "#1F4E79")
        ax_right.axvline(mean_val, color=color, linestyle="--", linewidth=1.2, alpha=0.9)
    ax_right.set_title(TEXT["env_dist_title"])
    ax_right.set_xlabel(TEXT["env_x_dist"])
    ax_right.set_ylabel(TEXT["env_y_dist"])
    ax_right.grid(True, linestyle="--", alpha=0.4)

    if values.shape[0]:
        max_val = values.max() * 1.15
        if max_val > 0:
            ax_left.set_ylim(0, max_val)
            ax_right.set_xlim(0, max_val)

    sns.despine(fig=fig)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)


def _plot_adaptation_curve(
    aligned: pd.DataFrame,
    baseline_wait: pd.DataFrame,
    baseline_reroute: pd.DataFrame,
    baseline_random: pd.DataFrame,
    out_path: Path,
    window: int,
    dist_name: Optional[str] = None,
    risk_q: float = DEFAULT_RISK_QUANTILE,
) -> None:
    if aligned.empty:
        raise ValueError("Unable to align rewards to decision trace.")

    x_train = np.arange(len(aligned))
    rl_smooth = _smooth_series(aligned["reward"], window)

    fig, ax_left = plt.subplots(1, 1, figsize=(13, 5), dpi=DEFAULT_DPI)
    palette = sns.color_palette("deep")

    phase_colors = {
        "train": "#E6F4EA",
        "implement": "#FDECEC",
        "eval": "#F0F0F0",
    }
    phases = aligned.get("phase", pd.Series(["train"] * len(aligned))).fillna("train")
    if not phases.empty:
        start = 0
        current = phases.iloc[0]
        for idx, phase in enumerate(phases):
            if phase != current:
                ax_left.axvspan(
                    start,
                    idx,
                    color=phase_colors.get(current, "#F0F0F0"),
                    alpha=0.3,
                    zorder=0,
                )
                start = idx
                current = phase
        ax_left.axvspan(
            start,
            len(phases),
            color=phase_colors.get(current, "#F0F0F0"),
            alpha=0.3,
            zorder=0,
        )

    ax_left.plot(x_train, rl_smooth, color=palette[0], linewidth=2.2, label="RL")

    if not baseline_wait.empty:
        baseline_wait = _sort_by_phase_table(baseline_wait)
        x_wait = np.linspace(0, len(aligned) - 1, len(baseline_wait))
        ax_left.plot(
            x_wait,
            _smooth_series(baseline_wait["reward"], max(3, window // 2)),
            color="gray",
            linestyle="--",
            linewidth=1.8,
            label=TEXT["always_wait"],
        )
    if not baseline_reroute.empty:
        baseline_reroute = _sort_by_phase_table(baseline_reroute)
        x_reroute = np.linspace(0, len(aligned) - 1, len(baseline_reroute))
        ax_left.plot(
            x_reroute,
            _smooth_series(baseline_reroute["reward"], max(3, window // 2)),
            color=palette[2],
            linestyle="--",
            linewidth=1.8,
            label=TEXT["always_reroute"],
        )
    if not baseline_random.empty:
        baseline_random = _sort_by_phase_table(baseline_random)
        x_random = np.linspace(0, len(aligned) - 1, len(baseline_random))
        ax_left.plot(
            x_random,
            _smooth_series(baseline_random["reward"], max(3, window // 2)),
            color=palette[3],
            linestyle="--",
            linewidth=1.8,
            label=TEXT["random_policy"],
        )


    ax_left.set_xlabel(TEXT["decision_step"])
    ax_left.set_ylabel(TEXT["smoothed_reward"])
    ax_left.set_title(TEXT["adapt_title"], pad=28)
    ax_left.set_ylim(-0.05, 1.05)

    ax_right = ax_left.twinx()
    gt_mean = pd.Series(dtype=float)
    if "gt_mean" in aligned.columns:
        _coerce_numeric(aligned, ["gt_mean"])
        gt_mean = aligned["gt_mean"]
    mean_changes = bool(gt_mean.notna().any() and gt_mean.nunique(dropna=True) > 1)

    variance_changes = False
    gt_std = pd.Series(dtype=float)
    if "gt_std" in aligned.columns:
        _coerce_numeric(aligned, ["gt_std"])
        gt_std = aligned["gt_std"]
        variance_changes = bool(gt_std.notna().any() and gt_std.nunique(dropna=True) > 1)
    elif dist_name:
        spec = _get_distribution_spec(dist_name)
        variance_spec = spec.get("variance")
        std_map: Dict[str, float] = {}
        if isinstance(variance_spec, dict):
            for k, v in variance_spec.items():
                try:
                    std_map[str(k)] = math.sqrt(max(0.0, float(v)))
                except Exception:
                    continue
        else:
            means_spec = spec.get("means")
            if isinstance(means_spec, dict):
                for k, v in means_spec.items():
                    if isinstance(v, dict) and "std" in v:
                        try:
                            std_map[str(k)] = float(v["std"])
                        except Exception:
                            continue
        if std_map and "phase_label" in aligned.columns:
            phase_vals = aligned["phase_label"].dropna().astype(str).unique().tolist()
            present = [std_map.get(v) for v in phase_vals if v in std_map]
            present = [v for v in present if v is not None and not math.isnan(v)]
            variance_changes = len(set(present)) > 1 if present else False
            gt_std = aligned["phase_label"].map(lambda v: std_map.get(str(v)) if pd.notna(v) else np.nan)
            gt_std = pd.to_numeric(gt_std, errors="coerce")

    use_gt_std = bool(variance_changes and not mean_changes and gt_std.notna().any())
    env_series = gt_std if use_gt_std else gt_mean
    env_label = "gt_std" if use_gt_std else "gt_mean"

    env_note = ""
    if dist_name:
        spec = _get_distribution_spec(dist_name)
        if _is_mixture_distribution(spec):
            env_series, env_note = _build_mixture_risk_series(aligned, spec, risk_q)
            env_label = f"risk_q{int(round(risk_q * 100))}"
            use_gt_std = False
            if env_series.empty or not env_series.notna().any():
                extra = spec.get("extra")
                if "phase_label" in aligned.columns:
                    series = aligned["phase_label"].map(
                        lambda v: _resolve_p_disaster(extra, v) if pd.notna(v) else np.nan
                    )
                    env_series = pd.to_numeric(series, errors="coerce")
                else:
                    p_const = _resolve_p_disaster(extra, "A")
                    env_series = pd.Series([p_const] * len(aligned), dtype=float)
                env_label = "p_disaster"
                env_note = ""

    if env_series.notna().any():
        if env_label == "gt_mean" and env_series.nunique(dropna=True) <= 1:
            print("Warning: gt_mean has no variance in this run.", file=sys.stderr)
        x_env = np.arange(len(env_series))
        mask = env_series.notna()
        ax_right.step(
            x_env[mask],
            env_series[mask],
            where="post",
            color="black",
            linewidth=1.4,
            linestyle=":",
            label=env_label,
        )
        ax_right.set_ylabel(_metric_label(env_label))
        if env_note:
            ax_right.text(
                0.02,
                0.95,
                env_note,
                transform=ax_right.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#999999"},
            )
        if use_gt_std and gt_mean.notna().any():
            const_mean = float(gt_mean.dropna().iloc[0])
            ax_right.text(
                0.02,
                0.95,
                f"均值恒定: {const_mean:.1f}",
                transform=ax_right.transAxes,
                ha="left",
                va="top",
                fontsize=11,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#999999"},
            )

    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    handles = handles_left + handles_right
    labels = labels_left + labels_right
    ncol = max(3, len(labels))
    ax_left.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=ncol,
        frameon=False,
        borderaxespad=0,
    )

    sns.despine(fig=fig)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)


def _plot_policy_heatmap(trace: pd.DataFrame, aligned: pd.DataFrame, out_path: Path) -> None:
    data = trace.copy()
    if "phase" not in data.columns:
        raise ValueError("rl_trace.csv missing phase column for heatmap.")
    data["phase"] = data["phase"].astype(str).str.lower()
    data = data[data["phase"].isin(["train", "implement"])]
    _coerce_numeric(data, ["action"])

    removal = data[data["stage"].isin(["send_action", "begin_removal"])]
    removal = removal[removal["action"].isin([0, 1])]
    insertion_finish = data[data["stage"] == "finish_insertion"]
    insertion_begin = data[data["stage"] == "begin_insertion"]
    insertion_finish = insertion_finish[insertion_finish["action"].isin([0, 1])]
    insertion_begin = insertion_begin[insertion_begin["action"].isin([0, 1])]

    phases = [p for p in ["train", "implement", "eval"] if p in data["phase"].dropna().unique()]
    if not phases:
        phases = sorted(data["phase"].dropna().unique())
    columns = [
        TEXT["removal_wait"],
        TEXT["removal_reroute"],
        TEXT["insert_accept"],
        TEXT["insert_reject"],
    ]
    counts = pd.DataFrame(0, index=phases, columns=columns, dtype=float)

    for phase in phases:
        rem_phase = removal[removal["phase"] == phase]
        ins_phase = insertion_finish[insertion_finish["phase"] == phase]
        if ins_phase.empty:
            ins_phase = insertion_begin[insertion_begin["phase"] == phase]
        counts.loc[phase, TEXT["removal_wait"]] = (rem_phase["action"] == 0).sum()
        counts.loc[phase, TEXT["removal_reroute"]] = (rem_phase["action"] == 1).sum()
        counts.loc[phase, TEXT["insert_accept"]] = (ins_phase["action"] == 0).sum()
        counts.loc[phase, TEXT["insert_reject"]] = (ins_phase["action"] == 1).sum()

    totals = counts.sum(axis=1).replace(0, np.nan)
    freq = counts.div(totals, axis=0) * 100
    freq.index = [PHASE_LABELS.get(p, p.title()) for p in freq.index]

    reward_columns = columns
    reward_means = pd.DataFrame(np.nan, index=phases, columns=reward_columns, dtype=float)
    if not aligned.empty and "reward" in aligned.columns:
        removal_rewards = aligned[aligned["action"].isin([0, 1])]
        for phase in phases:
            sub = removal_rewards[removal_rewards["phase"] == phase]
            if sub.empty:
                continue
            reward_means.loc[phase, TEXT["removal_wait"]] = sub.loc[sub["action"] == 0, "reward"].mean()
            reward_means.loc[phase, TEXT["removal_reroute"]] = sub.loc[sub["action"] == 1, "reward"].mean()

    insertion_reward = data[data["stage"].isin(["finish_insertion", "begin_insertion"])]
    insertion_reward = insertion_reward[insertion_reward["action"].isin([0, 1])]
    insertion_reward = insertion_reward[insertion_reward["reward"].notna()]
    insertion_reward = insertion_reward[insertion_reward["reward"] != -10000000]
    if not insertion_reward.empty:
        for phase in phases:
            sub = insertion_reward[insertion_reward["phase"] == phase]
            if sub.empty:
                continue
            reward_means.loc[phase, TEXT["insert_accept"]] = sub.loc[sub["action"] == 0, "reward"].mean()
            reward_means.loc[phase, TEXT["insert_reject"]] = sub.loc[sub["action"] == 1, "reward"].mean()

    reward_means.index = [PHASE_LABELS.get(p, p.title()) for p in reward_means.index]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), dpi=DEFAULT_DPI)
    ax_left, ax_right = axes

    sns.heatmap(
        freq,
        ax=ax_left,
        cmap="RdBu_r",
        vmin=0,
        vmax=100,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": f"{TEXT['heatmap_freq_title']}(%)"},
    )
    ax_left.set_title(TEXT["heatmap_freq_title"])
    ax_left.set_xlabel(TEXT["heatmap_x"])
    ax_left.set_ylabel(TEXT["heatmap_y"])

    mask = reward_means.isna()
    cmap = sns.color_palette("RdBu_r", as_cmap=True).copy()
    cmap.set_bad(color="#D9D9D9")
    sns.heatmap(
        reward_means,
        ax=ax_right,
        cmap=cmap,
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        mask=mask,
        cbar_kws={"label": TEXT["heatmap_reward_title"]},
    )
    ax_right.set_title(TEXT["heatmap_reward_title"])
    ax_right.set_xlabel(TEXT["heatmap_x"])
    ax_right.set_ylabel("")

    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)


def _plot_cumulative_advantage(
    aligned: pd.DataFrame,
    baseline_wait: pd.DataFrame,
    baseline_reroute: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.2), dpi=DEFAULT_DPI)
    if aligned.empty or (baseline_wait.empty and baseline_reroute.empty):
        ax.text(
            0.5,
            0.5,
            TEXT["missing_baseline"],
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=DEFAULT_DPI)
        plt.close(fig)
        return

    rl_rewards = aligned["reward"].to_numpy()
    x = np.arange(len(rl_rewards))

    def _cum_adv(rl: np.ndarray, baseline: np.ndarray) -> np.ndarray:
        rl_series = pd.Series(rl, dtype=float)
        baseline_series = pd.Series(baseline, dtype=float).reindex(rl_series.index)
        diff = (rl_series - baseline_series).fillna(0.0)
        return diff.cumsum().to_numpy()

    any_line = False
    if not baseline_wait.empty:
        baseline_wait = _sort_by_phase_table(baseline_wait)
        wait_rewards = baseline_wait["reward"].to_numpy()
        if len(rl_rewards) != len(wait_rewards):
            print(
                f"Warning: cumulative advantage length mismatch (RL={len(rl_rewards)}, wait={len(wait_rewards)}).",
                file=sys.stderr,
            )
        ax.plot(
            x,
            _cum_adv(rl_rewards, wait_rewards),
            color="#1F4E79",
            linewidth=2.2,
            label=TEXT["always_wait"],
        )
        any_line = True

    if not baseline_reroute.empty:
        baseline_reroute = _sort_by_phase_table(baseline_reroute)
        reroute_rewards = baseline_reroute["reward"].to_numpy()
        if len(rl_rewards) != len(reroute_rewards):
            print(
                f"Warning: cumulative advantage length mismatch (RL={len(rl_rewards)}, reroute={len(reroute_rewards)}).",
                file=sys.stderr,
            )
        ax.plot(
            x,
            _cum_adv(rl_rewards, reroute_rewards),
            color="#55A868",
            linewidth=2.2,
            label=TEXT["always_reroute"],
        )
        any_line = True

    if not any_line:
        ax.text(
            0.5,
            0.5,
            TEXT["missing_baseline_short"],
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=DEFAULT_DPI)
        plt.close(fig)
        return

    ax.axhline(0, color="#999999", linewidth=1, linestyle="--")
    ax.set_title(TEXT["cum_adv_title"])
    ax.set_xlabel(TEXT["cum_adv_x"])
    ax.set_ylabel(TEXT["cum_adv_y"])
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ALNS-RL paper figures.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing logs.")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="Smoothing window size.")
    parser.add_argument(
        "--risk-q",
        type=float,
        default=DEFAULT_RISK_QUANTILE,
        help="Risk quantile (0-1) used for mixture distributions on Fig2 right axis.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    trace_path = run_dir / "rl_trace.csv"
    if not trace_path.exists():
        raise FileNotFoundError(f"rl_trace.csv not found: {trace_path}")

    trace = _read_csv(trace_path)
    _coerce_numeric(trace, ["table_number", "gt_mean", "severity", "action", "reward", "ts"])

    dist_name: Optional[str] = None
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        value = meta.get("distribution")
        if value:
            dist_name = str(value)

    training = _load_rl_training(run_dir)
    baseline_wait = _load_baseline_events(run_dir / "baseline_wait.csv", "wait")
    baseline_reroute = _load_baseline_events(run_dir / "baseline_reroute.csv", "reroute")
    baseline_random = _load_baseline_events(run_dir / "baseline_random.csv", "random")
    aligned = _prepare_aligned_decisions(trace, training)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    plt.rcParams["figure.dpi"] = DEFAULT_DPI
    plt.rcParams["savefig.dpi"] = DEFAULT_DPI
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = FONT_SANS
    plt.rcParams["axes.unicode_minus"] = False

    out_dir = run_dir / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_environment_shift(run_dir, out_dir / "fig1_environment.pdf")
    _plot_adaptation_curve(
        aligned,
        baseline_wait,
        baseline_reroute,
        baseline_random,
        out_dir / "fig2_adaptation.pdf",
        args.window,
        dist_name,
        risk_q=args.risk_q,
    )
    _plot_policy_heatmap(trace, aligned, out_dir / "fig3_policy_heatmap.pdf")
    _plot_cumulative_advantage(aligned, baseline_wait, baseline_reroute, out_dir / "fig4_cumulative_advantage.pdf")

    print("Saved figures to:", out_dir)
    print("  fig1_environment.pdf")
    print("  fig2_adaptation.pdf")
    print("  fig3_policy_heatmap.pdf")
    print("  fig4_cumulative_advantage.pdf")


if __name__ == "__main__":
    main()

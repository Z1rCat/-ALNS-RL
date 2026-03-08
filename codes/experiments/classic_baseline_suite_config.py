from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List


ROOT_DIR = Path(__file__).resolve().parents[2]
DISTRIBUTION_CONFIG_PATH = ROOT_DIR / "distribution_config.json"

DEFAULT_SEEDS: List[int] = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
DEFAULT_REQUEST_NUMBERS: List[int] = [30]

TRAINABLE_VARIANTS: List[str] = [
    "A2C",
    "PPO",
    "PPO_LSTM",
    "RARL",
    "PLR_UED",
]

BASELINE_POLICY_LABELS: List[str] = [
    "random",
    "always1",
    "always0",
]

REPORT_ALGORITHM_ORDER: List[str] = TRAINABLE_VARIANTS + BASELINE_POLICY_LABELS
REPORT_FAMILY_ORDER: List[str] = ["M", "R", "O", "F1", "F2", "G"]

MAIN_TABLE_DISTS: List[str] = [
    "M_10",
    "M_60",
    "M_120",
    "R_10_90",
    "R_30_80",
    "O_10_90",
    "O_90_10",
    "O_30_80",
    "O_60_20",
    "O_10_120",
    "O_120_10",
    "F1_10_90",
    "F1_90_10",
    "F2_10_90",
    "F2_30_80",
    "G_10_90_50",
    "G_10_40_90",
    "G_40_80_10",
    "G_30_60_90",
]


def load_distribution_names(config_path: Path = DISTRIBUTION_CONFIG_PATH) -> List[str]:
    raw = json.loads(config_path.read_text(encoding="utf-8-sig"))
    items = raw.get("distributions", [])
    names: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if name:
            names.append(name)
    return names


def validate_distribution_subset(
    dist_names: Iterable[str],
    *,
    config_path: Path = DISTRIBUTION_CONFIG_PATH,
) -> None:
    available = set(load_distribution_names(config_path=config_path))
    requested = [str(name).strip() for name in dist_names if str(name).strip()]
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(
            f"distribution_config.json missing requested distributions: {missing}"
        )


def family_of_dist(dist_name: str) -> str:
    raw = str(dist_name or "").strip()
    if not raw:
        return ""
    if raw.startswith("F1_"):
        return "F1"
    if raw.startswith("F2_"):
        return "F2"
    return raw.split("_", 1)[0]


def canonical_baseline_label(raw: str) -> str:
    value = str(raw or "").strip().lower()
    aliases = {
        "random": "random",
        "rand": "random",
        "always0": "always0",
        "always_0": "always0",
        "alwayswait": "always0",
        "always_wait": "always0",
        "wait": "always0",
        "always1": "always1",
        "always_1": "always1",
        "alwaysreroute": "always1",
        "always_reroute": "always1",
        "reroute": "always1",
    }
    return aliases.get(value, value)


def algorithm_order_key(label: str) -> tuple[int, str]:
    raw = str(label or "").strip()
    try:
        return REPORT_ALGORITHM_ORDER.index(raw), raw
    except ValueError:
        return len(REPORT_ALGORITHM_ORDER), raw


def family_order_key(label: str) -> tuple[int, str]:
    raw = str(label or "").strip()
    try:
        return REPORT_FAMILY_ORDER.index(raw), raw
    except ValueError:
        return len(REPORT_FAMILY_ORDER), raw

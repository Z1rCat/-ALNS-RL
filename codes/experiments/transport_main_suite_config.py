from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT_DIR = Path(__file__).resolve().parents[2]
DISTRIBUTION_CONFIG_PATH = ROOT_DIR / "distribution_config.json"

DEFAULT_SEEDS: List[int] = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
DEFAULT_REQUEST_NUMBERS: List[int] = [30]

DEFAULT_BASELINE_VARIANTS: List[str] = [
    "A2C",
    "PPO",
    "PPO_LSTM",
    "RARL",
    "PLR_UED",
]

MEAN_GRID: List[int] = [10, 30, 60, 90, 120]


def _ordered_pair_names(prefix: str, pairs: Sequence[Tuple[int, int]]) -> List[str]:
    return [f"{prefix}_{int(a)}_{int(b)}" for a, b in pairs]


def _upper_triangle_pair_names(prefix: str, means: Sequence[int]) -> List[str]:
    out: List[str] = []
    vals = [int(x) for x in means]
    for idx, a in enumerate(vals):
        for b in vals[idx + 1 :]:
            out.append(f"{prefix}_{a}_{b}")
    return out


def _directed_pair_names(prefix: str, means: Sequence[int]) -> List[str]:
    out: List[str] = []
    vals = [int(x) for x in means]
    for a in vals:
        for b in vals:
            if a == b:
                continue
            out.append(f"{prefix}_{a}_{b}")
    return out


def _dedupe_keep_order(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in values:
        item = str(raw or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


FAMILY_REGISTRY: Dict[str, List[str]] = {
    "M_core": [f"M_{m}" for m in MEAN_GRID],
    "M_grid": [f"M_{m}" for m in MEAN_GRID],
    "R_core": _ordered_pair_names(
        "R",
        [
            (10, 30),
            (30, 60),
            (60, 90),
            (90, 120),
            (10, 90),
            (30, 90),
            (10, 120),
        ],
    ),
    "R_grid": _upper_triangle_pair_names("R", MEAN_GRID),
    "O_core": _ordered_pair_names(
        "O",
        [
            (10, 30),
            (30, 10),
            (30, 60),
            (60, 30),
            (60, 90),
            (90, 60),
            (90, 120),
            (120, 90),
            (10, 90),
            (90, 10),
            (10, 120),
            (120, 10),
            (30, 90),
            (90, 30),
        ],
    ),
    "O_grid": _directed_pair_names("O", MEAN_GRID),
    "F1_core": _ordered_pair_names(
        "F1",
        [
            (10, 90),
            (90, 10),
            (30, 90),
            (90, 30),
            (10, 120),
            (120, 10),
        ],
    ),
    "F1_grid": _directed_pair_names("F1", MEAN_GRID),
    "F2_core": _ordered_pair_names(
        "F2",
        [
            (10, 90),
            (90, 10),
            (30, 90),
            (90, 30),
            (10, 120),
            (120, 10),
        ],
    ),
    "F2_grid": _directed_pair_names("F2", MEAN_GRID),
    "G_core": [
        "G_10_30_60",
        "G_30_60_90",
        "G_60_90_120",
        "G_10_60_30",
        "G_30_90_60",
        "G_60_120_90",
    ],
    "G_backbone": [
        "G_10_30_60",
        "G_30_60_90",
        "G_60_90_120",
        "G_10_60_30",
        "G_30_90_60",
        "G_60_120_90",
    ],
    "legacy_offgrid": [
        "R_30_80",
        "O_30_80",
        "O_60_20",
        "F2_30_80",
        "G_10_90_50",
        "G_10_40_90",
        "G_40_80_10",
    ],
}

WAVE_REGISTRY: Dict[str, List[str]] = {
    "smoke": _dedupe_keep_order(
        [
            "M_60",
            "R_10_120",
            "O_10_120",
            "O_120_10",
            "F1_10_120",
            "F2_120_10",
            "G_60_90_120",
        ]
    ),
    "main_36": _dedupe_keep_order(
        [
            "M_10",
            "M_30",
            "M_60",
            "M_90",
            "M_120",
            "R_10_30",
            "R_30_60",
            "R_60_90",
            "R_90_120",
            "R_10_90",
            "R_30_90",
            "R_10_120",
            "O_10_30",
            "O_30_10",
            "O_30_60",
            "O_60_30",
            "O_60_90",
            "O_90_60",
            "O_90_120",
            "O_120_90",
            "O_10_120",
            "O_120_10",
            "F1_10_90",
            "F1_90_10",
            "F1_10_120",
            "F1_120_10",
            "F2_10_90",
            "F2_90_10",
            "F2_10_120",
            "F2_120_10",
            "G_10_30_60",
            "G_30_60_90",
            "G_60_90_120",
            "G_10_60_30",
            "G_30_90_60",
            "G_60_120_90",
        ]
    ),
    "core_shift": _dedupe_keep_order(
        FAMILY_REGISTRY["M_core"] + FAMILY_REGISTRY["R_core"] + FAMILY_REGISTRY["O_core"]
    ),
    "memory_generalization": _dedupe_keep_order(
        FAMILY_REGISTRY["F1_core"] + FAMILY_REGISTRY["F2_core"] + FAMILY_REGISTRY["G_core"]
    ),
    "full_main": _dedupe_keep_order(
        FAMILY_REGISTRY["M_core"]
        + FAMILY_REGISTRY["R_core"]
        + FAMILY_REGISTRY["O_core"]
        + FAMILY_REGISTRY["F1_core"]
        + FAMILY_REGISTRY["F2_core"]
        + FAMILY_REGISTRY["G_core"]
    ),
    "full_pair_grid": _dedupe_keep_order(
        FAMILY_REGISTRY["M_grid"]
        + FAMILY_REGISTRY["R_grid"]
        + FAMILY_REGISTRY["O_grid"]
        + FAMILY_REGISTRY["F1_grid"]
        + FAMILY_REGISTRY["F2_grid"]
        + FAMILY_REGISTRY["G_backbone"]
    ),
    "legacy_appendix": _dedupe_keep_order(FAMILY_REGISTRY["legacy_offgrid"]),
}


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


def available_wave_names() -> List[str]:
    return list(WAVE_REGISTRY.keys())


def resolve_wave_names(raw_waves: Sequence[str] | None) -> List[str]:
    values = _dedupe_keep_order(raw_waves or [])
    if not values:
        return ["full_main"]
    unknown = [name for name in values if name not in WAVE_REGISTRY]
    if unknown:
        raise ValueError(f"unknown wave(s): {unknown}; available={available_wave_names()}")
    return values


def resolve_distributions_for_waves(
    wave_names: Sequence[str],
    *,
    incremental: bool = True,
) -> Dict[str, List[str]]:
    resolved: Dict[str, List[str]] = {}
    seen = set()
    for wave_name in resolve_wave_names(wave_names):
        items = list(WAVE_REGISTRY[wave_name])
        if incremental:
            items = [name for name in items if name not in seen]
        for name in items:
            seen.add(name)
        resolved[wave_name] = items
    return resolved


def family_of_dist(dist_name: str) -> str:
    raw = str(dist_name or "").strip()
    if not raw:
        return ""
    if raw.startswith("F1_"):
        return "F1"
    if raw.startswith("F2_"):
        return "F2"
    return raw.split("_", 1)[0]

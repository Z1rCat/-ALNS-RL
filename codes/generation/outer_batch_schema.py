from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import os


ALLOWED_PATTERNS = {"ab", "aba", "abba", "random_mix"}
ALLOWED_DISTS = {"normal", "lognormal"}
DEFAULT_MIN_NUM_FILES = 3
DEFAULT_MAX_NUM_FILES = 200


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def resolve_num_file_bounds() -> Tuple[int, int]:
    min_files = int(max(1, _env_int("OUTER_BATCH_MIN_FILES", DEFAULT_MIN_NUM_FILES)))
    max_files = int(max(min_files, _env_int("OUTER_BATCH_MAX_FILES", DEFAULT_MAX_NUM_FILES)))
    return min_files, max_files


@dataclass(frozen=True)
class OuterBatchAction:
    mu_a: float
    mu_b: float
    ratio_a: float
    num_files: int
    seed: int
    pattern: str = "ab"
    std_a: Optional[float] = None
    std_b: Optional[float] = None
    dist: str = "normal"


def validate_action(action: OuterBatchAction) -> None:
    errors: List[str] = []
    min_files, max_files = resolve_num_file_bounds()
    if action.num_files < int(min_files) or action.num_files > int(max_files):
        errors.append(f"num_files must be in [{int(min_files)}, {int(max_files)}]")
    if not (0.0 < float(action.ratio_a) < 1.0):
        errors.append("ratio_a must be in (0, 1)")
    if float(action.mu_a) <= 0 or float(action.mu_b) <= 0:
        errors.append("mu_a and mu_b must be > 0")
    if action.std_a is not None and float(action.std_a) <= 0:
        errors.append("std_a must be > 0 when provided")
    if action.std_b is not None and float(action.std_b) <= 0:
        errors.append("std_b must be > 0 when provided")
    if action.pattern not in ALLOWED_PATTERNS:
        errors.append(f"pattern must be one of {sorted(ALLOWED_PATTERNS)}")
    if action.dist not in ALLOWED_DISTS:
        errors.append(f"dist must be one of {sorted(ALLOWED_DISTS)}")
    if errors:
        raise ValueError("; ".join(errors))


def phase_counts(num_files: int, ratio_a: float) -> Tuple[int, int]:
    n_a = int(np.floor(float(ratio_a) * int(num_files) + 0.5))
    n_a = max(1, min(int(num_files) - 1, n_a))
    n_b = int(num_files) - n_a
    return n_a, n_b


def build_phase_labels(
    num_files: int,
    ratio_a: float,
    pattern: str,
    seed: int,
) -> List[str]:
    n_a, n_b = phase_counts(num_files, ratio_a)
    if pattern == "ab":
        return (["A"] * n_a) + (["B"] * n_b)
    if pattern == "aba":
        left_a = (n_a + 1) // 2
        right_a = n_a - left_a
        return (["A"] * left_a) + (["B"] * n_b) + (["A"] * right_a)
    if pattern == "abba":
        left_a = (n_a + 1) // 2
        right_a = n_a - left_a
        left_b = n_b // 2
        right_b = n_b - left_b
        return (["A"] * left_a) + (["B"] * left_b) + (["B"] * right_b) + (["A"] * right_a)
    labels = (["A"] * n_a) + (["B"] * n_b)
    rng = np.random.RandomState(int(seed))
    rng.shuffle(labels)
    return labels


def action_to_dict(action: OuterBatchAction) -> Dict[str, object]:
    return {
        "mu_a": float(action.mu_a),
        "mu_b": float(action.mu_b),
        "ratio_a": float(action.ratio_a),
        "num_files": int(action.num_files),
        "seed": int(action.seed),
        "pattern": str(action.pattern),
        "std_a": None if action.std_a is None else float(action.std_a),
        "std_b": None if action.std_b is None else float(action.std_b),
        "dist": str(action.dist),
    }

from __future__ import annotations

import argparse
import csv
import json
import itertools
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ROOT_DIR = Path(__file__).resolve().parents[2]
CODES_DIR = ROOT_DIR / "codes"
GENERATION_DIR = CODES_DIR / "generation"
FORCED_EDRL_CONVERGE_MINORITY_FLOOR = 0.01

if str(GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATION_DIR))
try:
    from outer_batch_schema import resolve_num_file_bounds
except Exception:
    def resolve_num_file_bounds() -> Tuple[int, int]:
        return 3, 200


def _append_csv_row(path: Path, fieldnames: List[str], row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _rewrite_csv_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _safe_int(value, default: int = -1) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return int(default)


def _get_max_iter_id(path: Path) -> int:
    rows = _read_csv_rows(path)
    best = -1
    for row in rows:
        rid = _safe_int(row.get("iter_id", ""), default=-1)
        if rid > best:
            best = rid
    return int(best)


def _drop_rows_ge_iter(path: Path, start_iter: int) -> None:
    rows = _read_csv_rows(path)
    if not rows:
        return
    keep: List[Dict[str, str]] = []
    for row in rows:
        rid = _safe_int(row.get("iter_id", ""), default=-1)
        if rid < 0 or rid < int(start_iter):
            keep.append(row)
    _rewrite_csv_rows(path, keep) if keep else path.unlink(missing_ok=True)


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _is_nan(x: float) -> bool:
    return x != x


def _calc_metrics(
    decision_rows: List[Dict[str, str]]
) -> Tuple[float, float, float, float, str]:
    if not decision_rows:
        return float("nan"), float("nan"), float("nan"), float("nan"), "na"
    rewards = []
    action0_count = 0
    action1_count = 0
    for row in decision_rows:
        rewards.append(_safe_float(row.get("reward", "")))
        phase_val = str(row.get("phase", "")).strip().lower()
        stage_mode_val = str(row.get("stage_mode", "")).strip().lower()
        is_train_row = (
            (not phase_val or phase_val in {"train", "phase1"})
            and (not stage_mode_val or stage_mode_val in {"train", "train_only"})
        )
        if not is_train_row:
            continue
        action_val = str(row.get("action", "")).strip()
        if action_val == "0":
            action0_count += 1
        elif action_val == "1":
            action1_count += 1
    reward_vals = [x for x in rewards if not _is_nan(x)]
    avg_reward = sum(reward_vals) / len(reward_vals) if reward_vals else float("nan")
    action_total = action0_count + action1_count
    if action_total <= 0:
        action0_rate = float("nan")
        action1_rate = float("nan")
        minority_rate = float("nan")
        minority_action = "na"
    else:
        action0_rate = float(action0_count) / float(action_total)
        action1_rate = float(action1_count) / float(action_total)
        if action0_rate <= action1_rate:
            minority_rate = float(action0_rate)
            minority_action = "0"
        else:
            minority_rate = float(action1_rate)
            minority_action = "1"
    return avg_reward, action0_rate, action1_rate, minority_rate, minority_action


def _trace_train_stage_family(row: Dict[str, str]) -> str:
    family = str(row.get("stage_family", "")).strip().lower()
    if family:
        return family
    stage = str(row.get("stage", "")).strip().lower()
    if "removal" in stage:
        return "removal"
    if "insertion" in stage:
        return "insertion"
    return ""


def _trace_is_train_finish_row(row: Dict[str, str]) -> bool:
    phase_val = str(row.get("phase", "")).strip().lower()
    if phase_val not in {"train", "phase1"}:
        return False
    stage = str(row.get("stage", "")).strip().lower()
    if not stage.startswith("finish_"):
        return False
    reward = _safe_float(row.get("reward", ""))
    if not _is_finite(reward):
        return False
    if reward <= -9999999.0:
        return False
    action_val = str(row.get("action", "")).strip()
    if action_val not in {"0", "1"}:
        return False
    return True


def _calc_saber_v1_trace_metrics(
    trace_rows: List[Dict[str, str]],
    hard_threshold: int,
    easy_threshold: int,
) -> Dict[str, float]:
    hard_rewards: List[float] = []
    hard_action1_reward: List[float] = []
    hard_action1: List[float] = []
    hard_wait: List[float] = []
    easy_wait_reward: List[float] = []
    easy_action1: List[float] = []
    removal_rewards: List[float] = []
    removal_count = 0
    hard_count = 0
    easy_count = 0
    insertion_hard_rewards: List[float] = []
    insertion_hard_count = 0
    for row in trace_rows:
        if not _trace_is_train_finish_row(row):
            continue
        family = _trace_train_stage_family(row)
        reward = _safe_float(row.get("reward", ""))
        action_val = str(row.get("action", "")).strip()
        action1 = 1.0 if action_val == "1" else 0.0
        action0 = 1.0 if action_val == "0" else 0.0
        severity = _safe_float(row.get("severity", ""))
        if family == "removal":
            removal_count += 1
            removal_rewards.append(float(reward))
            if _is_finite(severity) and float(severity) >= float(hard_threshold):
                hard_count += 1
                hard_rewards.append(float(reward))
                hard_action1_reward.append(float(action1 * reward))
                hard_action1.append(float(action1))
                hard_wait.append(float(action0))
            if _is_finite(severity) and float(severity) <= float(easy_threshold):
                easy_count += 1
                easy_wait_reward.append(float(action0 * reward))
                easy_action1.append(float(action1))
        elif family == "insertion":
            if _is_finite(severity) and float(severity) >= float(hard_threshold):
                insertion_hard_count += 1
                insertion_hard_rewards.append(float(reward))
    return {
        "removal_count": float(removal_count),
        "hard_count": float(hard_count),
        "easy_count": float(easy_count),
        "insertion_hard_count": float(insertion_hard_count),
        "avg_reward_removal": _safe_mean(removal_rewards, default=float("nan")),
        "Q_hard_rem": _safe_mean(hard_rewards, default=float("nan")),
        "R_hard_rem": _safe_mean(hard_action1_reward, default=float("nan")),
        "P_easy_wait": _safe_mean(easy_wait_reward, default=float("nan")),
        "hard_action1_rate": _safe_mean(hard_action1, default=float("nan")),
        "hard_wait_share": _safe_mean(hard_wait, default=float("nan")),
        "easy_action1_rate": _safe_mean(easy_action1, default=float("nan")),
        "M_ins": _safe_mean(insertion_hard_rewards, default=float("nan")),
    }


def _is_finite(x: float) -> bool:
    return (x == x) and math.isfinite(x)


def _safe_mean(values: List[float], default: float = 0.0) -> float:
    clean = [float(v) for v in values if _is_finite(float(v))]
    if not clean:
        return float(default)
    return float(sum(clean)) / float(len(clean))


def _binary_entropy01(p: float) -> float:
    pv = float(_finite_or(p, 0.5))
    pv = min(1.0, max(0.0, pv))
    if pv <= 1e-12 or pv >= 1.0 - 1e-12:
        return 0.0
    # Normalized binary entropy in [0, 1].
    return float((-(pv * math.log(pv) + (1.0 - pv) * math.log(1.0 - pv))) / math.log(2.0))


def _window_converged(
    history: List[Dict[str, float]],
    patience: int,
    max_abs_dj: float,
    max_obj_range: float,
    min_minority_rate: float,
) -> bool:
    p = max(1, int(patience))
    if len(history) < p:
        return False
    window = history[-p:]
    dj_vals = [float(item.get("dJ", float("nan"))) for item in window]
    obj_vals = [float(item.get("objective", float("nan"))) for item in window]
    minority_vals = [float(item.get("minority_rate", float("nan"))) for item in window]
    if not all(_is_finite(v) for v in dj_vals + obj_vals + minority_vals):
        return False
    if any(abs(v) > float(max_abs_dj) for v in dj_vals):
        return False
    if (max(obj_vals) - min(obj_vals)) > float(max_obj_range):
        return False
    if any(v < float(min_minority_rate) for v in minority_vals):
        return False
    return True


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_str_list(raw: str) -> List[str]:
    return [str(x.strip()) for x in raw.split(",") if x.strip()]


def _action_key(action: Dict[str, object]) -> Tuple[int, int, float, int, str]:
    return (
        int(action["mu_a"]),
        int(action["mu_b"]),
        float(action["ratio_a"]),
        int(action["num_files"]),
        str(action["pattern"]),
    )


def _derive_iter_seed(base_seed: int, iter_id: int) -> int:
    """Deterministic per-iteration seed (action-independent)."""
    b = int(base_seed) & 0xFFFFFFFF
    i = int(iter_id) & 0xFFFFFFFF
    mixed = (1664525 * (b ^ 0x9E3779B9) + 1013904223 + 2654435761 * i) & 0xFFFFFFFF
    return int(mixed % 9_999_999) + 1


def _phase2_frozen_key(action: Dict[str, object]) -> str:
    # Ignore num_files so phase3 can reuse phase2 frozen estimates across N values.
    return (
        f"{int(action['mu_a'])}|{int(action['mu_b'])}|"
        f"{float(action['ratio_a']):.8f}|{str(action['pattern'])}"
    )


def _normalize_allowed_indices(total_actions: int, allowed_indices: Optional[List[int]]) -> List[int]:
    total = max(0, int(total_actions))
    if total <= 0:
        return []
    if not allowed_indices:
        return list(range(total))
    cleaned: List[int] = []
    seen = set()
    for idx in allowed_indices:
        i = int(idx)
        if i < 0 or i >= total:
            continue
        if i in seen:
            continue
        seen.add(i)
        cleaned.append(i)
    return cleaned if cleaned else list(range(total))


def _sample_action(
    rng: random.Random,
    mu_choices: List[int],
    ratio_choices: List[float],
    num_file_choices: List[int],
    pattern_choices: List[str],
) -> Dict[str, object]:
    mu_a = rng.choice(mu_choices)
    mu_b = rng.choice(mu_choices)
    while mu_b == mu_a and len(mu_choices) > 1:
        mu_b = rng.choice(mu_choices)
    return {
        "mu_a": int(mu_a),
        "mu_b": int(mu_b),
        "ratio_a": float(rng.choice(ratio_choices)),
        "num_files": int(rng.choice(num_file_choices)),
        "pattern": str(rng.choice(pattern_choices)),
    }


def _build_action_space(
    mu_choices: List[int],
    ratio_choices: List[float],
    num_file_choices: List[int],
    pattern_choices: List[str],
) -> List[Dict[str, object]]:
    action_space: List[Dict[str, object]] = []
    allow_same_mu = len(mu_choices) <= 1
    for mu_a, mu_b, ratio_a, n_files, pattern in itertools.product(
        mu_choices, mu_choices, ratio_choices, num_file_choices, pattern_choices
    ):
        if (not allow_same_mu) and int(mu_a) == int(mu_b):
            continue
        action_space.append(
            {
                "mu_a": int(mu_a),
                "mu_b": int(mu_b),
                "ratio_a": float(ratio_a),
                "num_files": int(n_files),
                "pattern": str(pattern),
            }
        )
    return action_space


def _build_action_space_v2_mu_only(
    mu_choices: List[int],
    fixed_ratio_a: float,
    num_file_choices: List[int],
    fixed_pattern: str,
) -> List[Dict[str, object]]:
    """EDRL v2 action space: mu varies; ratio/pattern fixed; N uses configured candidates."""
    action_space: List[Dict[str, object]] = []
    seen = set()
    for mu in mu_choices:
        mu_i = int(mu)
        for n_files in num_file_choices:
            n_i = int(n_files)
            sig = (mu_i, n_i)
            if sig in seen:
                continue
            seen.add(sig)
            action_space.append(
                {
                    "mu_a": int(mu_i),
                    "mu_b": int(mu_i),
                    "ratio_a": float(fixed_ratio_a),
                    "num_files": int(n_i),
                    "pattern": str(fixed_pattern),
                }
            )
    return action_space


def _action_signature(action: Dict[str, object]) -> str:
    return (
        f"{int(action['mu_a'])}|{int(action['mu_b'])}|"
        f"{float(action['ratio_a']):.8f}|{int(action['num_files'])}|{str(action['pattern'])}"
    )


def _softmax(logits: List[float], temperature: float) -> List[float]:
    if not logits:
        return []
    t = max(1e-8, float(temperature))
    max_logit = max(logits)
    exps = [math.exp((float(x) - max_logit) / t) for x in logits]
    denom = sum(exps)
    if denom <= 0:
        return [1.0 / float(len(logits)) for _ in logits]
    return [float(x) / float(denom) for x in exps]


def _sample_index(rng: random.Random, probs: List[float]) -> int:
    if not probs:
        raise ValueError("empty probability list")
    r = float(rng.random())
    s = 0.0
    for i, p in enumerate(probs):
        s += max(0.0, float(p))
        if r <= s:
            return i
    return len(probs) - 1


def _load_or_init_pg_state(
    policy_state_path: Path,
    action_space: List[Dict[str, object]],
    reset: bool,
) -> Dict[str, object]:
    signatures = [_action_signature(a) for a in action_space]
    if reset and policy_state_path.exists():
        policy_state_path.unlink()

    if policy_state_path.exists():
        try:
            with policy_state_path.open("r", encoding="utf-8") as f:
                state = json.load(f)
            saved_signatures = list(state.get("action_signatures", []))
            saved_logits = list(state.get("logits", []))
            if saved_signatures == signatures and len(saved_logits) == len(signatures):
                state["logits"] = [float(x) for x in saved_logits]
                state["baseline"] = float(state.get("baseline", 0.0))
                state["steps"] = int(state.get("steps", 0))
                state["action_signatures"] = signatures
                return state
        except Exception:
            pass

    return {
        "algo": "pg_bandit_v1",
        "steps": 0,
        "baseline": 0.0,
        "logits": [0.0 for _ in action_space],
        "action_signatures": signatures,
        "last_update_ts": int(time.time()),
    }


def _save_pg_state(policy_state_path: Path, state: Dict[str, object]) -> None:
    policy_state_path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(state)
    state["last_update_ts"] = int(time.time())
    with policy_state_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=True, indent=2)


def _choose_action_pg(
    rng: random.Random,
    action_space: List[Dict[str, object]],
    state: Dict[str, object],
    temperature: float,
    allowed_indices: Optional[List[int]] = None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    logits = [float(x) for x in list(state.get("logits", []))]
    if len(logits) != len(action_space):
        logits = [0.0 for _ in action_space]
        state["logits"] = logits
    probs = _softmax(logits, float(temperature))
    allowed = _normalize_allowed_indices(len(action_space), allowed_indices)
    sub_probs = [max(0.0, float(probs[i])) for i in allowed]
    denom = sum(sub_probs)
    if denom <= 0.0:
        sub_probs = [1.0 / float(max(1, len(allowed))) for _ in allowed]
    else:
        sub_probs = [float(p) / float(denom) for p in sub_probs]
    local_idx = _sample_index(rng, sub_probs)
    action_idx = int(allowed[local_idx])
    action = dict(action_space[action_idx])
    entropy = 0.0
    for p in probs:
        pp = max(1e-12, float(p))
        entropy += -pp * math.log(pp)
    return action, {
        "action_idx": int(action_idx),
        "action_prob": float(probs[action_idx]),
        "entropy": float(entropy),
        "probs": probs,
    }


def _update_pg_state(
    state: Dict[str, object],
    action_idx: int,
    probs: List[float],
    objective: float,
    lr: float,
    baseline_momentum: float,
) -> Tuple[float, float]:
    logits = [float(x) for x in list(state.get("logits", []))]
    if len(logits) != len(probs):
        raise ValueError("policy logits/probs length mismatch")
    steps = int(state.get("steps", 0))
    baseline = float(state.get("baseline", 0.0))
    if steps <= 0:
        baseline = float(objective)
    advantage = float(objective) - float(baseline)
    step_lr = float(lr)
    for i, p in enumerate(probs):
        grad = (1.0 if int(i) == int(action_idx) else 0.0) - float(p)
        logits[i] += step_lr * advantage * grad

    m = min(0.999, max(0.0, float(baseline_momentum)))
    baseline = m * baseline + (1.0 - m) * float(objective)
    state["logits"] = logits
    state["baseline"] = float(baseline)
    state["steps"] = int(steps + 1)
    return float(advantage), float(baseline)


def _save_json_state(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data["last_update_ts"] = int(time.time())
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


def _load_or_init_ucb_state(
    policy_state_path: Path,
    action_space: List[Dict[str, object]],
    reset: bool,
) -> Dict[str, object]:
    signatures = [_action_signature(a) for a in action_space]
    if reset and policy_state_path.exists():
        policy_state_path.unlink()
    if policy_state_path.exists():
        try:
            with policy_state_path.open("r", encoding="utf-8") as f:
                state = json.load(f)
            if list(state.get("action_signatures", [])) == signatures:
                count = [float(x) for x in list(state.get("count", []))]
                sum_obj = [float(x) for x in list(state.get("sum_obj", []))]
                if len(count) == len(signatures) and len(sum_obj) == len(signatures):
                    return {
                        "algo": "ucb_bandit_v2",
                        "steps": int(state.get("steps", 0)),
                        "count": count,
                        "sum_obj": sum_obj,
                        "action_signatures": signatures,
                    }
        except Exception:
            pass
    return {
        "algo": "ucb_bandit_v2",
        "steps": 0,
        "count": [0.0 for _ in action_space],
        "sum_obj": [0.0 for _ in action_space],
        "action_signatures": signatures,
    }


def _load_or_init_ts_state(
    policy_state_path: Path,
    action_space: List[Dict[str, object]],
    reset: bool,
) -> Dict[str, object]:
    signatures = [_action_signature(a) for a in action_space]
    if reset and policy_state_path.exists():
        policy_state_path.unlink()
    if policy_state_path.exists():
        try:
            with policy_state_path.open("r", encoding="utf-8") as f:
                state = json.load(f)
            if list(state.get("action_signatures", [])) == signatures:
                count = [float(x) for x in list(state.get("count", []))]
                sum_obj = [float(x) for x in list(state.get("sum_obj", []))]
                if len(count) == len(signatures) and len(sum_obj) == len(signatures):
                    return {
                        "algo": "thompson_bandit_v1",
                        "steps": int(state.get("steps", 0)),
                        "count": count,
                        "sum_obj": sum_obj,
                        "action_signatures": signatures,
                    }
        except Exception:
            pass
    return {
        "algo": "thompson_bandit_v1",
        "steps": 0,
        "count": [0.0 for _ in action_space],
        "sum_obj": [0.0 for _ in action_space],
        "action_signatures": signatures,
    }


def _candidate_indices(
    rng: random.Random,
    total_actions: int,
    candidate_pool: int,
    allowed_indices: Optional[List[int]] = None,
) -> List[int]:
    candidates = _normalize_allowed_indices(total_actions, allowed_indices)
    total = len(candidates)
    if total <= 0:
        return []
    if int(candidate_pool) <= 0 or int(candidate_pool) >= int(total):
        return list(candidates)
    return rng.sample(list(candidates), int(candidate_pool))


def _choose_action_ucb(
    rng: random.Random,
    action_space: List[Dict[str, object]],
    state: Dict[str, object],
    ucb_c: float,
    candidate_pool: int,
    allowed_indices: Optional[List[int]] = None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    count = [float(x) for x in list(state.get("count", []))]
    sum_obj = [float(x) for x in list(state.get("sum_obj", []))]
    if len(count) != len(action_space) or len(sum_obj) != len(action_space):
        count = [0.0 for _ in action_space]
        sum_obj = [0.0 for _ in action_space]
        state["count"] = count
        state["sum_obj"] = sum_obj

    base_indices = _normalize_allowed_indices(len(action_space), allowed_indices)
    indices = _candidate_indices(
        rng,
        len(action_space),
        int(candidate_pool),
        allowed_indices=base_indices,
    )
    unseen = [i for i in base_indices if float(count[i]) <= 1e-12]
    if unseen:
        action_idx = int(rng.choice(unseen))
        action = dict(action_space[action_idx])
        return action, {
            "action_idx": int(action_idx),
            "score": float("inf"),
            "mean_obj": 0.0,
            "count": float(count[action_idx]),
            "unseen_pick": 1,
        }

    total_count = sum(max(0.0, float(c)) for c in count) + 1.0
    best_idx = -1
    best_score = -1e30
    best_mean = float("nan")
    for i in indices:
        c = max(1e-12, float(count[i]))
        mean_obj = float(sum_obj[i]) / float(c)
        bonus = float(ucb_c) * math.sqrt(math.log(float(total_count)) / float(c))
        score = mean_obj + bonus
        if score > best_score:
            best_score = score
            best_idx = int(i)
            best_mean = mean_obj
    if best_idx < 0:
        best_idx = int(rng.choice(base_indices))
    action = dict(action_space[best_idx])
    return action, {
        "action_idx": int(best_idx),
        "score": float(best_score),
        "mean_obj": float(best_mean),
        "count": float(count[best_idx]),
    }


def _update_ucb_state(
    state: Dict[str, object],
    action_idx: int,
    objective: float,
    decay: float,
) -> None:
    count = [float(x) for x in list(state.get("count", []))]
    sum_obj = [float(x) for x in list(state.get("sum_obj", []))]
    if len(count) != len(sum_obj):
        raise ValueError("ucb state length mismatch")
    d = min(1.0, max(0.0, float(decay)))
    if d < 1.0:
        count = [float(c) * d for c in count]
        sum_obj = [float(s) * d for s in sum_obj]
    ai = int(action_idx)
    if ai < 0 or ai >= len(count):
        raise IndexError("ucb action idx out of range")
    count[ai] += 1.0
    sum_obj[ai] += float(objective)
    state["count"] = count
    state["sum_obj"] = sum_obj
    state["steps"] = int(state.get("steps", 0)) + 1


def _choose_action_ts(
    rng: random.Random,
    action_space: List[Dict[str, object]],
    state: Dict[str, object],
    prior_mean: float,
    prior_std: float,
    obs_std: float,
    candidate_pool: int,
    allowed_indices: Optional[List[int]] = None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    count = [float(x) for x in list(state.get("count", []))]
    sum_obj = [float(x) for x in list(state.get("sum_obj", []))]
    if len(count) != len(action_space) or len(sum_obj) != len(action_space):
        count = [0.0 for _ in action_space]
        sum_obj = [0.0 for _ in action_space]
        state["count"] = count
        state["sum_obj"] = sum_obj

    base_indices = _normalize_allowed_indices(len(action_space), allowed_indices)
    indices = _candidate_indices(
        rng,
        len(action_space),
        int(candidate_pool),
        allowed_indices=base_indices,
    )
    prior_var = max(1e-8, float(prior_std) ** 2)
    obs_var = max(1e-8, float(obs_std) ** 2)
    best_idx = -1
    best_draw = -1e30
    best_post_mean = float("nan")
    best_post_var = float("nan")
    for i in indices:
        c = max(0.0, float(count[i]))
        if c <= 1e-12:
            post_mean = float(prior_mean)
            post_var = float(prior_var)
        else:
            sample_mean = float(sum_obj[i]) / float(c)
            post_var = 1.0 / (1.0 / prior_var + c / obs_var)
            post_mean = post_var * (float(prior_mean) / prior_var + c * sample_mean / obs_var)
        draw = rng.gauss(post_mean, math.sqrt(max(1e-12, post_var)))
        if draw > best_draw:
            best_draw = draw
            best_idx = int(i)
            best_post_mean = float(post_mean)
            best_post_var = float(post_var)
    if best_idx < 0:
        best_idx = int(rng.choice(base_indices))
    action = dict(action_space[best_idx])
    return action, {
        "action_idx": int(best_idx),
        "ts_draw": float(best_draw),
        "posterior_mean": float(best_post_mean),
        "posterior_var": float(best_post_var),
        "count": float(count[best_idx]),
    }


def _update_ts_state(
    state: Dict[str, object],
    action_idx: int,
    objective: float,
    decay: float,
) -> None:
    count = [float(x) for x in list(state.get("count", []))]
    sum_obj = [float(x) for x in list(state.get("sum_obj", []))]
    if len(count) != len(sum_obj):
        raise ValueError("ts state length mismatch")
    d = min(1.0, max(0.0, float(decay)))
    if d < 1.0:
        count = [float(c) * d for c in count]
        sum_obj = [float(s) * d for s in sum_obj]
    ai = int(action_idx)
    if ai < 0 or ai >= len(count):
        raise IndexError("ts action idx out of range")
    count[ai] += 1.0
    sum_obj[ai] += float(objective)
    state["count"] = count
    state["sum_obj"] = sum_obj
    state["steps"] = int(state.get("steps", 0)) + 1


def _bootstrap_bandit_state_with_topk(
    state: Dict[str, object],
    action_space: List[Dict[str, object]],
    topk_action_ids: List[int],
    agg: Dict[int, Dict[str, float]],
    prior_count: float,
) -> None:
    pc = max(0.0, float(prior_count))
    state["count"] = [0.0 for _ in action_space]
    state["sum_obj"] = [0.0 for _ in action_space]
    for aid in topk_action_ids:
        i = int(aid)
        if i < 0 or i >= len(action_space):
            continue
        item = agg.get(i, {"sum": 0.0, "count": 0.0})
        cnt = float(item.get("count", 0.0))
        mean_obj = (float(item.get("sum", 0.0)) / cnt) if cnt > 0.0 else 0.0
        state["count"][i] = float(pc)
        state["sum_obj"][i] = float(pc) * float(mean_obj)
    state["steps"] = int(state.get("steps", 0)) + 1


def _finite_or(value: float, default: float = 0.0) -> float:
    return float(value) if _is_finite(float(value)) else float(default)


def _clip(value: float, lo: float, hi: float) -> float:
    return min(float(hi), max(float(lo), float(value)))


def _build_outer_policy_obs(
    *,
    last_j: float,
    last_dj: float,
    last_action0_rate: float,
    last_action1_rate: float,
    last_minority_rate: float,
    last_objective: float,
    last_policy_entropy: float,
    recent_j_mean: float,
    recent_action1_mean: float,
    recent_entropy_mean: float,
    iter_phase: str,
    iter_idx: int,
    total_iters: int,
) -> List[float]:
    progress = 0.0
    if int(total_iters) > 0:
        progress = float(iter_idx) / float(total_iters)
    phase_val = 1.0 if str(iter_phase).strip().lower() == "phase3" else 0.0
    return [
        _clip(_finite_or(last_j, 0.0), -1.0, 1.0),
        _clip(_finite_or(last_dj, 0.0), -1.0, 1.0),
        _clip(_finite_or(last_action0_rate, 0.5), 0.0, 1.0),
        _clip(_finite_or(last_action1_rate, 0.5), 0.0, 1.0),
        _clip(_finite_or(last_minority_rate, 0.0), 0.0, 1.0),
        _clip(_finite_or(last_objective, 0.0), -2.0, 2.0),
        _clip(_finite_or(last_policy_entropy, 0.0), 0.0, 1.0),
        _clip(_finite_or(recent_j_mean, 0.0), -1.0, 1.0),
        _clip(_finite_or(recent_action1_mean, 0.5), 0.0, 1.0),
        _clip(_finite_or(recent_entropy_mean, 0.0), 0.0, 1.0),
        _clip(float(phase_val), 0.0, 1.0),
        _clip(float(progress), 0.0, 1.0),
    ]


def _safe_obs(vec: List[float], state_dim: int) -> List[float]:
    out = [0.0 for _ in range(max(1, int(state_dim)))]
    for i in range(min(len(out), len(vec))):
        out[i] = float(_finite_or(vec[i], 0.0))
    return out


def _load_or_init_rarl_dqn_state(
    policy_state_path: Path,
    action_space: List[Dict[str, object]],
    reset: bool,
    replay_capacity: int,
    state_dim: int,
) -> Dict[str, object]:
    signatures = [_action_signature(a) for a in action_space]
    n_actions = len(action_space)
    state_dim = max(1, int(state_dim))
    replay_capacity = max(1, int(replay_capacity))
    if reset and policy_state_path.exists():
        policy_state_path.unlink()

    def _new_state() -> Dict[str, object]:
        local_rng = random.Random(1337)
        scale = 0.01
        weights = [
            [float(local_rng.uniform(-scale, scale)) for _ in range(state_dim)]
            for _ in range(n_actions)
        ]
        bias = [0.0 for _ in range(n_actions)]
        return {
            "algo": "rarl_dqn_linear_v1",
            "steps": 0,
            "train_updates": 0,
            "state_dim": int(state_dim),
            "replay_capacity": int(replay_capacity),
            "action_signatures": signatures,
            "weights": weights,
            "bias": list(bias),
            "target_weights": [list(row) for row in weights],
            "target_bias": list(bias),
            "replay": [],
        }

    if not policy_state_path.exists():
        return _new_state()
    try:
        with policy_state_path.open("r", encoding="utf-8") as f:
            saved = json.load(f)
        if list(saved.get("action_signatures", [])) != signatures:
            return _new_state()
        saved_dim = int(saved.get("state_dim", state_dim))
        if saved_dim != state_dim:
            return _new_state()
        weights = list(saved.get("weights", []))
        bias = list(saved.get("bias", []))
        target_weights = list(saved.get("target_weights", []))
        target_bias = list(saved.get("target_bias", []))
        if len(weights) != n_actions or len(bias) != n_actions:
            return _new_state()
        if len(target_weights) != n_actions or len(target_bias) != n_actions:
            target_weights = [list(row) for row in weights]
            target_bias = [float(x) for x in bias]
        clean_weights: List[List[float]] = []
        clean_target_weights: List[List[float]] = []
        for i in range(n_actions):
            row = [float(_finite_or(x, 0.0)) for x in list(weights[i])]
            if len(row) != state_dim:
                return _new_state()
            clean_weights.append(row)
            trow = [float(_finite_or(x, 0.0)) for x in list(target_weights[i])]
            if len(trow) != state_dim:
                trow = list(row)
            clean_target_weights.append(trow)
        clean_bias = [float(_finite_or(x, 0.0)) for x in list(bias)]
        clean_target_bias = [float(_finite_or(x, 0.0)) for x in list(target_bias)]
        replay_clean: List[Dict[str, object]] = []
        for item in list(saved.get("replay", [])):
            s = _safe_obs(list(item.get("s", [])), state_dim)
            s2 = _safe_obs(list(item.get("s2", [])), state_dim)
            a = int(max(0, min(n_actions - 1, _safe_int(item.get("a", 0), default=0))))
            r = float(_finite_or(_safe_float(item.get("r", 0.0)), 0.0))
            done = int(1 if str(item.get("done", 0)).strip() in {"1", "true", "True"} else 0)
            replay_clean.append({"s": s, "a": a, "r": r, "s2": s2, "done": done})
        if len(replay_clean) > replay_capacity:
            replay_clean = replay_clean[-int(replay_capacity):]
        return {
            "algo": "rarl_dqn_linear_v1",
            "steps": int(max(0, _safe_int(saved.get("steps", 0), default=0))),
            "train_updates": int(max(0, _safe_int(saved.get("train_updates", 0), default=0))),
            "state_dim": int(state_dim),
            "replay_capacity": int(replay_capacity),
            "action_signatures": signatures,
            "weights": clean_weights,
            "bias": clean_bias,
            "target_weights": clean_target_weights,
            "target_bias": clean_target_bias,
            "replay": replay_clean,
        }
    except Exception:
        return _new_state()


def _q_values_linear(weights: List[List[float]], bias: List[float], obs: List[float]) -> List[float]:
    out: List[float] = []
    for i, row in enumerate(weights):
        q = float(bias[i]) if i < len(bias) else 0.0
        for j, w in enumerate(row):
            if j < len(obs):
                q += float(w) * float(obs[j])
        out.append(float(q))
    return out


def _rarl_schedule_epsilon(
    steps: int,
    eps_start: float,
    eps_end: float,
    eps_decay_iters: int,
) -> float:
    es = max(0.0, float(eps_start))
    ee = max(0.0, float(eps_end))
    if int(eps_decay_iters) <= 0:
        return float(ee)
    frac = min(1.0, max(0.0, float(steps) / float(max(1, int(eps_decay_iters)))))
    return float(es + (ee - es) * frac)


def _choose_action_rarl_dqn(
    rng: random.Random,
    action_space: List[Dict[str, object]],
    state: Dict[str, object],
    obs: List[float],
    epsilon: float,
    allowed_indices: Optional[List[int]] = None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    n_actions = len(action_space)
    if n_actions <= 0:
        raise ValueError("empty action space")
    weights = [list(row) for row in list(state.get("weights", []))]
    bias = [float(x) for x in list(state.get("bias", []))]
    q_vals = _q_values_linear(weights=weights, bias=bias, obs=obs)
    allowed = _normalize_allowed_indices(n_actions, allowed_indices)
    allowed_q = [float(q_vals[i]) for i in allowed] if q_vals else []
    eps = min(1.0, max(0.0, float(epsilon)))
    explore = float(rng.random()) < eps
    if explore:
        action_idx = int(rng.choice(allowed))
    else:
        best_q = max(allowed_q) if allowed_q else 0.0
        best_idx = [i for i in allowed if abs(float(q_vals[i]) - float(best_q)) <= 1e-12] if q_vals else list(allowed)
        action_idx = int(rng.choice(best_idx if best_idx else list(allowed)))
    action = dict(action_space[action_idx])
    state["steps"] = int(state.get("steps", 0)) + 1
    return action, {
        "action_idx": int(action_idx),
        "epsilon": float(eps),
        "explore": int(1 if explore else 0),
        "q_selected": float(q_vals[action_idx]) if q_vals else float("nan"),
        "q_max": float(max(allowed_q)) if q_vals and allowed_q else float("nan"),
    }


def _rarl_push_transition(
    state: Dict[str, object],
    obs: List[float],
    action_idx: int,
    reward: float,
    next_obs: List[float],
    done: int,
) -> None:
    replay = list(state.get("replay", []))
    cap = max(1, int(state.get("replay_capacity", 1)))
    replay.append(
        {
            "s": list(obs),
            "a": int(action_idx),
            "r": float(reward),
            "s2": list(next_obs),
            "done": int(1 if int(done) else 0),
        }
    )
    if len(replay) > cap:
        replay = replay[-cap:]
    state["replay"] = replay


def _rarl_sync_target(state: Dict[str, object]) -> None:
    state["target_weights"] = [list(row) for row in list(state.get("weights", []))]
    state["target_bias"] = [float(x) for x in list(state.get("bias", []))]


def _rarl_train_dqn(
    rng: random.Random,
    state: Dict[str, object],
    batch_size: int,
    updates: int,
    min_replay: int,
    gamma: float,
    lr: float,
    target_sync_every: int,
) -> Dict[str, float]:
    replay = list(state.get("replay", []))
    if not replay:
        return {"updates": 0.0, "loss": float("nan")}
    weights = [list(row) for row in list(state.get("weights", []))]
    bias = [float(x) for x in list(state.get("bias", []))]
    target_weights = [list(row) for row in list(state.get("target_weights", []))]
    target_bias = [float(x) for x in list(state.get("target_bias", []))]
    n_actions = len(weights)
    if n_actions <= 0:
        return {"updates": 0.0, "loss": float("nan")}
    bsz = max(1, int(batch_size))
    upd = max(0, int(updates))
    replay_min = max(1, int(min_replay))
    if len(replay) < replay_min:
        return {"updates": 0.0, "loss": float("nan")}
    g = float(gamma)
    step_lr = max(1e-8, float(lr))
    sync_every = max(1, int(target_sync_every))
    total_loss = 0.0
    done_updates = 0
    for _ in range(upd):
        if len(replay) < bsz:
            break
        idxs = rng.sample(range(len(replay)), bsz)
        loss_sum = 0.0
        for idx in idxs:
            tr = replay[idx]
            s = [float(_finite_or(x, 0.0)) for x in list(tr.get("s", []))]
            s2 = [float(_finite_or(x, 0.0)) for x in list(tr.get("s2", []))]
            a = int(max(0, min(n_actions - 1, _safe_int(tr.get("a", 0), default=0))))
            r = float(_finite_or(_safe_float(tr.get("r", 0.0)), 0.0))
            done = int(1 if str(tr.get("done", 0)).strip() in {"1", "true", "True"} else 0)

            q_sa = _q_values_linear(weights=weights, bias=bias, obs=s)[a]
            q_next_vals = _q_values_linear(weights=target_weights, bias=target_bias, obs=s2)
            q_next = max(q_next_vals) if q_next_vals else 0.0
            target = r if done else (r + g * q_next)
            td = _clip(target - q_sa, -5.0, 5.0)
            loss_sum += float(td * td)
            for j in range(len(weights[a])):
                sval = s[j] if j < len(s) else 0.0
                weights[a][j] += step_lr * float(td) * float(sval)
            bias[a] += step_lr * float(td)
        done_updates += 1
        total_loss += float(loss_sum) / float(max(1, bsz))
        train_updates = int(state.get("train_updates", 0)) + 1
        state["train_updates"] = int(train_updates)
        if (train_updates % sync_every) == 0:
            target_weights = [list(row) for row in weights]
            target_bias = [float(x) for x in bias]
    state["weights"] = weights
    state["bias"] = bias
    state["target_weights"] = target_weights
    state["target_bias"] = target_bias
    if done_updates <= 0:
        return {"updates": 0.0, "loss": float("nan")}
    return {"updates": float(done_updates), "loss": float(total_loss) / float(done_updates)}


def _extract_dynamic_index(file_path: Path) -> Optional[int]:
    m = re.search(r"congestion(\d+)\.xlsx$", str(file_path).replace("\\", "/"), flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _resolve_request_dir(root: Path, request_number: int) -> Path:
    candidate = root / f"R{int(request_number)}"
    if candidate.exists() and candidate.is_dir():
        return candidate
    return root


def _scan_dynamic_files(root: Path, request_number: int) -> List[Path]:
    r_dir = _resolve_request_dir(root, request_number)
    if not r_dir.exists():
        return []
    files = list(r_dir.glob("Intermodal_EGS_data_dynamic_congestion*.xlsx"))
    files.sort(key=lambda p: (_extract_dynamic_index(p) is None, _extract_dynamic_index(p) or 0, str(p)))
    return files


def _iter_curriculum_alpha(
    iter_idx: int,
    alpha_start: float,
    alpha_end: float,
    alpha_horizon: int,
) -> float:
    start = float(alpha_start)
    end = float(alpha_end)
    horizon = int(alpha_horizon)
    if horizon <= 1:
        return end
    ratio = min(1.0, max(0.0, float(iter_idx - 1) / float(max(1, horizon - 1))))
    return start + (end - start) * ratio


def _materialize_curriculum_batch(
    iter_idx: int,
    request_number: int,
    num_files: int,
    outer_iter_dir: Path,
    base_root: Path,
    out_mix_root: Path,
    alpha_outer: float,
    replay_file_pool: List[Path],
    replay_ratio: float,
    rng: random.Random,
) -> Dict[str, object]:
    outer_files = _scan_dynamic_files(outer_iter_dir, request_number)
    base_files = _scan_dynamic_files(base_root, request_number)
    if not outer_files:
        raise RuntimeError(f"outer curriculum source empty: {outer_iter_dir}")
    if not base_files:
        raise RuntimeError(f"base curriculum source empty: {base_root}")

    alpha = min(1.0, max(0.0, float(alpha_outer)))
    rr = min(1.0, max(0.0, float(replay_ratio)))
    mix_iter_dir = out_mix_root / f"iter_{int(iter_idx):03d}"
    mix_r_dir = mix_iter_dir / f"R{int(request_number)}"
    mix_r_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    outer_count = 0
    base_count = 0
    replay_count = 0
    for i in range(int(num_files)):
        dst = mix_r_dir / f"Intermodal_EGS_data_dynamic_congestion{i}.xlsx"
        source_type = "base"
        source_path: Path
        if float(rng.random()) < alpha:
            if replay_file_pool and float(rng.random()) < rr:
                source_type = "replay"
                source_path = Path(rng.choice(replay_file_pool))
                replay_count += 1
            else:
                source_type = "outer"
                source_path = outer_files[i % len(outer_files)]
                outer_count += 1
        else:
            source_path = base_files[i % len(base_files)]
            base_count += 1
        shutil.copy2(str(source_path), str(dst))
        rows.append(
            {
                "iter_id": int(iter_idx),
                "logical_idx": int(i),
                "source_type": str(source_type),
                "source_file": str(source_path.resolve()),
                "mixed_file": str(dst.resolve()),
            }
        )

    manifest = {
        "schema_version": 1,
        "iter_id": int(iter_idx),
        "request_number": int(request_number),
        "num_files": int(num_files),
        "alpha_outer": float(alpha),
        "replay_ratio": float(rr),
        "counts": {
            "outer": int(outer_count),
            "base": int(base_count),
            "replay": int(replay_count),
        },
        "paths": {
            "outer_iter_dir": str(outer_iter_dir.resolve()),
            "base_root": str(base_root.resolve()),
            "mix_iter_dir": str(mix_iter_dir.resolve()),
        },
        "files": rows,
    }
    manifest_path = mix_iter_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return {
        "mix_iter_dir": mix_iter_dir,
        "mix_root": out_mix_root,
        "manifest_path": manifest_path,
        "outer_count": int(outer_count),
        "base_count": int(base_count),
        "replay_count": int(replay_count),
        "rows": rows,
    }


def _choose_action(
    rng: random.Random,
    action_space: List[Dict[str, object]],
    iter_idx: int,
    warmup_iters: int,
    allowed_indices: Optional[List[int]] = None,
) -> Dict[str, object]:
    if not action_space:
        raise ValueError("empty action space")
    allowed = _normalize_allowed_indices(len(action_space), allowed_indices)
    if int(iter_idx) <= int(warmup_iters):
        action = dict(action_space[int(rng.choice(allowed))])
    else:
        action = dict(action_space[int(rng.choice(allowed))])
    return action


def _action_template(action: Dict[str, object]) -> Dict[str, object]:
    return {
        "mu_a": int(action["mu_a"]),
        "mu_b": int(action["mu_b"]),
        "ratio_a": float(action["ratio_a"]),
        "num_files": int(action["num_files"]),
        "pattern": str(action["pattern"]),
    }


def _materialize_action_from_template(template: Dict[str, object]) -> Dict[str, object]:
    return dict(_action_template(template))


def _phase_allowed_action_indices(
    action_space: List[Dict[str, object]],
    iter_phase: str,
    phase2_fixed_n: int,
    phase3_n_choices: List[int],
) -> List[int]:
    phase = str(iter_phase).strip().lower()
    if phase == "phase2":
        if int(phase2_fixed_n) <= 0:
            return list(range(len(action_space)))
        target = int(phase2_fixed_n)
        idxs = [i for i, a in enumerate(action_space) if int(a.get("num_files", -1)) == target]
        return idxs if idxs else list(range(len(action_space)))
    if phase == "phase3":
        allowed_n = {int(x) for x in phase3_n_choices}
        if not allowed_n:
            return list(range(len(action_space)))
        idxs = [i for i, a in enumerate(action_space) if int(a.get("num_files", -1)) in allowed_n]
        return idxs if idxs else list(range(len(action_space)))
    return list(range(len(action_space)))


def _load_or_init_plr_buffer(
    buffer_path: Path,
    action_space: List[Dict[str, object]],
    maxlen: int,
    reset: bool,
) -> deque:
    maxlen = int(max(1, maxlen))
    if reset and buffer_path.exists():
        buffer_path.unlink()
    entries = deque(maxlen=maxlen)
    by_sig = {_action_signature(a): _action_template(a) for a in action_space}
    if not buffer_path.exists():
        return entries
    try:
        with buffer_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for raw in list(payload.get("entries", [])):
            sig = str(raw.get("signature", "")).strip()
            if (not sig) or (sig not in by_sig):
                continue
            score_ema = _safe_float(raw.get("score_ema", 0.0))
            if not _is_finite(score_ema):
                score_ema = 0.0
            last_score = _safe_float(raw.get("last_score", score_ema))
            if not _is_finite(last_score):
                last_score = float(score_ema)
            entries.append(
                {
                    "signature": str(sig),
                    "action": dict(by_sig[sig]),
                    "score_ema": float(score_ema),
                    "last_score": float(last_score),
                    "n_seen": int(max(0, _safe_int(raw.get("n_seen", 0), default=0))),
                    "n_sampled": int(max(0, _safe_int(raw.get("n_sampled", 0), default=0))),
                    "last_iter": int(max(0, _safe_int(raw.get("last_iter", 0), default=0))),
                }
            )
    except Exception:
        pass
    return entries


def _save_plr_buffer(buffer_path: Path, entries: deque) -> None:
    buffer_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "algo": "plr_level_replay_v1",
        "maxlen": int(entries.maxlen or len(entries)),
        "entries": list(entries),
        "last_update_ts": int(time.time()),
    }
    with buffer_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def _plr_sample_replay_index(
    rng: random.Random,
    entries: deque,
    min_weight: float,
) -> int:
    if not entries:
        raise ValueError("empty plr replay buffer")
    floor = max(1e-12, float(min_weight))
    scores = []
    for item in entries:
        v = _safe_float(item.get("score_ema", 0.0))
        if not _is_finite(v):
            v = 0.0
        scores.append(float(v))
    min_score = min(scores) if scores else 0.0
    weights = [max(floor, float(s) - float(min_score) + floor) for s in scores]
    total = sum(weights)
    if total <= 0:
        probs = [1.0 / float(len(weights)) for _ in weights]
    else:
        probs = [float(w) / float(total) for w in weights]
    return int(_sample_index(rng, probs))


def _choose_action_plr_mixed(
    rng: random.Random,
    action_space: List[Dict[str, object]],
    replay_entries: deque,
    p_new: float,
    min_weight: float,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    use_new = (len(replay_entries) <= 0) or (float(rng.random()) < float(p_new))
    if use_new:
        base = dict(rng.choice(action_space))
        action = _materialize_action_from_template(base)
        return action, {
            "source": "new",
            "buffer_size": int(len(replay_entries)),
            "entry_weight": "",
            "entry_score_ema": "",
        }
    ridx = _plr_sample_replay_index(rng=rng, entries=replay_entries, min_weight=float(min_weight))
    entry = replay_entries[ridx]
    entry["n_sampled"] = int(entry.get("n_sampled", 0)) + 1
    action = _materialize_action_from_template(dict(entry.get("action", {})))
    return action, {
        "source": "replay",
        "buffer_size": int(len(replay_entries)),
        "entry_index": int(ridx),
        "entry_weight": "",
        "entry_score_ema": float(entry.get("score_ema", 0.0)),
    }


def _update_plr_buffer(
    replay_entries: deque,
    action: Dict[str, object],
    score: float,
    ema_alpha: float,
    iter_idx: int,
) -> Dict[str, object]:
    sig = _action_signature(action)
    alpha = min(1.0, max(0.0, float(ema_alpha)))
    score_now = float(score) if _is_finite(float(score)) else 0.0
    for item in replay_entries:
        if str(item.get("signature", "")) != str(sig):
            continue
        old = _safe_float(item.get("score_ema", 0.0))
        if not _is_finite(old):
            old = 0.0
        new = (1.0 - alpha) * float(old) + alpha * float(score_now)
        item["score_ema"] = float(new)
        item["last_score"] = float(score_now)
        item["n_seen"] = int(item.get("n_seen", 0)) + 1
        item["last_iter"] = int(iter_idx)
        return {
            "signature": str(sig),
            "score_ema": float(new),
            "n_seen": int(item["n_seen"]),
            "event": "update",
        }
    replay_entries.append(
        {
            "signature": str(sig),
            "action": dict(_action_template(action)),
            "score_ema": float(score_now),
            "last_score": float(score_now),
            "n_seen": 1,
            "n_sampled": 0,
            "last_iter": int(iter_idx),
        }
    )
    return {
        "signature": str(sig),
        "score_ema": float(score_now),
        "n_seen": 1,
        "event": "insert",
    }


def _plr_entry_stats(replay_entries: deque, action: Dict[str, object]) -> Tuple[int, float, int]:
    sig = _action_signature(action)
    for item in replay_entries:
        if str(item.get("signature", "")) != str(sig):
            continue
        n_seen = int(max(0, _safe_int(item.get("n_seen", 0), default=0)))
        score_ema = _safe_float(item.get("score_ema", 0.0))
        if not _is_finite(score_ema):
            score_ema = 0.0
        n_sampled = int(max(0, _safe_int(item.get("n_sampled", 0), default=0)))
        return int(n_seen), float(score_ema), int(n_sampled)
    return 0, 0.0, 0


def _update_action_objective(
    actions_csv: Path,
    iter_id: int,
    objective_score: float,
    extra_updates: Optional[Dict[str, object]] = None,
) -> None:
    rows = _read_csv_rows(actions_csv)
    if not rows:
        return
    changed = False
    updates = dict(extra_updates or {})
    updates["objective_score"] = float(objective_score)

    def _format_csv_value(v: object) -> str:
        if v is None:
            return ""
        if isinstance(v, bool):
            return "1" if v else "0"
        if isinstance(v, int):
            return str(int(v))
        if isinstance(v, float):
            if not _is_finite(float(v)):
                return ""
            return f"{float(v):.8f}"
        return str(v)

    for row in rows:
        try:
            rid = int(str(row.get("iter_id", "")).strip())
        except Exception:
            continue
        if rid == int(iter_id):
            for k, v in updates.items():
                row[str(k)] = _format_csv_value(v)
            changed = True
    if not changed:
        return
    fields: List[str] = []
    for row in rows:
        for k in row.keys():
            if str(k) not in fields:
                fields.append(str(k))
    for k in updates.keys():
        ks = str(k)
        if ks not in fields:
            fields.append(ks)
    for row in rows:
        for k in fields:
            if k not in row:
                row[k] = ""
    with actions_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _build_phase2_frozen_stats(actions_rows: List[Dict[str, str]]) -> Tuple[Dict[str, Dict[str, float]], float]:
    stats: Dict[str, Dict[str, float]] = {}
    global_sum = 0.0
    global_count = 0.0
    for row in actions_rows:
        phase = str(row.get("phase", "")).strip().lower()
        if phase != "phase2":
            continue
        j_val = _safe_float(row.get("J", ""))
        if not _is_finite(j_val):
            continue
        try:
            key = (
                f"{int(float(row.get('mu_a', 'nan')))}|"
                f"{int(float(row.get('mu_b', 'nan')))}|"
                f"{float(row.get('p', 'nan')):.8f}|"
                f"{str(row.get('pattern', '')).strip()}"
            )
        except Exception:
            continue
        item = stats.get(key, {"sum": 0.0, "count": 0.0})
        item["sum"] = float(item["sum"]) + float(j_val)
        item["count"] = float(item["count"]) + 1.0
        stats[key] = item
        global_sum += float(j_val)
        global_count += 1.0
    global_mean = (global_sum / global_count) if global_count > 0.0 else float("nan")
    return stats, float(global_mean)


def _update_phase2_frozen_stats(
    stats: Dict[str, Dict[str, float]],
    action: Dict[str, object],
    j_val: float,
) -> None:
    if not _is_finite(float(j_val)):
        return
    key = _phase2_frozen_key(action)
    item = stats.get(key, {"sum": 0.0, "count": 0.0})
    item["sum"] = float(item["sum"]) + float(j_val)
    item["count"] = float(item["count"]) + 1.0
    stats[key] = item


def _lookup_phase2_frozen_mean(
    stats: Dict[str, Dict[str, float]],
    action: Dict[str, object],
    fallback: float,
) -> float:
    key = _phase2_frozen_key(action)
    item = stats.get(key, None)
    if item is not None:
        cnt = float(item.get("count", 0.0))
        if cnt > 0.0:
            return float(item.get("sum", 0.0)) / float(cnt)
    return float(fallback)


def _compute_phase2_topk_action_ids(train_rows: List[Dict[str, str]], k: int) -> List[int]:
    kk = max(0, int(k))
    if kk <= 0:
        return []
    agg: Dict[int, Dict[str, float]] = {}
    for row in train_rows:
        if str(row.get("phase", "")).strip().lower() != "phase2":
            continue
        aid = _safe_int(row.get("policy_action_id", ""), default=-1)
        obj = _safe_float(row.get("objective_score", ""))
        if aid < 0 or (not _is_finite(obj)):
            continue
        item = agg.get(aid, {"sum": 0.0, "count": 0.0})
        item["sum"] = float(item["sum"]) + float(obj)
        item["count"] = float(item["count"]) + 1.0
        agg[aid] = item
    ranked: List[Tuple[int, float, float]] = []
    for aid, item in agg.items():
        cnt = float(item.get("count", 0.0))
        if cnt <= 0.0:
            continue
        mean_obj = float(item.get("sum", 0.0)) / float(cnt)
        ranked.append((int(aid), float(mean_obj), float(cnt)))
    ranked.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return [int(x[0]) for x in ranked[:kk]]


def _compute_phase2_topk_action_ids_from_agg(
    agg: Dict[int, Dict[str, float]],
    k: int,
) -> List[int]:
    kk = max(0, int(k))
    if kk <= 0:
        return []
    ranked: List[Tuple[int, float, float]] = []
    for aid, item in agg.items():
        cnt = float(item.get("count", 0.0))
        if cnt <= 0.0:
            continue
        mean_obj = float(item.get("sum", 0.0)) / float(cnt)
        ranked.append((int(aid), float(mean_obj), float(cnt)))
    ranked.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return [int(x[0]) for x in ranked[:kk]]


def _run_generator_with_retry(
    cmd_base: List[str],
    workers: int,
    retry_max: int,
) -> None:
    attempt = 0
    cur_workers = max(1, int(workers))
    max_attempts = max(1, int(retry_max) + 1)
    while True:
        attempt += 1
        cmd = list(cmd_base)
        cmd.extend(["--workers", str(int(cur_workers))])
        try:
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError as exc:
            if attempt >= max_attempts:
                raise
            next_workers = max(1, int(cur_workers // 2))
            if next_workers == cur_workers and cur_workers > 1:
                next_workers = int(cur_workers - 1)
            if next_workers == cur_workers:
                raise
            print(
                f"[OUTER][GEN][RETRY] attempt={attempt}/{max_attempts} "
                f"workers={cur_workers} failed (code={exc.returncode}), retry workers={next_workers}"
            )
            cur_workers = int(next_workers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--dist-name", type=str, default="O_10_90")
    parser.add_argument("--request-number", type=int, default=30)
    parser.add_argument("--algorithm", type=str, default="PPO_NEW")
    parser.add_argument("--algo-version", type=str, default="v3")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--table-base", type=int, default=400)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--gen-retry-max", type=int, default=2, help="max retries for generation on worker/memory failure")
    parser.add_argument("--base-ckpt", type=str, default="")
    parser.add_argument(
        "--resume-mode",
        type=str,
        default="auto",
        choices=["auto", "none"],
        help="auto: continue from max completed iter in outer_train_round.csv; none: restart from iter=1",
    )
    parser.add_argument("--mu-choices", type=str, default="10,30,60,90")
    parser.add_argument("--ratio-choices", type=str, default="0.2,0.3,0.5,0.7,0.8")
    parser.add_argument("--num-file-choices", type=str, default="5,10,15")
    parser.add_argument("--pattern-choices", type=str, default="ab,random_mix")
    parser.add_argument(
        "--action-space-version",
        type=str,
        default="v1",
        choices=["v1", "v2"],
        help="v1: full (mu_a,mu_b,p,n,pattern); v2: mu-only with fixed p/n/pattern",
    )
    parser.add_argument("--v2-fixed-ratio-a", type=float, default=0.5, help="fixed ratio_a when action-space-version=v2")
    parser.add_argument("--v2-fixed-pattern", type=str, default="ab", help="fixed pattern when action-space-version=v2")
    parser.add_argument(
        "--v2-fixed-num-files",
        type=int,
        default=0,
        help="fixed num_files when action-space-version=v2; 0 means auto-resolve",
    )
    parser.add_argument(
        "--disable-fixed-n-sync",
        action="store_true",
        help="do not auto-sync num-file choices to inner_fixed_n in fixed_n mode",
    )
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--verify-batch", action="store_true")
    parser.add_argument(
        "--policy-mode",
        type=str,
        default="ts",
        choices=["random", "ucb", "pg", "ts", "rarl_dqn"],
        help="outer policy selector: random baseline / UCB / policy-gradient / Thompson sampling / explicit RARL DQN",
    )
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--candidate-pool", type=int, default=24)
    parser.add_argument("--ucb-c", type=float, default=0.35)
    parser.add_argument("--policy-decay", type=float, default=1.00, help="bandit state decay for non-stationarity")
    parser.add_argument("--policy-lr", type=float, default=0.15)
    parser.add_argument("--policy-temp", type=float, default=1.0)
    parser.add_argument("--policy-baseline-momentum", type=float, default=0.9)
    parser.add_argument("--ts-prior-mean", type=float, default=0.0)
    parser.add_argument("--ts-prior-std", type=float, default=0.5)
    parser.add_argument("--ts-obs-std", type=float, default=0.05)
    parser.add_argument("--rarl-k1", type=int, default=2, help="RARL alternation: collect K1 interactions before DQN updates")
    parser.add_argument("--rarl-k2", type=int, default=8, help="RARL alternation: run K2 DQN updates at each update step")
    parser.add_argument("--rarl-gamma", type=float, default=0.95, help="discount factor for RARL DQN")
    parser.add_argument("--rarl-lr", type=float, default=0.02, help="learning rate for RARL DQN")
    parser.add_argument("--rarl-replay-size", type=int, default=4000, help="replay capacity for RARL DQN")
    parser.add_argument("--rarl-batch-size", type=int, default=32, help="mini-batch size for RARL DQN")
    parser.add_argument(
        "--rarl-min-replay",
        type=int,
        default=64,
        help="minimum replay size before running RARL DQN updates",
    )
    parser.add_argument("--rarl-target-sync", type=int, default=50, help="sync target network every N DQN updates")
    parser.add_argument("--rarl-eps-start", type=float, default=0.35, help="initial epsilon for RARL DQN")
    parser.add_argument("--rarl-eps-end", type=float, default=0.05, help="final epsilon for RARL DQN")
    parser.add_argument(
        "--rarl-eps-decay-iters",
        type=int,
        default=120,
        help="linear epsilon decay horizon in outer iterations",
    )
    parser.add_argument(
        "--rarl-state-window",
        type=int,
        default=5,
        help="window size for recent outer metrics used by RARL state encoder",
    )
    parser.add_argument(
        "--rarl-force-objective",
        type=int,
        default=1,
        choices=[0, 1],
        help="force objective-mode=rarl when policy-mode=rarl_dqn to avoid mixed semantics",
    )
    parser.add_argument(
        "--rarl-zero-sum-strict",
        type=int,
        default=1,
        choices=[0, 1],
        help="when objective-mode=rarl, train DQN with strict zero-sum reward (-J) instead of objective score",
    )
    parser.add_argument("--policy-state-path", type=str, default="")
    parser.add_argument("--policy-reset", action="store_true")
    parser.add_argument(
        "--rho-target",
        type=float,
        default=0.22,
        help="target minimum rate for the under-represented action (min(action0_rate, action1_rate))",
    )
    parser.add_argument(
        "--rho-floor",
        type=float,
        default=0.10,
        help="hard floor for the under-represented action rate",
    )
    parser.add_argument(
        "--rho-floor-weight",
        type=float,
        default=4.0,
        help="penalty weight for falling below rho-floor",
    )
    parser.add_argument("--lambda-dj", type=float, default=1.0)
    parser.add_argument("--eta-collapse", type=float, default=1.4)
    parser.add_argument(
        "--objective-mode",
        type=str,
        default="edrl",
        choices=["edrl", "rarl", "plr", "saber_v0", "saber_v1"],
        help="edrl: +w_a*(1-J_frozen)+w_m*G_minority-D (phase2/phase3); rarl: pure adversarial (-J); plr: learning potential (|dJ|); saber_v0: learnability-gated LP+J+novelty; saber_v1: hardest-slice gated score over train trace removal rows",
    )
    parser.add_argument(
        "--edrl-version",
        type=str,
        default="v1",
        choices=["v1", "v3", "v4"],
        help="EDRL objective variant: v1=challenge+minority-feasibility; v3=v1+LP in phase3; v4=PLR/UED-guided curriculum objective",
    )
    parser.add_argument(
        "--phase2-freeze-inner",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="freeze inner model init at theta_phase1 during phase2",
    )
    parser.add_argument(
        "--phase2-difficulty-objective",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use unified difficulty objective for phase2/phase3 when objective-mode=edrl",
    )
    parser.add_argument("--phase2-hard-weight", type=float, default=1.00, help="phase2 w_a: weight for challenge term (1-J_frozen)")
    parser.add_argument(
        "--phase2-drop-weight",
        type=float,
        default=0.25,
        help="legacy no-op (kept for backward compatibility)",
    )
    parser.add_argument(
        "--phase2-proxy-weight",
        type=float,
        default=0.15,
        help="legacy no-op (kept for backward compatibility)",
    )
    parser.add_argument(
        "--phase2-minority-reward-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reward increasing the phase1-minority action rate under phase2 generated data",
    )
    parser.add_argument(
        "--phase2-minority-reward-weight",
        type=float,
        default=1.20,
        help="phase2/phase3 w_m: weight for minority gain G_minority=max(0, rate_t-rate_phase1_minority)",
    )
    parser.add_argument(
        "--edrl-v3-minority-abs-weight",
        type=float,
        default=0.50,
        help="EDRL-v3 extra weight on absolute minority rate to avoid zero-gradient minority reward",
    )
    parser.add_argument(
        "--phase2-too-hard-weight",
        type=float,
        default=0.35,
        help="phase2 w_too: weight for too-hard penalty max(0, J_low - J_frozen)^2",
    )
    parser.add_argument(
        "--phase2-j-ref",
        type=float,
        default=0.90,
        help="legacy no-op (kept for backward compatibility)",
    )
    parser.add_argument(
        "--phase2-j-low",
        type=float,
        default=0.20,
        help="phase2 too-hard floor J_low for anti-collapse penalty",
    )
    parser.add_argument(
        "--plr-level-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable PLR level replay (mix new actions with prioritized replay by objective score)",
    )
    parser.add_argument("--plr-p-new", type=float, default=0.5, help="probability to sample a new action in PLR mode")
    parser.add_argument("--plr-buffer-size", type=int, default=200, help="max replay levels in PLR buffer")
    parser.add_argument(
        "--plr-priority-ema-alpha",
        type=float,
        default=0.3,
        help="EMA alpha for PLR priority update",
    )
    parser.add_argument(
        "--plr-min-weight",
        type=float,
        default=1e-6,
        help="minimum sampling weight for PLR replay levels",
    )
    parser.add_argument(
        "--plr-buffer-path",
        type=str,
        default="",
        help="optional PLR buffer state file path; default: post_stage/outer_policy_plr_buffer.json",
    )
    parser.add_argument("--plr-dj-weight", type=float, default=1.0, help="weight on |dJ| for objective_mode=plr")
    parser.add_argument("--plr-j-weight", type=float, default=0.1, help="small reward tie-breaker for objective_mode=plr")
    parser.add_argument("--saber-v0-dj-weight", type=float, default=0.45, help="V0 weight on |dJ| inside gated score")
    parser.add_argument("--saber-v0-j-weight", type=float, default=0.25, help="V0 weight on J inside gated score")
    parser.add_argument("--saber-v0-novelty-weight", type=float, default=0.15, help="V0 weight on novelty inside gated score")
    parser.add_argument("--saber-v0-j-center", type=float, default=0.55, help="V0 learnability gate center over J_frozen")
    parser.add_argument("--saber-v0-j-sigma", type=float, default=0.18, help="V0 learnability gate sigma over J_frozen")
    parser.add_argument("--saber-v1-hard-threshold", type=int, default=5, help="V1 hard severity threshold over train trace removal rows")
    parser.add_argument("--saber-v1-easy-threshold", type=int, default=3, help="V1 easy severity threshold over train trace removal rows")
    parser.add_argument("--saber-v1-q-weight", type=float, default=0.28, help="V1 weight on Q_hard_rem")
    parser.add_argument("--saber-v1-r-weight", type=float, default=0.18, help="V1 weight on R_hard_rem")
    parser.add_argument("--saber-v1-easy-weight", type=float, default=0.15, help="V1 weight on easy wait preservation")
    parser.add_argument("--saber-v1-dq-weight", type=float, default=0.10, help="V1 weight on |dQ_hard_rem|")
    parser.add_argument("--saber-v1-novelty-weight", type=float, default=0.05, help="V1 weight on novelty")
    parser.add_argument("--saber-v1-no-success-weight", type=float, default=0.08, help="V1 light penalty when hard slice is present but yields no successful hard recovery (R_hard<=0)")
    parser.add_argument("--saber-v1-j-center", type=float, default=0.55, help="V1 learnability gate center over J_frozen")
    parser.add_argument("--saber-v1-j-sigma", type=float, default=0.18, help="V1 learnability gate sigma over J_frozen")
    parser.add_argument("--saber-v1-success-r-threshold", type=float, default=0.01, help="V1 success trigger threshold on R_hard_rem for sticky replay exploitation")
    parser.add_argument("--saber-v1-sticky-replay-iters", type=int, default=3, help="V1 exploitation: force replay the last successful hard candidate for up to this many later phase3 iterations")
    parser.add_argument("--saber-v1-success-budget-mult", type=float, default=2.0, help="V1 exploitation: when sticky replay is active, multiply the candidate n_files / inner budget by this factor")
    parser.add_argument("--saber-v1-success-min-num-files", type=int, default=15, help="V1 exploitation: minimum n_files to use when sticky replay is active")
    parser.add_argument(
        "--edrl-v3-dj-weight",
        type=float,
        default=0.6,
        help="EDRL-v3 learning-potential weight on |dJ| (phase3)",
    )
    parser.add_argument(
        "--edrl-v3-j-weight",
        type=float,
        default=0.1,
        help="EDRL-v3 tie-breaker weight on J (phase3)",
    )
    parser.add_argument(
        "--edrl-v3-level-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable PLR-style level replay in EDRL-v3",
    )
    parser.add_argument(
        "--edrl-v3-replay-phase3-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="if enabled, EDRL-v3 level replay is applied only in phase3",
    )
    parser.add_argument(
        "--edrl-v4-level-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable PLR/UED-style level replay in EDRL-v4",
    )
    parser.add_argument(
        "--edrl-v4-replay-phase3-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="if enabled, EDRL-v4 level replay is applied only in phase3",
    )
    parser.add_argument("--edrl-v4-challenge-weight", type=float, default=0.40, help="V4 weight on learnability-gated challenge term")
    parser.add_argument("--edrl-v4-lp-weight", type=float, default=1.00, help="V4 weight on learning potential |dJ|")
    parser.add_argument("--edrl-v4-j-weight", type=float, default=0.10, help="V4 small tie-breaker weight on J")
    parser.add_argument("--edrl-v4-entropy-weight", type=float, default=0.35, help="V4 weight on policy entropy H(pi)")
    parser.add_argument(
        "--edrl-v4-minority-weight",
        type=float,
        default=1.20,
        help="V4 weight on minority-action gain term",
    )
    parser.add_argument(
        "--edrl-v4-minority-abs-weight",
        type=float,
        default=0.80,
        help="V4 absolute minority-rate bonus weight",
    )
    parser.add_argument("--edrl-v4-novelty-weight", type=float, default=0.20, help="V4 weight on PLR novelty bonus 1/sqrt(1+n_seen)")
    parser.add_argument("--edrl-v4-j-center", type=float, default=0.55, help="V4 UED learnability center on J_frozen")
    parser.add_argument("--edrl-v4-j-sigma", type=float, default=0.20, help="V4 UED learnability bandwidth on J_frozen")
    parser.add_argument("--edrl-v4-p-new-k", type=float, default=0.80, help="V4 adaptive p_new gain by entropy gap")
    parser.add_argument("--edrl-v4-p-new-min", type=float, default=0.20, help="V4 lower clamp of adaptive p_new")
    parser.add_argument("--edrl-v4-p-new-max", type=float, default=0.90, help="V4 upper clamp of adaptive p_new")
    parser.add_argument("--edrl-v4-entropy-target", type=float, default=0.25, help="V4 target policy entropy for adaptive p_new")
    parser.add_argument(
        "--collapse-gap-power",
        type=float,
        default=2.0,
        help="power for collapse-gap penalty max(0, rho_target-rho_min)^p",
    )
    parser.add_argument(
        "--rho-floor-hard-weight",
        type=float,
        default=12.0,
        help="extra quadratic penalty weight for max(0, rho_floor-rho_min)^2",
    )
    parser.add_argument("--path-penalty", type=float, default=2.0)
    parser.add_argument("--validate-path-map", action="store_true")
    parser.add_argument("--min-path-read-rate", type=float, default=0.99)
    parser.add_argument(
        "--inner-stop-mode",
        type=str,
        default="fixed_n",
        choices=["fixed_n", "converge"],
        help="inner train_only stop mode",
    )
    parser.add_argument(
        "--inner-fixed-n",
        type=int,
        default=0,
        help="fixed inner budget N per iter when inner-stop-mode=fixed_n; 0 means use action n_files",
    )
    parser.add_argument(
        "--phase2-fixed-num-files",
        type=int,
        default=5,
        help="force action n_files to this value in phase2 (<=0 disables override)",
    )
    parser.add_argument(
        "--phase3-num-file-choices",
        type=str,
        default="5,10,15",
        help="allowed n_files candidates in phase3; empty means no phase3-only filtering",
    )
    parser.add_argument(
        "--outer-phase",
        type=str,
        default="phase2",
        choices=["phase2", "phase3", "auto"],
        help="phase2=outer optimization only; phase3=S4 joint curriculum; auto=phase2->phase3 by convergence",
    )
    parser.add_argument("--outer-auto-stop", action="store_true", help="enable convergence-based early stop")
    parser.add_argument("--phase2-min-iters", type=int, default=5, help="minimum phase2 iterations before convergence check")
    parser.add_argument("--phase2-max-iters", type=int, default=40, help="hard cap of phase2 iterations in auto mode")
    parser.add_argument("--phase3-min-iters", type=int, default=10, help="minimum phase3 iterations before convergence check")
    parser.add_argument("--phase3-max-iters", type=int, default=50, help="hard cap of phase3 iterations when outer-auto-stop=1")
    parser.add_argument(
        "--phase2-inner-ppo-new-ent-coef",
        type=float,
        default=0.0,
        help="phase2 override for inner PPO_NEW entropy coefficient (RL_PPO_NEW_ENT_COEF)",
    )
    parser.add_argument(
        "--phase3-inner-ppo-new-ent-coef",
        type=float,
        default=0.02,
        help="phase3 override for inner PPO_NEW entropy coefficient (RL_PPO_NEW_ENT_COEF)",
    )
    parser.add_argument("--converge-patience", type=int, default=2, help="phase2 consecutive window length for convergence")
    parser.add_argument("--phase3-converge-patience", type=int, default=1, help="phase3 consecutive window length for convergence")
    parser.add_argument("--converge-max-abs-dj", type=float, default=0.20, help="max abs(dJ) allowed in converge window")
    parser.add_argument("--converge-max-obj-range", type=float, default=0.50, help="max objective range allowed in converge window")
    parser.add_argument(
        "--phase2-converge-max-abs-dj",
        type=float,
        default=0.80,
        help="phase2-only max abs(dJ) allowed in converge window (<=0 uses --converge-max-abs-dj)",
    )
    parser.add_argument(
        "--phase2-converge-max-obj-range",
        type=float,
        default=1.00,
        help="phase2-only max objective range in converge window (<=0 uses --converge-max-obj-range)",
    )
    parser.add_argument(
        "--converge-minority-floor",
        type=float,
        default=0.00,
        help="minimum minority_rate in converge window (EDRL may hard-force its own floor); <0 means use rho-floor for non-EDRL",
    )
    parser.add_argument("--phase3-topk-k", type=int, default=3, help="use phase2 top-k actions as phase3 warm-start priority set")
    parser.add_argument(
        "--phase3-topk-warmup-iters",
        type=int,
        default=6,
        help="phase3 warm-start iterations sampled from phase2 top-k (0 disables)",
    )
    parser.add_argument(
        "--phase3-topk-prior-count",
        type=float,
        default=3.0,
        help="pseudo-count used when bootstrapping TS/UCB with phase2 top-k priors",
    )
    parser.add_argument("--curriculum-enable", action="store_true", help="enable mixed curriculum dataset build")
    parser.add_argument("--curriculum-base-root", type=str, default="", help="base dataset root for curriculum mix")
    parser.add_argument("--curriculum-alpha-start", type=float, default=0.70, help="outer ratio at phase3 start")
    parser.add_argument("--curriculum-alpha-end", type=float, default=0.35, help="outer ratio at phase3 end")
    parser.add_argument(
        "--curriculum-alpha-horizon",
        type=int,
        default=25,
        help="alpha schedule horizon in iters (0 means use total iterations)",
    )
    parser.add_argument("--curriculum-replay-ratio", type=float, default=0.2, help="replay sampling ratio inside outer bucket")
    parser.add_argument("--curriculum-replay-max-iters", type=int, default=5, help="max prior outer iters used as replay pool")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(int(args.seed))
    run_root = Path(args.run_id)
    if not run_root.is_absolute():
        run_root = (CODES_DIR / "logs" / str(args.run_id)).resolve()
    post_stage_dir = run_root / "post_stage"
    out_root = post_stage_dir / "outer_batches"
    curriculum_root = post_stage_dir / "outer_curriculum"
    ckpt_dir = post_stage_dir / "checkpoints"
    out_root.mkdir(parents=True, exist_ok=True)
    curriculum_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    mu_choices = _parse_int_list(args.mu_choices)
    ratio_choices = _parse_float_list(args.ratio_choices)
    num_file_choices = _parse_int_list(args.num_file_choices)
    phase3_num_file_choices = _parse_int_list(args.phase3_num_file_choices)
    pattern_choices = _parse_str_list(args.pattern_choices)
    min_files, max_files = resolve_num_file_bounds()
    invalid_n_choices = [int(n) for n in num_file_choices if int(n) < int(min_files) or int(n) > int(max_files)]
    if invalid_n_choices:
        raise ValueError(
            f"invalid num-file-choices={invalid_n_choices}, allowed range is "
            f"[{int(min_files)}, {int(max_files)}] (set by OUTER_BATCH_MIN_FILES/OUTER_BATCH_MAX_FILES)"
        )
    if not phase3_num_file_choices:
        phase3_num_file_choices = list(num_file_choices)
    invalid_phase3_n = [
        int(n) for n in phase3_num_file_choices if int(n) < int(min_files) or int(n) > int(max_files)
    ]
    if invalid_phase3_n:
        raise ValueError(
            f"invalid phase3-num-file-choices={invalid_phase3_n}, allowed range is "
            f"[{int(min_files)}, {int(max_files)}]"
        )
    phase2_fixed_num_files = int(args.phase2_fixed_num_files)
    if phase2_fixed_num_files > 0 and (
        phase2_fixed_num_files < int(min_files) or phase2_fixed_num_files > int(max_files)
    ):
        raise ValueError(
            f"phase2-fixed-num-files={phase2_fixed_num_files} out of range "
            f"[{int(min_files)}, {int(max_files)}]"
        )
    # Hard disable phase3 top-k warmstart.
    # Keep CLI flags for backward compatibility, but ignore them in runtime.
    phase3_topk_k = 0
    phase3_topk_warmup_iters = 0
    phase3_topk_prior_count = 0.0
    action_num_file_choices: List[int] = []
    for n in list(num_file_choices) + list(phase3_num_file_choices):
        nn = int(n)
        if nn not in action_num_file_choices:
            action_num_file_choices.append(nn)
    if phase2_fixed_num_files > 0 and int(phase2_fixed_num_files) not in action_num_file_choices:
        action_num_file_choices.append(int(phase2_fixed_num_files))
    if (
        str(args.inner_stop_mode) == "fixed_n"
        and int(args.inner_fixed_n) > 0
        and not bool(args.disable_fixed_n_sync)
    ):
        fixed_n = int(args.inner_fixed_n)
        if fixed_n < int(min_files):
            print(
                f"[OUTER][INIT][WARN] inner_fixed_n={fixed_n} is below generator lower bound ({int(min_files)}); "
                "skip num_file sync."
            )
        elif fixed_n > int(max_files):
            raise ValueError(
                f"inner_fixed_n={fixed_n} exceeds generator upper bound ({int(max_files)}); "
                "adjust inner_fixed_n or OUTER_BATCH_MAX_FILES."
            )
        else:
            if action_num_file_choices != [fixed_n]:
                print(
                    f"[OUTER][INIT] sync num_file_choices/phase3_num_file_choices -> [{fixed_n}] "
                    f"(inner_stop_mode=fixed_n, inner_fixed_n={fixed_n})"
                )
            action_num_file_choices = [fixed_n]
            phase3_num_file_choices = [fixed_n]
            phase2_fixed_num_files = int(fixed_n)

    action_space_version = str(getattr(args, "action_space_version", "v1")).strip().lower()
    if action_space_version not in {"v1", "v2"}:
        raise ValueError(f"unsupported action-space-version: {action_space_version}")

    if not mu_choices:
        raise ValueError("mu-choices must be non-empty")
    if not (ratio_choices and action_num_file_choices and pattern_choices):
        raise ValueError("ratio-choices/num-file-choices/pattern-choices must be non-empty")

    v2_fixed_ratio_a = float("nan")
    v2_fixed_num_files = -1
    v2_fixed_pattern = ""
    if action_space_version == "v2":
        v2_fixed_ratio_a = float(args.v2_fixed_ratio_a)
        if not (0.0 < float(v2_fixed_ratio_a) < 1.0):
            raise ValueError(f"v2-fixed-ratio-a must be in (0,1), got {v2_fixed_ratio_a}")
        raw_pattern = str(args.v2_fixed_pattern).strip()
        if not raw_pattern:
            raw_pattern = str(pattern_choices[0])
        if raw_pattern not in set(pattern_choices):
            print(
                f"[OUTER][INIT][WARN] v2-fixed-pattern={raw_pattern} not in pattern_choices={pattern_choices}; "
                f"use pattern_choices[0]={pattern_choices[0]}"
            )
            raw_pattern = str(pattern_choices[0])
        v2_fixed_pattern = str(raw_pattern)
        if int(args.v2_fixed_num_files) > 0:
            v2_fixed_num_files = int(args.v2_fixed_num_files)
        elif str(args.inner_stop_mode) == "fixed_n" and int(args.inner_fixed_n) > 0:
            v2_fixed_num_files = int(args.inner_fixed_n)
        else:
            v2_fixed_num_files = int(action_num_file_choices[0])
        if v2_fixed_num_files < int(min_files) or v2_fixed_num_files > int(max_files):
            raise ValueError(
                f"v2 fixed num_files={v2_fixed_num_files} out of range "
                f"[{int(min_files)}, {int(max_files)}]"
            )
        v2_num_file_choices = [int(v2_fixed_num_files)] if int(args.v2_fixed_num_files) > 0 else list(action_num_file_choices)
        action_space = _build_action_space_v2_mu_only(
            mu_choices=mu_choices,
            fixed_ratio_a=float(v2_fixed_ratio_a),
            num_file_choices=v2_num_file_choices,
            fixed_pattern=str(v2_fixed_pattern),
        )
    else:
        action_space = _build_action_space(mu_choices, ratio_choices, action_num_file_choices, pattern_choices)
    if not action_space:
        raise ValueError("empty action space")
    action_to_idx = {_action_signature(a): i for i, a in enumerate(action_space)}

    training_csv = run_root / "rl_training.csv"
    decision_csv = run_root / "rl_decision.csv"
    trace_csv = run_root / "rl_trace.csv"
    path_map_csv = post_stage_dir / "outer_path_map.csv"
    generation_csv = post_stage_dir / "outer_generation.csv"
    curriculum_csv = post_stage_dir / "outer_curriculum.csv"
    actions_csv = post_stage_dir / "outer_actions.csv"
    train_round_csv = post_stage_dir / "outer_train_round.csv"
    plr_stats_csv = post_stage_dir / "outer_plr_stats.csv"
    rarl_stats_csv = post_stage_dir / "outer_rarl_stats.csv"
    ckpt_prev = Path(str(args.base_ckpt)).resolve() if str(args.base_ckpt).strip() else None
    requested_phase = str(args.outer_phase).strip().lower()
    auto_phase = requested_phase == "auto"
    phase_mode = "phase2" if auto_phase else requested_phase
    outer_auto_stop = bool(args.outer_auto_stop) or auto_phase
    curriculum_flag = bool(args.curriculum_enable)
    # Hard rule: phase2 never uses curriculum. Curriculum is required only when
    # phase3 can run (phase3 or auto).
    curriculum_required = requested_phase in {"phase3", "auto"}
    alpha_horizon = int(args.curriculum_alpha_horizon) if int(args.curriculum_alpha_horizon) > 0 else int(args.iterations)
    replay_history: deque[Path] = deque(maxlen=max(0, int(args.curriculum_replay_max_iters)))
    base_root: Optional[Path] = None
    if curriculum_required:
        base_root_raw = str(args.curriculum_base_root or "").strip()
        if not base_root_raw:
            raise ValueError("curriculum is required but --curriculum-base-root is empty")
        base_root = Path(base_root_raw).resolve()
        if not base_root.exists():
            raise FileNotFoundError(f"curriculum base root not found: {base_root}")

    start_iter = 1
    completed_max_iter = -1
    if str(args.resume_mode) == "auto" and (not bool(args.policy_reset)):
        completed_max_iter = _get_max_iter_id(train_round_csv)
        if completed_max_iter >= 0:
            start_iter = int(completed_max_iter + 1)
            if start_iter > int(args.iterations):
                print(
                    f"[OUTER][INIT] resume_mode=auto found completed iter={completed_max_iter}; "
                    f"target iterations={int(args.iterations)} already satisfied, nothing to run."
                )
                return
            # Drop stale partial rows (e.g., generator wrote iter K but train_round failed before append).
            _drop_rows_ge_iter(actions_csv, start_iter)
            _drop_rows_ge_iter(generation_csv, start_iter)
            _drop_rows_ge_iter(curriculum_csv, start_iter)
            _drop_rows_ge_iter(path_map_csv, start_iter)
            _drop_rows_ge_iter(plr_stats_csv, start_iter)
            _drop_rows_ge_iter(rarl_stats_csv, start_iter)
    elif str(args.resume_mode) == "auto" and bool(args.policy_reset):
        print("[OUTER][INIT] policy_reset=1, disable auto resume and restart from iter=1")
    if str(args.resume_mode) == "none" and train_round_csv.exists():
        print(
            "[OUTER][INIT][WARN] resume_mode=none with existing outer_train_round.csv; "
            "iter_id rows will append from 1 and may duplicate prior runs."
        )

    print(f"[OUTER][INIT] run_root={run_root}")
    print(f"[OUTER][INIT] out_root={out_root}")
    print(
        f"[OUTER][INIT] resume_mode={args.resume_mode} start_iter={start_iter} "
        f"target_iter={int(args.iterations)}"
    )
    print(
        f"[OUTER][INIT] phase_request={requested_phase} phase_start={phase_mode} "
        f"auto_stop={int(outer_auto_stop)} curriculum_flag={int(curriculum_flag)} "
        f"curriculum_root={curriculum_root}"
    )
    if curriculum_required and base_root is not None:
        print(
            f"[OUTER][INIT] curriculum_base_root={base_root} "
            f"alpha_start={float(args.curriculum_alpha_start):.4f} "
            f"alpha_end={float(args.curriculum_alpha_end):.4f} horizon={alpha_horizon} "
            f"replay_ratio={float(args.curriculum_replay_ratio):.4f}"
        )
    print(
        f"[OUTER][INIT] converge: "
        f"phase2[p={int(args.converge_patience)},"
        f"max_abs_dj={float(args.phase2_converge_max_abs_dj):.4f},"
        f"max_obj_range={float(args.phase2_converge_max_obj_range):.4f}] "
        f"phase3[p={int(args.phase3_converge_patience)},"
        f"max_abs_dj={float(args.converge_max_abs_dj):.4f},"
        f"max_obj_range={float(args.converge_max_obj_range):.4f}] "
        f"phase2[min={int(args.phase2_min_iters)},max={int(args.phase2_max_iters)}] "
        f"phase3[min={int(args.phase3_min_iters)},max={int(args.phase3_max_iters)}]"
    )
    print(
        f"[OUTER][INIT] policy_mode={args.policy_mode} warmup={int(args.warmup_iters)} "
        f"candidate_pool={int(args.candidate_pool)} ucb_c={float(args.ucb_c):.4f} "
        f"decay={float(args.policy_decay):.4f}"
    )
    objective_mode = str(args.objective_mode).strip().lower()
    edrl_version = str(args.edrl_version).strip().lower()
    policy_mode = str(args.policy_mode).strip().lower()
    saber_v0_mode = objective_mode == "saber_v0"
    saber_v1_mode = objective_mode == "saber_v1"
    if policy_mode == "rarl_dqn" and objective_mode != "rarl":
        if int(args.rarl_force_objective) == 1:
            print(
                f"[OUTER][INIT][RARL] force objective_mode: {objective_mode} -> rarl "
                "(--rarl-force-objective=1)"
            )
            objective_mode = "rarl"
        else:
            print(
                f"[OUTER][INIT][WARN] policy_mode=rarl_dqn with objective_mode={objective_mode}; "
                "DQN reward will use current objective_score."
            )
    edrl_v3_mode = objective_mode == "edrl" and edrl_version == "v3"
    edrl_v4_mode = objective_mode == "edrl" and edrl_version == "v4"
    plr_replay_enabled = bool(args.plr_level_replay) and objective_mode in {"plr", "saber_v0", "saber_v1"}
    edrl_v3_replay_enabled = bool(args.edrl_v3_level_replay) and edrl_v3_mode
    edrl_v4_replay_enabled = bool(args.edrl_v4_level_replay) and edrl_v4_mode
    level_replay_enabled = bool(plr_replay_enabled or edrl_v3_replay_enabled or edrl_v4_replay_enabled)
    phase2_freeze_inner = bool(args.phase2_freeze_inner)
    phase2_difficulty_objective = bool(args.phase2_difficulty_objective) and objective_mode == "edrl"
    phase2_anchor_ckpt = ckpt_prev if (ckpt_prev is not None and ckpt_prev.exists()) else None
    if phase2_freeze_inner and phase2_anchor_ckpt is None:
        print(
            "[OUTER][INIT][WARN] phase2 freeze enabled but base checkpoint is missing; "
            "if phase2 runs, execution will fail. pass --base-ckpt or use --no-phase2-freeze-inner."
        )
    phase2_j_low = float(args.phase2_j_low)
    phase2_minority_reward_enable = bool(args.phase2_minority_reward_enable)
    phase2_minority_reward_weight = max(0.0, float(args.phase2_minority_reward_weight))
    edrl_v3_minority_abs_weight = max(0.0, float(args.edrl_v3_minority_abs_weight))
    phase2_baseline_path = post_stage_dir / "phase2_action_baseline.json"
    phase1_action0_base = 0.5
    phase1_action1_base = 0.5
    phase1_minority_action = "0"
    phase1_minority_rate = 0.5
    baseline_loaded = False
    if phase2_difficulty_objective and phase2_minority_reward_enable:
        if phase2_baseline_path.exists():
            try:
                with phase2_baseline_path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                a0 = float(payload.get("action0_rate", float("nan")))
                a1 = float(payload.get("action1_rate", float("nan")))
                ma = str(payload.get("minority_action", "")).strip()
                mr = float(payload.get("minority_rate", float("nan")))
                if _is_finite(a0) and _is_finite(a1) and ma in {"0", "1"} and _is_finite(mr):
                    phase1_action0_base = float(a0)
                    phase1_action1_base = float(a1)
                    phase1_minority_action = str(ma)
                    phase1_minority_rate = float(max(0.0, min(1.0, mr)))
                    baseline_loaded = True
            except Exception:
                baseline_loaded = False
        if not baseline_loaded:
            historical_rows = _read_csv_rows(decision_csv)
            _, a0, a1, mr, ma = _calc_metrics(historical_rows)
            if _is_finite(a0) and _is_finite(a1) and str(ma) in {"0", "1"} and _is_finite(mr):
                phase1_action0_base = float(a0)
                phase1_action1_base = float(a1)
                phase1_minority_action = str(ma)
                phase1_minority_rate = float(max(0.0, min(1.0, mr)))
            try:
                phase2_baseline_path.parent.mkdir(parents=True, exist_ok=True)
                with phase2_baseline_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "action0_rate": float(phase1_action0_base),
                            "action1_rate": float(phase1_action1_base),
                            "minority_action": str(phase1_minority_action),
                            "minority_rate": float(phase1_minority_rate),
                            "source_rows": int(len(historical_rows)),
                            "saved_at_ts": int(time.time()),
                        },
                        f,
                        ensure_ascii=True,
                        indent=2,
                    )
            except Exception:
                pass
    plr_p_new = min(1.0, max(0.0, float(args.plr_p_new)))
    plr_priority_ema_alpha = min(1.0, max(0.0, float(args.plr_priority_ema_alpha)))
    plr_min_weight = max(1e-12, float(args.plr_min_weight))
    plr_buffer_size = max(1, int(args.plr_buffer_size))
    _buffer_default_name = "outer_policy_plr_buffer.json"
    if saber_v0_mode and (not edrl_v3_replay_enabled) and (not edrl_v4_replay_enabled):
        _buffer_default_name = "outer_policy_saber_v0_buffer.json"
    if saber_v1_mode and (not edrl_v3_replay_enabled) and (not edrl_v4_replay_enabled):
        _buffer_default_name = "outer_policy_saber_v1_buffer.json"
    if edrl_v3_replay_enabled and (not plr_replay_enabled):
        _buffer_default_name = "outer_policy_edrl_v3_buffer.json"
    if edrl_v4_replay_enabled and (not plr_replay_enabled) and (not edrl_v3_replay_enabled):
        _buffer_default_name = "outer_policy_edrl_v4_buffer.json"
    plr_buffer_path = (
        Path(str(args.plr_buffer_path)).resolve()
        if str(args.plr_buffer_path).strip()
        else post_stage_dir / _buffer_default_name
    )
    plr_buffer = _load_or_init_plr_buffer(
        buffer_path=plr_buffer_path,
        action_space=action_space,
        maxlen=int(plr_buffer_size),
        reset=bool(args.policy_reset),
    ) if level_replay_enabled else deque(maxlen=int(plr_buffer_size))
    plr_total_samples = 0
    plr_new_samples = 0
    plr_replay_samples = 0
    plr_recent_sources: deque = deque(maxlen=20)
    if level_replay_enabled and str(args.resume_mode) == "auto" and (not bool(args.policy_reset)):
        prev_plr_rows = _read_csv_rows(plr_stats_csv)
        if prev_plr_rows:
            last = prev_plr_rows[-1]
            plr_total_samples = max(0, _safe_int(last.get("total_samples", 0), default=0))
            plr_new_samples = max(0, _safe_int(last.get("new_samples", 0), default=0))
            plr_replay_samples = max(0, _safe_int(last.get("replay_samples", 0), default=0))
            print(
                f"[OUTER][INIT] resume plr_stats total={plr_total_samples} "
                f"new={plr_new_samples} replay={plr_replay_samples}"
            )
    if str(args.policy_mode) == "pg":
        print(
            f"[OUTER][INIT] pg lr={float(args.policy_lr):.4f} "
            f"temp={float(args.policy_temp):.4f} "
            f"baseline_m={float(args.policy_baseline_momentum):.4f}"
        )
    if str(args.policy_mode) == "ts":
        print(
            f"[OUTER][INIT] ts prior_mean={float(args.ts_prior_mean):.4f} "
            f"prior_std={float(args.ts_prior_std):.4f} obs_std={float(args.ts_obs_std):.4f}"
        )
    if str(args.policy_mode) == "rarl_dqn":
        print(
            f"[OUTER][INIT] rarl_dqn "
            f"K1={int(args.rarl_k1)} K2={int(args.rarl_k2)} "
            f"gamma={float(args.rarl_gamma):.4f} lr={float(args.rarl_lr):.5f} "
            f"replay={int(args.rarl_replay_size)} batch={int(args.rarl_batch_size)} "
            f"min_replay={int(args.rarl_min_replay)} state_window={int(args.rarl_state_window)} "
            f"target_sync={int(args.rarl_target_sync)} "
            f"eps=[{float(args.rarl_eps_start):.3f}->{float(args.rarl_eps_end):.3f}] "
            f"decay_iters={int(args.rarl_eps_decay_iters)} "
            f"zero_sum_strict={int(args.rarl_zero_sum_strict)}"
        )
    if objective_mode == "rarl":
        print("[OUTER][INIT] objective_mode=rarl formula: -J - path_penalty*D")
    elif objective_mode == "plr":
        print(
            f"[OUTER][INIT] objective_mode=plr formula: "
            f"+plr_dj_weight*|dJ| + plr_j_weight*J - path_penalty*D "
            f"(plr_dj_weight={float(args.plr_dj_weight):.4f},plr_j_weight={float(args.plr_j_weight):.4f})"
        )
        print(
            f"[OUTER][INIT] plr_replay={int(plr_replay_enabled)} "
            f"p_new={float(plr_p_new):.4f} buffer_size={int(plr_buffer_size)} "
            f"ema_alpha={float(plr_priority_ema_alpha):.4f} "
            f"min_weight={float(plr_min_weight):.6f} buffer={plr_buffer_path} "
            f"loaded_levels={int(len(plr_buffer))}"
        )
    elif saber_v0_mode:
        print(
            f"[OUTER][INIT] objective_mode=saber_v0 formula: "
            f"L(J_frozen) * (w_dj*|dJ| + w_j*J + w_n*novelty) - D "
            f"(w_dj={float(args.saber_v0_dj_weight):.4f},"
            f"w_j={float(args.saber_v0_j_weight):.4f},"
            f"w_n={float(args.saber_v0_novelty_weight):.4f},"
            f"j_center={float(args.saber_v0_j_center):.4f},"
            f"j_sigma={float(args.saber_v0_j_sigma):.4f})"
        )
        print(
            f"[OUTER][INIT] saber_v0_replay={int(plr_replay_enabled)} "
            f"p_new={float(plr_p_new):.4f} buffer_size={int(plr_buffer_size)} "
            f"ema_alpha={float(plr_priority_ema_alpha):.4f} "
            f"min_weight={float(plr_min_weight):.6f} buffer={plr_buffer_path} "
            f"loaded_levels={int(len(plr_buffer))}"
        )
    elif saber_v1_mode:
        print(
            f"[OUTER][INIT] objective_mode=saber_v1 formula: "
            f"I(hard_count>0) * L(J_frozen) * (w_q*Q_hard_rem + w_r*R_hard_rem + "
            f"w_e*P_easy_wait + w_dq*|dQ_hard_rem| + w_n*novelty) - D - "
            f"w_ns*I(R_hard<=0)*wait_hard "
            f"(w_q={float(args.saber_v1_q_weight):.4f},"
            f"w_r={float(args.saber_v1_r_weight):.4f},"
            f"w_e={float(args.saber_v1_easy_weight):.4f},"
            f"w_dq={float(args.saber_v1_dq_weight):.4f},"
            f"w_n={float(args.saber_v1_novelty_weight):.4f},"
            f"w_ns={float(args.saber_v1_no_success_weight):.4f},"
            f"hard>={int(args.saber_v1_hard_threshold)},"
            f"easy<={int(args.saber_v1_easy_threshold)},"
            f"j_center={float(args.saber_v1_j_center):.4f},"
            f"j_sigma={float(args.saber_v1_j_sigma):.4f})"
        )
        print(
            f"[OUTER][INIT] saber_v1_replay={int(plr_replay_enabled)} "
            f"p_new={float(plr_p_new):.4f} buffer_size={int(plr_buffer_size)} "
            f"ema_alpha={float(plr_priority_ema_alpha):.4f} "
            f"min_weight={float(plr_min_weight):.6f} buffer={plr_buffer_path} "
            f"loaded_levels={int(len(plr_buffer))} "
            f"success_r>={float(args.saber_v1_success_r_threshold):.4f} "
            f"sticky_iters={int(args.saber_v1_sticky_replay_iters)} "
            f"budget_mult={float(args.saber_v1_success_budget_mult):.2f} "
            f"min_n={int(args.saber_v1_success_min_num_files)}"
        )
    else:
        if edrl_v3_mode:
            print(
                "[OUTER][INIT] objective_mode=edrl(v3) formula: "
                "phase2: +w_a*(1-J_frozen)+w_m*G_minority-D; "
                "phase3: phase2 + w_dj*|dJ| + w_j*J"
            )
        elif edrl_v4_mode:
            print(
                "[OUTER][INIT] objective_mode=edrl(v4, PLR/UED-driven) formula: "
                "+w_c*(1-J_frozen)*L(J_frozen) + w_lp*|dJ| + w_H*H(pi) + "
                "w_m*G_minority + w_n*novelty + w_j*J - D"
            )
        else:
            print(
                "[OUTER][INIT] objective_mode=edrl(v1) formula: "
                "+w_a*(1-J_frozen) + w_m*G_minority - D "
                "(phase2/phase3 unified)"
            )
        if edrl_v4_mode:
            print(
                f"[OUTER][INIT] phase2 freeze_inner={int(phase2_freeze_inner)} "
                f"anchor_ckpt={phase2_anchor_ckpt} "
                f"difficulty_objective={int(phase2_difficulty_objective)} "
                f"(phase2/phase3 obj = V4 PLR/UED-driven; D=path_penalty*missing_ratio)"
            )
        else:
            print(
                f"[OUTER][INIT] phase2 freeze_inner={int(phase2_freeze_inner)} "
                f"anchor_ckpt={phase2_anchor_ckpt} "
                f"difficulty_objective={int(phase2_difficulty_objective)} "
                f"(phase2/phase3 obj = w_a*(1-J_frozen) + w_m*G_minority - D; "
                f"w_a={float(args.phase2_hard_weight):.3f},"
                f"w_m={float(phase2_minority_reward_weight):.3f},"
                f"w_too=0.000,"
                f"J_low(disabled)={float(args.phase2_j_low):.4f})"
            )
        if edrl_v3_mode:
            print(
                f"[OUTER][INIT] edrl_v3 extras "
                f"(w_dj={float(args.edrl_v3_dj_weight):.3f},"
                f"w_j={float(args.edrl_v3_j_weight):.3f},"
                f"minor_abs_w={float(edrl_v3_minority_abs_weight):.3f},"
                f"replay={int(edrl_v3_replay_enabled)},"
                f"replay_phase3_only={int(bool(args.edrl_v3_replay_phase3_only))},"
                f"replay_p_new={float(plr_p_new):.3f},"
                f"replay_buffer={int(plr_buffer_size)})"
            )
            if edrl_v3_replay_enabled:
                print(
                    f"[OUTER][INIT] edrl_v3 replay buffer={plr_buffer_path} "
                    f"loaded_levels={int(len(plr_buffer))}"
                )
        if edrl_v4_mode:
            print(
                f"[OUTER][INIT] edrl_v4 weights "
                f"(w_c={float(args.edrl_v4_challenge_weight):.3f},"
                f"w_lp={float(args.edrl_v4_lp_weight):.3f},"
                f"w_j={float(args.edrl_v4_j_weight):.3f},"
                f"w_H={float(args.edrl_v4_entropy_weight):.3f},"
                f"w_m={float(args.edrl_v4_minority_weight):.3f},"
                f"w_m_abs={float(args.edrl_v4_minority_abs_weight):.3f},"
                f"w_n={float(args.edrl_v4_novelty_weight):.3f},"
                f"J_center={float(args.edrl_v4_j_center):.3f},"
                f"J_sigma={float(args.edrl_v4_j_sigma):.3f})"
            )
            print(
                f"[OUTER][INIT] edrl_v4 replay "
                f"(enabled={int(edrl_v4_replay_enabled)},"
                f"phase3_only={int(bool(args.edrl_v4_replay_phase3_only))},"
                f"p_new_base={float(plr_p_new):.3f},"
                f"p_new_range=[{float(args.edrl_v4_p_new_min):.3f},{float(args.edrl_v4_p_new_max):.3f}],"
                f"k={float(args.edrl_v4_p_new_k):.3f},"
                f"H_target={float(args.edrl_v4_entropy_target):.3f},"
                f"buffer={int(plr_buffer_size)})"
            )
            if edrl_v4_replay_enabled:
                print(
                    f"[OUTER][INIT] edrl_v4 replay buffer={plr_buffer_path} "
                    f"loaded_levels={int(len(plr_buffer))}"
                )
        if phase2_minority_reward_enable:
            print(
                f"[OUTER][INIT] phase2 minority baseline "
                f"(a0={phase1_action0_base:.6f},a1={phase1_action1_base:.6f},"
                f"minority={phase1_minority_action}:{phase1_minority_rate:.6f}) "
                f"path={phase2_baseline_path}"
            )
        if (
            abs(float(args.phase2_drop_weight) - 0.25) > 1e-12
            or abs(float(args.phase2_proxy_weight) - 0.15) > 1e-12
            or abs(float(args.phase2_j_ref) - 0.90) > 1e-12
        ):
            print(
                "[OUTER][INIT][INFO] phase2_drop_weight/phase2_proxy_weight/phase2_j_ref are legacy no-op "
                "parameters (kept only for backward compatibility)."
            )
    if action_space_version == "v2":
        print(
            f"[OUTER][INIT] action_space=v2(mu-only) mu={mu_choices} "
            f"fixed(ratio={float(v2_fixed_ratio_a):.4f},pattern={v2_fixed_pattern}) "
            f"n_choices={action_num_file_choices} "
            f"n_bounds=[{int(min_files)},{int(max_files)}] size={len(action_space)}"
        )
    else:
        print(
            f"[OUTER][INIT] action_space=v1(full) mu={mu_choices} ratio={ratio_choices} "
            f"n={action_num_file_choices} (bounds=[{int(min_files)},{int(max_files)}]) "
            f"pattern={pattern_choices} size={len(action_space)}"
        )
    print(
        f"[OUTER][INIT] phase_n_policy phase2_fixed_n={int(phase2_fixed_num_files)} "
        f"phase3_n_choices={phase3_num_file_choices}"
    )
    print("[OUTER][INIT] phase3_topk warmstart disabled (k=0,warm_iters=0,prior_count=0.0)")
    print(
        f"[OUTER][INIT] action-seed policy: deterministic seed per iter "
        f"(seed = f(base_seed={int(args.seed)}, iter_id)), not part of action space"
    )
    if curriculum_flag and requested_phase == "phase2":
        print(
            "[OUTER][INIT][WARN] --curriculum-enable is ignored in phase2 "
            "(hard rule: curriculum is phase3-only)."
        )
    print(
        f"[OUTER][INIT] inner_stop_mode={args.inner_stop_mode} "
        f"inner_fixed_n={int(args.inner_fixed_n)}"
    )
    learn_n_mode = (str(args.inner_stop_mode) == "fixed_n") and (int(args.inner_fixed_n) <= 0)
    if learn_n_mode:
        print(
            f"[OUTER][INIT] learn_n_mode=1 (inner budget follows outer action n_files, "
            f"choices={action_num_file_choices})"
        )

    prev_avg_reward = float("nan")
    pg_state: Dict[str, object] = {}
    ucb_state: Dict[str, object] = {}
    ts_state: Dict[str, object] = {}
    rarl_dqn_state: Dict[str, object] = {}
    pg_state_path: Optional[Path] = None
    ucb_state_path: Optional[Path] = None
    ts_state_path: Optional[Path] = None
    rarl_dqn_state_path: Optional[Path] = None
    policy_last_j = float("nan")
    policy_last_dj = 0.0
    policy_last_action0_rate = float("nan")
    policy_last_action1_rate = float("nan")
    policy_last_minority_rate = float("nan")
    policy_last_objective = 0.0
    policy_last_entropy = float("nan")
    policy_last_saber_v1_q_hard = float("nan")
    rarl_window = max(1, int(args.rarl_state_window))
    rarl_recent_j: deque = deque(maxlen=int(rarl_window))
    rarl_recent_action1: deque = deque(maxlen=int(rarl_window))
    rarl_recent_entropy: deque = deque(maxlen=int(rarl_window))
    rarl_state_dim = 12
    state_path_raw = str(args.policy_state_path).strip()
    if policy_mode == "pg":
        pg_state_path = Path(state_path_raw).resolve() if state_path_raw else post_stage_dir / "outer_policy_pg.json"
        pg_state = _load_or_init_pg_state(
            policy_state_path=pg_state_path,
            action_space=action_space,
            reset=bool(args.policy_reset),
        )
        print(f"[OUTER][INIT] pg state={pg_state_path}")
    elif policy_mode == "ucb":
        ucb_state_path = Path(state_path_raw).resolve() if state_path_raw else post_stage_dir / "outer_policy_ucb.json"
        ucb_state = _load_or_init_ucb_state(
            policy_state_path=ucb_state_path,
            action_space=action_space,
            reset=bool(args.policy_reset),
        )
        print(f"[OUTER][INIT] ucb state={ucb_state_path}")
    elif policy_mode == "ts":
        ts_state_path = Path(state_path_raw).resolve() if state_path_raw else post_stage_dir / "outer_policy_ts.json"
        ts_state = _load_or_init_ts_state(
            policy_state_path=ts_state_path,
            action_space=action_space,
            reset=bool(args.policy_reset),
        )
        print(f"[OUTER][INIT] ts state={ts_state_path}")
    elif policy_mode == "rarl_dqn":
        rarl_dqn_state_path = (
            Path(state_path_raw).resolve() if state_path_raw else post_stage_dir / "outer_policy_rarl_dqn.json"
        )
        rarl_dqn_state = _load_or_init_rarl_dqn_state(
            policy_state_path=rarl_dqn_state_path,
            action_space=action_space,
            reset=bool(args.policy_reset),
            replay_capacity=int(args.rarl_replay_size),
            state_dim=int(rarl_state_dim),
        )
        _rarl_sync_target(rarl_dqn_state)
        print(
            f"[OUTER][INIT] rarl_dqn state={rarl_dqn_state_path} "
            f"steps={int(rarl_dqn_state.get('steps', 0))} "
            f"updates={int(rarl_dqn_state.get('train_updates', 0))} "
            f"replay={len(list(rarl_dqn_state.get('replay', [])))}"
        )

    if objective_mode == "edrl":
        converge_minority_floor = float(FORCED_EDRL_CONVERGE_MINORITY_FLOOR)
        if abs(float(args.converge_minority_floor) - float(converge_minority_floor)) > 1e-12:
            print(
                f"[OUTER][INIT] force converge_minority_floor for EDRL: "
                f"{float(args.converge_minority_floor):.4f} -> {float(converge_minority_floor):.4f}"
            )
    else:
        converge_minority_floor = (
            float(args.converge_minority_floor)
            if float(args.converge_minority_floor) >= 0.0
            else float(args.rho_floor)
        )
    print(
        f"[OUTER][INIT] converge_minority_floor_effective={float(converge_minority_floor):.4f}"
    )
    phase_history: Dict[str, List[Dict[str, float]]] = {"phase2": [], "phase3": []}
    phase_iter_count: Dict[str, int] = {"phase2": 0, "phase3": 0}
    train_rows = _read_csv_rows(train_round_csv)
    actions_rows_hist = _read_csv_rows(actions_csv)
    phase2_frozen_stats, phase2_frozen_global_mean = _build_phase2_frozen_stats(actions_rows_hist)
    phase2_frozen_global_sum = float(sum(float(v.get("sum", 0.0)) for v in phase2_frozen_stats.values()))
    phase2_frozen_global_count = float(sum(float(v.get("count", 0.0)) for v in phase2_frozen_stats.values()))
    phase2_topk_obj_agg: Dict[int, Dict[str, float]] = {}
    for row in train_rows:
        ph = str(row.get("phase", "")).strip().lower()
        if ph not in phase_history:
            continue
        phase_iter_count[ph] += 1
        phase_history[ph].append(
            {
                "iter_id": float(_safe_int(row.get("iter_id", ""), default=-1)),
                "objective": _safe_float(row.get("objective_score", "")),
                "dJ": _safe_float(row.get("dJ", "")),
                "minority_rate": _safe_float(row.get("minority_rate", "")),
            }
        )
        if policy_mode == "rarl_dqn":
            hist_j = _safe_float(row.get("J", row.get("avg_reward", "")))
            hist_a1 = _safe_float(row.get("action1_rate", ""))
            hist_h = _safe_float(row.get("policy_entropy", ""))
            if not _is_finite(hist_h):
                hist_h = _binary_entropy01(hist_a1)
            if _is_finite(hist_j):
                rarl_recent_j.append(float(hist_j))
            if _is_finite(hist_a1):
                rarl_recent_action1.append(float(hist_a1))
            if _is_finite(hist_h):
                rarl_recent_entropy.append(float(hist_h))
        if ph == "phase2":
            aid = _safe_int(row.get("policy_action_id", ""), default=-1)
            obj = _safe_float(row.get("objective_score", ""))
            if aid >= 0 and _is_finite(obj):
                item = phase2_topk_obj_agg.get(aid, {"sum": 0.0, "count": 0.0})
                item["sum"] = float(item["sum"]) + float(obj)
                item["count"] = float(item["count"]) + 1.0
                phase2_topk_obj_agg[aid] = item
    if auto_phase:
        # Resume phase in auto mode from historical rows.
        phase_mode = "phase3" if phase_iter_count["phase3"] > 0 else "phase2"

    phase3_topk_ids: List[int] = []
    phase3_topk_warm_remaining = 0
    saber_v1_sticky_action: Optional[Dict[str, object]] = None
    saber_v1_sticky_signature = ""
    saber_v1_sticky_remaining = 0
    saber_v1_sticky_trigger_iter = -1
    if phase_mode == "phase3" and phase_iter_count.get("phase3", 0) <= 0 and phase3_topk_k > 0:
        phase3_topk_ids = _compute_phase2_topk_action_ids(train_rows, phase3_topk_k)
        phase3_topk_warm_remaining = int(phase3_topk_warmup_iters) if phase3_topk_ids else 0
        if phase3_topk_ids and policy_mode in {"ts", "ucb"} and phase3_topk_prior_count > 0.0:
            target_state = ts_state if policy_mode == "ts" else ucb_state
            _bootstrap_bandit_state_with_topk(
                state=target_state,
                action_space=action_space,
                topk_action_ids=phase3_topk_ids,
                agg=phase2_topk_obj_agg,
                prior_count=float(phase3_topk_prior_count),
            )
            if policy_mode == "ts" and ts_state_path is not None:
                _save_json_state(ts_state_path, ts_state)
            if policy_mode == "ucb" and ucb_state_path is not None:
                _save_json_state(ucb_state_path, ucb_state)
            print(
                f"[OUTER][INIT] bootstrap {policy_mode} from historical phase2 topk "
                f"(k={len(phase3_topk_ids)}, prior_count={float(phase3_topk_prior_count):.3f})"
            )

    if completed_max_iter >= 1:
        for row in reversed(train_rows):
            rid = _safe_int(row.get("iter_id", ""), default=-1)
            if rid == int(completed_max_iter):
                prev_avg_reward = _safe_float(row.get("avg_reward", ""))
                policy_last_j = _safe_float(row.get("J", row.get("avg_reward", "")))
                policy_last_dj = _safe_float(row.get("dJ", ""))
                policy_last_action0_rate = _safe_float(row.get("action0_rate", ""))
                policy_last_action1_rate = _safe_float(row.get("action1_rate", ""))
                policy_last_minority_rate = _safe_float(row.get("minority_rate", ""))
                policy_last_objective = _safe_float(row.get("objective_score", ""))
                policy_last_entropy = _safe_float(row.get("policy_entropy", ""))
                policy_last_saber_v1_q_hard = _safe_float(row.get("saber_v1_q_hard", ""))
                if not _is_finite(policy_last_entropy):
                    policy_last_entropy = _binary_entropy01(policy_last_action1_rate)
                break
        if ckpt_prev is None:
            resume_ckpt = ckpt_dir / f"theta_iter{int(completed_max_iter):03d}.zip"
            if resume_ckpt.exists():
                ckpt_prev = resume_ckpt
        if replay_history.maxlen and replay_history.maxlen > 0:
            hist_start = max(1, int(start_iter) - int(replay_history.maxlen))
            for hid in range(hist_start, int(start_iter)):
                hist_dir = out_root / f"iter_{int(hid):03d}"
                if hist_dir.exists():
                    replay_history.append(hist_dir)
        print(
            f"[OUTER][INIT] resume from iter={int(completed_max_iter)} "
            f"prev_avg_reward={prev_avg_reward} ckpt_prev={ckpt_prev} "
            f"phase_now={phase_mode} phase2_done={phase_iter_count['phase2']} "
            f"phase3_done={phase_iter_count['phase3']}"
        )

    for iter_idx in range(int(start_iter), int(args.iterations) + 1):
        iter_phase = str(phase_mode)
        # Hard rule: curriculum is enabled only in phase3.
        # Even if --curriculum-enable is passed, phase2 keeps pure outer-batch training.
        phase_curriculum_enabled = iter_phase == "phase3"
        iter_level_replay_enabled = bool(level_replay_enabled)
        if edrl_v3_replay_enabled and bool(args.edrl_v3_replay_phase3_only):
            iter_level_replay_enabled = (str(iter_phase).strip().lower() == "phase3")
        if edrl_v4_replay_enabled and bool(args.edrl_v4_replay_phase3_only):
            iter_level_replay_enabled = (str(iter_phase).strip().lower() == "phase3")
        pg_ctx: Dict[str, object] = {}
        bandit_ctx: Dict[str, object] = {}
        rarl_ctx: Dict[str, object] = {}
        rarl_train_ctx: Dict[str, object] = {"updates": 0.0, "loss": float("nan")}
        plr_ctx: Dict[str, object] = {"source": "policy", "buffer_size": int(len(plr_buffer))}
        plr_topk = 0
        plr_topk_covered = 0
        plr_topk_coverage = float("nan")
        plr_topk_sample_share = float("nan")
        plr_recent_replay_ratio = float("nan")
        plr_replay_ratio = float("nan")
        plr_total_n_sampled = 0
        plr_p_new_iter = float(plr_p_new)
        plr_recent_entropy = float("nan")
        action_idx = -1
        iter_action_source = "policy"
        policy_obs: List[float] = []
        phase_allowed_indices = _phase_allowed_action_indices(
            action_space=action_space,
            iter_phase=str(iter_phase),
            phase2_fixed_n=int(phase2_fixed_num_files),
            phase3_n_choices=list(phase3_num_file_choices),
        )
        allowed_set = set(phase_allowed_indices)
        if policy_mode == "rarl_dqn":
            recent_j_mean = _safe_mean(list(rarl_recent_j), default=_finite_or(policy_last_j, 0.0))
            recent_action1_mean = _safe_mean(list(rarl_recent_action1), default=_finite_or(policy_last_action1_rate, 0.5))
            recent_entropy_mean = _safe_mean(list(rarl_recent_entropy), default=_binary_entropy01(policy_last_action1_rate))
            policy_obs = _build_outer_policy_obs(
                last_j=float(policy_last_j),
                last_dj=float(policy_last_dj),
                last_action0_rate=float(policy_last_action0_rate),
                last_action1_rate=float(policy_last_action1_rate),
                last_minority_rate=float(policy_last_minority_rate),
                last_objective=float(policy_last_objective),
                last_policy_entropy=float(policy_last_entropy),
                recent_j_mean=float(recent_j_mean),
                recent_action1_mean=float(recent_action1_mean),
                recent_entropy_mean=float(recent_entropy_mean),
                iter_phase=str(iter_phase),
                iter_idx=int(iter_idx),
                total_iters=int(args.iterations),
            )
        force_topk_pick = False
        saber_v1_sticky_active = False
        saber_v1_success_trigger = 0
        saber_v1_budget_boost = 1.0
        saber_v1_budget_num_files = 0
        saber_v1_sticky_remaining_before = int(saber_v1_sticky_remaining)
        if (
            saber_v1_mode
            and str(iter_phase).strip().lower() == "phase3"
            and int(saber_v1_sticky_remaining) > 0
            and saber_v1_sticky_action is not None
        ):
            sticky_sig = str(saber_v1_sticky_signature or _action_signature(saber_v1_sticky_action))
            sticky_idx = int(action_to_idx.get(sticky_sig, -1))
            if sticky_idx in allowed_set:
                action_idx = int(sticky_idx)
                action = _materialize_action_from_template(dict(saber_v1_sticky_action))
                n_seen_sticky, entry_score_sticky, n_sampled_sticky = _plr_entry_stats(plr_buffer, action)
                plr_ctx = {
                    "source": "sticky_replay",
                    "buffer_size": int(len(plr_buffer)),
                    "entry_index": "",
                    "entry_weight": "",
                    "entry_score_ema": float(entry_score_sticky),
                    "sticky_trigger_iter": int(saber_v1_sticky_trigger_iter),
                    "sticky_remaining_before": int(saber_v1_sticky_remaining),
                    "sticky_n_seen": int(n_seen_sticky),
                    "sticky_n_sampled": int(n_sampled_sticky),
                }
                iter_action_source = "saber_v1_sticky_replay"
                saber_v1_sticky_active = True
                saber_v1_sticky_remaining = max(0, int(saber_v1_sticky_remaining) - 1)
                force_topk_pick = True
        if (
            (not force_topk_pick)
            and str(iter_phase).strip().lower() == "phase3"
            and int(phase3_topk_warm_remaining) > 0
            and phase3_topk_ids
        ):
            topk_candidates = [
                int(i)
                for i in phase3_topk_ids
                if int(i) in allowed_set and 0 <= int(i) < len(action_space)
            ]
            if topk_candidates:
                action_idx = int(rng.choice(topk_candidates))
                action = dict(action_space[action_idx])
                bandit_ctx = {"action_idx": int(action_idx), "phase3_topk": 1}
                iter_action_source = "phase3_topk_warmstart"
                phase3_topk_warm_remaining = max(0, int(phase3_topk_warm_remaining) - 1)
                force_topk_pick = True

        if not force_topk_pick:
            if iter_level_replay_enabled and edrl_v4_mode:
                recent_hist = list(phase_history.get(str(iter_phase), []))
                if recent_hist:
                    entropy_vals = [
                        float(item.get("policy_entropy", float("nan")))
                        for item in recent_hist[-5:]
                        if _is_finite(float(item.get("policy_entropy", float("nan"))))
                    ]
                    if entropy_vals:
                        plr_recent_entropy = _safe_mean(entropy_vals, default=float("nan"))
                if not _is_finite(plr_recent_entropy):
                    plr_recent_entropy = _binary_entropy01(policy_last_action1_rate)
                p_dyn = float(plr_p_new) + float(args.edrl_v4_p_new_k) * (
                    float(args.edrl_v4_entropy_target) - float(plr_recent_entropy)
                )
                if len(plr_buffer) < 8:
                    p_dyn = max(p_dyn, float(args.edrl_v4_p_new_max) - 0.1)
                plr_p_new_iter = min(
                    float(args.edrl_v4_p_new_max),
                    max(float(args.edrl_v4_p_new_min), float(p_dyn)),
                )
            if iter_level_replay_enabled:
                action, plr_ctx = _choose_action_plr_mixed(
                    rng=rng,
                    action_space=action_space,
                    replay_entries=plr_buffer,
                    p_new=float(plr_p_new_iter),
                    min_weight=float(plr_min_weight),
                )
                action_idx = int(action_to_idx[_action_signature(action)])
                if action_idx not in allowed_set:
                    action_idx = int(rng.choice(phase_allowed_indices))
                    action = dict(action_space[action_idx])
                    plr_ctx["source"] = "policy_filtered"
                if edrl_v4_mode:
                    plr_ctx["p_new_iter"] = float(plr_p_new_iter)
                    plr_ctx["recent_entropy"] = (
                        "" if _is_nan(plr_recent_entropy) else float(plr_recent_entropy)
                    )
            elif policy_mode == "pg":
                action, pg_ctx = _choose_action_pg(
                    rng=rng,
                    action_space=action_space,
                    state=pg_state,
                    temperature=float(args.policy_temp),
                    allowed_indices=phase_allowed_indices,
                )
                action_idx = int(pg_ctx.get("action_idx", -1))
            elif policy_mode == "ucb":
                if int(iter_idx) <= int(args.warmup_iters):
                    action = _choose_action(
                        rng=rng,
                        action_space=action_space,
                        iter_idx=int(iter_idx),
                        warmup_iters=int(args.warmup_iters),
                        allowed_indices=phase_allowed_indices,
                    )
                    action_idx = int(action_to_idx[_action_signature(action)])
                    bandit_ctx = {"action_idx": int(action_idx), "warmup": 1}
                else:
                    action, bandit_ctx = _choose_action_ucb(
                        rng=rng,
                        action_space=action_space,
                        state=ucb_state,
                        ucb_c=float(args.ucb_c),
                        candidate_pool=int(args.candidate_pool),
                        allowed_indices=phase_allowed_indices,
                    )
                    action_idx = int(bandit_ctx.get("action_idx", -1))
            elif policy_mode == "ts":
                if int(iter_idx) <= int(args.warmup_iters):
                    action = _choose_action(
                        rng=rng,
                        action_space=action_space,
                        iter_idx=int(iter_idx),
                        warmup_iters=int(args.warmup_iters),
                        allowed_indices=phase_allowed_indices,
                    )
                    action_idx = int(action_to_idx[_action_signature(action)])
                    bandit_ctx = {"action_idx": int(action_idx), "warmup": 1}
                else:
                    action, bandit_ctx = _choose_action_ts(
                        rng=rng,
                        action_space=action_space,
                        state=ts_state,
                        prior_mean=float(args.ts_prior_mean),
                        prior_std=float(args.ts_prior_std),
                        obs_std=float(args.ts_obs_std),
                        candidate_pool=int(args.candidate_pool),
                        allowed_indices=phase_allowed_indices,
                    )
                    action_idx = int(bandit_ctx.get("action_idx", -1))
            elif policy_mode == "rarl_dqn":
                epsilon = _rarl_schedule_epsilon(
                    steps=int(rarl_dqn_state.get("steps", 0)),
                    eps_start=float(args.rarl_eps_start),
                    eps_end=float(args.rarl_eps_end),
                    eps_decay_iters=int(args.rarl_eps_decay_iters),
                )
                action, rarl_ctx = _choose_action_rarl_dqn(
                    rng=rng,
                    action_space=action_space,
                    state=rarl_dqn_state,
                    obs=list(policy_obs),
                    epsilon=float(epsilon),
                    allowed_indices=phase_allowed_indices,
                )
                action_idx = int(rarl_ctx.get("action_idx", -1))
            else:
                action = _choose_action(
                    rng=rng,
                    action_space=action_space,
                    iter_idx=int(iter_idx),
                    warmup_iters=int(args.warmup_iters),
                    allowed_indices=phase_allowed_indices,
                )
                action_idx = int(action_to_idx[_action_signature(action)])
        action["seed"] = int(_derive_iter_seed(base_seed=int(args.seed), iter_id=int(iter_idx)))
        iter_tag = f"{iter_idx:03d}"
        iter_dir = out_root / f"iter_{iter_tag}"
        action_num_files = int(action["num_files"])
        if str(iter_phase).strip().lower() == "phase2" and int(phase2_fixed_num_files) > 0:
            action_num_files = int(phase2_fixed_num_files)
        if saber_v1_mode and bool(saber_v1_sticky_active):
            boosted_n = int(action_num_files)
            boost_mult = max(1.0, float(args.saber_v1_success_budget_mult))
            if boost_mult > 1.0:
                boosted_n = max(boosted_n, int(math.ceil(float(action_num_files) * float(boost_mult))))
            boosted_n = max(boosted_n, int(max(0, int(args.saber_v1_success_min_num_files))))
            boosted_n = max(int(min_files), min(int(max_files), int(boosted_n)))
            saber_v1_budget_boost = float(boost_mult)
            saber_v1_budget_num_files = int(boosted_n)
            action_num_files = int(boosted_n)
        inner_budget_n = int(args.inner_fixed_n) if int(args.inner_fixed_n) > 0 else int(action_num_files)
        if saber_v1_mode and bool(saber_v1_sticky_active):
            inner_budget_n = max(int(inner_budget_n), int(action_num_files))
        if int(saber_v1_budget_num_files) <= 0:
            saber_v1_budget_num_files = int(action_num_files)
        if str(args.inner_stop_mode) == "fixed_n" and inner_budget_n <= 0:
            raise ValueError("inner fixed N budget must be > 0")
        dynamic_table_base = 0 if str(args.inner_stop_mode) == "fixed_n" else int(args.table_base)
        curriculum_alpha = float("nan")
        train_data_root = iter_dir
        train_manifest_path = post_stage_dir / "manifest.json"
        curriculum_outer_files = 0
        curriculum_base_files = 0
        curriculum_replay_files = 0

        print(
            f"[OUTER][ITER] iter={iter_tag}/{int(args.iterations):03d} "
            f"phase={iter_phase} "
            f"action=(muA={action['mu_a']},muB={action['mu_b']},p={action['ratio_a']},"
            f"n={action_num_files},seed={action['seed']},pattern={action['pattern']}) "
            f"inner=(mode={args.inner_stop_mode},budget_n={inner_budget_n},table_base={dynamic_table_base})"
        )
        if str(args.policy_mode) == "pg" and not iter_level_replay_enabled:
            print(
                f"[OUTER][PG] iter={iter_tag} action_id={int(pg_ctx.get('action_idx', -1))} "
                f"prob={float(pg_ctx.get('action_prob', float('nan'))):.6f} "
                f"entropy={float(pg_ctx.get('entropy', float('nan'))):.6f}"
            )
        if iter_level_replay_enabled:
            print(
                f"[OUTER][PLR] iter={iter_tag} source={str(plr_ctx.get('source', 'na'))} "
                f"buffer_size={int(plr_ctx.get('buffer_size', len(plr_buffer)))} "
                f"entry_index={plr_ctx.get('entry_index', '')} "
                f"entry_score_ema={plr_ctx.get('entry_score_ema', '')}"
            )
        if saber_v1_mode and bool(saber_v1_sticky_active):
            print(
                f"[OUTER][SABER-STICKY] iter={iter_tag} "
                f"trigger_iter={int(saber_v1_sticky_trigger_iter)} "
                f"remaining_after_pick={int(saber_v1_sticky_remaining)} "
                f"boost_n={int(action_num_files)} "
                f"budget_mult={float(saber_v1_budget_boost):.2f}"
            )
        if policy_mode == "ucb" and not iter_level_replay_enabled:
            print(
                f"[OUTER][UCB] iter={iter_tag} action_id={int(bandit_ctx.get('action_idx', -1))} "
                f"score={float(bandit_ctx.get('score', float('nan'))):.6f} "
                f"mean={float(bandit_ctx.get('mean_obj', float('nan'))):.6f} "
                f"count={float(bandit_ctx.get('count', 0.0)):.3f}"
            )
        if policy_mode == "ts" and not iter_level_replay_enabled:
            print(
                f"[OUTER][TS] iter={iter_tag} action_id={int(bandit_ctx.get('action_idx', -1))} "
                f"draw={float(bandit_ctx.get('ts_draw', float('nan'))):.6f} "
                f"post_mean={float(bandit_ctx.get('posterior_mean', float('nan'))):.6f} "
                f"count={float(bandit_ctx.get('count', 0.0)):.3f}"
            )
        if policy_mode == "rarl_dqn" and not iter_level_replay_enabled:
            reward_mode = "objective"
            if objective_mode == "rarl" and int(args.rarl_zero_sum_strict) == 1:
                reward_mode = "-J"
            print(
                f"[OUTER][RARL] iter={iter_tag} action_id={int(rarl_ctx.get('action_idx', -1))} "
                f"eps={float(rarl_ctx.get('epsilon', float('nan'))):.4f} "
                f"q_sel={float(rarl_ctx.get('q_selected', float('nan'))):.6f} "
                f"explore={int(rarl_ctx.get('explore', 0))} reward={reward_mode}"
            )
        gen_cmd = [
            args.python_bin,
            str(CODES_DIR / "generation" / "generate_outer_small_batch.py"),
            "--run-id",
            str(args.run_id),
            "--iter-id",
            str(iter_idx),
            "--request-number",
            str(int(args.request_number)),
            "--mu-a",
            str(action["mu_a"]),
            "--mu-b",
            str(action["mu_b"]),
            "--ratio-a",
            str(action["ratio_a"]),
            "--num-files",
            str(int(action_num_files)),
            "--seed",
            str(action["seed"]),
            "--pattern",
            str(action["pattern"]),
            "--out-root",
            str(out_root),
        ]
        if args.verify_batch:
            gen_cmd.append("--verify")
        _run_generator_with_retry(
            cmd_base=gen_cmd,
            workers=int(args.workers),
            retry_max=int(args.gen_retry_max),
        )

        if phase_curriculum_enabled:
            curriculum_alpha = _iter_curriculum_alpha(
                iter_idx=int(iter_idx),
                alpha_start=float(args.curriculum_alpha_start),
                alpha_end=float(args.curriculum_alpha_end),
                alpha_horizon=int(alpha_horizon),
            )
            replay_pool: List[Path] = []
            for hist_iter_dir in replay_history:
                replay_pool.extend(_scan_dynamic_files(hist_iter_dir, int(args.request_number)))
            mix_result = _materialize_curriculum_batch(
                iter_idx=int(iter_idx),
                request_number=int(args.request_number),
                num_files=int(action_num_files),
                outer_iter_dir=iter_dir,
                base_root=base_root,  # type: ignore[arg-type]
                out_mix_root=curriculum_root,
                alpha_outer=float(curriculum_alpha),
                replay_file_pool=replay_pool,
                replay_ratio=float(args.curriculum_replay_ratio),
                rng=rng,
            )
            train_data_root = Path(mix_result["mix_iter_dir"])
            train_manifest_path = Path(mix_result["manifest_path"])
            curriculum_outer_files = int(mix_result["outer_count"])
            curriculum_base_files = int(mix_result["base_count"])
            curriculum_replay_files = int(mix_result["replay_count"])
            _append_csv_row(
                post_stage_dir / "outer_curriculum.csv",
                [
                    "iter_id",
                    "phase",
                    "alpha_outer",
                    "num_files",
                    "outer_files",
                    "base_files",
                    "replay_files",
                    "outer_iter_dir",
                    "mix_iter_dir",
                    "manifest",
                ],
                {
                    "iter_id": int(iter_idx),
                    "phase": str(iter_phase),
                    "alpha_outer": float(curriculum_alpha),
                    "num_files": int(action_num_files),
                    "outer_files": int(curriculum_outer_files),
                    "base_files": int(curriculum_base_files),
                    "replay_files": int(curriculum_replay_files),
                    "outer_iter_dir": str(iter_dir),
                    "mix_iter_dir": str(train_data_root),
                    "manifest": str(train_manifest_path),
                },
            )
            print(
                f"[OUTER][S4] iter={iter_tag} alpha_outer={float(curriculum_alpha):.4f} "
                f"mix=(outer={curriculum_outer_files},base={curriculum_base_files},replay={curriculum_replay_files}) "
                f"train_data_root={train_data_root}"
            )
        replay_history.append(iter_dir)

        iter_ckpt = ckpt_dir / f"theta_iter{iter_tag}.zip"
        before_decisions = len(_read_csv_rows(decision_csv))
        before_trace = len(_read_csv_rows(trace_csv))
        before_training = len(_read_csv_rows(training_csv))
        before_paths = len(_read_csv_rows(path_map_csv))

        train_cmd = [
            args.python_bin,
            str(CODES_DIR / "Dynamic_master34959.py"),
            "--dist_name",
            str(args.dist_name),
            "--request_number",
            str(int(args.request_number)),
            "--algorithm",
            str(args.algorithm),
            "--algo_version",
            str(args.algo_version),
            "--stage-mode",
            "train_only",
            "--run-name",
            str(args.run_id),
            "--seed",
            str(int(args.seed)),
            "--skip-generator",
            "--external-data-root",
            str(train_data_root),
            "--save-model-path",
            str(iter_ckpt),
        ]
        ckpt_in_path: Optional[Path] = ckpt_prev if (ckpt_prev and ckpt_prev.exists()) else None
        if iter_phase == "phase2" and phase2_freeze_inner:
            if phase2_anchor_ckpt is None or (not phase2_anchor_ckpt.exists()):
                raise FileNotFoundError(
                    "phase2 freeze requires theta_phase1 checkpoint but it is missing; "
                    "pass --base-ckpt <theta_phase1.zip> or use --no-phase2-freeze-inner"
                )
            ckpt_in_path = phase2_anchor_ckpt
        if ckpt_in_path and ckpt_in_path.exists():
            train_cmd.extend(["--init-model-path", str(ckpt_in_path)])

        env = os.environ.copy()
        env["RL_DYNAMIC_INDEX_MODE"] = "mod"
        env["RL_DYNAMIC_FILE_COUNT"] = str(int(action_num_files))
        env["RL_DYNAMIC_TABLE_BASE"] = str(int(dynamic_table_base))
        env["RL_DYNAMIC_STRICT_PATH"] = "1"
        env["RL_DYNAMIC_MANIFEST"] = str(train_manifest_path)
        env["RL_DYNAMIC_PATH_MAP_CSV"] = str(path_map_csv)
        env["RL_OUTER_ITER_ID"] = str(int(iter_idx))
        env["RL_TRAIN_ONLY_STOP_MODE"] = str(args.inner_stop_mode)
        if str(args.inner_stop_mode) == "fixed_n":
            env["RL_TRAIN_ONLY_FIXED_TABLES"] = str(int(inner_budget_n))
            env["RL_TRAIN_ONLY_EARLY_STOP"] = "0"
        inner_ppo_new_ent_coef = float(
            args.phase2_inner_ppo_new_ent_coef if str(iter_phase).strip().lower() == "phase2" else args.phase3_inner_ppo_new_ent_coef
        )
        if str(args.algorithm).strip().upper() == "PPO_NEW":
            env["RL_PPO_NEW_ENT_COEF"] = str(float(inner_ppo_new_ent_coef))
        env["RUN_ID"] = str(args.run_id)

        start_train = time.time()
        subprocess.run(train_cmd, check=True, env=env)
        elapsed = time.time() - start_train

        all_decisions = _read_csv_rows(decision_csv)
        new_decisions = all_decisions[before_decisions:]
        all_trace_rows = _read_csv_rows(trace_csv)
        new_trace_rows = all_trace_rows[before_trace:]
        avg_reward, action0_rate, action1_rate, minority_rate, minority_action = _calc_metrics(new_decisions)
        saber_v1_metrics = _calc_saber_v1_trace_metrics(
            new_trace_rows,
            hard_threshold=int(args.saber_v1_hard_threshold),
            easy_threshold=int(args.saber_v1_easy_threshold),
        )
        iter_policy_entropy = _binary_entropy01(action1_rate)
        new_training_rows = len(_read_csv_rows(training_csv)) - before_training

        all_path_rows = _read_csv_rows(path_map_csv)
        new_path_rows = all_path_rows[before_paths:]
        missing_paths = 0
        read_ok_paths = 0
        table_numbers_all = set()
        table_numbers_dynamic = set()
        for row in new_path_rows:
            exists_val = str(row.get("exists", "")).strip().lower()
            if exists_val in {"0", "false", "no", ""}:
                missing_paths += 1
            read_ok_val = str(row.get("read_ok", "")).strip().lower()
            if read_ok_val in {"1", "true", "yes", "y"}:
                read_ok_paths += 1
            module_name = str(row.get("module", "")).strip()
            table_num = None
            try:
                table_num = int(str(row.get("table_number", "")).strip())
            except Exception:
                table_num = None
            if table_num is not None:
                table_numbers_all.add(int(table_num))
            if module_name == "dynamic_RL34959":
                if table_num is not None:
                    table_numbers_dynamic.add(int(table_num))
        # Use all modules as the primary processed-table metric to avoid
        # false partial-counts when one module's path-map row is intermittently missing.
        processed_tables = len(table_numbers_all)
        processed_tables_dynamic = len(table_numbers_dynamic)
        d_penalty = float(missing_paths) / float(max(1, len(new_path_rows)))
        read_rate = float(read_ok_paths) / float(max(1, len(new_path_rows)))
        if args.validate_path_map and read_rate < float(args.min_path_read_rate):
            raise RuntimeError(
                f"path read rate below threshold at iter={iter_tag}: "
                f"{read_rate:.6f} < {float(args.min_path_read_rate):.6f}"
            )
        if str(args.inner_stop_mode) == "fixed_n":
            if processed_tables >= int(inner_budget_n):
                stop_reason = "fixed_budget_n"
            else:
                stop_reason = "fixed_budget_n_partial"
        else:
            stop_reason = "converge_or_boundary"

        j_val = 0.0 if _is_nan(avg_reward) else float(avg_reward)
        if _is_nan(prev_avg_reward) or _is_nan(avg_reward):
            dj_val = 0.0
        else:
            dj_val = float(avg_reward) - float(prev_avg_reward)
        rho_eff = 0.0 if _is_nan(minority_rate) else float(minority_rate)
        collapse_gap = max(0.0, float(args.rho_target) - float(rho_eff))
        collapse_penalty = float(args.eta_collapse) * (float(collapse_gap) ** float(args.collapse_gap_power))
        b_val = -float(collapse_penalty)
        rho_floor_gap = max(0.0, float(args.rho_floor) - float(rho_eff))
        floor_soft_penalty = float(args.rho_floor_weight) * float(rho_floor_gap)
        floor_hard_penalty = float(args.rho_floor_hard_weight) * (float(rho_floor_gap) ** 2.0)
        phase2_j_frozen = 0.0
        phase2_challenge = 0.0
        phase2_too_hard = 0.0
        phase2_g_minority = 0.0
        phase2_term_challenge = 0.0
        phase2_term_minority = 0.0
        phase2_term_lp_dj = 0.0
        phase2_term_lp_j = 0.0
        phase2_term_too_hard = 0.0
        phase2_term_feasibility = 0.0
        phase2_v4_learnability = 0.0
        phase2_v4_entropy = 0.0
        phase2_v4_novelty = 0.0
        phase2_v4_n_seen = 0
        phase2_v4_n_sampled = 0
        phase2_v4_entry_score = 0.0
        saber_v0_learnability = 0.0
        saber_v0_novelty = 0.0
        saber_v0_n_seen = 0
        saber_v0_n_sampled = 0
        saber_v0_entry_score = 0.0
        saber_v0_term_lp_dj = 0.0
        saber_v0_term_lp_j = 0.0
        saber_v0_term_novelty = 0.0
        saber_v0_term_feasibility = 0.0
        saber_v1_learnability = 0.0
        saber_v1_novelty = 0.0
        saber_v1_n_seen = 0
        saber_v1_n_sampled = 0
        saber_v1_entry_score = 0.0
        saber_v1_hard_count = 0
        saber_v1_easy_count = 0
        saber_v1_q_hard = float("nan")
        saber_v1_r_hard = float("nan")
        saber_v1_p_easy = float("nan")
        saber_v1_m_ins = float("nan")
        saber_v1_hard_action1_rate = float("nan")
        saber_v1_hard_wait_share = float("nan")
        saber_v1_easy_action1_rate = float("nan")
        saber_v1_dq_hard = 0.0
        saber_v1_term_q_hard = 0.0
        saber_v1_term_r_hard = 0.0
        saber_v1_term_p_easy = 0.0
        saber_v1_term_dq_hard = 0.0
        saber_v1_term_novelty = 0.0
        saber_v1_no_success_flag = 0.0
        saber_v1_term_no_success = 0.0
        saber_v1_term_feasibility = 0.0
        phase2_j_frozen_source = "na"
        objective_formula = str(objective_mode)
        if objective_mode == "rarl":
            # Pure adversarial objective: maximize difficulty for the learner.
            objective = (
                -j_val
                - float(args.path_penalty) * d_penalty
            )
            b_val = 0.0
            floor_soft_penalty = 0.0
            floor_hard_penalty = 0.0
            objective_formula = "rarl"
        elif objective_mode == "plr":
            # PLR/UED-style learning potential objective.
            objective = (
                + float(args.plr_dj_weight) * abs(float(dj_val))
                + float(args.plr_j_weight) * j_val
                - float(args.path_penalty) * d_penalty
            )
            b_val = 0.0
            floor_soft_penalty = 0.0
            floor_hard_penalty = 0.0
            objective_formula = "plr"
        elif objective_mode == "saber_v0":
            if str(iter_phase).strip().lower() == "phase2":
                phase2_j_frozen = max(0.0, min(1.0, float(j_val)))
                phase2_j_frozen_source = "phase2_current_eval"
            else:
                fallback_frozen = (
                    float(phase2_frozen_global_mean)
                    if _is_finite(float(phase2_frozen_global_mean))
                    else max(0.0, min(1.0, float(j_val)))
                )
                phase2_j_frozen = max(
                    0.0,
                    min(
                        1.0,
                        float(
                            _lookup_phase2_frozen_mean(
                                stats=phase2_frozen_stats,
                                action=action,
                                fallback=float(fallback_frozen),
                            )
                        ),
                    ),
                )
                phase2_j_frozen_source = "phase2_lookup"

            sigma = max(1e-6, float(args.saber_v0_j_sigma))
            z = (float(phase2_j_frozen) - float(args.saber_v0_j_center)) / sigma
            saber_v0_learnability = float(math.exp(-0.5 * (z ** 2.0)))
            n_seen, entry_score, n_sampled = _plr_entry_stats(plr_buffer, action)
            saber_v0_n_seen = int(n_seen)
            saber_v0_entry_score = float(entry_score)
            saber_v0_n_sampled = int(n_sampled)
            saber_v0_novelty = float(1.0 / math.sqrt(1.0 + float(max(0, n_seen))))
            saber_v0_term_lp_dj = float(args.saber_v0_dj_weight) * abs(float(dj_val))
            saber_v0_term_lp_j = float(args.saber_v0_j_weight) * float(j_val)
            saber_v0_term_novelty = float(args.saber_v0_novelty_weight) * float(saber_v0_novelty)
            saber_v0_term_feasibility = float(d_penalty)
            objective = (
                float(saber_v0_learnability)
                * (
                    float(saber_v0_term_lp_dj)
                    + float(saber_v0_term_lp_j)
                    + float(saber_v0_term_novelty)
                )
                - float(saber_v0_term_feasibility)
            )
            b_val = 0.0
            floor_soft_penalty = 0.0
            floor_hard_penalty = 0.0
            objective_formula = "saber_v0"
        elif objective_mode == "saber_v1":
            if str(iter_phase).strip().lower() == "phase2":
                phase2_j_frozen = max(0.0, min(1.0, float(j_val)))
                phase2_j_frozen_source = "phase2_current_eval"
            else:
                fallback_frozen = (
                    float(phase2_frozen_global_mean)
                    if _is_finite(float(phase2_frozen_global_mean))
                    else max(0.0, min(1.0, float(j_val)))
                )
                phase2_j_frozen = max(
                    0.0,
                    min(
                        1.0,
                        float(
                            _lookup_phase2_frozen_mean(
                                stats=phase2_frozen_stats,
                                action=action,
                                fallback=float(fallback_frozen),
                            )
                        ),
                    ),
                )
                phase2_j_frozen_source = "phase2_lookup"

            sigma = max(1e-6, float(args.saber_v1_j_sigma))
            z = (float(phase2_j_frozen) - float(args.saber_v1_j_center)) / sigma
            saber_v1_learnability = float(math.exp(-0.5 * (z ** 2.0)))
            n_seen, entry_score, n_sampled = _plr_entry_stats(plr_buffer, action)
            saber_v1_n_seen = int(n_seen)
            saber_v1_entry_score = float(entry_score)
            saber_v1_n_sampled = int(n_sampled)
            saber_v1_novelty = float(1.0 / math.sqrt(1.0 + float(max(0, n_seen))))
            saber_v1_hard_count = int(_finite_or(saber_v1_metrics.get("hard_count", 0.0), 0.0))
            saber_v1_easy_count = int(_finite_or(saber_v1_metrics.get("easy_count", 0.0), 0.0))
            saber_v1_q_hard = _safe_float(saber_v1_metrics.get("Q_hard_rem", float("nan")))
            saber_v1_r_hard = _safe_float(saber_v1_metrics.get("R_hard_rem", float("nan")))
            saber_v1_p_easy = _safe_float(saber_v1_metrics.get("P_easy_wait", float("nan")))
            saber_v1_m_ins = _safe_float(saber_v1_metrics.get("M_ins", float("nan")))
            saber_v1_hard_action1_rate = _safe_float(saber_v1_metrics.get("hard_action1_rate", float("nan")))
            saber_v1_hard_wait_share = _safe_float(saber_v1_metrics.get("hard_wait_share", float("nan")))
            saber_v1_easy_action1_rate = _safe_float(saber_v1_metrics.get("easy_action1_rate", float("nan")))
            if _is_finite(saber_v1_q_hard) and _is_finite(policy_last_saber_v1_q_hard):
                saber_v1_dq_hard = float(saber_v1_q_hard) - float(policy_last_saber_v1_q_hard)
            else:
                saber_v1_dq_hard = 0.0
            saber_v1_term_q_hard = float(args.saber_v1_q_weight) * float(_finite_or(saber_v1_q_hard, 0.0))
            saber_v1_term_r_hard = float(args.saber_v1_r_weight) * float(_finite_or(saber_v1_r_hard, 0.0))
            saber_v1_term_p_easy = float(args.saber_v1_easy_weight) * float(_finite_or(saber_v1_p_easy, 0.0))
            saber_v1_term_dq_hard = float(args.saber_v1_dq_weight) * abs(float(saber_v1_dq_hard))
            saber_v1_term_novelty = float(args.saber_v1_novelty_weight) * float(saber_v1_novelty)
            saber_v1_no_success_flag = (
                1.0
                if (
                    int(saber_v1_hard_count) > 0
                    and float(_finite_or(saber_v1_r_hard, 0.0)) <= 1e-12
                )
                else 0.0
            )
            saber_v1_term_no_success = (
                float(args.saber_v1_no_success_weight)
                * float(saber_v1_no_success_flag)
                * float(_finite_or(saber_v1_hard_wait_share, 0.0))
            )
            saber_v1_term_feasibility = float(d_penalty)
            hard_gate = 1.0 if int(saber_v1_hard_count) > 0 else 0.0
            objective = (
                float(hard_gate)
                * float(saber_v1_learnability)
                * (
                    float(saber_v1_term_q_hard)
                    + float(saber_v1_term_r_hard)
                    + float(saber_v1_term_p_easy)
                    + float(saber_v1_term_dq_hard)
                    + float(saber_v1_term_novelty)
                )
                - float(saber_v1_term_no_success)
                - float(saber_v1_term_feasibility)
            )
            b_val = 0.0
            floor_soft_penalty = 0.0
            floor_hard_penalty = 0.0
            objective_formula = "saber_v1"
        else:
            if phase2_difficulty_objective:
                # Unified phase2/phase3 objective: challenge + minority gain - feasibility.
                if str(iter_phase).strip().lower() == "phase2":
                    phase2_j_frozen = max(0.0, min(1.0, float(j_val)))
                    phase2_j_frozen_source = "phase2_current_eval"
                else:
                    fallback_frozen = (
                        float(phase2_frozen_global_mean)
                        if _is_finite(float(phase2_frozen_global_mean))
                        else max(0.0, min(1.0, float(j_val)))
                    )
                    phase2_j_frozen = max(
                        0.0,
                        min(
                            1.0,
                            float(
                                _lookup_phase2_frozen_mean(
                                    stats=phase2_frozen_stats,
                                    action=action,
                                    fallback=float(fallback_frozen),
                                )
                            ),
                        ),
                    )
                    phase2_j_frozen_source = "phase2_lookup"

                phase2_challenge = max(0.0, 1.0 - float(phase2_j_frozen))
                if phase2_minority_reward_enable:
                    minority_target_rate = float(action0_rate) if str(phase1_minority_action) == "0" else float(action1_rate)
                    if not _is_finite(minority_target_rate):
                        minority_target_rate = 0.0
                    if edrl_v4_mode:
                        phase2_g_minority = (
                            max(0.0, float(minority_target_rate) - float(phase1_minority_rate))
                            + float(args.edrl_v4_minority_abs_weight) * max(0.0, float(minority_target_rate))
                        )
                    elif edrl_v3_mode:
                        # EDRL-v3: keep delta reward and add absolute minority-rate term
                        # to avoid zero-gradient collapse when phase1 baseline is hard to exceed.
                        phase2_g_minority = (
                            max(0.0, float(minority_target_rate) - float(phase1_minority_rate))
                            + float(edrl_v3_minority_abs_weight) * max(0.0, float(minority_target_rate))
                        )
                    else:
                        phase2_g_minority = max(0.0, float(minority_target_rate) - float(phase1_minority_rate))
                if edrl_v4_mode:
                    sigma = max(1e-6, float(args.edrl_v4_j_sigma))
                    z = (float(phase2_j_frozen) - float(args.edrl_v4_j_center)) / sigma
                    phase2_v4_learnability = float(math.exp(-0.5 * (z ** 2.0)))
                    phase2_v4_entropy = float(_binary_entropy01(action1_rate))
                    n_seen, entry_score, n_sampled = _plr_entry_stats(plr_buffer, action)
                    phase2_v4_n_seen = int(n_seen)
                    phase2_v4_entry_score = float(entry_score)
                    phase2_v4_n_sampled = int(n_sampled)
                    phase2_v4_novelty = float(1.0 / math.sqrt(1.0 + float(max(0, n_seen))))

                    challenge_eff = float(phase2_challenge) * float(phase2_v4_learnability)
                    phase2_term_challenge = float(args.edrl_v4_challenge_weight) * float(challenge_eff)
                    phase2_term_lp_dj = float(args.edrl_v4_lp_weight) * abs(float(dj_val))
                    phase2_term_lp_j = float(args.edrl_v4_j_weight) * float(j_val)
                    phase2_term_minority = float(args.edrl_v4_minority_weight) * float(phase2_g_minority)
                    phase2_term_too_hard = float(args.edrl_v4_entropy_weight) * float(phase2_v4_entropy)
                    phase2_too_hard = float(phase2_v4_novelty)
                else:
                    phase2_term_challenge = float(args.phase2_hard_weight) * float(phase2_challenge)
                    phase2_term_minority = float(phase2_minority_reward_weight) * float(phase2_g_minority)
                    if edrl_v3_mode and str(iter_phase).strip().lower() == "phase3":
                        phase2_term_lp_dj = float(args.edrl_v3_dj_weight) * abs(float(dj_val))
                        phase2_term_lp_j = float(args.edrl_v3_j_weight) * float(j_val)
                    phase2_term_too_hard = 0.0
                    phase2_too_hard = 0.0
                phase2_term_feasibility = float(d_penalty)
                if edrl_v4_mode:
                    objective = (
                        + float(phase2_term_challenge)
                        + float(phase2_term_lp_dj)
                        + float(phase2_term_lp_j)
                        + float(phase2_term_minority)
                        + float(phase2_term_too_hard)
                        + float(args.edrl_v4_novelty_weight) * float(phase2_v4_novelty)
                        - float(phase2_term_feasibility)
                    )
                else:
                    objective = (
                        + float(phase2_term_challenge)
                        + float(phase2_term_minority)
                        + float(phase2_term_lp_dj)
                        + float(phase2_term_lp_j)
                        - float(phase2_term_feasibility)
                    )
                b_val = 0.0
                floor_soft_penalty = 0.0
                floor_hard_penalty = 0.0
                if edrl_v4_mode:
                    objective_formula = "phase23_plr_ued_v4"
                else:
                    objective_formula = "phase23_difficulty_minor_v3_lp" if edrl_v3_mode else "phase23_difficulty_minor_v1"
            else:
                objective = (
                    +j_val
                    + float(args.lambda_dj) * dj_val
                    + b_val
                    - float(args.path_penalty) * d_penalty
                    - float(floor_soft_penalty)
                    - float(floor_hard_penalty)
                )
                objective_formula = "edrl_phase3_legacy"

        if (
            (
                objective_mode == "edrl"
                and phase2_difficulty_objective
            )
            or objective_mode == "saber_v0"
            or objective_mode == "saber_v1"
        ) and str(iter_phase).strip().lower() == "phase2":
            _update_phase2_frozen_stats(
                stats=phase2_frozen_stats,
                action=action,
                j_val=float(phase2_j_frozen),
            )
            phase2_frozen_global_sum += float(phase2_j_frozen)
            phase2_frozen_global_count += 1.0
            if phase2_frozen_global_count > 0.0:
                phase2_frozen_global_mean = float(phase2_frozen_global_sum) / float(phase2_frozen_global_count)
        pg_action_id = ""
        pg_action_prob = ""
        pg_advantage = ""
        pg_baseline = ""
        pg_entropy = ""
        policy_score = ""
        policy_post_mean = ""
        policy_post_var = ""
        if iter_level_replay_enabled:
            pg_action_id = int(action_idx)
        elif policy_mode == "pg":
            probs = [float(x) for x in list(pg_ctx.get("probs", []))]
            pg_action_id = int(pg_ctx.get("action_idx", -1))
            pg_action_prob = float(pg_ctx.get("action_prob", float("nan")))
            pg_entropy = float(pg_ctx.get("entropy", float("nan")))
            pg_adv, pg_base = _update_pg_state(
                state=pg_state,
                action_idx=int(pg_action_id),
                probs=probs,
                objective=float(objective),
                lr=float(args.policy_lr),
                baseline_momentum=float(args.policy_baseline_momentum),
            )
            pg_advantage = float(pg_adv)
            pg_baseline = float(pg_base)
            if pg_state_path is not None:
                _save_pg_state(pg_state_path, pg_state)
        elif policy_mode == "ucb":
            pg_action_id = int(action_idx)
            policy_score = float(bandit_ctx.get("score", float("nan")))
            _update_ucb_state(
                state=ucb_state,
                action_idx=int(action_idx),
                objective=float(objective),
                decay=float(args.policy_decay),
            )
            if ucb_state_path is not None:
                _save_json_state(ucb_state_path, ucb_state)
        elif policy_mode == "ts":
            pg_action_id = int(action_idx)
            policy_score = float(bandit_ctx.get("ts_draw", float("nan")))
            policy_post_mean = float(bandit_ctx.get("posterior_mean", float("nan")))
            policy_post_var = float(bandit_ctx.get("posterior_var", float("nan")))
            _update_ts_state(
                state=ts_state,
                action_idx=int(action_idx),
                objective=float(objective),
                decay=float(args.policy_decay),
            )
            if ts_state_path is not None:
                _save_json_state(ts_state_path, ts_state)
        elif policy_mode == "rarl_dqn":
            pg_action_id = int(action_idx)
            pg_action_prob = float(rarl_ctx.get("epsilon", float("nan")))
            policy_score = float(rarl_ctx.get("q_selected", float("nan")))
            policy_post_mean = float(rarl_ctx.get("q_max", float("nan")))
            rarl_policy_reward = float(objective)
            if objective_mode == "rarl" and int(args.rarl_zero_sum_strict) == 1:
                rarl_policy_reward = -float(j_val)
            rarl_ctx["policy_reward"] = float(rarl_policy_reward)
            next_recent_j = list(rarl_recent_j) + [float(j_val)]
            next_recent_action1 = list(rarl_recent_action1) + [float(action1_rate)]
            next_recent_entropy = list(rarl_recent_entropy) + [float(iter_policy_entropy)]
            next_obs = _build_outer_policy_obs(
                last_j=float(j_val),
                last_dj=float(dj_val),
                last_action0_rate=float(action0_rate),
                last_action1_rate=float(action1_rate),
                last_minority_rate=float(rho_eff),
                last_objective=float(objective),
                last_policy_entropy=float(iter_policy_entropy),
                recent_j_mean=float(_safe_mean(next_recent_j, default=j_val)),
                recent_action1_mean=float(_safe_mean(next_recent_action1, default=action1_rate)),
                recent_entropy_mean=float(_safe_mean(next_recent_entropy, default=iter_policy_entropy)),
                iter_phase=str(iter_phase),
                iter_idx=int(iter_idx + 1),
                total_iters=int(args.iterations),
            )
            obs_dim = int(max(1, _safe_int(rarl_dqn_state.get("state_dim", rarl_state_dim), default=rarl_state_dim)))
            _rarl_push_transition(
                state=rarl_dqn_state,
                obs=_safe_obs(list(policy_obs), state_dim=obs_dim),
                action_idx=int(action_idx),
                reward=float(rarl_policy_reward),
                next_obs=_safe_obs(list(next_obs), state_dim=obs_dim),
                done=0,
            )
            do_update = int(args.rarl_k1) <= 1 or (int(rarl_dqn_state.get("steps", 0)) % int(max(1, args.rarl_k1)) == 0)
            updates_to_run = int(args.rarl_k2) if do_update else 0
            rarl_train_ctx = _rarl_train_dqn(
                rng=rng,
                state=rarl_dqn_state,
                batch_size=int(args.rarl_batch_size),
                updates=int(updates_to_run),
                min_replay=int(args.rarl_min_replay),
                gamma=float(args.rarl_gamma),
                lr=float(args.rarl_lr),
                target_sync_every=int(args.rarl_target_sync),
            )
            pg_advantage = float(rarl_train_ctx.get("loss", float("nan")))
            pg_baseline = float(rarl_train_ctx.get("updates", 0.0))
            pg_entropy = float(len(list(rarl_dqn_state.get("replay", []))))
            if rarl_dqn_state_path is not None:
                _save_json_state(rarl_dqn_state_path, rarl_dqn_state)

        plr_update: Dict[str, object] = {}
        if (
            saber_v1_mode
            and int(saber_v1_hard_count) > 0
            and float(_finite_or(saber_v1_r_hard, 0.0)) >= float(args.saber_v1_success_r_threshold)
        ):
            saber_v1_sticky_action = dict(_action_template(action))
            saber_v1_sticky_signature = _action_signature(action)
            saber_v1_sticky_remaining = max(0, int(args.saber_v1_sticky_replay_iters))
            saber_v1_sticky_trigger_iter = int(iter_idx)
            saber_v1_success_trigger = 1

        if iter_level_replay_enabled:
            plr_update = _update_plr_buffer(
                replay_entries=plr_buffer,
                action=action,
                score=float(objective),
                ema_alpha=float(plr_priority_ema_alpha),
                iter_idx=int(iter_idx),
            )
            _save_plr_buffer(plr_buffer_path, plr_buffer)
            src = str(plr_ctx.get("source", "policy")).strip().lower()
            if src in {"new", "replay", "sticky_replay"}:
                plr_total_samples += 1
                if src == "new":
                    plr_new_samples += 1
                else:
                    plr_replay_samples += 1
                plr_recent_sources.append("replay" if src == "sticky_replay" else src)
            if plr_total_samples > 0:
                plr_replay_ratio = float(plr_replay_samples) / float(plr_total_samples)
            if len(plr_recent_sources) > 0:
                rr = sum(1 for s in list(plr_recent_sources) if str(s) == "replay")
                plr_recent_replay_ratio = float(rr) / float(len(plr_recent_sources))
            entries_sorted = sorted(
                list(plr_buffer),
                key=lambda it: _safe_float(it.get("score_ema", 0.0)),
                reverse=True,
            )
            plr_topk = int(min(10, len(entries_sorted)))
            if plr_topk > 0:
                topk_entries = entries_sorted[:plr_topk]
                plr_topk_covered = int(
                    sum(1 for it in topk_entries if _safe_int(it.get("n_sampled", 0), default=0) > 0)
                )
                plr_topk_coverage = float(plr_topk_covered) / float(plr_topk)
                topk_n = sum(max(0, _safe_int(it.get("n_sampled", 0), default=0)) for it in topk_entries)
                plr_total_n_sampled = int(
                    sum(max(0, _safe_int(it.get("n_sampled", 0), default=0)) for it in entries_sorted)
                )
                if plr_total_n_sampled > 0:
                    plr_topk_sample_share = float(topk_n) / float(plr_total_n_sampled)
            _append_csv_row(
                plr_stats_csv,
                [
                    "iter_id",
                    "phase",
                    "source",
                    "buffer_size",
                    "total_samples",
                    "new_samples",
                    "replay_samples",
                    "replay_ratio",
                    "recent_replay_ratio_w20",
                    "topk",
                    "topk_covered",
                    "topk_coverage",
                    "topk_sample_share",
                    "buffer_total_n_sampled",
                    "entry_index",
                    "entry_score_ema",
                    "p_new_iter",
                    "recent_entropy",
                    "update_event",
                    "update_score_ema",
                ],
                {
                    "iter_id": int(iter_idx),
                    "phase": str(iter_phase),
                    "source": str(src),
                    "buffer_size": int(len(plr_buffer)),
                    "total_samples": int(plr_total_samples),
                    "new_samples": int(plr_new_samples),
                    "replay_samples": int(plr_replay_samples),
                    "replay_ratio": "" if _is_nan(plr_replay_ratio) else float(plr_replay_ratio),
                    "recent_replay_ratio_w20": "" if _is_nan(plr_recent_replay_ratio) else float(plr_recent_replay_ratio),
                    "topk": int(plr_topk),
                    "topk_covered": int(plr_topk_covered),
                    "topk_coverage": "" if _is_nan(plr_topk_coverage) else float(plr_topk_coverage),
                    "topk_sample_share": "" if _is_nan(plr_topk_sample_share) else float(plr_topk_sample_share),
                    "buffer_total_n_sampled": int(plr_total_n_sampled),
                    "entry_index": plr_ctx.get("entry_index", ""),
                    "entry_score_ema": plr_ctx.get("entry_score_ema", ""),
                    "p_new_iter": plr_ctx.get("p_new_iter", float(plr_p_new_iter)),
                    "recent_entropy": plr_ctx.get(
                        "recent_entropy",
                        "" if _is_nan(plr_recent_entropy) else float(plr_recent_entropy),
                    ),
                    "update_event": plr_update.get("event", ""),
                    "update_score_ema": plr_update.get("score_ema", ""),
                },
            )

        _append_csv_row(
            post_stage_dir / "outer_train_round.csv",
            [
                "iter_id",
                "phase",
                "ckpt_in",
                "ckpt_out",
                "learn_steps",
                "avg_reward",
                "action1_rate",
                "elapsed_s",
                "objective_score",
                "objective_formula",
                "J",
                "dJ",
                "B",
                "rho_eff",
                "rho_floor_gap",
                "D",
                "phase2_j_frozen",
                "phase2_j_frozen_source",
                "phase2_challenge",
                "phase2_g_minority",
                "phase2_too_hard_sq",
                "phase2_term_challenge",
                "phase2_term_minority",
                "phase2_term_lp_dj",
                "phase2_term_lp_j",
                "phase2_term_too_hard",
                "phase2_term_feasibility",
                "phase2_j_low",
                "phase2_w_a",
                "phase2_w_m",
                "phase2_w_too",
                "v4_learnability",
                "v4_entropy",
                "v4_novelty",
                "v4_n_seen",
                "v4_n_sampled",
                "v4_entry_score",
                "v4_p_new_iter",
                "v4_recent_entropy",
                "saber_v0_learnability",
                "saber_v0_novelty",
                "saber_v0_n_seen",
                "saber_v0_n_sampled",
                "saber_v0_entry_score",
                "saber_v0_term_lp_dj",
                "saber_v0_term_lp_j",
                "saber_v0_term_novelty",
                "saber_v0_term_feasibility",
                "saber_v1_learnability",
                "saber_v1_novelty",
                "saber_v1_n_seen",
                "saber_v1_n_sampled",
                "saber_v1_entry_score",
                "saber_v1_hard_count",
                "saber_v1_easy_count",
                "saber_v1_q_hard",
                "saber_v1_r_hard",
                "saber_v1_p_easy",
                "saber_v1_m_ins",
                "saber_v1_hard_action1_rate",
                "saber_v1_hard_wait_share",
                "saber_v1_easy_action1_rate",
                "saber_v1_dq_hard",
                "saber_v1_term_q_hard",
                "saber_v1_term_r_hard",
                "saber_v1_term_p_easy",
                "saber_v1_term_dq_hard",
                "saber_v1_term_novelty",
                "saber_v1_no_success_flag",
                "saber_v1_term_no_success",
                "saber_v1_term_feasibility",
                "saber_v1_success_trigger",
                "saber_v1_sticky_active",
                "saber_v1_sticky_remaining_before",
                "saber_v1_sticky_remaining_after",
                "saber_v1_sticky_trigger_iter",
                "saber_v1_budget_boost",
                "saber_v1_budget_num_files",
                "path_missing",
                "path_total",
                "path_read_ok",
                "path_read_rate",
                "inner_stop_mode",
                "table_budget_n",
                "processed_tables",
                "processed_tables_dynamic",
                "stop_reason",
                "policy_mode",
                "objective_mode",
                "policy_action_id",
                "policy_action_prob",
                "policy_score",
                "policy_post_mean",
                "policy_post_var",
                "policy_advantage",
                "policy_baseline",
                "policy_entropy",
                "curriculum_enabled",
                "curriculum_alpha",
                "curriculum_outer_files",
                "curriculum_base_files",
                "curriculum_replay_files",
                "train_data_root",
                "inner_ppo_new_ent_coef",
                "action0_rate",
                "minority_action",
                "minority_rate",
                "action_source",
                "plr_buffer_size",
                "plr_entry_index",
                "plr_entry_score_ema",
                "plr_event",
                "plr_score_ema",
            ],
            {
                "iter_id": int(iter_idx),
                "phase": str(iter_phase),
                "ckpt_in": "" if ckpt_in_path is None else str(ckpt_in_path),
                "ckpt_out": str(iter_ckpt),
                "learn_steps": int(new_training_rows),
                "avg_reward": "" if _is_nan(avg_reward) else float(avg_reward),
                "action1_rate": "" if _is_nan(action1_rate) else float(action1_rate),
                "elapsed_s": float(elapsed),
                "objective_score": float(objective),
                "objective_formula": str(objective_formula),
                "J": float(j_val),
                "dJ": float(dj_val),
                "B": float(b_val),
                "rho_eff": float(rho_eff),
                "rho_floor_gap": float(rho_floor_gap),
                "D": float(d_penalty),
                "phase2_j_frozen": float(phase2_j_frozen),
                "phase2_j_frozen_source": str(phase2_j_frozen_source),
                "phase2_challenge": float(phase2_challenge),
                "phase2_g_minority": float(phase2_g_minority),
                "phase2_too_hard_sq": float(phase2_too_hard),
                "phase2_term_challenge": float(phase2_term_challenge),
                "phase2_term_minority": float(phase2_term_minority),
                "phase2_term_lp_dj": float(phase2_term_lp_dj),
                "phase2_term_lp_j": float(phase2_term_lp_j),
                "phase2_term_too_hard": float(phase2_term_too_hard),
                "phase2_term_feasibility": float(phase2_term_feasibility),
                "phase2_j_low": float(phase2_j_low),
                "phase2_w_a": float(args.phase2_hard_weight),
                "phase2_w_m": float(phase2_minority_reward_weight),
                "phase2_w_too": 0.0,
                "v4_learnability": float(phase2_v4_learnability),
                "v4_entropy": float(phase2_v4_entropy),
                "v4_novelty": float(phase2_v4_novelty),
                "v4_n_seen": int(phase2_v4_n_seen),
                "v4_n_sampled": int(phase2_v4_n_sampled),
                "v4_entry_score": float(phase2_v4_entry_score),
                "v4_p_new_iter": float(plr_p_new_iter),
                "v4_recent_entropy": "" if _is_nan(plr_recent_entropy) else float(plr_recent_entropy),
                "saber_v0_learnability": float(saber_v0_learnability),
                "saber_v0_novelty": float(saber_v0_novelty),
                "saber_v0_n_seen": int(saber_v0_n_seen),
                "saber_v0_n_sampled": int(saber_v0_n_sampled),
                "saber_v0_entry_score": float(saber_v0_entry_score),
                "saber_v0_term_lp_dj": float(saber_v0_term_lp_dj),
                "saber_v0_term_lp_j": float(saber_v0_term_lp_j),
                "saber_v0_term_novelty": float(saber_v0_term_novelty),
                "saber_v0_term_feasibility": float(saber_v0_term_feasibility),
                "saber_v1_learnability": float(saber_v1_learnability),
                "saber_v1_novelty": float(saber_v1_novelty),
                "saber_v1_n_seen": int(saber_v1_n_seen),
                "saber_v1_n_sampled": int(saber_v1_n_sampled),
                "saber_v1_entry_score": float(saber_v1_entry_score),
                "saber_v1_hard_count": int(saber_v1_hard_count),
                "saber_v1_easy_count": int(saber_v1_easy_count),
                "saber_v1_q_hard": "" if _is_nan(saber_v1_q_hard) else float(saber_v1_q_hard),
                "saber_v1_r_hard": "" if _is_nan(saber_v1_r_hard) else float(saber_v1_r_hard),
                "saber_v1_p_easy": "" if _is_nan(saber_v1_p_easy) else float(saber_v1_p_easy),
                "saber_v1_m_ins": "" if _is_nan(saber_v1_m_ins) else float(saber_v1_m_ins),
                "saber_v1_hard_action1_rate": "" if _is_nan(saber_v1_hard_action1_rate) else float(saber_v1_hard_action1_rate),
                "saber_v1_hard_wait_share": "" if _is_nan(saber_v1_hard_wait_share) else float(saber_v1_hard_wait_share),
                "saber_v1_easy_action1_rate": "" if _is_nan(saber_v1_easy_action1_rate) else float(saber_v1_easy_action1_rate),
                "saber_v1_dq_hard": float(saber_v1_dq_hard),
                "saber_v1_term_q_hard": float(saber_v1_term_q_hard),
                "saber_v1_term_r_hard": float(saber_v1_term_r_hard),
                "saber_v1_term_p_easy": float(saber_v1_term_p_easy),
                "saber_v1_term_dq_hard": float(saber_v1_term_dq_hard),
                "saber_v1_term_novelty": float(saber_v1_term_novelty),
                "saber_v1_no_success_flag": float(saber_v1_no_success_flag),
                "saber_v1_term_no_success": float(saber_v1_term_no_success),
                "saber_v1_term_feasibility": float(saber_v1_term_feasibility),
                "saber_v1_success_trigger": int(saber_v1_success_trigger),
                "saber_v1_sticky_active": int(saber_v1_sticky_active),
                "saber_v1_sticky_remaining_before": int(saber_v1_sticky_remaining_before),
                "saber_v1_sticky_remaining_after": int(saber_v1_sticky_remaining),
                "saber_v1_sticky_trigger_iter": int(saber_v1_sticky_trigger_iter),
                "saber_v1_budget_boost": float(saber_v1_budget_boost),
                "saber_v1_budget_num_files": int(saber_v1_budget_num_files),
                "path_missing": int(missing_paths),
                "path_total": int(len(new_path_rows)),
                "path_read_ok": int(read_ok_paths),
                "path_read_rate": float(read_rate),
                "inner_stop_mode": str(args.inner_stop_mode),
                "table_budget_n": int(inner_budget_n),
                "processed_tables": int(processed_tables),
                "processed_tables_dynamic": int(processed_tables_dynamic),
                "stop_reason": str(stop_reason),
                "policy_mode": str(policy_mode),
                "objective_mode": str(objective_mode),
                "policy_action_id": pg_action_id,
                "policy_action_prob": pg_action_prob,
                "policy_score": policy_score,
                "policy_post_mean": policy_post_mean,
                "policy_post_var": policy_post_var,
                "policy_advantage": pg_advantage,
                "policy_baseline": pg_baseline,
                "policy_entropy": pg_entropy,
                "curriculum_enabled": int(phase_curriculum_enabled),
                "curriculum_alpha": "" if _is_nan(float(curriculum_alpha)) else float(curriculum_alpha),
                "curriculum_outer_files": int(curriculum_outer_files),
                "curriculum_base_files": int(curriculum_base_files),
                "curriculum_replay_files": int(curriculum_replay_files),
                "train_data_root": str(train_data_root),
                "inner_ppo_new_ent_coef": float(inner_ppo_new_ent_coef),
                "action0_rate": "" if _is_nan(action0_rate) else float(action0_rate),
                "minority_action": str(minority_action),
                "minority_rate": "" if _is_nan(minority_rate) else float(minority_rate),
                "action_source": (
                    str(plr_ctx.get("source", "policy"))
                    if (iter_level_replay_enabled and str(iter_action_source) == "policy")
                    else str(iter_action_source)
                ),
                "plr_buffer_size": int(len(plr_buffer)),
                "plr_entry_index": plr_ctx.get("entry_index", ""),
                "plr_entry_score_ema": plr_ctx.get("entry_score_ema", ""),
                "plr_event": plr_update.get("event", ""),
                "plr_score_ema": plr_update.get("score_ema", ""),
            },
        )
        if policy_mode == "rarl_dqn" and not iter_level_replay_enabled:
            _append_csv_row(
                rarl_stats_csv,
                [
                    "iter_id",
                    "phase",
                    "action_id",
                    "epsilon",
                    "explore",
                    "q_selected",
                    "q_max",
                    "policy_reward",
                    "dqn_updates",
                    "dqn_loss",
                    "replay_size",
                    "objective_score",
                    "J",
                    "dJ",
                    "policy_entropy",
                    "recent_j_mean",
                    "recent_action1_mean",
                    "recent_entropy_mean",
                    "minority_rate",
                ],
                {
                    "iter_id": int(iter_idx),
                    "phase": str(iter_phase),
                    "action_id": int(rarl_ctx.get("action_idx", -1)),
                    "epsilon": float(_safe_float(rarl_ctx.get("epsilon", float("nan")))),
                    "explore": int(_safe_int(rarl_ctx.get("explore", 0), default=0)),
                    "q_selected": float(_safe_float(rarl_ctx.get("q_selected", float("nan")))),
                    "q_max": float(_safe_float(rarl_ctx.get("q_max", float("nan")))),
                    "policy_reward": float(_safe_float(rarl_ctx.get("policy_reward", float("nan")))),
                    "dqn_updates": float(_safe_float(rarl_train_ctx.get("updates", 0.0))),
                    "dqn_loss": float(_safe_float(rarl_train_ctx.get("loss", float("nan")))),
                    "replay_size": int(len(list(rarl_dqn_state.get("replay", [])))),
                    "objective_score": float(objective),
                    "J": float(j_val),
                    "dJ": float(dj_val),
                    "policy_entropy": float(iter_policy_entropy),
                    "recent_j_mean": float(_safe_mean(list(rarl_recent_j), default=j_val)),
                    "recent_action1_mean": float(_safe_mean(list(rarl_recent_action1), default=action1_rate)),
                    "recent_entropy_mean": float(_safe_mean(list(rarl_recent_entropy), default=iter_policy_entropy)),
                    "minority_rate": float(rho_eff),
                },
            )
        _update_action_objective(
            actions_csv,
            int(iter_idx),
            float(objective),
            extra_updates={
                "phase": str(iter_phase),
                "objective_mode": str(objective_mode),
                "objective_formula": str(objective_formula),
                "J": float(j_val),
                "dJ": float(dj_val),
                "B": float(b_val),
                "D": float(d_penalty),
                "phase2_j_frozen": float(phase2_j_frozen),
                "phase2_j_frozen_source": str(phase2_j_frozen_source),
                "phase2_challenge": float(phase2_challenge),
                "phase2_g_minority": float(phase2_g_minority),
                "phase2_too_hard_sq": float(phase2_too_hard),
                "phase2_term_challenge": float(phase2_term_challenge),
                "phase2_term_minority": float(phase2_term_minority),
                "phase2_term_too_hard": float(phase2_term_too_hard),
                "phase2_term_lp_dj": float(phase2_term_lp_dj),
                "phase2_term_lp_j": float(phase2_term_lp_j),
                "phase2_term_feasibility": float(phase2_term_feasibility),
                "phase2_j_low": float(phase2_j_low),
                "phase2_w_a": float(args.phase2_hard_weight),
                "phase2_w_m": float(phase2_minority_reward_weight),
                "phase2_w_too": 0.0,
                "v4_learnability": float(phase2_v4_learnability),
                "v4_entropy": float(phase2_v4_entropy),
                "v4_novelty": float(phase2_v4_novelty),
                "v4_n_seen": int(phase2_v4_n_seen),
                "v4_n_sampled": int(phase2_v4_n_sampled),
                "v4_entry_score": float(phase2_v4_entry_score),
                "v4_p_new_iter": float(plr_p_new_iter),
                "v4_recent_entropy": "" if _is_nan(plr_recent_entropy) else float(plr_recent_entropy),
                "saber_v0_learnability": float(saber_v0_learnability),
                "saber_v0_novelty": float(saber_v0_novelty),
                "saber_v0_n_seen": int(saber_v0_n_seen),
                "saber_v0_n_sampled": int(saber_v0_n_sampled),
                "saber_v0_entry_score": float(saber_v0_entry_score),
                "saber_v0_term_lp_dj": float(saber_v0_term_lp_dj),
                "saber_v0_term_lp_j": float(saber_v0_term_lp_j),
                "saber_v0_term_novelty": float(saber_v0_term_novelty),
                "saber_v0_term_feasibility": float(saber_v0_term_feasibility),
                "saber_v1_learnability": float(saber_v1_learnability),
                "saber_v1_novelty": float(saber_v1_novelty),
                "saber_v1_n_seen": int(saber_v1_n_seen),
                "saber_v1_n_sampled": int(saber_v1_n_sampled),
                "saber_v1_entry_score": float(saber_v1_entry_score),
                "saber_v1_hard_count": int(saber_v1_hard_count),
                "saber_v1_easy_count": int(saber_v1_easy_count),
                "saber_v1_q_hard": "" if _is_nan(saber_v1_q_hard) else float(saber_v1_q_hard),
                "saber_v1_r_hard": "" if _is_nan(saber_v1_r_hard) else float(saber_v1_r_hard),
                "saber_v1_p_easy": "" if _is_nan(saber_v1_p_easy) else float(saber_v1_p_easy),
                "saber_v1_m_ins": "" if _is_nan(saber_v1_m_ins) else float(saber_v1_m_ins),
                "saber_v1_hard_action1_rate": "" if _is_nan(saber_v1_hard_action1_rate) else float(saber_v1_hard_action1_rate),
                "saber_v1_hard_wait_share": "" if _is_nan(saber_v1_hard_wait_share) else float(saber_v1_hard_wait_share),
                "saber_v1_easy_action1_rate": "" if _is_nan(saber_v1_easy_action1_rate) else float(saber_v1_easy_action1_rate),
                "saber_v1_dq_hard": float(saber_v1_dq_hard),
                "saber_v1_term_q_hard": float(saber_v1_term_q_hard),
                "saber_v1_term_r_hard": float(saber_v1_term_r_hard),
                "saber_v1_term_p_easy": float(saber_v1_term_p_easy),
                "saber_v1_term_dq_hard": float(saber_v1_term_dq_hard),
                "saber_v1_term_novelty": float(saber_v1_term_novelty),
                "saber_v1_no_success_flag": float(saber_v1_no_success_flag),
                "saber_v1_term_no_success": float(saber_v1_term_no_success),
                "saber_v1_term_feasibility": float(saber_v1_term_feasibility),
                "saber_v1_success_trigger": int(saber_v1_success_trigger),
                "saber_v1_sticky_active": int(saber_v1_sticky_active),
                "saber_v1_sticky_remaining_before": int(saber_v1_sticky_remaining_before),
                "saber_v1_sticky_remaining_after": int(saber_v1_sticky_remaining),
                "saber_v1_sticky_trigger_iter": int(saber_v1_sticky_trigger_iter),
                "saber_v1_budget_boost": float(saber_v1_budget_boost),
                "saber_v1_budget_num_files": int(saber_v1_budget_num_files),
                "inner_ppo_new_ent_coef": float(inner_ppo_new_ent_coef),
            },
        )
        print(
            f"[OUTER][TRAIN] iter={iter_tag} rows={new_training_rows} "
            f"avg_reward={avg_reward:.6f} action0_rate={action0_rate:.6f} "
            f"action1_rate={action1_rate:.6f} minority={minority_action}:{rho_eff:.6f} "
            f"inner_ent={float(inner_ppo_new_ent_coef):.4f} "
            f"elapsed={elapsed:.2f}s "
            f"processed_tables={processed_tables} processed_tables_dynamic={processed_tables_dynamic} "
            f"stop_reason={stop_reason}"
        )
        if objective_formula in {"phase23_difficulty_minor_v1", "phase23_difficulty_minor_v3_lp", "phase23_plr_ued_v4", "saber_v0", "saber_v1"}:
            if objective_formula == "phase23_plr_ued_v4":
                print(
                    f"[OUTER][CHECK] iter={iter_tag} objective={objective:.6f} "
                    f"(J_frozen={phase2_j_frozen:.6f},challenge={phase2_challenge:.6f},"
                    f"L={phase2_v4_learnability:.6f},|dJ|={abs(float(dj_val)):.6f},"
                    f"H={phase2_v4_entropy:.6f},G_minority={phase2_g_minority:.6f},"
                    f"novelty={phase2_v4_novelty:.6f},D={phase2_term_feasibility:.6f}) "
                    f"path_missing={missing_paths}/{len(new_path_rows)} "
                    f"path_read_ok={read_ok_paths}/{len(new_path_rows)}"
                )
            elif objective_formula == "phase23_difficulty_minor_v3_lp":
                print(
                    f"[OUTER][CHECK] iter={iter_tag} objective={objective:.6f} "
                    f"(J_frozen={phase2_j_frozen:.6f},challenge={phase2_challenge:.6f},"
                    f"G_minority={phase2_g_minority:.6f},LP_dJ={phase2_term_lp_dj:.6f},"
                    f"LP_J={phase2_term_lp_j:.6f},D={phase2_term_feasibility:.6f}) "
                    f"path_missing={missing_paths}/{len(new_path_rows)} "
                    f"path_read_ok={read_ok_paths}/{len(new_path_rows)}"
                )
            elif objective_formula == "saber_v0":
                print(
                    f"[OUTER][CHECK] iter={iter_tag} objective={objective:.6f} "
                    f"(J_frozen={phase2_j_frozen:.6f},L={saber_v0_learnability:.6f},"
                    f"|dJ|={abs(float(dj_val)):.6f},J={j_val:.6f},"
                    f"novelty={saber_v0_novelty:.6f},D={saber_v0_term_feasibility:.6f}) "
                    f"path_missing={missing_paths}/{len(new_path_rows)} "
                    f"path_read_ok={read_ok_paths}/{len(new_path_rows)}"
                )
            elif objective_formula == "saber_v1":
                print(
                    f"[OUTER][CHECK] iter={iter_tag} objective={objective:.6f} "
                    f"(J_frozen={phase2_j_frozen:.6f},L={saber_v1_learnability:.6f},"
                    f"Q_hard={_finite_or(saber_v1_q_hard, float('nan')):.6f},"
                    f"R_hard={_finite_or(saber_v1_r_hard, float('nan')):.6f},"
                    f"P_easy={_finite_or(saber_v1_p_easy, float('nan')):.6f},"
                    f"|dQ|={abs(float(saber_v1_dq_hard)):.6f},"
                    f"novelty={saber_v1_novelty:.6f},hard_n={saber_v1_hard_count},D={saber_v1_term_feasibility:.6f}) "
                    f"path_missing={missing_paths}/{len(new_path_rows)} "
                    f"path_read_ok={read_ok_paths}/{len(new_path_rows)}"
                )
            else:
                print(
                    f"[OUTER][CHECK] iter={iter_tag} objective={objective:.6f} "
                    f"(J_frozen={phase2_j_frozen:.6f},challenge={phase2_challenge:.6f},"
                    f"G_minority={phase2_g_minority:.6f},D={phase2_term_feasibility:.6f}) "
                    f"path_missing={missing_paths}/{len(new_path_rows)} "
                    f"path_read_ok={read_ok_paths}/{len(new_path_rows)}"
                )
        else:
            print(
                f"[OUTER][CHECK] iter={iter_tag} objective={objective:.6f} "
                f"(J={j_val:.6f},dJ={dj_val:.6f},B={b_val:.6f},"
                f"rho_min={rho_eff:.6f}({minority_action}),"
                f"collapse_gap={collapse_gap:.6f},collapse_penalty={collapse_penalty:.6f},"
                f"rho_floor_gap={rho_floor_gap:.6f},"
                f"floor_soft={floor_soft_penalty:.6f},floor_hard={floor_hard_penalty:.6f},"
                f"D={d_penalty:.6f}) "
                f"path_missing={missing_paths}/{len(new_path_rows)} "
                f"path_read_ok={read_ok_paths}/{len(new_path_rows)}"
            )
        if saber_v1_mode and int(saber_v1_success_trigger) == 1:
            print(
                f"[OUTER][SABER-SUCCESS] iter={iter_tag} "
                f"R_hard={float(_finite_or(saber_v1_r_hard, 0.0)):.6f} "
                f"sticky_iters={int(saber_v1_sticky_remaining)} "
                f"signature={saber_v1_sticky_signature}"
            )
        if objective_formula in {"phase23_difficulty_minor_v1", "phase23_difficulty_minor_v3_lp", "phase23_plr_ued_v4", "saber_v0", "saber_v1"}:
            if objective_formula == "phase23_plr_ued_v4":
                print(
                    f"[OUTER][OBJ] iter={iter_tag} phase={iter_phase} "
                    f"objective = +w_c*(1-J_frozen)*L + w_lp*|dJ| + w_j*J + "
                    f"w_H*H(pi) + w_m*G_minority + w_n*novelty - D | "
                    f"J_frozen={phase2_j_frozen:.6f} source={phase2_j_frozen_source} "
                    f"(1-J_frozen)={phase2_challenge:.6f} L={phase2_v4_learnability:.6f} "
                    f"H={phase2_v4_entropy:.6f} G_minority={phase2_g_minority:.6f} "
                    f"novelty={phase2_v4_novelty:.6f}(n_seen={phase2_v4_n_seen},n_sampled={phase2_v4_n_sampled}) "
                    f"|dJ|={abs(float(dj_val)):.6f} J={j_val:.6f} D={phase2_term_feasibility:.6f} "
                    f"term_challenge={phase2_term_challenge:.6f} term_lp={phase2_term_lp_dj:.6f} "
                    f"term_j={phase2_term_lp_j:.6f} term_entropy={phase2_term_too_hard:.6f} "
                    f"term_minority={phase2_term_minority:.6f} term_novelty="
                    f"{float(args.edrl_v4_novelty_weight) * float(phase2_v4_novelty):.6f} "
                    f"w_c={float(args.edrl_v4_challenge_weight):.4f} "
                    f"w_lp={float(args.edrl_v4_lp_weight):.4f} "
                    f"w_j={float(args.edrl_v4_j_weight):.4f} "
                    f"w_H={float(args.edrl_v4_entropy_weight):.4f} "
                    f"w_m={float(args.edrl_v4_minority_weight):.4f} "
                    f"w_n={float(args.edrl_v4_novelty_weight):.4f} "
                    f"p_new_iter={float(plr_p_new_iter):.4f}"
                )
            elif objective_formula == "phase23_difficulty_minor_v3_lp":
                print(
                    f"[OUTER][OBJ] iter={iter_tag} phase={iter_phase} "
                    f"objective = +w_a*(1-J_frozen) + w_m*G_minority + w_dj*|dJ| + w_j*J - D | "
                    f"J_frozen={phase2_j_frozen:.6f} "
                    f"source={phase2_j_frozen_source} "
                    f"(1-J_frozen)={phase2_challenge:.6f} "
                    f"G_minority={phase2_g_minority:.6f} "
                    f"|dJ|={abs(float(dj_val)):.6f} "
                    f"J={j_val:.6f} "
                    f"D={phase2_term_feasibility:.6f} "
                    f"term_challenge={phase2_term_challenge:.6f} "
                    f"term_minority={phase2_term_minority:.6f} "
                    f"term_lp_dj={phase2_term_lp_dj:.6f} "
                    f"term_lp_j={phase2_term_lp_j:.6f} "
                    f"w_a={float(args.phase2_hard_weight):.4f} "
                    f"w_m={float(phase2_minority_reward_weight):.4f} "
                    f"w_dj={float(args.edrl_v3_dj_weight):.4f} "
                    f"w_j={float(args.edrl_v3_j_weight):.4f} "
                    f"w_too=0.0000"
                )
            elif objective_formula == "saber_v0":
                print(
                    f"[OUTER][OBJ] iter={iter_tag} phase={iter_phase} "
                    f"objective = L(J_frozen) * (w_dj*|dJ| + w_j*J + w_n*novelty) - D | "
                    f"J_frozen={phase2_j_frozen:.6f} source={phase2_j_frozen_source} "
                    f"L={saber_v0_learnability:.6f} "
                    f"|dJ|={abs(float(dj_val)):.6f} J={j_val:.6f} "
                    f"novelty={saber_v0_novelty:.6f}(n_seen={saber_v0_n_seen},n_sampled={saber_v0_n_sampled}) "
                    f"D={saber_v0_term_feasibility:.6f} "
                    f"term_lp_dj={saber_v0_term_lp_dj:.6f} "
                    f"term_j={saber_v0_term_lp_j:.6f} "
                    f"term_novelty={saber_v0_term_novelty:.6f} "
                    f"w_dj={float(args.saber_v0_dj_weight):.4f} "
                    f"w_j={float(args.saber_v0_j_weight):.4f} "
                    f"w_n={float(args.saber_v0_novelty_weight):.4f} "
                    f"j_center={float(args.saber_v0_j_center):.4f} "
                    f"j_sigma={float(args.saber_v0_j_sigma):.4f} "
                    f"p_new_iter={float(plr_p_new_iter):.4f}"
                )
            elif objective_formula == "saber_v1":
                print(
                    f"[OUTER][OBJ] iter={iter_tag} phase={iter_phase} "
                    f"objective = I(hard_n>0) * L(J_frozen) * (w_q*Q_hard + w_r*R_hard + "
                    f"w_e*P_easy + w_dq*|dQ_hard| + w_n*novelty) - w_ns*I(no_success)*wait_hard - D | "
                    f"J_frozen={phase2_j_frozen:.6f} source={phase2_j_frozen_source} "
                    f"L={saber_v1_learnability:.6f} hard_n={int(saber_v1_hard_count)} easy_n={int(saber_v1_easy_count)} "
                    f"Q_hard={_finite_or(saber_v1_q_hard, float('nan')):.6f} "
                    f"R_hard={_finite_or(saber_v1_r_hard, float('nan')):.6f} "
                    f"P_easy={_finite_or(saber_v1_p_easy, float('nan')):.6f} "
                    f"wait_hard={_finite_or(saber_v1_hard_wait_share, float('nan')):.6f} "
                    f"no_success={float(saber_v1_no_success_flag):.0f} "
                    f"|dQ_hard|={abs(float(saber_v1_dq_hard)):.6f} "
                    f"novelty={saber_v1_novelty:.6f}(n_seen={saber_v1_n_seen},n_sampled={saber_v1_n_sampled}) "
                    f"term_no_success={saber_v1_term_no_success:.6f} "
                    f"D={saber_v1_term_feasibility:.6f} "
                    f"term_q={saber_v1_term_q_hard:.6f} "
                    f"term_r={saber_v1_term_r_hard:.6f} "
                    f"term_easy={saber_v1_term_p_easy:.6f} "
                    f"term_dq={saber_v1_term_dq_hard:.6f} "
                    f"term_novelty={saber_v1_term_novelty:.6f} "
                    f"w_q={float(args.saber_v1_q_weight):.4f} "
                    f"w_r={float(args.saber_v1_r_weight):.4f} "
                    f"w_e={float(args.saber_v1_easy_weight):.4f} "
                    f"w_dq={float(args.saber_v1_dq_weight):.4f} "
                    f"w_n={float(args.saber_v1_novelty_weight):.4f} "
                    f"w_ns={float(args.saber_v1_no_success_weight):.4f} "
                    f"hard>={int(args.saber_v1_hard_threshold)} "
                    f"easy<={int(args.saber_v1_easy_threshold)} "
                    f"p_new_iter={float(plr_p_new_iter):.4f}"
                )
            else:
                print(
                    f"[OUTER][OBJ] iter={iter_tag} phase={iter_phase} "
                    f"objective = +w_a*(1-J_frozen) + w_m*G_minority - D | "
                    f"J_frozen={phase2_j_frozen:.6f} "
                    f"source={phase2_j_frozen_source} "
                    f"(1-J_frozen)={phase2_challenge:.6f} "
                    f"G_minority={phase2_g_minority:.6f} "
                    f"D={phase2_term_feasibility:.6f} "
                    f"term_challenge={phase2_term_challenge:.6f} "
                    f"term_minority={phase2_term_minority:.6f} "
                    f"w_a={float(args.phase2_hard_weight):.4f} "
                    f"w_m={float(phase2_minority_reward_weight):.4f} "
                    f"w_too=0.0000"
                )
        if iter_level_replay_enabled:
            print(
                f"[OUTER][PLR] iter={iter_tag} buffer_update={plr_update.get('event', 'na')} "
                f"score_ema={plr_update.get('score_ema', 'na')} "
                f"levels={len(plr_buffer)}"
            )
            print(
                f"[OUTER][PLR-STATS] iter={iter_tag} "
                f"samples(total={plr_total_samples},new={plr_new_samples},replay={plr_replay_samples},"
                f"replay_ratio={plr_replay_ratio:.3f},recent_w20={plr_recent_replay_ratio:.3f}) "
                f"top{plr_topk}_coverage={plr_topk_coverage:.3f} "
                f"topk_sample_share={plr_topk_sample_share:.3f} "
                f"p_new_iter={float(plr_p_new_iter):.3f}"
            )
        if policy_mode == "pg" and not iter_level_replay_enabled:
            print(
                f"[OUTER][PG] iter={iter_tag} advantage={float(pg_advantage):.6f} "
                f"baseline={float(pg_baseline):.6f}"
            )
        if policy_mode == "rarl_dqn" and not iter_level_replay_enabled:
            print(
                f"[OUTER][RARL] iter={iter_tag} "
                f"updates={int(_finite_or(_safe_float(rarl_train_ctx.get('updates', 0.0)), 0.0))} "
                f"loss={float(_safe_float(rarl_train_ctx.get('loss', float('nan')))):.6f} "
                f"replay={len(list(rarl_dqn_state.get('replay', [])))} "
                f"policy_reward={float(_safe_float(rarl_ctx.get('policy_reward', float('nan')))):.6f}"
            )
        phase_iter_count[iter_phase] = int(phase_iter_count.get(iter_phase, 0)) + 1
        phase_history.setdefault(iter_phase, []).append(
            {
                "iter_id": float(iter_idx),
                "objective": float(objective),
                "dJ": float(dj_val),
                "minority_rate": float(rho_eff),
            }
        )
        if str(iter_phase).strip().lower() == "phase2" and int(action_idx) >= 0 and _is_finite(float(objective)):
            item = phase2_topk_obj_agg.get(int(action_idx), {"sum": 0.0, "count": 0.0})
            item["sum"] = float(item["sum"]) + float(objective)
            item["count"] = float(item["count"]) + 1.0
            phase2_topk_obj_agg[int(action_idx)] = item
        policy_last_j = float(j_val)
        policy_last_dj = float(dj_val)
        policy_last_action0_rate = float(action0_rate)
        policy_last_action1_rate = float(action1_rate)
        policy_last_minority_rate = float(rho_eff)
        policy_last_objective = float(objective)
        policy_last_entropy = float(iter_policy_entropy)
        if _is_finite(saber_v1_q_hard):
            policy_last_saber_v1_q_hard = float(saber_v1_q_hard)
        if _is_finite(j_val):
            rarl_recent_j.append(float(j_val))
        if _is_finite(action1_rate):
            rarl_recent_action1.append(float(action1_rate))
        if _is_finite(iter_policy_entropy):
            rarl_recent_entropy.append(float(iter_policy_entropy))
        prev_avg_reward = avg_reward
        if iter_phase == "phase2" and phase2_freeze_inner:
            ckpt_prev = phase2_anchor_ckpt
        else:
            ckpt_prev = iter_ckpt

        phase_min_iters = int(args.phase2_min_iters) if iter_phase == "phase2" else int(args.phase3_min_iters)
        phase_max_iters = int(args.phase2_max_iters) if iter_phase == "phase2" else int(args.phase3_max_iters)
        phase_iters_done = int(phase_iter_count.get(iter_phase, 0))
        phase_max_abs_dj = float(args.converge_max_abs_dj)
        phase_max_obj_range = float(args.converge_max_obj_range)
        if iter_phase == "phase2":
            p2_abs = float(args.phase2_converge_max_abs_dj)
            p2_rng = float(args.phase2_converge_max_obj_range)
            if p2_abs > 0.0:
                phase_max_abs_dj = float(p2_abs)
            if p2_rng > 0.0:
                phase_max_obj_range = float(p2_rng)
        phase_patience = int(args.converge_patience) if iter_phase == "phase2" else int(args.phase3_converge_patience)
        phase_converged = _window_converged(
            history=phase_history.get(iter_phase, []),
            patience=max(1, int(phase_patience)),
            max_abs_dj=float(phase_max_abs_dj),
            max_obj_range=float(phase_max_obj_range),
            min_minority_rate=float(converge_minority_floor),
        )
        reached_phase_cap = phase_iters_done >= phase_max_iters

        if auto_phase and iter_phase == "phase2":
            if reached_phase_cap or (phase_iters_done >= phase_min_iters and phase_converged):
                reason = "phase2_max_iters" if reached_phase_cap else "phase2_converged"
                phase3_topk_ids = _compute_phase2_topk_action_ids_from_agg(
                    agg=phase2_topk_obj_agg,
                    k=int(phase3_topk_k),
                )
                phase3_topk_warm_remaining = int(phase3_topk_warmup_iters) if phase3_topk_ids else 0
                if phase3_topk_ids:
                    topk_desc = []
                    for aid in phase3_topk_ids:
                        if aid < 0 or aid >= len(action_space):
                            continue
                        sig = _action_signature(action_space[aid])
                        it = phase2_topk_obj_agg.get(aid, {"sum": 0.0, "count": 0.0})
                        cnt = float(it.get("count", 0.0))
                        mean_obj = (float(it.get("sum", 0.0)) / cnt) if cnt > 0.0 else float("nan")
                        topk_desc.append(f"{aid}:{sig}:mean={mean_obj:.4f}:n={cnt:.0f}")
                    print(
                        f"[OUTER][PHASE] phase2_topk(k={len(phase3_topk_ids)}) "
                        f"{'; '.join(topk_desc)}"
                    )
                    if policy_mode in {"ts", "ucb"} and phase3_topk_prior_count > 0.0:
                        target_state = ts_state if policy_mode == "ts" else ucb_state
                        _bootstrap_bandit_state_with_topk(
                            state=target_state,
                            action_space=action_space,
                            topk_action_ids=phase3_topk_ids,
                            agg=phase2_topk_obj_agg,
                            prior_count=float(phase3_topk_prior_count),
                        )
                        if policy_mode == "ts" and ts_state_path is not None:
                            _save_json_state(ts_state_path, ts_state)
                        if policy_mode == "ucb" and ucb_state_path is not None:
                            _save_json_state(ucb_state_path, ucb_state)
                        print(
                            f"[OUTER][PHASE] bootstrap {policy_mode} with phase2_topk priors "
                            f"(prior_count={float(phase3_topk_prior_count):.3f}, "
                            f"warm_iters={int(phase3_topk_warm_remaining)})"
                        )
                phase_mode = "phase3"
                print(
                    f"[OUTER][PHASE] switch phase2->phase3 at iter={iter_tag} "
                    f"reason={reason} phase2_iters={phase_iters_done}"
                )
                continue

        if outer_auto_stop:
            if reached_phase_cap:
                print(
                    f"[OUTER][STOP] phase={iter_phase} iter={iter_tag} "
                    f"reason=phase_max_iters({phase_max_iters})"
                )
                break
            if phase_iters_done >= phase_min_iters and phase_converged:
                print(
                    f"[OUTER][STOP] phase={iter_phase} iter={iter_tag} "
                    f"reason=converged window={max(1, int(phase_patience))}"
                )
                break

    print(f"[OUTER][DONE] requested_phase={requested_phase} final_phase={phase_mode} loop finished")


if __name__ == "__main__":
    main()

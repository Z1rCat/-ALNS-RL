#!/usr/bin/env python3
"""
Upper-bound probe for PPO_NEW (v1/v2) trace logs.

Goals:
1) Build supervised datasets from rl_trace.csv to predict immediate reward (0/1).
2) Support two reconstruction modes:
   - A: strict alignment with current PPO_NEW v2 env input timing.
   - B: cross-sample rolling variant as optimistic reference.
3) Report per-run metrics and pooled metrics (with scenario_id feature in pooled).

The script is diagnostic-only and does not modify RL training.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import warnings
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SKLEARN_AVAILABLE = True
SKLEARN_IMPORT_ERROR = ""
try:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except Exception as exc:
    SKLEARN_AVAILABLE = False
    SKLEARN_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    ConvergenceWarning = Warning  # type: ignore[assignment]
    LogisticRegression = None  # type: ignore[assignment]
    MLPClassifier = None  # type: ignore[assignment]
    Pipeline = None  # type: ignore[assignment]
    StandardScaler = None  # type: ignore[assignment]

TORCH_AVAILABLE = True
TORCH_IMPORT_ERROR = ""
try:
    import torch
    import torch.nn as nn
except Exception as exc:
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


BEGIN_TO_STAGE_BIT = {
    "begin_removal": 0.0,
    "begin_insertion": 1.0,
}


@dataclass
class Sample:
    run_id: str
    scenario_id: str
    mode: str
    feature_kind: str
    phase: str
    table_number: int
    action: int
    reward: int
    x: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upper-bound probe for PPO_NEW trace logs")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="run directory or parent directory containing multiple run_* folders",
    )
    parser.add_argument(
        "--feature-kind",
        type=str,
        default="Xt",
        choices=["xt", "Xt", "both"],
        help="use x_t, X_t, or both",
    )
    parser.add_argument(
        "--n-stack",
        type=int,
        default=4,
        help="window size for X_t stacking (default: 4)",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default="phase_table",
        choices=["phase_table", "table_only", "phase_only"],
        help="dataset split rule",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for reproducibility",
    )
    parser.add_argument(
        "--min-action-samples",
        type=int,
        default=20,
        help="minimum samples per action for optimistic proxy",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="A,B",
        help="reconstruction modes, comma-separated: A,B",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default="probe_report.json",
        help="per-run report file name",
    )
    parser.add_argument(
        "--pooled-report-name",
        type=str,
        default="probe_report_pooled.json",
        help="pooled report file name (saved under input root)",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if text == "":
            return None
        return float(text)
    except Exception:
        return None


def to_int(value: Any) -> Optional[int]:
    val = to_float(value)
    if val is None:
        return None
    try:
        return int(round(val))
    except Exception:
        return None


def safe_phase(value: Any) -> str:
    return str(value or "").strip().lower()


def parse_obs_delay_severity(row: Dict[str, Any]) -> Optional[np.ndarray]:
    delay = to_float(row.get("delay_tolerance", ""))
    severity = to_float(row.get("severity", ""))
    if delay is None or severity is None:
        return None
    return np.asarray([delay, severity], dtype=np.float32)


def discover_run_dirs(input_path: Path) -> List[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"input path does not exist: {input_path}")
    if input_path.is_dir() and (input_path / "rl_trace.csv").exists():
        return [input_path.resolve()]
    if not input_path.is_dir():
        raise ValueError(f"input path must be a directory: {input_path}")
    run_dirs: List[Path] = []
    for child in sorted(input_path.iterdir()):
        if child.is_dir() and (child / "rl_trace.csv").exists():
            run_dirs.append(child.resolve())
    if not run_dirs:
        raise FileNotFoundError(f"no run folders with rl_trace.csv found under: {input_path}")
    return run_dirs


def load_scenario_id(run_dir: Path) -> str:
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            scenario = str(payload.get("distribution", "")).strip()
            if scenario:
                return scenario
        except Exception:
            pass
    name = run_dir.name
    for token in ["O_10_60", "O_10_30", "F1_10_30", "F1_10_60"]:
        if token in name:
            return token
    return "UNKNOWN"


def iter_trace_rows(trace_path: Path) -> Iterable[Dict[str, Any]]:
    with trace_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, restkey="__extra__")
        for row in reader:
            yield row


def compose_xt(
    obs_now: np.ndarray,
    prev_obs: np.ndarray,
    stage_bit: float,
    prev_action: float,
    prev_reward: float,
) -> np.ndarray:
    obs_now = np.asarray(obs_now, dtype=np.float32).reshape(-1)
    prev_obs = np.asarray(prev_obs, dtype=np.float32).reshape(-1)
    delta = (obs_now - prev_obs).astype(np.float32)
    return np.concatenate(
        [
            obs_now,
            np.asarray([stage_bit], dtype=np.float32),
            np.asarray([prev_action], dtype=np.float32),
            np.asarray([prev_reward], dtype=np.float32),
            delta,
        ],
        axis=0,
    ).astype(np.float32)


def stack_xt(x_now: np.ndarray, history: Sequence[np.ndarray], k: int) -> np.ndarray:
    x_now = np.asarray(x_now, dtype=np.float32).reshape(-1)
    k = max(1, int(k))
    if k == 1:
        return x_now
    frames: List[np.ndarray] = [x_now]
    for item in history[: k - 1]:
        frames.append(np.asarray(item, dtype=np.float32).reshape(-1))
    while len(frames) < k:
        frames.append(x_now.copy())
    return np.concatenate(frames, axis=0).astype(np.float32)


def build_samples_for_run(
    run_dir: Path,
    mode: str,
    feature_kind: str,
    n_stack: int,
) -> Tuple[List[Sample], Dict[str, Any]]:
    mode = mode.upper()
    if mode not in {"A", "B"}:
        raise ValueError(f"unsupported mode: {mode}")
    feature_kind = "Xt" if feature_kind == "Xt" else "xt"
    n_stack = max(1, int(n_stack))
    if feature_kind == "xt":
        n_stack = 1

    run_id = run_dir.name
    scenario_id = load_scenario_id(run_dir)
    trace_path = run_dir / "rl_trace.csv"

    stats: Dict[str, Any] = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "mode": mode,
        "feature_kind": feature_kind,
        "n_stack": n_stack,
        "rows_total": 0,
        "rows_receive_reward": 0,
        "rows_kept": 0,
        "dropped_missing_stage_bit": 0,
        "dropped_missing_obs_now": 0,
        "dropped_invalid_action_reward": 0,
        "fallback_prev_obs_count": 0,
    }

    stage_bit_by_key: Dict[Tuple[str, str], float] = {}
    begin_obs_by_key: Dict[Tuple[str, str], np.ndarray] = {}

    rolling_state: Dict[Tuple[str, str], Dict[str, Any]] = {}

    samples: List[Sample] = []

    for row in iter_trace_rows(trace_path):
        stats["rows_total"] += 1
        stage = str(row.get("stage", "")).strip()
        request = str(row.get("request", "")).strip()
        vehicle = str(row.get("vehicle", "")).strip()
        key = (request, vehicle)

        if stage in BEGIN_TO_STAGE_BIT:
            stage_bit_by_key[key] = BEGIN_TO_STAGE_BIT[stage]
            begin_obs = parse_obs_delay_severity(row)
            if begin_obs is not None:
                begin_obs_by_key[key] = begin_obs
            continue

        if stage != "receive_reward":
            continue

        stats["rows_receive_reward"] += 1

        phase = safe_phase(row.get("phase", ""))
        table_number = to_int(row.get("table_number", ""))
        action = to_int(row.get("action", ""))
        reward = to_int(row.get("reward", ""))
        obs_now = parse_obs_delay_severity(row)
        stage_bit = stage_bit_by_key.get(key, None)

        if obs_now is None:
            stats["dropped_missing_obs_now"] += 1
            continue
        if action not in (0, 1) or reward not in (0, 1) or table_number is None:
            stats["dropped_invalid_action_reward"] += 1
            continue
        if stage_bit is None:
            stats["dropped_missing_stage_bit"] += 1
            continue

        begin_obs = begin_obs_by_key.get(key, None)
        if begin_obs is None:
            begin_obs = obs_now.copy()
            stats["fallback_prev_obs_count"] += 1

        if mode == "A":
            # Strict PPO_NEW v2 alignment:
            # prev_action/prev_reward reset to 0 (episode_length=1 path),
            # prev_obs from begin state of the same (request, vehicle) trajectory.
            x_now = compose_xt(
                obs_now=obs_now,
                prev_obs=begin_obs,
                stage_bit=float(stage_bit),
                prev_action=0.0,
                prev_reward=0.0,
            )
            if feature_kind == "xt":
                feat = x_now
            else:
                x0 = compose_xt(
                    obs_now=begin_obs,
                    prev_obs=begin_obs,
                    stage_bit=float(stage_bit),
                    prev_action=0.0,
                    prev_reward=0.0,
                )
                frames = [x_now] + [x0.copy() for _ in range(n_stack - 1)]
                feat = np.concatenate(frames, axis=0).astype(np.float32)
        else:
            # Cross-sample rolling reference over (request, vehicle) trajectory.
            state = rolling_state.get(key)
            if state is None:
                prev_obs = begin_obs.copy()
                prev_action = 0.0
                prev_reward = 0.0
                x_hist: Deque[np.ndarray] = deque(maxlen=max(1, n_stack - 1))
            else:
                prev_obs = np.asarray(state["prev_obs"], dtype=np.float32)
                prev_action = float(state["prev_action"])
                prev_reward = float(state["prev_reward"])
                x_hist = state["x_hist"]

            x_now = compose_xt(
                obs_now=obs_now,
                prev_obs=prev_obs,
                stage_bit=float(stage_bit),
                prev_action=prev_action,
                prev_reward=prev_reward,
            )
            if feature_kind == "xt":
                feat = x_now
            else:
                feat = stack_xt(x_now, list(x_hist), n_stack)

            x_hist.appendleft(x_now.copy())
            rolling_state[key] = {
                "prev_obs": obs_now.copy(),
                "prev_action": float(action),
                "prev_reward": float(reward),
                "x_hist": x_hist,
            }

        sample = Sample(
            run_id=run_id,
            scenario_id=scenario_id,
            mode=mode,
            feature_kind=feature_kind,
            phase=phase,
            table_number=int(table_number),
            action=int(action),
            reward=int(reward),
            x=feat.astype(np.float32),
        )
        samples.append(sample)
        stats["rows_kept"] += 1

    return samples, stats


def split_indices(samples: Sequence[Sample], split_mode: str) -> Tuple[np.ndarray, np.ndarray]:
    train_idx: List[int] = []
    test_idx: List[int] = []
    for i, s in enumerate(samples):
        if split_mode == "phase_table":
            if s.phase == "train" and 0 <= s.table_number <= 799:
                train_idx.append(i)
            elif s.phase == "implement" and 800 <= s.table_number <= 999:
                test_idx.append(i)
        elif split_mode == "table_only":
            if 0 <= s.table_number <= 799:
                train_idx.append(i)
            elif 800 <= s.table_number <= 999:
                test_idx.append(i)
        elif split_mode == "phase_only":
            if s.phase == "train":
                train_idx.append(i)
            elif s.phase == "implement":
                test_idx.append(i)
        else:
            raise ValueError(f"unsupported split_mode: {split_mode}")
    return np.asarray(train_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)


def make_matrix_and_labels(
    samples: Sequence[Sample],
    indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
    if indices.size == 0:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            [],
            np.zeros((0,), dtype=np.int64),
        )
    feats = [samples[i].x for i in indices.tolist()]
    x = np.vstack(feats).astype(np.float32)
    y = np.asarray([samples[i].reward for i in indices.tolist()], dtype=np.int64)
    a = np.asarray([samples[i].action for i in indices.tolist()], dtype=np.int64)
    scenarios = [samples[i].scenario_id for i in indices.tolist()]
    tables = np.asarray([samples[i].table_number for i in indices.tolist()], dtype=np.int64)
    return x, y, a, scenarios, tables


def add_scenario_onehot(
    x: np.ndarray,
    scenarios: Sequence[str],
    scenario_vocab: Sequence[str],
) -> np.ndarray:
    if x.shape[0] == 0:
        return x
    if not scenario_vocab:
        return x
    vocab = list(scenario_vocab)
    idx_map = {name: i for i, name in enumerate(vocab)}
    onehot = np.zeros((x.shape[0], len(vocab)), dtype=np.float32)
    for i, sid in enumerate(scenarios):
        j = idx_map.get(sid, None)
        if j is not None:
            onehot[i, j] = 1.0
    return np.concatenate([x, onehot], axis=1).astype(np.float32)


def with_action_feature(x: np.ndarray, action: np.ndarray) -> np.ndarray:
    if x.shape[0] != action.shape[0]:
        raise ValueError("x and action row count mismatch")
    return np.concatenate([x, action.reshape(-1, 1).astype(np.float32)], axis=1).astype(np.float32)


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(np.mean((y_true == y_pred).astype(np.float32)))


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    classes = np.unique(y_true)
    if classes.size < 2:
        return None
    recalls: List[float] = []
    for c in [0, 1]:
        mask = y_true == c
        if int(mask.sum()) == 0:
            return None
        recalls.append(float(np.mean((y_pred[mask] == c).astype(np.float32))))
    return float(np.mean(recalls))


def _roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, y_score.size + 1, dtype=np.float64)
    # tie handling: average ranks within ties
    sorted_scores = y_score[order]
    i = 0
    while i < y_score.size:
        j = i + 1
        while j < y_score.size and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg_rank = float(np.mean(np.arange(i + 1, j + 1, dtype=np.float64)))
            ranks[order[i:j]] = avg_rank
        i = j
    rank_sum_pos = float(np.sum(ranks[y_true == 1]))
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _average_precision_binary(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return None
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1).astype(np.float64)
    fp = np.cumsum(y_sorted == 0).astype(np.float64)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / float(n_pos)
    # AP = sum over threshold steps of precision * delta_recall
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    delta = recall - recall_prev
    ap = float(np.sum(precision * delta))
    return ap


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray]) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    out: Dict[str, Any] = {
        "n": int(y_true.size),
        "positive_rate": float(np.mean(y_true)) if y_true.size else None,
        "accuracy": _accuracy(y_true, y_pred) if y_true.size else None,
        "balanced_accuracy": None,
        "roc_auc": None,
        "pr_auc": None,
    }
    if y_true.size == 0:
        return out
    classes = np.unique(y_true)
    if classes.size >= 2:
        out["balanced_accuracy"] = _balanced_accuracy(y_true, y_pred)
        if y_prob is not None:
            out["roc_auc"] = _roc_auc_binary(y_true, y_prob)
            out["pr_auc"] = _average_precision_binary(y_true, y_prob)
    return out


def metrics_by_action(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    action: np.ndarray,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for a_val in [0, 1]:
        mask = action == a_val
        if int(mask.sum()) == 0:
            out[str(a_val)] = {"n": 0, "available": False}
            continue
        m = binary_metrics(y_true[mask], y_pred[mask], None if y_prob is None else y_prob[mask])
        m["available"] = True
        out[str(a_val)] = m
    return out


class TorchBinaryWrapper:
    def __init__(self, net: Any, mean: np.ndarray, std: np.ndarray, backend: str) -> None:
        self.net = net
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self._probe_backend = backend

    def _norm(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return ((x - self.mean) / self.std).astype(np.float32)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not TORCH_AVAILABLE:
            raise ImportError(f"torch unavailable: {TORCH_IMPORT_ERROR}")
        x_norm = self._norm(x)
        with torch.no_grad():
            logits = self.net(torch.from_numpy(x_norm))
            probs1 = torch.sigmoid(logits).cpu().numpy().reshape(-1).astype(np.float32)
        probs0 = (1.0 - probs1).astype(np.float32)
        return np.stack([probs0, probs1], axis=1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        probs1 = self.predict_proba(x)[:, 1]
        return (probs1 >= 0.5).astype(np.int64)


def _fit_torch_binary(
    x_train_sa: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    hidden_layers: Sequence[int],
) -> TorchBinaryWrapper:
    if not TORCH_AVAILABLE:
        raise ImportError(f"torch unavailable: {TORCH_IMPORT_ERROR}")
    torch.manual_seed(int(seed))

    x = np.asarray(x_train_sa, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_norm = ((x - mean) / std).astype(np.float32)

    xt = torch.from_numpy(x_norm)
    yt = torch.from_numpy(y)

    layers: List[Any] = []
    in_dim = int(x_norm.shape[1])
    for hidden_dim in hidden_layers:
        layers.append(nn.Linear(in_dim, int(hidden_dim)))
        layers.append(nn.ReLU())
        in_dim = int(hidden_dim)
    layers.append(nn.Linear(in_dim, 1))
    net = nn.Sequential(*layers)

    pos = float(np.sum(y))
    neg = float(y.shape[0] - pos)
    pos = max(pos, 1.0)
    neg = max(neg, 1.0)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 350 if not hidden_layers else 550
    net.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = net(xt)
        loss = criterion(logits, yt)
        loss.backward()
        optimizer.step()

    net.eval()
    backend = "torch_logreg" if not hidden_layers else "torch_mlp"
    return TorchBinaryWrapper(net=net, mean=mean, std=std, backend=backend)


def fit_logreg(x_train_sa: np.ndarray, y_train: np.ndarray, seed: int) -> Any:
    if SKLEARN_AVAILABLE:
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=5000,
                        random_state=seed,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(x_train_sa, y_train)
        setattr(model, "_probe_backend", "sklearn_logreg")
        return model
    return _fit_torch_binary(x_train_sa=x_train_sa, y_train=y_train, seed=seed, hidden_layers=[])


def fit_mlp(x_train_sa: np.ndarray, y_train: np.ndarray, seed: int) -> Any:
    if SKLEARN_AVAILABLE:
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        alpha=1e-4,
                        max_iter=600,
                        random_state=seed,
                        early_stopping=True,
                    ),
                ),
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(x_train_sa, y_train)
        setattr(model, "_probe_backend", "sklearn_mlp")
        return model
    return _fit_torch_binary(x_train_sa=x_train_sa, y_train=y_train, seed=seed, hidden_layers=[64, 32])


def predict_prob_positive(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(x)
        if prob.ndim == 2 and prob.shape[1] >= 2:
            return prob[:, 1].astype(np.float32)
        if prob.ndim == 2 and prob.shape[1] == 1:
            return prob[:, 0].astype(np.float32)
    pred = model.predict(x)
    return np.asarray(pred, dtype=np.float32)


def build_trivial_baselines(
    y_train: np.ndarray,
    y_test: np.ndarray,
    a_train: np.ndarray,
    a_test: np.ndarray,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    if y_train.size == 0 or y_test.size == 0:
        result["majority"] = {"available": False, "reason": "empty_train_or_test"}
        result["action_rate"] = {"available": False, "reason": "empty_train_or_test"}
        return result

    # Baseline 1: majority class
    p_global = float(np.mean(y_train))
    majority_label = 1 if p_global >= 0.5 else 0
    pred_majority = np.full_like(y_test, fill_value=majority_label)
    prob_majority = np.full(shape=y_test.shape, fill_value=p_global, dtype=np.float32)
    result["majority"] = {
        "available": True,
        "overall": binary_metrics(y_test, pred_majority, prob_majority),
        "by_action": metrics_by_action(y_test, pred_majority, prob_majority, a_test),
        "notes": {"majority_label": int(majority_label), "train_positive_rate": p_global},
    }

    # Baseline 2: empirical reward rate by action from train set
    p_by_action: Dict[int, float] = {}
    for a_val in [0, 1]:
        mask = a_train == a_val
        if int(mask.sum()) == 0:
            p_by_action[a_val] = p_global
        else:
            p_by_action[a_val] = float(np.mean(y_train[mask]))
    prob_action = np.asarray([p_by_action.get(int(a), p_global) for a in a_test], dtype=np.float32)
    pred_action = (prob_action >= 0.5).astype(np.int64)
    result["action_rate"] = {
        "available": True,
        "overall": binary_metrics(y_test, pred_action, prob_action),
        "by_action": metrics_by_action(y_test, pred_action, prob_action, a_test),
        "notes": {
            "train_positive_rate": p_global,
            "p_reward_given_action": {str(k): float(v) for k, v in p_by_action.items()},
        },
    }
    return result


def optimistic_proxy_report(
    model: Any,
    x_base_test: np.ndarray,
    train_action_counts: Dict[int, int],
    test_action_counts: Dict[int, int],
    min_action_samples: int,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "available": False,
        "value": None,
        "reason": None,
        "train_action_counts": {str(k): int(v) for k, v in train_action_counts.items()},
        "test_action_counts": {str(k): int(v) for k, v in test_action_counts.items()},
        "note": "",
    }
    c0 = int(train_action_counts.get(0, 0))
    c1 = int(train_action_counts.get(1, 0))
    if c0 < min_action_samples or c1 < min_action_samples:
        report["reason"] = (
            f"insufficient_train_action_samples(min={min_action_samples}, action0={c0}, action1={c1})"
        )
        return report
    if x_base_test.shape[0] == 0:
        report["reason"] = "empty_test_set"
        return report
    try:
        x0 = with_action_feature(x_base_test, np.zeros((x_base_test.shape[0],), dtype=np.int64))
        x1 = with_action_feature(x_base_test, np.ones((x_base_test.shape[0],), dtype=np.int64))
        p0 = predict_prob_positive(model, x0)
        p1 = predict_prob_positive(model, x1)
        proxy = float(np.mean(np.maximum(p0, p1)))
        report["available"] = True
        report["value"] = proxy
        t0 = int(test_action_counts.get(0, 0))
        t1 = int(test_action_counts.get(1, 0))
        if t0 < min_action_samples or t1 < min_action_samples:
            report["note"] = (
                "computed_with_test_action_imbalance; use as extrapolation_reference"
            )
    except Exception as exc:
        report["reason"] = f"proxy_failed:{type(exc).__name__}:{exc}"
    return report


def evaluate_dataset(
    samples: Sequence[Sample],
    split_mode: str,
    seed: int,
    min_action_samples: int,
    include_scenario_feature: bool = False,
    scenario_vocab: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    train_idx, test_idx = split_indices(samples, split_mode=split_mode)

    x_train_base, y_train, a_train, s_train, table_train = make_matrix_and_labels(samples, train_idx)
    x_test_base, y_test, a_test, s_test, table_test = make_matrix_and_labels(samples, test_idx)

    if include_scenario_feature:
        vocab = list(scenario_vocab or sorted(set(s_train + s_test)))
        x_train_base = add_scenario_onehot(x_train_base, s_train, vocab)
        x_test_base = add_scenario_onehot(x_test_base, s_test, vocab)
    else:
        vocab = []

    x_train_sa = with_action_feature(x_train_base, a_train) if x_train_base.shape[0] else x_train_base
    x_test_sa = with_action_feature(x_test_base, a_test) if x_test_base.shape[0] else x_test_base

    train_tables = set(int(x) for x in table_train.tolist())
    test_tables = set(int(x) for x in table_test.tolist())
    table_overlap = sorted(train_tables.intersection(test_tables))

    split_summary = {
        "split_mode": split_mode,
        "n_train": int(y_train.size),
        "n_test": int(y_test.size),
        "train_positive_rate": float(np.mean(y_train)) if y_train.size else None,
        "test_positive_rate": float(np.mean(y_test)) if y_test.size else None,
        "train_action_counts": {str(k): int(v) for k, v in Counter(a_train.tolist()).items()},
        "test_action_counts": {str(k): int(v) for k, v in Counter(a_test.tolist()).items()},
        "train_table_minmax": (
            [int(np.min(table_train)), int(np.max(table_train))]
            if table_train.size
            else None
        ),
        "test_table_minmax": (
            [int(np.min(table_test)), int(np.max(table_test))]
            if table_test.size
            else None
        ),
        "table_overlap_count": int(len(table_overlap)),
        "table_overlap_examples": table_overlap[:20],
        "feature_dim_base": int(x_train_base.shape[1]) if x_train_base.ndim == 2 and x_train_base.shape[0] else int(x_test_base.shape[1]) if x_test_base.ndim == 2 and x_test_base.shape[0] else 0,
        "feature_dim_with_action": int(x_train_sa.shape[1]) if x_train_sa.ndim == 2 and x_train_sa.shape[0] else int(x_test_sa.shape[1]) if x_test_sa.ndim == 2 and x_test_sa.shape[0] else 0,
        "scenario_vocab_for_features": vocab,
    }

    baselines = build_trivial_baselines(y_train, y_test, a_train, a_test)

    result: Dict[str, Any] = {
        "split_summary": split_summary,
        "baselines": baselines,
        "backend_availability": {
            "sklearn_available": bool(SKLEARN_AVAILABLE),
            "sklearn_import_error": SKLEARN_IMPORT_ERROR,
            "torch_available": bool(TORCH_AVAILABLE),
            "torch_import_error": TORCH_IMPORT_ERROR,
        },
        "models": {},
    }

    if y_train.size == 0 or y_test.size == 0:
        result["models"]["logreg"] = {"available": False, "reason": "empty_train_or_test"}
        result["models"]["mlp"] = {"available": False, "reason": "empty_train_or_test"}
        return result

    if np.unique(y_train).size < 2:
        reason = "train_labels_single_class"
        result["models"]["logreg"] = {"available": False, "reason": reason}
        result["models"]["mlp"] = {"available": False, "reason": reason}
        return result

    if not SKLEARN_AVAILABLE and not TORCH_AVAILABLE:
        reason = (
            "no_model_backend_available:"
            f"sklearn=({SKLEARN_IMPORT_ERROR});torch=({TORCH_IMPORT_ERROR})"
        )
        result["models"]["logreg"] = {"available": False, "reason": reason}
        result["models"]["mlp"] = {"available": False, "reason": reason}
        return result

    train_action_counts = {
        0: int(np.sum(a_train == 0)),
        1: int(np.sum(a_train == 1)),
    }
    test_action_counts = {
        0: int(np.sum(a_test == 0)),
        1: int(np.sum(a_test == 1)),
    }

    # Model: Logistic Regression
    try:
        model_lr = fit_logreg(x_train_sa, y_train, seed=seed)
        prob_lr = predict_prob_positive(model_lr, x_test_sa)
        pred_lr = (prob_lr >= 0.5).astype(np.int64)
        result["models"]["logreg"] = {
            "available": True,
            "backend": getattr(model_lr, "_probe_backend", "unknown"),
            "overall": binary_metrics(y_test, pred_lr, prob_lr),
            "by_action": metrics_by_action(y_test, pred_lr, prob_lr, a_test),
            "optimistic_proxy": optimistic_proxy_report(
                model=model_lr,
                x_base_test=x_test_base,
                train_action_counts=train_action_counts,
                test_action_counts=test_action_counts,
                min_action_samples=min_action_samples,
            ),
        }
    except Exception as exc:
        result["models"]["logreg"] = {
            "available": False,
            "reason": f"fit_or_eval_failed:{type(exc).__name__}:{exc}",
        }

    # Model: MLP
    try:
        model_mlp = fit_mlp(x_train_sa, y_train, seed=seed)
        prob_mlp = predict_prob_positive(model_mlp, x_test_sa)
        pred_mlp = (prob_mlp >= 0.5).astype(np.int64)
        result["models"]["mlp"] = {
            "available": True,
            "backend": getattr(model_mlp, "_probe_backend", "unknown"),
            "overall": binary_metrics(y_test, pred_mlp, prob_mlp),
            "by_action": metrics_by_action(y_test, pred_mlp, prob_mlp, a_test),
            "optimistic_proxy": optimistic_proxy_report(
                model=model_mlp,
                x_base_test=x_test_base,
                train_action_counts=train_action_counts,
                test_action_counts=test_action_counts,
                min_action_samples=min_action_samples,
            ),
        }
    except Exception as exc:
        result["models"]["mlp"] = {
            "available": False,
            "reason": f"fit_or_eval_failed:{type(exc).__name__}:{exc}",
        }

    return result


def summarize_console(report: Dict[str, Any]) -> None:
    run_id = report.get("run_id", "")
    scenario = report.get("scenario_id", "")
    print(f"[RUN] {run_id} scenario={scenario}")
    entries = report.get("entries", {})
    for key, item in entries.items():
        split_summary = item.get("evaluation", {}).get("split_summary", {})
        n_train = split_summary.get("n_train", 0)
        n_test = split_summary.get("n_test", 0)
        lr = item.get("evaluation", {}).get("models", {}).get("logreg", {})
        mlp = item.get("evaluation", {}).get("models", {}).get("mlp", {})

        def _fmt_model(m: Dict[str, Any]) -> str:
            if not m.get("available", False):
                return f"NA({m.get('reason', 'unavailable')})"
            overall = m.get("overall", {})
            acc = overall.get("accuracy", None)
            bacc = overall.get("balanced_accuracy", None)
            backend = str(m.get("backend", "unknown"))
            return f"{backend}:acc={acc:.4f}, bacc={bacc:.4f}" if acc is not None and bacc is not None else f"{backend}:partial"

        print(
            f"  - {key}: n_train={n_train}, n_test={n_test}, "
            f"logreg[{_fmt_model(lr)}], mlp[{_fmt_model(mlp)}]"
        )


def _normalize_modes(modes: Any) -> List[str]:
    items = [x.strip().upper() for x in str(modes).split(",") if str(x).strip()]
    items = [m for m in items if m in {"A", "B"}]
    return items if items else ["A", "B"]


def _normalize_feature_kinds(feature_kind: str) -> List[str]:
    return ["xt", "Xt"] if feature_kind == "both" else [feature_kind]


def probe_one_run(
    run_dir: Path,
    *,
    feature_kind: str = "Xt",
    n_stack: int = 4,
    split_mode: str = "phase_table",
    seed: int = 42,
    min_action_samples: int = 20,
    modes: str = "A,B",
    report_name: str = "probe_report.json",
    write_report: bool = True,
) -> Dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
    trace_path = run_dir / "rl_trace.csv"
    if not trace_path.exists():
        raise FileNotFoundError(f"rl_trace.csv not found: {trace_path}")

    set_seed(int(seed))
    scenario_id = load_scenario_id(run_dir)
    run_report: Dict[str, Any] = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "scenario_id": scenario_id,
        "entries": {},
    }
    for mode in _normalize_modes(modes):
        for fkind in _normalize_feature_kinds(feature_kind):
            entry_key = f"mode_{mode}_{fkind}"
            samples, build_stats = build_samples_for_run(
                run_dir=run_dir,
                mode=mode,
                feature_kind=fkind,
                n_stack=int(n_stack),
            )
            eval_report = evaluate_dataset(
                samples=samples,
                split_mode=split_mode,
                seed=int(seed),
                min_action_samples=int(min_action_samples),
                include_scenario_feature=False,
                scenario_vocab=None,
            )
            run_report["entries"][entry_key] = {
                "build_stats": build_stats,
                "evaluation": eval_report,
            }

    if write_report:
        out_path = run_dir / str(report_name)
        out_path.write_text(json.dumps(run_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_report


def run_probe(args: argparse.Namespace) -> Dict[str, Any]:
    set_seed(int(args.seed))

    input_path = Path(args.input).resolve()
    run_dirs = discover_run_dirs(input_path)

    modes = _normalize_modes(args.modes)
    feature_kinds = _normalize_feature_kinds(args.feature_kind)

    all_run_reports: List[Dict[str, Any]] = []
    pooled_buckets: Dict[Tuple[str, str], List[Sample]] = defaultdict(list)

    for run_dir in run_dirs:
        scenario_id = load_scenario_id(run_dir)
        run_report: Dict[str, Any] = {
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "scenario_id": scenario_id,
            "entries": {},
        }
        for mode in modes:
            for fkind in feature_kinds:
                entry_key = f"mode_{mode}_{fkind}"
                samples, build_stats = build_samples_for_run(
                    run_dir=run_dir,
                    mode=mode,
                    feature_kind=fkind,
                    n_stack=int(args.n_stack),
                )
                eval_report = evaluate_dataset(
                    samples=samples,
                    split_mode=args.split_mode,
                    seed=int(args.seed),
                    min_action_samples=int(args.min_action_samples),
                    include_scenario_feature=False,
                    scenario_vocab=None,
                )
                run_report["entries"][entry_key] = {
                    "build_stats": build_stats,
                    "evaluation": eval_report,
                }
                pooled_buckets[(mode, fkind)].extend(samples)

        # Save per-run report
        per_run_path = run_dir / str(args.report_name)
        per_run_path.write_text(
            json.dumps(run_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        all_run_reports.append(run_report)
        summarize_console(run_report)
        print(f"  -> saved: {per_run_path}")

    # Build pooled report
    pooled_report: Dict[str, Any] = {
        "input_root": str(input_path),
        "run_count": len(run_dirs),
        "modes": modes,
        "feature_kinds": feature_kinds,
        "split_mode": args.split_mode,
        "entries": {},
    }
    for (mode, fkind), samples in pooled_buckets.items():
        key = f"mode_{mode}_{fkind}"
        scenario_vocab = sorted({s.scenario_id for s in samples})
        eval_report = evaluate_dataset(
            samples=samples,
            split_mode=args.split_mode,
            seed=int(args.seed),
            min_action_samples=int(args.min_action_samples),
            include_scenario_feature=True,
            scenario_vocab=scenario_vocab,
        )
        pooled_report["entries"][key] = {
            "n_samples": int(len(samples)),
            "scenario_vocab": scenario_vocab,
            "evaluation": eval_report,
        }

    pooled_path = input_path / str(args.pooled_report_name)
    pooled_path.write_text(
        json.dumps(pooled_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[POOLED] saved: {pooled_path}")

    master_report = {
        "args": vars(args),
        "run_reports": [
            {
                "run_id": rr.get("run_id"),
                "scenario_id": rr.get("scenario_id"),
                "report_path": str(Path(rr.get("run_dir", ".")) / str(args.report_name)),
            }
            for rr in all_run_reports
        ],
        "pooled_report_path": str(pooled_path),
    }
    return master_report


def main() -> int:
    args = parse_args()
    summary = run_probe(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

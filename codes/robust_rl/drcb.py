from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def _stable_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""


@dataclass
class BanditSnapshot:
    algo: str
    regime_id: str
    context_id: str
    drift_score: float


class DriftRobustContextualBandit:
    """
    Drift-robust contextual bandit with exponential forgetting + per-regime buckets.

    Why this fits this project well:
    - In `codes/dynamic_RL34959.py`, `episode_length = 1` for ALNS mode, so each decision is a 1-step episode.
      This makes the problem much closer to a contextual bandit than a long-horizon MDP.
    - ALNS steps are expensive: this method is sample-efficient and trains online.
    - Action space is usually 2 (wait/keep vs reroute): linear models are stable and interpretable.

    API compatibility goals (minimum):
    - `predict(obs)` -> (action, None)  (SB3-like)
    - `learn(total_timesteps)` drives the env internally (SB3-like)
    """

    def __init__(
        self,
        env: Any,
        *,
        seed: Optional[int] = None,
        decay: float = 0.995,
        ridge: float = 1.0,
        ucb_alpha: float = 0.2,
        use_regime_buckets: bool = True,
        context_getter: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        self.env = env
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        self.decay = float(decay)
        self.ridge = float(ridge)
        self.ucb_alpha = float(ucb_alpha)
        self.use_regime_buckets = bool(use_regime_buckets)
        self.context_getter = context_getter or (lambda: {})

        # Infer action count if possible; default to 2.
        n_actions = 2
        try:
            n_actions = int(getattr(env.action_space, "n", n_actions))
        except Exception:
            n_actions = 2
        self.n_actions = max(2, n_actions)

        self._dim: Optional[int] = None
        self._buckets: Dict[str, Dict[str, Any]] = {}
        self.last_snapshot = BanditSnapshot(algo="DRCB", regime_id="", context_id="", drift_score=0.0)

    def _get_regime_id(self, ctx: Dict[str, Any]) -> str:
        if not self.use_regime_buckets:
            return "__global__"
        phase_label = _stable_str(ctx.get("phase_label", ""))
        return phase_label if phase_label else "__unknown__"

    def _get_context_id(self, ctx: Dict[str, Any]) -> str:
        phase_label = _stable_str(ctx.get("phase_label", ""))
        gt_mean = _as_float(ctx.get("gt_mean"))
        if phase_label and gt_mean is not None:
            return f"{phase_label}|gt_mean={gt_mean:g}"
        if phase_label:
            return phase_label
        if gt_mean is not None:
            return f"gt_mean={gt_mean:g}"
        return ""

    def _features(self, obs: Any, ctx: Dict[str, Any]) -> np.ndarray:
        obs_arr = np.asarray(obs, dtype=float).reshape(-1)
        gt_mean = _as_float(ctx.get("gt_mean"))
        extra = []
        if gt_mean is not None:
            extra.append(gt_mean)
        x = np.concatenate(([1.0], obs_arr, np.asarray(extra, dtype=float)))
        if self._dim is None:
            self._dim = int(x.shape[0])
        return x

    def _bucket(self, regime_id: str) -> Dict[str, Any]:
        bucket = self._buckets.get(regime_id)
        if bucket is not None:
            return bucket
        dim = self._dim or 1
        bucket = {
            "A": [np.eye(dim, dtype=float) * self.ridge for _ in range(self.n_actions)],
            "b": [np.zeros((dim,), dtype=float) for _ in range(self.n_actions)],
        }
        self._buckets[regime_id] = bucket
        return bucket

    def _ensure_dim(self, dim: int) -> None:
        if self._dim is None:
            self._dim = dim
            return
        if self._dim == dim:
            return
        # Dimension change should not happen in this project unless observation definition changes.
        # Fail loudly to avoid silent corruption.
        raise ValueError(f"Observation/context feature dim changed: {self._dim} -> {dim}")

    def predict(self, obs: Any, deterministic: bool = True) -> Tuple[int, None]:
        ctx = dict(self.context_getter() or {})
        regime_id = self._get_regime_id(ctx)
        context_id = self._get_context_id(ctx)

        x = self._features(obs, ctx)
        self._ensure_dim(int(x.shape[0]))
        bucket = self._bucket(regime_id)

        scores = []
        for a in range(self.n_actions):
            A = bucket["A"][a]
            b = bucket["b"][a]
            try:
                w = np.linalg.solve(A, b)
                invA_x = np.linalg.solve(A, x)
                bonus = self.ucb_alpha * float(np.sqrt(max(0.0, float(x @ invA_x))))
            except Exception:
                w = np.zeros_like(b)
                bonus = 0.0
            mean = float(w @ x)
            scores.append(mean + bonus)

        action = int(np.argmax(scores))
        self.last_snapshot = BanditSnapshot(
            algo="DRCB",
            regime_id=regime_id if regime_id != "__unknown__" else "",
            context_id=context_id,
            drift_score=0.0,
        )
        return action, None

    def learn(self, total_timesteps: int = 1) -> "DriftRobustContextualBandit":
        steps = int(max(0, total_timesteps))
        for _ in range(steps):
            obs = self.env.reset()
            action, _ = self.predict(obs, deterministic=False)
            next_obs, reward, done, info = self.env.step(action)
            _ = next_obs, done, info
            self._update(obs, action, reward)
        return self

    def _update(self, obs: Any, action: int, reward: Any) -> None:
        ctx = dict(self.context_getter() or {})
        regime_id = self._get_regime_id(ctx)
        x = self._features(obs, ctx)
        self._ensure_dim(int(x.shape[0]))
        bucket = self._bucket(regime_id)

        r = _as_float(reward)
        if r is None:
            return

        a = int(action)
        if not (0 <= a < self.n_actions):
            return

        bucket["A"][a] = bucket["A"][a] * self.decay + np.outer(x, x)
        bucket["b"][a] = bucket["b"][a] * self.decay + x * float(r)

    def save(self, path: str | Path) -> None:
        out = {
            "algo": "DRCB",
            "n_actions": self.n_actions,
            "dim": self._dim,
            "decay": self.decay,
            "ridge": self.ridge,
            "ucb_alpha": self.ucb_alpha,
            "use_regime_buckets": self.use_regime_buckets,
            "buckets": {},
        }
        for key, bucket in self._buckets.items():
            out["buckets"][key] = {
                "A": [a.tolist() for a in bucket["A"]],
                "b": [b.tolist() for b in bucket["b"]],
            }
        Path(path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, env: Any, *, context_getter: Optional[Callable[[], Dict[str, Any]]] = None) -> "DriftRobustContextualBandit":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(
            env,
            seed=None,
            decay=float(payload.get("decay", 0.995)),
            ridge=float(payload.get("ridge", 1.0)),
            ucb_alpha=float(payload.get("ucb_alpha", 0.2)),
            use_regime_buckets=bool(payload.get("use_regime_buckets", True)),
            context_getter=context_getter,
        )
        model.n_actions = int(payload.get("n_actions", model.n_actions))
        model._dim = payload.get("dim", None)
        buckets = payload.get("buckets", {}) or {}
        for key, bucket in buckets.items():
            model._buckets[str(key)] = {
                "A": [np.asarray(a, dtype=float) for a in bucket.get("A", [])],
                "b": [np.asarray(b, dtype=float) for b in bucket.get("b", [])],
            }
        return model


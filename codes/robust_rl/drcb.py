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


def _ema(old: float, new: float, alpha: float) -> float:
    alpha = float(min(max(alpha, 0.0), 1.0))
    return alpha * float(new) + (1.0 - alpha) * float(old)


@dataclass
class BanditSnapshot:
    algo: str
    regime_id: str
    context_id: str
    drift_score: float
    effective_ucb: float = 0.0
    risk_penalty: float = 0.0


class DriftRobustContextualBandit:
    """
    DRCB-v2:
    - Linear contextual bandit with exponential forgetting.
    - Drift-aware exploration scaling.
    - Conservative deterministic inference (risk-adjusted score).
    """

    def __init__(
        self,
        env: Any,
        *,
        seed: Optional[int] = None,
        decay: float = 0.995,
        ridge: float = 1.0,
        ucb_alpha: float = 0.4,
        risk_alpha: float = 0.1,
        drift_alpha: float = 0.05,
        drift_scale: float = 1.5,
        drift_cap: float = 2.0,
        warm_start_min_pulls: int = 8,
        impl_eps: float = 0.08,
        include_entity_ids: bool = False,
        use_regime_buckets: bool = True,
        use_context_features: bool = True,
        context_getter: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        self.env = env
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        self.decay = float(decay)
        self.ridge = float(ridge)
        self.ucb_alpha = float(ucb_alpha)
        self.risk_alpha = float(risk_alpha)
        self.drift_alpha = float(drift_alpha)
        self.drift_scale = float(drift_scale)
        self.drift_cap = float(max(0.0, drift_cap))
        self.warm_start_min_pulls = int(max(0, warm_start_min_pulls))
        self.impl_eps = float(min(max(impl_eps, 0.0), 1.0))
        self.include_entity_ids = bool(include_entity_ids)
        self.use_regime_buckets = bool(use_regime_buckets)
        self.use_context_features = bool(use_context_features)
        self.context_getter = context_getter or (lambda: {})

        n_actions = 2
        try:
            n_actions = int(getattr(env.action_space, "n", n_actions))
        except Exception:
            n_actions = 2
        self.n_actions = max(2, n_actions)

        self._dim: Optional[int] = None
        self._buckets: Dict[str, Dict[str, Any]] = {}
        self._stats: Dict[str, Dict[str, float]] = {}
        self.last_snapshot = BanditSnapshot(algo="DRCB", regime_id="", context_id="", drift_score=0.0)

    def _get_regime_id(self, ctx: Dict[str, Any]) -> str:
        if not self.use_regime_buckets:
            return "__global__"
        stage = _stable_str(ctx.get("stage", "")).strip().lower()
        severity = _stable_str(ctx.get("severity", "")).strip()
        phase = _stable_str(ctx.get("phase_label", "")).strip()
        if stage and severity:
            return f"{stage}|sev={severity}"
        if stage:
            return stage
        if severity:
            return f"sev={severity}"
        if phase:
            return phase
        return "__unknown__"

    def _get_context_id(self, ctx: Dict[str, Any]) -> str:
        keys = ("stage", "severity", "table_number", "phase_label", "gt_mean")
        parts = []
        for key in keys:
            val = ctx.get(key, "")
            if val in ("", None):
                continue
            if key == "gt_mean":
                fval = _as_float(val)
                if fval is not None:
                    parts.append(f"{key}={fval:g}")
                continue
            parts.append(f"{key}={val}")
        return "|".join(parts)

    def _context_feature_vector(self, ctx: Dict[str, Any]) -> np.ndarray:
        if not self.use_context_features:
            return np.zeros((0,), dtype=float)
        feats = []
        # Keep only bounded/normalized signals by default.
        numeric_specs = [
            ("severity", 6.0),
            ("delay_tolerance", 200.0),
            ("current_time", 200.0),
            ("table_number", 1000.0),
        ]
        for key, denom in numeric_specs:
            fval = _as_float(ctx.get(key))
            if fval is None:
                continue
            feats.append(float(np.tanh(float(fval) / float(max(1e-6, denom)))))

        # Optional only: IDs are high-cardinality/noisy, so default OFF.
        if self.include_entity_ids:
            veh = _as_float(ctx.get("vehicle"))
            req = _as_float(ctx.get("request"))
            if veh is not None:
                feats.append(float(np.tanh(veh / 500.0)))
            if req is not None:
                feats.append(float(np.tanh(req / 1e6)))

        stage = _stable_str(ctx.get("stage", "")).strip().lower()
        feats.append(1.0 if "remove" in stage else 0.0)
        feats.append(1.0 if "insert" in stage else 0.0)
        return np.asarray(feats, dtype=float)

    def _features(self, obs: Any, ctx: Dict[str, Any]) -> np.ndarray:
        obs_arr = np.asarray(obs, dtype=float).reshape(-1)
        ctx_arr = self._context_feature_vector(ctx)
        x = np.concatenate(([1.0], obs_arr, ctx_arr))
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
            "n": [0 for _ in range(self.n_actions)],
        }
        self._buckets[regime_id] = bucket
        return bucket

    def _regime_stats(self, regime_id: str) -> Dict[str, float]:
        stats = self._stats.get(regime_id)
        if stats is not None:
            return stats
        stats = {"reward_ema": 0.0, "residual_ema": 0.0}
        self._stats[regime_id] = stats
        return stats

    def _ensure_dim(self, dim: int) -> None:
        if self._dim is None:
            self._dim = dim
            return
        if self._dim == dim:
            return
        raise ValueError(f"Observation/context feature dim changed: {self._dim} -> {dim}")

    def _mean_and_sigma(self, A: np.ndarray, b: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
        try:
            w = np.linalg.solve(A, b)
            invA_x = np.linalg.solve(A, x)
            sigma = float(np.sqrt(max(0.0, float(x @ invA_x))))
            mean = float(w @ x)
            return mean, sigma
        except Exception:
            return 0.0, 0.0

    def _unwrap_obs(self, reset_out: Any) -> Any:
        if isinstance(reset_out, tuple) and len(reset_out) >= 1:
            return reset_out[0]
        return reset_out

    def _predict_single(self, obs: Any, deterministic: bool) -> int:
        ctx = dict(self.context_getter() or {})
        regime_id = self._get_regime_id(ctx)
        context_id = self._get_context_id(ctx)
        stats = self._regime_stats(regime_id)

        x = self._features(obs, ctx)
        self._ensure_dim(int(x.shape[0]))
        bucket = self._bucket(regime_id)

        drift_score = float(max(0.0, stats.get("residual_ema", 0.0)))
        drift_scale = min(self.drift_cap, drift_score)
        effective_ucb = float(self.ucb_alpha * (1.0 + self.drift_scale * drift_scale))
        risk_penalty = float(self.risk_alpha * (1.0 + 0.5 * drift_scale))
        phase = _stable_str(ctx.get("phase", "")).strip().lower()

        # Low-budget stochasticity only for implementation phase.
        if not deterministic and phase == "implement" and self.impl_eps > 0.0:
            if self.rng.rand() < self.impl_eps:
                action = int(self.rng.randint(self.n_actions))
                self.last_snapshot = BanditSnapshot(
                    algo="DRCB",
                    regime_id=regime_id if regime_id != "__unknown__" else "",
                    context_id=context_id,
                    drift_score=drift_score,
                    effective_ucb=effective_ucb,
                    risk_penalty=risk_penalty,
                )
                return action

        # Anti-collapse warm start during learning: force each action to be sampled
        # a few times per regime before pure score-driven selection.
        counts = list(bucket.get("n", [0 for _ in range(self.n_actions)]))
        if not deterministic and self.warm_start_min_pulls > 0:
            min_count = min(counts) if counts else 0
            if min_count < self.warm_start_min_pulls:
                candidates = [idx for idx, c in enumerate(counts) if c == min_count]
                action = int(candidates[self.rng.randint(len(candidates))])
                self.last_snapshot = BanditSnapshot(
                    algo="DRCB",
                    regime_id=regime_id if regime_id != "__unknown__" else "",
                    context_id=context_id,
                    drift_score=drift_score,
                    effective_ucb=effective_ucb,
                    risk_penalty=risk_penalty,
                )
                return action

        scores = []
        for a in range(self.n_actions):
            mean, sigma = self._mean_and_sigma(bucket["A"][a], bucket["b"][a], x)
            if deterministic:
                # Apply conservative penalty proportionally to observed drift.
                drift_weight = min(1.0, max(0.0, drift_score / 0.2))
                score = mean - (risk_penalty * drift_weight) * sigma
            else:
                score = mean + effective_ucb * sigma
            scores.append(score)

        action = int(np.argmax(scores))
        self.last_snapshot = BanditSnapshot(
            algo="DRCB",
            regime_id=regime_id if regime_id != "__unknown__" else "",
            context_id=context_id,
            drift_score=drift_score,
            effective_ucb=effective_ucb,
            risk_penalty=risk_penalty,
        )
        return action

    def predict(
        self,
        observation: Any,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        _ = episode_start
        obs_arr = np.asarray(observation)
        if obs_arr.ndim >= 2:
            actions = []
            for i in range(obs_arr.shape[0]):
                actions.append(self._predict_single(obs_arr[i], deterministic=deterministic))
            return np.asarray(actions, dtype=int), state

        action = self._predict_single(observation, deterministic=deterministic)
        return np.asarray([action], dtype=int), state

    def learn(self, total_timesteps: int = 1) -> "DriftRobustContextualBandit":
        steps = int(max(0, total_timesteps))
        for _ in range(steps):
            obs = self._unwrap_obs(self.env.reset())
            ctx = dict(self.context_getter() or {})
            action, _ = self.predict(obs, deterministic=False)
            action_scalar = int(np.asarray(action).squeeze())
            step_out = self.env.step(action_scalar)
            if isinstance(step_out, tuple) and len(step_out) == 5:
                next_obs, reward, terminated, truncated, info = step_out
                done = bool(terminated) or bool(truncated)
            else:
                next_obs, reward, done, info = step_out
            _ = next_obs, done, info
            self._update(obs, action_scalar, reward, ctx=ctx)
        return self

    def _update(self, obs: Any, action: int, reward: Any, *, ctx: Optional[Dict[str, Any]] = None) -> None:
        ctx = dict(ctx or self.context_getter() or {})
        regime_id = self._get_regime_id(ctx)
        x = self._features(obs, ctx)
        self._ensure_dim(int(x.shape[0]))
        bucket = self._bucket(regime_id)
        stats = self._regime_stats(regime_id)

        r = _as_float(reward)
        if r is None:
            return
        a = int(action)
        if not (0 <= a < self.n_actions):
            return

        bucket["A"][a] = bucket["A"][a] * self.decay + np.outer(x, x)
        bucket["b"][a] = bucket["b"][a] * self.decay + x * float(r)
        bucket["n"][a] = int(bucket["n"][a]) + 1

        old_reward_ema = float(stats.get("reward_ema", 0.0))
        reward_ema = _ema(old_reward_ema, float(r), self.drift_alpha)
        residual = abs(float(r) - reward_ema)
        residual_ema = _ema(float(stats.get("residual_ema", 0.0)), residual, self.drift_alpha)
        stats["reward_ema"] = reward_ema
        stats["residual_ema"] = residual_ema

    def save(self, path: str | Path) -> None:
        out = {
            "algo": "DRCB",
            "n_actions": self.n_actions,
            "dim": self._dim,
            "decay": self.decay,
            "ridge": self.ridge,
            "ucb_alpha": self.ucb_alpha,
            "risk_alpha": self.risk_alpha,
            "drift_alpha": self.drift_alpha,
            "drift_scale": self.drift_scale,
            "drift_cap": self.drift_cap,
            "warm_start_min_pulls": self.warm_start_min_pulls,
            "impl_eps": self.impl_eps,
            "include_entity_ids": self.include_entity_ids,
            "use_regime_buckets": self.use_regime_buckets,
            "use_context_features": self.use_context_features,
            "stats": self._stats,
            "buckets": {},
        }
        for key, bucket in self._buckets.items():
            out["buckets"][key] = {
                "A": [a.tolist() for a in bucket["A"]],
                "b": [b.tolist() for b in bucket["b"]],
                "n": [int(n) for n in bucket.get("n", [0 for _ in range(self.n_actions)])],
            }
        Path(path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        path: str | Path,
        env: Any,
        *,
        context_getter: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> "DriftRobustContextualBandit":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(
            env,
            seed=None,
            decay=float(payload.get("decay", 0.995)),
            ridge=float(payload.get("ridge", 1.0)),
            ucb_alpha=float(payload.get("ucb_alpha", 0.4)),
            risk_alpha=float(payload.get("risk_alpha", 0.1)),
            drift_alpha=float(payload.get("drift_alpha", 0.05)),
            drift_scale=float(payload.get("drift_scale", 1.5)),
            drift_cap=float(payload.get("drift_cap", 2.0)),
            warm_start_min_pulls=int(payload.get("warm_start_min_pulls", 8)),
            impl_eps=float(payload.get("impl_eps", 0.08)),
            include_entity_ids=bool(payload.get("include_entity_ids", False)),
            use_regime_buckets=bool(payload.get("use_regime_buckets", True)),
            use_context_features=bool(payload.get("use_context_features", True)),
            context_getter=context_getter,
        )
        model.n_actions = int(payload.get("n_actions", model.n_actions))
        model._dim = payload.get("dim", None)
        model._stats = dict(payload.get("stats", {}) or {})
        buckets = payload.get("buckets", {}) or {}
        for key, bucket in buckets.items():
            model._buckets[str(key)] = {
                "A": [np.asarray(a, dtype=float) for a in bucket.get("A", [])],
                "b": [np.asarray(b, dtype=float) for b in bucket.get("b", [])],
                "n": [int(n) for n in bucket.get("n", [0 for _ in range(model.n_actions)])],
            }
        return model


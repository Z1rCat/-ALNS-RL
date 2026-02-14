from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import torch as th
    from sb3_contrib import QRDQN
except Exception as exc:  # pragma: no cover
    raise ImportError("QRDQNCVaRAgent requires torch + sb3-contrib.") from exc


@dataclass
class QRDQNCVaRConfig:
    cvar_alpha: float = 0.25
    n_quantiles: int = 64
    learning_rate: float = 1e-3
    buffer_size: int = 50000
    learning_starts: int = 200
    batch_size: int = 64
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 500
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.02
    max_grad_norm: float = 10.0
    device: str = "cpu"


class QRDQNCVaRAgent:
    """
    Thin wrapper around sb3-contrib QRDQN with CVaR inference.
    - Train with QRDQN.
    - During deterministic inference, choose action by lower-tail (CVaR) value.
    """

    def __init__(
        self,
        env: Any,
        config: Optional[QRDQNCVaRConfig] = None,
        *,
        seed: Optional[int] = None,
    ) -> None:
        self.env = env
        self.config = config or QRDQNCVaRConfig()
        self.seed = seed

        self.model = QRDQN(
            "MlpPolicy",
            env,
            learning_rate=float(self.config.learning_rate),
            buffer_size=int(self.config.buffer_size),
            learning_starts=int(self.config.learning_starts),
            batch_size=int(self.config.batch_size),
            train_freq=int(self.config.train_freq),
            gradient_steps=int(self.config.gradient_steps),
            target_update_interval=int(self.config.target_update_interval),
            exploration_fraction=float(self.config.exploration_fraction),
            exploration_initial_eps=float(self.config.exploration_initial_eps),
            exploration_final_eps=float(self.config.exploration_final_eps),
            max_grad_norm=float(self.config.max_grad_norm),
            policy_kwargs={"n_quantiles": int(self.config.n_quantiles)},
            verbose=1,
            seed=seed,
            device=self.config.device,
        )

    def learn(self, total_timesteps: int = 1, **kwargs) -> "QRDQNCVaRAgent":
        self.model.learn(total_timesteps=int(max(0, total_timesteps)), **kwargs)
        return self

    def _cvar_action(self, obs: Any) -> np.ndarray:
        obs_tensor, vec_env = self.model.policy.obs_to_tensor(obs)
        with th.no_grad():
            quantiles = self.model.quantile_net(obs_tensor)  # [B, Nq, Na]
            n_q = int(quantiles.shape[1])
            alpha = float(min(max(self.config.cvar_alpha, 1e-6), 1.0))
            k = max(1, int(np.ceil(alpha * n_q)))
            sorted_q, _ = th.sort(quantiles, dim=1)
            cvar_values = sorted_q[:, :k, :].mean(dim=1)  # [B, Na]
            action = th.argmax(cvar_values, dim=1)
        action_np = action.cpu().numpy()
        if not vec_env:
            action_np = np.array([action_np[0]])
        return action_np

    def predict(
        self,
        observation: Any,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        if deterministic:
            action = self._cvar_action(observation)
            return action, state
        return self.model.predict(observation, state=state, episode_start=episode_start, deterministic=False)

    def save(self, path: str) -> None:
        self.model.save(path)

    @classmethod
    def load(
        cls,
        path: str,
        env: Any,
        config: Optional[QRDQNCVaRConfig] = None,
    ) -> "QRDQNCVaRAgent":
        agent = cls(env, config=config)
        agent.model = QRDQN.load(path, env=env)
        return agent

    def __getattr__(self, item: str) -> Any:
        # Delegate unknown attributes to underlying SB3 model.
        return getattr(self.model, item)

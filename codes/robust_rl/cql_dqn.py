from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

try:
    import torch as th
    import torch.nn.functional as F
    from stable_baselines3 import DQN
except Exception as exc:  # pragma: no cover
    raise ImportError("DiscreteCQLAgent requires torch + stable-baselines3.") from exc


@dataclass
class CQLConfig:
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
    cql_alpha: float = 1.0
    cql_temp: float = 1.0
    device: str = "cpu"


class DiscreteCQLDQN(DQN):
    """
    DQN + conservative regularization:
    loss = TD_loss + alpha * (logsumexp(Q(s,*)) - Q(s,a_data))
    """

    def __init__(self, *args, cql_alpha: float = 1.0, cql_temp: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cql_alpha = float(cql_alpha)
        self.cql_temp = float(max(1e-6, cql_temp))
        self.last_cql_metrics: dict[str, float] = {
            "cql_alpha": float(self.cql_alpha),
            "cql_temp": float(self.cql_temp),
            "cql_updates": 0.0,
            "cql_td_loss": 0.0,
            "cql_cql_loss": 0.0,
            "cql_q_mean": 0.0,
            "cql_q_std": 0.0,
            "cql_q_max": 0.0,
            "cql_q_taken": 0.0,
            "cql_lse_q": 0.0,
            "cql_ood_q_gap": 0.0,
        }

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:  # pragma: no cover
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        td_losses = []
        cql_losses = []
        q_mean_list = []
        q_std_list = []
        q_max_list = []
        q_taken_list = []
        lse_list = []
        ood_gap_list = []
        grad_steps = int(max(0, gradient_steps))

        for _ in range(grad_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            with th.no_grad():
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1.0 - replay_data.dones) * self.gamma * next_q_values

            current_q_values = self.q_net(replay_data.observations)
            action_index = replay_data.actions.long()
            if action_index.ndim == 1:
                action_index = action_index.unsqueeze(-1)
            current_q_selected = th.gather(current_q_values, dim=1, index=action_index)

            td_loss = F.smooth_l1_loss(current_q_selected, target_q_values)
            lse = th.logsumexp(current_q_values / self.cql_temp, dim=1, keepdim=True) * self.cql_temp
            cql_loss = (lse - current_q_selected).mean()
            loss = td_loss + float(self.cql_alpha) * cql_loss

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            losses.append(float(loss.item()))
            td_losses.append(float(td_loss.item()))
            cql_losses.append(float(cql_loss.item()))
            with th.no_grad():
                q_mean_list.append(float(current_q_values.mean().item()))
                q_std_list.append(float(current_q_values.std().item()))
                q_max_list.append(float(current_q_values.max(dim=1)[0].mean().item()))
                q_taken_list.append(float(current_q_selected.mean().item()))
                lse_list.append(float(lse.mean().item()))
                ood_gap_list.append(float((lse - current_q_selected).mean().item()))

        self._n_updates += grad_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if losses:
            self.logger.record("train/loss", float(np.mean(losses)))
            self.logger.record("train/td_loss", float(np.mean(td_losses)))
            self.logger.record("train/cql_loss", float(np.mean(cql_losses)))
            self.logger.record("train/cql_q_mean", float(np.mean(q_mean_list)))
            self.logger.record("train/cql_q_std", float(np.mean(q_std_list)))
            self.logger.record("train/cql_q_max", float(np.mean(q_max_list)))
            self.logger.record("train/cql_q_taken", float(np.mean(q_taken_list)))
            self.logger.record("train/cql_lse_q", float(np.mean(lse_list)))
            self.logger.record("train/cql_ood_q_gap", float(np.mean(ood_gap_list)))
        self.last_cql_metrics = {
            "cql_alpha": float(self.cql_alpha),
            "cql_temp": float(self.cql_temp),
            "cql_updates": float(grad_steps),
            "cql_td_loss": float(np.mean(td_losses)) if td_losses else 0.0,
            "cql_cql_loss": float(np.mean(cql_losses)) if cql_losses else 0.0,
            "cql_q_mean": float(np.mean(q_mean_list)) if q_mean_list else 0.0,
            "cql_q_std": float(np.mean(q_std_list)) if q_std_list else 0.0,
            "cql_q_max": float(np.mean(q_max_list)) if q_max_list else 0.0,
            "cql_q_taken": float(np.mean(q_taken_list)) if q_taken_list else 0.0,
            "cql_lse_q": float(np.mean(lse_list)) if lse_list else 0.0,
            "cql_ood_q_gap": float(np.mean(ood_gap_list)) if ood_gap_list else 0.0,
        }


class DiscreteCQLAgent:
    """
    Thin wrapper around DiscreteCQLDQN to keep project API consistent.
    """

    def __init__(self, env: Any, config: Optional[CQLConfig] = None, *, seed: Optional[int] = None) -> None:
        self.env = env
        self.config = config or CQLConfig()
        self.model = DiscreteCQLDQN(
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
            cql_alpha=float(self.config.cql_alpha),
            cql_temp=float(self.config.cql_temp),
            verbose=1,
            seed=seed,
            device=self.config.device,
        )

    def learn(self, total_timesteps: int = 1, **kwargs) -> "DiscreteCQLAgent":
        self.model.learn(total_timesteps=int(max(0, total_timesteps)), **kwargs)
        return self

    @property
    def last_cql_metrics(self) -> dict[str, float]:
        try:
            return dict(getattr(self.model, "last_cql_metrics", {}) or {})
        except Exception:
            return {}

    def predict(
        self,
        observation: Any,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        return self.model.predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=bool(deterministic),
        )

    def save(self, path: str) -> None:
        self.model.save(path)

    @classmethod
    def load(cls, path: str, env: Any, config: Optional[CQLConfig] = None) -> "DiscreteCQLAgent":
        agent = cls(env=env, config=config)
        agent.model = DiscreteCQLDQN.load(path, env=env, device=agent.config.device)
        return agent

    def __getattr__(self, item: str) -> Any:
        return getattr(self.model, item)

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch as th
    import torch.nn as nn
    from sb3_contrib.common.utils import quantile_huber_loss
except Exception as exc:  # pragma: no cover
    raise ImportError("BE_CVAR_DQN requires torch + sb3-contrib.") from exc


@dataclass
class BECVaRDQNConfig:
    history_len: int = 20
    belief_dim: int = 16
    hidden_dim: int = 64
    n_heads: int = 3
    n_quantiles: int = 51

    gamma: float = 0.99
    learning_rate: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 50000
    learning_starts: int = 200
    train_freq: int = 1
    gradient_steps: int = 1
    target_update_interval: int = 500
    tau: float = 1.0
    max_grad_norm: float = 10.0

    cvar_alpha: float = 0.2
    uncertainty_beta: float = 0.2
    loss_ens_coef: float = 0.01
    loss_belief_coef: float = 1e-4

    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    exploration_fraction: float = 0.3
    impl_eps: float = 0.05

    device: str = "cpu"


class ReplayBuffer:
    def __init__(self, max_size: int) -> None:
        self.max_size = int(max(1, max_size))
        self._data: Deque[Tuple[np.ndarray, int, float, np.ndarray, float, np.ndarray, np.ndarray]] = deque(
            maxlen=self.max_size
        )

    def __len__(self) -> int:
        return len(self._data)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: float,
        tokens: np.ndarray,
        tokens_next: np.ndarray,
    ) -> None:
        self._data.append((obs, int(action), float(reward), next_obs, float(done), tokens, tokens_next))

    def sample(self, batch_size: int) -> List[Tuple[np.ndarray, int, float, np.ndarray, float, np.ndarray, np.ndarray]]:
        if not self._data:
            return []
        data = list(self._data)
        if len(data) <= batch_size:
            return data
        idx = np.random.choice(len(data), size=batch_size, replace=False)
        return [data[i] for i in idx]


class BeliefEncoder(nn.Module):
    def __init__(self, token_dim: int, hidden_dim: int, belief_dim: int) -> None:
        super().__init__()
        self.token_proj = nn.Linear(token_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, belief_dim, num_layers=1, batch_first=True)

    def forward(self, tokens: th.Tensor) -> th.Tensor:
        # tokens: [B, H, D]
        x = th.relu(self.token_proj(tokens))
        _, h = self.gru(x)
        return h[-1]


class EnsembleQuantileQ(nn.Module):
    def __init__(self, obs_dim: int, belief_dim: int, hidden_dim: int, action_dim: int, n_heads: int, n_quantiles: int) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.n_heads = int(n_heads)
        self.n_quantiles = int(n_quantiles)
        in_dim = int(obs_dim + belief_dim)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, self.action_dim * self.n_quantiles) for _ in range(self.n_heads)])

    def forward(self, obs: th.Tensor, belief: th.Tensor) -> th.Tensor:
        # returns [B, K, A, Nq]
        x = th.cat([obs, belief], dim=-1)
        h = self.trunk(x)
        outs = []
        for head in self.heads:
            q = head(h).view(-1, self.action_dim, self.n_quantiles)
            outs.append(q)
        return th.stack(outs, dim=1)


class BeliefEnsembleCvaRDQN:
    """
    Belief + Ensemble + Distributional Q (CVaR decision).
    API-compatible subset for this project:
      - learn(total_timesteps)
      - predict(obs, state=None, episode_start=None, deterministic=False)
    """

    def __init__(
        self,
        env: Any,
        config: Optional[BECVaRDQNConfig] = None,
        *,
        seed: Optional[int] = None,
        context_getter: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        self.env = env
        self.cfg = config or BECVaRDQNConfig()
        self.context_getter = context_getter or (lambda: {})
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        self.device = th.device(self.cfg.device)

        self.obs_dim = int(np.prod(getattr(env.observation_space, "shape", (1,))))
        self.action_dim = int(getattr(env.action_space, "n", 2))
        self.action_dim = max(2, self.action_dim)
        self.token_dim = self.obs_dim + self.action_dim + 1

        self.belief_net = BeliefEncoder(
            token_dim=self.token_dim,
            hidden_dim=int(self.cfg.hidden_dim),
            belief_dim=int(self.cfg.belief_dim),
        ).to(self.device)
        self.q_net = EnsembleQuantileQ(
            obs_dim=self.obs_dim,
            belief_dim=int(self.cfg.belief_dim),
            hidden_dim=int(self.cfg.hidden_dim),
            action_dim=self.action_dim,
            n_heads=int(self.cfg.n_heads),
            n_quantiles=int(self.cfg.n_quantiles),
        ).to(self.device)
        self.target_q_net = EnsembleQuantileQ(
            obs_dim=self.obs_dim,
            belief_dim=int(self.cfg.belief_dim),
            hidden_dim=int(self.cfg.hidden_dim),
            action_dim=self.action_dim,
            n_heads=int(self.cfg.n_heads),
            n_quantiles=int(self.cfg.n_quantiles),
        ).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = th.optim.Adam(
            list(self.belief_net.parameters()) + list(self.q_net.parameters()),
            lr=float(self.cfg.learning_rate),
        )

        self.buffer = ReplayBuffer(int(self.cfg.buffer_size))
        self.history: Deque[np.ndarray] = deque(maxlen=max(1, int(self.cfg.history_len)))
        self.total_steps = 0
        self.update_steps = 0
        self.exploration_rate = float(self.cfg.exploration_initial_eps)
        self.exploration_decay_steps = max(1, int(10000 * float(self.cfg.exploration_fraction)))
        self.last_losses: Dict[str, float] = {}

    def _unwrap_obs(self, reset_out: Any) -> np.ndarray:
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def _unwrap_step(self, step_out: Any) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
        if isinstance(step_out, tuple) and len(step_out) == 5:
            next_obs, reward, terminated, truncated, info = step_out
            done = float(bool(terminated) or bool(truncated))
            return np.asarray(next_obs, dtype=np.float32).reshape(-1), float(reward), done, dict(info or {})
        next_obs, reward, done, info = step_out
        return np.asarray(next_obs, dtype=np.float32).reshape(-1), float(reward), float(bool(done)), dict(info or {})

    def _onehot_action(self, action: int) -> np.ndarray:
        x = np.zeros((self.action_dim,), dtype=np.float32)
        a = int(max(0, min(self.action_dim - 1, int(action))))
        x[a] = 1.0
        return x

    def _append_history(self, obs: np.ndarray, action: int, reward: float) -> None:
        token = np.concatenate([obs.astype(np.float32), self._onehot_action(action), np.asarray([float(reward)], dtype=np.float32)])
        self.history.append(token.astype(np.float32))

    def _history_tokens(self) -> np.ndarray:
        H = int(self.cfg.history_len)
        out = np.zeros((H, self.token_dim), dtype=np.float32)
        if not self.history:
            return out
        hist = list(self.history)[-H:]
        start = H - len(hist)
        out[start:] = np.asarray(hist, dtype=np.float32)
        return out

    def _cvar_stats(self, quantiles: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # quantiles: [B, K, A, Nq]
        n_q = int(quantiles.shape[-1])
        alpha = float(min(max(self.cfg.cvar_alpha, 1e-6), 1.0))
        k = max(1, int(np.ceil(alpha * n_q)))
        sorted_q, _ = th.sort(quantiles, dim=-1)
        cvar = sorted_q[..., :k].mean(dim=-1)  # [B, K, A]
        mean_cvar = cvar.mean(dim=1)  # [B, A]
        uncertainty = cvar.var(dim=1, unbiased=False)  # [B, A]
        scores = mean_cvar - float(self.cfg.uncertainty_beta) * uncertainty
        return scores, mean_cvar, uncertainty

    def _forward_quantiles(self, obs_np: np.ndarray, tokens_np: np.ndarray, use_target: bool = False) -> th.Tensor:
        obs_t = th.as_tensor(obs_np, dtype=th.float32, device=self.device)
        tok_t = th.as_tensor(tokens_np, dtype=th.float32, device=self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        if tok_t.ndim == 2:
            tok_t = tok_t.unsqueeze(0)
        net = self.target_q_net if use_target else self.q_net
        belief = self.belief_net(tok_t)
        return net(obs_t, belief)

    def _select_action(self, obs: np.ndarray, deterministic: bool) -> int:
        quant = self._forward_quantiles(obs, self._history_tokens(), use_target=False)
        scores, _, _ = self._cvar_stats(quant)
        greedy = int(th.argmax(scores[0]).item())
        if deterministic:
            return greedy

        ctx = dict(self.context_getter() or {})
        phase = str(ctx.get("phase", "")).strip().lower()
        eps = float(self.cfg.impl_eps) if phase == "implement" else float(self.exploration_rate)
        if self.rng.rand() < eps:
            return int(self.rng.randint(self.action_dim))
        return greedy

    def _update_exploration(self) -> None:
        progress = min(1.0, float(self.total_steps) / float(max(1, self.exploration_decay_steps)))
        start = float(self.cfg.exploration_initial_eps)
        end = float(self.cfg.exploration_final_eps)
        self.exploration_rate = start + progress * (end - start)

    def _maybe_update_target(self) -> None:
        self.update_steps += 1
        interval = int(max(1, self.cfg.target_update_interval))
        if self.update_steps % interval != 0:
            return
        tau = float(self.cfg.tau)
        if tau >= 1.0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
            return
        with th.no_grad():
            for p, tp in zip(self.q_net.parameters(), self.target_q_net.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * p.data)

    def _train_step(self) -> None:
        batch = self.buffer.sample(int(self.cfg.batch_size))
        if not batch:
            return

        obs = np.asarray([b[0] for b in batch], dtype=np.float32)
        actions = np.asarray([b[1] for b in batch], dtype=np.int64)
        rewards = np.asarray([b[2] for b in batch], dtype=np.float32)
        next_obs = np.asarray([b[3] for b in batch], dtype=np.float32)
        dones = np.asarray([b[4] for b in batch], dtype=np.float32)
        tokens = np.asarray([b[5] for b in batch], dtype=np.float32)
        tokens_next = np.asarray([b[6] for b in batch], dtype=np.float32)

        obs_t = th.as_tensor(obs, dtype=th.float32, device=self.device)
        actions_t = th.as_tensor(actions, dtype=th.long, device=self.device)
        rewards_t = th.as_tensor(rewards, dtype=th.float32, device=self.device)
        next_obs_t = th.as_tensor(next_obs, dtype=th.float32, device=self.device)
        dones_t = th.as_tensor(dones, dtype=th.float32, device=self.device)
        tokens_t = th.as_tensor(tokens, dtype=th.float32, device=self.device)
        tokens_next_t = th.as_tensor(tokens_next, dtype=th.float32, device=self.device)

        belief = self.belief_net(tokens_t)
        q_all = self.q_net(obs_t, belief)  # [B,K,A,Nq]
        B, K, _, Nq = q_all.shape
        action_index = actions_t.view(B, 1, 1, 1).expand(B, K, 1, Nq)
        current_q = th.gather(q_all, dim=2, index=action_index).squeeze(2)  # [B,K,Nq]

        with th.no_grad():
            belief_next = self.belief_net(tokens_next_t)
            q_next_online = self.q_net(next_obs_t, belief_next)
            next_scores, _, _ = self._cvar_stats(q_next_online)  # [B,A]
            next_actions = th.argmax(next_scores, dim=1)  # [B]

            q_next_target = self.target_q_net(next_obs_t, belief_next)
            next_index = next_actions.view(B, 1, 1, 1).expand(B, K, 1, Nq)
            next_q = th.gather(q_next_target, dim=2, index=next_index).squeeze(2)  # [B,K,Nq]

            target_q = rewards_t.view(B, 1, 1) + float(self.cfg.gamma) * (1.0 - dones_t.view(B, 1, 1)) * next_q

        loss_q = quantile_huber_loss(current_q, target_q, sum_over_quantiles=True)
        loss_ens = current_q.var(dim=1, unbiased=False).mean()
        loss_belief = belief.pow(2).mean()
        loss = loss_q + float(self.cfg.loss_ens_coef) * loss_ens + float(self.cfg.loss_belief_coef) * loss_belief

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.belief_net.parameters()) + list(self.q_net.parameters()), float(self.cfg.max_grad_norm))
        self.optimizer.step()
        self._maybe_update_target()

        self.last_losses = {
            "be_loss_q": float(loss_q.item()),
            "be_loss_ens": float(loss_ens.item()),
            "be_loss_belief": float(loss_belief.item()),
            "be_loss_total": float(loss.item()),
        }

    def predict(
        self,
        observation: Any,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        _ = episode_start
        obs_arr = np.asarray(observation, dtype=np.float32)
        if obs_arr.ndim >= 2:
            actions = [self._select_action(obs_arr[i].reshape(-1), deterministic=deterministic) for i in range(obs_arr.shape[0])]
            return np.asarray(actions, dtype=int), state
        a = self._select_action(obs_arr.reshape(-1), deterministic=deterministic)
        return np.asarray([a], dtype=int), state

    def learn(self, total_timesteps: int = 1, **kwargs) -> "BeliefEnsembleCvaRDQN":
        _ = kwargs
        steps = int(max(0, total_timesteps))
        for _ in range(steps):
            obs = self._unwrap_obs(self.env.reset())
            tokens = self._history_tokens()
            action_arr, _ = self.predict(obs, deterministic=False)
            action = int(np.asarray(action_arr).squeeze())

            next_obs, reward, done, info = self._unwrap_step(self.env.step(action))
            _ = info

            self._append_history(obs, action, reward)
            tokens_next = self._history_tokens()
            self.buffer.add(obs, action, reward, next_obs, done, tokens, tokens_next)

            self.total_steps += 1
            self._update_exploration()

            if len(self.buffer) >= int(self.cfg.learning_starts) and self.total_steps % int(max(1, self.cfg.train_freq)) == 0:
                for _ in range(int(max(1, self.cfg.gradient_steps))):
                    self._train_step()
        return self

    def observe_impl(self, obs: np.ndarray, action: int, reward: float) -> None:
        # Lightweight online adaptation in implementation: update belief history only.
        self._append_history(np.asarray(obs, dtype=np.float32).reshape(-1), int(action), float(reward))

    def save(self, path: str) -> None:
        payload = {
            "cfg": vars(self.cfg),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "total_steps": self.total_steps,
            "update_steps": self.update_steps,
            "exploration_rate": self.exploration_rate,
            "belief_net": self.belief_net.state_dict(),
            "q_net": self.q_net.state_dict(),
            "target_q_net": self.target_q_net.state_dict(),
        }
        th.save(payload, path)

    @classmethod
    def load(
        cls,
        path: str,
        env: Any,
        config: Optional[BECVaRDQNConfig] = None,
        *,
        context_getter: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> "BeliefEnsembleCvaRDQN":
        payload = th.load(path, map_location="cpu")
        cfg = config or BECVaRDQNConfig(**dict(payload.get("cfg", {})))
        agent = cls(env, config=cfg, context_getter=context_getter)
        agent.total_steps = int(payload.get("total_steps", 0))
        agent.update_steps = int(payload.get("update_steps", 0))
        agent.exploration_rate = float(payload.get("exploration_rate", cfg.exploration_final_eps))
        agent.belief_net.load_state_dict(payload["belief_net"])
        agent.q_net.load_state_dict(payload["q_net"])
        agent.target_q_net.load_state_dict(payload["target_q_net"])
        return agent

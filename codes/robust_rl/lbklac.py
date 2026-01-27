from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Tuple
from collections import deque
import copy

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical
except Exception as exc:  # pragma: no cover - torch is required at runtime
    raise ImportError("LBKLAC requires torch to be installed.") from exc


@dataclass
class LBKLACConfig:
    # History / belief
    history_len: int = 50
    belief_dim: int = 20
    transformer_layers: int = 2
    transformer_heads: int = 2
    transformer_hidden_dim: int = 64
    transformer_dropout: float = 0.1
    use_causal_mask: bool = True
    belief_kl_sigma: float = 1.0

    # Replay / update
    buffer_size: int = 1000
    batch_size: int = 32
    update_every: int = 4
    update_iters: int = 1
    recent_window: int = 200

    # Loss weights / optimizer
    gamma: float = 0.9
    bootstrap_value: bool = False  # False: 单步 bandit 目标 r；True: r + γV(s')
    beta: float = 0.2
    c_v: float = 0.5
    c_h: float = 0.01
    learning_rate: float = 5e-4
    grad_clip: float = 1.0
    use_ppo_clip: bool = False
    policy_clip: float = 0.2

    # Trust region (delta) + OOD detection
    delta_init: float = 0.1
    delta_min: float = 0.01
    delta_max: float = 0.5
    kappa_up: float = 1.5
    kappa_down: float = 0.9
    tau_kl: float = 0.2
    eps_ood: float = 0.5
    ood_metric: str = "both"  # value_residual | belief_smooth_penalty | both

    # Optional stage input (one-hot)
    use_stage: bool = False
    stage_dim: int = 2

    # Device
    device: str = "auto"


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    tokens: np.ndarray
    tokens_next: np.ndarray
    prev_action_onehot: np.ndarray
    prev_reward: float
    old_logp: float


class ReplayBuffer:
    def __init__(self, max_size: int, recent_window: int) -> None:
        self._data: Deque[Transition] = deque(maxlen=max_size)
        self._recent_window = int(max(0, recent_window))

    def __len__(self) -> int:
        return len(self._data)

    def add(self, transition: Transition) -> None:
        self._data.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        if not self._data:
            return []
        data = list(self._data)
        if self._recent_window > 0:
            window = min(len(data), self._recent_window)
            data = data[-window:]
        if len(data) <= batch_size:
            return data
        replace = len(data) < batch_size
        idx = np.random.choice(len(data), size=batch_size, replace=replace)
        return [data[i] for i in idx]


class TransformerBelief(nn.Module):
    def __init__(
        self,
        token_dim: int,
        hidden_dim: int,
        belief_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_len: int,
        use_causal_mask: bool,
    ) -> None:
        super().__init__()
        self.token_proj = nn.Linear(token_dim, hidden_dim)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="relu",
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(hidden_dim, belief_dim)
        self.use_causal_mask = use_causal_mask

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # tokens: [batch, seq, token_dim]
        batch, seq, _ = tokens.shape
        device = tokens.device
        pos_ids = torch.arange(seq, device=device).unsqueeze(0).expand(batch, seq)
        x = self.token_proj(tokens) + self.pos_emb(pos_ids)
        x = x.transpose(0, 1)  # [seq, batch, hidden]
        key_padding = torch.arange(seq, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        mask = self._causal_mask(seq, device) if self.use_causal_mask and seq > 1 else None
        out = self.encoder(x, mask=mask, src_key_padding_mask=key_padding)
        out = out.transpose(0, 1)  # [batch, seq, hidden]
        last_idx = (lengths - 1).clamp(min=0)
        last = out[torch.arange(batch, device=device), last_idx]
        return self.out_proj(last)


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        belief_dim: int,
        hidden_dim: int,
        action_dim: int,
        extra_dim: int = 0,
    ) -> None:
        super().__init__()
        self.extra_dim = int(max(0, extra_dim))
        in_dim = obs_dim + belief_dim + self.extra_dim
        self.actor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        obs: torch.Tensor,
        belief: torch.Tensor,
        prev_ar: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.extra_dim > 0 and prev_ar is not None:
            x = torch.cat([obs, belief, prev_ar], dim=-1)
        else:
            x = torch.cat([obs, belief], dim=-1)
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value


def _resolve_device(device: str) -> torch.device:
    if device and device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LBKLACAgent:
    """
    Latent Belief KL-Regularized Actor-Critic.

    The agent maintains a history deque across single-step episodes and
    updates a transformer-based belief embedding.
    """

    def __init__(
        self,
        env: Any,
        config: Optional[LBKLACConfig] = None,
        *,
        seed: Optional[int] = None,
        stage_getter: Optional[Callable[[], Optional[Iterable[float]]]] = None,
    ) -> None:
        self.env = env
        self.config = config or LBKLACConfig()
        self.stage_getter = stage_getter

        obs_dim = int(np.prod(getattr(env.observation_space, "shape", (1,))))
        action_dim = int(getattr(env.action_space, "n", 2))
        action_dim = max(2, action_dim)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = _resolve_device(self.config.device)

        token_dim = obs_dim  # token 仅包含状态（可选拼接 stage）
        if self.config.use_stage:
            token_dim += int(self.config.stage_dim)
        self.token_dim = token_dim

        self.prev_ar_dim = self.action_dim + 1  # one-hot action + reward
        self.belief_net = TransformerBelief(
            token_dim=token_dim,
            hidden_dim=self.config.transformer_hidden_dim,
            belief_dim=self.config.belief_dim,
            num_layers=self.config.transformer_layers,
            num_heads=self.config.transformer_heads,
            dropout=self.config.transformer_dropout,
            max_len=max(2, self.config.history_len),
            use_causal_mask=self.config.use_causal_mask,
        ).to(self.device)
        self.actor_critic = ActorCritic(
            obs_dim=obs_dim,
            belief_dim=self.config.belief_dim,
            hidden_dim=self.config.transformer_hidden_dim,
            action_dim=action_dim,
            extra_dim=self.prev_ar_dim,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.belief_net.parameters()) + list(self.actor_critic.parameters()),
            lr=self.config.learning_rate,
        )

        self.buffer = ReplayBuffer(self.config.buffer_size, self.config.recent_window)
        self.history: Deque[Tuple[np.ndarray, int, float, np.ndarray]] = deque(
            maxlen=max(1, self.config.history_len - 1)
        )
        self.delta_t = float(self.config.delta_init)
        self.step_count = 0
        self.last_update_metrics: Dict[str, float] = {}

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def _stage_vec(self) -> np.ndarray:
        if not self.config.use_stage or self.stage_getter is None:
            return np.zeros((0,), dtype=np.float32)
        try:
            stage = self.stage_getter()
        except Exception:
            stage = None
        if stage is None:
            return np.zeros((int(self.config.stage_dim),), dtype=np.float32)
        stage_arr = np.asarray(list(stage), dtype=np.float32).reshape(-1)
        if stage_arr.size != int(self.config.stage_dim):
            return np.zeros((int(self.config.stage_dim),), dtype=np.float32)
        return stage_arr

    def _token(self, obs: np.ndarray, stage_vec: np.ndarray) -> np.ndarray:
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        if stage_vec.size > 0:
            return np.concatenate([obs_arr, stage_vec])
        return obs_arr

    def _build_tokens(self, history: Deque[Tuple[np.ndarray, int, float, np.ndarray]], obs: np.ndarray) -> np.ndarray:
        tokens: List[np.ndarray] = []
        for past_obs, past_action, past_reward, past_stage in history:
            tokens.append(self._token(past_obs, past_stage))
        current_stage = self._stage_vec()
        tokens.append(self._token(obs, current_stage))
        if len(tokens) > self.config.history_len:
            tokens = tokens[-self.config.history_len :]
        return np.stack(tokens, axis=0)

    def _pad_tokens(self, token_list: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = torch.tensor([t.shape[0] for t in token_list], dtype=torch.long, device=self.device)
        max_len = int(max(lengths).item()) if len(token_list) else 1
        max_len = max(max_len, 1)
        batch = torch.zeros((len(token_list), max_len, self.token_dim), dtype=torch.float32, device=self.device)
        for i, tokens in enumerate(token_list):
            batch[i, : tokens.shape[0], :] = torch.tensor(tokens, dtype=torch.float32, device=self.device)
        return batch, lengths

    def _action_onehot(self, action: int) -> np.ndarray:
        onehot = np.zeros((self.action_dim,), dtype=np.float32)
        if 0 <= int(action) < self.action_dim:
            onehot[int(action)] = 1.0
        return onehot

    def _prev_ar_from_history(self) -> Tuple[np.ndarray, float]:
        if not self.history:
            return np.zeros((self.action_dim,), dtype=np.float32), 0.0
        last_action = int(self.history[-1][1])
        last_reward = float(self.history[-1][2])
        return self._action_onehot(last_action), last_reward

    def _belief_smooth_penalty(self, b_t: torch.Tensor, b_prev: torch.Tensor) -> torch.Tensor:
        sigma2 = max(1e-6, float(self.config.belief_kl_sigma) ** 2)
        return 0.5 * ((b_t - b_prev) ** 2).sum(dim=-1) / sigma2

    def _update_delta(self, value_residual: float, belief_smooth_penalty: float) -> None:
        ood_metric = (self.config.ood_metric or "both").lower()
        ood = False
        if ood_metric in ("value_residual", "both") and value_residual > self.config.eps_ood:
            ood = True
        if ood_metric in ("belief_kl", "belief_smooth_penalty", "both") and belief_smooth_penalty > self.config.tau_kl:
            ood = True
        if ood:
            self.delta_t = min(self.config.delta_max, self.config.kappa_up * self.delta_t)
        else:
            self.delta_t = max(self.config.delta_min, self.config.kappa_down * self.delta_t)

    def act(self, obs: np.ndarray, *, deterministic: bool = False) -> Dict[str, Any]:
        tokens = self._build_tokens(self.history, obs)
        tokens_t, lengths = self._pad_tokens([tokens])
        belief = self.belief_net(tokens_t, lengths)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).reshape(1, -1)
        prev_action, prev_reward = self._prev_ar_from_history()
        prev_ar = np.concatenate([prev_action, np.array([prev_reward], dtype=np.float32)])
        prev_ar_t = torch.tensor(prev_ar, dtype=torch.float32, device=self.device).reshape(1, -1)
        logits, value = self.actor_critic(obs_t, belief, prev_ar_t)
        dist = Categorical(logits=logits)
        if deterministic:
            action = int(torch.argmax(logits, dim=-1).item())
        else:
            action = int(dist.sample().item())
        logp = float(dist.log_prob(torch.tensor([action], device=self.device)).item())
        entropy = float(dist.entropy().item())
        action_prob = float(torch.exp(torch.tensor(logp)).item())
        return {
            "action": action,
            "logp": logp,
            "entropy": entropy,
            "action_prob": action_prob,
            "belief": belief.detach().cpu().numpy().reshape(-1),
            "value": float(value.item()),
            "tokens": tokens,
        }

    def observe(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        *,
        tokens: Optional[np.ndarray] = None,
        old_logp: float = 0.0,
        record: bool = True,
        update: bool = True,
    ) -> Dict[str, Any]:
        self.step_count += 1
        tokens = tokens if tokens is not None else self._build_tokens(self.history, obs)
        tokens_t, lengths = self._pad_tokens([tokens])
        belief = self.belief_net(tokens_t, lengths)
        if tokens.shape[0] > 1:
            tokens_prev = tokens[:-1]
        else:
            tokens_prev = tokens
        tokens_prev_t, lengths_prev = self._pad_tokens([tokens_prev])
        belief_prev = self.belief_net(tokens_prev_t, lengths_prev)
        belief_smooth_penalty = float(self._belief_smooth_penalty(belief, belief_prev).item())

        prev_action, prev_reward = self._prev_ar_from_history()
        prev_ar = np.concatenate([prev_action, np.array([prev_reward], dtype=np.float32)])
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).reshape(1, -1)
        prev_ar_t = torch.tensor(prev_ar, dtype=torch.float32, device=self.device).reshape(1, -1)
        _, value = self.actor_critic(obs_t, belief, prev_ar_t)
        value_residual = float(reward - value.item())
        self._update_delta(value_residual, belief_smooth_penalty)

        stage_vec = self._stage_vec()
        self.history.append((np.asarray(obs, dtype=np.float32), int(action), float(reward), stage_vec))
        tokens_next = self._build_tokens(self.history, next_obs)

        if record:
            self.buffer.add(
                Transition(
                    obs=np.asarray(obs, dtype=np.float32),
                    action=int(action),
                    reward=float(reward),
                    next_obs=np.asarray(next_obs, dtype=np.float32),
                    tokens=tokens,
                    tokens_next=tokens_next,
                    prev_action_onehot=prev_action,
                    prev_reward=float(prev_reward),
                    old_logp=float(old_logp),
                )
            )

        update_metrics: Dict[str, float] = {}
        if update and self.step_count % self.config.update_every == 0:
            update_metrics = self.update()
            self.last_update_metrics = update_metrics

        return {
            "belief_smooth_penalty": belief_smooth_penalty,
            "value_residual": value_residual,
            "delta_t": self.delta_t,
            "bootstrap": int(bool(self.config.bootstrap_value)),
            **update_metrics,
        }

    def update(self) -> Dict[str, float]:
        if len(self.buffer) < max(1, self.config.batch_size):
            return {}
        metrics: Dict[str, float] = {}
        for _ in range(max(1, self.config.update_iters)):
            batch = self.buffer.sample(self.config.batch_size)
            if not batch:
                break
            obs = torch.tensor([t.obs for t in batch], dtype=torch.float32, device=self.device)
            actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device)
            rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
            next_obs = torch.tensor([t.next_obs for t in batch], dtype=torch.float32, device=self.device)

            actions_safe = torch.clamp(actions, min=0, max=self.action_dim - 1)
            prev_action = torch.tensor(
                [t.prev_action_onehot for t in batch], dtype=torch.float32, device=self.device
            )
            prev_reward = torch.tensor([t.prev_reward for t in batch], dtype=torch.float32, device=self.device).unsqueeze(-1)
            prev_ar = torch.cat([prev_action, prev_reward], dim=-1)

            prev_action_next = torch.zeros((len(batch), self.action_dim), dtype=torch.float32, device=self.device)
            prev_action_next.scatter_(1, actions_safe.view(-1, 1), 1.0)
            prev_reward_next = rewards.unsqueeze(-1)
            prev_ar_next = torch.cat([prev_action_next, prev_reward_next], dim=-1)

            tokens_list = [t.tokens for t in batch]
            tokens_next_list = [t.tokens_next for t in batch]
            tokens_prev_list = [t.tokens[:-1] if t.tokens.shape[0] > 1 else t.tokens for t in batch]

            tokens_t, lengths = self._pad_tokens(tokens_list)
            tokens_next_t, lengths_next = self._pad_tokens(tokens_next_list)
            tokens_prev_t, lengths_prev = self._pad_tokens(tokens_prev_list)

            def _compute_losses() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                belief = self.belief_net(tokens_t, lengths)
                belief_next = self.belief_net(tokens_next_t, lengths_next)
                belief_prev = self.belief_net(tokens_prev_t, lengths_prev)

                logits, value = self.actor_critic(obs, belief, prev_ar)
                _, next_value = self.actor_critic(next_obs, belief_next, prev_ar_next)

                dist = Categorical(logits=logits)
                logp = dist.log_prob(actions_safe)
                entropy = dist.entropy().mean()

                if self.config.bootstrap_value:
                    # 使用 r + γV(s') 做平滑的 TD 目标
                    target = rewards + self.config.gamma * next_value.detach()
                    advantage = target - value.detach()
                else:
                    # 单步 episode：target = r，advantage = r - V(s,b)
                    target = rewards
                    advantage = rewards - value.detach()

                if self.config.use_ppo_clip:
                    old_logp = torch.tensor([t.old_logp for t in batch], dtype=torch.float32, device=self.device)
                    ratio = torch.exp(logp - old_logp)
                    unclipped = ratio * advantage
                    clipped = torch.clamp(ratio, 1.0 - self.config.policy_clip, 1.0 + self.config.policy_clip) * advantage
                    loss_pi = -torch.min(unclipped, clipped).mean()
                else:
                    loss_pi = -(logp * advantage).mean()

                loss_v = 0.5 * (value - target).pow(2).mean()
                loss_smooth = self._belief_smooth_penalty(belief, belief_prev).mean()
                loss_entropy = -entropy
                loss_total = loss_pi + self.config.c_v * loss_v + self.config.beta * loss_smooth + self.config.c_h * loss_entropy
                return loss_total, loss_pi, loss_v, loss_smooth, loss_entropy, logits, belief

            loss_total, loss_pi, loss_v, loss_smooth, loss_entropy, logits, _ = _compute_losses()
            old_logits = logits.detach()
            old_state = copy.deepcopy(self.actor_critic.state_dict())
            old_belief_state = copy.deepcopy(self.belief_net.state_dict())
            old_opt_state = copy.deepcopy(self.optimizer.state_dict())

            self.optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            if self.config.grad_clip and self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    list(self.actor_critic.parameters()) + list(self.belief_net.parameters()),
                    max_norm=self.config.grad_clip,
                )
            self.optimizer.step()

            with torch.no_grad():
                belief_new = self.belief_net(tokens_t, lengths)
                new_logits, _ = self.actor_critic(obs, belief_new, prev_ar)
                policy_kl = (
                    torch.distributions.kl_divergence(
                        Categorical(logits=old_logits), Categorical(logits=new_logits)
                    )
                    .mean()
                    .item()
                )

            tr_scaled = 0
            tr_scale = 1.0
            if policy_kl > self.delta_t:
                tr_scaled = 1
                tr_scale = float(self.delta_t / (policy_kl + 1e-8))
                # 回滚到旧参数后重新计算 loss，再按比例缩放梯度
                self.actor_critic.load_state_dict(old_state)
                self.belief_net.load_state_dict(old_belief_state)
                self.optimizer.load_state_dict(old_opt_state)

                loss_total, loss_pi, loss_v, loss_smooth, loss_entropy, logits, _ = _compute_losses()
                old_logits = logits.detach()
                self.optimizer.zero_grad(set_to_none=True)
                loss_total.backward()
                for p in list(self.actor_critic.parameters()) + list(self.belief_net.parameters()):
                    if p.grad is not None:
                        p.grad.mul_(tr_scale)
                if self.config.grad_clip and self.config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        list(self.actor_critic.parameters()) + list(self.belief_net.parameters()),
                        max_norm=self.config.grad_clip,
                    )
                self.optimizer.step()

                with torch.no_grad():
                    belief_new = self.belief_net(tokens_t, lengths)
                    new_logits, _ = self.actor_critic(obs, belief_new, prev_ar)
                    policy_kl = (
                        torch.distributions.kl_divergence(
                            Categorical(logits=old_logits), Categorical(logits=new_logits)
                        )
                        .mean()
                        .item()
                    )
            metrics = {
                "loss_pi": float(loss_pi.item()),
                "loss_v": float(loss_v.item()),
                "loss_kl": float(loss_smooth.item()),
                "loss_entropy": float(loss_entropy.item()),
                "policy_kl": float(policy_kl),
                "trust_region_scaled": float(tr_scaled),
                "trust_region_scale": float(tr_scale),
            }
        return metrics

    def learn(self, total_timesteps: int = 1, *, on_step: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        steps = int(max(0, total_timesteps))
        for _ in range(steps):
            obs = self.env.reset()
            act_info = self.act(obs, deterministic=False)
            action = int(act_info["action"])
            next_obs, reward, done, info = self.env.step(action)
            _ = done

            tokens = act_info.get("tokens")
            step_metrics = self.observe(
                obs,
                action,
                reward,
                next_obs,
                tokens=tokens,
                old_logp=float(act_info.get("logp", 0.0)),
                update=True,
            )

            payload = {
                "obs": obs,
                "next_obs": next_obs,
                "action": action,
                "reward": float(reward),
                "action_prob": float(act_info.get("action_prob", 0.0)),
                "entropy": float(act_info.get("entropy", 0.0)),
            }
            payload.update(step_metrics)
            if self.last_update_metrics:
                payload.setdefault("policy_kl", self.last_update_metrics.get("policy_kl", payload.get("policy_kl", 0.0)))
            if isinstance(info, dict):
                payload["info"] = info
            if on_step is not None:
                on_step(payload)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[int, None]:
        info = self.act(obs, deterministic=deterministic)
        return int(info["action"]), None

    def reset_history(self) -> None:
        self.history.clear()

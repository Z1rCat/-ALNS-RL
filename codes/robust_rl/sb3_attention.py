from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple
from collections import deque

import numpy as np

try:
    import torch
    import torch.nn as nn
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
except Exception as exc:  # pragma: no cover - torch/SB3 required at runtime
    raise ImportError("SB3 attention components require torch and stable-baselines3.") from exc

try:
    # Optional: only needed when using MoE policy/algos.
    import torch.nn.functional as F
    from gymnasium import spaces  # SB3 v2 uses gymnasium
    import torch as th
    from stable_baselines3.common.policies import ActorCriticPolicy
except Exception:  # pragma: no cover
    ActorCriticPolicy = None  # type: ignore[assignment]

try:
    import gym
    from gym.spaces import Box
except Exception as exc:  # pragma: no cover - gym required at runtime
    raise ImportError("SB3 attention components require gym to be installed.") from exc


@dataclass
class HATConfig:
    history_len: int = 20
    embed_dim: int = 64
    num_heads: int = 2
    num_layers: int = 2
    dropout: float = 0.1
    feature_dim: int = 64


class HistoryAttentionWrapper(gym.Wrapper):
    """
    HAT (History-Attention Transform) wrapper.

    Observation becomes a fixed-length sequence of
    [obs_t, prev_action_onehot, prev_reward] for the last H steps.
    """

    def __init__(
        self,
        env: gym.Env,
        history_len: int = 20,
        *,
        keep_history: bool = False,
        stage_dim: int = 0,
        stage_getter=None,
    ) -> None:
        super().__init__(env)
        self.history_len = int(max(1, history_len))
        self.keep_history = bool(keep_history)
        self.action_dim = int(getattr(env.action_space, "n", 2))
        obs_shape = getattr(env.observation_space, "shape", (1,))
        self.obs_dim = int(np.prod(obs_shape))
        self.stage_dim = int(max(0, stage_dim))
        self.stage_getter = stage_getter
        self.feature_dim = self.obs_dim + self.action_dim + 1 + self.stage_dim
        self._history: Deque[np.ndarray] = deque(maxlen=self.history_len)
        self._last_action = np.zeros((self.action_dim,), dtype=np.float32)
        self._last_reward = 0.0
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.history_len, self.feature_dim),
            dtype=np.float32,
        )

    def _onehot(self, action: int) -> np.ndarray:
        vec = np.zeros((self.action_dim,), dtype=np.float32)
        if 0 <= int(action) < self.action_dim:
            vec[int(action)] = 1.0
        return vec

    def _append_obs(self, obs: np.ndarray) -> None:
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        ar = np.concatenate([self._last_action, np.array([self._last_reward], dtype=np.float32)])
        if self.stage_dim > 0 and self.stage_getter is not None:
            try:
                stage_val = np.asarray(self.stage_getter(), dtype=np.float32).reshape(-1)
            except Exception:
                stage_val = np.zeros((self.stage_dim,), dtype=np.float32)
            if stage_val.size != self.stage_dim:
                stage_val = np.zeros((self.stage_dim,), dtype=np.float32)
        else:
            stage_val = np.zeros((self.stage_dim,), dtype=np.float32)
        feat = np.concatenate([obs_arr, ar, stage_val]).astype(np.float32)
        self._history.append(feat)

    def _stack_history(self) -> np.ndarray:
        if len(self._history) < self.history_len:
            pad = self.history_len - len(self._history)
            padding = [np.zeros((self.feature_dim,), dtype=np.float32) for _ in range(pad)]
            data = padding + list(self._history)
        else:
            data = list(self._history)
        return np.stack(data, axis=0)

    def reset(self, **kwargs):  # gym 0.21 compat
        obs = self.env.reset(**kwargs)
        if not self.keep_history or not self._history:
            self._history.clear()
            self._last_action = np.zeros((self.action_dim,), dtype=np.float32)
            self._last_reward = 0.0
        self._append_obs(obs)
        return self._stack_history()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = self.env.step(action)
        self._last_action = self._onehot(action)
        self._last_reward = float(reward)
        self._append_obs(obs)
        return self._stack_history(), reward, done, info


class AttentionExtractor(BaseFeaturesExtractor):
    """
    Lightweight attention encoder for HAT observations: [B, H, D].
    """

    def __init__(self, observation_space: Box, config: HATConfig) -> None:
        super().__init__(observation_space, features_dim=config.feature_dim)
        if len(observation_space.shape) != 2:
            raise ValueError(f"Expected [H, D] observation, got {observation_space.shape}.")
        seq_len, token_dim = observation_space.shape
        self.token_dim = int(token_dim)
        self.seq_len = int(seq_len)
        self.embed = nn.Linear(self.token_dim, config.embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_len, config.embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.embed_dim * 2,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.proj = nn.Linear(config.embed_dim, config.feature_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.float()
        x = self.embed(x) + self.pos_emb[:, : x.shape[1], :]
        x = self.encoder(x)
        x = self.norm(x)
        pooled = x[:, -1, :]
        return self.proj(pooled)


@dataclass
class MoEConfig:
    # Experts
    num_experts: int = 2
    expert_hidden_dim: int = 64
    expert_layers: int = 1  # 1: Linear->ReLU->Out; 2: add one extra hidden

    # Gate
    gate_hidden_dim: int = 32
    stage_dim: int = 2  # removal/insertion one-hot from wrapper last token

    # Regularization (used by MoE PPO/A2C subclasses)
    gate_entropy_coef: float = 0.01  # encourage non-collapsed gating
    load_balance_coef: float = 0.01  # encourage mean(g_k) ~= 1/K within a batch

    # Logging (rolling mean over last N forwards; used for rl_trace.csv columns)
    log_window: int = 50

    # Inference-only acceleration
    hard_inference: bool = False  # hard/top-1 gating only for predict()/implement


class _MoEHead(nn.Module):
    """
    Lightweight MoE head for discrete action spaces.

    Inputs:
      - belief/features: [B, D]
      - stage one-hot:  [B, stage_dim] (extracted from last token of HAT obs)

    Outputs:
      - mixed logits: [B, A]
      - mixed value:  [B]
      - gate probs:   [B, K]
      - expert logits/value (for logging)
      - regularizers (gate entropy, load-balance)
    """

    def __init__(self, in_dim: int, action_dim: int, cfg: MoEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_dim = int(in_dim)
        self.action_dim = int(action_dim)
        k = int(max(1, cfg.num_experts))
        self.k = k

        gate_in = self.in_dim + int(max(0, cfg.stage_dim))
        self.gate = nn.Sequential(
            nn.Linear(gate_in, int(cfg.gate_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(cfg.gate_hidden_dim), k),
        )

        def _mlp(out_dim: int) -> nn.Module:
            h = int(cfg.expert_hidden_dim)
            if int(cfg.expert_layers) <= 1:
                return nn.Sequential(nn.Linear(gate_in, h), nn.ReLU(), nn.Linear(h, out_dim))
            return nn.Sequential(
                nn.Linear(gate_in, h),
                nn.ReLU(),
                nn.Linear(h, h),
                nn.ReLU(),
                nn.Linear(h, out_dim),
            )

        self.expert_pi = nn.ModuleList([_mlp(self.action_dim) for _ in range(k)])
        self.expert_v = nn.ModuleList([_mlp(1) for _ in range(k)])

    def forward(self, feat: th.Tensor, stage: Optional[th.Tensor], *, hard: bool) -> Dict[str, th.Tensor]:
        if stage is None:
            x = feat
            stage = th.zeros((feat.shape[0], 0), device=feat.device, dtype=feat.dtype)
        else:
            x = th.cat([feat, stage], dim=-1)

        gate_logits = self.gate(x)
        gate_probs = th.softmax(gate_logits, dim=-1)

        # Expert outputs
        logits_k = th.stack([m(x) for m in self.expert_pi], dim=1)  # [B, K, A]
        value_k = th.stack([m(x).squeeze(-1) for m in self.expert_v], dim=1)  # [B, K]

        # --- Hard requirement: soft MoE must mix in probability space (not logits) ---
        # probs_k: [B, K, A]
        probs_k = th.softmax(logits_k, dim=-1)

        if hard:
            idx = th.argmax(gate_probs, dim=-1)  # [B]
            b_idx = th.arange(feat.shape[0], device=feat.device)
            mixed_probs = probs_k[b_idx, idx]
            mixed_logits = th.log(mixed_probs.clamp_min(1e-8))
            mixed_value = value_k[b_idx, idx]
            selected = idx
        else:
            w = gate_probs.unsqueeze(-1)  # [B, K, 1]
            mixed_probs = (w * probs_k).sum(dim=1)  # [B, A]
            mixed_logits = th.log(mixed_probs.clamp_min(1e-8))
            mixed_value = (gate_probs * value_k).sum(dim=1)
            selected = th.full((feat.shape[0],), -1, device=feat.device, dtype=th.long)

        # Regularizers (tensors with grad)
        gate_entropy = -(gate_probs * th.log(gate_probs.clamp_min(1e-8))).sum(dim=-1).mean()
        mean_gate = gate_probs.mean(dim=0)  # [K]
        target = th.full_like(mean_gate, 1.0 / float(self.k))
        load_balance = (mean_gate - target).pow(2).sum()

        # Expert diversity: symmetric KL between expert distributions (K=2 supported; uses first two experts).
        div = th.zeros((), device=feat.device, dtype=feat.dtype)
        if self.k >= 2:
            p0 = probs_k[:, 0, :].clamp_min(1e-8)
            p1 = probs_k[:, 1, :].clamp_min(1e-8)
            kl01 = (p0 * (th.log(p0) - th.log(p1))).sum(dim=-1)
            kl10 = (p1 * (th.log(p1) - th.log(p0))).sum(dim=-1)
            div = (kl01 + kl10).mean()

        return {
            "mixed_logits": mixed_logits,
            "mixed_value": mixed_value,
            "gate_probs": gate_probs,
            "gate_entropy": gate_entropy,
            "load_balance": load_balance,
            "div": div,
            "expert_logits": logits_k,
            "expert_probs": probs_k,
            "expert_value": value_k,
            "expert_selected": selected,
        }


class HATMoEActorCriticPolicy(ActorCriticPolicy):
    """
    SB3 policy: HAT (features_extractor) + lightweight MoE head (K=2 by default).

    - Transformer (HAT) runs once to produce features/belief b_t.
    - Gate sees [b_t, stage_onehot] (stage extracted from the last token, not from any phase_label).
    - Two tiny expert heads output policy logits/value; mixed by gate.
    """

    def __init__(self, *args, moe_config: Optional[MoEConfig] = None, **kwargs) -> None:
        if ActorCriticPolicy is None:
            raise ImportError("stable-baselines3 is required for HATMoEActorCriticPolicy.")
        self.moe_config = moe_config or MoEConfig()
        super().__init__(*args, **kwargs)

        # NOTE: self.features_dim is set by the (HAT) features_extractor.
        action_dim = int(getattr(self.action_space, "n", 2))
        self._moe = _MoEHead(self.features_dim, action_dim, self.moe_config)

        # Rolling window for rl_trace.csv mean fields.
        win = int(max(1, self.moe_config.log_window))
        self._log_gate0: Deque[float] = deque(maxlen=win)
        self._log_gate1: Deque[float] = deque(maxlen=win)
        self._log_ent: Deque[float] = deque(maxlen=win)
        self._log_e0_p1: Deque[float] = deque(maxlen=win)
        self._log_e1_p1: Deque[float] = deque(maxlen=win)
        self._log_sel: Deque[int] = deque(maxlen=win)

        # Batch-level regularizers used by MoE PPO/A2C subclasses.
        self._moe_cache: Optional[Dict[str, th.Tensor]] = None

    def _extract_stage(self, obs: th.Tensor) -> Optional[th.Tensor]:
        # HAT obs: [B, H, token_dim], stage dims at the end of token.
        sd = int(max(0, self.moe_config.stage_dim))
        if sd <= 0:
            return None
        if obs.ndim != 3:
            return None
        if obs.shape[-1] < sd:
            return None
        return obs[:, -1, -sd:].float()

    def _dist_from_logits(self, logits: th.Tensor):
        dist = self.action_dist.proba_distribution(logits)  # type: ignore[attr-defined]
        return dist

    def _update_logs(self, out: Dict[str, th.Tensor]) -> None:
        with th.no_grad():
            g = out["gate_probs"].detach()
            if g.shape[-1] >= 2:
                self._log_gate0.append(float(g[:, 0].mean().cpu().item()))
                self._log_gate1.append(float(g[:, 1].mean().cpu().item()))
            ent = out["gate_entropy"].detach()
            self._log_ent.append(float(ent.cpu().item()))

            # Expert action=1 prob means (binary action space assumed for interpretability).
            e_probs = out["expert_probs"].detach()  # [B, K, A]
            if e_probs.shape[-1] >= 2 and e_probs.shape[1] >= 2:
                p1 = e_probs[:, :, 1]  # [B, K]
                self._log_e0_p1.append(float(p1[:, 0].mean().cpu().item()))
                self._log_e1_p1.append(float(p1[:, 1].mean().cpu().item()))

            sel = out["expert_selected"].detach()
            if sel.numel() > 0:
                self._log_sel.append(int(sel[0].cpu().item()))

            # Optional: expert diversity for interpretability (not required to log every step)
            if not hasattr(self, "_log_div"):
                self._log_div = deque(maxlen=int(max(1, self.moe_config.log_window)))
            try:
                self._log_div.append(float(out["div"].detach().cpu().item()))
            except Exception:
                pass

    def get_moe_log(self) -> Dict[str, float]:
        def _mean(xs):
            return float(sum(xs) / len(xs)) if xs else 0.0

        # Names follow rl_trace.csv requested columns (mean over rolling window).
        out = {
            "gate_prob_0_mean": _mean(list(self._log_gate0)),
            "gate_prob_1_mean": _mean(list(self._log_gate1)),
            "gate_entropy_mean": _mean(list(self._log_ent)),
            "expert0_action1_prob_mean": _mean(list(self._log_e0_p1)),
            "expert1_action1_prob_mean": _mean(list(self._log_e1_p1)),
        }
        if hasattr(self, "_log_div"):
            out["moe_div_mean"] = _mean(list(getattr(self, "_log_div")))
        else:
            out["moe_div_mean"] = 0.0
        if self.moe_config.hard_inference and self._log_sel:
            sel = list(self._log_sel)
            out["expert_selected_ratio"] = float(sum(1 for x in sel if x == 1) / len(sel))
        else:
            out["expert_selected_ratio"] = 0.0
        return out

    def pop_moe_cache(self) -> Dict[str, th.Tensor]:
        """
        Minibatch-aligned cache.

        Hard requirement:
        - evaluate_actions() stores the current minibatch stats into _moe_cache
        - train() pops immediately after evaluate_actions() to ensure alignment
        """
        if self._moe_cache is None:
            raise RuntimeError("MoE cache is empty: pop_moe_cache() called without a preceding evaluate_actions().")
        out = self._moe_cache
        self._moe_cache = None
        return out

    def _moe_forward(self, obs: th.Tensor, *, hard: bool, cache: bool) -> Tuple[th.Tensor, th.Tensor]:
        feat = self.extract_features(obs)
        stage = self._extract_stage(obs)
        out = self._moe(feat, stage, hard=hard)

        if cache:
            # Cache tensors needed by the current minibatch loss (with grad).
            self._moe_cache = {
                "gate_probs": out["gate_probs"],
                "gate_entropy": out["gate_entropy"],
                "load_balance": out["load_balance"],
                "div": out["div"],
            }

        self._update_logs(out)
        return out["mixed_logits"], out["mixed_value"]

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        hard = bool(self.moe_config.hard_inference) and (not self.training)
        logits, values = self._moe_forward(obs, hard=hard, cache=False)
        dist = self._dist_from_logits(logits)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        # Training always uses soft MoE for stability.
        logits, values = self._moe_forward(obs, hard=False, cache=True)
        dist = self._dist_from_logits(logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: th.Tensor):
        logits, _ = self._moe_forward(obs, hard=bool(self.moe_config.hard_inference) and (not self.training), cache=False)
        return self._dist_from_logits(logits)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        _, values = self._moe_forward(obs, hard=False, cache=False)
        return values


class MoEPPO:  # pragma: no cover - runtime integration
    """
    Thin PPO wrapper that adds MoE regularizers (gate entropy + load balance) to PPO loss.

    This keeps SB3 training loop intact except for the extra term:
      L = L_PPO + c_v * L_V + c_ent * L_ent  +  c_lb * LB(g)  -  c_gate * H(g)
    """

    @staticmethod
    def wrap(sb3_ppo_cls):
        from stable_baselines3 import PPO as _PPO  # local import

        class _MoEPPO(_PPO):
            def __init__(
                self,
                *args,
                moe_gate_entropy_coef: float = 0.01,
                moe_load_balance_coef: float = 0.01,
                moe_div_coef: float = 0.005,
                **kwargs,
            ):
                super().__init__(*args, **kwargs)
                self.moe_gate_entropy_coef = float(moe_gate_entropy_coef)
                self.moe_load_balance_coef = float(moe_load_balance_coef)
                self.moe_div_coef = float(moe_div_coef)

            def train(self) -> None:
                # Copy of SB3 PPO.train() with two extra regularizer terms.
                self.policy.set_training_mode(True)
                self._update_learning_rate(self.policy.optimizer)
                clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
                if self.clip_range_vf is not None:
                    clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

                entropy_losses = []
                pg_losses, value_losses = [], []
                clip_fractions = []
                moe_ent_vals = []
                moe_lb_vals = []
                moe_div_vals = []

                continue_training = True
                for epoch in range(self.n_epochs):
                    approx_kl_divs = []
                    for rollout_data in self.rollout_buffer.get(self.batch_size):
                        actions = rollout_data.actions
                        if isinstance(self.action_space, spaces.Discrete):
                            actions = rollout_data.actions.long().flatten()
                        if self.use_sde:
                            self.policy.reset_noise(self.batch_size)

                        values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                        values = values.flatten()

                        advantages = rollout_data.advantages
                        if self.normalize_advantage and len(advantages) > 1:
                            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                        ratio = th.exp(log_prob - rollout_data.old_log_prob)
                        policy_loss_1 = advantages * ratio
                        policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                        policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                        pg_losses.append(policy_loss.item())
                        clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                        clip_fractions.append(clip_fraction)

                        if self.clip_range_vf is None:
                            values_pred = values
                        else:
                            values_pred = rollout_data.old_values + th.clamp(
                                values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                            )
                        value_loss = F.mse_loss(rollout_data.returns, values_pred)
                        value_losses.append(value_loss.item())

                        if entropy is None:
                            entropy_loss = -th.mean(-log_prob)
                        else:
                            entropy_loss = -th.mean(entropy)
                        entropy_losses.append(entropy_loss.item())

                        moe_ent = moe_lb = moe_div = None
                        if hasattr(self.policy, "pop_moe_cache"):
                            try:
                                cache = self.policy.pop_moe_cache()
                                moe_ent = cache.get("gate_entropy")
                                moe_lb = cache.get("load_balance")
                                moe_div = cache.get("div")
                            except Exception as exc:
                                if not hasattr(self, "_moe_cache_warned"):
                                    print("[warn] MoE cache pop failed:", exc)
                                    self._moe_cache_warned = True
                        moe_term = 0.0
                        if moe_ent is not None and self.moe_gate_entropy_coef != 0.0:
                            moe_term = moe_term - self.moe_gate_entropy_coef * moe_ent
                            moe_ent_vals.append(float(moe_ent.detach().cpu().item()))
                        if moe_lb is not None and self.moe_load_balance_coef != 0.0:
                            moe_term = moe_term + self.moe_load_balance_coef * moe_lb
                            moe_lb_vals.append(float(moe_lb.detach().cpu().item()))
                        if moe_div is not None and self.moe_div_coef != 0.0:
                            # Encourage expert differentiation: maximize symmetric KL => subtract in loss.
                            moe_term = moe_term - self.moe_div_coef * moe_div
                            moe_div_vals.append(float(moe_div.detach().cpu().item()))

                        loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + moe_term

                        with th.no_grad():
                            log_ratio = log_prob - rollout_data.old_log_prob
                            approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                            approx_kl_divs.append(approx_kl_div)

                        if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                            continue_training = False
                            if self.verbose >= 1:
                                print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                            break

                        self.policy.optimizer.zero_grad()
                        loss.backward()
                        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                        self.policy.optimizer.step()

                    self._n_updates += 1
                    if not continue_training:
                        break

                from stable_baselines3.common.utils import explained_variance

                explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
                self.logger.record("train/entropy_loss", np.mean(entropy_losses))
                self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
                self.logger.record("train/value_loss", np.mean(value_losses))
                self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
                self.logger.record("train/clip_fraction", np.mean(clip_fractions))
                self.logger.record("train/loss", loss.item())
                self.logger.record("train/explained_variance", explained_var)
                if moe_ent_vals:
                    self.logger.record("train/moe_gate_entropy", float(np.mean(moe_ent_vals)))
                if moe_lb_vals:
                    self.logger.record("train/moe_load_balance", float(np.mean(moe_lb_vals)))
                if moe_div_vals:
                    self.logger.record("train/moe_div", float(np.mean(moe_div_vals)))
                if hasattr(self.policy, "log_std"):
                    self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
                self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
                self.logger.record("train/clip_range", clip_range)
                if self.clip_range_vf is not None:
                    self.logger.record("train/clip_range_vf", clip_range_vf)

        return _MoEPPO


class MoEA2C:  # pragma: no cover - runtime integration
    """
    Thin A2C wrapper that adds MoE regularizers (gate entropy + load balance) to A2C loss.
    """

    @staticmethod
    def wrap(sb3_a2c_cls):
        from stable_baselines3 import A2C as _A2C  # local import

        class _MoEA2C(_A2C):
            def __init__(
                self,
                *args,
                moe_gate_entropy_coef: float = 0.01,
                moe_load_balance_coef: float = 0.01,
                moe_div_coef: float = 0.005,
                **kwargs,
            ):
                super().__init__(*args, **kwargs)
                self.moe_gate_entropy_coef = float(moe_gate_entropy_coef)
                self.moe_load_balance_coef = float(moe_load_balance_coef)
                self.moe_div_coef = float(moe_div_coef)

            def train(self) -> None:
                self.policy.set_training_mode(True)
                self._update_learning_rate(self.policy.optimizer)

                moe_ent_val = None
                moe_lb_val = None
                moe_div_val = None

                for rollout_data in self.rollout_buffer.get(batch_size=None):
                    actions = rollout_data.actions
                    if isinstance(self.action_space, spaces.Discrete):
                        actions = actions.long().flatten()

                    values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                    values = values.flatten()

                    advantages = rollout_data.advantages
                    if self.normalize_advantage:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    policy_loss = -(advantages * log_prob).mean()
                    value_loss = F.mse_loss(rollout_data.returns, values)

                    if entropy is None:
                        entropy_loss = -th.mean(-log_prob)
                    else:
                        entropy_loss = -th.mean(entropy)

                    moe_ent = moe_lb = moe_div = None
                    if hasattr(self.policy, "pop_moe_cache"):
                        try:
                            cache = self.policy.pop_moe_cache()
                            moe_ent = cache.get("gate_entropy")
                            moe_lb = cache.get("load_balance")
                            moe_div = cache.get("div")
                        except Exception as exc:
                            if not hasattr(self, "_moe_cache_warned"):
                                print("[warn] MoE cache pop failed:", exc)
                                self._moe_cache_warned = True
                    moe_term = 0.0
                    if moe_ent is not None and self.moe_gate_entropy_coef != 0.0:
                        moe_term = moe_term - self.moe_gate_entropy_coef * moe_ent
                        moe_ent_val = float(moe_ent.detach().cpu().item())
                    if moe_lb is not None and self.moe_load_balance_coef != 0.0:
                        moe_term = moe_term + self.moe_load_balance_coef * moe_lb
                        moe_lb_val = float(moe_lb.detach().cpu().item())
                    if moe_div is not None and self.moe_div_coef != 0.0:
                        moe_term = moe_term - self.moe_div_coef * moe_div
                        moe_div_val = float(moe_div.detach().cpu().item())

                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + moe_term

                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()

                from stable_baselines3.common.utils import explained_variance

                explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
                self._n_updates += 1
                self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
                self.logger.record("train/explained_variance", explained_var)
                self.logger.record("train/entropy_loss", entropy_loss.item())
                self.logger.record("train/policy_loss", policy_loss.item())
                self.logger.record("train/value_loss", value_loss.item())
                if moe_ent_val is not None:
                    self.logger.record("train/moe_gate_entropy", moe_ent_val)
                if moe_lb_val is not None:
                    self.logger.record("train/moe_load_balance", moe_lb_val)
                if moe_div_val is not None:
                    self.logger.record("train/moe_div", moe_div_val)
                if hasattr(self.policy, "log_std"):
                    self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        return _MoEA2C

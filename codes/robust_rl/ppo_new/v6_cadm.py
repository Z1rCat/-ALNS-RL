from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import explained_variance


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _to_discrete_actions(action_space: spaces.Space, actions: th.Tensor) -> th.Tensor:
    if isinstance(action_space, spaces.Discrete):
        return actions.long().flatten()
    return actions


def _extract_p_action1(dist: Any) -> Optional[th.Tensor]:
    try:
        if isinstance(dist.distribution, th.distributions.Categorical):
            probs = dist.distribution.probs
            if probs is not None and probs.ndim >= 2 and probs.shape[1] >= 2:
                return probs[:, 1]
    except Exception:
        return None
    return None


def _infer_base_x_dim(x_dim: int, oracle_ctx_dim: int) -> Optional[int]:
    base_x_dim = int(x_dim - max(0, int(oracle_ctx_dim)))
    if base_x_dim > 3 and (base_x_dim - 3) % 2 == 0:
        return int(base_x_dim)
    for cand_ctx in range(0, min(64, x_dim + 1)):
        cand_base = x_dim - cand_ctx
        if cand_base <= 3:
            continue
        if (cand_base - 3) % 2 == 0:
            return int(cand_base)
    return None


def _extract_latest_severity_from_obs(
    observations: th.Tensor,
    *,
    n_stack: int,
    oracle_ctx_dim: int,
) -> Tuple[Optional[th.Tensor], str]:
    try:
        obs = observations.detach().float()
        if obs.ndim != 2:
            obs = obs.view(obs.shape[0], -1)
        if obs.shape[1] <= 0:
            return None, "obs_empty"
        n_stack = int(max(1, n_stack))
        if obs.shape[1] % n_stack != 0:
            return None, "obs_dim_not_divisible_by_stack"
        x_dim = int(obs.shape[1] // n_stack)
        x_now = obs[:, :x_dim]
        base_x_dim = _infer_base_x_dim(x_dim=x_dim, oracle_ctx_dim=int(oracle_ctx_dim))
        if base_x_dim is None:
            return None, "layout_unresolved"
        d = int((base_x_dim - 3) // 2)
        if d < 2:
            return None, "severity_not_available"
        severity = x_now[:, 1]
        return severity, "obs_next_severity"
    except Exception:
        return None, "severity_extract_error"


def _build_transition_dataset(
    rollout_buffer,
    *,
    device: th.device,
    n_stack: int,
    oracle_ctx_dim: int,
    max_transitions: int,
) -> Tuple[Optional[Dict[str, th.Tensor]], str]:
    try:
        obs = rollout_buffer.observations
        actions = rollout_buffer.actions
        rewards = rollout_buffer.rewards
        episode_starts = rollout_buffer.episode_starts
    except Exception:
        return None, "rollout_buffer_missing_fields"

    try:
        if obs is None or actions is None or rewards is None or episode_starts is None:
            return None, "rollout_buffer_empty"
        n_steps = int(obs.shape[0])
        n_envs = int(obs.shape[1])
        if n_steps <= 1 or n_envs <= 0:
            return None, "rollout_buffer_too_short"

        obs_t = obs[:-1].reshape(-1, *obs.shape[2:])
        obs_tp1 = obs[1:].reshape(-1, *obs.shape[2:])
        act_t = actions[:-1].reshape(-1, *actions.shape[2:])
        rew_t = rewards[:-1].reshape(-1)
        ep_tp1 = episode_starts[1:].reshape(-1)

        valid_mask = (np.asarray(ep_tp1).reshape(-1) < 0.5)
        used_fallback = False
        if valid_mask.sum() > 0:
            obs_t = obs_t[valid_mask]
            obs_tp1 = obs_tp1[valid_mask]
            act_t = act_t[valid_mask]
            rew_t = rew_t[valid_mask]
        else:
            # Fallback for single-step episodes (common in this project):
            # use current-frame severity and reward_t as proxy dynamics target.
            used_fallback = True
            obs_t = obs.reshape(-1, *obs.shape[2:])
            obs_tp1 = obs_t
            act_t = actions.reshape(-1, *actions.shape[2:])
            rew_t = rewards.reshape(-1)

        if int(max_transitions) > 0 and len(obs_t) > int(max_transitions):
            idx = np.random.choice(len(obs_t), size=int(max_transitions), replace=False)
            obs_t = obs_t[idx]
            obs_tp1 = obs_tp1[idx]
            act_t = act_t[idx]
            rew_t = rew_t[idx]

        obs_t_th = th.as_tensor(obs_t, device=device).float()
        obs_tp1_th = th.as_tensor(obs_tp1, device=device).float()
        act_t_th = th.as_tensor(act_t, device=device)
        if act_t_th.ndim > 1:
            act_t_th = act_t_th.squeeze(-1)
        rew_t_th = th.as_tensor(rew_t, device=device).float().flatten()

        next_severity, sev_source = _extract_latest_severity_from_obs(
            obs_tp1_th,
            n_stack=int(n_stack),
            oracle_ctx_dim=int(oracle_ctx_dim),
        )
        if next_severity is None:
            return None, sev_source
        if used_fallback:
            sev_source = "obs_curr_severity_fallback"

        data = {
            "obs": obs_t_th,
            "actions": act_t_th.long().flatten(),
            "reward": rew_t_th,
            "next_severity": next_severity.float().flatten(),
        }
        return data, sev_source
    except Exception:
        return None, "transition_build_error"


class CadmAuxPolicy(ActorCriticPolicy):
    """
    CaDM-adapted policy:
    add a forward predictor head for (next_severity, reward_t) from (h_t, a_t).
    """

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        n_actions = int(getattr(self.action_space, "n", 2) or 2)
        n_actions = max(2, n_actions)
        self.cadm_n_actions = int(n_actions)
        hidden = int(max(32, self.mlp_extractor.latent_dim_pi))
        in_dim = int(self.mlp_extractor.latent_dim_pi + self.cadm_n_actions)
        self.cadm_head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        ).to(self.device)
        if self.ortho_init:
            self.cadm_head.apply(lambda m: self.init_weights(m, gain=1.0))
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _latent(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        return latent_pi, latent_vf

    def _action_one_hot(self, actions: th.Tensor) -> th.Tensor:
        act = actions.long().flatten()
        act = th.clamp(act, min=0, max=self.cadm_n_actions - 1)
        return F.one_hot(act, num_classes=self.cadm_n_actions).float()

    def predict_dynamics(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        latent_pi, _ = self._latent(obs)
        act_oh = self._action_one_hot(actions).to(latent_pi.device)
        x = th.cat([latent_pi, act_oh], dim=1)
        out = self.cadm_head(x)
        pred_next_sev = out[:, 0]
        pred_reward = out[:, 1]
        return pred_next_sev, pred_reward

    def evaluate_actions_with_cadm(self, obs: th.Tensor, actions: th.Tensor):
        latent_pi, latent_vf = self._latent(obs)
        dist = self._get_action_dist_from_latent(latent_pi)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.value_net(latent_vf)
        pred_next_sev, pred_reward = self.predict_dynamics(obs, actions)
        return values, log_prob, entropy, pred_next_sev, pred_reward


@dataclass
class CadmAuxConfig:
    beta: float = 0.05
    lambda_next: float = 1.0
    lambda_reward: float = 0.5
    aux_batch_size: int = 64
    max_transitions: int = 4096


class CadmAuxPPO(PPO):
    """
    PPO + auxiliary forward prediction loss (CaDM adaptation):
      L_total = L_PPO + beta * (lambda_next * MSE(next_severity) + lambda_reward * MSE(reward_t))
    """

    def __init__(
        self,
        *args,
        cadm_beta: float = 0.05,
        cadm_lambda_next: float = 1.0,
        cadm_lambda_reward: float = 0.5,
        cadm_aux_batch_size: int = 64,
        cadm_max_transitions: int = 4096,
        **kwargs,
    ) -> None:
        self.cadm_cfg = CadmAuxConfig(
            beta=float(max(0.0, cadm_beta)),
            lambda_next=float(max(0.0, cadm_lambda_next)),
            lambda_reward=float(max(0.0, cadm_lambda_reward)),
            aux_batch_size=int(max(1, cadm_aux_batch_size)),
            max_transitions=int(max(128, cadm_max_transitions)),
        )
        self.last_v6_metrics: Dict[str, Any] = {}
        super().__init__(*args, **kwargs)

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        approx_kl_divs_all = []
        action1_rate_list: list[float] = []
        p_action1_list: list[float] = []
        reward_a0_list: list[float] = []
        reward_a1_list: list[float] = []
        cadm_aux_loss_list: list[float] = []
        cadm_next_loss_list: list[float] = []
        cadm_reward_loss_list: list[float] = []
        cadm_next_mae_list: list[float] = []
        cadm_reward_mae_list: list[float] = []

        continue_training = True
        loss = None

        ext = getattr(self.policy, "features_extractor", None)
        n_stack = int(getattr(ext, "n_stack", 1) or 1)
        oracle_ctx_dim = int(getattr(ext, "oracle_ctx_dim", 0) or 0)
        trans_data, cadm_source = _build_transition_dataset(
            self.rollout_buffer,
            device=self.device,
            n_stack=n_stack,
            oracle_ctx_dim=oracle_ctx_dim,
            max_transitions=int(self.cadm_cfg.max_transitions),
        )
        trans_n = int(trans_data["obs"].shape[0]) if trans_data is not None else 0
        cadm_enabled = (
            trans_data is not None
            and trans_n > 0
            and self.cadm_cfg.beta > 0.0
            and hasattr(self.policy, "predict_dynamics")
        )
        cadm_aux_batches = 0

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = _to_discrete_actions(self.action_space, rollout_data.actions)
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                if hasattr(self.policy, "evaluate_actions_with_cadm"):
                    values, log_prob, entropy, _, _ = self.policy.evaluate_actions_with_cadm(
                        rollout_data.observations, actions
                    )
                else:
                    values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(float(policy_loss.item()))
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(float(clip_fraction))

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(float(value_loss.item()))

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(float(entropy_loss.item()))

                cadm_aux_loss = th.zeros((), device=self.device)
                if cadm_enabled and trans_data is not None:
                    aux_bs = int(min(self.cadm_cfg.aux_batch_size, trans_n))
                    idx = th.randint(low=0, high=trans_n, size=(aux_bs,), device=self.device)
                    obs_aux = trans_data["obs"].index_select(0, idx)
                    act_aux = trans_data["actions"].index_select(0, idx)
                    tgt_next = trans_data["next_severity"].index_select(0, idx)
                    tgt_rew = trans_data["reward"].index_select(0, idx)
                    pred_next, pred_rew = self.policy.predict_dynamics(obs_aux, act_aux)
                    loss_next = F.mse_loss(pred_next, tgt_next)
                    loss_rew = F.mse_loss(pred_rew, tgt_rew)
                    cadm_aux_loss = (
                        float(self.cadm_cfg.lambda_next) * loss_next
                        + float(self.cadm_cfg.lambda_reward) * loss_rew
                    )
                    cadm_aux_batches += 1
                    cadm_next_loss_list.append(float(loss_next.item()))
                    cadm_reward_loss_list.append(float(loss_rew.item()))
                    cadm_aux_loss_list.append(float(cadm_aux_loss.item()))
                    with th.no_grad():
                        cadm_next_mae_list.append(float((pred_next - tgt_next).abs().mean().item()))
                        cadm_reward_mae_list.append(float((pred_rew - tgt_rew).abs().mean().item()))

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + float(self.cadm_cfg.beta) * cadm_aux_loss
                )

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(float(approx_kl_div))

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                with th.no_grad():
                    mask0 = actions == 0
                    mask1 = actions == 1
                    action1_rate_list.append(float(mask1.float().mean().item()))
                    try:
                        dist = self.policy.get_distribution(rollout_data.observations)
                        p1 = _extract_p_action1(dist)
                        if p1 is not None:
                            p_action1_list.append(float(p1.mean().item()))
                    except Exception:
                        pass
                    returns = rollout_data.returns.detach().float().flatten()
                    if int(mask0.sum().item()) > 0:
                        reward_a0_list.append(float(returns[mask0].mean().item()))
                    if int(mask1.sum().item()) > 0:
                        reward_a1_list.append(float(returns[mask1].mean().item()))

            self._n_updates += 1
            approx_kl_divs_all.extend(approx_kl_divs)
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses) if entropy_losses else 0.0)
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses) if pg_losses else 0.0)
        self.logger.record("train/value_loss", np.mean(value_losses) if value_losses else 0.0)
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs_all) if approx_kl_divs_all else 0.0)
        self.logger.record("train/clip_fraction", np.mean(clip_fractions) if clip_fractions else 0.0)
        self.logger.record("train/loss", 0.0 if loss is None else float(loss.item()))
        self.logger.record("train/explained_variance", float(explained_var))
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        diag: Dict[str, Any] = {
            "action1_rate": _safe_mean(action1_rate_list),
            "p_action1": _safe_mean(p_action1_list),
            "reward_given_action0": _safe_mean(reward_a0_list),
            "reward_given_action1": _safe_mean(reward_a1_list),
            "cadm_enabled": int(cadm_enabled and cadm_aux_batches > 0),
            "cadm_source": str(cadm_source),
            "cadm_transitions": int(trans_n),
            "cadm_aux_batches": int(cadm_aux_batches),
            "cadm_aux_loss": _safe_mean(cadm_aux_loss_list),
            "cadm_nextsev_loss": _safe_mean(cadm_next_loss_list),
            "cadm_reward_loss": _safe_mean(cadm_reward_loss_list),
            "cadm_nextsev_mae": _safe_mean(cadm_next_mae_list),
            "cadm_reward_mae": _safe_mean(cadm_reward_mae_list),
        }
        self.last_v6_metrics = diag
        for k, v in diag.items():
            try:
                if isinstance(v, (int, float, np.floating)):
                    self.logger.record(f"train/{k}", float(v))
            except Exception:
                pass

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import explained_variance


def _to_discrete_actions(action_space: spaces.Space, actions: th.Tensor) -> th.Tensor:
    if isinstance(action_space, spaces.Discrete):
        return actions.long().flatten()
    return actions


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _extract_p_action1(dist: Any) -> Optional[th.Tensor]:
    try:
        if isinstance(dist.distribution, th.distributions.Categorical):
            probs = dist.distribution.probs
            if probs is not None and probs.ndim >= 2 and probs.shape[1] >= 2:
                return probs[:, 1]
    except Exception:
        return None
    return None


def _binary_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    if y_true.size == 0 or y_score.size == 0:
        return None
    y_true = y_true.astype(np.int64).reshape(-1)
    y_score = y_score.astype(np.float64).reshape(-1)
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    pos_ranks = ranks[y_true == 1]
    auc = (np.sum(pos_ranks) - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


@dataclass
class ABPPOConfig:
    freq_epsilon: float = 1e-6
    w_max: float = 8.0
    group_adv_norm: bool = True
    group_min_samples: int = 8


class ActionBalancedPPO(PPO):
    """
    V5.1: Action-Balanced PPO.
    Only modifies the policy loss term with inverse-frequency sample weights.
    """

    def __init__(
        self,
        *args,
        ab_freq_epsilon: float = 1e-6,
        ab_w_max: float = 8.0,
        ab_group_adv_norm: bool = True,
        ab_group_min_samples: int = 8,
        **kwargs,
    ) -> None:
        self.ab_cfg = ABPPOConfig(
            freq_epsilon=float(max(1e-12, ab_freq_epsilon)),
            w_max=float(max(1.0, ab_w_max)),
            group_adv_norm=bool(ab_group_adv_norm),
            group_min_samples=int(max(1, ab_group_min_samples)),
        )
        self.last_v5_metrics: Dict[str, Any] = {}
        super().__init__(*args, **kwargs)

    def _normalize_advantages(self, advantages: th.Tensor, actions: th.Tensor) -> th.Tensor:
        if not self.normalize_advantage or len(advantages) <= 1:
            return advantages
        if not self.ab_cfg.group_adv_norm:
            return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        mask0 = actions == 0
        mask1 = actions == 1
        n0 = int(mask0.sum().item())
        n1 = int(mask1.sum().item())
        if n0 < self.ab_cfg.group_min_samples or n1 < self.ab_cfg.group_min_samples:
            return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        out = advantages.clone()
        for mask in (mask0, mask1):
            adv_g = advantages[mask]
            out[mask] = (adv_g - adv_g.mean()) / (adv_g.std() + 1e-8)
        return out

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

        f0_list: list[float] = []
        f1_list: list[float] = []
        w0_list: list[float] = []
        w1_list: list[float] = []
        action1_rate_list: list[float] = []
        p_action1_list: list[float] = []
        reward_a0_list: list[float] = []
        reward_a1_list: list[float] = []

        continue_training = True
        loss = None

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = _to_discrete_actions(self.action_space, rollout_data.actions)

                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = self._normalize_advantages(rollout_data.advantages, actions)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_obj = th.min(policy_loss_1, policy_loss_2)

                # --- Action-balanced weighting (V5.1 core change) ---
                eps = self.ab_cfg.freq_epsilon
                f0 = (actions == 0).float().mean()
                f1 = (actions == 1).float().mean()
                w0 = th.clamp(1.0 / (f0 + eps), min=1.0, max=self.ab_cfg.w_max)
                w1 = th.clamp(1.0 / (f1 + eps), min=1.0, max=self.ab_cfg.w_max)
                sample_w = th.where(actions == 0, w0, w1)
                policy_loss = -(sample_w * policy_obj).mean()

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

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

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

                # Diagnostics for acceptance checks.
                with th.no_grad():
                    f0_list.append(float(f0.item()))
                    f1_list.append(float(f1.item()))
                    w0_list.append(float(w0.item()))
                    w1_list.append(float(w1.item()))
                    action1_rate_list.append(float(f1.item()))

                    try:
                        dist_now = self.policy.get_distribution(rollout_data.observations)
                        p1 = _extract_p_action1(dist_now)
                        if p1 is not None:
                            p_action1_list.append(float(p1.mean().item()))
                    except Exception:
                        pass

                    returns = rollout_data.returns.detach().float().flatten()
                    a0 = actions == 0
                    a1 = actions == 1
                    if int(a0.sum().item()) > 0:
                        reward_a0_list.append(float(returns[a0].mean().item()))
                    if int(a1.sum().item()) > 0:
                        reward_a1_list.append(float(returns[a1].mean().item()))

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

        diag = {
            "abppo_f0": _safe_mean(f0_list),
            "abppo_f1": _safe_mean(f1_list),
            "abppo_w0": _safe_mean(w0_list),
            "abppo_w1": _safe_mean(w1_list),
            "action1_rate": _safe_mean(action1_rate_list),
            "p_action1": _safe_mean(p_action1_list),
            "reward_given_action0": _safe_mean(reward_a0_list),
            "reward_given_action1": _safe_mean(reward_a1_list),
        }
        self.last_v5_metrics = diag
        for k, v in diag.items():
            try:
                self.logger.record(f"train/{k}", float(v))
            except Exception:
                pass


class ActionConditionalCriticPolicy(ActorCriticPolicy):
    """
    V5.2 policy:
      critic outputs Q(s,0), Q(s,1), and PPO uses Q(s,a_taken) as value baseline.
    """

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        n_actions = int(getattr(self.action_space, "n", 2))
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, n_actions).to(self.device)
        if self.ortho_init:
            self.value_net.apply(lambda m: self.init_weights(m, gain=1.0))
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

    def _q_values(self, latent_vf: th.Tensor) -> th.Tensor:
        return self.value_net(latent_vf)

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        latent_pi, latent_vf = self._latent(obs)
        dist = self._get_action_dist_from_latent(latent_pi)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        q = self._q_values(latent_vf)
        act = actions.long().flatten()
        q_taken = q.gather(1, act.unsqueeze(1)).squeeze(1)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, q_taken.unsqueeze(-1), log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        latent_pi, latent_vf = self._latent(obs)
        dist = self._get_action_dist_from_latent(latent_pi)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        q = self._q_values(latent_vf)
        act = actions.long().flatten()
        q_taken = q.gather(1, act.unsqueeze(1)).squeeze(1)
        return q_taken.unsqueeze(-1), log_prob, entropy

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        # For rollout tail bootstrap where action is unknown, use expectation under current policy.
        latent_pi, latent_vf = self._latent(obs)
        dist = self._get_action_dist_from_latent(latent_pi)
        q = self._q_values(latent_vf)
        p1 = _extract_p_action1(dist)
        if p1 is None or q.shape[1] < 2:
            return q.mean(dim=1, keepdim=True)
        p1 = p1.view(-1, 1)
        v = q[:, 0:1] * (1.0 - p1) + q[:, 1:2] * p1
        return v

    def predict_q_values(self, obs: th.Tensor) -> th.Tensor:
        _, latent_vf = self._latent(obs)
        return self._q_values(latent_vf)


class QCriticPPO(PPO):
    """
    V5.2 PPO wrapper:
      keep PPO actor objective unchanged, but collect Q/adv/action diagnostics.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.last_v5_metrics: Dict[str, Any] = {}
        super().__init__(*args, **kwargs)

    def train(self) -> None:
        super().train()
        q0_list: list[float] = []
        q1_list: list[float] = []
        q_taken_list: list[float] = []
        adv_mean_list: list[float] = []
        adv_std_list: list[float] = []
        adv0_mean_list: list[float] = []
        adv0_std_list: list[float] = []
        adv1_mean_list: list[float] = []
        adv1_std_list: list[float] = []
        action1_rate_list: list[float] = []
        p_action1_list: list[float] = []
        reward_a0_list: list[float] = []
        reward_a1_list: list[float] = []

        with th.no_grad():
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = _to_discrete_actions(self.action_space, rollout_data.actions)
                try:
                    if hasattr(self.policy, "predict_q_values"):
                        q_all = self.policy.predict_q_values(rollout_data.observations)
                        if q_all.ndim == 2 and q_all.shape[1] >= 2:
                            q0 = q_all[:, 0]
                            q1 = q_all[:, 1]
                            q_taken = q_all.gather(1, actions.long().unsqueeze(1)).squeeze(1)
                            q0_list.append(float(q0.mean().item()))
                            q1_list.append(float(q1.mean().item()))
                            q_taken_list.append(float(q_taken.mean().item()))
                except Exception:
                    pass

                adv = rollout_data.advantages.detach().float().flatten()
                if adv.numel() > 0:
                    adv_mean_list.append(float(adv.mean().item()))
                    adv_std_list.append(float(adv.std().item()))
                mask0 = actions == 0
                mask1 = actions == 1
                if int(mask0.sum().item()) > 0:
                    adv0 = adv[mask0]
                    adv0_mean_list.append(float(adv0.mean().item()))
                    adv0_std_list.append(float(adv0.std().item()))
                if int(mask1.sum().item()) > 0:
                    adv1 = adv[mask1]
                    adv1_mean_list.append(float(adv1.mean().item()))
                    adv1_std_list.append(float(adv1.std().item()))

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

        diag = {
            "qcritic_q0": _safe_mean(q0_list),
            "qcritic_q1": _safe_mean(q1_list),
            "qcritic_q_taken": _safe_mean(q_taken_list),
            "qcritic_adv_mean": _safe_mean(adv_mean_list),
            "qcritic_adv_std": _safe_mean(adv_std_list),
            "qcritic_adv0_mean": _safe_mean(adv0_mean_list),
            "qcritic_adv0_std": _safe_mean(adv0_std_list),
            "qcritic_adv1_mean": _safe_mean(adv1_mean_list),
            "qcritic_adv1_std": _safe_mean(adv1_std_list),
            "action1_rate": _safe_mean(action1_rate_list),
            "p_action1": _safe_mean(p_action1_list),
            "reward_given_action0": _safe_mean(reward_a0_list),
            "reward_given_action1": _safe_mean(reward_a1_list),
        }
        self.last_v5_metrics = diag
        for k, v in diag.items():
            try:
                self.logger.record(f"train/{k}", float(v))
            except Exception:
                pass


class AuxWeakPolicy(ActorCriticPolicy):
    """
    V5.3 policy: add a weak-label auxiliary binary head on shared policy latent.
    """

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        self.aux_head = nn.Linear(self.mlp_extractor.latent_dim_pi, 1).to(self.device)
        if self.ortho_init:
            self.aux_head.apply(lambda m: self.init_weights(m, gain=1.0))
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

    def evaluate_actions_with_aux(self, obs: th.Tensor, actions: th.Tensor):
        latent_pi, latent_vf = self._latent(obs)
        dist = self._get_action_dist_from_latent(latent_pi)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.value_net(latent_vf)
        aux_logits = self.aux_head(latent_pi).squeeze(-1)
        return values, log_prob, entropy, aux_logits


@dataclass
class AuxWeakConfig:
    lambda_aux: float = 0.2
    severity_threshold: float = 0.5


def _infer_weak_labels_from_obs(
    observations: th.Tensor,
    *,
    n_stack: int,
    oracle_ctx_dim: int,
    severity_threshold: float,
) -> Tuple[Optional[th.Tensor], str]:
    """
    Immediate-rule weak labels:
      y=1 iff latest-frame severity >= threshold
    This uses only current observation (no future information).
    """
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

        base_x_dim = int(x_dim - max(0, int(oracle_ctx_dim)))
        if base_x_dim <= 3 or (base_x_dim - 3) % 2 != 0:
            resolved = False
            for cand_ctx in range(0, min(64, x_dim + 1)):
                cand_base = x_dim - cand_ctx
                if cand_base <= 3:
                    continue
                if (cand_base - 3) % 2 == 0:
                    base_x_dim = int(cand_base)
                    resolved = True
                    break
            if not resolved:
                return None, "layout_unresolved"
        d = int((base_x_dim - 3) // 2)
        if d < 2:
            return None, "severity_not_available"

        # o_t = first D dims in latest frame, severity is o_t[1] in current pipeline.
        severity = x_now[:, 1]
        labels = (severity >= float(severity_threshold)).float()
        return labels, "obs_severity_rule"
    except Exception:
        return None, "weak_label_inference_error"


class AuxWeakPPO(PPO):
    """
    V5.3: PPO core + lambda_aux * BCE(aux_head(h), weak_label).
    If weak labels cannot be reliably inferred from immediate signals, aux is auto-disabled.
    """

    def __init__(
        self,
        *args,
        aux_lambda: float = 0.2,
        aux_severity_threshold: float = 0.5,
        **kwargs,
    ) -> None:
        self.aux_cfg = AuxWeakConfig(
            lambda_aux=float(max(0.0, aux_lambda)),
            severity_threshold=float(aux_severity_threshold),
        )
        self.last_v5_metrics: Dict[str, Any] = {}
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
        aux_losses = []
        y_mean_list = []
        yhat_mean_list = []
        aux_acc_list = []
        aux_auc_list = []
        action1_rate_list = []
        p_action1_list = []
        reward_a0_list = []
        reward_a1_list = []

        continue_training = True
        loss = None
        weak_label_source = "disabled"
        aux_enabled_any = False

        # Try to infer stack/layout from extractor (v3-compatible).
        ext = getattr(self.policy, "features_extractor", None)
        n_stack = int(getattr(ext, "n_stack", 1) or 1)
        oracle_ctx_dim = int(getattr(ext, "oracle_ctx_dim", 0) or 0)

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = _to_discrete_actions(self.action_space, rollout_data.actions)
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                if hasattr(self.policy, "evaluate_actions_with_aux"):
                    values, log_prob, entropy, aux_logits = self.policy.evaluate_actions_with_aux(
                        rollout_data.observations, actions
                    )
                else:
                    values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                    aux_logits = None
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

                aux_loss = th.zeros((), device=self.device)
                if aux_logits is not None and self.aux_cfg.lambda_aux > 0.0:
                    labels, source = _infer_weak_labels_from_obs(
                        rollout_data.observations,
                        n_stack=n_stack,
                        oracle_ctx_dim=oracle_ctx_dim,
                        severity_threshold=self.aux_cfg.severity_threshold,
                    )
                    weak_label_source = source
                    if labels is not None and labels.numel() == aux_logits.numel():
                        aux_enabled_any = True
                        labels = labels.to(device=aux_logits.device, dtype=aux_logits.dtype)
                        aux_loss = F.binary_cross_entropy_with_logits(aux_logits, labels)
                        aux_losses.append(float(aux_loss.item()))

                        with th.no_grad():
                            prob = th.sigmoid(aux_logits)
                            pred = (prob >= 0.5).float()
                            y_mean_list.append(float(labels.mean().item()))
                            yhat_mean_list.append(float(prob.mean().item()))
                            aux_acc = (pred == labels).float().mean()
                            aux_acc_list.append(float(aux_acc.item()))
                            y_np = labels.detach().cpu().numpy()
                            p_np = prob.detach().cpu().numpy()
                            auc_val = _binary_auc_score(y_np, p_np)
                            if auc_val is not None:
                                aux_auc_list.append(float(auc_val))

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.aux_cfg.lambda_aux * aux_loss

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

        diag = {
            "aux_enabled": int(aux_enabled_any),
            "aux_source": weak_label_source,
            "aux_loss": _safe_mean(aux_losses),
            "aux_y_mean": _safe_mean(y_mean_list),
            "aux_yhat_mean": _safe_mean(yhat_mean_list),
            "aux_acc": _safe_mean(aux_acc_list),
            "aux_auc": _safe_mean(aux_auc_list),
            "action1_rate": _safe_mean(action1_rate_list),
            "p_action1": _safe_mean(p_action1_list),
            "reward_given_action0": _safe_mean(reward_a0_list),
            "reward_given_action1": _safe_mean(reward_a1_list),
        }
        self.last_v5_metrics = diag
        for k, v in diag.items():
            try:
                if isinstance(v, (int, float, np.floating)):
                    self.logger.record(f"train/{k}", float(v))
            except Exception:
                pass

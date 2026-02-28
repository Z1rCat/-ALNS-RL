from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3 import PPO
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


@dataclass
class CvarPPOConfig:
    alpha: float = 0.30
    beta: float = 1.0
    w_max: float = 3.0
    eps: float = 1e-8


class CvarPPO(PPO):
    """
    V6 core:
    policy-loss reweighting on low-return tail samples (CVaR-style emphasis).
    """

    def __init__(
        self,
        *args,
        cvar_alpha: float = 0.30,
        cvar_beta: float = 1.0,
        cvar_w_max: float = 3.0,
        cvar_eps: float = 1e-8,
        **kwargs,
    ) -> None:
        alpha = float(cvar_alpha)
        beta = float(cvar_beta)
        w_max = float(cvar_w_max)
        eps = float(cvar_eps)
        self.cvar_cfg = CvarPPOConfig(
            alpha=float(min(1.0, max(0.0, alpha))),
            beta=float(max(0.0, beta)),
            w_max=float(max(1.0, w_max)),
            eps=float(max(1e-12, eps)),
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

        q_alpha_list: list[float] = []
        tail_frac_list: list[float] = []
        weight_mean_list: list[float] = []
        tail_reward_list: list[float] = []
        non_tail_reward_list: list[float] = []
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

                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_obj = th.min(policy_loss_1, policy_loss_2)

                # --- V6 core: CVaR-style low-return tail reweighting ---
                returns = rollout_data.returns.detach().float().flatten()
                if returns.numel() > 0:
                    q_alpha = th.quantile(returns, q=float(self.cvar_cfg.alpha))
                    tail_mask = returns <= q_alpha
                    sample_w = 1.0 + float(self.cvar_cfg.beta) * tail_mask.float()
                    sample_w = th.clamp(sample_w, min=1.0, max=float(self.cvar_cfg.w_max))
                    sample_w = sample_w / (sample_w.mean() + float(self.cvar_cfg.eps))
                else:
                    q_alpha = th.tensor(0.0, device=policy_obj.device, dtype=policy_obj.dtype)
                    tail_mask = th.zeros_like(policy_obj).bool()
                    sample_w = th.ones_like(policy_obj)

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

                # Diagnostics required by V6 acceptance checks.
                with th.no_grad():
                    if returns.numel() > 0:
                        q_alpha_list.append(float(q_alpha.item()))
                        tail_frac_list.append(float(tail_mask.float().mean().item()))
                        weight_mean_list.append(float(sample_w.mean().item()))
                        if int(tail_mask.sum().item()) > 0:
                            tail_reward_list.append(float(returns[tail_mask].mean().item()))
                        non_tail_mask = ~tail_mask
                        if int(non_tail_mask.sum().item()) > 0:
                            non_tail_reward_list.append(float(returns[non_tail_mask].mean().item()))

                    mask0 = actions == 0
                    mask1 = actions == 1
                    action1_rate_list.append(float(mask1.float().mean().item()))

                    try:
                        dist_now = self.policy.get_distribution(rollout_data.observations)
                        p1 = _extract_p_action1(dist_now)
                        if p1 is not None:
                            p_action1_list.append(float(p1.mean().item()))
                    except Exception:
                        pass

                    if returns.numel() > 0 and int(mask0.sum().item()) > 0:
                        reward_a0_list.append(float(returns[mask0].mean().item()))
                    if returns.numel() > 0 and int(mask1.sum().item()) > 0:
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
            "cvar_alpha": float(self.cvar_cfg.alpha),
            "cvar_beta": float(self.cvar_cfg.beta),
            "cvar_quantile": _safe_mean(q_alpha_list),
            "cvar_tail_frac": _safe_mean(tail_frac_list),
            "cvar_weight_mean": _safe_mean(weight_mean_list),
            "cvar_tail_reward": _safe_mean(tail_reward_list),
            "cvar_non_tail_reward": _safe_mean(non_tail_reward_list),
            "action1_rate": _safe_mean(action1_rate_list),
            "p_action1": _safe_mean(p_action1_list),
            "reward_given_action0": _safe_mean(reward_a0_list),
            "reward_given_action1": _safe_mean(reward_a1_list),
        }
        self.last_v6_metrics = diag
        for k, v in diag.items():
            try:
                self.logger.record(f"train/{k}", float(v))
            except Exception:
                pass

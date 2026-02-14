from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple
from collections import deque

import numpy as np

try:
    import torch as th
    import torch.nn as nn
    import torch.nn.functional as F
    from gymnasium import spaces
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import explained_variance
except Exception as exc:  # pragma: no cover
    raise ImportError("HAT-PPO-PDI requires torch + stable-baselines3.") from exc


@dataclass
class PDIConfig:
    # modulation
    kappa: float = 1.0
    temp_coef: float = 0.0
    # losses
    lambda_future: float = 0.2
    lambda_teach: float = 0.1
    lambda_actfail: float = 0.2
    # future window
    future_h: int = 5
    # teacher normalization
    gt_mean_norm: float = 100.0
    # phase classification (optional)
    phase_classes: int = 0
    # logging
    log_window: int = 50


class HATPdiPolicy(ActorCriticPolicy):
    """
    HAT + stage-conditioned actor/critic + PDI heads.

    The HAT (Transformer) runs in the features_extractor; this policy only consumes
    c_t (features) and stage_onehot (from the last token in obs) to select heads.
    """

    def __init__(self, *args, pdi_config: Optional[PDIConfig] = None, **kwargs) -> None:
        self.pdi_config = pdi_config or PDIConfig()
        super().__init__(*args, **kwargs)

        feat_dim = int(self.features_dim)
        action_dim = int(getattr(self.action_space, "n", 2))

        def _mlp(out_dim: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(feat_dim, 64),
                nn.ReLU(),
                nn.Linear(64, out_dim),
            )

        # stage-specific actor/critic
        self.actor_removal = _mlp(action_dim)
        self.actor_insertion = _mlp(action_dim)
        self.critic_removal = _mlp(1)
        self.critic_insertion = _mlp(1)

        # PDI heads
        self.future_head = _mlp(1)  # predicts future failure rate (logit)
        self.gt_mean_head = _mlp(1)  # teacher regression (normalized)
        if int(self.pdi_config.phase_classes) > 0:
            self.phase_head = _mlp(int(self.pdi_config.phase_classes))
        else:
            self.phase_head = None

        # action-conditional failure logits (stage-specific)
        self.fail_removal = _mlp(2)
        self.fail_insertion = _mlp(2)

        # cache (minibatch aligned)
        self._pdi_cache: Optional[Dict[str, th.Tensor]] = None

        # rolling logs
        win = int(max(1, self.pdi_config.log_window))
        self._log_future: Deque[float] = deque(maxlen=win)
        self._log_pref: Deque[float] = deque(maxlen=win)
        self._log_hard: Deque[float] = deque(maxlen=win)
        self._log_fail0: Deque[float] = deque(maxlen=win)
        self._log_fail1: Deque[float] = deque(maxlen=win)

    def _extract_stage(self, obs: th.Tensor) -> Optional[th.Tensor]:
        # obs shape: [B, H, D], stage onehot is the last 2 dims of last token
        if obs.ndim != 3 or obs.shape[-1] < 2:
            return None
        return obs[:, -1, -2:].float()

    def _stage_mask(self, stage_vec: Optional[th.Tensor]) -> th.Tensor:
        if stage_vec is None:
            # default to removal
            return th.zeros((1,), device=next(self.parameters()).device, dtype=th.bool)
        return stage_vec[:, 1] > stage_vec[:, 0]  # True => insertion

    def _select_heads(self, feat: th.Tensor, stage_vec: Optional[th.Tensor]):
        mask = self._stage_mask(stage_vec)
        logits_r = self.actor_removal(feat)
        logits_i = self.actor_insertion(feat)
        logits = th.where(mask.unsqueeze(-1), logits_i, logits_r)

        v_r = self.critic_removal(feat).squeeze(-1)
        v_i = self.critic_insertion(feat).squeeze(-1)
        values = th.where(mask, v_i, v_r)

        fail_r = self.fail_removal(feat)
        fail_i = self.fail_insertion(feat)
        fail_logits = th.where(mask.unsqueeze(-1), fail_i, fail_r)

        return logits, values, fail_logits

    def _apply_modulation(self, logits: th.Tensor, fail_logits: th.Tensor, future_logit: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # p_fail for each action
        p_fail = th.sigmoid(fail_logits)
        # P_pref = logit(p_fail0) - logit(p_fail1)
        p_pref = fail_logits[:, 0] - fail_logits[:, 1]
        # Pr(a=1) = sigmoid(logit1 + kappa * P_pref)
        kappa = float(self.pdi_config.kappa)
        if kappa != 0.0:
            logits = logits.clone()
            logits[:, 1] = logits[:, 1] + kappa * p_pref
        # temperature based on difficulty (future failure)
        temp_coef = float(self.pdi_config.temp_coef)
        if temp_coef != 0.0:
            hard = th.sigmoid(future_logit).detach()
            temp = 1.0 + temp_coef * hard
            logits = logits / temp.unsqueeze(-1)
        return logits, p_pref, p_fail

    def _update_logs(self, future_logit: th.Tensor, p_pref: th.Tensor, p_fail: th.Tensor) -> None:
        with th.no_grad():
            future = th.sigmoid(future_logit).mean().item()
            pref = p_pref.mean().item()
            hard = th.sigmoid(future_logit).mean().item()
            self._log_future.append(float(future))
            self._log_pref.append(float(pref))
            self._log_hard.append(float(hard))
            self._log_fail0.append(float(p_fail[:, 0].mean().item()))
            self._log_fail1.append(float(p_fail[:, 1].mean().item()))

    def get_pdi_log(self) -> Dict[str, float]:
        def _mean(xs):
            return float(sum(xs) / len(xs)) if xs else 0.0

        return {
            "pdi_future_mean": _mean(list(self._log_future)),
            "pdi_pref_mean": _mean(list(self._log_pref)),
            "pdi_hard_mean": _mean(list(self._log_hard)),
            "pdi_fail0_mean": _mean(list(self._log_fail0)),
            "pdi_fail1_mean": _mean(list(self._log_fail1)),
        }

    def pop_pdi_cache(self) -> Dict[str, th.Tensor]:
        if self._pdi_cache is None:
            raise RuntimeError("PDI cache empty: evaluate_actions() not called.")
        cache = self._pdi_cache
        self._pdi_cache = None
        return cache

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        feat = self.extract_features(obs)
        stage_vec = self._extract_stage(obs)
        logits, values, fail_logits = self._select_heads(feat, stage_vec)
        future_logit = self.future_head(feat).squeeze(-1)
        logits, p_pref, p_fail = self._apply_modulation(logits, fail_logits, future_logit)
        self._update_logs(future_logit, p_pref, p_fail)

        dist = self.action_dist.proba_distribution(logits)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        feat = self.extract_features(obs)
        stage_vec = self._extract_stage(obs)
        logits, values, fail_logits = self._select_heads(feat, stage_vec)
        future_logit = self.future_head(feat).squeeze(-1)
        logits, p_pref, p_fail = self._apply_modulation(logits, fail_logits, future_logit)

        # cache for PPO auxiliary losses (minibatch aligned)
        cache = {
            "future_logit": future_logit,
            "fail_logits": fail_logits,
            "p_pref": p_pref,
            "gt_mean_pred": self.gt_mean_head(feat).squeeze(-1),
        }
        if self.phase_head is not None:
            cache["phase_logits"] = self.phase_head(feat)
        self._pdi_cache = cache
        self._update_logs(future_logit, p_pref, p_fail)

        dist = self.action_dist.proba_distribution(logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return values, log_prob, entropy

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        feat = self.extract_features(obs)
        stage_vec = self._extract_stage(obs)
        _, values, _ = self._select_heads(feat, stage_vec)
        return values


class HATPdiPPO(PPO):
    """
    PPO with auxiliary PDI losses:
      L_total = L_PPO + lambda1 * L_future + lambda2 * L_teach + lambda3 * L_actfail
    """

    def __init__(
        self,
        *args,
        pdi_config: Optional[PDIConfig] = None,
        **kwargs,
    ):
        self.pdi_config = pdi_config or PDIConfig()
        super().__init__(*args, **kwargs)

    def train(self) -> None:
        # Copy of SB3 PPO.train() with added PDI losses.
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        pdi_future_losses = []
        pdi_teach_losses = []
        pdi_actfail_losses = []

        # Prepare indices and flattened arrays (include rewards)
        rb = self.rollout_buffer
        indices = np.random.permutation(rb.buffer_size * rb.n_envs)
        if not rb.generator_ready:
            for name in ["observations", "actions", "values", "log_probs", "advantages", "returns", "rewards"]:
                rb.__dict__[name] = rb.swap_and_flatten(rb.__dict__[name])
            rb.generator_ready = True

        rewards = rb.rewards.flatten()
        # future failure rate targets
        H = max(1, int(self.pdi_config.future_h))
        y_future = np.zeros_like(rewards, dtype=np.float32)
        for i in range(len(rewards)):
            end = min(len(rewards), i + H)
            y_future[i] = np.mean(1.0 - rewards[i:end]) if end > i else 0.0

        # teacher targets from dynamic_RL34959 (best-effort)
        gt_mean_targets = None
        phase_targets = None
        try:
            from core import dynamic_RL34959 as dyn
            if hasattr(dyn, "PDI_GT_MEAN_LIST"):
                gt_list = list(dyn.PDI_GT_MEAN_LIST)
                if len(gt_list) >= len(rewards):
                    gt_list = gt_list[-len(rewards):]
                gt_mean_targets = np.array(gt_list, dtype=np.float32)
            if hasattr(dyn, "PDI_PHASE_LIST"):
                phase_list = list(dyn.PDI_PHASE_LIST)
                if len(phase_list) >= len(rewards):
                    phase_list = phase_list[-len(rewards):]
                phase_targets = phase_list
        except Exception:
            gt_mean_targets = None
            phase_targets = None

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            start_idx = 0
            batch_size = self.batch_size
            size = rb.buffer_size * rb.n_envs
            while start_idx < size:
                batch_inds = indices[start_idx : start_idx + batch_size]
                start_idx += batch_size
                data = (
                    rb.observations[batch_inds],
                    rb.actions[batch_inds],
                    rb.values[batch_inds].flatten(),
                    rb.log_probs[batch_inds].flatten(),
                    rb.advantages[batch_inds].flatten(),
                    rb.returns[batch_inds].flatten(),
                )
                observations, actions, old_values, old_log_prob, advantages, returns = map(self.rollout_buffer.to_torch, data)

                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(observations, actions)
                values = values.flatten()

                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())

                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = old_values + th.clamp(values - old_values, -clip_range_vf, clip_range_vf)
                value_loss = F.mse_loss(returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # PDI auxiliary losses
                pdi_cache = self.policy.pop_pdi_cache()
                future_logit = pdi_cache["future_logit"]
                fail_logits = pdi_cache["fail_logits"]
                gt_pred = pdi_cache["gt_mean_pred"]

                y_fut = th.tensor(y_future[batch_inds], device=observations.device, dtype=th.float32)
                loss_future = F.binary_cross_entropy_with_logits(future_logit, y_fut)
                pdi_future_losses.append(float(loss_future.detach().cpu().item()))

                loss_teach = th.tensor(0.0, device=observations.device)
                if gt_mean_targets is not None:
                    gt = gt_mean_targets[batch_inds]
                    gt = np.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0)
                    gt = th.tensor(gt, device=observations.device, dtype=th.float32) / max(1e-6, float(self.pdi_config.gt_mean_norm))
                    loss_teach = F.mse_loss(gt_pred, gt)
                    pdi_teach_losses.append(float(loss_teach.detach().cpu().item()))

                loss_actfail = th.tensor(0.0, device=observations.device)
                try:
                    fail_target = 1.0 - th.tensor(rewards[batch_inds], device=observations.device, dtype=th.float32)
                    action_idx = actions.long().view(-1, 1)
                    fail_logit_action = th.gather(fail_logits, 1, action_idx).squeeze(1)
                    loss_actfail = F.binary_cross_entropy_with_logits(fail_logit_action, fail_target)
                    pdi_actfail_losses.append(float(loss_actfail.detach().cpu().item()))
                except Exception:
                    pass

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + float(self.pdi_config.lambda_future) * loss_future
                    + float(self.pdi_config.lambda_teach) * loss_teach
                    + float(self.pdi_config.lambda_actfail) * loss_actfail
                )

                with th.no_grad():
                    log_ratio = log_prob - old_log_prob
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

        explained_var = explained_variance(rb.values.flatten(), rb.returns.flatten())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if pdi_future_losses:
            self.logger.record("train/pdi_future_loss", float(np.mean(pdi_future_losses)))
        if pdi_teach_losses:
            self.logger.record("train/pdi_teach_loss", float(np.mean(pdi_teach_losses)))
        if pdi_actfail_losses:
            self.logger.record("train/pdi_actfail_loss", float(np.mean(pdi_actfail_losses)))
        # expose for external csv logger (dynamic_RL34959.log_training_row)
        self.last_pdi_losses = {
            "pdi_future_loss": float(np.mean(pdi_future_losses)) if pdi_future_losses else "",
            "pdi_teach_loss": float(np.mean(pdi_teach_losses)) if pdi_teach_losses else "",
            "pdi_actfail_loss": float(np.mean(pdi_actfail_losses)) if pdi_actfail_losses else "",
        }
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

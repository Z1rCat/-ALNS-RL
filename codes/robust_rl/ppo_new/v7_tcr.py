from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

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


def _safe_mean(values: List[float]) -> float:
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
class TCRConfig:
    enable: bool = True
    action1_rate_threshold: float = 0.08
    reward_gap_threshold: float = 0.05
    min_action1_samples: int = 12
    min_action0_samples: int = 12
    min_group_steps: int = 32
    trigger_window_steps: int = 400
    a1_reward_quantile: float = 0.70
    reward_margin_over_a0: float = 0.0
    buffer_size_per_group: int = 512
    aux_coef: float = 0.03
    aux_batch_size: int = 64
    teacher_mode: str = "none"  # none | biased_student
    teacher_action1_logit_bias: float = 0.35
    eps: float = 1e-8


class TriggeredCounterfactualPPO(PPO):
    """
    Triggered Counterfactual/Teacher Replay (TCR):
    - Trigger condition per group:
      action1_rate <= tau_rho, reward_gap = E[r|a=1]-E[r|a=0] >= tau_r,
      N(a=1) and N(a=0) above minimums.
    - Collect high-value a=1 samples into a per-group replay buffer.
    - Keep PPO on-policy objective unchanged; add a small auxiliary term
      on replay observations only.
    """

    def __init__(
        self,
        *args,
        tcr_enable: bool = True,
        tcr_action1_rate_threshold: float = 0.08,
        tcr_reward_gap_threshold: float = 0.05,
        tcr_min_action1_samples: int = 12,
        tcr_min_action0_samples: int = 12,
        tcr_min_group_steps: int = 32,
        tcr_trigger_window_steps: int = 400,
        tcr_a1_reward_quantile: float = 0.70,
        tcr_reward_margin_over_a0: float = 0.0,
        tcr_buffer_size_per_group: int = 512,
        tcr_aux_coef: float = 0.03,
        tcr_aux_batch_size: int = 64,
        tcr_teacher_mode: str = "none",
        tcr_teacher_action1_logit_bias: float = 0.35,
        tcr_eps: float = 1e-8,
        **kwargs,
    ) -> None:
        self.tcr_cfg = TCRConfig(
            enable=bool(tcr_enable),
            action1_rate_threshold=float(max(0.0, min(1.0, tcr_action1_rate_threshold))),
            reward_gap_threshold=float(tcr_reward_gap_threshold),
            min_action1_samples=int(max(1, tcr_min_action1_samples)),
            min_action0_samples=int(max(1, tcr_min_action0_samples)),
            min_group_steps=int(max(1, tcr_min_group_steps)),
            trigger_window_steps=int(max(16, tcr_trigger_window_steps)),
            a1_reward_quantile=float(max(0.0, min(1.0, tcr_a1_reward_quantile))),
            reward_margin_over_a0=float(tcr_reward_margin_over_a0),
            buffer_size_per_group=int(max(16, tcr_buffer_size_per_group)),
            aux_coef=float(max(0.0, tcr_aux_coef)),
            aux_batch_size=int(max(1, tcr_aux_batch_size)),
            teacher_mode=str(tcr_teacher_mode or "none").strip().lower(),
            teacher_action1_logit_bias=float(tcr_teacher_action1_logit_bias),
            eps=float(max(1e-12, tcr_eps)),
        )
        if self.tcr_cfg.teacher_mode not in {"none", "biased_student"}:
            self.tcr_cfg.teacher_mode = "none"

        self._tcr_buffer: Dict[str, Deque[Dict[str, Any]]] = {}
        self._tcr_recent_records: Deque[Dict[str, Any]] = deque(maxlen=self.tcr_cfg.trigger_window_steps)
        self._tcr_active_groups: List[str] = []
        self.last_tcr_metrics: Dict[str, Any] = {}
        super().__init__(*args, **kwargs)

    def _tcr_total_buffer_size(self) -> int:
        return int(sum(len(buf) for buf in self._tcr_buffer.values()))

    def _tcr_push_sample(self, group: str, obs: np.ndarray, action: int, reward: float) -> None:
        group_key = str(group or "unknown")
        if group_key not in self._tcr_buffer:
            self._tcr_buffer[group_key] = deque(maxlen=self.tcr_cfg.buffer_size_per_group)
        obs_arr = np.asarray(obs, dtype=np.float32)
        self._tcr_buffer[group_key].append(
            {
                "obs": obs_arr.copy(),
                "action": int(action),
                "reward": float(reward),
            }
        )

    def _tcr_sample_obs_batch(self, groups: List[str], batch_size: int) -> Optional[np.ndarray]:
        samples: List[Dict[str, Any]] = []
        for group in groups:
            samples.extend(list(self._tcr_buffer.get(group, [])))
        if not samples:
            return None

        first_shape = None
        filtered: List[Dict[str, Any]] = []
        for sample in samples:
            obs = np.asarray(sample.get("obs"), dtype=np.float32)
            if obs.size == 0:
                continue
            if first_shape is None:
                first_shape = obs.shape
            if obs.shape == first_shape:
                filtered.append(sample)
        if not filtered:
            return None

        n = min(int(batch_size), len(filtered))
        if n <= 0:
            return None
        idx = np.random.choice(len(filtered), size=n, replace=False)
        obs_batch = np.stack([np.asarray(filtered[i]["obs"], dtype=np.float32) for i in idx], axis=0)
        return obs_batch.astype(np.float32)

    def _tcr_auxiliary_loss(self) -> Optional[th.Tensor]:
        if not self.tcr_cfg.enable or self.tcr_cfg.aux_coef <= 0.0:
            return None
        if not self._tcr_active_groups:
            return None
        obs_batch = self._tcr_sample_obs_batch(self._tcr_active_groups, self.tcr_cfg.aux_batch_size)
        if obs_batch is None:
            return None

        obs_t = th.as_tensor(obs_batch, device=self.device)
        dist = self.policy.get_distribution(obs_t)
        p1 = _extract_p_action1(dist)
        if p1 is None:
            return None

        if self.tcr_cfg.teacher_mode == "biased_student":
            try:
                logits = dist.distribution.logits
                if logits is None or logits.shape[-1] < 2:
                    return -th.log(th.clamp(p1, min=self.tcr_cfg.eps)).mean()
                with th.no_grad():
                    t_logits = logits.detach().clone()
                    t_logits[..., 1] = t_logits[..., 1] + float(self.tcr_cfg.teacher_action1_logit_bias)
                    teacher_probs = th.softmax(t_logits, dim=-1)
                log_probs = th.log_softmax(logits, dim=-1)
                return F.kl_div(log_probs, teacher_probs, reduction="batchmean")
            except Exception:
                return -th.log(th.clamp(p1, min=self.tcr_cfg.eps)).mean()

        # V1 default: behavior-cloning style push for action=1 on selected replay samples.
        return -th.log(th.clamp(p1, min=self.tcr_cfg.eps)).mean()

    def tcr_consume_rollout(self, records: List[Dict[str, Any]]) -> None:
        """
        Consumes rollout-level records collected by callback:
        each record needs {group, action, reward, obs}.
        """
        if not self.tcr_cfg.enable:
            self._tcr_active_groups = []
            self.last_tcr_metrics = {
                "tcr_enabled": 0,
                "tcr_rollout_groups": 0,
                "tcr_trigger_events": 0,
                "tcr_triggered_groups": 0,
                "tcr_trigger_group_ids": "",
                "tcr_new_samples": 0,
                "tcr_buffer_size": int(self._tcr_total_buffer_size()),
                "tcr_action1_rate_trigger_mean": 0.0,
                "tcr_reward_gap_trigger_mean": 0.0,
                "tcr_aux_loss": 0.0,
                "tcr_aux_applied_batches": 0,
                "tcr_teacher_mode": self.tcr_cfg.teacher_mode,
            }
            return

        normalized_current: List[Dict[str, Any]] = []
        for rec in records or []:
            try:
                group = str(rec.get("group", "") or "unknown")
                action = int(rec.get("action", 0))
                reward = float(rec.get("reward", 0.0))
                obs = np.asarray(rec.get("obs"), dtype=np.float32)
            except Exception:
                continue
            if obs.size == 0:
                continue
            item = {"group": group, "action": action, "reward": reward, "obs": obs}
            normalized_current.append(item)
            self._tcr_recent_records.append(item)

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for rec in list(self._tcr_recent_records):
            grouped.setdefault(str(rec["group"]), []).append(rec)

        current_grouped: Dict[str, List[Dict[str, Any]]] = {}
        for rec in normalized_current:
            current_grouped.setdefault(str(rec["group"]), []).append(rec)

        triggered_groups: List[str] = []
        trigger_action1_rates: List[float] = []
        trigger_reward_gaps: List[float] = []
        new_samples = 0

        for group, rows in grouped.items():
            n = len(rows)
            if n < self.tcr_cfg.min_group_steps:
                continue

            a0_rewards = [r["reward"] for r in rows if int(r["action"]) == 0]
            a1_rows = [r for r in rows if int(r["action"]) == 1]
            a1_rewards = [r["reward"] for r in a1_rows]
            n0 = len(a0_rewards)
            n1 = len(a1_rewards)
            if n1 < self.tcr_cfg.min_action1_samples or n0 < self.tcr_cfg.min_action0_samples:
                continue

            action1_rate = float(n1 / max(1, n))
            r0 = float(np.mean(a0_rewards)) if a0_rewards else 0.0
            r1 = float(np.mean(a1_rewards)) if a1_rewards else 0.0
            reward_gap = float(r1 - r0)

            collapse_hit = action1_rate <= float(self.tcr_cfg.action1_rate_threshold)
            value_hit = reward_gap >= float(self.tcr_cfg.reward_gap_threshold)
            if not (collapse_hit and value_hit):
                continue

            triggered_groups.append(group)
            trigger_action1_rates.append(action1_rate)
            trigger_reward_gaps.append(reward_gap)

            q = float(np.quantile(np.asarray(a1_rewards, dtype=np.float32), self.tcr_cfg.a1_reward_quantile))
            keep_threshold = max(q, r0 + float(self.tcr_cfg.reward_margin_over_a0))
            source_rows = current_grouped.get(group, [])
            for sample in source_rows:
                if int(sample["action"]) != 1:
                    continue
                if float(sample["reward"]) >= keep_threshold:
                    self._tcr_push_sample(
                        group=group,
                        obs=np.asarray(sample["obs"], dtype=np.float32),
                        action=1,
                        reward=float(sample["reward"]),
                    )
                    new_samples += 1

        self._tcr_active_groups = list(triggered_groups)
        self.last_tcr_metrics = {
            "tcr_enabled": 1,
            "tcr_rollout_groups": int(len(grouped)),
            "tcr_trigger_events": int(len(triggered_groups)),
            "tcr_triggered_groups": int(len(triggered_groups)),
            "tcr_trigger_group_ids": "|".join(triggered_groups[:8]),
            "tcr_new_samples": int(new_samples),
            "tcr_buffer_size": int(self._tcr_total_buffer_size()),
            "tcr_action1_rate_trigger_mean": _safe_mean(trigger_action1_rates),
            "tcr_reward_gap_trigger_mean": _safe_mean(trigger_reward_gaps),
            "tcr_aux_loss": 0.0,
            "tcr_aux_applied_batches": 0,
            "tcr_teacher_mode": self.tcr_cfg.teacher_mode,
        }

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses: List[float] = []
        pg_losses: List[float] = []
        value_losses: List[float] = []
        clip_fractions: List[float] = []
        approx_kl_divs_all: List[float] = []

        action1_rate_list: List[float] = []
        p_action1_list: List[float] = []
        reward_a0_list: List[float] = []
        reward_a1_list: List[float] = []
        tcr_aux_loss_list: List[float] = []
        tcr_aux_applied_batches = 0

        continue_training = True
        loss = None

        for epoch in range(self.n_epochs):
            approx_kl_divs: List[float] = []
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

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Auxiliary replay/distillation head, enabled only when trigger fires.
                aux_loss = self._tcr_auxiliary_loss()
                if aux_loss is not None and self.tcr_cfg.aux_coef > 0.0:
                    loss = loss + float(self.tcr_cfg.aux_coef) * aux_loss
                    tcr_aux_loss_list.append(float(aux_loss.item()))
                    tcr_aux_applied_batches += 1

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
                        dist_now = self.policy.get_distribution(rollout_data.observations)
                        p1 = _extract_p_action1(dist_now)
                        if p1 is not None:
                            p_action1_list.append(float(p1.mean().item()))
                    except Exception:
                        pass

                    returns = rollout_data.returns.detach().float().flatten()
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

        metrics = dict(self.last_tcr_metrics or {})
        metrics.update(
            {
                "action1_rate": _safe_mean(action1_rate_list),
                "p_action1": _safe_mean(p_action1_list),
                "reward_given_action0": _safe_mean(reward_a0_list),
                "reward_given_action1": _safe_mean(reward_a1_list),
                "tcr_aux_loss": _safe_mean(tcr_aux_loss_list),
                "tcr_aux_applied_batches": int(tcr_aux_applied_batches),
                "tcr_buffer_size": int(self._tcr_total_buffer_size()),
                "tcr_teacher_mode": self.tcr_cfg.teacher_mode,
            }
        )
        self.last_tcr_metrics = metrics
        for key, value in metrics.items():
            try:
                if isinstance(value, (int, float, np.integer, np.floating)):
                    self.logger.record(f"train/{key}", float(value))
            except Exception:
                pass

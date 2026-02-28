from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import torch as th

from .v7_tcr import TriggeredCounterfactualPPO, _extract_p_action1, _safe_mean


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -40.0, 40.0))
    return float(1.0 / (1.0 + np.exp(-x)))


@dataclass
class V8AConfig:
    quality_temp: float = 0.15
    support_power: float = 0.50
    novelty_power: float = 1.00
    min_keep_weight: float = 0.15
    max_keep_weight: float = 4.00
    kl_guard_coef: float = 0.02
    ref_mix: float = 0.50


@dataclass
class V9AConfig:
    kappa_base: float = 1.00
    kappa_slope: float = 2.00
    kappa_min: float = 0.50
    kappa_max: float = 3.00
    gap_ema_decay: float = 0.80


class PPOPostV8A(TriggeredCounterfactualPPO):
    """
    PPO_NEW V8-A:
    - Keep V3 pipeline unchanged.
    - Trigger by collapse/value conditions on per-group rollout stats.
    - Build a quality-weighted replay of high-value action=1 samples.
    - Keep on-policy PPO objective unchanged, add a small weighted aux term.
    """

    def __init__(
        self,
        *args,
        v8_quality_temp: float = 0.15,
        v8_support_power: float = 0.50,
        v8_novelty_power: float = 1.00,
        v8_min_keep_weight: float = 0.15,
        v8_max_keep_weight: float = 4.00,
        v8_kl_guard_coef: float = 0.02,
        v8_ref_mix: float = 0.50,
        **kwargs,
    ) -> None:
        self.v8_cfg = V8AConfig(
            quality_temp=float(max(1e-6, v8_quality_temp)),
            support_power=float(max(0.0, v8_support_power)),
            novelty_power=float(max(0.0, v8_novelty_power)),
            min_keep_weight=float(max(0.0, v8_min_keep_weight)),
            max_keep_weight=float(max(0.05, v8_max_keep_weight)),
            kl_guard_coef=float(max(0.0, v8_kl_guard_coef)),
            ref_mix=float(np.clip(v8_ref_mix, 0.0, 1.0)),
        )
        if self.v8_cfg.max_keep_weight < self.v8_cfg.min_keep_weight:
            self.v8_cfg.max_keep_weight = self.v8_cfg.min_keep_weight

        self._v8_rollout_stats: Dict[str, Any] = {}
        self._v8_aux_boost_history: List[float] = []
        self._v8_aux_kl_history: List[float] = []
        self._v8_aux_weight_history: List[float] = []
        self._v8_aux_scale_history: List[float] = []
        self.last_v8_metrics: Dict[str, Any] = {}
        super().__init__(*args, **kwargs)

    @staticmethod
    def _clip_weight(value: float, w_min: float, w_max: float) -> float:
        return float(np.clip(float(value), float(w_min), float(w_max)))

    def _v8_reset_aux_trackers(self) -> None:
        self._v8_aux_boost_history = []
        self._v8_aux_kl_history = []
        self._v8_aux_weight_history = []
        self._v8_aux_scale_history = []

    def _estimate_action1_prob(self, obs: np.ndarray) -> float:
        try:
            obs_t = th.as_tensor(np.asarray(obs, dtype=np.float32)[None, ...], device=self.device)
            with th.no_grad():
                dist = self.policy.get_distribution(obs_t)
                p1 = _extract_p_action1(dist)
                if p1 is None or int(p1.numel()) <= 0:
                    return 0.5
                return float(np.clip(float(p1.reshape(-1)[0].item()), 1e-6, 1.0 - 1e-6))
        except Exception:
            return 0.5

    def _v8_push_sample(
        self,
        group: str,
        obs: np.ndarray,
        action: int,
        reward: float,
        sample_weight: float,
        p1_ref: float,
        group_gap: float,
    ) -> None:
        group_key = str(group or "unknown")
        if group_key not in self._tcr_buffer:
            self._tcr_buffer[group_key] = deque(maxlen=self.tcr_cfg.buffer_size_per_group)
        self._tcr_buffer[group_key].append(
            {
                "obs": np.asarray(obs, dtype=np.float32).copy(),
                "action": int(action),
                "reward": float(reward),
                "weight": float(sample_weight),
                "p1_ref": float(np.clip(p1_ref, 1e-6, 1.0 - 1e-6)),
                "group_gap": float(group_gap),
            }
        )

    def _v8_sample_batch(self, groups: List[str], batch_size: int) -> Optional[Dict[str, np.ndarray]]:
        rows: List[Dict[str, Any]] = []
        for group in groups:
            rows.extend(list(self._tcr_buffer.get(group, [])))
        if not rows:
            return None

        first_shape = None
        filtered: List[Dict[str, Any]] = []
        for row in rows:
            obs = np.asarray(row.get("obs"), dtype=np.float32)
            if obs.size == 0:
                continue
            if first_shape is None:
                first_shape = obs.shape
            if obs.shape != first_shape:
                continue
            filtered.append(row)
        if not filtered:
            return None

        n = min(int(batch_size), len(filtered))
        if n <= 0:
            return None
        idx = np.random.choice(len(filtered), size=n, replace=False)
        obs_batch = np.stack([np.asarray(filtered[i]["obs"], dtype=np.float32) for i in idx], axis=0).astype(np.float32)
        w_batch = np.asarray([float(filtered[i].get("weight", 1.0)) for i in idx], dtype=np.float32)
        p1_ref_batch = np.asarray([float(filtered[i].get("p1_ref", 0.5)) for i in idx], dtype=np.float32)
        return {
            "obs": obs_batch,
            "weight": w_batch,
            "p1_ref": p1_ref_batch,
        }

    def _v8_aux_scale(self) -> float:
        return 1.0

    def _v8_compute_aux_terms(self, batch: Dict[str, np.ndarray]) -> Optional[Dict[str, th.Tensor]]:
        obs_batch = batch.get("obs", None)
        if obs_batch is None:
            return None

        obs_t = th.as_tensor(obs_batch, device=self.device)
        dist = self.policy.get_distribution(obs_t)
        p1 = _extract_p_action1(dist)
        if p1 is None:
            return None
        eps = float(self.tcr_cfg.eps)

        w_t = th.as_tensor(batch["weight"], device=self.device, dtype=th.float32)
        ref_t = th.as_tensor(batch["p1_ref"], device=self.device, dtype=th.float32)
        if w_t.numel() != p1.numel() or ref_t.numel() != p1.numel():
            return None

        # Keep relative weights but normalize batch scale.
        w_t = th.clamp(w_t, min=0.0)
        w_t = w_t / th.clamp(w_t.mean(), min=eps)
        p1 = th.clamp(p1.float(), min=eps, max=1.0 - eps)
        ref_t = th.clamp(ref_t.float(), min=eps, max=1.0 - eps)

        boost_vec = -th.log(p1)
        boost_loss = th.mean(w_t * boost_vec)

        cur_probs = th.stack([1.0 - p1, p1], dim=1)
        ref_probs = th.stack([1.0 - ref_t, ref_t], dim=1)
        # Optional mix with uniform prior to avoid hard overfitting to stale reference.
        if self.v8_cfg.ref_mix > 0.0:
            mix = float(self.v8_cfg.ref_mix)
            ref_probs = (1.0 - mix) * ref_probs + mix * 0.5
        ref_probs = th.clamp(ref_probs, min=eps, max=1.0)
        cur_probs = th.clamp(cur_probs, min=eps, max=1.0)
        kl_vec = th.sum(ref_probs * (th.log(ref_probs) - th.log(cur_probs)), dim=1)
        kl_loss = th.mean(w_t * kl_vec)
        return {
            "boost_loss": boost_loss,
            "kl_loss": kl_loss,
            "weight_mean": th.mean(w_t),
        }

    def _tcr_auxiliary_loss(self) -> Optional[th.Tensor]:
        if not self.tcr_cfg.enable or self.tcr_cfg.aux_coef <= 0.0:
            return None
        if not self._tcr_active_groups:
            return None

        batch = self._v8_sample_batch(self._tcr_active_groups, self.tcr_cfg.aux_batch_size)
        if batch is None:
            return None
        terms = self._v8_compute_aux_terms(batch)
        if terms is None:
            return None

        scale = float(max(0.0, self._v8_aux_scale()))
        aux_loss = scale * terms["boost_loss"] + float(self.v8_cfg.kl_guard_coef) * terms["kl_loss"]

        try:
            self._v8_aux_boost_history.append(float(terms["boost_loss"].detach().cpu().item()))
            self._v8_aux_kl_history.append(float(terms["kl_loss"].detach().cpu().item()))
            self._v8_aux_weight_history.append(float(terms["weight_mean"].detach().cpu().item()))
            self._v8_aux_scale_history.append(float(scale))
        except Exception:
            pass
        return aux_loss

    def tcr_consume_rollout(self, records: List[Dict[str, Any]]) -> None:
        if not self.tcr_cfg.enable:
            self._tcr_active_groups = []
            self._v8_rollout_stats = {
                "v8_trigger_quality_mean": 0.0,
                "v8_trigger_gap_score_mean": 0.0,
                "v8_selected_weight_mean": 0.0,
            }
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
                "tcr_teacher_mode": "none",
            }
            self.last_v8_metrics = dict(self.last_tcr_metrics)
            self.last_v8_metrics.update(
                {
                    "v8_enabled": 0,
                    "v8_trigger_quality_mean": 0.0,
                    "v8_trigger_gap_score_mean": 0.0,
                    "v8_selected_weight_mean": 0.0,
                    "v8_aux_boost_loss": 0.0,
                    "v8_aux_kl_guard_loss": 0.0,
                    "v8_aux_scale": 0.0,
                }
            )
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
        trigger_quality: List[float] = []
        trigger_gap_score: List[float] = []
        selected_weights: List[float] = []
        new_samples = 0

        tau_rho = float(self.tcr_cfg.action1_rate_threshold)
        tau_gap = float(self.tcr_cfg.reward_gap_threshold)
        eps = float(self.tcr_cfg.eps)

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

            collapse_hit = action1_rate <= tau_rho
            value_hit = reward_gap >= tau_gap
            if not (collapse_hit and value_hit):
                continue

            collapse_strength = max(0.0, (tau_rho - action1_rate) / max(tau_rho, eps))
            value_strength = _sigmoid((reward_gap - tau_gap) / max(self.v8_cfg.quality_temp, eps))
            support_strength = min(1.0, float(n1) / max(1.0, 2.0 * float(self.tcr_cfg.min_action1_samples)))
            gap_score = collapse_strength * max(0.0, reward_gap - tau_gap)
            group_quality = (
                (collapse_strength ** self.v8_cfg.novelty_power)
                * value_strength
                * (support_strength ** self.v8_cfg.support_power)
            )
            group_quality = self._clip_weight(
                group_quality,
                self.v8_cfg.min_keep_weight,
                self.v8_cfg.max_keep_weight,
            )

            triggered_groups.append(group)
            trigger_action1_rates.append(action1_rate)
            trigger_reward_gaps.append(reward_gap)
            trigger_quality.append(group_quality)
            trigger_gap_score.append(gap_score)

            q = float(np.quantile(np.asarray(a1_rewards, dtype=np.float32), self.tcr_cfg.a1_reward_quantile))
            keep_threshold = max(q, r0 + float(self.tcr_cfg.reward_margin_over_a0))
            for sample in current_grouped.get(group, []):
                if int(sample["action"]) != 1:
                    continue
                if float(sample["reward"]) < keep_threshold:
                    continue
                p1_ref = self._estimate_action1_prob(np.asarray(sample["obs"], dtype=np.float32))
                novelty = 1.0 - p1_ref
                sample_weight = self._clip_weight(
                    group_quality * (1.0 + novelty),
                    self.v8_cfg.min_keep_weight,
                    self.v8_cfg.max_keep_weight,
                )
                self._v8_push_sample(
                    group=group,
                    obs=np.asarray(sample["obs"], dtype=np.float32),
                    action=1,
                    reward=float(sample["reward"]),
                    sample_weight=sample_weight,
                    p1_ref=p1_ref,
                    group_gap=reward_gap,
                )
                selected_weights.append(sample_weight)
                new_samples += 1

        self._tcr_active_groups = list(triggered_groups)
        self._v8_rollout_stats = {
            "v8_trigger_quality_mean": _safe_mean(trigger_quality),
            "v8_trigger_gap_score_mean": _safe_mean(trigger_gap_score),
            "v8_selected_weight_mean": _safe_mean(selected_weights),
        }

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
            "tcr_teacher_mode": "none",
        }
        self.last_v8_metrics = dict(self.last_tcr_metrics)
        self.last_v8_metrics.update(
            {
                "v8_enabled": 1,
                "v8_trigger_quality_mean": _safe_mean(trigger_quality),
                "v8_trigger_gap_score_mean": _safe_mean(trigger_gap_score),
                "v8_selected_weight_mean": _safe_mean(selected_weights),
                "v8_aux_boost_loss": 0.0,
                "v8_aux_kl_guard_loss": 0.0,
                "v8_aux_scale": self._v8_aux_scale(),
            }
        )

    def train(self) -> None:
        self._v8_reset_aux_trackers()
        super().train()
        metrics = dict(self.last_tcr_metrics or {})
        metrics.update(
            {
                "v8_enabled": int(bool(self.tcr_cfg.enable)),
                "v8_trigger_quality_mean": float(self._v8_rollout_stats.get("v8_trigger_quality_mean", 0.0)),
                "v8_trigger_gap_score_mean": float(self._v8_rollout_stats.get("v8_trigger_gap_score_mean", 0.0)),
                "v8_selected_weight_mean": float(self._v8_rollout_stats.get("v8_selected_weight_mean", 0.0)),
                "v8_aux_boost_loss": _safe_mean(self._v8_aux_boost_history),
                "v8_aux_kl_guard_loss": _safe_mean(self._v8_aux_kl_history),
                "v8_aux_weight_mean": _safe_mean(self._v8_aux_weight_history),
                "v8_aux_scale": _safe_mean(self._v8_aux_scale_history),
            }
        )
        self.last_v8_metrics = metrics
        # Keep compatibility with existing training logger branch.
        self.last_tcr_metrics = dict(metrics)
        for key, value in metrics.items():
            try:
                if isinstance(value, (int, float, np.integer, np.floating)):
                    self.logger.record(f"train/{key}", float(value))
            except Exception:
                pass


class PPOPostV9A(PPOPostV8A):
    """
    PPO_NEW V9-A:
    - V8-A + adaptive auxiliary scale kappa(gap_score) with EMA smoothing.
    """

    def __init__(
        self,
        *args,
        v9_kappa_base: float = 1.00,
        v9_kappa_slope: float = 2.00,
        v9_kappa_min: float = 0.50,
        v9_kappa_max: float = 3.00,
        v9_gap_ema_decay: float = 0.80,
        **kwargs,
    ) -> None:
        self.v9_cfg = V9AConfig(
            kappa_base=float(v9_kappa_base),
            kappa_slope=float(v9_kappa_slope),
            kappa_min=float(v9_kappa_min),
            kappa_max=float(v9_kappa_max),
            gap_ema_decay=float(np.clip(v9_gap_ema_decay, 0.0, 0.999)),
        )
        if self.v9_cfg.kappa_max < self.v9_cfg.kappa_min:
            self.v9_cfg.kappa_max = self.v9_cfg.kappa_min
        self._v9_gap_score_ema = 0.0
        self._v9_kappa = float(np.clip(self.v9_cfg.kappa_base, self.v9_cfg.kappa_min, self.v9_cfg.kappa_max))
        self._v9_kappa_history: List[float] = []
        self.last_v9_metrics: Dict[str, Any] = {}
        super().__init__(*args, **kwargs)

    def _v8_aux_scale(self) -> float:
        return float(self._v9_kappa)

    def tcr_consume_rollout(self, records: List[Dict[str, Any]]) -> None:
        super().tcr_consume_rollout(records)
        gap_score = float(self._v8_rollout_stats.get("v8_trigger_gap_score_mean", 0.0))
        decay = float(self.v9_cfg.gap_ema_decay)
        self._v9_gap_score_ema = decay * float(self._v9_gap_score_ema) + (1.0 - decay) * gap_score
        raw_kappa = float(self.v9_cfg.kappa_base + self.v9_cfg.kappa_slope * self._v9_gap_score_ema)
        self._v9_kappa = float(np.clip(raw_kappa, self.v9_cfg.kappa_min, self.v9_cfg.kappa_max))

        updated = dict(self.last_v8_metrics or self.last_tcr_metrics or {})
        updated.update(
            {
                "v9_enabled": int(bool(self.tcr_cfg.enable)),
                "v9_gap_score_ema": float(self._v9_gap_score_ema),
                "v9_kappa": float(self._v9_kappa),
                "v9_kappa_base": float(self.v9_cfg.kappa_base),
            }
        )
        self.last_v8_metrics = dict(updated)
        self.last_tcr_metrics = dict(updated)

    def train(self) -> None:
        self._v9_kappa_history = []
        super().train()
        if self._v8_aux_scale_history:
            self._v9_kappa_history.extend([float(v) for v in self._v8_aux_scale_history])
        metrics = dict(self.last_v8_metrics or self.last_tcr_metrics or {})
        metrics.update(
            {
                "v9_enabled": int(bool(self.tcr_cfg.enable)),
                "v9_gap_score_ema": float(self._v9_gap_score_ema),
                "v9_kappa": float(self._v9_kappa),
                "v9_kappa_mean": _safe_mean(self._v9_kappa_history),
                "v9_kappa_base": float(self.v9_cfg.kappa_base),
            }
        )
        self.last_v9_metrics = metrics
        self.last_tcr_metrics = dict(metrics)
        for key, value in metrics.items():
            try:
                if isinstance(value, (int, float, np.integer, np.floating)):
                    self.logger.record(f"train/{key}", float(value))
            except Exception:
                pass

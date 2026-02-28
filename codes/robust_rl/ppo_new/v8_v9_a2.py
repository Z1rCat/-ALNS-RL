from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch as th

from gymnasium import spaces

from .v7_tcr import TriggeredCounterfactualPPO, _safe_mean


def _clip_float(value: float, lo: float, hi: float) -> float:
    return float(np.clip(float(value), float(lo), float(hi)))


def _safe_entropy(probs: np.ndarray, eps: float = 1e-8) -> float:
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    if p.size <= 0:
        return 0.0
    p = np.clip(p, eps, 1.0)
    p = p / max(float(np.sum(p)), eps)
    return float(-np.sum(p * np.log(p)))


def _ansi_wrap(text: str, color_code: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"\033[{color_code}m{text}\033[0m"


@dataclass
class V8A2Config:
    ratio_threshold: float = 0.10
    min_group_steps: int = 1
    gen_budget_scale: float = 2.0
    gen_budget_max_per_group: int = 128
    gen_min_total_per_rollout: int = 50
    gen_min_per_group: int = 0
    gen_mix_alpha: float = 2.0
    gen_noise_std: float = 0.01
    min_keep_weight: float = 0.15
    max_keep_weight: float = 4.00
    kl_guard_coef: float = 0.02
    ref_mix: float = 0.50
    eps: float = 1e-8


@dataclass
class V9A2Config:
    tau_entropy_eta: float = 0.05
    shortage_ema_decay: float = 0.80
    kappa_base: float = 1.00
    kappa_slope: float = 2.00
    kappa_min: float = 0.50
    kappa_max: float = 3.00


class PPOPostV8A2(TriggeredCounterfactualPPO):
    """
    PPO_NEW V8-A.2:
    - Trigger only by low action ratio (any action, not only action=1).
    - For each triggered group, generate targeted synthetic data for the under-represented action.
    - Keep PPO objective on-policy; use a small auxiliary targeted loss on generated/replay data.
    """

    def __init__(
        self,
        *args,
        v8a2_ratio_threshold: float = 0.10,
        v8a2_min_group_steps: int = 1,
        v8a2_gen_budget_scale: float = 2.0,
        v8a2_gen_budget_max_per_group: int = 128,
        v8a2_gen_min_total_per_rollout: int = 50,
        v8a2_gen_min_per_group: int = 0,
        v8a2_gen_mix_alpha: float = 2.0,
        v8a2_gen_noise_std: float = 0.01,
        v8a2_min_keep_weight: float = 0.15,
        v8a2_max_keep_weight: float = 4.00,
        v8a2_kl_guard_coef: float = 0.02,
        v8a2_ref_mix: float = 0.50,
        v8a2_eps: float = 1e-8,
        **kwargs,
    ) -> None:
        self.v8a2_cfg = V8A2Config(
            ratio_threshold=float(max(0.0, min(0.95, v8a2_ratio_threshold))),
            min_group_steps=int(max(1, v8a2_min_group_steps)),
            gen_budget_scale=float(max(0.0, v8a2_gen_budget_scale)),
            gen_budget_max_per_group=int(max(1, v8a2_gen_budget_max_per_group)),
            gen_min_total_per_rollout=int(max(0, v8a2_gen_min_total_per_rollout)),
            gen_min_per_group=int(max(0, v8a2_gen_min_per_group)),
            gen_mix_alpha=float(max(0.1, v8a2_gen_mix_alpha)),
            gen_noise_std=float(max(0.0, v8a2_gen_noise_std)),
            min_keep_weight=float(max(0.0, v8a2_min_keep_weight)),
            max_keep_weight=float(max(0.05, v8a2_max_keep_weight)),
            kl_guard_coef=float(max(0.0, v8a2_kl_guard_coef)),
            ref_mix=float(np.clip(v8a2_ref_mix, 0.0, 1.0)),
            eps=float(max(1e-12, v8a2_eps)),
        )
        if self.v8a2_cfg.max_keep_weight < self.v8a2_cfg.min_keep_weight:
            self.v8a2_cfg.max_keep_weight = self.v8a2_cfg.min_keep_weight

        self._a2_rollout_stats: Dict[str, Any] = {}
        self._a2_aux_ce_history: List[float] = []
        self._a2_aux_kl_history: List[float] = []
        self._a2_aux_scale_history: List[float] = []
        self._a2_aux_weight_history: List[float] = []
        self.last_v8a2_metrics: Dict[str, Any] = {}
        self._phase2_console = str(os.environ.get("RL_PHASE2_CONSOLE", "1")).strip() == "1"
        self._phase2_color = str(os.environ.get("RL_PHASE2_COLOR", "1")).strip() == "1"
        super().__init__(*args, **kwargs)

    def _phase2_log(self, stage: str, payload: Dict[str, Any], level: str = "info") -> None:
        if not self._phase2_console:
            return
        if level == "warn":
            color = "33"
        elif level == "good":
            color = "92"
        else:
            color = "96"
        tag = _ansi_wrap(f"[PHASE2][{stage}]", color, self._phase2_color)
        kv = " ".join([f"{k}={v}" for k, v in payload.items()])
        print(f"{tag} {kv}")

    def _num_actions(self) -> int:
        try:
            if isinstance(self.action_space, spaces.Discrete):
                n = int(self.action_space.n)
                if n > 1:
                    return n
        except Exception:
            pass
        return 2

    def _estimate_action_probs(self, obs: np.ndarray) -> np.ndarray:
        n_actions = self._num_actions()
        default = np.full((n_actions,), 1.0 / float(n_actions), dtype=np.float32)
        try:
            obs_t = th.as_tensor(np.asarray(obs, dtype=np.float32)[None, ...], device=self.device)
            with th.no_grad():
                dist = self.policy.get_distribution(obs_t)
                probs_t = getattr(dist, "distribution", None)
                probs_t = getattr(probs_t, "probs", None)
                if probs_t is None:
                    return default
                p = np.asarray(probs_t.detach().cpu().numpy(), dtype=np.float32).reshape(1, -1)[0]
                if p.size < n_actions:
                    out = default.copy()
                    out[: p.size] = p[:]
                    out = out / max(float(np.sum(out)), self.v8a2_cfg.eps)
                    return out.astype(np.float32)
                p = p[:n_actions]
                p = p / max(float(np.sum(p)), self.v8a2_cfg.eps)
                return p.astype(np.float32)
        except Exception:
            return default

    def _a2_push_sample(
        self,
        group: str,
        obs: np.ndarray,
        target_action: int,
        sample_weight: float,
        ref_probs: np.ndarray,
        synthetic: int,
        shortage: float,
    ) -> None:
        group_key = str(group or "unknown")
        if group_key not in self._tcr_buffer:
            self._tcr_buffer[group_key] = deque(maxlen=self.tcr_cfg.buffer_size_per_group)
        self._tcr_buffer[group_key].append(
            {
                "obs": np.asarray(obs, dtype=np.float32).copy(),
                "target_action": int(target_action),
                "weight": float(sample_weight),
                "ref_probs": np.asarray(ref_probs, dtype=np.float32).copy(),
                "synthetic": int(synthetic),
                "shortage": float(shortage),
            }
        )

    def _a2_sample_batch(self, groups: List[str], batch_size: int) -> Optional[Dict[str, np.ndarray]]:
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
        target_batch = np.asarray([int(filtered[i].get("target_action", 0)) for i in idx], dtype=np.int64)
        weight_batch = np.asarray([float(filtered[i].get("weight", 1.0)) for i in idx], dtype=np.float32)
        ref_batch = np.stack([np.asarray(filtered[i].get("ref_probs"), dtype=np.float32) for i in idx], axis=0).astype(np.float32)
        return {
            "obs": obs_batch,
            "target_action": target_batch,
            "weight": weight_batch,
            "ref_probs": ref_batch,
        }

    def _a2_aux_scale(self) -> float:
        return 1.0

    def _a2_version_tag(self) -> str:
        return "V8-A.2"

    def _a2_group_tau(self, probs: np.ndarray) -> float:
        _ = probs
        return float(self.v8a2_cfg.ratio_threshold)

    def _a2_after_rollout(self, shortage_mean: float) -> None:
        _ = shortage_mean

    def _a2_allow_synthetic(self) -> bool:
        return True

    def _a2_adjust_budget(self, budget: int, shortage: float, total: int, group: str) -> int:
        _ = shortage
        _ = total
        _ = group
        return int(max(0, budget))

    def _a2_real_reward_boost(self, reward_value: float, low_reward_mean: float, other_reward_mean: float) -> float:
        _ = reward_value
        _ = low_reward_mean
        _ = other_reward_mean
        return 1.0

    @staticmethod
    def _a2_group_reward_stats(rows: List[Dict[str, Any]], low_action: int, n_actions: int, eps: float) -> Tuple[float, float]:
        low_rewards: List[float] = []
        other_rewards: List[float] = []
        for row in rows:
            try:
                a = int(row.get("action", -1))
                r = float(row.get("reward", 0.0))
            except Exception:
                continue
            if a < 0 or a >= n_actions:
                continue
            if a == int(low_action):
                low_rewards.append(r)
            else:
                other_rewards.append(r)
        low_mean = float(np.mean(low_rewards)) if low_rewards else 0.0
        other_mean = float(np.mean(other_rewards)) if other_rewards else float(low_mean)
        if not np.isfinite(low_mean):
            low_mean = 0.0
        if not np.isfinite(other_mean):
            other_mean = float(low_mean)
        # Keep values bounded to avoid exploding sample weights.
        low_mean = float(np.clip(low_mean, -1.0 / max(eps, 1e-8), 1.0 / max(eps, 1e-8)))
        other_mean = float(np.clip(other_mean, -1.0 / max(eps, 1e-8), 1.0 / max(eps, 1e-8)))
        return low_mean, other_mean

    def _a2_reset_aux_trackers(self) -> None:
        self._a2_aux_ce_history = []
        self._a2_aux_kl_history = []
        self._a2_aux_scale_history = []
        self._a2_aux_weight_history = []

    def _a2_compute_aux_terms(self, batch: Dict[str, np.ndarray]) -> Optional[Dict[str, th.Tensor]]:
        obs_batch = batch.get("obs", None)
        if obs_batch is None:
            return None

        target_batch = batch.get("target_action", None)
        weight_batch = batch.get("weight", None)
        ref_batch = batch.get("ref_probs", None)
        if target_batch is None or weight_batch is None or ref_batch is None:
            return None

        obs_t = th.as_tensor(obs_batch, device=self.device, dtype=th.float32)
        target_t = th.as_tensor(target_batch, device=self.device, dtype=th.long)
        weight_t = th.as_tensor(weight_batch, device=self.device, dtype=th.float32)
        ref_t = th.as_tensor(ref_batch, device=self.device, dtype=th.float32)

        dist = self.policy.get_distribution(obs_t)
        probs_t = getattr(dist, "distribution", None)
        probs_t = getattr(probs_t, "probs", None)
        if probs_t is None:
            return None

        eps = float(self.v8a2_cfg.eps)
        cur_probs = th.clamp(probs_t.float(), min=eps, max=1.0)
        n_actions = int(cur_probs.shape[1])

        if ref_t.ndim != 2 or int(ref_t.shape[1]) != n_actions:
            return None
        if target_t.numel() != int(cur_probs.shape[0]) or weight_t.numel() != int(cur_probs.shape[0]):
            return None

        target_t = th.clamp(target_t, min=0, max=n_actions - 1)
        weight_t = th.clamp(weight_t, min=0.0)
        weight_t = weight_t / th.clamp(weight_t.mean(), min=eps)

        row_idx = th.arange(cur_probs.shape[0], device=self.device)
        p_target = th.clamp(cur_probs[row_idx, target_t], min=eps, max=1.0)
        ce_vec = -th.log(p_target)
        ce_loss = th.mean(weight_t * ce_vec)

        one_hot = th.nn.functional.one_hot(target_t, num_classes=n_actions).float()
        ref_probs = (1.0 - float(self.v8a2_cfg.ref_mix)) * ref_t + float(self.v8a2_cfg.ref_mix) * one_hot
        ref_probs = th.clamp(ref_probs, min=eps, max=1.0)
        ref_probs = ref_probs / th.clamp(ref_probs.sum(dim=1, keepdim=True), min=eps)
        cur_probs = cur_probs / th.clamp(cur_probs.sum(dim=1, keepdim=True), min=eps)

        kl_vec = th.sum(ref_probs * (th.log(ref_probs) - th.log(cur_probs)), dim=1)
        kl_loss = th.mean(weight_t * kl_vec)
        return {
            "ce_loss": ce_loss,
            "kl_loss": kl_loss,
            "weight_mean": th.mean(weight_t),
        }

    def _tcr_auxiliary_loss(self) -> Optional[th.Tensor]:
        if not self.tcr_cfg.enable or self.tcr_cfg.aux_coef <= 0.0:
            return None
        if not self._tcr_active_groups:
            return None

        batch = self._a2_sample_batch(self._tcr_active_groups, self.tcr_cfg.aux_batch_size)
        if batch is None:
            return None
        terms = self._a2_compute_aux_terms(batch)
        if terms is None:
            return None

        scale = float(max(0.0, self._a2_aux_scale()))
        aux_loss = scale * terms["ce_loss"] + float(self.v8a2_cfg.kl_guard_coef) * terms["kl_loss"]
        try:
            self._a2_aux_ce_history.append(float(terms["ce_loss"].detach().cpu().item()))
            self._a2_aux_kl_history.append(float(terms["kl_loss"].detach().cpu().item()))
            self._a2_aux_scale_history.append(float(scale))
            self._a2_aux_weight_history.append(float(terms["weight_mean"].detach().cpu().item()))
        except Exception:
            pass
        return aux_loss

    def _build_synthetic_obs(self, obs_pool: List[np.ndarray]) -> Optional[np.ndarray]:
        if not obs_pool:
            return None
        if len(obs_pool) == 1:
            base = np.asarray(obs_pool[0], dtype=np.float32)
            if self.v8a2_cfg.gen_noise_std > 0.0:
                noise = np.random.normal(0.0, self.v8a2_cfg.gen_noise_std, size=base.shape).astype(np.float32)
                base = base + noise
            return base.astype(np.float32)
        i1 = int(np.random.randint(0, len(obs_pool)))
        i2 = int(np.random.randint(0, len(obs_pool)))
        o1 = np.asarray(obs_pool[i1], dtype=np.float32)
        o2 = np.asarray(obs_pool[i2], dtype=np.float32)
        lam = float(np.random.beta(self.v8a2_cfg.gen_mix_alpha, self.v8a2_cfg.gen_mix_alpha))
        syn = lam * o1 + (1.0 - lam) * o2
        if self.v8a2_cfg.gen_noise_std > 0.0:
            noise = np.random.normal(0.0, self.v8a2_cfg.gen_noise_std, size=syn.shape).astype(np.float32)
            syn = syn + noise
        return syn.astype(np.float32)

    def tcr_consume_rollout(self, records: List[Dict[str, Any]]) -> None:
        if not self.tcr_cfg.enable:
            self._tcr_active_groups = []
            self._a2_rollout_stats = {
                "v8a2_trigger_shortage_mean": 0.0,
                "v8a2_trigger_low_ratio_mean": 0.0,
                "v8a2_generated_samples": 0,
                "v8a2_generated_target": 0,
                "v8a2_target_action0_samples": 0,
                "v8a2_target_action1_samples": 0,
                "v8a2_target_action_mode": "",
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
                "phase2_active": 0,
                "phase2_stage": "idle",
                "phase2_triggered_groups": 0,
                "phase2_new_samples": 0,
                "phase2_generated_samples": 0,
                "phase2_generated_target": 0,
                "phase2_generated_shortfall": 0,
                "phase2_target_action_mode": "",
                "phase2_aux_applied_batches": 0,
                "phase2_aux_ce_loss": 0.0,
                "phase2_aux_kl_loss": 0.0,
                "phase2_kappa": 0.0,
            }
            self.last_v8a2_metrics = dict(self.last_tcr_metrics)
            self.last_v8a2_metrics.update(
                {
                    "v8a2_enabled": 0,
                    "v8a2_trigger_shortage_mean": 0.0,
                    "v8a2_trigger_low_ratio_mean": 0.0,
                    "v8a2_generated_samples": 0,
                    "v8a2_generated_target": 0,
                    "v8a2_target_action0_samples": 0,
                    "v8a2_target_action1_samples": 0,
                    "v8a2_target_action_mode": "",
                    "v8a2_aux_ce_loss": 0.0,
                    "v8a2_aux_kl_guard_loss": 0.0,
                    "v8a2_aux_scale": 0.0,
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

        n_actions = self._num_actions()
        eps = float(self.v8a2_cfg.eps)
        triggered_groups: List[str] = []
        trigger_action1_rates: List[float] = []
        shortage_list: List[float] = []
        low_ratio_list: List[float] = []
        new_samples = 0
        generated_samples = 0
        action0_targets = 0
        action1_targets = 0
        target_action_ids: List[int] = []

        generation_plans: List[Dict[str, Any]] = []

        for group, rows in grouped.items():
            n = len(rows)
            if n < self.v8a2_cfg.min_group_steps:
                continue
            counts = np.zeros((n_actions,), dtype=np.int64)
            obs_pool: List[np.ndarray] = []
            current_low_samples: List[Tuple[np.ndarray, float]] = []
            for row in rows:
                a = int(row.get("action", -1))
                if 0 <= a < n_actions:
                    counts[a] += 1
                obs = np.asarray(row.get("obs"), dtype=np.float32)
                if obs.size > 0:
                    obs_pool.append(obs)
            total = int(np.sum(counts))
            if total <= 0 or not obs_pool:
                continue

            probs = counts.astype(np.float64) / max(float(total), eps)
            low_action = int(np.argmin(probs))
            low_ratio = float(probs[low_action])
            tau_g = float(self._a2_group_tau(probs))
            shortage = max(0.0, tau_g - low_ratio)
            if shortage <= 0.0:
                continue

            low_reward_mean, other_reward_mean = self._a2_group_reward_stats(
                rows=rows,
                low_action=low_action,
                n_actions=n_actions,
                eps=eps,
            )

            triggered_groups.append(group)
            shortage_list.append(shortage)
            low_ratio_list.append(low_ratio)
            trigger_action1_rates.append(float(probs[1]) if n_actions > 1 else float(low_ratio))
            target_action_ids.append(low_action)

            for row in current_grouped.get(group, []):
                if int(row.get("action", -1)) != low_action:
                    continue
                obs = np.asarray(row.get("obs"), dtype=np.float32)
                if obs.size > 0:
                    current_low_samples.append((obs, float(row.get("reward", 0.0))))

            severity = shortage / max(tau_g, eps)
            base_weight = _clip_float(1.0 + severity, self.v8a2_cfg.min_keep_weight, self.v8a2_cfg.max_keep_weight)
            # Keep real low-action samples from current rollout first.
            for obs, reward_value in current_low_samples:
                ref_probs = self._estimate_action_probs(obs)
                novelty = 1.0 - float(ref_probs[min(low_action, ref_probs.shape[0] - 1)])
                reward_boost = self._a2_real_reward_boost(
                    reward_value=float(reward_value),
                    low_reward_mean=float(low_reward_mean),
                    other_reward_mean=float(other_reward_mean),
                )
                sample_weight = _clip_float(
                    base_weight * (1.0 + novelty) * float(reward_boost),
                    self.v8a2_cfg.min_keep_weight,
                    self.v8a2_cfg.max_keep_weight,
                )
                self._a2_push_sample(
                    group=group,
                    obs=obs,
                    target_action=low_action,
                    sample_weight=sample_weight,
                    ref_probs=ref_probs,
                    synthetic=0,
                    shortage=shortage,
                )
                new_samples += 1
                if low_action == 0:
                    action0_targets += 1
                elif low_action == 1:
                    action1_targets += 1
            budget = int(np.ceil(self.v8a2_cfg.gen_budget_scale * shortage * float(total)))
            budget = int(max(budget, self.v8a2_cfg.gen_min_per_group))
            budget = int(max(0, min(budget, self.v8a2_cfg.gen_budget_max_per_group)))
            budget = int(self._a2_adjust_budget(budget=budget, shortage=shortage, total=total, group=group))
            generation_plans.append(
                {
                    "group": group,
                    "obs_pool": obs_pool,
                    "low_action": low_action,
                    "shortage": shortage,
                    "base_weight": base_weight,
                    "low_reward_mean": float(low_reward_mean),
                    "other_reward_mean": float(other_reward_mean),
                    "budget": budget,
                    "max_budget": int(self.v8a2_cfg.gen_budget_max_per_group),
                }
            )

        # Ensure minimum synthetic volume per rollout when at least one group is triggered.
        target_generated = int(sum(int(plan["budget"]) for plan in generation_plans))
        min_total = int(max(0, self.v8a2_cfg.gen_min_total_per_rollout))
        if generation_plans and self._a2_allow_synthetic() and target_generated < min_total:
            shortage_to_add = int(min_total - target_generated)
            # Prefer groups with larger shortage.
            order = sorted(range(len(generation_plans)), key=lambda i: float(generation_plans[i]["shortage"]), reverse=True)
            ptr = 0
            guard = 0
            while shortage_to_add > 0 and guard < 100000:
                guard += 1
                i = order[ptr % len(order)]
                ptr += 1
                cur = int(generation_plans[i]["budget"])
                mx = int(generation_plans[i]["max_budget"])
                if cur >= mx:
                    if all(int(generation_plans[j]["budget"]) >= int(generation_plans[j]["max_budget"]) for j in order):
                        break
                    continue
                generation_plans[i]["budget"] = cur + 1
                shortage_to_add -= 1
        target_generated = int(sum(int(plan["budget"]) for plan in generation_plans))

        # Materialize synthetic samples according to final budgets.
        if self._a2_allow_synthetic():
            for plan in generation_plans:
                obs_pool = list(plan.get("obs_pool", []) or [])
                group = str(plan.get("group", "unknown"))
                low_action = int(plan.get("low_action", 0))
                shortage = float(plan.get("shortage", 0.0))
                base_weight = float(plan.get("base_weight", 1.0))
                low_reward_mean = float(plan.get("low_reward_mean", 0.0))
                other_reward_mean = float(plan.get("other_reward_mean", low_reward_mean))
                budget = int(plan.get("budget", 0))
                for _ in range(max(0, budget)):
                    syn = self._build_synthetic_obs(obs_pool)
                    if syn is None:
                        break
                    ref_probs = self._estimate_action_probs(syn)
                    novelty = 1.0 - float(ref_probs[min(low_action, ref_probs.shape[0] - 1)])
                    reward_boost = self._a2_real_reward_boost(
                        reward_value=float(low_reward_mean),
                        low_reward_mean=float(low_reward_mean),
                        other_reward_mean=float(other_reward_mean),
                    )
                    sample_weight = _clip_float(
                        base_weight * (1.0 + novelty) * float(reward_boost),
                        self.v8a2_cfg.min_keep_weight,
                        self.v8a2_cfg.max_keep_weight,
                    )
                    self._a2_push_sample(
                        group=group,
                        obs=syn,
                        target_action=low_action,
                        sample_weight=sample_weight,
                        ref_probs=ref_probs,
                        synthetic=1,
                        shortage=shortage,
                    )
                    new_samples += 1
                    generated_samples += 1
                    if low_action == 0:
                        action0_targets += 1
                    elif low_action == 1:
                        action1_targets += 1
        else:
            target_generated = 0

        generated_shortfall = int(max(0, target_generated - int(generated_samples)))

        shortage_mean = _safe_mean(shortage_list)
        self._a2_after_rollout(shortage_mean)

        mode_target = ""
        if target_action_ids:
            uniq, cnt = np.unique(np.asarray(target_action_ids, dtype=np.int64), return_counts=True)
            mode_target = str(int(uniq[int(np.argmax(cnt))]))

        self._tcr_active_groups = list(triggered_groups)
        self._a2_rollout_stats = {
            "v8a2_trigger_shortage_mean": shortage_mean,
            "v8a2_trigger_low_ratio_mean": _safe_mean(low_ratio_list),
            "v8a2_generated_samples": int(generated_samples),
            "v8a2_generated_target": int(target_generated),
            "v8a2_generated_shortfall": int(generated_shortfall),
            "v8a2_target_action0_samples": int(action0_targets),
            "v8a2_target_action1_samples": int(action1_targets),
            "v8a2_target_action_mode": mode_target,
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
            "tcr_reward_gap_trigger_mean": 0.0,
            "tcr_aux_loss": 0.0,
            "tcr_aux_applied_batches": 0,
            "tcr_teacher_mode": "none",
            "phase2_active": int(len(triggered_groups) > 0),
            "phase2_stage": "generate" if len(triggered_groups) > 0 else "idle",
            "phase2_triggered_groups": int(len(triggered_groups)),
            "phase2_new_samples": int(new_samples),
            "phase2_generated_samples": int(generated_samples),
            "phase2_generated_target": int(target_generated),
            "phase2_generated_shortfall": int(generated_shortfall),
            "phase2_target_action_mode": mode_target,
            "phase2_aux_applied_batches": 0,
            "phase2_aux_ce_loss": 0.0,
            "phase2_aux_kl_loss": 0.0,
            "phase2_kappa": float(self._a2_aux_scale()),
        }
        self.last_v8a2_metrics = dict(self.last_tcr_metrics)
        self.last_v8a2_metrics.update(
            {
                "v8a2_enabled": 1,
                "v8a2_trigger_shortage_mean": shortage_mean,
                "v8a2_trigger_low_ratio_mean": _safe_mean(low_ratio_list),
                "v8a2_generated_samples": int(generated_samples),
                "v8a2_generated_target": int(target_generated),
                "v8a2_generated_shortfall": int(generated_shortfall),
                "v8a2_target_action0_samples": int(action0_targets),
                "v8a2_target_action1_samples": int(action1_targets),
                "v8a2_target_action_mode": mode_target,
                "v8a2_aux_ce_loss": 0.0,
                "v8a2_aux_kl_guard_loss": 0.0,
                "v8a2_aux_scale": float(self._a2_aux_scale()),
            }
        )
        if len(triggered_groups) > 0:
            self._phase2_log(
                "GENERATE",
                {
                    "ver": self._a2_version_tag(),
                    "groups": int(len(triggered_groups)),
                    "target_mode": mode_target if mode_target != "" else "na",
                    "new_samples": int(new_samples),
                    "generated": int(generated_samples),
                    "target_generated": int(target_generated),
                    "shortfall": int(generated_shortfall),
                    "shortage_mean": f"{shortage_mean:.4f}",
                },
                level="good",
            )

    def train(self) -> None:
        self._a2_reset_aux_trackers()
        super().train()
        aux_batches = int((self.last_tcr_metrics or {}).get("tcr_aux_applied_batches", 0) or 0)
        phase2_active = int((self.last_tcr_metrics or {}).get("phase2_active", 0) or 0)
        if aux_batches > 0:
            phase2_stage = "aux_train"
        elif phase2_active > 0:
            phase2_stage = "generate"
        else:
            phase2_stage = "idle"
        metrics = dict(self.last_tcr_metrics or {})
        metrics.update(
            {
                "v8a2_enabled": int(bool(self.tcr_cfg.enable)),
                "v8a2_trigger_shortage_mean": float(self._a2_rollout_stats.get("v8a2_trigger_shortage_mean", 0.0)),
                "v8a2_trigger_low_ratio_mean": float(self._a2_rollout_stats.get("v8a2_trigger_low_ratio_mean", 0.0)),
                "v8a2_generated_samples": int(self._a2_rollout_stats.get("v8a2_generated_samples", 0)),
                "v8a2_generated_target": int(self._a2_rollout_stats.get("v8a2_generated_target", 0)),
                "v8a2_generated_shortfall": int(self._a2_rollout_stats.get("v8a2_generated_shortfall", 0)),
                "v8a2_target_action0_samples": int(self._a2_rollout_stats.get("v8a2_target_action0_samples", 0)),
                "v8a2_target_action1_samples": int(self._a2_rollout_stats.get("v8a2_target_action1_samples", 0)),
                "v8a2_target_action_mode": str(self._a2_rollout_stats.get("v8a2_target_action_mode", "")),
                "v8a2_aux_ce_loss": _safe_mean(self._a2_aux_ce_history),
                "v8a2_aux_kl_guard_loss": _safe_mean(self._a2_aux_kl_history),
                "v8a2_aux_weight_mean": _safe_mean(self._a2_aux_weight_history),
                "v8a2_aux_scale": _safe_mean(self._a2_aux_scale_history),
                "phase2_active": int(phase2_active > 0 or aux_batches > 0),
                "phase2_stage": phase2_stage,
                "phase2_triggered_groups": int((self.last_tcr_metrics or {}).get("tcr_triggered_groups", 0) or 0),
                "phase2_new_samples": int((self.last_tcr_metrics or {}).get("tcr_new_samples", 0) or 0),
                "phase2_generated_samples": int(self._a2_rollout_stats.get("v8a2_generated_samples", 0)),
                "phase2_generated_target": int(self._a2_rollout_stats.get("v8a2_generated_target", 0)),
                "phase2_generated_shortfall": int(self._a2_rollout_stats.get("v8a2_generated_shortfall", 0)),
                "phase2_target_action_mode": str(self._a2_rollout_stats.get("v8a2_target_action_mode", "")),
                "phase2_aux_applied_batches": int(aux_batches),
                "phase2_aux_ce_loss": _safe_mean(self._a2_aux_ce_history),
                "phase2_aux_kl_loss": _safe_mean(self._a2_aux_kl_history),
                "phase2_kappa": _safe_mean(self._a2_aux_scale_history),
            }
        )
        self.last_v8a2_metrics = metrics
        self.last_tcr_metrics = dict(metrics)
        if aux_batches > 0:
            self._phase2_log(
                "AUX-TRAIN",
                {
                    "ver": self._a2_version_tag(),
                    "batches": int(aux_batches),
                    "generated": int(self._a2_rollout_stats.get("v8a2_generated_samples", 0)),
                    "target_generated": int(self._a2_rollout_stats.get("v8a2_generated_target", 0)),
                    "aux_ce": f"{_safe_mean(self._a2_aux_ce_history):.4f}",
                    "aux_kl": f"{_safe_mean(self._a2_aux_kl_history):.4f}",
                    "scale": f"{_safe_mean(self._a2_aux_scale_history):.4f}",
                },
                level="good",
            )
        for key, value in metrics.items():
            try:
                if isinstance(value, (int, float, np.integer, np.floating)):
                    self.logger.record(f"train/{key}", float(value))
            except Exception:
                pass


class PPOPostV9A2(PPOPostV8A2):
    """
    PPO_NEW V9-A.2:
    - V8-A.2 + adaptive trigger threshold and adaptive aux scale from shortage EMA.
    """

    def __init__(
        self,
        *args,
        v9a2_tau_entropy_eta: float = 0.05,
        v9a2_shortage_ema_decay: float = 0.80,
        v9a2_kappa_base: float = 1.00,
        v9a2_kappa_slope: float = 2.00,
        v9a2_kappa_min: float = 0.50,
        v9a2_kappa_max: float = 3.00,
        **kwargs,
    ) -> None:
        self.v9a2_cfg = V9A2Config(
            tau_entropy_eta=float(max(0.0, v9a2_tau_entropy_eta)),
            shortage_ema_decay=float(np.clip(v9a2_shortage_ema_decay, 0.0, 0.999)),
            kappa_base=float(v9a2_kappa_base),
            kappa_slope=float(v9a2_kappa_slope),
            kappa_min=float(v9a2_kappa_min),
            kappa_max=float(v9a2_kappa_max),
        )
        if self.v9a2_cfg.kappa_max < self.v9a2_cfg.kappa_min:
            self.v9a2_cfg.kappa_max = self.v9a2_cfg.kappa_min
        self._v9a2_shortage_ema = 0.0
        self._v9a2_kappa = _clip_float(self.v9a2_cfg.kappa_base, self.v9a2_cfg.kappa_min, self.v9a2_cfg.kappa_max)
        self._v9a2_kappa_history: List[float] = []
        self.last_v9a2_metrics: Dict[str, Any] = {}
        super().__init__(*args, **kwargs)

    def _a2_group_tau(self, probs: np.ndarray) -> float:
        base_tau = float(self.v8a2_cfg.ratio_threshold)
        n_actions = max(2, int(np.asarray(probs).size))
        h = _safe_entropy(np.asarray(probs, dtype=np.float64), eps=self.v8a2_cfg.eps)
        h_max = float(np.log(float(n_actions)))
        h_norm = float(h / max(h_max, self.v8a2_cfg.eps))
        tau = base_tau + float(self.v9a2_cfg.tau_entropy_eta) * (1.0 - h_norm)
        return _clip_float(tau, 0.0, 0.95)

    def _a2_after_rollout(self, shortage_mean: float) -> None:
        decay = float(self.v9a2_cfg.shortage_ema_decay)
        self._v9a2_shortage_ema = decay * float(self._v9a2_shortage_ema) + (1.0 - decay) * float(shortage_mean)
        raw = float(self.v9a2_cfg.kappa_base + self.v9a2_cfg.kappa_slope * self._v9a2_shortage_ema)
        self._v9a2_kappa = _clip_float(raw, self.v9a2_cfg.kappa_min, self.v9a2_cfg.kappa_max)

    def _a2_aux_scale(self) -> float:
        return float(self._v9a2_kappa)

    def _a2_version_tag(self) -> str:
        return "V9-A.2"

    def train(self) -> None:
        self._v9a2_kappa_history = []
        super().train()
        if self._a2_aux_scale_history:
            self._v9a2_kappa_history.extend([float(v) for v in self._a2_aux_scale_history])
        metrics = dict(self.last_v8a2_metrics or self.last_tcr_metrics or {})
        metrics.update(
            {
                "v9a2_enabled": int(bool(self.tcr_cfg.enable)),
                "v9a2_shortage_ema": float(self._v9a2_shortage_ema),
                "v9a2_kappa": float(self._v9a2_kappa),
                "v9a2_kappa_mean": _safe_mean(self._v9a2_kappa_history),
                "v9a2_kappa_base": float(self.v9a2_cfg.kappa_base),
                "phase2_kappa": float(self._v9a2_kappa),
            }
        )
        self.last_v9a2_metrics = metrics
        self.last_tcr_metrics = dict(metrics)
        if int(metrics.get("phase2_active", 0) or 0) > 0:
            self._phase2_log(
                "ADAPT",
                {
                    "ver": self._a2_version_tag(),
                    "shortage_ema": f"{float(self._v9a2_shortage_ema):.4f}",
                    "kappa": f"{float(self._v9a2_kappa):.4f}",
                },
                level="info",
            )
        for key, value in metrics.items():
            try:
                if isinstance(value, (int, float, np.integer, np.floating)):
                    self.logger.record(f"train/{key}", float(value))
            except Exception:
                pass


class _PPOPostV10Base(PPOPostV8A2):
    """
    V10 base:
    - same trigger (low action ratio for any action)
    - supports large-scale targeted generation + real-interaction weighting
    - variant toggles are implemented by subclass hooks.
    """

    def __init__(
        self,
        *args,
        v10_reward_constraint_scale: float = 1.0,
        **kwargs,
    ) -> None:
        self.v10_reward_constraint_scale = float(max(0.0, v10_reward_constraint_scale))
        super().__init__(*args, **kwargs)

    def _a2_version_tag(self) -> str:
        return "V10"

    def _a2_real_reward_boost(self, reward_value: float, low_reward_mean: float, other_reward_mean: float) -> float:
        scale = float(max(0.0, self.v10_reward_constraint_scale))
        if scale <= 0.0:
            return 1.0
        # If low action has relatively better real reward signal, amplify correction weight.
        rel = float(reward_value - other_reward_mean)
        boost = 1.0 + scale * rel
        return _clip_float(boost, 0.25, 3.0)


class PPOPostV10A(_PPOPostV10Base):
    """
    V10-A: full mode
      - targeted large-scale action correction
      - real interaction constraint enabled
      - KL guard enabled
    """

    def _a2_version_tag(self) -> str:
        return "V10-A"


class PPOPostV10B(_PPOPostV10Base):
    """
    V10-B: remove KL guard
      - targeted large-scale action correction
      - real interaction constraint enabled
      - KL guard disabled
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.v8a2_cfg.kl_guard_coef = 0.0

    def _a2_version_tag(self) -> str:
        return "V10-B"


class PPOPostV10C(_PPOPostV10Base):
    """
    V10-C: balance-only
      - targeted large-scale action correction only
      - no real-reward weighting
      - no KL guard
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs.pop("v10_reward_constraint_scale", None)
        super().__init__(*args, v10_reward_constraint_scale=0.0, **kwargs)
        self.v8a2_cfg.kl_guard_coef = 0.0
        self.v8a2_cfg.ref_mix = 1.0

    def _a2_version_tag(self) -> str:
        return "V10-C"


class PPOPostV10D(_PPOPostV10Base):
    """
    V10-D: strong balance-only
      - stronger generation budget for underrepresented action
      - no real-reward weighting
      - no KL guard
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs.pop("v10_reward_constraint_scale", None)
        super().__init__(*args, v10_reward_constraint_scale=0.0, **kwargs)
        self.v8a2_cfg.kl_guard_coef = 0.0
        self.v8a2_cfg.ref_mix = 1.0
        self.v8a2_cfg.gen_budget_scale = max(3.0, float(self.v8a2_cfg.gen_budget_scale))
        self.v8a2_cfg.gen_min_total_per_rollout = max(50, int(self.v8a2_cfg.gen_min_total_per_rollout))
        self.v8a2_cfg.gen_budget_max_per_group = max(256, int(self.v8a2_cfg.gen_budget_max_per_group))

    def _a2_adjust_budget(self, budget: int, shortage: float, total: int, group: str) -> int:
        _ = shortage
        _ = total
        _ = group
        return int(max(0, int(np.ceil(1.5 * float(budget)))))

    def _a2_version_tag(self) -> str:
        return "V10-D"


class PPOPostV10E(_PPOPostV10Base):
    """
    V10-E: real-only source
      - keep targeted correction on real samples
      - disable synthetic generation
      - no KL guard
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.v8a2_cfg.kl_guard_coef = 0.0

    def _a2_allow_synthetic(self) -> bool:
        return False

    def _a2_version_tag(self) -> str:
        return "V10-E"

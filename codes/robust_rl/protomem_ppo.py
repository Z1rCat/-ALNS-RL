from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
import os
import sys

import numpy as np

try:
    import gym
    from gym.spaces import Box
except Exception as exc:  # pragma: no cover
    raise ImportError("ProtoMem-PPO wrapper requires gym.") from exc

try:
    import torch as th
    import torch.nn as nn
    import torch.nn.functional as F
    try:
        from gymnasium import spaces
    except Exception:  # pragma: no cover
        from gym import spaces  # type: ignore
    from stable_baselines3 import PPO
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.utils import explained_variance
except Exception as exc:  # pragma: no cover
    raise ImportError("ProtoMem-PPO requires torch + stable-baselines3.") from exc


def _c(text: str, color: str = "", bold: bool = False) -> str:
    if not (getattr(sys, "stdout", None) and sys.stdout.isatty()):
        return text
    palette = {
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
    }
    codes = []
    if bold:
        codes.append("1")
    if color in palette:
        codes.append(palette[color])
    if not codes:
        return text
    return f"\033[{';'.join(codes)}m{text}\033[0m"


@dataclass
class ProtoMemConfig:
    # Eq.(1) input shape
    input_mode: str = "full"  # full | obs
    include_stage: bool = True
    include_prev_action_reward: bool = True
    stage_dim: int = 2

    # Eq.(2-9) architecture
    num_prototypes: int = 32
    mem_dim: int = 64
    hidden_dim: int = 64
    tau: float = 0.5

    # Eq.(12-17) regularizers
    lambda_sparse: float = 1e-3
    lambda_div: float = 3e-4
    lambda_stable: float = 0.0
    lambda_aux: float = 0.0

    # Eq.(15) stable replay
    stable_buffer_per_phase: int = 300
    stable_batch_ratio: float = 0.25
    stable_warmup_updates: int = 0

    # Eq.(21) smoothing (kept for later stage; disabled by default)
    use_smooth: bool = False
    smooth_alpha: float = 0.1
    smooth_train_test_consistent: bool = True

    # optimizer
    mem_lr_scale: float = 0.5

    # numerics
    eps: float = 1e-8

    # wrapper reset boundaries
    keep_state_across_reset: bool = True
    reset_prev_on_table_switch: bool = True
    reset_prev_on_phase_switch: bool = True


class RouteReplayBuffer:
    """
    Stratified replay for Eq.(15): stores (x, i*) grouped by phase/table metadata.
    Metadata is for sampling only and never injected into policy input.
    """

    def __init__(self, per_phase_capacity: int = 300) -> None:
        cap = int(max(1, per_phase_capacity))
        self._by_phase: Dict[str, Deque[Tuple[np.ndarray, int, int]]] = defaultdict(lambda: deque(maxlen=cap))

    def __len__(self) -> int:
        return int(sum(len(v) for v in self._by_phase.values()))

    def add(self, x: np.ndarray, idx: int, phase_label: str, table_id: int = -1) -> None:
        key = str(phase_label or "UNK")
        try:
            arr = np.asarray(x, dtype=np.float32).reshape(-1)
            self._by_phase[key].append((arr, int(idx), int(table_id)))
        except Exception:
            return

    def sample(self, batch_size: int, rng: np.random.Generator) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        total = len(self)
        if total <= 0:
            return None, None
        n = int(max(1, batch_size))
        keys = [k for k, v in self._by_phase.items() if len(v) > 0]
        if not keys:
            return None, None

        rng.shuffle(keys)
        per = max(1, n // len(keys))
        picks: List[Tuple[np.ndarray, int]] = []

        # first pass: near-uniform across phases
        for k in keys:
            arr = list(self._by_phase[k])
            if not arr:
                continue
            take = min(per, len(arr), n - len(picks))
            if take <= 0:
                continue
            idxs = rng.choice(len(arr), size=take, replace=False)
            for j in idxs:
                x, i_star, _table = arr[int(j)]
                picks.append((x, int(i_star)))
            if len(picks) >= n:
                break

        # fill remainder from all phases
        if len(picks) < n:
            flat: List[Tuple[np.ndarray, int]] = []
            for k in keys:
                for x, i_star, _table in self._by_phase[k]:
                    flat.append((x, int(i_star)))
            if flat:
                need = n - len(picks)
                idxs = rng.choice(len(flat), size=min(need, len(flat)), replace=False)
                for j in idxs:
                    picks.append(flat[int(j)])

        if not picks:
            return None, None
        obs = np.stack([p[0] for p in picks], axis=0).astype(np.float32)
        labels = np.asarray([p[1] for p in picks], dtype=np.int64)
        return obs, labels


class ProtoMemInputWrapper(gym.Wrapper):
    """
    Builds Eq.(1) input without modifying the base environment state logic.
    x_t = [obs, stage, prev_action, prev_reward] when enabled.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        include_stage: bool,
        include_prev_action_reward: bool,
        stage_dim: int,
        keep_state: bool,
        stage_getter=None,
        table_getter=None,
        phase_getter=None,
        reset_prev_on_table_switch: bool = True,
        reset_prev_on_phase_switch: bool = True,
    ) -> None:
        super().__init__(env)
        self.include_stage = bool(include_stage)
        self.include_prev_action_reward = bool(include_prev_action_reward)
        self.stage_dim = int(max(0, stage_dim if include_stage else 0))
        self.keep_state = bool(keep_state)
        self.stage_getter = stage_getter
        self.table_getter = table_getter
        self.phase_getter = phase_getter
        self.reset_prev_on_table_switch = bool(reset_prev_on_table_switch)
        self.reset_prev_on_phase_switch = bool(reset_prev_on_phase_switch)

        self.action_dim = int(getattr(env.action_space, "n", 2))
        obs_shape = getattr(env.observation_space, "shape", (1,))
        self.obs_dim = int(np.prod(obs_shape))

        extra_prev = self.action_dim + 1 if self.include_prev_action_reward else 0
        self.feature_dim = self.obs_dim + self.stage_dim + extra_prev

        self._last_action = np.zeros((self.action_dim,), dtype=np.float32)
        self._last_reward = 0.0
        self._last_table = None
        self._last_phase = None

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.feature_dim,),
            dtype=np.float32,
        )

    def _onehot(self, action: int) -> np.ndarray:
        vec = np.zeros((self.action_dim,), dtype=np.float32)
        try:
            idx = int(action)
            if 0 <= idx < self.action_dim:
                vec[idx] = 1.0
        except Exception:
            pass
        return vec

    def reset_inference_state(self, *, reset_prev: bool = True) -> None:
        if reset_prev:
            self._last_action = np.zeros((self.action_dim,), dtype=np.float32)
            self._last_reward = 0.0
        self._last_table = None
        self._last_phase = None

    def _safe_table(self):
        if self.table_getter is None:
            return None
        try:
            return self.table_getter()
        except Exception:
            return None

    def _safe_phase(self):
        if self.phase_getter is None:
            return None
        try:
            return self.phase_getter()
        except Exception:
            return None

    def _safe_stage(self) -> np.ndarray:
        if not self.include_stage or self.stage_getter is None:
            return np.zeros((self.stage_dim,), dtype=np.float32)
        try:
            stage = np.asarray(self.stage_getter(), dtype=np.float32).reshape(-1)
            if stage.size != self.stage_dim:
                return np.zeros((self.stage_dim,), dtype=np.float32)
            return stage
        except Exception:
            return np.zeros((self.stage_dim,), dtype=np.float32)

    def _check_boundary_and_reset_prev(self) -> None:
        cur_table = self._safe_table()
        cur_phase = self._safe_phase()
        table_changed = self._last_table is not None and cur_table is not None and cur_table != self._last_table
        phase_changed = self._last_phase is not None and cur_phase is not None and str(cur_phase) != str(self._last_phase)

        if (table_changed and self.reset_prev_on_table_switch) or (phase_changed and self.reset_prev_on_phase_switch):
            self._last_action = np.zeros((self.action_dim,), dtype=np.float32)
            self._last_reward = 0.0

        self._last_table = cur_table
        self._last_phase = cur_phase

    def _build_obs(self, obs: np.ndarray) -> np.ndarray:
        self._check_boundary_and_reset_prev()
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        feats: List[np.ndarray] = [obs_arr]
        if self.include_stage:
            feats.append(self._safe_stage())
        if self.include_prev_action_reward:
            feats.append(self._last_action)
            feats.append(np.asarray([self._last_reward], dtype=np.float32))
        return np.concatenate(feats, axis=0).astype(np.float32)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if not self.keep_state:
            self._last_action = np.zeros((self.action_dim,), dtype=np.float32)
            self._last_reward = 0.0
        return self._build_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._last_action = self._onehot(int(np.asarray(action).squeeze()))
        try:
            self._last_reward = float(reward)
        except Exception:
            self._last_reward = 0.0
        return self._build_obs(obs), reward, done, info


class ProtoMemPolicy(ActorCriticPolicy):
    def __init__(self, *args, pm_config: Optional[ProtoMemConfig] = None, **kwargs) -> None:
        self.pm_config = pm_config or ProtoMemConfig()
        super().__init__(*args, **kwargs)

        action_dim = int(getattr(self.action_space, "n", 2))
        x_dim = int(np.prod(self.observation_space.shape))
        h_dim = int(max(8, self.pm_config.hidden_dim))
        mem_dim = int(max(4, self.pm_config.mem_dim))
        n_proto = int(max(2, self.pm_config.num_prototypes))

        self.query_encoder = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, mem_dim),
        )
        self.backbone = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )
        self.film_head = nn.Linear(mem_dim, h_dim * 2)

        self.actor_head = nn.Linear(h_dim, action_dim)
        self.critic_head = nn.Linear(h_dim, 1)
        self.aux_head = nn.Linear(mem_dim, 1)

        self.proto_mem = nn.Parameter(th.randn(n_proto, mem_dim) * 0.02)

        self._cache: Optional[Dict[str, th.Tensor]] = None

        self._log_route_entropy: Deque[float] = deque(maxlen=100)
        self._log_route_conf: Deque[float] = deque(maxlen=100)
        self._log_route_conf_p25: Deque[float] = deque(maxlen=100)
        self._log_route_conf_p75: Deque[float] = deque(maxlen=100)
        self._log_top1_entropy: Deque[float] = deque(maxlen=100)

        # Eq.(21) non-parameter runtime state
        self.register_buffer("_smooth_state", th.zeros((1, mem_dim), dtype=th.float32), persistent=False)

    def get_memory_parameters(self) -> List[nn.Parameter]:
        return [self.proto_mem]

    def reset_inference_state(self, reason: str = "") -> None:
        with th.no_grad():
            if hasattr(self, "_smooth_state"):
                self._smooth_state.zero_()

    def _flat_obs(self, obs: th.Tensor) -> th.Tensor:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        return obs.float().view(obs.shape[0], -1)

    def _route(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        q = self.query_encoder(x)
        tau = float(max(1e-6, self.pm_config.tau))

        # Eq.(21): optional smoothing; default off.
        q_used = q
        if bool(self.pm_config.use_smooth):
            alpha = float(np.clip(self.pm_config.smooth_alpha, 0.0, 1.0))
            if self._smooth_state.shape[0] != q.shape[0]:
                self._smooth_state = th.zeros_like(q)
            self._smooth_state = (1.0 - alpha) * self._smooth_state + alpha * q
            q_used = self._smooth_state

        route_logits = th.matmul(q_used, self.proto_mem.t()) / tau
        route_probs = th.softmax(route_logits, dim=-1)
        ctx = th.matmul(route_probs, self.proto_mem)
        return route_logits, route_probs, ctx

    def _core(self, obs: th.Tensor) -> Dict[str, th.Tensor]:
        x = self._flat_obs(obs)
        route_logits, route_probs, ctx = self._route(x)

        h = self.backbone(x)
        gamma_beta = self.film_head(ctx)
        gamma, beta = th.chunk(gamma_beta, 2, dim=-1)
        h_tilde = (1.0 + gamma) * h + beta

        action_logits = self.actor_head(h_tilde)
        values = self.critic_head(h_tilde).squeeze(-1)
        aux_logits = self.aux_head(ctx).squeeze(-1)

        with th.no_grad():
            eps = float(max(1e-12, self.pm_config.eps))
            ent = -(route_probs * th.log(route_probs.clamp_min(eps))).sum(dim=-1).mean().item()
            conf_values = route_probs.max(dim=-1).values
            conf = conf_values.mean().item()
            conf_p25 = th.quantile(conf_values, 0.25).item()
            conf_p75 = th.quantile(conf_values, 0.75).item()
            top1 = route_probs.argmax(dim=-1)
            bins = th.bincount(top1, minlength=route_probs.shape[-1]).float()
            p = bins / bins.sum().clamp_min(1.0)
            top1_ent = -(p * th.log(p.clamp_min(eps))).sum().item()
            self._log_route_entropy.append(float(ent))
            self._log_route_conf.append(float(conf))
            self._log_route_conf_p25.append(float(conf_p25))
            self._log_route_conf_p75.append(float(conf_p75))
            self._log_top1_entropy.append(float(top1_ent))

        return {
            "route_logits": route_logits,
            "route_probs": route_probs,
            "ctx": ctx,
            "action_logits": action_logits,
            "values": values,
            "aux_logits": aux_logits,
        }

    def route_logits_from_obs(self, obs: th.Tensor) -> th.Tensor:
        x = self._flat_obs(obs)
        q = self.query_encoder(x)
        tau = float(max(1e-6, self.pm_config.tau))
        return th.matmul(q, self.proto_mem.t()) / tau

    def get_protomem_log(self) -> Dict[str, float]:
        def _mean(vs: Deque[float]) -> float:
            return float(sum(vs) / len(vs)) if vs else 0.0

        return {
            "pm_route_entropy_mean": _mean(self._log_route_entropy),
            "pm_route_confidence_mean": _mean(self._log_route_conf),
            "pm_route_confidence_p25": _mean(self._log_route_conf_p25),
            "pm_route_confidence_p75": _mean(self._log_route_conf_p75),
            "pm_proto_top1_entropy": _mean(self._log_top1_entropy),
            "pm_param_norm_m": float(self.proto_mem.detach().norm().cpu().item()),
        }

    def pop_protomem_cache(self) -> Dict[str, th.Tensor]:
        if self._cache is None:
            raise RuntimeError("ProtoMem cache is empty: evaluate_actions() not called.")
        cache = self._cache
        self._cache = None
        return cache

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        out = self._core(obs)
        dist = self.action_dist.proba_distribution(out["action_logits"])
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        return actions, out["values"], log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        out = self._core(obs)
        dist = self.action_dist.proba_distribution(out["action_logits"])
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        self._cache = {
            "route_logits": out["route_logits"],
            "route_probs": out["route_probs"],
            "aux_logits": out["aux_logits"],
        }
        return out["values"], log_prob, entropy

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        return self._core(obs)["values"]


class ProtoMemPPO(PPO):
    """
    SB3-compatible PPO subclass with additive ProtoMem losses.
    PPO clipped surrogate remains unchanged.
    """

    def __init__(self, *args, pm_config: Optional[ProtoMemConfig] = None, **kwargs):
        self.pm_config = pm_config or ProtoMemConfig()
        super().__init__(*args, **kwargs)

        self._stable_buffer = RouteReplayBuffer(per_phase_capacity=int(self.pm_config.stable_buffer_per_phase))
        self._rng = np.random.default_rng(42)
        self.last_protomem_losses: Dict[str, float] = {}
        self._setup_mem_param_groups()
        if self.verbose >= 1:
            print(
                f"{_c('[ProtoMem]', 'cyan', True)} init "
                f"N={self.pm_config.num_prototypes} d={self.pm_config.mem_dim} "
                f"tau={self.pm_config.tau} "
                f"lambda(sparse/div/stable/aux)="
                f"{self.pm_config.lambda_sparse}/{self.pm_config.lambda_div}/{self.pm_config.lambda_stable}/{self.pm_config.lambda_aux}"
            )

    def _setup_mem_param_groups(self) -> None:
        if not hasattr(self.policy, "get_memory_parameters"):
            return
        mem_params = list(self.policy.get_memory_parameters())
        if not mem_params:
            return
        mem_ids = {id(p) for p in mem_params}
        base_params = [p for p in self.policy.parameters() if p.requires_grad and id(p) not in mem_ids]

        old_opt = self.policy.optimizer
        opt_cls = old_opt.__class__
        defaults = dict(old_opt.defaults)
        base_lr = float(old_opt.param_groups[0].get("lr", defaults.get("lr", 3e-4)))
        mem_lr = base_lr * float(self.pm_config.mem_lr_scale)

        opt_kwargs = {k: v for k, v in defaults.items() if k != "lr"}
        self.policy.optimizer = opt_cls(
            [
                {"params": base_params, "lr": base_lr},
                {"params": mem_params, "lr": mem_lr},
            ],
            lr=base_lr,
            **opt_kwargs,
        )

    def _sync_mem_lr(self) -> None:
        if len(self.policy.optimizer.param_groups) < 2:
            return
        base_lr = float(self.policy.optimizer.param_groups[0].get("lr", 3e-4))
        self.policy.optimizer.param_groups[1]["lr"] = base_lr * float(self.pm_config.mem_lr_scale)

    def _get_phase_labels(self, n: int) -> List[str]:
        labels: List[str] = ["UNK"] * int(max(0, n))
        try:
            from core import dynamic_RL34959 as dyn

            phase_list = list(getattr(dyn, "PDI_PHASE_LIST", []) or [])
            if len(phase_list) >= n and n > 0:
                tail = phase_list[-n:]
                labels = [str(x) if x not in (None, "") else "UNK" for x in tail]
        except Exception:
            pass
        return labels

    def _update_stable_buffer(self, obs_np: np.ndarray, route_logits_np: np.ndarray, phase_labels: List[str]) -> None:
        if obs_np is None or route_logits_np is None:
            return
        try:
            top1 = np.argmax(route_logits_np, axis=1)
            m = min(len(obs_np), len(top1), len(phase_labels))
            for i in range(m):
                phase = phase_labels[i] if i < len(phase_labels) else "UNK"
                self._stable_buffer.add(obs_np[i], int(top1[i]), phase_label=phase, table_id=-1)
        except Exception:
            return

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        self._sync_mem_lr()
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses: List[float] = []
        pg_losses: List[float] = []
        value_losses: List[float] = []
        clip_fractions: List[float] = []
        approx_kl_divs: List[float] = []

        sparse_losses: List[float] = []
        div_losses: List[float] = []
        stable_losses: List[float] = []
        aux_losses: List[float] = []
        grad_m_norms: List[float] = []

        rb = self.rollout_buffer
        indices = np.random.permutation(rb.buffer_size * rb.n_envs)
        if not rb.generator_ready:
            for name in ["observations", "actions", "values", "log_probs", "advantages", "returns", "rewards"]:
                rb.__dict__[name] = rb.swap_and_flatten(rb.__dict__[name])
            rb.generator_ready = True

        rewards = rb.rewards.flatten()
        phase_labels = self._get_phase_labels(int(len(rewards)))

        # Keep Eq.(15) buffer updated from latest rollout routes.
        with th.no_grad():
            obs_all = self.rollout_buffer.to_torch(rb.observations)
            logits_all = self.policy.route_logits_from_obs(obs_all).detach().cpu().numpy()
        self._update_stable_buffer(rb.observations, logits_all, phase_labels)
        phase_top1_entropy = 0.0
        try:
            top1_all = np.argmax(logits_all, axis=1)
            by_phase: Dict[str, List[int]] = defaultdict(list)
            for i, ph in enumerate(phase_labels[: len(top1_all)]):
                by_phase[str(ph or "UNK")].append(int(top1_all[i]))
            ent_vals: List[float] = []
            eps = float(max(1e-12, self.pm_config.eps))
            for vals in by_phase.values():
                if not vals:
                    continue
                counts = np.bincount(np.asarray(vals, dtype=np.int64), minlength=int(self.pm_config.num_prototypes)).astype(np.float64)
                p = counts / max(1.0, counts.sum())
                ent = -float(np.sum(p * np.log(np.clip(p, eps, None))))
                ent_vals.append(ent)
            if ent_vals:
                phase_top1_entropy = float(np.mean(ent_vals))
        except Exception:
            phase_top1_entropy = 0.0

        continue_training = True
        for epoch in range(self.n_epochs):
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
                observations, actions, old_values, old_log_prob, advantages, returns = map(
                    self.rollout_buffer.to_torch, data
                )

                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(observations, actions)
                values = values.flatten()

                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Eq.(10)-(11): PPO core unchanged.
                ratio = th.exp(log_prob - old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(float(policy_loss.detach().cpu().item()))

                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(float(clip_fraction))

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = old_values + th.clamp(values - old_values, -clip_range_vf, clip_range_vf)
                value_loss = F.mse_loss(returns, values_pred)
                value_losses.append(float(value_loss.detach().cpu().item()))

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(float(entropy_loss.detach().cpu().item()))

                pm_cache = self.policy.pop_protomem_cache()
                route_logits = pm_cache["route_logits"]
                route_probs = pm_cache["route_probs"]
                aux_logits = pm_cache["aux_logits"]

                eps = float(max(1e-12, self.pm_config.eps))

                # Eq.(12): L_sparse = E[H(w)] and minimize directly.
                loss_sparse = -(route_probs * th.log(route_probs.clamp_min(eps))).sum(dim=-1).mean()
                sparse_losses.append(float(loss_sparse.detach().cpu().item()))

                # Eq.(13): Frobenius penalty for prototype diversity.
                M = self.policy.proto_mem
                M_tilde = M / (th.norm(M, p="fro") + eps)
                gram = th.matmul(M_tilde, M_tilde.t())
                target = th.eye(M.shape[0], device=M.device, dtype=M.dtype) / float(M.shape[0])
                loss_div = th.sum((gram - target) ** 2)
                div_losses.append(float(loss_div.detach().cpu().item()))

                # Eq.(15): CE on logits for numeric stability.
                loss_stable = th.tensor(0.0, device=observations.device)
                if (
                    float(self.pm_config.lambda_stable) > 0.0
                    and self._n_updates >= int(self.pm_config.stable_warmup_updates)
                    and len(self._stable_buffer) > 0
                ):
                    sb = max(1, int(len(batch_inds) * float(self.pm_config.stable_batch_ratio)))
                    sb_obs_np, sb_idx_np = self._stable_buffer.sample(sb, self._rng)
                    if sb_obs_np is not None and sb_idx_np is not None:
                        sb_obs = th.as_tensor(sb_obs_np, device=observations.device, dtype=observations.dtype)
                        sb_idx = th.as_tensor(sb_idx_np, device=observations.device, dtype=th.long)
                        sb_logits = self.policy.route_logits_from_obs(sb_obs)
                        loss_stable = F.cross_entropy(sb_logits, sb_idx)
                stable_losses.append(float(loss_stable.detach().cpu().item()))

                # Eq.(16): optional weak supervision.
                loss_aux = th.tensor(0.0, device=observations.device)
                if float(self.pm_config.lambda_aux) > 0.0:
                    fail_target = 1.0 - th.as_tensor(rewards[batch_inds], device=observations.device, dtype=th.float32)
                    loss_aux = F.binary_cross_entropy_with_logits(aux_logits, fail_target)
                aux_losses.append(float(loss_aux.detach().cpu().item()))

                # Eq.(17): total loss.
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + float(self.pm_config.lambda_sparse) * loss_sparse
                    + float(self.pm_config.lambda_div) * loss_div
                    + float(self.pm_config.lambda_stable) * loss_stable
                    + float(self.pm_config.lambda_aux) * loss_aux
                )

                with th.no_grad():
                    log_ratio = log_prob - old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(float(approx_kl_div))

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at epoch {epoch} due to max kl: {approx_kl_div:.4f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                with th.no_grad():
                    grad_m = getattr(self.policy.proto_mem, "grad", None)
                    if grad_m is not None:
                        grad_m_norms.append(float(grad_m.norm().detach().cpu().item()))
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(rb.values.flatten(), rb.returns.flatten())
        self.logger.record("train/entropy_loss", float(np.mean(entropy_losses)) if entropy_losses else 0.0)
        self.logger.record("train/policy_gradient_loss", float(np.mean(pg_losses)) if pg_losses else 0.0)
        self.logger.record("train/value_loss", float(np.mean(value_losses)) if value_losses else 0.0)
        self.logger.record("train/approx_kl", float(np.mean(approx_kl_divs)) if approx_kl_divs else 0.0)
        self.logger.record("train/clip_fraction", float(np.mean(clip_fractions)) if clip_fractions else 0.0)
        self.logger.record("train/explained_variance", float(explained_var))

        self.logger.record("train/pm_loss_sparse", float(np.mean(sparse_losses)) if sparse_losses else 0.0)
        self.logger.record("train/pm_loss_div", float(np.mean(div_losses)) if div_losses else 0.0)
        self.logger.record("train/pm_loss_stable", float(np.mean(stable_losses)) if stable_losses else 0.0)
        self.logger.record("train/pm_loss_aux", float(np.mean(aux_losses)) if aux_losses else 0.0)
        self.logger.record("train/pm_grad_norm_m", float(np.mean(grad_m_norms)) if grad_m_norms else 0.0)
        self.logger.record("train/pm_proto_top1_entropy_over_phase", float(phase_top1_entropy))

        pm_log = self.policy.get_protomem_log() if hasattr(self.policy, "get_protomem_log") else {}
        for k, v in pm_log.items():
            self.logger.record(f"train/{k}", float(v))

        self.last_protomem_losses = {
            "pm_loss_sparse": float(np.mean(sparse_losses)) if sparse_losses else "",
            "pm_loss_div": float(np.mean(div_losses)) if div_losses else "",
            "pm_loss_stable": float(np.mean(stable_losses)) if stable_losses else "",
            "pm_loss_aux": float(np.mean(aux_losses)) if aux_losses else "",
            "pm_grad_norm_m": float(np.mean(grad_m_norms)) if grad_m_norms else "",
            "pm_route_entropy_mean": float(pm_log.get("pm_route_entropy_mean", 0.0)) if pm_log else "",
            "pm_route_confidence_mean": float(pm_log.get("pm_route_confidence_mean", 0.0)) if pm_log else "",
            "pm_route_confidence_p25": float(pm_log.get("pm_route_confidence_p25", 0.0)) if pm_log else "",
            "pm_route_confidence_p75": float(pm_log.get("pm_route_confidence_p75", 0.0)) if pm_log else "",
            "pm_proto_top1_entropy": float(pm_log.get("pm_proto_top1_entropy", 0.0)) if pm_log else "",
            "pm_proto_top1_entropy_over_phase": float(phase_top1_entropy),
            "pm_param_norm_m": float(pm_log.get("pm_param_norm_m", 0.0)) if pm_log else "",
        }

        print_every = int(os.environ.get("PM_PRINT_EVERY", "1"))
        if self.verbose >= 1 and (print_every <= 1 or (self._n_updates % print_every == 0)):
            pg = float(np.mean(pg_losses)) if pg_losses else 0.0
            vl = float(np.mean(value_losses)) if value_losses else 0.0
            sp = float(np.mean(sparse_losses)) if sparse_losses else 0.0
            dv = float(np.mean(div_losses)) if div_losses else 0.0
            st = float(np.mean(stable_losses)) if stable_losses else 0.0
            ax = float(np.mean(aux_losses)) if aux_losses else 0.0
            gm = float(np.mean(grad_m_norms)) if grad_m_norms else 0.0
            re = float(pm_log.get("pm_route_entropy_mean", 0.0)) if pm_log else 0.0
            rc = float(pm_log.get("pm_route_confidence_mean", 0.0)) if pm_log else 0.0
            t1 = float(pm_log.get("pm_proto_top1_entropy", 0.0)) if pm_log else 0.0
            ph = float(phase_top1_entropy)
            print(
                f"{_c('[ProtoMem]', 'cyan', True)} "
                f"{_c('upd', 'blue', True)}={self._n_updates} "
                f"pg={pg:.3f} v={vl:.3f} "
                f"{_c('sp', 'yellow', True)}={sp:.3f} "
                f"{_c('div', 'yellow', True)}={dv:.4f} "
                f"st={st:.3f} aux={ax:.3f} "
                f"{_c('routeH', 'magenta', True)}={re:.3f} "
                f"{_c('conf', 'green', True)}={rc:.3f} "
                f"{_c('top1H', 'green', True)}={t1:.3f} "
                f"{_c('phaseH', 'green', True)}={ph:.3f} "
                f"{_c('gradM', 'red', True)}={gm:.3f}"
            )

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

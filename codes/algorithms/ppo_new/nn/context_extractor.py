from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
try:
    from gymnasium import spaces
except Exception:  # pragma: no cover
    from gym import spaces

try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
except Exception as exc:  # pragma: no cover
    raise ImportError("stable_baselines3 is required for StackedContextExtractor") from exc


def _make_mlp(in_dim: int, out_dim: int, use_layernorm: bool = True) -> nn.Sequential:
    layers = [
        nn.Linear(int(in_dim), int(out_dim)),
        nn.SiLU(),
    ]
    if use_layernorm:
        layers.append(nn.LayerNorm(int(out_dim)))
    return nn.Sequential(*layers)


def _build_adapter(out_dim: int, hidden_dim: int, activation: str) -> nn.Sequential:
    act_name = str(activation or "silu").strip().lower()
    if act_name == "relu":
        act = nn.ReLU
    else:
        act = nn.SiLU
    return nn.Sequential(
        nn.Linear(int(out_dim), int(hidden_dim)),
        act(),
        nn.Linear(int(hidden_dim), int(out_dim)),
    )


class StackedContextExtractor(BaseFeaturesExtractor):
    """
    PPO_NEW v3/v4 extractor for stacked augmented observations.

    Expected per-frame base x_t layout:
        [o_t (D), stage_bit (1), prev_action (1), prev_reward (1), delta_o_t (D)]
    Optional oracle context (v4.2) is appended at the tail and routed into meta branch.
    Stacked obs layout:
        [x_t, x_{t-1}, ..., x_{t-k+1}]  (newest -> oldest)
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        n_stack: int = 4,
        use_branch: bool = True,
        use_layernorm: bool = True,
        embed_dim: int = 32,
        out_dim: int = 64,
        conv_dilation_1: int = 1,
        conv_dilation_2: int = 1,
        enable_adapter: bool = False,
        adapter_hidden_dim: int = 64,
        adapter_activation: str = "silu",
        force_alpha_zero: bool = False,
        alpha_use_tanh: bool = True,
        alpha_limit: float = 1.0,
        alpha_log_interval: int = 0,
        alpha_log_path: str = "",
        oracle_ctx_dim: int = 0,
    ) -> None:
        self.n_stack = int(max(1, n_stack))
        self.use_branch = bool(use_branch)
        self.use_layernorm = bool(use_layernorm)
        self.embed_dim = int(embed_dim)
        self.out_dim = int(out_dim)
        self.conv_dilation_1 = int(max(1, conv_dilation_1))
        self.conv_dilation_2 = int(max(1, conv_dilation_2))
        self.enable_adapter = bool(enable_adapter)
        self.adapter_hidden_dim = int(max(1, adapter_hidden_dim))
        self.adapter_activation = str(adapter_activation or "silu")
        self.force_alpha_zero = bool(force_alpha_zero)
        self.alpha_use_tanh = bool(alpha_use_tanh)
        self.alpha_limit = float(alpha_limit)
        self.alpha_log_interval = int(max(0, alpha_log_interval))
        self.alpha_log_path = str(alpha_log_path or "").strip()
        self.oracle_ctx_dim = int(max(0, oracle_ctx_dim))
        self._alpha_log_step = 0

        obs_shape = tuple(observation_space.shape or ())
        if len(obs_shape) != 1:
            raise ValueError(f"StackedContextExtractor expects 1D obs, got shape={obs_shape}")
        obs_dim = int(obs_shape[0])
        if obs_dim <= 0:
            raise ValueError(f"invalid obs_dim={obs_dim}")
        if obs_dim % self.n_stack != 0:
            raise ValueError(f"obs_dim={obs_dim} is not divisible by n_stack={self.n_stack}")

        self.obs_dim = obs_dim
        self.x_dim = obs_dim // self.n_stack
        self.base_x_dim = int(self.x_dim - self.oracle_ctx_dim)
        if self.base_x_dim <= 3 and self.oracle_ctx_dim <= 0:
            # backward-compatible fallback: infer tail dims if not provided.
            inferred = None
            for cand_ctx in range(0, min(64, self.x_dim + 1)):
                cand_base = self.x_dim - cand_ctx
                if cand_base <= 3:
                    continue
                if (cand_base - 3) % 2 == 0:
                    inferred = cand_ctx
                    self.oracle_ctx_dim = int(cand_ctx)
                    self.base_x_dim = int(cand_base)
                    break
            if inferred is None:
                raise ValueError(f"invalid x_dim={self.x_dim}; cannot infer 2D+3 base layout")
        elif self.base_x_dim <= 3:
            raise ValueError(
                f"invalid x_dim={self.x_dim}; oracle_ctx_dim={self.oracle_ctx_dim} leaves base_x_dim={self.base_x_dim}"
            )
        self.D = (self.base_x_dim - 3) // 2
        if (2 * self.D + 3) != self.base_x_dim:
            raise ValueError(
                f"invalid x_dim={self.x_dim}; base_x_dim={self.base_x_dim}, "
                f"oracle_ctx_dim={self.oracle_ctx_dim}, D={self.D}"
            )
        if self.D <= 0:
            raise ValueError(f"invalid feature split D={self.D}, x_dim={self.x_dim}")

        super().__init__(observation_space, features_dim=self.out_dim)

        if self.use_branch:
            self.o_encoder = _make_mlp(self.D, 16, use_layernorm=self.use_layernorm)
            self.meta_encoder = _make_mlp(3 + self.oracle_ctx_dim, 8, use_layernorm=self.use_layernorm)
            self.delta_encoder = _make_mlp(self.D, 16, use_layernorm=self.use_layernorm)
            token_in_dim = 16 + 8 + 16
            self.token_proj = nn.Linear(token_in_dim, self.embed_dim)
            self.token_norm = nn.LayerNorm(self.embed_dim) if self.use_layernorm else nn.Identity()
        else:
            self.token_proj = nn.Linear(self.x_dim, self.embed_dim)
            self.token_norm = nn.LayerNorm(self.embed_dim) if self.use_layernorm else nn.Identity()

        self.temporal = nn.Sequential(
            nn.Conv1d(
                self.embed_dim,
                self.out_dim,
                kernel_size=3,
                padding=self.conv_dilation_1,
                dilation=self.conv_dilation_1,
            ),
            nn.SiLU(),
            nn.Conv1d(
                self.out_dim,
                self.out_dim,
                kernel_size=3,
                padding=self.conv_dilation_2,
                dilation=self.conv_dilation_2,
            ),
            nn.SiLU(),
        )
        self.out_norm = nn.LayerNorm(self.out_dim) if self.use_layernorm else nn.Identity()
        if self.enable_adapter:
            self.adapter_norm = nn.LayerNorm(self.out_dim)
            self.adapter = _build_adapter(
                out_dim=self.out_dim,
                hidden_dim=self.adapter_hidden_dim,
                activation=self.adapter_activation,
            )
            # alpha=0 ensures strict v3-equivalent behavior at initialization.
            self.alpha = nn.Parameter(torch.zeros(1), requires_grad=not self.force_alpha_zero)
        else:
            self.adapter_norm = nn.Identity()
            self.adapter = nn.Identity()
            self.register_parameter("alpha", None)

    def _split_xt(self, xt: torch.Tensor) -> Dict[str, torch.Tensor]:
        # xt: (batch, n_stack, x_dim)
        base = xt[..., : self.base_x_dim]
        o = base[..., : self.D]
        meta_base = base[..., self.D : self.D + 3]
        delta = base[..., self.D + 3 :]
        if self.oracle_ctx_dim > 0:
            oracle = xt[..., self.base_x_dim : self.base_x_dim + self.oracle_ctx_dim]
            meta = torch.cat([meta_base, oracle], dim=-1)
        else:
            meta = meta_base
        return {"o": o, "meta": meta, "delta": delta}

    def _effective_alpha(self, h: torch.Tensor) -> torch.Tensor:
        if (not self.enable_adapter) or self.alpha is None or self.force_alpha_zero:
            return h.new_zeros(1)
        alpha = self.alpha
        if self.alpha_use_tanh:
            alpha = torch.tanh(alpha)
        if self.alpha_limit > 0:
            alpha = torch.clamp(alpha, min=-self.alpha_limit, max=self.alpha_limit)
        return alpha.to(dtype=h.dtype, device=h.device)

    def _maybe_log_alpha(self, alpha_eff: torch.Tensor) -> None:
        if (not self.enable_adapter) or (not self.training):
            return
        self._alpha_log_step += 1
        if self.alpha_log_interval <= 0:
            return
        if (self._alpha_log_step % self.alpha_log_interval) != 0:
            return
        try:
            alpha_raw = 0.0 if self.alpha is None else float(self.alpha.detach().cpu().item())
            alpha_eff_v = float(alpha_eff.detach().cpu().item())
            msg = (
                f"[PPO_NEW_V4] alpha_raw={alpha_raw:.6f} alpha_eff={alpha_eff_v:.6f} "
                f"step={self._alpha_log_step} force_alpha_zero={int(self.force_alpha_zero)}"
            )
            print(msg)
            if self.alpha_log_path:
                line = f"{datetime.now().isoformat()} {msg}\n"
                Path(self.alpha_log_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.alpha_log_path, "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception:
            pass

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (batch, obs_dim)
        if observations.ndim != 2:
            observations = observations.view(observations.shape[0], -1)
        batch = int(observations.shape[0])
        xt = observations.view(batch, self.n_stack, self.x_dim)

        if self.use_branch:
            parts = self._split_xt(xt)
            o_emb = self.o_encoder(parts["o"])
            meta_emb = self.meta_encoder(parts["meta"])
            delta_emb = self.delta_encoder(parts["delta"])
            token = torch.cat([o_emb, meta_emb, delta_emb], dim=-1)
            token = self.token_proj(token)
            token = self.token_norm(token)
        else:
            token = self.token_proj(xt)
            token = self.token_norm(token)

        # newest -> oldest order is preserved as provided in stacked observation.
        token = token.permute(0, 2, 1)  # (batch, embed_dim, n_stack)
        h = self.temporal(token)
        h = h.mean(dim=-1)  # temporal mean pooling
        h = self.out_norm(h)
        if self.enable_adapter:
            alpha_eff = self._effective_alpha(h)
            adapted = self.adapter(self.adapter_norm(h))
            h = h + alpha_eff * adapted
            self._maybe_log_alpha(alpha_eff)
        return h

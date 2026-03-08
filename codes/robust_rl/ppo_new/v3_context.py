from __future__ import annotations

import os
from typing import Any, Dict

try:
    from stable_baselines3 import PPO
except Exception:
    PPO = None
try:
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.distributions import CategoricalDistribution
except Exception:
    ActorCriticPolicy = None
    CategoricalDistribution = None

from .v5_variants import (
    ActionBalancedPPO,
    ActionConditionalCriticPolicy,
    AuxWeakPPO,
    AuxWeakPolicy,
    QCriticPPO,
)
from .v6_variants import CvarPPO
from .v6_cadm import CadmAuxPolicy, CadmAuxPPO
from .v7_tcr import TriggeredCounterfactualPPO
from .v8_v9_a import PPOPostV8A, PPOPostV9A
from .v8_v9_a2 import (
    PPOPostV8A2,
    PPOPostV9A2,
    PPOPostV10A,
    PPOPostV10B,
    PPOPostV10C,
    PPOPostV10D,
    PPOPostV10E,
)


class Action1LogitBiasPolicy(ActorCriticPolicy if ActorCriticPolicy is not None else object):
    """
    Minimal diagnostic policy:
    add a fixed bias to action-1 logit for Discrete(2) action space.
    """

    def __init__(self, *args, action1_logit_bias: float = 0.3, **kwargs):
        self.action1_logit_bias = float(action1_logit_bias)
        if ActorCriticPolicy is None:
            raise ImportError("stable_baselines3 is required for Action1LogitBiasPolicy.")
        super().__init__(*args, **kwargs)

    def _get_action_dist_from_latent(self, latent_pi):
        if CategoricalDistribution is None or not isinstance(self.action_dist, CategoricalDistribution):
            return super()._get_action_dist_from_latent(latent_pi)
        action_logits = self.action_net(latent_pi)
        try:
            if action_logits.shape[-1] >= 2 and self.action1_logit_bias != 0.0:
                action_logits = action_logits.clone()
                action_logits[..., 1] = action_logits[..., 1] + float(self.action1_logit_bias)
        except Exception:
            pass
        return self.action_dist.proba_distribution(action_logits=action_logits)


def _build_model_v3_impl(
    env,
    seed,
    device="cpu",
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
    **kwargs,
):
    if PPO is None:
        raise ImportError("stable_baselines3 is required for PPO_NEW v3.")
    try:
        from algorithms.ppo_new.nn.context_extractor import StackedContextExtractor
    except Exception as exc:
        raise ImportError("Failed to import StackedContextExtractor for PPO_NEW v3") from exc

    ppo_kwargs: Dict[str, Any] = dict(kwargs or {})
    ppo_cls = ppo_kwargs.pop("ppo_cls", PPO)
    policy = ppo_kwargs.pop("policy", "MlpPolicy")
    if "force_alpha_zero" not in ppo_kwargs:
        env_force_alpha = os.environ.get("RL_PPO_NEW_FORCE_ALPHA_ZERO", "").strip()
        if env_force_alpha:
            ppo_kwargs["force_alpha_zero"] = env_force_alpha == "1"
    if "alpha_log_interval" not in ppo_kwargs:
        env_alpha_log_interval = os.environ.get("RL_PPO_NEW_ALPHA_LOG_INTERVAL", "").strip()
        if env_alpha_log_interval:
            try:
                ppo_kwargs["alpha_log_interval"] = int(env_alpha_log_interval)
            except Exception:
                pass
    if "alpha_log_path" not in ppo_kwargs:
        env_alpha_log_path = os.environ.get("RL_PPO_NEW_ALPHA_LOG_PATH", "").strip()
        if env_alpha_log_path:
            ppo_kwargs["alpha_log_path"] = env_alpha_log_path

    env_stack = getattr(env, "_aug_window_k", None)
    try:
        env_stack = int(env_stack) if env_stack is not None else None
    except Exception:
        env_stack = None
    env_oracle_ctx_dim = getattr(env, "_oracle_ctx_dim", 0)
    try:
        env_oracle_ctx_dim = int(env_oracle_ctx_dim) if env_oracle_ctx_dim is not None else 0
    except Exception:
        env_oracle_ctx_dim = 0

    stack_k = int(ppo_kwargs.pop("n_stack", n_stack if n_stack is not None else 4))
    if env_stack is not None and env_stack > 0:
        stack_k = env_stack
    stack_k = max(1, stack_k)

    d1 = int(ppo_kwargs.pop("conv_dilation_1", conv_dilation_1))
    d2 = int(ppo_kwargs.pop("conv_dilation_2", conv_dilation_2))
    d1 = max(1, d1)
    d2 = max(1, d2)
    enable_adapter = bool(ppo_kwargs.pop("enable_adapter", enable_adapter))
    adapter_hidden_dim = int(ppo_kwargs.pop("adapter_hidden_dim", adapter_hidden_dim))
    adapter_hidden_dim = max(1, adapter_hidden_dim)
    adapter_activation = str(ppo_kwargs.pop("adapter_activation", adapter_activation))
    force_alpha_zero = bool(ppo_kwargs.pop("force_alpha_zero", force_alpha_zero))
    alpha_use_tanh = bool(ppo_kwargs.pop("alpha_use_tanh", alpha_use_tanh))
    alpha_limit = float(ppo_kwargs.pop("alpha_limit", alpha_limit))
    alpha_log_interval = int(ppo_kwargs.pop("alpha_log_interval", alpha_log_interval))
    alpha_log_interval = max(0, alpha_log_interval)
    alpha_log_path = str(ppo_kwargs.pop("alpha_log_path", alpha_log_path) or "")

    incoming_policy_kwargs = ppo_kwargs.pop("policy_kwargs", None)
    policy_kwargs: Dict[str, Any] = dict(incoming_policy_kwargs or {})
    policy_kwargs["features_extractor_class"] = StackedContextExtractor
    policy_kwargs["features_extractor_kwargs"] = {
        "n_stack": int(stack_k),
        "use_branch": bool(use_branch),
        "use_layernorm": bool(use_layernorm),
        "embed_dim": int(embed_dim),
        "out_dim": int(out_dim),
        "conv_dilation_1": int(d1),
        "conv_dilation_2": int(d2),
        "enable_adapter": bool(enable_adapter),
        "adapter_hidden_dim": int(adapter_hidden_dim),
        "adapter_activation": str(adapter_activation),
        "force_alpha_zero": bool(force_alpha_zero),
        "alpha_use_tanh": bool(alpha_use_tanh),
        "alpha_limit": float(alpha_limit),
        "alpha_log_interval": int(alpha_log_interval),
        "alpha_log_path": str(alpha_log_path),
        "oracle_ctx_dim": int(max(0, env_oracle_ctx_dim)),
    }

    ppo_kwargs.setdefault("n_steps", 10)
    ppo_kwargs.setdefault("verbose", 1)

    return ppo_cls(policy, env, device=device, seed=seed, policy_kwargs=policy_kwargs, **ppo_kwargs)


def build_model_v3(
    env,
    seed,
    device="cpu",
    n_stack: int = 4,
    use_branch: bool = True,
    use_layernorm: bool = True,
    embed_dim: int = 32,
    out_dim: int = 64,
    **kwargs,
):
    return _build_model_v3_impl(
        env=env,
        seed=seed,
        device=device,
        n_stack=int(n_stack),
        use_branch=bool(use_branch),
        use_layernorm=bool(use_layernorm),
        embed_dim=int(embed_dim),
        out_dim=int(out_dim),
        conv_dilation_1=1,
        conv_dilation_2=1,
        **kwargs,
    )


def build_model_v31(
    env,
    seed,
    device="cpu",
    n_stack: int = 8,
    use_branch: bool = True,
    use_layernorm: bool = True,
    embed_dim: int = 32,
    out_dim: int = 64,
    **kwargs,
):
    return _build_model_v3_impl(
        env=env,
        seed=seed,
        device=device,
        n_stack=int(n_stack),
        use_branch=bool(use_branch),
        use_layernorm=bool(use_layernorm),
        embed_dim=int(embed_dim),
        out_dim=int(out_dim),
        conv_dilation_1=1,
        conv_dilation_2=1,
        **kwargs,
    )


def build_model_v32(
    env,
    seed,
    device="cpu",
    n_stack: int = 4,
    use_branch: bool = True,
    use_layernorm: bool = True,
    embed_dim: int = 32,
    out_dim: int = 64,
    **kwargs,
):
    return _build_model_v3_impl(
        env=env,
        seed=seed,
        device=device,
        n_stack=int(n_stack),
        use_branch=bool(use_branch),
        use_layernorm=bool(use_layernorm),
        embed_dim=int(embed_dim),
        out_dim=int(out_dim),
        conv_dilation_1=1,
        conv_dilation_2=2,
        **kwargs,
    )


def build_model_v4(
    env,
    seed,
    device="cpu",
    n_stack: int = 4,
    use_branch: bool = True,
    use_layernorm: bool = True,
    embed_dim: int = 32,
    out_dim: int = 64,
    force_alpha_zero: bool = False,
    alpha_log_interval: int = 200,
    **kwargs,
):
    extra_kwargs: Dict[str, Any] = dict(kwargs or {})
    alpha_log_path = str(extra_kwargs.pop("alpha_log_path", "") or "")
    return _build_model_v3_impl(
        env=env,
        seed=seed,
        device=device,
        n_stack=int(n_stack),
        use_branch=bool(use_branch),
        use_layernorm=bool(use_layernorm),
        embed_dim=int(embed_dim),
        out_dim=int(out_dim),
        conv_dilation_1=1,
        conv_dilation_2=1,
        enable_adapter=True,
        adapter_hidden_dim=64,
        adapter_activation="silu",
        force_alpha_zero=bool(force_alpha_zero),
        alpha_use_tanh=True,
        alpha_limit=1.0,
        alpha_log_interval=int(alpha_log_interval),
        alpha_log_path=alpha_log_path,
        **extra_kwargs,
    )


def build_model_v41(
    env,
    seed,
    device="cpu",
    n_stack: int = 4,
    use_branch: bool = True,
    use_layernorm: bool = True,
    embed_dim: int = 32,
    out_dim: int = 64,
    force_alpha_zero: bool = False,
    alpha_log_interval: int = 200,
    **kwargs,
):
    return build_model_v4(
        env=env,
        seed=seed,
        device=device,
        n_stack=int(n_stack),
        use_branch=bool(use_branch),
        use_layernorm=bool(use_layernorm),
        embed_dim=int(embed_dim),
        out_dim=int(out_dim),
        force_alpha_zero=bool(force_alpha_zero),
        alpha_log_interval=int(alpha_log_interval),
        **kwargs,
    )


def build_model_v42_phase(
    env,
    seed,
    device="cpu",
    n_stack: int = 4,
    use_branch: bool = True,
    use_layernorm: bool = True,
    embed_dim: int = 32,
    out_dim: int = 64,
    **kwargs,
):
    # Keep PPO settings aligned with v3; only observation carries oracle context.
    return build_model_v3(
        env=env,
        seed=seed,
        device=device,
        n_stack=int(n_stack),
        use_branch=bool(use_branch),
        use_layernorm=bool(use_layernorm),
        embed_dim=int(embed_dim),
        out_dim=int(out_dim),
        **kwargs,
    )


def build_model_v42_mean(
    env,
    seed,
    device="cpu",
    n_stack: int = 4,
    use_branch: bool = True,
    use_layernorm: bool = True,
    embed_dim: int = 32,
    out_dim: int = 64,
    **kwargs,
):
    # Keep PPO settings aligned with v3; only observation carries oracle context.
    return build_model_v3(
        env=env,
        seed=seed,
        device=device,
        n_stack=int(n_stack),
        use_branch=bool(use_branch),
        use_layernorm=bool(use_layernorm),
        embed_dim=int(embed_dim),
        out_dim=int(out_dim),
        **kwargs,
    )


def build_model_v43_ent(
    env,
    seed,
    device="cpu",
    n_stack: int = 4,
    use_branch: bool = True,
    use_layernorm: bool = True,
    embed_dim: int = 32,
    out_dim: int = 64,
    ent_coef: float = 0.02,
    **kwargs,
):
    extra_kwargs: Dict[str, Any] = dict(kwargs or {})
    if "ent_coef" not in extra_kwargs:
        try:
            extra_kwargs["ent_coef"] = float(os.environ.get("RL_PPO_NEW_ENT_COEF", str(ent_coef)))
        except Exception:
            extra_kwargs["ent_coef"] = float(ent_coef)
    return build_model_v3(
        env=env,
        seed=seed,
        device=device,
        n_stack=int(n_stack),
        use_branch=bool(use_branch),
        use_layernorm=bool(use_layernorm),
        embed_dim=int(embed_dim),
        out_dim=int(out_dim),
        **extra_kwargs,
    )


def build_model_v43_logit_bias(
    env,
    seed,
    device="cpu",
    n_stack: int = 4,
    use_branch: bool = True,
    use_layernorm: bool = True,
    embed_dim: int = 32,
    out_dim: int = 64,
    action1_logit_bias: float = 0.3,
    **kwargs,
):
    extra_kwargs: Dict[str, Any] = dict(kwargs or {})
    try:
        bias_val = float(extra_kwargs.pop("action1_logit_bias", os.environ.get("RL_PPO_NEW_LOGIT_BIAS", str(action1_logit_bias))))
    except Exception:
        bias_val = float(action1_logit_bias)

    incoming_policy_kwargs = dict(extra_kwargs.pop("policy_kwargs", None) or {})
    incoming_policy_kwargs["action1_logit_bias"] = float(bias_val)
    extra_kwargs["policy_kwargs"] = incoming_policy_kwargs
    # master currently passes policy='MlpPolicy'; drop it to avoid duplicate policy kwarg.
    extra_kwargs.pop("policy", None)

    return _build_model_v3_impl(
        env=env,
        seed=seed,
        device=device,
        n_stack=int(n_stack),
        use_branch=bool(use_branch),
        use_layernorm=bool(use_layernorm),
        embed_dim=int(embed_dim),
        out_dim=int(out_dim),
        conv_dilation_1=1,
        conv_dilation_2=1,
        policy=Action1LogitBiasPolicy,
        **extra_kwargs,
    )


def build_model_v51_abppo(
    env,
    seed,
    device="cpu",
    n_stack: int = 4,
    use_branch: bool = True,
    use_layernorm: bool = True,
    embed_dim: int = 32,
    out_dim: int = 64,
    ab_freq_epsilon: float = 1e-6,
    ab_w_max: float = 8.0,
    ab_group_adv_norm: bool = True,
    ab_group_min_samples: int = 8,
    **kwargs,
):
    extra_kwargs: Dict[str, Any] = dict(kwargs or {})
    extra_kwargs.setdefault("ab_freq_epsilon", float(os.environ.get("RL_AB_FREQ_EPS", str(ab_freq_epsilon))))
    extra_kwargs.setdefault("ab_w_max", float(os.environ.get("RL_AB_W_MAX", str(ab_w_max))))
    extra_kwargs.setdefault("ab_group_adv_norm", os.environ.get("RL_AB_GROUP_ADV_NORM", "1").strip() == "1")
    extra_kwargs.setdefault("ab_group_min_samples", int(os.environ.get("RL_AB_GROUP_MIN", str(ab_group_min_samples))))
    return _build_model_v3_impl(
        env=env,
        seed=seed,
        device=device,
        n_stack=int(n_stack),
        use_branch=bool(use_branch),
        use_layernorm=bool(use_layernorm),
        embed_dim=int(embed_dim),
        out_dim=int(out_dim),
        conv_dilation_1=1,
        conv_dilation_2=1,
        ppo_cls=ActionBalancedPPO,
        **extra_kwargs,
    )


def build_model_v52_qcritic(
    env,
    seed,
    device="cpu",
    n_stack: int = 4,
    use_branch: bool = True,
    use_layernorm: bool = True,
    embed_dim: int = 32,
    out_dim: int = 64,
    **kwargs,
):
    extra_kwargs: Dict[str, Any] = dict(kwargs or {})
    # master currently passes policy='MlpPolicy'; drop it to avoid duplicate policy kwarg.
    extra_kwargs.pop("policy", None)
    return _build_model_v3_impl(
        env=env,
        seed=seed,
        device=device,
        n_stack=int(n_stack),
        use_branch=bool(use_branch),
        use_layernorm=bool(use_layernorm),
        embed_dim=int(embed_dim),
        out_dim=int(out_dim),
        conv_dilation_1=1,
        conv_dilation_2=1,
        ppo_cls=QCriticPPO,
        policy=ActionConditionalCriticPolicy,
        **extra_kwargs,
    )


def build_model_v53_auxweak(
    env,
    seed,
    device="cpu",
    n_stack: int = 4,
    use_branch: bool = True,
    use_layernorm: bool = True,
    embed_dim: int = 32,
    out_dim: int = 64,
    aux_lambda: float = 0.2,
    aux_severity_threshold: float = 0.5,
    **kwargs,
):
    extra_kwargs: Dict[str, Any] = dict(kwargs or {})
    extra_kwargs.setdefault("aux_lambda", float(os.environ.get("RL_AUX_LAMBDA", str(aux_lambda))))
    extra_kwargs.setdefault(
        "aux_severity_threshold",
        float(os.environ.get("RL_AUX_SEVERITY_THRESHOLD", str(aux_severity_threshold))),
    )
    # master currently passes policy='MlpPolicy'; drop it to avoid duplicate policy kwarg.
    extra_kwargs.pop("policy", None)
    return _build_model_v3_impl(
        env=env,
        seed=seed,
        device=device,
        n_stack=int(n_stack),
        use_branch=bool(use_branch),
        use_layernorm=bool(use_layernorm),
        embed_dim=int(embed_dim),
        out_dim=int(out_dim),
        conv_dilation_1=1,
        conv_dilation_2=1,
        ppo_cls=AuxWeakPPO,
        policy=AuxWeakPolicy,
        **extra_kwargs,
    )


def build_model_v61_cvarppo(
    env,
    seed,
    device="cpu",
    cvar_alpha: float = 0.30,
    cvar_beta: float = 1.0,
    cvar_w_max: float = 3.0,
    **kwargs,
):
    """
    V6.1:
      - plain PPO observation interface (no v3 extractor)
      - only CVaR-style tail reweighting in policy loss
    """
    if PPO is None:
        raise ImportError("stable_baselines3 is required for PPO_NEW v6.1.")
    extra_kwargs: Dict[str, Any] = dict(kwargs or {})
    extra_kwargs.setdefault("cvar_alpha", float(os.environ.get("RL_PPO_NEW_CVAR_ALPHA", str(cvar_alpha))))
    extra_kwargs.setdefault("cvar_beta", float(os.environ.get("RL_PPO_NEW_CVAR_BETA", str(cvar_beta))))
    extra_kwargs.setdefault("cvar_w_max", float(os.environ.get("RL_PPO_NEW_CVAR_WMAX", str(cvar_w_max))))

    # master currently passes policy='MlpPolicy'; keep exactly one policy argument.
    policy = extra_kwargs.pop("policy", "MlpPolicy")
    extra_kwargs.setdefault("n_steps", 10)
    extra_kwargs.setdefault("verbose", 1)
    return CvarPPO(policy, env, device=device, seed=seed, **extra_kwargs)


def build_model_v62_v3cvar(
    env,
    seed,
    device="cpu",
    n_stack: int = 4,
    use_branch: bool = True,
    use_layernorm: bool = True,
    embed_dim: int = 32,
    out_dim: int = 64,
    cvar_alpha: float = 0.30,
    cvar_beta: float = 1.0,
    cvar_w_max: float = 3.0,
    **kwargs,
):
    """
    V6.2:
      - keep V3 representation (aug obs + stacking + context extractor)
      - apply the same CVaR-style loss reweighting as V6.1
    """
    extra_kwargs: Dict[str, Any] = dict(kwargs or {})
    extra_kwargs.setdefault("cvar_alpha", float(os.environ.get("RL_PPO_NEW_CVAR_ALPHA", str(cvar_alpha))))
    extra_kwargs.setdefault("cvar_beta", float(os.environ.get("RL_PPO_NEW_CVAR_BETA", str(cvar_beta))))
    extra_kwargs.setdefault("cvar_w_max", float(os.environ.get("RL_PPO_NEW_CVAR_WMAX", str(cvar_w_max))))
    return _build_model_v3_impl(
        env=env,
        seed=seed,
        device=device,
        n_stack=int(n_stack),
        use_branch=bool(use_branch),
        use_layernorm=bool(use_layernorm),
        embed_dim=int(embed_dim),
        out_dim=int(out_dim),
        conv_dilation_1=1,
        conv_dilation_2=1,
        ppo_cls=CvarPPO,
        **extra_kwargs,
    )


def build_model_v63_cadm(
    env,
    seed,
    device="cpu",
    n_stack: int = 4,
    use_branch: bool = True,
    use_layernorm: bool = True,
    embed_dim: int = 32,
    out_dim: int = 64,
    cadm_beta: float = 0.05,
    cadm_lambda_next: float = 1.0,
    cadm_lambda_reward: float = 0.5,
    cadm_aux_batch_size: int = 64,
    cadm_max_transitions: int = 4096,
    **kwargs,
):
    """
    V6.3 CaDM-adapted PPO:
      - keep V3 representation (stacked context extractor)
      - add auxiliary forward prediction loss on (next severity, reward_t)
    """
    extra_kwargs: Dict[str, Any] = dict(kwargs or {})
    extra_kwargs.setdefault("cadm_beta", float(os.environ.get("RL_CADM_BETA", str(cadm_beta))))
    extra_kwargs.setdefault("cadm_lambda_next", float(os.environ.get("RL_CADM_LAMBDA_NEXT", str(cadm_lambda_next))))
    extra_kwargs.setdefault("cadm_lambda_reward", float(os.environ.get("RL_CADM_LAMBDA_REWARD", str(cadm_lambda_reward))))
    extra_kwargs.setdefault(
        "cadm_aux_batch_size",
        int(os.environ.get("RL_CADM_AUX_BATCH_SIZE", str(cadm_aux_batch_size))),
    )
    extra_kwargs.setdefault(
        "cadm_max_transitions",
        int(os.environ.get("RL_CADM_MAX_TRANSITIONS", str(cadm_max_transitions))),
    )
    # master currently passes policy='MlpPolicy'; drop it to avoid duplicate policy kwarg.
    extra_kwargs.pop("policy", None)
    return _build_model_v3_impl(
        env=env,
        seed=seed,
        device=device,
        n_stack=int(n_stack),
        use_branch=bool(use_branch),
        use_layernorm=bool(use_layernorm),
        embed_dim=int(embed_dim),
        out_dim=int(out_dim),
        conv_dilation_1=1,
        conv_dilation_2=1,
        ppo_cls=CadmAuxPPO,
        policy=CadmAuxPolicy,
        **extra_kwargs,
    )


def build_model_v71_poolppo(
    env,
    seed,
    device="cpu",
    **kwargs,
):
    """
    V7.1:
      - PPO-compatible plain observation branch (for protocol-level ablation)
      - no extra loss/network trick
    """
    if PPO is None:
        raise ImportError("stable_baselines3 is required for PPO_NEW v7.1.")
    extra_kwargs: Dict[str, Any] = dict(kwargs or {})
    policy = extra_kwargs.pop("policy", "MlpPolicy")
    extra_kwargs.setdefault("n_steps", 10)
    extra_kwargs.setdefault("verbose", 1)
    return PPO(policy, env, device=device, seed=seed, **extra_kwargs)


def build_model_v72_poolv3(
    env,
    seed,
    device="cpu",
    n_stack: int = 4,
    use_branch: bool = True,
    use_layernorm: bool = True,
    embed_dim: int = 32,
    out_dim: int = 64,
    **kwargs,
):
    """
    V7.2:
      - keep V3 representation branch (aug obs + stacking + context extractor)
      - no extra objective trick
    """
    return build_model_v3(
        env=env,
        seed=seed,
        device=device,
        n_stack=int(n_stack),
        use_branch=bool(use_branch),
        use_layernorm=bool(use_layernorm),
        embed_dim=int(embed_dim),
        out_dim=int(out_dim),
        **kwargs,
    )


def build_model_v73_tcrppo(
    env,
    seed,
    device="cpu",
    **kwargs,
):
    """
    V7.3:
      - PPO-compatible plain observation branch
      - Triggered Counterfactual/Teacher Replay auxiliary objective
    """
    if PPO is None:
        raise ImportError("stable_baselines3 is required for PPO_NEW v7.3.")
    extra_kwargs: Dict[str, Any] = dict(kwargs or {})
    policy = extra_kwargs.pop("policy", "MlpPolicy")
    extra_kwargs.setdefault("n_steps", 10)
    extra_kwargs.setdefault("verbose", 1)
    return TriggeredCounterfactualPPO(policy, env, device=device, seed=seed, **extra_kwargs)


def build_model_v74_tcrv3(
    env,
    seed,
    device="cpu",
    n_stack: int = 4,
    use_branch: bool = True,
    use_layernorm: bool = True,
    embed_dim: int = 32,
    out_dim: int = 64,
    **kwargs,
):
    """
    V7.4:
      - keep V3 representation branch (aug obs + stacking + context extractor)
      - Triggered Counterfactual/Teacher Replay auxiliary objective
    """
    return _build_model_v3_impl(
        env=env,
        seed=seed,
        device=device,
        n_stack=int(n_stack),
        use_branch=bool(use_branch),
        use_layernorm=bool(use_layernorm),
        embed_dim=int(embed_dim),
        out_dim=int(out_dim),
        conv_dilation_1=1,
        conv_dilation_2=1,
        ppo_cls=TriggeredCounterfactualPPO,
        **kwargs,
    )


def build_model_v8_a(
    env,
    seed,
    device="cpu",
    n_stack: int = 4,
    use_branch: bool = True,
    use_layernorm: bool = True,
    embed_dim: int = 32,
    out_dim: int = 64,
    **kwargs,
):
    """
    V8-A:
      - keep V3 representation branch (aug obs + stacking + context extractor)
      - quality-weighted triggered replay with KL-guarded auxiliary term
    """
    return _build_model_v3_impl(
        env=env,
        seed=seed,
        device=device,
        n_stack=int(n_stack),
        use_branch=bool(use_branch),
        use_layernorm=bool(use_layernorm),
        embed_dim=int(embed_dim),
        out_dim=int(out_dim),
        conv_dilation_1=1,
        conv_dilation_2=1,
        ppo_cls=PPOPostV8A,
        **kwargs,
    )


def build_model_v9_a(
    env,
    seed,
    device="cpu",
    n_stack: int = 4,
    use_branch: bool = True,
    use_layernorm: bool = True,
    embed_dim: int = 32,
    out_dim: int = 64,
    **kwargs,
):
    """
    V9-A:
      - keep V3 representation branch (aug obs + stacking + context extractor)
      - V8-A plus adaptive kappa scaling from rollout gap-score EMA
    """
    return _build_model_v3_impl(
        env=env,
        seed=seed,
        device=device,
        n_stack=int(n_stack),
        use_branch=bool(use_branch),
        use_layernorm=bool(use_layernorm),
        embed_dim=int(embed_dim),
        out_dim=int(out_dim),
        conv_dilation_1=1,
        conv_dilation_2=1,
        ppo_cls=PPOPostV9A,
        **kwargs,
    )


def build_model_v8_a2(
    env,
    seed,
    device="cpu",
    n_stack: int = 4,
    use_branch: bool = True,
    use_layernorm: bool = True,
    embed_dim: int = 32,
    out_dim: int = 64,
    **kwargs,
):
    """
    V8-A.2:
      - keep V3 representation branch (aug obs + stacking + context extractor)
      - trigger when any action ratio is too low; targeted synthetic replay correction
    """
    return _build_model_v3_impl(
        env=env,
        seed=seed,
        device=device,
        n_stack=int(n_stack),
        use_branch=bool(use_branch),
        use_layernorm=bool(use_layernorm),
        embed_dim=int(embed_dim),
        out_dim=int(out_dim),
        conv_dilation_1=1,
        conv_dilation_2=1,
        ppo_cls=PPOPostV8A2,
        **kwargs,
    )


def build_model_v9_a2(
    env,
    seed,
    device="cpu",
    n_stack: int = 4,
    use_branch: bool = True,
    use_layernorm: bool = True,
    embed_dim: int = 32,
    out_dim: int = 64,
    **kwargs,
):
    """
    V9-A.2:
      - keep V3 representation branch (aug obs + stacking + context extractor)
      - V8-A.2 plus adaptive threshold and adaptive correction strength
    """
    return _build_model_v3_impl(
        env=env,
        seed=seed,
        device=device,
        n_stack=int(n_stack),
        use_branch=bool(use_branch),
        use_layernorm=bool(use_layernorm),
        embed_dim=int(embed_dim),
        out_dim=int(out_dim),
        conv_dilation_1=1,
        conv_dilation_2=1,
        ppo_cls=PPOPostV9A2,
        **kwargs,
    )


def _build_model_v10_impl(
    ppo_cls,
    env,
    seed,
    device="cpu",
    n_stack: int = 4,
    use_branch: bool = True,
    use_layernorm: bool = True,
    embed_dim: int = 32,
    out_dim: int = 64,
    **kwargs,
):
    """
    V10 family:
      - keep V3 representation branch (aug obs + stacking + context extractor)
      - targeted large-scale action correction in Phase2-style auxiliary replay
    """
    return _build_model_v3_impl(
        env=env,
        seed=seed,
        device=device,
        n_stack=int(n_stack),
        use_branch=bool(use_branch),
        use_layernorm=bool(use_layernorm),
        embed_dim=int(embed_dim),
        out_dim=int(out_dim),
        conv_dilation_1=1,
        conv_dilation_2=1,
        ppo_cls=ppo_cls,
        **kwargs,
    )


def build_model_v10_a(
    env,
    seed,
    device="cpu",
    n_stack: int = 4,
    use_branch: bool = True,
    use_layernorm: bool = True,
    embed_dim: int = 32,
    out_dim: int = 64,
    **kwargs,
):
    return _build_model_v10_impl(
        ppo_cls=PPOPostV10A,
        env=env,
        seed=seed,
        device=device,
        n_stack=int(n_stack),
        use_branch=bool(use_branch),
        use_layernorm=bool(use_layernorm),
        embed_dim=int(embed_dim),
        out_dim=int(out_dim),
        **kwargs,
    )


def build_model_v10_b(
    env,
    seed,
    device="cpu",
    n_stack: int = 4,
    use_branch: bool = True,
    use_layernorm: bool = True,
    embed_dim: int = 32,
    out_dim: int = 64,
    **kwargs,
):
    return _build_model_v10_impl(
        ppo_cls=PPOPostV10B,
        env=env,
        seed=seed,
        device=device,
        n_stack=int(n_stack),
        use_branch=bool(use_branch),
        use_layernorm=bool(use_layernorm),
        embed_dim=int(embed_dim),
        out_dim=int(out_dim),
        **kwargs,
    )


def build_model_v10_c(
    env,
    seed,
    device="cpu",
    n_stack: int = 4,
    use_branch: bool = True,
    use_layernorm: bool = True,
    embed_dim: int = 32,
    out_dim: int = 64,
    **kwargs,
):
    return _build_model_v10_impl(
        ppo_cls=PPOPostV10C,
        env=env,
        seed=seed,
        device=device,
        n_stack=int(n_stack),
        use_branch=bool(use_branch),
        use_layernorm=bool(use_layernorm),
        embed_dim=int(embed_dim),
        out_dim=int(out_dim),
        **kwargs,
    )


def build_model_v10_d(
    env,
    seed,
    device="cpu",
    n_stack: int = 4,
    use_branch: bool = True,
    use_layernorm: bool = True,
    embed_dim: int = 32,
    out_dim: int = 64,
    **kwargs,
):
    return _build_model_v10_impl(
        ppo_cls=PPOPostV10D,
        env=env,
        seed=seed,
        device=device,
        n_stack=int(n_stack),
        use_branch=bool(use_branch),
        use_layernorm=bool(use_layernorm),
        embed_dim=int(embed_dim),
        out_dim=int(out_dim),
        **kwargs,
    )


def build_model_v10_e(
    env,
    seed,
    device="cpu",
    n_stack: int = 4,
    use_branch: bool = True,
    use_layernorm: bool = True,
    embed_dim: int = 32,
    out_dim: int = 64,
    **kwargs,
):
    return _build_model_v10_impl(
        ppo_cls=PPOPostV10E,
        env=env,
        seed=seed,
        device=device,
        n_stack=int(n_stack),
        use_branch=bool(use_branch),
        use_layernorm=bool(use_layernorm),
        embed_dim=int(embed_dim),
        out_dim=int(out_dim),
        **kwargs,
    )

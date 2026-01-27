from __future__ import annotations

import json
import os
from typing import Any, Dict


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if raw == "":
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if raw == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if raw == "":
        return bool(default)
    return raw in {"1", "true", "yes", "y", "on"}


def _get_str(name: str, default: str) -> str:
    raw = os.getenv(name, "").strip()
    return raw if raw != "" else str(default)


LBKLAC_DEFAULTS: Dict[str, Any] = {
    # History / belief
    "history_len": _get_int("LBKLAC_HISTORY_LEN", 50),
    "belief_dim": _get_int("LBKLAC_BELIEF_DIM", 20),
    "transformer_layers": _get_int("LBKLAC_T_LAYERS", 2),
    "transformer_heads": _get_int("LBKLAC_T_HEADS", 2),
    "transformer_hidden_dim": _get_int("LBKLAC_T_HIDDEN", 64),
    "transformer_dropout": _get_float("LBKLAC_T_DROPOUT", 0.1),
    "use_causal_mask": _get_bool("LBKLAC_CAUSAL_MASK", True),
    "belief_kl_sigma": _get_float("LBKLAC_BELIEF_KL_SIGMA", 1.0),
    # Replay / update
    "buffer_size": _get_int("LBKLAC_BUFFER_SIZE", 1000),
    "batch_size": _get_int("LBKLAC_BATCH_SIZE", 32),
    "update_every": _get_int("LBKLAC_UPDATE_EVERY", 4),
    "update_iters": _get_int("LBKLAC_UPDATE_ITERS", 1),
    "recent_window": _get_int("LBKLAC_RECENT_WINDOW", 200),
    # Loss weights / optimizer
    "gamma": _get_float("LBKLAC_GAMMA", 0.9),
    "bootstrap_value": _get_bool("LBKLAC_BOOTSTRAP", False),
    "beta": _get_float("LBKLAC_BETA", 0.2),
    "c_v": _get_float("LBKLAC_CV", 0.5),
    "c_h": _get_float("LBKLAC_CH", 0.01),
    "learning_rate": _get_float("LBKLAC_LR", 5e-4),
    "grad_clip": _get_float("LBKLAC_GRAD_CLIP", 1.0),
    "use_ppo_clip": _get_bool("LBKLAC_USE_PPO_CLIP", False),
    "policy_clip": _get_float("LBKLAC_POLICY_CLIP", 0.2),
    # Trust region + OOD
    "delta_init": _get_float("LBKLAC_DELTA_INIT", 0.1),
    "delta_min": _get_float("LBKLAC_DELTA_MIN", 0.01),
    "delta_max": _get_float("LBKLAC_DELTA_MAX", 0.5),
    "kappa_up": _get_float("LBKLAC_KAPPA_UP", 1.5),
    "kappa_down": _get_float("LBKLAC_KAPPA_DOWN", 0.9),
    "tau_kl": _get_float("LBKLAC_TAU_KL", 0.2),
    "eps_ood": _get_float("LBKLAC_EPS_OOD", 0.5),
    "ood_metric": _get_str("LBKLAC_OOD_METRIC", "both"),
    # Optional stage input
    "use_stage": _get_bool("LBKLAC_USE_STAGE", False),
    "stage_dim": _get_int("LBKLAC_STAGE_DIM", 2),
    # Device
    "device": _get_str("LBKLAC_DEVICE", "auto"),
}


def get_lbklac_config() -> Dict[str, Any]:
    cfg = dict(LBKLAC_DEFAULTS)
    raw = os.getenv("LBKLAC_CONFIG_JSON", "").strip()
    if raw:
        try:
            override = json.loads(raw)
            if isinstance(override, dict):
                cfg.update(override)
        except Exception:
            pass
    return cfg

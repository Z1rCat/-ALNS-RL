"""
Minimal, plug-in friendly RL/decision-making components designed for:
- expensive environment steps (ALNS),
- non-stationary / drifting distributions,
- small discrete action spaces (often {0,1}),
- existing ALNS-RL logging/evaluation chain (rl_trace.csv / rl_training.csv / baselines / plots / metrics).
"""

from .drcb import DriftRobustContextualBandit
from .lbklac import LBKLACAgent, LBKLACConfig
from .sb3_attention import HATConfig, HistoryAttentionWrapper, AttentionExtractor
from .protomem_ppo import ProtoMemConfig, ProtoMemInputWrapper, ProtoMemPolicy, ProtoMemPPO
from .ppo_new import build_model as build_ppo_new_model

__all__ = [
    "DriftRobustContextualBandit",
    "LBKLACAgent",
    "LBKLACConfig",
    "HATConfig",
    "HistoryAttentionWrapper",
    "AttentionExtractor",
    "ProtoMemConfig",
    "ProtoMemInputWrapper",
    "ProtoMemPolicy",
    "ProtoMemPPO",
    "build_ppo_new_model",
]

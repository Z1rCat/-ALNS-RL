import pandas as pd
import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
import numpy as np
import copy
import random
import os
import json
import csv
import atexit
import matplotlib.pyplot as plt
try:
    from stable_baselines3 import DQN, PPO, A2C, DDPG, HER, SAC, TD3
    from stable_baselines3.common.evaluation import evaluate_policy as sb3_evaluate_policy
    from stable_baselines3.common.vec_env import VecFrameStack
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from stable_baselines3.common.utils import explained_variance
    _SB3_AVAILABLE = True
except Exception:
    DQN = PPO = A2C = DDPG = HER = SAC = TD3 = None
    VecFrameStack = None
    sb3_evaluate_policy = None
    BaseCallback = None
    CallbackList = None
    explained_variance = None
    _SB3_AVAILABLE = False
try:
    from sb3_contrib import RecurrentPPO
    _SB3_CONTRIB_AVAILABLE = True
except Exception:
    RecurrentPPO = None
    _SB3_CONTRIB_AVAILABLE = False
try:
    from sb3_contrib import QRDQN as SB3_QRDQN
    _QRDQN_AVAILABLE = True
except Exception:
    SB3_QRDQN = None
    _QRDQN_AVAILABLE = False
try:
    from robust_rl.lbklac import LBKLACAgent, LBKLACConfig
    _LBKLAC_AVAILABLE = True
except Exception:
    LBKLACAgent = None
    LBKLACConfig = None
    _LBKLAC_AVAILABLE = False
try:
    from robust_rl.cql_dqn import DiscreteCQLAgent, CQLConfig
    _CQL_AVAILABLE = True
except Exception:
    DiscreteCQLAgent = None
    CQLConfig = None
    _CQL_AVAILABLE = False
try:
    from core import config as rl_config
except Exception:
    rl_config = None
try:
    from line_profiler import LineProfiler
except ImportError:
    class LineProfiler:
        def __call__(self, *args, **kwargs):
            def wrapper(func):
                return func
            return wrapper
        def print_stats(self):
            pass
import timeit
import time
from core import Intermodal_ALNS34959
import sys
from core import Dynamic_ALNS_RL34959
import cProfile
import pstats
import io
from pathlib import Path
from core import rl_logging
from collections import deque
# from Intermodal_ALNS34959 import parallel_read_excel, parallel_save_excel
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter(os.path.join('Training', 'Logs'))
# writer.add_scalar(tag, scalar_value, global_step=None, walltime=None)
# for epoch in range(100):
#     mAP = eval(model)
#     writer.add_scalar('mAP', mAP, epoch)
# writer.add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')
# writer.add_images(tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW')
import wrapt
if 'builtins' not in dir() or not hasattr(builtins, 'profile'):
    import builtins

def profile(func):
    def inner(*args, **kwargs):
        return func(*args, **kwargs)

    return inner


builtins.__dict__['profile'] = profile


def resolve_seed():
    seed_val = os.environ.get("RL_SEED", "").strip()
    if not seed_val:
        return None
    try:
        return int(seed_val)
    except ValueError:
        return None


def set_global_seed(seed_val):
    if seed_val is None:
        return
    random.seed(seed_val)
    np.random.seed(seed_val)
    try:
        import torch
        torch.manual_seed(seed_val)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_val)
    except Exception:
        pass


def get_stop_flag_path():
    return os.environ.get("STOP_FLAG_FILE", "34959.txt")


HAT_REWARD_EMA = 0.0
HAT_DRIFT_EMA = 0.0
HAT_ACTION_EMA = 0.0
HAT_STEP = 0
HAT_GATE_STATE = 0
HAT_BASE_CLIP = None
HAT_BASE_ENT = None
HAT_BASE_KL = None
_IMPL_REMOVAL_IDX = 0
_IMPL_INSERTION_IDX = 0
LSTM_CHAIN_LEN = 1
LSTM_CHAIN_STEP = 0
USE_LSTM = False
STAGE_IN_OBS = False
USE_AUGMENTED_OBS = False
ALGO_VERSION = (os.environ.get("RL_ALGO_VERSION", "v1") or "v1").strip().lower()
PPO_NEW_WINDOW_K = 1
STAGE_MODE = (os.environ.get("RL_STAGE_MODE", "train_eval") or "train_eval").strip().lower()
INIT_MODEL_PATH = (os.environ.get("RL_INIT_MODEL_PATH", "") or "").strip()
SAVE_MODEL_PATH = (os.environ.get("RL_SAVE_MODEL_PATH", "") or "").strip()
ORACLE_CTX_MODE = (os.environ.get("RL_ORACLE_CTX_MODE", "none") or "none").strip().lower()
try:
    ORACLE_GT_MEAN_NORM = float(os.environ.get("RL_ORACLE_GT_MEAN_NORM", "100.0"))
except Exception:
    ORACLE_GT_MEAN_NORM = 100.0
try:
    ORACLE_PHASE_CLASSES = int(os.environ.get("RL_ORACLE_PHASE_CLASSES", "0") or 0)
except Exception:
    ORACLE_PHASE_CLASSES = 0
PDI_GT_MEAN_LIST = []
PDI_PHASE_LIST = []
PDI_REWARD_LIST = []
_CHECKPOINT_SAVED_ON_STOP = False
_PHASE1_HIST_SAVED_KEYS = set()
_PHASE1_HIST_RECORDED_PATHS = set()
_PHASE1_HIST_LAST_TABLE_NUMBER = None
_PHASE1_HIST_LAST_COMPLETED_TABLES = 0
PHASE1_HIST_FIELDS = [
    "ts",
    "run_dir",
    "run_name",
    "distribution",
    "algorithm",
    "algo_version",
    "seed",
    "stage_mode",
    "table_number",
    "completed_train_tables",
    "checkpoint_name",
    "checkpoint_path",
    "trigger",
]


def _normalize_stage_mode(value):
    mode = str(value or "train_eval").strip().lower()
    if mode not in {"train_eval", "train_only", "eval_only"}:
        mode = "train_eval"
    return mode


def _phase1_hist_every_tables():
    try:
        return max(0, int(os.environ.get("RL_PHASE1_HIST_EVERY_TABLES", "0") or 0))
    except Exception:
        return 0


def _phase1_hist_enabled():
    return _normalize_stage_mode(os.environ.get("RL_STAGE_MODE", "train_eval")) == "train_only" and _phase1_hist_every_tables() > 0


def _phase1_hist_paths():
    run_dir = Path(rl_logging.get_run_dir())
    raw_ckpt_dir = str(os.environ.get("RL_PHASE1_HIST_CKPT_DIR", "") or "").strip()
    raw_manifest = str(os.environ.get("RL_PHASE1_HIST_MANIFEST", "") or "").strip()
    ckpt_dir = Path(raw_ckpt_dir).resolve() if raw_ckpt_dir else (run_dir / "post_stage" / "checkpoints" / "phase1_history").resolve()
    manifest_path = Path(raw_manifest).resolve() if raw_manifest else (run_dir / "post_stage" / "phase1_ckpt_manifest.csv").resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    return ckpt_dir, manifest_path


def _append_phase1_hist_manifest(checkpoint_path, table_number, completed_train_tables, trigger):
    _, manifest_path = _phase1_hist_paths()
    checkpoint_path = Path(str(checkpoint_path)).resolve()
    row = {
        "ts": time.time(),
        "run_dir": str(rl_logging.get_run_dir()),
        "run_name": Path(str(rl_logging.get_run_dir())).name,
        "distribution": str(os.environ.get("SCENARIO_NAME", "") or globals().get("SCENARIO_NAME", "") or ""),
        "algorithm": str(os.environ.get("RL_ALGORITHM", "") or globals().get("algorithm", "") or ""),
        "algo_version": str(os.environ.get("RL_ALGO_VERSION", "") or ""),
        "seed": str(os.environ.get("RL_SEED", "") or ""),
        "stage_mode": str(os.environ.get("RL_STAGE_MODE", "") or ""),
        "table_number": "" if table_number is None else int(table_number),
        "completed_train_tables": int(completed_train_tables) if completed_train_tables is not None else "",
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_path": str(checkpoint_path),
        "trigger": str(trigger),
    }
    file_exists = manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PHASE1_HIST_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def maybe_save_phase1_history_checkpoint(table_number, completed_train_tables, trigger="periodic"):
    global _PHASE1_HIST_LAST_TABLE_NUMBER, _PHASE1_HIST_LAST_COMPLETED_TABLES
    _PHASE1_HIST_LAST_TABLE_NUMBER = int(table_number)
    _PHASE1_HIST_LAST_COMPLETED_TABLES = int(completed_train_tables)
    if not _phase1_hist_enabled():
        return None
    every_tables = _phase1_hist_every_tables()
    if every_tables <= 0 or int(completed_train_tables) <= 0:
        return None
    if int(completed_train_tables) % int(every_tables) != 0:
        return None
    save_key = (str(trigger), int(table_number), int(completed_train_tables))
    if save_key in _PHASE1_HIST_SAVED_KEYS:
        return None
    model_obj = globals().get("model", None)
    if model_obj is None:
        return None
    ckpt_dir, _ = _phase1_hist_paths()
    out_path = ckpt_dir / f"theta_phase1_t{int(table_number):04d}_n{int(completed_train_tables):04d}.zip"
    try:
        model_obj.save(str(out_path))
        _append_phase1_hist_manifest(
            checkpoint_path=out_path,
            table_number=int(table_number),
            completed_train_tables=int(completed_train_tables),
            trigger=str(trigger),
        )
        _PHASE1_HIST_SAVED_KEYS.add(save_key)
        print(
            "[RL][PHASE1_HIST] "
            f"saved periodic checkpoint table={int(table_number)} "
            f"completed={int(completed_train_tables)} path={out_path}"
        )
        return str(out_path)
    except Exception as exc:
        print(f"[RL][PHASE1_HIST][WARN] failed to save periodic checkpoint: {exc}")
        return None


def _record_phase1_final_checkpoint(saved_path):
    if not _phase1_hist_enabled():
        return
    path_obj = Path(str(saved_path)).resolve()
    path_key = str(path_obj).lower()
    if path_key in _PHASE1_HIST_RECORDED_PATHS:
        return
    table_number = globals().get("_PHASE1_HIST_LAST_TABLE_NUMBER", None)
    completed_train_tables = globals().get("_PHASE1_HIST_LAST_COMPLETED_TABLES", 0)
    try:
        _append_phase1_hist_manifest(
            checkpoint_path=path_obj,
            table_number=table_number,
            completed_train_tables=completed_train_tables,
            trigger="final_save_model_path",
        )
        _PHASE1_HIST_RECORDED_PATHS.add(path_key)
    except Exception as exc:
        print(f"[RL][PHASE1_HIST][WARN] failed to record final checkpoint manifest: {exc}")


def _maybe_load_model_checkpoint(model_obj):
    path = str(globals().get("INIT_MODEL_PATH", "") or "").strip()
    if not path:
        return
    if not os.path.exists(path):
        print(f"[RL] init model checkpoint not found: {path}")
        return
    loaded = False
    if hasattr(model_obj, "set_parameters"):
        try:
            model_obj.set_parameters(path, exact_match=False, device="cpu")
            loaded = True
        except Exception:
            loaded = False
    if not loaded and hasattr(model_obj, "load"):
        try:
            cls = model_obj.__class__
            new_model = cls.load(path, env=model_obj.get_env(), device="cpu")
            if hasattr(new_model, "set_env"):
                new_model.set_env(model_obj.get_env())
            return new_model
        except Exception:
            pass
    if loaded:
        print(f"[RL] loaded model parameters from: {path}")
    else:
        print(f"[RL] failed to load checkpoint from: {path}")
    return model_obj


def _maybe_save_model_checkpoint(model_obj):
    path = str(globals().get("SAVE_MODEL_PATH", "") or "").strip()
    if not path:
        return
    try:
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        model_obj.save(path)
        _record_phase1_final_checkpoint(path)
        print(f"[RL] saved model checkpoint to: {path}")
    except Exception as exc:
        print(f"[RL] failed to save checkpoint to {path}: {exc}")


def _maybe_save_checkpoint_on_stop_exit():
    global _CHECKPOINT_SAVED_ON_STOP
    if _CHECKPOINT_SAVED_ON_STOP:
        return
    if str(globals().get("STAGE_MODE", "") or "").strip().lower() != "train_only":
        return
    path = str(globals().get("SAVE_MODEL_PATH", "") or "").strip()
    if not path:
        return
    model_obj = globals().get("model", None)
    if model_obj is None:
        return
    _maybe_save_model_checkpoint(model_obj)
    _CHECKPOINT_SAVED_ON_STOP = True


def _normalize_oracle_ctx_mode(mode_val):
    mode = str(mode_val or "none").strip().lower()
    return mode if mode in {"none", "phase", "mean"} else "none"


def _ema_update(prev, value, alpha):
    if prev is None:
        return float(value)
    return (1.0 - alpha) * float(prev) + alpha * float(value)


def _hat_is_active():
    try:
        return os.environ.get("RL_HAT", "0").strip() == "1" and algorithm in ("PPO", "A2C", "PPO_HAT_PDI")
    except Exception:
        return False


def _hat_gate():
    drift_hi = float(os.environ.get("HAT_GATE_DRIFT_HI", "0.2"))
    reward_low = float(os.environ.get("HAT_GATE_REWARD_LOW", "0.6"))
    return 1 if (HAT_DRIFT_EMA > drift_hi or HAT_REWARD_EMA < reward_low) else 0


def _hat_update_train_params():
    global HAT_BASE_CLIP, HAT_BASE_ENT, HAT_BASE_KL
    if not _hat_is_active() or implement == 1:
        return
    if model is None:
        return
    scale = float(os.environ.get("HAT_DRIFT_SCALE", "1.5"))
    max_scale = float(os.environ.get("HAT_DRIFT_MAX_SCALE", "3.0"))
    adj = min(max_scale, 1.0 + scale * HAT_DRIFT_EMA)
    if algorithm == "PPO":
        if HAT_BASE_CLIP is None:
            HAT_BASE_CLIP = getattr(model, "clip_range", 0.2)
        base_clip = HAT_BASE_CLIP if isinstance(HAT_BASE_CLIP, float) else 0.2
        model.clip_range = lambda _: base_clip * adj
        if HAT_BASE_KL is None:
            HAT_BASE_KL = getattr(model, "target_kl", None)
        if HAT_BASE_KL is not None:
            model.target_kl = HAT_BASE_KL * adj
    elif algorithm == "A2C":
        if HAT_BASE_ENT is None:
            HAT_BASE_ENT = getattr(model, "ent_coef", 0.0)
        model.ent_coef = float(HAT_BASE_ENT) * adj


def _hat_update_stats(reward, action):
    global HAT_REWARD_EMA, HAT_DRIFT_EMA, HAT_ACTION_EMA, HAT_STEP, HAT_GATE_STATE
    alpha = float(os.environ.get("HAT_EMA_ALPHA", "0.05"))
    HAT_STEP += 1
    HAT_REWARD_EMA = _ema_update(HAT_REWARD_EMA, reward, alpha)
    drift = abs(float(reward) - HAT_REWARD_EMA)
    HAT_DRIFT_EMA = _ema_update(HAT_DRIFT_EMA, drift, alpha)
    HAT_ACTION_EMA = _ema_update(HAT_ACTION_EMA, action, alpha)
    HAT_GATE_STATE = _hat_gate()


def _hat_predict_probs(model, obs):
    try:
        obs_tensor = model.policy.obs_to_tensor(obs)[0]
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.detach().cpu().numpy().squeeze()
        return probs
    except Exception:
        return None


def _predict_action1_prob(model, obs):
    probs = _hat_predict_probs(model, obs)
    try:
        if probs is None:
            return None
        probs_arr = np.asarray(probs, dtype=float).reshape(-1)
        if probs_arr.shape[0] < 2:
            return None
        return float(probs_arr[1])
    except Exception:
        return None


def _hat_select_action(model, obs):
    probs = _hat_predict_probs(model, obs)
    if probs is None or len(probs) < 2:
        action, _ = model.predict(obs, deterministic=True)
        try:
            return int(np.array(action).squeeze()), {"gate": 0, "p1": None, "tau": None}
        except Exception:
            return int(action), {"gate": 0, "p1": None, "tau": None}
    p1 = float(probs[1])
    tau_high = float(os.environ.get("HAT_TAU_HIGH", "0.55"))
    tau_low = float(os.environ.get("HAT_TAU_LOW", "0.35"))
    gate = _hat_gate()
    tau = tau_low if gate == 1 else tau_high
    action = 1 if p1 >= tau else 0
    return int(action), {"gate": gate, "p1": p1, "tau": tau}


def _hat_update_history_wrapper(env, action, reward):
    if not _hat_is_active():
        return False
    target = env
    seen = set()
    while target is not None and id(target) not in seen:
        seen.add(id(target))
        if hasattr(target, "_last_action") and hasattr(target, "_last_reward") and hasattr(target, "_onehot"):
            try:
                target._last_action = target._onehot(int(action))
                target._last_reward = float(reward)
                return True
            except Exception:
                return False
        target = getattr(target, "env", None)
    return False


def _flush_impl_reward_lists(env):
    global _IMPL_REMOVAL_IDX, _IMPL_INSERTION_IDX
    try:
        removal_rewards = getattr(Dynamic_ALNS_RL34959, "removal_reward_list_in_implementation", [])
        removal_states = getattr(Dynamic_ALNS_RL34959, "removal_state_list_in_implementation", [])
        removal_actions = getattr(Dynamic_ALNS_RL34959, "removal_action_list_in_implementation", [])
    except Exception:
        removal_rewards, removal_states, removal_actions = [], [], []

    while _IMPL_REMOVAL_IDX < len(removal_rewards):
        reward = removal_rewards[_IMPL_REMOVAL_IDX]
        state_row = removal_states[_IMPL_REMOVAL_IDX] if _IMPL_REMOVAL_IDX < len(removal_states) else {}
        if hasattr(state_row, "to_dict"):
            state_row = state_row.to_dict()
        if not isinstance(state_row, dict):
            state_row = {}
        action_val = removal_actions[_IMPL_REMOVAL_IDX] if _IMPL_REMOVAL_IDX < len(removal_actions) else state_row.get("action", "")
        state_row["action"] = action_val
        try:
            _decision_finalize(
                row=state_row,
                reward=reward,
                action=action_val,
                stage="receive_reward",
                source="RL",
                impl_stream="removal",
                impl_list_idx=int(_IMPL_REMOVAL_IDX),
            )
        except Exception:
            pass
        try:
            log_trace_from_row(state_row, "receive_reward", action=action_val, reward=reward, source="RL")
        except Exception:
            pass
        if _hat_is_active() and implement == 1:
            try:
                _hat_update_stats(float(reward), float(action_val))
                _hat_update_history_wrapper(env, action_val, reward)
            except Exception:
                pass
        try:
            Intermodal_ALNS34959.log_impl_reward(reward)
        except Exception:
            pass
        if algorithm == "DRCB" and os.environ.get("DRCB_IMPL_ONLINE_UPDATE", "1").strip() == "1":
            try:
                row_order = [
                    "uncertainty_index", "uncertainty_type", "request", "vehicle",
                    "delay_tolerance", "passed_terminals", "current_time", "action", "reward",
                ]
                row_series = pd.Series({k: state_row.get(k, "") for k in row_order})
                obs_upd = get_state(row_series)
                model._update(obs_upd, int(action_val), float(reward))
            except Exception:
                pass
        elif algorithm == "BE_CVAR_DQN" and os.environ.get("BE_IMPL_ONLINE_OBS", "1").strip() == "1":
            try:
                row_order = [
                    "uncertainty_index", "uncertainty_type", "request", "vehicle",
                    "delay_tolerance", "passed_terminals", "current_time", "action", "reward",
                ]
                row_series = pd.Series({k: state_row.get(k, "") for k in row_order})
                obs_upd = get_state(row_series)
                if hasattr(model, "observe_impl"):
                    model.observe_impl(obs_upd, int(action_val), float(reward))
            except Exception:
                pass
        _IMPL_REMOVAL_IDX += 1

    try:
        insertion_rewards = getattr(Dynamic_ALNS_RL34959, "insertion_reward_list_in_implementation", [])
        insertion_states = getattr(Dynamic_ALNS_RL34959, "insertion_state_list_in_implementation", [])
        insertion_actions = getattr(Dynamic_ALNS_RL34959, "insertion_action_list_in_implementation", [])
    except Exception:
        insertion_rewards, insertion_states, insertion_actions = [], [], []

    while _IMPL_INSERTION_IDX < len(insertion_rewards):
        reward = insertion_rewards[_IMPL_INSERTION_IDX]
        state_row = insertion_states[_IMPL_INSERTION_IDX] if _IMPL_INSERTION_IDX < len(insertion_states) else {}
        if hasattr(state_row, "to_dict"):
            state_row = state_row.to_dict()
        if not isinstance(state_row, dict):
            state_row = {}
        action_val = insertion_actions[_IMPL_INSERTION_IDX] if _IMPL_INSERTION_IDX < len(insertion_actions) else state_row.get("action", "")
        state_row["action"] = action_val
        try:
            _decision_finalize(
                row=state_row,
                reward=reward,
                action=action_val,
                stage="receive_reward",
                source="RL",
                impl_stream="insertion",
                impl_list_idx=int(_IMPL_INSERTION_IDX),
            )
        except Exception:
            pass
        try:
            log_trace_from_row(state_row, "receive_reward", action=action_val, reward=reward, source="RL")
        except Exception:
            pass
        if _hat_is_active() and implement == 1:
            try:
                _hat_update_stats(float(reward), float(action_val))
                _hat_update_history_wrapper(env, action_val, reward)
            except Exception:
                pass
        try:
            Intermodal_ALNS34959.log_impl_reward(reward)
        except Exception:
            pass
        if algorithm == "DRCB" and os.environ.get("DRCB_IMPL_ONLINE_UPDATE", "1").strip() == "1":
            try:
                row_order = [
                    "uncertainty_index", "uncertainty_type", "request", "vehicle",
                    "delay_tolerance", "passed_terminals", "current_time", "action", "reward",
                ]
                row_series = pd.Series({k: state_row.get(k, "") for k in row_order})
                obs_upd = get_state(row_series)
                model._update(obs_upd, int(action_val), float(reward))
            except Exception:
                pass
        elif algorithm == "BE_CVAR_DQN" and os.environ.get("BE_IMPL_ONLINE_OBS", "1").strip() == "1":
            try:
                row_order = [
                    "uncertainty_index", "uncertainty_type", "request", "vehicle",
                    "delay_tolerance", "passed_terminals", "current_time", "action", "reward",
                ]
                row_series = pd.Series({k: state_row.get(k, "") for k in row_order})
                obs_upd = get_state(row_series)
                if hasattr(model, "observe_impl"):
                    model.observe_impl(obs_upd, int(action_val), float(reward))
            except Exception:
                pass
        _IMPL_INSERTION_IDX += 1


def profile():

    lp = LineProfiler()

    @wrapt.decorator
    def wrapper(func, instance, args, kwargs):
        # global lp
        lp_wrapper = lp(func)
        res = lp_wrapper(*args, **kwargs)
        lp.print_stats()
        # lp.dump_stats(path + current_save + '/better_obj_record' + current_save + '.txt')
        return res

    return wrapper

# ===== 路径配置 =====
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DYNAMIC_PATH_MAP_SEEN = set()

WAIT_TIMEOUT_S = float(os.environ.get("RL_WAIT_TIMEOUT_S", "0") or 0)
WAIT_LOG_INTERVAL_S = float(os.environ.get("RL_WAIT_LOG_INTERVAL_S", "5") or 5)
WAIT_SLEEP_S = float(os.environ.get("RL_WAIT_SLEEP_S", "0.01") or 0.01)
LBKLAC_CUSTOM_LOGGING = False


def _wait_watchdog(stage, start_ts, last_log_ts, row_dict=None):
    now = time.time()
    if WAIT_LOG_INTERVAL_S > 0 and now - last_log_ts >= WAIT_LOG_INTERVAL_S:
        print(f"[RL] wait {stage} {now - start_ts:.1f}s")
        last_log_ts = now
    if WAIT_TIMEOUT_S > 0 and now - start_ts >= WAIT_TIMEOUT_S:
        try:
            log_trace_from_row(row_dict or {}, f"timeout_{stage}", source="RL")
        except Exception:
            pass
        if os.environ.get("RL_WAIT_ABORT", "0") == "1":
            return True, last_log_ts, now
        start_ts = now
    return False, last_log_ts, start_ts

def _env_int(name, default):
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


SEND_ACTION_MAX_WRITE_RETRY = max(1, _env_int("RL_SEND_ACTION_MAX_WRITE_RETRY", 1000))
SEND_ACTION_MAX_RESELECT = max(1, _env_int("RL_SEND_ACTION_MAX_RESELECT", 40))
SEND_ACTION_MAX_WAIT_SLOT_LOOPS = max(1, _env_int("RL_SEND_ACTION_MAX_WAIT_SLOT_LOOPS", 60000))
SEND_ACTION_MAX_CONFIRM_RETRY = max(1, _env_int("RL_SEND_ACTION_MAX_CONFIRM_RETRY", 1000))
SEND_ACTION_ERROR_LOG_EVERY = max(1, _env_int("RL_SEND_ACTION_ERROR_LOG_EVERY", 20))


def _resolve_dynamic_index(table_number):
    mode = (os.environ.get("RL_DYNAMIC_INDEX_MODE", "direct") or "direct").strip().lower()
    if mode not in {"direct", "mod"}:
        mode = "direct"
    file_count = _env_int("RL_DYNAMIC_FILE_COUNT", 0)
    table_base = _env_int("RL_DYNAMIC_TABLE_BASE", 0)
    mapped_idx = int(table_number)
    if mode == "mod":
        if file_count <= 0:
            mode = "direct"
            file_count = 0
        else:
            mapped_idx = (int(table_number) - int(table_base)) % int(file_count)
    return mode, int(mapped_idx), int(file_count), int(table_base)


def _resolve_path_map_csv():
    explicit = os.environ.get("RL_DYNAMIC_PATH_MAP_CSV", "").strip()
    if explicit:
        return Path(explicit)
    manifest_path = os.environ.get("RL_DYNAMIC_MANIFEST", "").strip()
    if manifest_path:
        manifest = Path(manifest_path)
        if manifest.suffix.lower() == ".json":
            return manifest.with_name("outer_path_map.csv")
        return manifest / "outer_path_map.csv"
    try:
        return Path(rl_logging.get_run_dir()) / "post_stage" / "outer_path_map.csv"
    except Exception:
        return None


def _append_outer_path_map(row):
    csv_path = _resolve_path_map_csv()
    if csv_path is None:
        return
    key = (
        row.get("request_number", ""),
        row.get("table_number", ""),
        row.get("mapped_idx", ""),
        row.get("mapped_file", ""),
        row.get("index_mode", ""),
    )
    if key in _DYNAMIC_PATH_MAP_SEEN:
        return
    _DYNAMIC_PATH_MAP_SEEN.add(key)
    fields = [
        "ts",
        "module",
        "iter_id",
        "phase",
        "stage_mode",
        "request_number",
        "table_number",
        "mapped_idx",
        "index_mode",
        "file_count",
        "table_base",
        "dynamic_root",
        "mapped_file",
        "exists",
        "read_ok",
        "strict_path",
    ]
    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        exists = csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in fields})
    except Exception:
        pass


def resolve_dynamic_data_path(request_number_in_R, table_number, duration_type, add_event_types):
    dynamic_root = os.environ.get("DYNAMIC_DATA_ROOT", "").strip()
    mode, mapped_idx, file_count, table_base = _resolve_dynamic_index(table_number)
    file_name = f"Intermodal_EGS_data_dynamic_congestion{mapped_idx}.xlsx"
    if dynamic_root:
        root = Path(dynamic_root)
        candidate_a = root / f"R{request_number_in_R}" / file_name
        candidate_b = root / file_name
        selected = candidate_a
        if candidate_a.exists():
            selected = candidate_a
        elif candidate_b.exists():
            selected = candidate_b
        strict_path = os.environ.get("RL_DYNAMIC_STRICT_PATH", "0").strip() == "1"
        selected_exists = selected.exists()
        _append_outer_path_map(
            {
                "ts": rl_logging.now_ts(),
                "module": "dynamic_RL34959",
                "iter_id": os.environ.get("RL_OUTER_ITER_ID", ""),
                "phase": "implement" if globals().get("implement", 0) == 1 else "train",
                "stage_mode": str(globals().get("STAGE_MODE", os.environ.get("RL_STAGE_MODE", "train_eval"))),
                "request_number": int(request_number_in_R),
                "table_number": int(table_number),
                "mapped_idx": int(mapped_idx),
                "index_mode": mode,
                "file_count": int(file_count),
                "table_base": int(table_base),
                "dynamic_root": str(root),
                "mapped_file": str(selected),
                "exists": int(selected_exists),
                "read_ok": int(selected_exists),
                "strict_path": int(strict_path),
            }
        )
        if strict_path and not selected_exists:
            raise FileNotFoundError(
                "[OUTER][ERROR] PATH_CHECK failed: "
                f"table={table_number} mapped_idx={mapped_idx} mode={mode} "
                f"request=R{request_number_in_R} expected_file={selected}"
            )
        return str(selected)
    base_dir = os.path.join(
        ROOT_DIR,
        "Uncertainties Dynamic planning under unexpected events",
        f"plot_distribution_targetInstances_disruption_{duration_type}_not_time_dependent",
    )
    if add_event_types == 1:
        base_dir = base_dir + "_event_types"
    return os.path.join(base_dir, f"R{request_number_in_R}", f"Intermodal_EGS_data_dynamic_congestion{table_number}.xlsx")

# ===== 日志工具 =====
global_step = 0
MIN_STEPS = 100
MAX_STEPS = 20000
SLIDING_WINDOW = 30
TARGET_REWARD = 0.5
recent_rewards = deque(maxlen=SLIDING_WINDOW)
CURRICULUM_REWARD_THRESHOLD = 0.85
CURRICULUM_SUCCESS_REQUIRED = 3
curriculum_converged = 0
curriculum_last_avg_reward = ""
SCENARIO_NAME = os.environ.get("SCENARIO_NAME", "")
# Baseline replay may call env.step without going through main(), so keep safe defaults.
algorithm = (os.environ.get("RL_ALGORITHM", "DQN") or "DQN").strip().upper()
mode = (os.environ.get("RL_MODE", "barge") or "barge").strip().lower()

TRACE_FIELDS = [
    "ts", "phase", "stage", "uncertainty_index", "request", "vehicle",
    "table_number", "table_id", "dynamic_t_begin", "duration_type", "gt_mean", "phase_label", "oracle_ctx_mode",
    "delay_tolerance", "severity", "passed_terminals", "current_time",
    "action", "reward", "action_meaning", "feasible", "source", "p_action1",
    # Drift/robustness interpretability fields (optional; safe to leave empty)
    "algo", "regime_id", "context_id", "drift_score",
    # LB-KLAC diagnostics
    "belief_smooth_penalty", "value_residual", "delta_t", "policy_kl", "action_prob", "entropy",
    "bootstrap", "trust_region_scaled", "trust_region_scale",
    # MoE (HAT+MoE) diagnostics (rolling means; safe to leave empty)
    "gate_prob_0_mean", "gate_prob_1_mean", "gate_entropy_mean",
    "expert0_action1_prob_mean", "expert1_action1_prob_mean",
    "expert_selected_ratio",
    "moe_div_mean",
    # PDI diagnostics
    "pdi_future_mean", "pdi_pref_mean", "pdi_hard_mean",
    "pdi_fail0_mean", "pdi_fail1_mean",
    # ProtoMem diagnostics
    "pm_route_entropy_mean", "pm_route_confidence_mean",
    "pm_route_confidence_p25", "pm_route_confidence_p75",
    "pm_proto_top1_entropy", "pm_param_norm_m",
]

current_gt_mean = ""
current_phase_label = ""
current_stage_label = ""
_last_phase_label_for_drift = None

TRAIN_FIELDS = [
    "ts", "phase", "step_idx", "reward", "avg_reward", "std_reward",
    "rolling_avg", "recent_count",
    "training_time", "implementation_time",
    # Optional training diagnostics for drift-robust algorithms
    "algo", "regime_id", "context_id", "drift_score",
    # Optional LB-KLAC diagnostics
    "loss_pi", "loss_v", "loss_kl", "loss_entropy",
    "policy_kl", "delta_t", "belief_smooth_penalty", "value_residual",
    "bootstrap", "trust_region_scaled", "trust_region_scale",
    # LSTM / PPO diagnostics
    "value_pred_mean", "value_pred_std",
    "advantage_mean", "advantage_std",
    "explained_variance", "policy_entropy", "lstm_hidden_norm",
    # Generic diagnostics for PPO_NEW v5+/v6+
    "action1_rate", "p_action1", "reward_given_action0", "reward_given_action1",
    # V5.1 AB-PPO diagnostics
    "abppo_f0", "abppo_f1", "abppo_w0", "abppo_w1",
    # V5.2 Q-critic diagnostics
    "qcritic_q0", "qcritic_q1", "qcritic_q_taken",
    "qcritic_adv_mean", "qcritic_adv_std",
    "qcritic_adv0_mean", "qcritic_adv0_std",
    "qcritic_adv1_mean", "qcritic_adv1_std",
    # V5.3 Aux-Weak diagnostics
    "aux_enabled", "aux_source", "aux_loss", "aux_y_mean", "aux_yhat_mean", "aux_acc", "aux_auc",
    # V6 CVaR-tail diagnostics
    "cvar_alpha", "cvar_beta", "cvar_quantile", "cvar_tail_frac",
    "cvar_weight_mean", "cvar_tail_reward", "cvar_non_tail_reward",
    # CQL diagnostics
    "cql_alpha", "cql_temp", "cql_updates", "cql_td_loss", "cql_cql_loss",
    "cql_q_mean", "cql_q_std", "cql_q_max", "cql_q_taken", "cql_lse_q", "cql_ood_q_gap",
    # V6.3 CaDM-adapted auxiliary dynamics diagnostics
    "cadm_enabled", "cadm_source", "cadm_transitions", "cadm_aux_batches",
    "cadm_aux_loss", "cadm_nextsev_loss", "cadm_reward_loss",
    "cadm_nextsev_mae", "cadm_reward_mae",
    # V7.3+/TCR diagnostics
    "tcr_enabled", "tcr_rollout_groups", "tcr_trigger_events", "tcr_triggered_groups",
    "tcr_trigger_group_ids", "tcr_new_samples", "tcr_buffer_size",
    "tcr_action1_rate_trigger_mean", "tcr_reward_gap_trigger_mean",
    "tcr_aux_loss", "tcr_aux_applied_batches", "tcr_teacher_mode",
    # V8-A diagnostics
    "v8_enabled", "v8_trigger_quality_mean", "v8_trigger_gap_score_mean",
    "v8_selected_weight_mean", "v8_aux_boost_loss", "v8_aux_kl_guard_loss",
    "v8_aux_weight_mean", "v8_aux_scale",
    # V9-A diagnostics
    "v9_enabled", "v9_gap_score_ema", "v9_kappa", "v9_kappa_mean", "v9_kappa_base",
    # V8-A.2/V9-A.2 diagnostics
    "v8a2_enabled", "v8a2_trigger_shortage_mean", "v8a2_trigger_low_ratio_mean",
    "v8a2_generated_samples", "v8a2_target_action0_samples", "v8a2_target_action1_samples",
    "v8a2_generated_target", "v8a2_generated_shortfall",
    "v8a2_target_action_mode", "v8a2_aux_ce_loss", "v8a2_aux_kl_guard_loss",
    "v8a2_aux_weight_mean", "v8a2_aux_scale",
    "v9a2_enabled", "v9a2_shortage_ema", "v9a2_kappa", "v9a2_kappa_mean", "v9a2_kappa_base",
    # Generic Phase2 visual markers
    "phase2_active", "phase2_stage", "phase2_triggered_groups",
    "phase2_new_samples", "phase2_generated_samples", "phase2_generated_target", "phase2_generated_shortfall", "phase2_target_action_mode",
    "phase2_aux_applied_batches", "phase2_aux_ce_loss", "phase2_aux_kl_loss", "phase2_kappa",
    # PDI training diagnostics
    "pdi_future_loss", "pdi_teach_loss", "pdi_actfail_loss",
    # ProtoMem training diagnostics
    "pm_loss_sparse", "pm_loss_div", "pm_loss_stable", "pm_loss_aux",
    "pm_grad_norm_m", "pm_route_entropy_mean", "pm_route_confidence_mean",
    "pm_route_confidence_p25", "pm_route_confidence_p75",
    "pm_proto_top1_entropy", "pm_proto_top1_entropy_over_phase", "pm_param_norm_m",
]

DECISION_FIELDS = [
    "ts_decision", "ts_reward",
    "run_id", "decision_seq", "decision_id",
    "phase", "stage_mode", "stage",
    "uncertainty_index", "request", "vehicle", "pair_index",
    "table_number", "dynamic_t_begin", "duration_type",
    "gt_mean", "phase_label", "severity",
    "stage_family", "action_meaning", "semantic_action",
    "action", "reward", "p_action1",
    "matched", "impl_stream", "impl_list_idx",
    "source", "h_index",
]

decision_seq = 0
_decision_open_by_id = {}
_decision_open_by_pair_key = {}
_decision_open_queue_by_signature = {}
_decision_open_order = deque()
_decision_h_ids = []
_decision_h_rows = []
_decision_h_flush_interval = max(1, int(os.environ.get("RL_DECISION_H_FLUSH_INTERVAL", "200") or 200))
_decision_h_last_flush_size = 0

def _drift_snapshot():
    """
    Best-effort drift/context snapshot, designed to be safe across:
    - SB3 algorithms (no changes required)
    - Baseline replay (fields will exist but may be empty)
    - New robust algorithms (can populate regime/context more richly)
    """
    global _last_phase_label_for_drift
    phase_label = current_phase_label
    drift = 0.0
    try:
        if _last_phase_label_for_drift is not None and str(phase_label) != str(_last_phase_label_for_drift):
            drift = 1.0
    except Exception:
        drift = 0.0
    _last_phase_label_for_drift = phase_label
    algo_name = globals().get("algorithm", "")
    regime_id = phase_label
    context_id = phase_label
    try:
        gt = float(current_gt_mean) if current_gt_mean != "" else None
        if gt is not None and phase_label not in (None, ""):
            context_id = f"{phase_label}|gt_mean={gt:g}"
    except Exception:
        pass
    return {
        "algo": algo_name,
        "regime_id": regime_id,
        "context_id": context_id,
        "drift_score": drift,
    }

def log_trace_from_row(row, stage, action=None, reward=None, feasible="", source="RL", extra=None):
    try:
        action_val = action if action is not None else row.get("action", "")
        action_meaning = ""
        try:
            if action_val in [-10000000, -10000000.0, ""]:
                action_meaning = ""
            else:
                a_int = int(action_val)
                if "insert" in stage:
                    action_meaning = "接受插入" if a_int == 0 else "拒绝插入"
                else:
                    action_meaning = "等待/保持" if a_int == 0 else "重新规划"
        except Exception:
            action_meaning = ""
        payload = {
            "ts": rl_logging.now_ts(),
            "phase": "implement" if implement == 1 else "train",
            "stage": stage,
            "uncertainty_index": row.get("uncertainty_index", ""),
            "request": row.get("request", ""),
            "vehicle": row.get("vehicle", ""),
            "table_number": getattr(Dynamic_ALNS_RL34959, "table_number", ""),
            "table_id": getattr(Dynamic_ALNS_RL34959, "table_number", ""),
            "dynamic_t_begin": getattr(Intermodal_ALNS34959, "dynamic_t_begin", ""),
            "duration_type": getattr(Intermodal_ALNS34959, "duration_type", ""),
            "gt_mean": current_gt_mean,
            "phase_label": current_phase_label,
            "oracle_ctx_mode": _normalize_oracle_ctx_mode(globals().get("ORACLE_CTX_MODE", "none")),
            "delay_tolerance": row.get("delay_tolerance", ""),
            "severity": globals().get("severity_level", ""),
            "passed_terminals": row.get("passed_terminals", ""),
            "current_time": row.get("current_time", ""),
            "action": action_val,
            "reward": reward if reward is not None else row.get("reward", ""),
            "action_meaning": action_meaning,
            "feasible": feasible,
            "source": source,
            "p_action1": "",
        }
        payload.update(_drift_snapshot())
        # Best-effort MoE stats from policy (no impact on non-MoE runs).
        try:
            if model is not None and hasattr(model, "policy") and hasattr(model.policy, "get_moe_log"):
                payload.update(model.policy.get_moe_log())
        except Exception:
            pass
        # Best-effort PDI stats from policy
        try:
            if model is not None and hasattr(model, "policy") and hasattr(model.policy, "get_pdi_log"):
                payload.update(model.policy.get_pdi_log())
        except Exception:
            pass
        # Best-effort ProtoMem stats from policy
        try:
            if model is not None and hasattr(model, "policy") and hasattr(model.policy, "get_protomem_log"):
                payload.update(model.policy.get_protomem_log())
        except Exception:
            pass
        if extra:
            payload.update(extra)
        rl_logging.append_row("rl_trace.csv", TRACE_FIELDS, payload)
    except Exception as e:
        print("log_trace_from_row error", e)

def log_training_row(phase, step_idx="", reward=None, avg_reward=None, std_reward=None,
                     rolling_avg=None, recent_count=None, training_time=None, implementation_time=None, extra=None):
    try:
        payload = {
            "ts": rl_logging.now_ts(),
            "phase": phase,
            "step_idx": step_idx,
            "reward": reward if reward is not None else "",
            "avg_reward": avg_reward if avg_reward is not None else "",
            "std_reward": std_reward if std_reward is not None else "",
            "rolling_avg": rolling_avg if rolling_avg is not None else "",
            "recent_count": recent_count if recent_count is not None else "",
            "training_time": training_time if training_time is not None else "",
            "implementation_time": implementation_time if implementation_time is not None else "",
        }
        payload.update(_drift_snapshot())
        if extra:
            payload.update(extra)
        rl_logging.append_row("rl_training.csv", TRAIN_FIELDS, payload)
    except Exception as e:
        print("log_training_row error", e)


def _maybe_console_phase2(extra, phase: str = "train"):
    if not isinstance(extra, dict):
        return
    if str(os.environ.get("RL_PHASE2_CONSOLE", "1")).strip() != "1":
        return
    try:
        active = int(float(extra.get("phase2_active", 0) or 0))
    except Exception:
        active = 0
    if active <= 0:
        return
    stage = str(extra.get("phase2_stage", "") or "")
    groups = int(float(extra.get("phase2_triggered_groups", 0) or 0))
    new_samples = int(float(extra.get("phase2_new_samples", 0) or 0))
    gen_samples = int(float(extra.get("phase2_generated_samples", 0) or 0))
    gen_target = int(float(extra.get("phase2_generated_target", 0) or 0))
    gen_shortfall = int(float(extra.get("phase2_generated_shortfall", 0) or 0))
    aux_batches = int(float(extra.get("phase2_aux_applied_batches", 0) or 0))
    kappa = extra.get("phase2_kappa", "")
    target_mode = str(extra.get("phase2_target_action_mode", "") or "")
    use_color = str(os.environ.get("RL_PHASE2_COLOR", "1")).strip() == "1"
    prefix = f"[PHASE2][{str(phase or '').upper()}]"
    if use_color:
        prefix = f"\033[95m{prefix}\033[0m"
    print(
        f"{prefix} stage={stage} groups={groups} new={new_samples} "
        f"generated={gen_samples}/{gen_target if gen_target > 0 else 'na'} "
        f"shortfall={gen_shortfall} aux_batches={aux_batches} "
        f"target_mode={target_mode if target_mode else 'na'} kappa={kappa if kappa != '' else 'na'}"
    )


def _decision_current_phase():
    return "implement" if int(globals().get("implement", 0) or 0) == 1 else "train"


def _decision_run_id():
    try:
        run_dir = rl_logging.get_run_dir()
        if run_dir is not None:
            return Path(run_dir).name
    except Exception:
        pass
    return str(os.environ.get("RL_RUN_ID", "") or "")


def _decision_to_dict(row):
    if isinstance(row, dict):
        return dict(row)
    try:
        if hasattr(row, "to_dict"):
            data = row.to_dict()
            if isinstance(data, dict):
                return dict(data)
    except Exception:
        pass
    return {}


def _decision_to_action_value(value):
    try:
        arr = np.asarray(value).reshape(-1)
        if arr.size == 0:
            return ""
        return int(arr[0])
    except Exception:
        return value if value is not None else ""


def _decision_to_float_or_empty(value):
    if value in ("", None):
        return ""
    try:
        arr = np.asarray(value).reshape(-1)
        if arr.size == 0:
            return ""
        return float(arr[0])
    except Exception:
        return ""


def _decision_capture_obs(obs_snapshot):
    if obs_snapshot is None:
        return None
    try:
        arr = np.asarray(obs_snapshot, dtype=np.float32)
        if arr.size == 0:
            return None
        return arr.reshape(-1).copy()
    except Exception:
        return None


def _decision_get_p_action1(trace_extra):
    if not isinstance(trace_extra, dict):
        return ""
    return _decision_to_float_or_empty(trace_extra.get("p_action1", ""))


def _decision_stage_family(impl_stream=""):
    stream = str(impl_stream or "").strip().lower()
    if stream in {"removal", "insertion"}:
        return stream
    label = str(globals().get("current_stage_label", "") or "").strip().lower()
    if "insert" in label:
        return "insertion"
    if "remove" in label:
        return "removal"
    return ""


def _decision_semantic_action(stage_family, action_val):
    try:
        a_int = int(action_val)
    except Exception:
        return ""
    family = str(stage_family or "").strip().lower()
    if family == "removal":
        return "wait" if a_int == 0 else "remove"
    if family == "insertion":
        return "insert" if a_int == 0 else "non_insert"
    if a_int == 0:
        return "action0"
    if a_int == 1:
        return "action1"
    return ""


def _decision_action_meaning(stage_family, action_val):
    semantic = _decision_semantic_action(stage_family, action_val)
    mapping = {
        "wait": "wait",
        "remove": "remove",
        "insert": "insert",
        "non_insert": "non_insert",
        "action0": "action0",
        "action1": "action1",
    }
    return mapping.get(semantic, "")


def _decision_signature(row_dict, phase):
    return "|".join(
        [
            str(phase),
            str(row_dict.get("uncertainty_index", "")),
            str(row_dict.get("vehicle", "")),
            str(row_dict.get("request", "")),
            str(row_dict.get("uncertainty_type", "")),
        ]
    )


def _decision_pair_key(row_dict, phase):
    return _decision_signature(row_dict, phase) + "|pair=" + str(row_dict.get("pair_index", ""))


def _decision_next_id(phase):
    global decision_seq
    decision_seq += 1
    seq = int(decision_seq)
    return seq, f"{_decision_run_id()}|{phase}|{seq}"


def _decision_queue_append(signature, decision_id):
    queue = _decision_open_queue_by_signature.get(signature)
    if queue is None:
        queue = deque()
        _decision_open_queue_by_signature[signature] = queue
    queue.append(decision_id)


def _decision_remove_open(decision_id):
    pending = _decision_open_by_id.pop(decision_id, None)
    if not isinstance(pending, dict):
        return
    pair_key = str(pending.get("pair_key", ""))
    if pair_key and _decision_open_by_pair_key.get(pair_key) == decision_id:
        _decision_open_by_pair_key.pop(pair_key, None)
    signature = str(pending.get("signature", ""))
    if signature in _decision_open_queue_by_signature:
        queue = _decision_open_queue_by_signature.get(signature)
        if queue is not None:
            new_queue = deque([item for item in queue if item != decision_id])
            if len(new_queue) > 0:
                _decision_open_queue_by_signature[signature] = new_queue
            else:
                _decision_open_queue_by_signature.pop(signature, None)
    try:
        _decision_open_order.remove(decision_id)
    except Exception:
        pass


def _decision_note_send(row, action, obs_snapshot=None, trace_extra=None, source="RL"):
    row_dict = _decision_to_dict(row)
    phase = _decision_current_phase()
    signature = _decision_signature(row_dict, phase)
    pair_key = _decision_pair_key(row_dict, phase)
    stage_family = _decision_stage_family("")
    action_value = _decision_to_action_value(action)
    severity = globals().get("severity_level", "")
    action_meaning = _decision_action_meaning(stage_family, action_value)
    semantic_action = _decision_semantic_action(stage_family, action_value)
    existing_id = _decision_open_by_pair_key.get(pair_key)
    if existing_id in _decision_open_by_id:
        pending = _decision_open_by_id.get(existing_id, {})
        if pending.get("obs_snapshot") is None:
            pending["obs_snapshot"] = _decision_capture_obs(obs_snapshot)
        p_action1 = _decision_get_p_action1(trace_extra)
        if p_action1 != "":
            pending["p_action1"] = p_action1
        if not pending.get("stage_family"):
            pending["stage_family"] = stage_family
        if pending.get("severity", "") in ("", None):
            pending["severity"] = severity
        if not pending.get("action_meaning"):
            pending["action_meaning"] = action_meaning
        if not pending.get("semantic_action"):
            pending["semantic_action"] = semantic_action
        _decision_open_by_id[existing_id] = pending
        return existing_id
    seq, decision_id = _decision_next_id(phase)
    pending = {
        "run_id": _decision_run_id(),
        "decision_seq": seq,
        "decision_id": decision_id,
        "phase": phase,
        "stage_mode": str(globals().get("STAGE_MODE", "")),
        "ts_decision": rl_logging.now_ts(),
        "action": action_value,
        "p_action1": _decision_get_p_action1(trace_extra),
        "severity": severity,
        "stage_family": stage_family,
        "action_meaning": action_meaning,
        "semantic_action": semantic_action,
        "row_dict": row_dict,
        "signature": signature,
        "pair_key": pair_key,
        "source": source,
        "obs_snapshot": _decision_capture_obs(obs_snapshot),
    }
    _decision_open_by_id[decision_id] = pending
    _decision_open_by_pair_key[pair_key] = decision_id
    _decision_queue_append(signature, decision_id)
    _decision_open_order.append(decision_id)
    return decision_id


def _decision_pick_pending_id(row_dict, phase):
    pair_key = _decision_pair_key(row_dict, phase)
    decision_id = _decision_open_by_pair_key.get(pair_key, "")
    if decision_id in _decision_open_by_id:
        return decision_id
    signature = _decision_signature(row_dict, phase)
    queue = _decision_open_queue_by_signature.get(signature)
    if queue is not None:
        while len(queue) > 0:
            candidate = queue[0]
            if candidate in _decision_open_by_id:
                return candidate
            queue.popleft()
        _decision_open_queue_by_signature.pop(signature, None)
    for candidate in list(_decision_open_order):
        pending = _decision_open_by_id.get(candidate)
        if isinstance(pending, dict) and str(pending.get("phase", "")) == str(phase):
            return candidate
    return ""


def _decision_extract_h64(obs_snapshot):
    if obs_snapshot is None:
        return None
    policy = getattr(globals().get("model", None), "policy", None)
    extractor = getattr(policy, "features_extractor", None)
    if extractor is None:
        return None
    if extractor.__class__.__name__ != "StackedContextExtractor":
        return None
    try:
        arr = np.asarray(obs_snapshot, dtype=np.float32).reshape(1, -1)
    except Exception:
        return None
    try:
        expected_dim = int(getattr(extractor, "obs_dim", 0) or 0)
    except Exception:
        expected_dim = 0
    if expected_dim > 0 and int(arr.shape[1]) != expected_dim:
        return None
    try:
        import torch
    except Exception:
        return None
    param = None
    try:
        param = next(extractor.parameters())
    except Exception:
        param = None
    device = param.device if param is not None else "cpu"
    was_training = bool(getattr(extractor, "training", False))
    try:
        extractor.eval()
        with torch.no_grad():
            obs_tensor = torch.as_tensor(arr, dtype=torch.float32, device=device)
            h = extractor(obs_tensor)
            h_arr = np.asarray(h.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
    except Exception:
        h_arr = None
    finally:
        try:
            extractor.train(was_training)
        except Exception:
            pass
    if h_arr is None or h_arr.size != 64:
        return None
    return h_arr


def _decision_flush_h_dump():
    global _decision_h_last_flush_size
    try:
        run_dir = rl_logging.get_run_dir()
        if run_dir is None:
            return
        out_path = Path(run_dir) / "h_dump.npz"
        decision_ids = np.asarray(_decision_h_ids, dtype=object)
        if _decision_h_rows:
            h_mat = np.vstack(_decision_h_rows).astype(np.float32, copy=False)
        else:
            h_mat = np.zeros((0, 64), dtype=np.float32)
        np.savez(out_path, decision_id=decision_ids, h=h_mat)
        _decision_h_last_flush_size = len(_decision_h_ids)
    except Exception:
        pass


def _decision_append_h(decision_id, h_vec):
    if h_vec is None:
        return -1
    try:
        h_arr = np.asarray(h_vec, dtype=np.float32).reshape(-1)
    except Exception:
        return -1
    if h_arr.size != 64:
        return -1
    h_index = len(_decision_h_ids)
    _decision_h_ids.append(str(decision_id))
    _decision_h_rows.append(h_arr)
    if (len(_decision_h_ids) - int(_decision_h_last_flush_size)) >= int(_decision_h_flush_interval):
        _decision_flush_h_dump()
    return int(h_index)


def _decision_finalize(row, reward, action=None, stage="receive_reward", source="RL", impl_stream="", impl_list_idx=""):
    row_dict = _decision_to_dict(row)
    phase = _decision_current_phase()
    decision_id = _decision_pick_pending_id(row_dict, phase)
    pending = _decision_open_by_id.get(decision_id) if decision_id else None
    matched = 1 if isinstance(pending, dict) else 0
    stage_family = _decision_stage_family(impl_stream=impl_stream if phase == "implement" else "")
    severity = globals().get("severity_level", "")

    if matched == 0:
        seq, decision_id = _decision_next_id(phase)
        action_value = _decision_to_action_value(action if action is not None else row_dict.get("action", ""))
        pending = {
            "run_id": _decision_run_id(),
            "decision_seq": seq,
            "decision_id": decision_id,
            "phase": phase,
            "stage_mode": str(globals().get("STAGE_MODE", "")),
            "ts_decision": rl_logging.now_ts(),
            "action": action_value,
            "p_action1": "",
            "severity": severity,
            "stage_family": stage_family,
            "action_meaning": _decision_action_meaning(stage_family, action_value),
            "semantic_action": _decision_semantic_action(stage_family, action_value),
            "row_dict": row_dict,
            "signature": _decision_signature(row_dict, phase),
            "pair_key": _decision_pair_key(row_dict, phase),
            "source": source,
            "obs_snapshot": None,
        }

    action_val = _decision_to_action_value(
        action if action is not None else pending.get("action", row_dict.get("action", ""))
    )
    p_action1 = pending.get("p_action1", "")
    reward_val = _decision_to_float_or_empty(reward)
    payload_stage_family = pending.get("stage_family", stage_family) or stage_family
    payload_severity = pending.get("severity", severity)
    action_meaning = pending.get("action_meaning", _decision_action_meaning(payload_stage_family, action_val))
    semantic_action = pending.get("semantic_action", _decision_semantic_action(payload_stage_family, action_val))

    h_index = -1
    if matched == 1:
        h_vec = _decision_extract_h64(pending.get("obs_snapshot"))
        h_index = _decision_append_h(decision_id, h_vec)

    payload = {
        "ts_decision": pending.get("ts_decision", rl_logging.now_ts()),
        "ts_reward": rl_logging.now_ts(),
        "run_id": pending.get("run_id", _decision_run_id()),
        "decision_seq": pending.get("decision_seq", ""),
        "decision_id": decision_id,
        "phase": phase,
        "stage_mode": pending.get("stage_mode", str(globals().get("STAGE_MODE", ""))),
        "stage": stage,
        "uncertainty_index": row_dict.get("uncertainty_index", pending.get("row_dict", {}).get("uncertainty_index", "")),
        "request": row_dict.get("request", pending.get("row_dict", {}).get("request", "")),
        "vehicle": row_dict.get("vehicle", pending.get("row_dict", {}).get("vehicle", "")),
        "pair_index": row_dict.get("pair_index", pending.get("row_dict", {}).get("pair_index", "")),
        "table_number": getattr(Dynamic_ALNS_RL34959, "table_number", ""),
        "dynamic_t_begin": getattr(Intermodal_ALNS34959, "dynamic_t_begin", ""),
        "duration_type": getattr(Intermodal_ALNS34959, "duration_type", ""),
        "gt_mean": current_gt_mean,
        "phase_label": str(current_phase_label or ""),
        "severity": payload_severity,
        "stage_family": payload_stage_family,
        "action_meaning": action_meaning,
        "semantic_action": semantic_action,
        "action": action_val,
        "reward": reward_val,
        "p_action1": p_action1,
        "matched": int(matched),
        "impl_stream": impl_stream if phase == "implement" else "",
        "impl_list_idx": impl_list_idx if phase == "implement" else "",
        "source": source,
        "h_index": h_index,
    }
    try:
        rl_logging.append_row("rl_decision.csv", DECISION_FIELDS, payload)
    except Exception:
        pass
    if matched == 1:
        _decision_remove_open(decision_id)


def _decision_flush_unmatched_rows():
    for decision_id in list(_decision_open_order):
        pending = _decision_open_by_id.get(decision_id)
        if not isinstance(pending, dict):
            continue
        row_dict = pending.get("row_dict", {}) if isinstance(pending.get("row_dict", {}), dict) else {}
        phase = str(pending.get("phase", _decision_current_phase()))
        payload = {
            "ts_decision": pending.get("ts_decision", rl_logging.now_ts()),
            "ts_reward": rl_logging.now_ts(),
            "run_id": pending.get("run_id", _decision_run_id()),
            "decision_seq": pending.get("decision_seq", ""),
            "decision_id": decision_id,
            "phase": phase,
            "stage_mode": pending.get("stage_mode", str(globals().get("STAGE_MODE", ""))),
            "stage": "unmatched",
            "uncertainty_index": row_dict.get("uncertainty_index", ""),
            "request": row_dict.get("request", ""),
            "vehicle": row_dict.get("vehicle", ""),
            "pair_index": row_dict.get("pair_index", ""),
            "table_number": getattr(Dynamic_ALNS_RL34959, "table_number", ""),
            "dynamic_t_begin": getattr(Intermodal_ALNS34959, "dynamic_t_begin", ""),
            "duration_type": getattr(Intermodal_ALNS34959, "duration_type", ""),
            "gt_mean": current_gt_mean,
            "phase_label": str(current_phase_label or ""),
            "severity": pending.get("severity", globals().get("severity_level", "")),
            "stage_family": pending.get("stage_family", ""),
            "action_meaning": pending.get("action_meaning", ""),
            "semantic_action": pending.get("semantic_action", ""),
            "action": pending.get("action", ""),
            "reward": "",
            "p_action1": pending.get("p_action1", ""),
            "matched": 0,
            "impl_stream": "",
            "impl_list_idx": "",
            "source": pending.get("source", "RL"),
            "h_index": -1,
        }
        try:
            rl_logging.append_row("rl_decision.csv", DECISION_FIELDS, payload)
        except Exception:
            pass
        _decision_remove_open(decision_id)


def _decision_finalize_on_exit():
    try:
        _decision_flush_unmatched_rows()
    except Exception:
        pass
    try:
        _decision_flush_h_dump()
    except Exception:
        pass


atexit.register(_decision_finalize_on_exit)


class LstmStatsCallback(BaseCallback):
    """
    Collect batch-level stats from RecurrentPPO rollout buffer and log to rl_training.csv.
    """

    def __init__(self):
        if BaseCallback is None:
            raise ImportError("stable_baselines3 is required for LstmStatsCallback.")
        super().__init__()

    def _on_rollout_end(self) -> None:
        try:
            rb = getattr(self.model, "rollout_buffer", None)
            if rb is None:
                return
            values = getattr(rb, "values", None)
            advantages = getattr(rb, "advantages", None)
            returns = getattr(rb, "returns", None)
            if values is None or advantages is None:
                return
            v = np.array(values).astype(float).reshape(-1)
            adv = np.array(advantages).astype(float).reshape(-1)
            v_mean = float(np.mean(v)) if v.size else 0.0
            v_std = float(np.std(v)) if v.size else 0.0
            a_mean = float(np.mean(adv)) if adv.size else 0.0
            a_std = float(np.std(adv)) if adv.size else 0.0

            exp_var = ""
            try:
                if explained_variance is not None and returns is not None:
                    exp_var = float(explained_variance(np.array(returns).reshape(-1), v))
            except Exception:
                exp_var = ""

            policy_entropy = ""
            try:
                # SB3 logs negative entropy as entropy_loss; convert back if available.
                ent_loss = self.model.logger.name_to_value.get("train/entropy_loss", None)
                if ent_loss is not None:
                    policy_entropy = float(-ent_loss)
            except Exception:
                policy_entropy = ""

            lstm_hidden_norm = ""
            try:
                lstm_states = getattr(rb, "lstm_states", None)
                if lstm_states is not None:
                    # lstm_states: (hidden, cell)
                    h = lstm_states[0]
                    lstm_hidden_norm = float(np.linalg.norm(np.array(h)))
            except Exception:
                lstm_hidden_norm = ""

            extra = {
                "value_pred_mean": v_mean,
                "value_pred_std": v_std,
                "advantage_mean": a_mean,
                "advantage_std": a_std,
                "explained_variance": exp_var,
                "policy_entropy": policy_entropy,
                "lstm_hidden_norm": lstm_hidden_norm,
            }
            log_training_row("train", step_idx=next_step(), extra=extra)
        except Exception:
            return


class TCRRolloutStatsCallback(BaseCallback):
    """
    Collect rollout step records (group/action/reward/obs) and pass to TCR-enabled PPO_NEW.
    """

    def __init__(self):
        if BaseCallback is None:
            raise ImportError("stable_baselines3 is required for TCRRolloutStatsCallback.")
        super().__init__()
        self._records = []

    @staticmethod
    def _to_item_list(value, expected_len):
        if expected_len <= 0:
            return []
        if isinstance(value, (list, tuple)):
            items = list(value)
            if len(items) >= expected_len:
                return items[:expected_len]
            if not items:
                return [None] * expected_len
            return items + [items[-1]] * (expected_len - len(items))
        arr = np.asarray(value)
        if expected_len == 1:
            return [arr]
        if arr.ndim >= 1 and arr.shape[0] == expected_len:
            return [arr[i] for i in range(expected_len)]
        return [arr for _ in range(expected_len)]

    @staticmethod
    def _to_scalar_int(value, default=0):
        try:
            arr = np.asarray(value).reshape(-1)
            if arr.size == 0:
                return int(default)
            return int(arr[0])
        except Exception:
            return int(default)

    @staticmethod
    def _to_scalar_float(value, default=0.0):
        try:
            arr = np.asarray(value).reshape(-1)
            if arr.size == 0:
                return float(default)
            return float(arr[0])
        except Exception:
            return float(default)

    def _on_step(self) -> bool:
        try:
            infos = self.locals.get("infos", None)
            if infos is None:
                return True
            if not isinstance(infos, (list, tuple)):
                infos = [infos]
            n_envs = len(infos)
            if n_envs <= 0:
                return True

            actions = self._to_item_list(self.locals.get("actions", None), n_envs)
            rewards = self._to_item_list(self.locals.get("rewards", None), n_envs)

            obs_source = self.locals.get("new_obs", None)
            if obs_source is None:
                obs_source = getattr(self.model, "_last_obs", None)
            obs_list = self._to_item_list(obs_source, n_envs)

            for i in range(n_envs):
                info_i = infos[i] if i < len(infos) and isinstance(infos[i], dict) else {}
                row_dict = info_i.get("row_dict", {}) if isinstance(info_i, dict) else {}
                try:
                    if hasattr(row_dict, "to_dict"):
                        row_dict = row_dict.to_dict()
                except Exception:
                    row_dict = {}

                phase_label = ""
                try:
                    phase_label = str(info_i.get("phase_label", "") or row_dict.get("phase_label", "") or "")
                except Exception:
                    phase_label = ""
                if phase_label == "":
                    phase_label = str(globals().get("current_phase_label", "") or "")
                if phase_label == "":
                    phase_label = "unknown"

                obs_i = obs_list[i] if i < len(obs_list) else None
                if obs_i is None:
                    continue
                obs_arr = np.asarray(obs_i, dtype=np.float32)
                if obs_arr.size == 0:
                    continue

                self._records.append(
                    {
                        "group": phase_label,
                        "action": self._to_scalar_int(actions[i] if i < len(actions) else 0),
                        "reward": self._to_scalar_float(rewards[i] if i < len(rewards) else 0.0),
                        "obs": obs_arr.copy(),
                    }
                )
        except Exception:
            pass
        return True

    def _on_rollout_end(self) -> None:
        try:
            if hasattr(self.model, "tcr_consume_rollout"):
                self.model.tcr_consume_rollout(self._records)
        except Exception:
            pass
        self._records = []


def evaluate_recurrent_policy(model, env, n_eval_episodes=1):
    """
    Minimal eval loop for RecurrentPPO that maintains LSTM state.
    """
    rewards = []
    for _ in range(int(n_eval_episodes)):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        lstm_state = None
        episode_start = True
        while True:
            action, lstm_state = model.predict(
                obs,
                state=lstm_state,
                episode_start=np.array([episode_start], dtype=bool),
                deterministic=True,
            )
            obs, reward, done, _ = env.step(action)
            ep_reward += float(reward)
            episode_start = bool(done)
            if done:
                break
        rewards.append(ep_reward)
    avg = float(np.mean(rewards)) if rewards else 0.0
    std = float(np.std(rewards)) if rewards else 0.0
    return avg, std

def next_step():
    global global_step
    global_step += 1
    return global_step

def save_plot_reward_list():
    if add_ALNS == 1:
        # plot all_rewards_list and save
        for reward_index in range(len(all_rewards_list)):
            # check_RL_ALNS_iteraction_bug()
            if (reward_index + 1) % iteration_numbers_unit == 0:
                average_reward, std_reward = np.mean(all_rewards_list[
                                                     reward_index + 1 - iteration_numbers_unit:reward_index + 1]), np.std(
                    all_rewards_list[
                    reward_index + 1 - iteration_numbers_unit:reward_index + 1])
                all_average_reward.append(average_reward)
                all_deviation.append(std_reward)
        # top_line = [a + b for a, b in zip(all_average_reward, all_deviation)]
        with open(Intermodal_ALNS34959.path + "/finite_horizon_length" + str(
                episode_length) + "_delay_reward_time_dependent" + str(
            time_dependent) + "_tenterminal_" + algorithm + "_" + mode + "_" + str(
            iteration_multiply) + "multiply" + 'reward_list.txt', 'w') as f:
            for reward in all_rewards_list:
                f.write(f"{reward}\n")
        print('all_rewards_list', all_rewards_list)
        plt.plot(range(1, len(all_rewards_list) + 1), all_rewards_list)
        # plt.fill_between(timestamps, bottom_line, top_line)
        plt.ylabel('Reward')
        plt.xlabel('Iteration')
        # plt.title('Congested terminals: ' + str(congested_terminals))
        # plt.show()
        plt.savefig(
            Intermodal_ALNS34959.path + "/finite_horizon_length" + str(
                episode_length) + "_delay_reward_time_dependent" + str(
                time_dependent) + "_tenterminal_" + algorithm + "_" + mode + "_" + str(
                iteration_multiply) + "multiply" + '.pdf',
            format='pdf', bbox_inches='tight')

def stop_wait():
    try:
        if os.path.exists(get_stop_flag_path()) and Intermodal_ALNS34959.ALNS_end_flag != 1:
            save_plot_reward_list()
            _maybe_save_checkpoint_on_stop_exit()
            sys.exit(78)
    except:
        if os.path.exists(get_stop_flag_path()):
            save_plot_reward_list()
            _maybe_save_checkpoint_on_stop_exit()
            sys.exit(78)
#@profile()
def send_action(action, trace_extra=None, obs_snapshot=None):
    if stop_everything_in_learning_and_go_to_implementation_phase == 1:
        return

    def _fail_fast(reason, err_text="", row_dict=None):
        msg = f"[RL][FATAL] send_action {reason}"
        if err_text:
            msg += f" | {err_text}"
        print(msg)
        try:
            log_trace_from_row(
                row_dict or {},
                "send_action_fatal",
                action=action,
                source="RL",
                extra={"reason": str(reason), "error": str(err_text)},
            )
        except Exception:
            pass
        raise SystemExit(124)

    break_flag = 0
    wait_start = time.time()
    last_log = wait_start
    while True:
        if stop_everything_in_learning_and_go_to_implementation_phase == 1:
            return
        if len(Intermodal_ALNS34959.state_reward_pairs) != 0:
            break
        print('len(Intermodal_ALNS34959.state_reward_pairs) == 0 in send_action function')
        timed_out, last_log, wait_start = _wait_watchdog("send_action_wait_pairs", wait_start, last_log)
        if timed_out:
            return
        if WAIT_SLEEP_S > 0:
            time.sleep(WAIT_SLEEP_S)

    wait_start = time.time()
    last_log = wait_start
    wait_slot_loops = 0
    reselect_count = 0
    last_write_error = ""
    pair_index = None

    while True:
        stop_wait()
        if stop_everything_in_learning_and_go_to_implementation_phase == 1:
            return

        slot_written = False
        for pair_index in Intermodal_ALNS34959.state_reward_pairs.index:
            try:
                check = Intermodal_ALNS34959.state_reward_pairs['uncertainty_index'][pair_index] == uncertainty_index and \
                    Intermodal_ALNS34959.state_reward_pairs['vehicle'][pair_index] == vehicle and \
                    Intermodal_ALNS34959.state_reward_pairs['request'][pair_index] == request and \
                    Intermodal_ALNS34959.state_reward_pairs['action'][pair_index] == -10000000
            except Exception:
                continue
            if implement == 0:
                if Intermodal_ALNS34959.after_action_review == 1:
                    check = check and Intermodal_ALNS34959.state_reward_pairs['uncertainty_type'][pair_index] == 'finish'
            else:
                check = check and Intermodal_ALNS34959.state_reward_pairs['uncertainty_type'][pair_index] == 'begin'

            if not check:
                continue

            write_retry = 0
            while True:
                stop_wait()
                if stop_everything_in_learning_and_go_to_implementation_phase == 1:
                    return
                try:
                    Intermodal_ALNS34959.state_reward_pairs['action'][pair_index] = action
                    try:
                        row_dict = dict(Intermodal_ALNS34959.state_reward_pairs.loc[pair_index])
                    except Exception:
                        row_dict = {}
                    row_dict["pair_index"] = pair_index
                    try:
                        _decision_note_send(
                            row=row_dict,
                            action=action,
                            obs_snapshot=obs_snapshot,
                            trace_extra=trace_extra,
                            source="RL",
                        )
                    except Exception:
                        pass
                    log_trace_from_row(row_dict, "send_action", action=action, source="RL", extra=trace_extra)
                    slot_written = True
                    break
                except Exception as exc:
                    write_retry += 1
                    last_write_error = f"{type(exc).__name__}: {exc}"
                    if write_retry == 1 or write_retry % SEND_ACTION_ERROR_LOG_EVERY == 0:
                        print(
                            f"[RL][WARN] send_action_write_retry pair={pair_index} "
                            f"retry={write_retry}/{SEND_ACTION_MAX_WRITE_RETRY} err={last_write_error}"
                        )
                    timed_out, last_log, wait_start = _wait_watchdog(
                        "send_action_write_retry",
                        wait_start,
                        last_log,
                    )
                    if timed_out:
                        return
                    if write_retry >= SEND_ACTION_MAX_WRITE_RETRY:
                        reselect_count += 1
                        print(
                            f"[RL][WARN] send_action reselect slot after write failures "
                            f"(reselect={reselect_count}/{SEND_ACTION_MAX_RESELECT}, pair={pair_index})"
                        )
                        break
                    if WAIT_SLEEP_S > 0:
                        time.sleep(WAIT_SLEEP_S)
                    continue

            if slot_written:
                break_flag = 1
                break

        if break_flag == 1 and pair_index is not None:
            confirm_retry = 0
            while True:
                stop_wait()
                if stop_everything_in_learning_and_go_to_implementation_phase == 1:
                    return
                try:
                    if Intermodal_ALNS34959.state_reward_pairs['action'][pair_index] != -10000000:
                        break
                    Intermodal_ALNS34959.state_reward_pairs['action'][pair_index] = action
                except Exception as exc:
                    confirm_retry += 1
                    err_text = f"{type(exc).__name__}: {exc}"
                    if confirm_retry == 1 or confirm_retry % SEND_ACTION_ERROR_LOG_EVERY == 0:
                        print(
                            f"[RL][WARN] send_action_confirm_slot_retry pair={pair_index} "
                            f"retry={confirm_retry}/{SEND_ACTION_MAX_CONFIRM_RETRY} err={err_text}"
                        )
                    if confirm_retry >= SEND_ACTION_MAX_CONFIRM_RETRY:
                        _fail_fast("confirm_retry_exhausted", err_text)
                timed_out, last_log, wait_start = _wait_watchdog(
                    "send_action_confirm_slot",
                    wait_start,
                    last_log,
                )
                if timed_out:
                    return
                if WAIT_SLEEP_S > 0:
                    time.sleep(WAIT_SLEEP_S)
            break

        if reselect_count >= SEND_ACTION_MAX_RESELECT:
            _fail_fast("reselect_exhausted", last_write_error)

        wait_slot_loops += 1
        if wait_slot_loops >= SEND_ACTION_MAX_WAIT_SLOT_LOOPS:
            _fail_fast("wait_slot_exhausted", f"loops={wait_slot_loops},last_err={last_write_error}")

        timed_out, last_log, wait_start = _wait_watchdog("send_action_wait_slot", wait_start, last_log)
        if timed_out:
            return
        if WAIT_SLEEP_S > 0:
            time.sleep(WAIT_SLEEP_S)
#@profile()
def get_state(chosen_pair, table_number=-1, request_number_in_R=-1, duration_type='x', dynamic_t_begin=-1):
    global severity_level
    #check_RL_ALNS_iteraction_bug()
    state_list = [None] * 13
    state_list[0] = chosen_pair[4] #delay tolerance
    passed_terminals = chosen_pair[5]
    if len(passed_terminals) < 10:
        for z in range(10 - len(passed_terminals)):
            passed_terminals.append(-1)
    state_list[1:11] = passed_terminals
    state_list[11] = chosen_pair[6] #current time
    uncertainty_index = chosen_pair[0]
    if table_number == -1:
        table_number = Dynamic_ALNS_RL34959.table_number
        request_number_in_R = Intermodal_ALNS34959.request_number_in_R
        duration_type = Intermodal_ALNS34959.duration_type
        dynamic_t_begin = Intermodal_ALNS34959.dynamic_t_begin
    # 防止索引越界：文件从 0 到 999
    table_number = max(0, min(table_number, 999))

    data_path = resolve_dynamic_data_path(request_number_in_R, table_number, duration_type, add_event_types)
    Data = pd.ExcelFile(data_path)
    global current_gt_mean, current_phase_label
    try:
        meta_df = pd.read_excel(Data, "__meta__")
        if "Property" in meta_df.columns and "Value" in meta_df.columns:
            meta_map = dict(zip(meta_df["Property"].astype(str), meta_df["Value"]))
            current_gt_mean = meta_map.get("gt_mean", "")
            current_phase_label = meta_map.get("phase_label", "")
        else:
            current_gt_mean = meta_df["gt_mean"].iloc[0] if "gt_mean" in meta_df.columns and len(meta_df) else ""
            current_phase_label = meta_df["phase_label"].iloc[0] if "phase_label" in meta_df.columns and len(meta_df) else ""
    except Exception:
        current_gt_mean = ""
        current_phase_label = ""
    # check_repeat_r_in_R_pool(), check_T_k_record_and_R()
    # below are travel time uncertainty, including delay and congestion at nodes and arcs
    R_change_dynamic_travel_time = pd.read_excel(Data, 'R_' + str(request_number_in_R) + '_' + str(
            dynamic_t_begin) + ' (2)')
    # data_path = "/data/yimeng/Uncertainties Dynamic planning under unexpected events/plot_distribution_targetInstances_disruption_" + duration_type + "_not_time_dependent/R" + str(
    #     request_number_in_R) + "/Intermodal_EGS_data_dynamic_congestion" + str(table_number) + ".xlsx"
    # Data = pd.ExcelFile(data_path)
    #
    for index in R_change_dynamic_travel_time.index:
        if uncertainty_index == R_change_dynamic_travel_time['uncertainty_index'][index]:
            if R_change_dynamic_travel_time['type'][index] == 'congestion':
                # if dynamic_RL_online.implement == 1:
                #     #then send state to RL
                #
                # else:
                duration = eval(R_change_dynamic_travel_time['duration'][index])
                break
    duration_length = duration[1] - duration[0]
    # if state_list[0] >= duration_length:
    #     severity_level = 0
    # else:
    #     severity_level = 1
    ###############
    # if duration_length <= 20:
    #     severity_level = 1
    # elif duration_length <= 25:
    #     severity_level = 2
    # elif duration_length <= 30:
    #     severity_level = 3
    # elif duration_length <= 35:
    #     severity_level = 4
    # elif duration_length <= 40:
    #     severity_level = 5
    # elif duration_length <= 45:
    #     severity_level = 6
    # elif duration_length <= 50:
    #     severity_level = 7
    # elif duration_length <= 55:
    #     severity_level = 8
    # elif duration_length <= 60:
    #     severity_level = 9
    # elif duration_length <= 65:
    #     severity_level = 10
    # elif duration_length <= 70:
    #     severity_level = 11
    # elif duration_length <= 75:
    #     severity_level = 12
    # elif duration_length <= 80:
    #     severity_level = 13
    # else:
    #     severity_level = 14
    #################
    if duration_length <= 20:
        severity_level = 1
    elif duration_length <= 40:
        severity_level = 2
    # *60
    elif duration_length <= 60:
        severity_level = 3
    # *60
    elif duration_length <= 80:
        severity_level = 4
    elif duration_length <= 100:
        severity_level = 5
    else:
        severity_level = 6
    # state_list[12] = severity_level
    number_of_severity_levels = 6
    if number_of_severity_levels > 2 and wrong_severity_level_with_probability != 0:
        number = int(np.random.choice([1, 2], size=(1,), p=[wrong_severity_level_with_probability,
                                                            1 - wrong_severity_level_with_probability]))
        if number == 1:
            # then the level is a wrong one
            severity_level = random.randint(1, number_of_severity_levels)
    if add_event_types == 1:

        event_type = R_change_dynamic_travel_time['event_types'][0]
        state_list = [state_list[0], severity_level, event_type]
    else:
        state_list = [state_list[0], severity_level]
    if STAGE_IN_OBS:
        label = str(current_stage_label or "").lower()
        if "insert" in label:
            stage_vec = [0.0, 1.0]
        elif "remove" in label:
            stage_vec = [1.0, 0.0]
        else:
            stage_vec = [0.0, 0.0]
        state_list = list(state_list) + stage_vec
    state = np.array(state_list, dtype=float)
    return state

def check_RL_ALNS_iteraction_bug():
    if implement == 1 and Intermodal_ALNS34959.ALNS_implement_start_RL_can_move == 1 and len(Intermodal_ALNS34959.state_reward_pairs) == 0:
        print('gfsfsfagsgfd')
        print('gfsfsfagsgfd')
class coordinationEnv(Env):
    def __init__(self):
        # Actions we can take, wait, go
        self.action_space = Discrete(2)
        # Cost array
        # self.observation_space = Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), high=np.array([24, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 24, 14]))
        if add_event_types == 1:
            raw_low = [0, 0, 0]
            raw_high = [200, 6, 6]
        else:
            raw_low = [0, 0]
            raw_high = [200, 6]
        self._raw_low = np.array(raw_low, dtype=np.float32)
        self._raw_high = np.array(raw_high, dtype=np.float32)
        self._raw_obs_dim = int(self._raw_low.shape[0])
        self.use_augmented_obs = bool(globals().get("USE_AUGMENTED_OBS", False))
        self.oracle_ctx_mode = _normalize_oracle_ctx_mode(globals().get("ORACLE_CTX_MODE", "none"))
        try:
            self._oracle_gt_mean_norm = float(globals().get("ORACLE_GT_MEAN_NORM", 100.0))
        except Exception:
            self._oracle_gt_mean_norm = 100.0
        if self._oracle_gt_mean_norm <= 0:
            self._oracle_gt_mean_norm = 100.0
        self._oracle_phase_to_id = {}
        self._oracle_mapping_path = None
        self._oracle_phase_dim = 0
        self._oracle_other_key = "__OTHER__"
        self._oracle_other_id = 0
        if self.use_augmented_obs and self.oracle_ctx_mode == "phase":
            phase_classes = int(globals().get("ORACLE_PHASE_CLASSES", 0) or 0)
            if phase_classes <= 0:
                phase_classes = self._infer_default_phase_classes()
            self._oracle_phase_dim = max(1, phase_classes)
            self._oracle_other_id = max(0, self._oracle_phase_dim - 1)
            self._oracle_phase_to_id = {self._oracle_other_key: self._oracle_other_id}
            self._oracle_mapping_path = Path(rl_logging.get_run_dir()) / "oracle_phase_label_map.json"
            self._save_phase_mapping(reason="init")
        if self.use_augmented_obs and self.oracle_ctx_mode == "phase":
            self._oracle_ctx_dim = 1 + self._oracle_phase_dim  # stage + onehot(phase)
        elif self.use_augmented_obs and self.oracle_ctx_mode == "mean":
            self._oracle_ctx_dim = 2  # stage + gt_mean_norm
        else:
            self._oracle_ctx_dim = 0
        if self.use_augmented_obs:
            self._aug_window_k = int(max(1, globals().get("PPO_NEW_WINDOW_K", 1)))
            delta_low = self._raw_low - self._raw_high
            delta_high = self._raw_high - self._raw_low
            frame_low = np.concatenate(
                [self._raw_low, np.array([0.0, 0.0, 0.0], dtype=np.float32), delta_low]
            ).astype(np.float32)
            frame_high = np.concatenate(
                [self._raw_high, np.array([1.0, 1.0, 1.0], dtype=np.float32), delta_high]
            ).astype(np.float32)
            if self._oracle_ctx_dim > 0:
                oracle_low, oracle_high = self._oracle_low_high()
                frame_low = np.concatenate([frame_low, oracle_low]).astype(np.float32)
                frame_high = np.concatenate([frame_high, oracle_high]).astype(np.float32)
            self._aug_x_dim = int(frame_low.shape[0])
            if self._aug_window_k > 1:
                low = np.tile(frame_low, self._aug_window_k).astype(np.float32)
                high = np.tile(frame_high, self._aug_window_k).astype(np.float32)
            else:
                low = frame_low
                high = frame_high
        else:
            self._aug_window_k = 1
            self._aug_x_dim = 0
            base_low = list(raw_low)
            base_high = list(raw_high)
            if STAGE_IN_OBS:
                base_low = base_low + [0, 0]
                base_high = base_high + [1, 1]
            low = np.array(base_low, dtype=np.float32)
            high = np.array(base_high, dtype=np.float32)
        self.observation_space = Box(low=low, high=high, dtype=np.float32)
        self._prev_o = None
        self._prev_action = 0.0
        self._prev_reward = 0.0
        self._stage_bit = 0.0
        self._x_history = deque(maxlen=max(1, self._aug_window_k))
        # self.state = [random.choice(range(0,24)), random.choice(range(0,11))]
        # Set coordination length
        self.horizon_length = 0
        # self.dis = 0

    def _infer_default_phase_classes(self):
        # Reserve one fallback slot (__OTHER__) to avoid shape changes mid-run.
        scenario = str(globals().get("SCENARIO_NAME", "") or "").upper()
        base = 3 if scenario.startswith("G_") else 2
        return int(base + 1)

    def _save_phase_mapping(self, reason="update"):
        if self._oracle_mapping_path is None:
            return
        try:
            payload = {
                "reason": str(reason),
                "oracle_ctx_mode": self.oracle_ctx_mode,
                "algo_version": str(globals().get("ALGO_VERSION", "")),
                "scenario_name": str(globals().get("SCENARIO_NAME", "")),
                "phase_dim": int(self._oracle_phase_dim),
                "other_key": self._oracle_other_key,
                "other_id": int(self._oracle_other_id),
                "phase_to_id": dict(self._oracle_phase_to_id),
                "gt_mean_norm_denom": float(self._oracle_gt_mean_norm),
                "updated_at": float(time.time()),
            }
            self._oracle_mapping_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._oracle_mapping_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _oracle_low_high(self):
        if self.oracle_ctx_mode == "phase":
            low = np.concatenate(
                [
                    np.array([0.0], dtype=np.float32),
                    np.zeros(int(self._oracle_phase_dim), dtype=np.float32),
                ]
            ).astype(np.float32)
            high = np.concatenate(
                [
                    np.array([1.0], dtype=np.float32),
                    np.ones(int(self._oracle_phase_dim), dtype=np.float32),
                ]
            ).astype(np.float32)
            return low, high
        if self.oracle_ctx_mode == "mean":
            # stage in [0,1], normalized mean kept in a conservative bounded range.
            return (
                np.array([0.0, -10.0], dtype=np.float32),
                np.array([1.0, 10.0], dtype=np.float32),
            )
        return (
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )

    def _phase_to_id(self, phase_label):
        label = str(phase_label or "").strip()
        if not label:
            label = "__EMPTY__"
        if label in self._oracle_phase_to_id:
            return int(self._oracle_phase_to_id[label])
        used_ids = set(int(v) for v in self._oracle_phase_to_id.values())
        assignable_ids = [
            idx
            for idx in range(int(self._oracle_phase_dim))
            if idx != int(self._oracle_other_id) and idx not in used_ids
        ]
        if assignable_ids:
            mapped = int(assignable_ids[0])
        else:
            mapped = int(self._oracle_other_id)
        self._oracle_phase_to_id[label] = mapped
        self._save_phase_mapping(reason="phase_label_seen")
        return mapped

    def _get_phase_onehot(self):
        if int(self._oracle_phase_dim) <= 0:
            return np.zeros(0, dtype=np.float32)
        phase_vec = np.zeros(int(self._oracle_phase_dim), dtype=np.float32)
        idx = self._phase_to_id(current_phase_label)
        if 0 <= idx < int(self._oracle_phase_dim):
            phase_vec[idx] = 1.0
        return phase_vec

    def _get_gt_mean_norm(self):
        try:
            gt = float(current_gt_mean)
        except Exception:
            gt = 0.0
        return float(gt / float(self._oracle_gt_mean_norm))

    def _compose_oracle_ctx(self, stage_bit):
        mode = self.oracle_ctx_mode
        if mode == "phase":
            phase_vec = self._get_phase_onehot()
            return np.concatenate(
                [np.array([float(stage_bit)], dtype=np.float32), phase_vec.astype(np.float32)]
            ).astype(np.float32)
        if mode == "mean":
            return np.array([float(stage_bit), float(self._get_gt_mean_norm())], dtype=np.float32)
        return np.zeros(0, dtype=np.float32)

    def _get_stage_bit(self):
        label = str(current_stage_label or "").lower()
        self._stage_bit = 1.0 if "insert" in label else 0.0
        return self._stage_bit

    def _extract_raw_obs(self, obs):
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        return arr[: self._raw_obs_dim]

    def _compose_augmented(self, obs_now, action_prev, reward_prev, prev_obs):
        obs_vec = np.asarray(obs_now, dtype=np.float32).reshape(-1)
        prev_vec = np.asarray(prev_obs, dtype=np.float32).reshape(-1)
        delta_vec = obs_vec - prev_vec
        stage_bit = float(self._get_stage_bit())
        pieces = [
            obs_vec,
            np.array([stage_bit], dtype=np.float32),
            np.array([float(action_prev)], dtype=np.float32),
            np.array([float(reward_prev)], dtype=np.float32),
            delta_vec.astype(np.float32),
        ]
        if self._oracle_ctx_dim > 0:
            pieces.append(self._compose_oracle_ctx(stage_bit))
        return np.concatenate(pieces).astype(np.float32)

    def _stack_augmented(self, x_now, reset_fill=False):
        x_vec = np.asarray(x_now, dtype=np.float32).reshape(-1)
        if self._aug_window_k <= 1:
            return x_vec
        if reset_fill or len(self._x_history) == 0:
            self._x_history.clear()
            for _ in range(self._aug_window_k):
                self._x_history.append(x_vec.copy())
        else:
            self._x_history.appendleft(x_vec.copy())
        return np.concatenate(list(self._x_history), axis=0).astype(np.float32)

    #@profile()
    def step(self, action):
        global state_action_reward_collect, all_rewards_list, wait_training_finish_last_iteration, state_action_reward_collect_for_evaluate, number_of_state_key, state_keys, iteration_times, RL_drop_finish, episode_length, next_state_reward_time_step, next_state_penalty_time_step, time_s, all_average_reward, all_deviation, timestamps, LSTM_CHAIN_LEN, LSTM_CHAIN_STEP, USE_LSTM
        # 将动作转为标量
        try:
            if isinstance(action, np.ndarray):
                action = int(action.squeeze())
            else:
                action = int(action)
        except Exception:
            pass
        info = {}
        next_state_raw = None

        # truck picks up containers at A, then go to B to transfer to barge, plan transshipment time is 30
        # between A and B, 300 km, truck speed 75 km/h, so 4 hour go to terminal B, truck on route 5/h
        # therefore, when truck arrives before 30, if wait, then 1/h, if store containers 5/h, if arrives after 30, 20/h
        # new case under uncertainty: but barge delayed, new transshipment time is 35
        #
        # Apply action
        # 0 wait 1/h
        # 1 go 5/h,
        # 2 store 20/h


        #choose T
        #Contargo/COSCO has two options, choose Rotterdam or Antwerp
        #                     Contargo
        #               Rotterdam  Antwerp
        #COSCO Rotterdam    (1,2)    (0,0)
        #      Antwerp      (0,0)    (2,1)
        # For COSCO's transportation, Rotterdam is the best choice and the profit is 2. For Contargo's transportation, Antwerp is the best choice
        # if both COSCO and Contargo choose unilateral action, i.e., COSCO choose Rotterdam and Contargo choose Antwerp, then reward is 0.
        # Only when they choose the same terminal, reward is positive.
        # if self.state[0] >= 10 or self.state[0] <= 14:
            # congestion_duration = random.choice(range(0,4))
            # congestion_duration = np.random.uniform(low=1, high=5)
        # if self.state[1]
        if add_ALNS == 1:
            if stop_everything_in_learning_and_go_to_implementation_phase == 1:
                return self.state, 0, True, {}
            if time_s >= total_timesteps2:
                wait_training_finish_last_iteration = 1


            #send the action to ALNS, and let it check the feasibility
            if evaluate == 1:
                if not state_keys or not state_action_reward_collect_for_evaluate:
                    reward = 0
                    all_rewards_list.append(reward)
                    if _hat_is_active():
                        _hat_update_stats(reward, action)
                        _hat_update_train_params()
                    if self.use_augmented_obs:
                        next_state_raw = self._extract_raw_obs(self.state)
                    return self.state, reward, True, {}
                state_key = random.choice(state_keys)
                action_map = state_action_reward_collect_for_evaluate.get(state_key, {})
                if action in action_map:
                    reward = action_map[action]
                elif action_map:
                    reward = random.choice(list(action_map.values()))
                else:
                    reward = 0
                all_rewards_list.append(reward)
                if _hat_is_active():
                    _hat_update_stats(reward, action)
                    _hat_update_train_params()
                if self.use_augmented_obs:
                    next_state_raw = self._extract_raw_obs(self.state)
            else:
                if stop_everything_in_learning_and_go_to_implementation_phase == 1:
                    return self.state, 0, True, {}
                send_action(action, obs_snapshot=self.state)

                #get the reward from ALNS
                wait_start = time.time()
                last_log = wait_start
                while True:
                    # print('step 1')
                    stop_wait()
                    if stop_everything_in_learning_and_go_to_implementation_phase == 1:
                        return self.state, 0, True, {}
                    break_flag = 0
                    for pair_index in Intermodal_ALNS34959.state_reward_pairs.index:
                        # try:
                            # print('RL', Intermodal_ALNS34959.state_reward_pairs)
                        try:
                        # print(Intermodal_ALNS34959.state_reward_pairs,pair_index)
                            check = Intermodal_ALNS34959.state_reward_pairs['uncertainty_index'][pair_index] == uncertainty_index and \
                                Intermodal_ALNS34959.state_reward_pairs['vehicle'][pair_index] == vehicle and Intermodal_ALNS34959.state_reward_pairs['request'][pair_index] == request and Intermodal_ALNS34959.state_reward_pairs['reward'][pair_index] != -10000000
                        except:
                            #continue while break current for loop
                            #IndexError: tuple index out of range may happen and I do not know why. Maybe ALNS is changing it and RL use it, so conflict
                            break
                        #     print(pair_index, Intermodal_ALNS34959.state_reward_pairs, 'IndexError: tuple index out of range')
                        #     sys.exit(-1)
                        if check:
                            reward = Intermodal_ALNS34959.state_reward_pairs['reward'][pair_index]
                            if type(reward).__module__ == 'numpy':
                                reward = reward[0,0]
                            all_rewards_list.append(reward)
                            if algorithm == "PPO_HAT_PDI" and evaluate == 0 and implement == 0:
                                try:
                                    PDI_REWARD_LIST.append(float(reward))
                                    PDI_GT_MEAN_LIST.append(float(current_gt_mean) if current_gt_mean != "" else 0.0)
                                    PDI_PHASE_LIST.append(str(current_phase_label))
                                except Exception:
                                    pass
                            recent_rewards.append(reward)
                            if _hat_is_active():
                                _hat_update_stats(reward, action)
                                _hat_update_train_params()
                            step_id = next_step()
                            if self.use_augmented_obs:
                                try:
                                    next_state_raw = self._extract_raw_obs(
                                        get_state(Intermodal_ALNS34959.state_reward_pairs.loc[pair_index])
                                    )
                                except Exception:
                                    next_state_raw = self._extract_raw_obs(self.state)
                            try:
                                row_dict = dict(Intermodal_ALNS34959.state_reward_pairs.loc[pair_index])
                            except:
                                row_dict = {}
                            row_dict["pair_index"] = pair_index
                            uncertainty_type = Intermodal_ALNS34959.state_reward_pairs['uncertainty_type'][pair_index]
                            info = {
                                "row_dict": row_dict,
                                "uncertainty_type": uncertainty_type,
                                "pair_index": pair_index,
                                "phase_label": str(current_phase_label or ""),
                            }
                            if not LBKLAC_CUSTOM_LOGGING:
                                log_training_row("implement" if implement == 1 else "train", step_idx=step_id, reward=reward)
                                try:
                                    _decision_finalize(
                                        row=row_dict,
                                        reward=reward,
                                        action=row_dict.get("action", action),
                                        stage="receive_reward",
                                        source="RL",
                                    )
                                except Exception:
                                    pass
                                log_trace_from_row(row_dict, "receive_reward", action=row_dict.get('action', ''), reward=reward, source="RL")
                            # parallel_save_excel(path + 'state_reward_pairs.xlsx', state_reward_pairs, 'state_reward_pairs')
                            #drop the finish
                            # Intermodal_ALNS34959.state_reward_pairs = Intermodal_ALNS34959.state_reward_pairs.drop(
                            #     labels=pair_index,
                            #     axis=0)
                            break_flag = 1
                            break
                        elif Intermodal_ALNS34959.state_reward_pairs['action'][pair_index] == -10000000:
                            send_action(action, obs_snapshot=self.state)
                        # except:
                        #     break
                    if break_flag == 1:
                        drop_record = 1
                        if drop_record == 1:
                            if uncertainty_type == 'finish':
                                for pair_index in Intermodal_ALNS34959.state_reward_pairs.index:
                                    if Intermodal_ALNS34959.state_reward_pairs['uncertainty_index'][
                                        pair_index] == uncertainty_index and Intermodal_ALNS34959.state_reward_pairs['request'][
                                                pair_index] == request and Intermodal_ALNS34959.state_reward_pairs['reward'][pair_index] != -10000000:
                                        # print('RL_drop', Intermodal_ALNS34959.state_reward_pairs)

                                        # collect the historical state_action_reward pairs
                                        while True:
                                            # print('step 2')
                                            stop_wait()
                                            if stop_everything_in_learning_and_go_to_implementation_phase == 1:
                                                return self.state, 0, True, {}
                                            try:
                                                add_row = list(Intermodal_ALNS34959.state_reward_pairs.loc[pair_index])
                                                break
                                            except:
                                                print("add_row = list(Intermodal_ALNS34959.state_reward_pairs.loc[pair_index])")
                                                continue
                                        if np.size(state_action_reward_collect) > 0:
                                            if not any(np.equal(state_action_reward_collect,add_row).all(1)):
                                                state_action_reward_collect = np.vstack([state_action_reward_collect, add_row])
                                                table_number_collect[len(state_action_reward_collect)-1] = [Dynamic_ALNS_RL34959.table_number, Intermodal_ALNS34959.request_number_in_R, Intermodal_ALNS34959.duration_type, Intermodal_ALNS34959.dynamic_t_begin]
                                        else:
                                            state_action_reward_collect = np.vstack(
                                                [state_action_reward_collect, add_row])
                                            table_number_collect[len(state_action_reward_collect)-1] = [Dynamic_ALNS_RL34959.table_number, Intermodal_ALNS34959.request_number_in_R, Intermodal_ALNS34959.duration_type, Intermodal_ALNS34959.dynamic_t_begin]
                                        break
                                        # remove two records of uncertainty begin and finish only when uncertainty finishes
                                        # Intermodal_ALNS34959.state_reward_pairs = Intermodal_ALNS34959.state_reward_pairs.drop(labels=pair_index,
                                        #                                                                   axis=0)
                                #clear all data in pairs
                                Intermodal_ALNS34959.state_reward_pairs = Intermodal_ALNS34959.state_reward_pairs.iloc[0:0]
                                        # print('RL_drop_finish', Intermodal_ALNS34959.state_reward_pairs)
                        RL_drop_finish = 1


                        break
                    timed_out, last_log, wait_start = _wait_watchdog("step_wait_reward", wait_start, last_log)
                    if timed_out:
                        return self.state, 0, True, info
                    if WAIT_SLEEP_S > 0:
                        time.sleep(WAIT_SLEEP_S)
        else:
            for terminal in range(10):
                if time_dependent == 0:
                    locals()['congestion_duration' + str(int(terminal))] = np.random.normal(eval('congestion_'+str(int(terminal)) + '_mean'),1)
                else:
                    locals()['congestion_duration' + str(int(terminal))] = np.random.normal(
                        self.state[11]%24/5, 1)
            # congestion_duration1 = np.random.normal(congestion_2_mean,1)
            # congestion_duration2 = np.random.normal(congestion_3_mean,1)
            # congestion_duration0 = np.random.gamma(congestion_1_mean, 1)
            # congestion_duration1 = np.random.gamma(congestion_2_mean, 1)
            time_s += 1
            if time_s % iteration_multiply == 0:
                timestamps.append(time_s)
                time_s_save = time_s
            #     model.save('congestion_terminal_mean_list' + '_20220220congestion_stochastic100000')
                # load
                # model = PPO.load("PPO2021113a0coordination")
                if USE_LSTM:
                    average_reward, deviation = evaluate_recurrent_policy(model, env, n_eval_episodes=iteration_numbers_unit)
                else:
                    if sb3_evaluate_policy is None:
                        raise ImportError("stable_baselines3 is required for evaluate_policy in non-ALNS mode.")
                    average_reward, deviation = sb3_evaluate_policy(model, env, n_eval_episodes=iteration_numbers_unit, render=False)
                all_average_reward.append(average_reward)
                all_deviation.append(average_reward)
                time_s = time_s_save

            if time_s == iteration_numbers_unit * iteration_multiply:
                # top_line = [a + b for a, b in zip(all_average_reward, all_deviation)]
                # bottom_line = [a - b for a, b in zip(all_average_reward, all_deviation)]
                real_average_reward = [element / episode_length for element in all_average_reward]
                plt.plot(timestamps, all_average_reward)
                # plt.fill_between(timestamps, bottom_line, top_line)
                plt.ylabel('Average Reward')
                plt.xlabel('Timestamp')
                # plt.title('Congested terminals: ' + str(congested_terminals))
                # plt.show()
                if repeat == 4:
                    plot_dir = os.path.join(
                        ROOT_DIR,
                        "Uncertainties Dynamic planning under unexpected events",
                        "Average reward plots",
                    )
                    os.makedirs(plot_dir, exist_ok=True)
                    plot_path = os.path.join(
                        plot_dir,
                        "finite_horizon_length"
                        + str(episode_length)
                        + "_delay_reward_time_dependent"
                        + str(time_dependent)
                        + "_tenterminal_"
                        + algorithm
                        + "_"
                        + mode
                        + "_"
                        + str(iteration_multiply)
                        + "multiply"
                        + ".pdf",
                    )
                    plt.savefig(plot_path, format="pdf", bbox_inches="tight")
            influenced_time = 0
            if non_stationary == 0 or (time_s  <= iteration_numbers_unit * iteration_multiply / 2):

                for i in range(1, 11):
                    terminal = self.state[i]
                    if terminal == -1:
                        break
                    # travel_time = 3
                    locals()['latter_terminal_influenced_time' + str(int(terminal))] = max(0, eval(
                        'congestion_duration' + str(int(self.state[i]))))

                    for j in range(1,i):
                        locals()['latter_terminal_influenced_time' + str(int(terminal))] = eval('latter_terminal_influenced_time' + str(int(terminal))) - eval('congestion_duration' + str(int(self.state[j]))) - eval('travel_time_' + mode)[int(self.state[j]), int(self.state[j+1])]
                    locals()['latter_terminal_influenced_time' + str(int(terminal))] = max(0, eval('latter_terminal_influenced_time' + str(int(terminal))))

                    influenced_time = influenced_time + eval('latter_terminal_influenced_time' +  str(int(terminal)))
            else:
                # influenced_time = np.random.normal(2,1)
                influenced_time = random.choice(range(0, 8))
            # if self.state[11] >= 18 or self.state[11] <= 8:
            #     if action == 0:
            #         self.state = 2
            #     else:
            #         self.state = 0
            # else:
            if next_state_reward_time_step == time_s:
                # reward = 1
                reward = -1
            elif next_state_penalty_time_step == time_s:
                # reward = -1
                reward = -3
            else:
                reward = -2

            if (action == 0 and (self.state[0] >= influenced_time)) or (action == 1 and (self.state[0] < influenced_time)):
                next_state_reward_time_step = time_s + 1
            else:
                next_state_penalty_time_step = time_s + 1


            new_seq = get_new_seq()
            self.state = np.array([[random.choice(range(0, 4)), new_seq[0], new_seq[1], new_seq[2], new_seq[3],
                                    new_seq[4], new_seq[5], new_seq[6], new_seq[7], new_seq[8], new_seq[9],
                                    random.choice(range(0, 24))]]).astype(float)

            # # Calculate reward
            # if self.state == 2:
            #     reward = 1
            # else:
            #     reward = 0
        # Reduce coordination length by 1 second
        self.horizon_length += 1
        # print(self.horizon_length)
            # Check if coordination is done
        if add_ALNS == 1 and USE_LSTM and int(LSTM_CHAIN_LEN) > 1:
            LSTM_CHAIN_STEP += 1
            done = LSTM_CHAIN_STEP >= int(LSTM_CHAIN_LEN)
            if done:
                LSTM_CHAIN_STEP = 0
            try:
                self.state = self.reset()
            except SystemExit:
                raise
            return self.state, reward, done, info
        if self.horizon_length == episode_length:
            done = True
        else:
            done = False

        # Apply temperature noise
        # self.state += random.randint(-1,1)
        # Set placeholder for info (if not already populated)
        if not info:
            info = {}
        if self.use_augmented_obs:
            if next_state_raw is None:
                next_state_raw = self._extract_raw_obs(self.state)
            next_state_raw = np.asarray(next_state_raw, dtype=np.float32).reshape(-1)
            prev_obs = self._prev_o
            if prev_obs is None or np.asarray(prev_obs).shape != next_state_raw.shape:
                prev_obs = next_state_raw
            x_now = self._compose_augmented(
                obs_now=next_state_raw,
                action_prev=self._prev_action,
                reward_prev=self._prev_reward,
                prev_obs=prev_obs,
            )
            self.state = self._stack_augmented(x_now, reset_fill=False)
            self._prev_o = next_state_raw.copy()
            self._prev_action = float(action)
            self._prev_reward = float(reward)
        print(time_s)
        # Return step information
        print(self.state, 'action', action,  reward)
        if add_ALNS == 1:
            time_s += 1
        return self.state, reward, done, info

    def render(self):
        # Implement viz
        pass

    #@profile()
    def reset(self):
        global wait_training_finish_last_iteration, number_of_state_key, state_keys, congested_terminals, uncertainty_index, vehicle, request
        #check_RL_ALNS_iteraction_bug()
        wait_training_finish_last_iteration = 0
        # Reset initial cost
        # self.state = np.array([[random.choice(range(0,24)),random.choice(range(0,4))]]).astype(float)
        #generate a random terminal sequence
        if add_ALNS == 0:
            new_seq = get_new_seq()
            self.state = np.array([random.choice(range(0, 4)),new_seq[0],new_seq[1],new_seq[2], new_seq[3],new_seq[4],new_seq[5],new_seq[6],new_seq[7],new_seq[8], new_seq[9], random.choice(range(0, 24))]).astype(float)
        else:
            if evaluate == 1:
                if not state_keys:
                    self.state = np.zeros(self.observation_space.shape, dtype=float)
                else:
                    self.state = np.array(random.choice(state_keys), dtype=float)

            else:
                if stop_everything_in_learning_and_go_to_implementation_phase == 1:
                    self.state = np.zeros(self.observation_space.shape, dtype=float)
                    return self.state
                #this is used for both learning and implement
                #read which terminals a vehicle passes
                break_flag = 0
                #check_RL_ALNS_iteraction_bug()
                wait_start = time.time()
                last_log = wait_start
                while True:
                    #check_RL_ALNS_iteraction_bug()
                    # if implement == 1 and ALNS_got_action_in_implementation == 0:
                    #     print('it should be 1')
                    # print('reset 1')
                    stop_wait()
                    if stop_everything_in_learning_and_go_to_implementation_phase == 1:
                        self.state = np.zeros(self.observation_space.shape, dtype=float)
                        return self.state
                    # Intermodal_ALNS34959.state_reward_pairs = parallel_read_excel(path + 'state_reward_pairs.xlsx', 'state_reward_pairs')
                    for pair_index in Intermodal_ALNS34959.state_reward_pairs.index:
                        #check_RL_ALNS_iteraction_bug()
                        while True:
                            #check_RL_ALNS_iteraction_bug()
                            # print('reset 2')
                            stop_wait()
                            try:
                                Intermodal_ALNS34959.ALNS_end_flag
                            except:
                                continue
                            if Intermodal_ALNS34959.ALNS_end_flag == 1:
                                save_plot_reward_list()
                                sys.exit('end_RL_due_ALNS_ends')
                            break
                        try:
                            if Intermodal_ALNS34959.state_reward_pairs['action'][pair_index] == -10000000:
                                #check_RL_ALNS_iteraction_bug()
                                chosen_pair = Intermodal_ALNS34959.state_reward_pairs.loc[pair_index]
                                self.state = get_state(chosen_pair)
                                uncertainty_index, vehicle, request = Intermodal_ALNS34959.state_reward_pairs['uncertainty_index'][pair_index], Intermodal_ALNS34959.state_reward_pairs['vehicle'][pair_index], Intermodal_ALNS34959.state_reward_pairs['request'][pair_index]
                                break_flag = 1
                                break
                        except:
                            continue

                    if break_flag == 1:
                        break
                    timed_out, last_log, wait_start = _wait_watchdog("reset_wait_state", wait_start, last_log)
                    if timed_out:
                        self.state = np.zeros(self.observation_space.shape, dtype=float)
                        return self.state
                    if WAIT_SLEEP_S > 0:
                        time.sleep(WAIT_SLEEP_S)
        # Reset coordination time
        self.horizon_length = 0
        # self.dis = 0
        if self.use_augmented_obs:
            raw_state = self._extract_raw_obs(self.state)
            self._prev_o = raw_state.copy()
            self._prev_action = 0.0
            self._prev_reward = 0.0
            x0 = self._compose_augmented(
                obs_now=raw_state,
                action_prev=self._prev_action,
                reward_prev=self._prev_reward,
                prev_obs=self._prev_o,
            )
            self.state = self._stack_augmented(x0, reset_fill=True)
        return self.state

def get_new_seq():
    sequence = [i for i in range(10)]
    new_sequence = copy.deepcopy(sequence)
    new_seq = []
    for i in range(len(sequence)):
        a_terminal = random.choice(sequence)
        sequence.remove(a_terminal)
        if random.choice([0, 1]) == 1:
            if len(new_seq) > 1:
                # continue
                if eval('travel_time_' + mode)[old_terminal, a_terminal] > 10000:
                    continue
            new_seq.append(a_terminal)
            old_terminal = a_terminal
    congested_terminals = copy.deepcopy(new_seq)
    for i in range(len(new_sequence) - len(new_seq)):
        new_seq.append(-1)
    return new_seq

def append_new_line(file_name, text_to_append):
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)

def main(algorithm2, mode2):
    global wrong_severity_level_with_probability, add_event_types, stop_everything_in_learning_and_go_to_implementation_phase, clear_pairs_done, ALNS_got_action_in_implementation, table_number_collect, state_action_reward_collect, all_rewards_list, wait_training_finish_last_iteration, state_action_reward_collect_for_evaluate, number_of_state_key, state_keys, evaluate, implement, iteration_times, RL_drop_finish, non_stationary, algorithm, time_dependent, episode_length, next_state_reward_time_step, next_state_penalty_time_step, total_timesteps2, iteration_multiply, add_ALNS, iteration_numbers_unit, mode, travel_time_barge, travel_time_train, travel_time_truck, time_s, model, env, all_average_reward,all_deviation, timestamps, repeat, sucess_times, curriculum_converged, curriculum_last_avg_reward, LBKLAC_CUSTOM_LOGGING, USE_LSTM, STAGE_IN_OBS, USE_AUGMENTED_OBS, ALGO_VERSION, PPO_NEW_WINDOW_K, STAGE_MODE, INIT_MODEL_PATH, SAVE_MODEL_PATH, LSTM_CHAIN_LEN, LSTM_CHAIN_STEP, ORACLE_CTX_MODE, ORACLE_GT_MEAN_NORM, ORACLE_PHASE_CLASSES, PDI_GT_MEAN_LIST, PDI_PHASE_LIST, PDI_REWARD_LIST, _CHECKPOINT_SAVED_ON_STOP
    add_event_types =0 
    stop_everything_in_learning_and_go_to_implementation_phase = 0
    _CHECKPOINT_SAVED_ON_STOP = False
    clear_pairs_done = 0
    LBKLAC_CUSTOM_LOGGING = False
    # only_stop_once_by_implementation = 0
    evaluate = 0
    implement = 0
    wrong_severity_level_with_probability = 0
    while True:
        try:
            with open(Intermodal_ALNS34959.path + "/" + 'wrong_severity_level_with_probability.txt', 'w') as f:
                f.write(f"{str(wrong_severity_level_with_probability)}\n")
            break
        except:
            pass
    RL_drop_finish = 0
    iteration_times = 0
    #actual
    algorithm, mode = algorithm2, mode2
    ALGO_VERSION = (os.environ.get("RL_ALGO_VERSION", "v1") or "v1").strip().lower()
    STAGE_MODE = _normalize_stage_mode(os.environ.get("RL_STAGE_MODE", "train_eval"))
    INIT_MODEL_PATH = (os.environ.get("RL_INIT_MODEL_PATH", "") or "").strip()
    SAVE_MODEL_PATH = (os.environ.get("RL_SAVE_MODEL_PATH", "") or "").strip()
    ppo_new_plain_obs_versions = (
        "v6.1_cvarppo", "v61_cvarppo", "v6_1_cvarppo",
        "v7.1_poolppo", "v71_poolppo", "v7_1_poolppo",
    )
    ppo_new_use_aug_default = algorithm == "PPO_NEW" and ALGO_VERSION not in ppo_new_plain_obs_versions
    USE_AUGMENTED_OBS = bool(
        ppo_new_use_aug_default or os.environ.get("RL_USE_AUGMENTED_OBS", "0").strip() == "1"
    )
    phase_versions = ("v4.2_phase", "v42_phase", "v4_2_phase")
    mean_versions = ("v4.2_mean", "v42_mean", "v4_2_mean")
    env_oracle_mode_raw = os.environ.get("RL_ORACLE_CTX_MODE", "").strip()
    if algorithm == "PPO_NEW" and ALGO_VERSION in phase_versions:
        default_oracle_mode = "phase"
    elif algorithm == "PPO_NEW" and ALGO_VERSION in mean_versions:
        default_oracle_mode = "mean"
    else:
        default_oracle_mode = "none"
    if env_oracle_mode_raw:
        ORACLE_CTX_MODE = _normalize_oracle_ctx_mode(env_oracle_mode_raw)
    else:
        ORACLE_CTX_MODE = default_oracle_mode
    try:
        ORACLE_GT_MEAN_NORM = float(os.environ.get("RL_ORACLE_GT_MEAN_NORM", "100.0"))
    except Exception:
        ORACLE_GT_MEAN_NORM = 100.0
    if ORACLE_GT_MEAN_NORM <= 0:
        ORACLE_GT_MEAN_NORM = 100.0
    try:
        ORACLE_PHASE_CLASSES = int(os.environ.get("RL_ORACLE_PHASE_CLASSES", "0") or 0)
    except Exception:
        ORACLE_PHASE_CLASSES = 0
    if algorithm == "PPO_NEW":
        if ALGO_VERSION in ("v3.1", "v31", "v3_1"):
            default_window = 8
        elif ALGO_VERSION in (
            "v2", "v3", "v3.2", "v32", "v3_2",
            "v4", "v4.1", "v41", "v4_1",
            "v4.2_phase", "v42_phase", "v4_2_phase",
            "v4.2_mean", "v42_mean", "v4_2_mean",
            "v4.3_ent", "v43_ent", "v4_3_ent",
            "v4.3_logit_bias", "v43_logit_bias", "v4_3_logit_bias",
            "v5.1_abppo", "v51_abppo", "v5_1_abppo",
            "v5.2_qcritic", "v52_qcritic", "v5_2_qcritic",
            "v5.3_auxweak", "v53_auxweak", "v5_3_auxweak",
            "v6.2_v3cvar", "v62_v3cvar", "v6_2_v3cvar",
            "v6.3_cadm", "v63_cadm", "v6_3_cadm",
            "v7.2_poolv3", "v72_poolv3", "v7_2_poolv3",
            "v8_a", "v8-a", "v8a", "pponew_v8_a", "pponewv8_a",
            "v9_a", "v9-a", "v9a", "pponew_v9_a", "pponewv9_a",
            "v8_a2", "v8-a2", "v8a2", "pponew_v8_a2", "pponewv8_a2",
            "v9_a2", "v9-a2", "v9a2", "pponew_v9_a2", "pponewv9_a2",
            "v10_a", "v10-a", "v10a", "pponew_v10_a", "pponewv10_a",
            "v10_b", "v10-b", "v10b", "pponew_v10_b", "pponewv10_b",
            "v10_c", "v10-c", "v10c", "pponew_v10_c", "pponewv10_c",
            "v10_d", "v10-d", "v10d", "pponew_v10_d", "pponewv10_d",
            "v10_e", "v10-e", "v10e", "pponew_v10_e", "pponewv10_e",
        ):
            default_window = 4
        else:
            default_window = 1
        try:
            PPO_NEW_WINDOW_K = int(os.environ.get("RL_PPO_NEW_WINDOW", str(default_window)))
        except Exception:
            PPO_NEW_WINDOW_K = default_window
        PPO_NEW_WINDOW_K = max(1, int(PPO_NEW_WINDOW_K))
        if ALGO_VERSION == "v1":
            PPO_NEW_WINDOW_K = 1
    else:
        PPO_NEW_WINDOW_K = 1
    if STAGE_MODE == "eval_only":
        implement = 1
        evaluate = 0
        stop_everything_in_learning_and_go_to_implementation_phase = 0
    else:
        implement = 0
    USE_LSTM = algorithm in ("PPO_LSTM", "REC_PPO", "RECURRENTPPO", "PPO_HAT_LSTM")
    if USE_LSTM:
        STAGE_IN_OBS = os.environ.get("RL_STAGE_IN_OBS", "1").strip() == "1"
        try:
            LSTM_CHAIN_LEN = int(os.environ.get("LSTM_CHAIN_LEN", "10"))
        except Exception:
            LSTM_CHAIN_LEN = 10
        LSTM_CHAIN_LEN = max(1, int(LSTM_CHAIN_LEN))
        LSTM_CHAIN_STEP = 0
    else:
        STAGE_IN_OBS = os.environ.get("RL_STAGE_IN_OBS", "0").strip() == "1"
        LSTM_CHAIN_LEN = 1
        LSTM_CHAIN_STEP = 0
    if USE_AUGMENTED_OBS:
        STAGE_IN_OBS = False
    # reset PDI buffers per run
    PDI_GT_MEAN_LIST = []
    PDI_PHASE_LIST = []
    PDI_REWARD_LIST = []
    seed_val = resolve_seed()
    set_global_seed(seed_val)
    episode_length = 1
    if USE_LSTM and int(LSTM_CHAIN_LEN) > 1:
        episode_length = int(LSTM_CHAIN_LEN)
    next_state_reward_time_step = -1
    next_state_penalty_time_step = -1
    wait_training_finish_last_iteration = 0
    add_ALNS = 1
    all_rewards_list = []
    if add_ALNS == 1:
        while True:
            stop_wait()
            try:
                Intermodal_ALNS34959.state_reward_pairs
            except:
                continue
            break
    iteration_numbers_unit = 1
    time_dependent = 0
    record_results = pd.DataFrame(columns=['congestion_terminal_mean_list', 'average_reward', 'deviation'])
    D_path = os.path.join(ROOT_DIR, "D_EGS - 10r.xlsx")
    # algorithm = 'PPO'
    # mode = 'barge'

    D_origin_barge = pd.read_excel(D_path, 'Barge')
    D_origin_train = pd.read_excel(D_path, 'Train')
    D_origin_truck = pd.read_excel(D_path, 'Truck')

    D_origin_barge = D_origin_barge.set_index('N')
    D_origin_train = D_origin_train.set_index('N')
    D_origin_truck = D_origin_truck.set_index('N')

    D_origin_barge = D_origin_barge.values
    D_origin_train = D_origin_train.values
    D_origin_truck = D_origin_truck.values

    travel_time_barge = D_origin_barge/15
    travel_time_train = D_origin_train/45
    travel_time_truck = D_origin_truck/75

    for repeat in range(1):
        congestion_terminal_mean_list = []
        for terminal in [i for i in range(10)]:
            globals()['congestion_' + str(int(terminal)) + '_mean'] = random.choice(range(0,4))
            congestion_terminal_mean_list.append(eval('congestion_' + str(int(terminal)) + '_mean'))
            # for congestion_2_mean in range(4):
            #     for congestion_3_mean in range(10):

        env=coordinationEnv()
        hat_policy_kwargs = None
        protomem_policy_kwargs = None
        protomem_cfg = None
        if algorithm == "PPO_PROTOMEM":
            try:
                from robust_rl.protomem_ppo import ProtoMemInputWrapper, ProtoMemConfig

                pm_input_mode = os.environ.get("PM_INPUT_MODE", "full").strip().lower()
                pm_include_stage = os.environ.get("PM_INCLUDE_STAGE", "1").strip() == "1"
                pm_include_prev = os.environ.get("PM_INCLUDE_PREV", "1").strip() == "1"
                if pm_input_mode == "obs":
                    pm_include_stage = False
                    pm_include_prev = False

                protomem_cfg = ProtoMemConfig(
                    input_mode=pm_input_mode,
                    include_stage=pm_include_stage,
                    include_prev_action_reward=pm_include_prev,
                    stage_dim=int(os.environ.get("PM_STAGE_DIM", "2")),
                    num_prototypes=int(os.environ.get("PM_NUM_PROTOTYPES", "32")),
                    mem_dim=int(os.environ.get("PM_MEM_DIM", "64")),
                    hidden_dim=int(os.environ.get("PM_HIDDEN_DIM", "64")),
                    tau=float(os.environ.get("PM_TAU", "0.5")),
                    lambda_sparse=float(os.environ.get("PM_LAMBDA_SPARSE", "0.001")),
                    lambda_div=float(os.environ.get("PM_LAMBDA_DIV", "0.0003")),
                    lambda_stable=float(os.environ.get("PM_LAMBDA_STABLE", "0.0")),
                    lambda_aux=float(os.environ.get("PM_LAMBDA_AUX", "0.0")),
                    stable_buffer_per_phase=int(os.environ.get("PM_STABLE_BUF_PER_PHASE", "300")),
                    stable_batch_ratio=float(os.environ.get("PM_STABLE_BATCH_RATIO", "0.25")),
                    stable_warmup_updates=int(os.environ.get("PM_STABLE_WARMUP_UPDATES", "0")),
                    use_smooth=os.environ.get("PM_USE_SMOOTH", "0").strip() == "1",
                    smooth_alpha=float(os.environ.get("PM_SMOOTH_ALPHA", "0.1")),
                    smooth_train_test_consistent=os.environ.get("PM_SMOOTH_TRAIN_TEST_CONSISTENT", "1").strip() == "1",
                    mem_lr_scale=float(os.environ.get("PM_MEM_LR_SCALE", "0.5")),
                    eps=float(os.environ.get("PM_EPS", "1e-8")),
                    keep_state_across_reset=os.environ.get("PM_KEEP_STATE", "1").strip() == "1",
                    reset_prev_on_table_switch=os.environ.get("PM_RESET_PREV_ON_TABLE_SWITCH", "1").strip() == "1",
                    reset_prev_on_phase_switch=os.environ.get("PM_RESET_PREV_ON_PHASE_SWITCH", "1").strip() == "1",
                )

                def _pm_stage_onehot():
                    label = str(current_stage_label or "").lower()
                    if "insert" in label:
                        return [0.0, 1.0]
                    if "remove" in label:
                        return [1.0, 0.0]
                    return [0.0, 0.0]

                env = ProtoMemInputWrapper(
                    env,
                    include_stage=protomem_cfg.include_stage,
                    include_prev_action_reward=protomem_cfg.include_prev_action_reward,
                    stage_dim=protomem_cfg.stage_dim,
                    keep_state=protomem_cfg.keep_state_across_reset,
                    stage_getter=_pm_stage_onehot,
                    table_getter=lambda: getattr(Dynamic_ALNS_RL34959, "table_number", None),
                    phase_getter=lambda: str(current_phase_label or ""),
                    reset_prev_on_table_switch=protomem_cfg.reset_prev_on_table_switch,
                    reset_prev_on_phase_switch=protomem_cfg.reset_prev_on_phase_switch,
                )
                protomem_policy_kwargs = {"pm_config": protomem_cfg}
                print("ProtoMem wrapper enabled:", "input_mode", protomem_cfg.input_mode, "N", protomem_cfg.num_prototypes)
            except Exception as exc:
                raise ImportError("ProtoMem-PPO wrapper init failed") from exc
        use_hat = os.environ.get("RL_HAT", "0").strip() == "1"
        if use_hat and algorithm in ("PPO", "A2C", "PPO_HAT_PDI", "PPO_HAT_LSTM"):
            try:
                from robust_rl.sb3_attention import HistoryAttentionWrapper, AttentionExtractor, HATConfig
                hat_cfg = HATConfig(
                    history_len=int(os.environ.get("HAT_HISTORY_LEN", "20")),
                    embed_dim=int(os.environ.get("HAT_EMBED_DIM", "64")),
                    num_heads=int(os.environ.get("HAT_HEADS", "2")),
                    num_layers=int(os.environ.get("HAT_LAYERS", "2")),
                    dropout=float(os.environ.get("HAT_DROPOUT", "0.1")),
                    feature_dim=int(os.environ.get("HAT_FEATURE_DIM", "64")),
                )
                keep_history = os.environ.get("HAT_KEEP_HISTORY", "1").strip() == "1"
                def _hat_stage_onehot():
                    label = str(current_stage_label or "").lower()
                    if "insert" in label:
                        return [0.0, 1.0]
                    if "remove" in label:
                        return [1.0, 0.0]
                    return [0.0, 0.0]

                env = HistoryAttentionWrapper(
                    env,
                    history_len=hat_cfg.history_len,
                    keep_history=keep_history,
                    stage_dim=2,
                    stage_getter=_hat_stage_onehot,
                )
                hat_policy_kwargs = {
                    "features_extractor_class": AttentionExtractor,
                    "features_extractor_kwargs": {"config": hat_cfg},
                }
                print("HAT enabled: history_len", hat_cfg.history_len)
            except Exception as exc:
                print("HAT enable failed, fallback to MlpPolicy:", exc)
                hat_policy_kwargs = None
        if seed_val is not None:
            try:
                env.seed(seed_val)
            except Exception:
                pass

        def _lbklac_on_step(payload):
            try:
                info = payload.get("info", {}) if isinstance(payload, dict) else {}
            except Exception:
                info = {}
            row_dict = {}
            try:
                row_dict = info.get("row_dict", {}) if isinstance(info, dict) else {}
                if hasattr(row_dict, "to_dict"):
                    row_dict = row_dict.to_dict()
            except Exception:
                row_dict = {}
            reward_val = payload.get("reward") if isinstance(payload, dict) else None
            action_val = payload.get("action") if isinstance(payload, dict) else None
            extra = {}
            for key in [
                "belief_smooth_penalty",
                "value_residual",
                "delta_t",
                "policy_kl",
                "action_prob",
                "entropy",
                "loss_pi",
                "loss_v",
                "loss_kl",
                "loss_entropy",
                "bootstrap",
                "trust_region_scaled",
                "trust_region_scale",
            ]:
                if isinstance(payload, dict) and key in payload:
                    extra[key] = payload.get(key)
            if "policy_kl" not in extra:
                extra["policy_kl"] = 0.0
            if "bootstrap" not in extra:
                extra["bootstrap"] = 0
            if "trust_region_scaled" not in extra:
                extra["trust_region_scaled"] = 0
            if "trust_region_scale" not in extra:
                extra["trust_region_scale"] = 1.0
            step_id = next_step()
            log_training_row(
                "implement" if implement == 1 else "train",
                step_idx=step_id,
                reward=reward_val,
                extra=extra,
            )
            log_trace_from_row(
                row_dict,
                "receive_reward",
                action=action_val if action_val is not None else row_dict.get("action", ""),
                reward=reward_val,
                source="RL",
                extra=extra,
            )

        def _lbklac_eval(agent, env_obj, n_eval_episodes):
            rewards = []
            for _ in range(max(1, n_eval_episodes)):
                obs_eval = env_obj.reset()
                action_eval, _ = agent.predict(obs_eval, deterministic=True)
                _, reward_eval, _, _ = env_obj.step(action_eval)
                rewards.append(reward_eval)
            if not rewards:
                return -1000, -1000
            return float(np.mean(rewards)), float(np.std(rewards))

        # env.observation_space.sample()
        # env.reset()
        # from stable_baselines3.common.env_checker import check_env
        # # check_env(env, warn=True)
        # episodes = 5
        # for episode in range(1, episodes + 1):
        #     state = env.reset()
        #     done = False
        #     score = 0
        #
        #     while not done:
        #         env.render()
        #         action = env.action_space.sample()
        #         n_state, reward, done, info = env.step(action)
        #         score += reward
        #     print('Episode:{} Score:{}'.format(episode, score))
        # env.close()


        log_path = os.path.join('Training', 'Logs')
        # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
        #default n_steps = 2048
        #while True:
         #   try:
        if algorithm == 'LBKLAC':
            if not _LBKLAC_AVAILABLE or LBKLACAgent is None or LBKLACConfig is None:
                raise ImportError("LBKLAC requires torch and robust_rl.lbklac to be available.")
            lbklac_kwargs = rl_config.get_lbklac_config() if rl_config else {}
            lbklac_cfg = LBKLACConfig(**lbklac_kwargs)
            model = LBKLACAgent(
                env,
                lbklac_cfg,
                seed=seed_val,
            )
            LBKLAC_CUSTOM_LOGGING = True
        elif algorithm == 'DRCB':
            from robust_rl.drcb import DriftRobustContextualBandit
            use_hidden_meta = os.environ.get("DRCB_INCLUDE_META", "0").strip() == "1"

            def _drcb_context_getter():
                ctx = {
                    "phase": "implement" if implement == 1 else "train",
                    "stage": current_stage_label,
                    "severity": globals().get("severity_level", ""),
                    "table_number": getattr(Dynamic_ALNS_RL34959, "table_number", ""),
                    "vehicle": globals().get("vehicle", ""),
                    "request": globals().get("request", ""),
                }
                if use_hidden_meta:
                    # Optional only: may leak privileged info in strict hidden-mode evaluation.
                    ctx["gt_mean"] = current_gt_mean
                    ctx["phase_label"] = current_phase_label
                return ctx

            model = DriftRobustContextualBandit(
                env,
                seed=seed_val,
                decay=float(os.environ.get("DRCB_DECAY", "0.995")),
                ridge=float(os.environ.get("DRCB_RIDGE", "1.0")),
                ucb_alpha=float(os.environ.get("DRCB_UCB_ALPHA", "0.4")),
                risk_alpha=float(os.environ.get("DRCB_RISK_ALPHA", "0.1")),
                drift_alpha=float(os.environ.get("DRCB_DRIFT_ALPHA", "0.05")),
                drift_scale=float(os.environ.get("DRCB_DRIFT_SCALE", "1.5")),
                drift_cap=float(os.environ.get("DRCB_DRIFT_CAP", "2.0")),
                warm_start_min_pulls=int(os.environ.get("DRCB_WARM_START_MIN_PULLS", "8")),
                impl_eps=float(os.environ.get("DRCB_IMPL_EPS", "0.08")),
                include_entity_ids=os.environ.get("DRCB_INCLUDE_ENTITY_IDS", "0").strip() == "1",
                use_regime_buckets=os.environ.get("DRCB_USE_REGIME_BUCKETS", "1").strip() == "1",
                use_context_features=os.environ.get("DRCB_USE_CONTEXT_FEATS", "1").strip() == "1",
                context_getter=_drcb_context_getter,
            )
        elif algorithm == 'DQN':
            if not _SB3_AVAILABLE or DQN is None:
                raise ImportError("stable_baselines3 is required for DQN. Please install stable-baselines3 + torch.")
            model = DQN('MlpPolicy', env, verbose=1, learning_starts=10, device='cpu', seed=seed_val)
        elif algorithm in {'CQL_DQN', 'CQL'}:
            if not _CQL_AVAILABLE or DiscreteCQLAgent is None or CQLConfig is None:
                raise ImportError("CQL_DQN requires robust_rl.cql_dqn + torch + stable-baselines3.")
            cql_cfg = CQLConfig(
                learning_rate=float(os.environ.get("CQL_LR", "0.001")),
                buffer_size=int(os.environ.get("CQL_BUFFER_SIZE", "50000")),
                learning_starts=int(os.environ.get("CQL_LEARNING_STARTS", "200")),
                batch_size=int(os.environ.get("CQL_BATCH_SIZE", "64")),
                train_freq=int(os.environ.get("CQL_TRAIN_FREQ", "4")),
                gradient_steps=int(os.environ.get("CQL_GRAD_STEPS", "1")),
                target_update_interval=int(os.environ.get("CQL_TARGET_UPDATE", "500")),
                exploration_fraction=float(os.environ.get("CQL_EXPL_FRACTION", "0.1")),
                exploration_initial_eps=float(os.environ.get("CQL_EXPL_INIT", "1.0")),
                exploration_final_eps=float(os.environ.get("CQL_EXPL_FINAL", "0.02")),
                max_grad_norm=float(os.environ.get("CQL_MAX_GRAD_NORM", "10.0")),
                cql_alpha=float(os.environ.get("CQL_ALPHA", "1.0")),
                cql_temp=float(os.environ.get("CQL_TEMP", "1.0")),
                device=os.environ.get("CQL_DEVICE", "cpu"),
            )
            model = DiscreteCQLAgent(
                env,
                config=cql_cfg,
                seed=seed_val,
            )
        elif algorithm == 'QRDQN_CVAR':
            if not _QRDQN_AVAILABLE or SB3_QRDQN is None:
                raise ImportError("sb3-contrib with QRDQN is required for QRDQN_CVAR.")
            from robust_rl.qrdqn_cvar import QRDQNCVaRAgent, QRDQNCVaRConfig

            qcfg = QRDQNCVaRConfig(
                cvar_alpha=float(os.environ.get("QRDQN_CVAR_ALPHA", "0.25")),
                n_quantiles=int(os.environ.get("QRDQN_N_QUANTILES", "64")),
                learning_rate=float(os.environ.get("QRDQN_LR", "0.001")),
                buffer_size=int(os.environ.get("QRDQN_BUFFER_SIZE", "50000")),
                learning_starts=int(os.environ.get("QRDQN_LEARNING_STARTS", "200")),
                batch_size=int(os.environ.get("QRDQN_BATCH_SIZE", "64")),
                train_freq=int(os.environ.get("QRDQN_TRAIN_FREQ", "4")),
                gradient_steps=int(os.environ.get("QRDQN_GRAD_STEPS", "1")),
                target_update_interval=int(os.environ.get("QRDQN_TARGET_UPDATE", "500")),
                exploration_fraction=float(os.environ.get("QRDQN_EXPL_FRACTION", "0.1")),
                exploration_initial_eps=float(os.environ.get("QRDQN_EXPL_INIT", "1.0")),
                exploration_final_eps=float(os.environ.get("QRDQN_EXPL_FINAL", "0.02")),
                max_grad_norm=float(os.environ.get("QRDQN_MAX_GRAD_NORM", "10.0")),
                device=os.environ.get("QRDQN_DEVICE", "cpu"),
            )
            model = QRDQNCVaRAgent(
                env,
                config=qcfg,
                seed=seed_val,
            )
        elif algorithm == 'BE_CVAR_DQN':
            from robust_rl.be_cvar_dqn import BeliefEnsembleCvaRDQN, BECVaRDQNConfig

            def _be_context_getter():
                return {
                    "phase": "implement" if implement == 1 else "train",
                    "stage": current_stage_label,
                    "severity": globals().get("severity_level", ""),
                    "table_number": getattr(Dynamic_ALNS_RL34959, "table_number", ""),
                }

            be_cfg = BECVaRDQNConfig(
                history_len=int(os.environ.get("BE_HISTORY_LEN", "20")),
                belief_dim=int(os.environ.get("BE_BELIEF_DIM", "16")),
                hidden_dim=int(os.environ.get("BE_HIDDEN_DIM", "64")),
                n_heads=int(os.environ.get("BE_N_HEADS", "3")),
                n_quantiles=int(os.environ.get("BE_N_QUANTILES", "51")),
                gamma=float(os.environ.get("BE_GAMMA", "0.99")),
                learning_rate=float(os.environ.get("BE_LR", "0.0003")),
                batch_size=int(os.environ.get("BE_BATCH_SIZE", "64")),
                buffer_size=int(os.environ.get("BE_BUFFER_SIZE", "50000")),
                learning_starts=int(os.environ.get("BE_LEARNING_STARTS", "200")),
                train_freq=int(os.environ.get("BE_TRAIN_FREQ", "1")),
                gradient_steps=int(os.environ.get("BE_GRAD_STEPS", "1")),
                target_update_interval=int(os.environ.get("BE_TARGET_UPDATE", "500")),
                tau=float(os.environ.get("BE_TAU", "1.0")),
                max_grad_norm=float(os.environ.get("BE_MAX_GRAD_NORM", "10.0")),
                cvar_alpha=float(os.environ.get("BE_CVAR_ALPHA", "0.2")),
                uncertainty_beta=float(os.environ.get("BE_UNCERTAINTY_BETA", "0.2")),
                loss_ens_coef=float(os.environ.get("BE_LOSS_ENS", "0.01")),
                loss_belief_coef=float(os.environ.get("BE_LOSS_BELIEF", "0.0001")),
                exploration_initial_eps=float(os.environ.get("BE_EXPL_INIT", "1.0")),
                exploration_final_eps=float(os.environ.get("BE_EXPL_FINAL", "0.05")),
                exploration_fraction=float(os.environ.get("BE_EXPL_FRACTION", "0.3")),
                impl_eps=float(os.environ.get("BE_IMPL_EPS", "0.05")),
                device=os.environ.get("BE_DEVICE", "cpu"),
            )
            model = BeliefEnsembleCvaRDQN(
                env,
                config=be_cfg,
                seed=seed_val,
                context_getter=_be_context_getter,
            )
        elif algorithm in ('PPO_LSTM', 'REC_PPO', 'RECURRENTPPO', 'PPO_HAT_LSTM'):
            if not _SB3_CONTRIB_AVAILABLE or RecurrentPPO is None:
                raise ImportError("sb3-contrib is required for RecurrentPPO. Please install sb3-contrib==2.3.0.")
            lstm_hidden_size = int(os.environ.get("LSTM_HIDDEN_SIZE", "64"))
            n_lstm_layers = int(os.environ.get("LSTM_LAYERS", "1"))
            shared_lstm = os.environ.get("LSTM_SHARED", "1").strip() == "1"
            enable_critic_lstm = os.environ.get("LSTM_CRITIC", "1").strip() == "1"
            # sb3-contrib constraint: choose exactly one of (shared_lstm) or (separate critic LSTM) or (no critic LSTM).
            # If shared_lstm is True, enable_critic_lstm must be False.
            if shared_lstm and enable_critic_lstm:
                enable_critic_lstm = False
            policy_kwargs = {
                "lstm_hidden_size": lstm_hidden_size,
                "n_lstm_layers": n_lstm_layers,
                "shared_lstm": shared_lstm,
                "enable_critic_lstm": enable_critic_lstm,
            }
            # Optional: LSTM after existing encoder (e.g., HAT features_extractor).
            if use_hat and os.environ.get("LSTM_AFTER_ENCODER", "0").strip() == "1":
                policy_kwargs.update(hat_policy_kwargs or {})
            n_steps = int(os.environ.get("LSTM_N_STEPS", "10"))
            batch_size = int(os.environ.get("LSTM_BATCH_SIZE", str(n_steps)))
            n_epochs = int(os.environ.get("LSTM_N_EPOCHS", "5"))
            learning_rate = float(os.environ.get("LSTM_LR", "0.0003"))
            gamma = float(os.environ.get("LSTM_GAMMA", "0.99"))
            gae_lambda = float(os.environ.get("LSTM_GAE_LAMBDA", "0.95"))
            clip_range = float(os.environ.get("LSTM_CLIP_RANGE", "0.2"))
            ent_coef = float(os.environ.get("LSTM_ENT_COEF", "0.0"))
            vf_coef = float(os.environ.get("LSTM_VF_COEF", "0.5"))
            model = RecurrentPPO(
                "MlpLstmPolicy",
                env,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                verbose=1,
                device='cpu',
                seed=seed_val,
                policy_kwargs=policy_kwargs,
            )
        elif algorithm == 'PPO_HAT_PDI':
            if not _SB3_AVAILABLE or PPO is None:
                raise ImportError("stable_baselines3 is required for PPO. Please install stable-baselines3 + torch.")
            from robust_rl.hat_ppo_pdi import HATPdiPolicy, HATPdiPPO, PDIConfig

            pdi_cfg = PDIConfig(
                kappa=float(os.environ.get("PDI_KAPPA", "1.0")),
                temp_coef=float(os.environ.get("PDI_TEMP_COEF", "0.0")),
                lambda_future=float(os.environ.get("PDI_LAMBDA_FUTURE", "0.2")),
                lambda_teach=float(os.environ.get("PDI_LAMBDA_TEACH", "0.1")),
                lambda_actfail=float(os.environ.get("PDI_LAMBDA_ACTFAIL", "0.2")),
                future_h=int(os.environ.get("PDI_FUTURE_H", "5")),
                gt_mean_norm=float(os.environ.get("PDI_GT_MEAN_NORM", "100.0")),
                phase_classes=int(os.environ.get("PDI_PHASE_CLASSES", "0")),
                log_window=int(os.environ.get("PDI_LOG_WINDOW", "50")),
            )
            policy_kwargs = dict(hat_policy_kwargs or {})
            policy_kwargs["pdi_config"] = pdi_cfg
            model = HATPdiPPO(
                HATPdiPolicy,
                env,
                n_steps=10,
                verbose=1,
                device='cpu',
                seed=seed_val,
                policy_kwargs=policy_kwargs,
                pdi_config=pdi_cfg,
            )
        elif algorithm == 'PPO_PROTOMEM':
            if not _SB3_AVAILABLE or PPO is None:
                raise ImportError("stable_baselines3 is required for PPO. Please install stable-baselines3 + torch.")
            from robust_rl.protomem_ppo import ProtoMemPolicy, ProtoMemPPO, ProtoMemConfig

            pm_cfg = protomem_cfg if isinstance(protomem_cfg, ProtoMemConfig) else ProtoMemConfig()
            policy_kwargs = dict(protomem_policy_kwargs or {})
            pm_n_steps = int(os.environ.get("PM_N_STEPS", "10"))
            pm_batch_size = int(os.environ.get("PM_BATCH_SIZE", str(pm_n_steps)))
            pm_n_epochs = int(os.environ.get("PM_N_EPOCHS", "5"))
            pm_lr = float(os.environ.get("PM_LR", "0.0003"))
            pm_gamma = float(os.environ.get("PM_GAMMA", "0.99"))
            pm_gae = float(os.environ.get("PM_GAE_LAMBDA", "0.95"))
            pm_clip = float(os.environ.get("PM_CLIP_RANGE", "0.2"))
            pm_ent = float(os.environ.get("PM_ENT_COEF", "0.0"))
            pm_vf = float(os.environ.get("PM_VF_COEF", "0.5"))
            pm_device = os.environ.get("RL_DEVICE", "cpu")
            model = ProtoMemPPO(
                ProtoMemPolicy,
                env,
                n_steps=pm_n_steps,
                batch_size=pm_batch_size,
                n_epochs=pm_n_epochs,
                learning_rate=pm_lr,
                gamma=pm_gamma,
                gae_lambda=pm_gae,
                clip_range=pm_clip,
                ent_coef=pm_ent,
                vf_coef=pm_vf,
                verbose=1,
                device=pm_device,
                seed=seed_val,
                policy_kwargs=policy_kwargs,
                pm_config=pm_cfg,
            )
            print("ProtoMem-PPO enabled:", "device", pm_device, "N", pm_cfg.num_prototypes, "d", pm_cfg.mem_dim)
        elif algorithm == 'PPO_NEW':
            if not _SB3_AVAILABLE or PPO is None:
                raise ImportError("stable_baselines3 is required for PPO_NEW. Please install stable-baselines3 + torch.")
            from robust_rl.ppo_new import build_model as build_ppo_new_model
            tcr_kwargs = {}
            ppo_new_extra_kwargs = {}

            ppo_new_ent_coef_raw = str(os.environ.get("RL_PPO_NEW_ENT_COEF", "")).strip()
            if ppo_new_ent_coef_raw != "":
                try:
                    ppo_new_extra_kwargs["ent_coef"] = float(ppo_new_ent_coef_raw)
                except Exception:
                    print(f"[RL][WARN] invalid RL_PPO_NEW_ENT_COEF='{ppo_new_ent_coef_raw}', ignored")

            ppo_new_lr_raw = str(os.environ.get("RL_PPO_NEW_LR", "")).strip()
            if ppo_new_lr_raw != "":
                try:
                    ppo_new_extra_kwargs["learning_rate"] = float(ppo_new_lr_raw)
                except Exception:
                    print(f"[RL][WARN] invalid RL_PPO_NEW_LR='{ppo_new_lr_raw}', ignored")

            tcr_family_versions = (
                "v7.3_tcrppo", "v73_tcrppo", "v7_3_tcrppo",
                "v7.4_tcrv3", "v74_tcrv3", "v7_4_tcrv3",
                "v8_a", "v8-a", "v8a", "pponew_v8_a", "pponewv8_a",
                "v9_a", "v9-a", "v9a", "pponew_v9_a", "pponewv9_a",
                "v8_a2", "v8-a2", "v8a2", "pponew_v8_a2", "pponewv8_a2",
                "v9_a2", "v9-a2", "v9a2", "pponew_v9_a2", "pponewv9_a2",
                "v10_a", "v10-a", "v10a", "pponew_v10_a", "pponewv10_a",
                "v10_b", "v10-b", "v10b", "pponew_v10_b", "pponewv10_b",
                "v10_c", "v10-c", "v10c", "pponew_v10_c", "pponewv10_c",
                "v10_d", "v10-d", "v10d", "pponew_v10_d", "pponewv10_d",
                "v10_e", "v10-e", "v10e", "pponew_v10_e", "pponewv10_e",
            )
            v8_family_versions = (
                "v8_a", "v8-a", "v8a", "pponew_v8_a", "pponewv8_a",
                "v9_a", "v9-a", "v9a", "pponew_v9_a", "pponewv9_a",
            )
            v9_family_versions = ("v9_a", "v9-a", "v9a", "pponew_v9_a", "pponewv9_a")
            v8a2_family_versions = (
                "v8_a2", "v8-a2", "v8a2", "pponew_v8_a2", "pponewv8_a2",
                "v9_a2", "v9-a2", "v9a2", "pponew_v9_a2", "pponewv9_a2",
                "v10_a", "v10-a", "v10a", "pponew_v10_a", "pponewv10_a",
                "v10_b", "v10-b", "v10b", "pponew_v10_b", "pponewv10_b",
                "v10_c", "v10-c", "v10c", "pponew_v10_c", "pponewv10_c",
                "v10_d", "v10-d", "v10d", "pponew_v10_d", "pponewv10_d",
                "v10_e", "v10-e", "v10e", "pponew_v10_e", "pponewv10_e",
            )
            v9a2_family_versions = ("v9_a2", "v9-a2", "v9a2", "pponew_v9_a2", "pponewv9_a2")
            if ALGO_VERSION in tcr_family_versions:
                tcr_kwargs = {
                    "tcr_enable": os.environ.get("RL_TCR_ENABLE", "1").strip() == "1",
                    "tcr_action1_rate_threshold": float(os.environ.get("RL_TCR_TAU_RHO", "0.08")),
                    "tcr_reward_gap_threshold": float(os.environ.get("RL_TCR_TAU_R", "0.05")),
                    "tcr_min_action1_samples": int(os.environ.get("RL_TCR_MIN_A1", "12")),
                    "tcr_min_action0_samples": int(os.environ.get("RL_TCR_MIN_A0", "12")),
                    "tcr_min_group_steps": int(os.environ.get("RL_TCR_MIN_GROUP_STEPS", "32")),
                    "tcr_trigger_window_steps": int(os.environ.get("RL_TCR_TRIGGER_WINDOW", "400")),
                    "tcr_a1_reward_quantile": float(os.environ.get("RL_TCR_A1_QUANTILE", "0.70")),
                    "tcr_reward_margin_over_a0": float(os.environ.get("RL_TCR_MARGIN_A0", "0.0")),
                    "tcr_buffer_size_per_group": int(os.environ.get("RL_TCR_BUFFER_PER_GROUP", "512")),
                    "tcr_aux_coef": float(os.environ.get("RL_TCR_BETA", "0.03")),
                    "tcr_aux_batch_size": int(os.environ.get("RL_TCR_AUX_BATCH", "64")),
                    "tcr_teacher_mode": os.environ.get("RL_TCR_TEACHER_MODE", "none").strip().lower(),
                    "tcr_teacher_action1_logit_bias": float(os.environ.get("RL_TCR_TEACHER_LOGIT_BIAS", "0.35")),
                }
                if ALGO_VERSION in v8_family_versions:
                    tcr_kwargs.update(
                        {
                            "v8_quality_temp": float(os.environ.get("RL_V8_QUALITY_TEMP", "0.15")),
                            "v8_support_power": float(os.environ.get("RL_V8_SUPPORT_POWER", "0.50")),
                            "v8_novelty_power": float(os.environ.get("RL_V8_NOVELTY_POWER", "1.00")),
                            "v8_min_keep_weight": float(os.environ.get("RL_V8_MIN_KEEP_WEIGHT", "0.15")),
                            "v8_max_keep_weight": float(os.environ.get("RL_V8_MAX_KEEP_WEIGHT", "4.00")),
                            "v8_kl_guard_coef": float(os.environ.get("RL_V8_KL_GUARD_COEF", "0.02")),
                            "v8_ref_mix": float(os.environ.get("RL_V8_REF_MIX", "0.50")),
                        }
                    )
                if ALGO_VERSION in v9_family_versions:
                    tcr_kwargs.update(
                        {
                            "v9_kappa_base": float(os.environ.get("RL_V9_KAPPA_BASE", "1.00")),
                            "v9_kappa_slope": float(os.environ.get("RL_V9_KAPPA_SLOPE", "2.00")),
                            "v9_kappa_min": float(os.environ.get("RL_V9_KAPPA_MIN", "0.50")),
                            "v9_kappa_max": float(os.environ.get("RL_V9_KAPPA_MAX", "3.00")),
                            "v9_gap_ema_decay": float(os.environ.get("RL_V9_GAP_EMA_DECAY", "0.80")),
                        }
                    )
                if ALGO_VERSION in v8a2_family_versions:
                    tcr_kwargs.update(
                        {
                            "v8a2_ratio_threshold": float(os.environ.get("RL_A2_TAU_RHO", os.environ.get("RL_TCR_TAU_RHO", "0.10"))),
                            "v8a2_min_group_steps": int(os.environ.get("RL_A2_MIN_GROUP_STEPS", "1")),
                            "v8a2_gen_budget_scale": float(os.environ.get("RL_A2_GEN_BUDGET_SCALE", "2.0")),
                            "v8a2_gen_budget_max_per_group": int(os.environ.get("RL_A2_GEN_MAX_PER_GROUP", "128")),
                            "v8a2_gen_min_total_per_rollout": int(os.environ.get("RL_A2_GEN_MIN_TOTAL", "50")),
                            "v8a2_gen_min_per_group": int(os.environ.get("RL_A2_GEN_MIN_PER_GROUP", "0")),
                            "v8a2_gen_mix_alpha": float(os.environ.get("RL_A2_GEN_MIX_ALPHA", "2.0")),
                            "v8a2_gen_noise_std": float(os.environ.get("RL_A2_GEN_NOISE_STD", "0.01")),
                            "v8a2_min_keep_weight": float(os.environ.get("RL_A2_MIN_KEEP_WEIGHT", "0.15")),
                            "v8a2_max_keep_weight": float(os.environ.get("RL_A2_MAX_KEEP_WEIGHT", "4.00")),
                            "v8a2_kl_guard_coef": float(os.environ.get("RL_A2_KL_GUARD_COEF", "0.02")),
                            "v8a2_ref_mix": float(os.environ.get("RL_A2_REF_MIX", "0.50")),
                        }
                    )
                if ALGO_VERSION in v9a2_family_versions:
                    tcr_kwargs.update(
                        {
                            "v9a2_tau_entropy_eta": float(os.environ.get("RL_V9A2_TAU_ENTROPY_ETA", "0.05")),
                            "v9a2_shortage_ema_decay": float(os.environ.get("RL_V9A2_SHORTAGE_EMA_DECAY", "0.80")),
                            "v9a2_kappa_base": float(os.environ.get("RL_V9A2_KAPPA_BASE", "1.00")),
                            "v9a2_kappa_slope": float(os.environ.get("RL_V9A2_KAPPA_SLOPE", "2.00")),
                            "v9a2_kappa_min": float(os.environ.get("RL_V9A2_KAPPA_MIN", "0.50")),
                            "v9a2_kappa_max": float(os.environ.get("RL_V9A2_KAPPA_MAX", "3.00")),
                        }
                    )
            model = build_ppo_new_model(
                env=env,
                seed=seed_val,
                device='cpu',
                algo_version=ALGO_VERSION,
                policy='MlpPolicy',
                n_steps=10,
                verbose=1,
                policy_kwargs=hat_policy_kwargs,
                **ppo_new_extra_kwargs,
                **tcr_kwargs,
            )
        elif algorithm == 'PPO':
            if not _SB3_AVAILABLE or PPO is None:
                raise ImportError("stable_baselines3 is required for PPO. Please install stable-baselines3 + torch.")
            use_moe = os.environ.get("RL_MOE", "0").strip() == "1"
            if use_hat and use_moe:
                # HAT+MoE: keep SB3 training loop; swap only policy head.
                from robust_rl.sb3_attention import HATMoEActorCriticPolicy, MoEConfig, MoEPPO

                moe_cfg = MoEConfig(
                    num_experts=int(os.environ.get("MOE_K", "2")),
                    expert_hidden_dim=int(os.environ.get("MOE_HIDDEN", "64")),
                    expert_layers=int(os.environ.get("MOE_LAYERS", "1")),
                    gate_hidden_dim=int(os.environ.get("MOE_GATE_HIDDEN", "32")),
                    stage_dim=int(os.environ.get("MOE_STAGE_DIM", "2")),
                    gate_entropy_coef=float(os.environ.get("MOE_GATE_ENT_COEF", "0.01")),
                    load_balance_coef=float(os.environ.get("MOE_LB_COEF", "0.01")),
                    # NOTE: div coef lives on the algorithm wrapper (loss term); policy logs it via out["div"].
                    log_window=int(os.environ.get("MOE_LOG_WINDOW", "50")),
                    hard_inference=os.environ.get("MOE_HARD_INFER", "0").strip() == "1",
                )
                policy_kwargs = dict(hat_policy_kwargs or {})
                policy_kwargs["moe_config"] = moe_cfg
                PPOCls = MoEPPO.wrap(PPO)
                model = PPOCls(
                    HATMoEActorCriticPolicy,
                    env,
                    n_steps=10,
                    verbose=1,
                    device="cpu",
                    seed=seed_val,
                    policy_kwargs=policy_kwargs,
                    moe_gate_entropy_coef=float(moe_cfg.gate_entropy_coef),
                    moe_load_balance_coef=float(moe_cfg.load_balance_coef),
                    moe_div_coef=float(os.environ.get("MOE_DIV_COEF", "0.005")),
                )
                print("MoE enabled: K", moe_cfg.num_experts, "hard_infer", int(moe_cfg.hard_inference))
            else:
                model = PPO('MlpPolicy', env, n_steps=10, verbose=1, device='cpu', seed=seed_val, policy_kwargs=hat_policy_kwargs)
        elif algorithm == 'A2C':
            if not _SB3_AVAILABLE or A2C is None:
                raise ImportError("stable_baselines3 is required for A2C. Please install stable-baselines3 + torch.")
            use_moe = os.environ.get("RL_MOE", "0").strip() == "1"
            if use_hat and use_moe:
                from robust_rl.sb3_attention import HATMoEActorCriticPolicy, MoEConfig, MoEA2C

                moe_cfg = MoEConfig(
                    num_experts=int(os.environ.get("MOE_K", "2")),
                    expert_hidden_dim=int(os.environ.get("MOE_HIDDEN", "64")),
                    expert_layers=int(os.environ.get("MOE_LAYERS", "1")),
                    gate_hidden_dim=int(os.environ.get("MOE_GATE_HIDDEN", "32")),
                    stage_dim=int(os.environ.get("MOE_STAGE_DIM", "2")),
                    gate_entropy_coef=float(os.environ.get("MOE_GATE_ENT_COEF", "0.01")),
                    load_balance_coef=float(os.environ.get("MOE_LB_COEF", "0.01")),
                    log_window=int(os.environ.get("MOE_LOG_WINDOW", "50")),
                    hard_inference=os.environ.get("MOE_HARD_INFER", "0").strip() == "1",
                )
                policy_kwargs = dict(hat_policy_kwargs or {})
                policy_kwargs["moe_config"] = moe_cfg
                A2CCls = MoEA2C.wrap(A2C)
                model = A2CCls(
                    HATMoEActorCriticPolicy,
                    env,
                    n_steps=10,
                    verbose=1,
                    device="cpu",
                    seed=seed_val,
                    policy_kwargs=policy_kwargs,
                    moe_gate_entropy_coef=float(moe_cfg.gate_entropy_coef),
                    moe_load_balance_coef=float(moe_cfg.load_balance_coef),
                    moe_div_coef=float(os.environ.get("MOE_DIV_COEF", "0.005")),
                )
                print("MoE enabled: K", moe_cfg.num_experts, "hard_infer", int(moe_cfg.hard_inference))
            else:
                model = A2C('MlpPolicy', env, n_steps=10, verbose=1, device='cpu', seed=seed_val, policy_kwargs=hat_policy_kwargs)
        else:
            if not _SB3_AVAILABLE:
                raise ImportError("stable_baselines3 is required for this algorithm. Please install stable-baselines3 + torch.")
            model = eval(algorithm + "('MlpPolicy', env, n_steps=10, verbose=1, device='cpu')")
            #break
           # except:
            #    continue
        maybe_loaded_model = _maybe_load_model_checkpoint(model)
        if maybe_loaded_model is not None:
            model = maybe_loaded_model
        lstm_callback = None
        if USE_LSTM and BaseCallback is not None:
            try:
                lstm_callback = LstmStatsCallback()
            except Exception:
                lstm_callback = None
        tcr_callback = None
        if BaseCallback is not None and hasattr(model, "tcr_consume_rollout"):
            try:
                tcr_callback = TCRRolloutStatsCallback()
            except Exception:
                tcr_callback = None
        learn_callback = None
        if lstm_callback is not None and tcr_callback is not None:
            if CallbackList is not None:
                learn_callback = CallbackList([lstm_callback, tcr_callback])
            else:
                learn_callback = lstm_callback
        elif lstm_callback is not None:
            learn_callback = lstm_callback
        elif tcr_callback is not None:
            learn_callback = tcr_callback

        def _reset_model_inference_state(reason=""):
            try:
                if model is not None and hasattr(model, "policy") and hasattr(model.policy, "reset_inference_state"):
                    model.policy.reset_inference_state(reason=reason)
            except Exception:
                pass

        _reset_model_inference_state("run_start")
        # #########imitation learning both baseline and baseline3 have bugs and can't be solved.
        # # baseline: ValueError: Cannot feed value of shape (1, 1, 11) for Tensor 'deepq/input/Ob:0', which has shape '(?, 11)'
        # #           and have error even when run official example
        # # baseline3's imitation learning need a library only can be used in Linux
        # generate_expert_traj(model, 'congestion', env, n_timesteps=int(1e5), n_episodes=10)
        # # Using only one expert trajectory
        # # you can specify `traj_limitation=-1` for using the whole dataset
        # dataset = ExpertDataset(expert_path='congestion.npz',
        #                         traj_limitation=1, batch_size=128)
        # # Pretrain the PPO2 model
        # model.pretrain(dataset, n_epochs=iteration_multiply)
        ############
        # info["TimeLimit.truncated"] = True
        time_s = 0
        all_average_reward = []
        all_deviation = []
        timestamps = []
        # for time_stamp in range(10000):
        #     model.learn(total_timesteps=10)
        start_time = timeit.default_timer()
        non_stationary = 0
        iteration_multiply = 1
        total_timesteps2 = iteration_numbers_unit * iteration_multiply
        sucess_times = 0
        curriculum_converged = 0
        curriculum_last_avg_reward = ""
        state_action_reward_collect = np.array(np.empty(shape=(0, 9)))
        table_number_collect = {}
        for number_of_learn_evaluate_loops in range(1000000000):
            if STAGE_MODE == "eval_only":
                break
            if implement == 1:
                break
            if STAGE_MODE == "train_only" and os.path.exists(get_stop_flag_path()):
                _maybe_save_model_checkpoint(model)
                return
            if algorithm == 'LBKLAC':
                model.learn(total_timesteps=total_timesteps2, on_step=_lbklac_on_step)
            else:
                if learn_callback is not None:
                    model.learn(total_timesteps=total_timesteps2, callback=learn_callback)
                else:
                    model.learn(total_timesteps=total_timesteps2)
            training_time = timeit.default_timer() - start_time
            extra = {}
            try:
                if hasattr(model, "last_pdi_losses"):
                    extra.update(dict(getattr(model, "last_pdi_losses", {}) or {}))
            except Exception:
                pass
            try:
                if hasattr(model, "last_protomem_losses"):
                    extra.update(dict(getattr(model, "last_protomem_losses", {}) or {}))
            except Exception:
                pass
            try:
                if hasattr(model, "last_v5_metrics"):
                    extra.update(dict(getattr(model, "last_v5_metrics", {}) or {}))
            except Exception:
                pass
            try:
                if hasattr(model, "last_v6_metrics"):
                    extra.update(dict(getattr(model, "last_v6_metrics", {}) or {}))
            except Exception:
                pass
            try:
                if hasattr(model, "last_cql_metrics"):
                    extra.update(dict(getattr(model, "last_cql_metrics", {}) or {}))
            except Exception:
                pass
            try:
                if hasattr(model, "last_tcr_metrics"):
                    extra.update(dict(getattr(model, "last_tcr_metrics", {}) or {}))
            except Exception:
                pass
            try:
                if hasattr(model, "last_v8_metrics"):
                    extra.update(dict(getattr(model, "last_v8_metrics", {}) or {}))
            except Exception:
                pass
            try:
                if hasattr(model, "last_v9_metrics"):
                    extra.update(dict(getattr(model, "last_v9_metrics", {}) or {}))
            except Exception:
                pass
            try:
                if hasattr(model, "last_v8a2_metrics"):
                    extra.update(dict(getattr(model, "last_v8a2_metrics", {}) or {}))
            except Exception:
                pass
            try:
                if hasattr(model, "last_v9a2_metrics"):
                    extra.update(dict(getattr(model, "last_v9a2_metrics", {}) or {}))
            except Exception:
                pass
            _maybe_console_phase2(extra, phase="train")
            if not extra:
                extra = None
            log_training_row("train", step_idx=global_step, training_time=training_time, extra=extra)
            try:
                with open(Intermodal_ALNS34959.path + "/finite_horizon_length" + str(
                        episode_length) + "_delay_reward_time_dependent" + str(
                    time_dependent) + "_tenterminal_" + algorithm + "_" + mode + "_" + str(
                    iteration_multiply) + "multiply" + 'training_time.txt', 'w') as f:
                    f.write(f"{str(training_time)}\n")
            except:
                pass
            #model.save('congestion_terminal_mean_list' + '_20220220congestion_stochastic100000')
            #load
            # model = PPO.load("PPO2021113a0coordination")
            # iteration_times += 1
            # if iteration_times > 5:
            evaluate = 1

            state_action_reward_collect_for_evaluate = {}

            list_of_collect_index = range(len(state_action_reward_collect))
            for collect_index in list_of_collect_index:
                chosen_pair = state_action_reward_collect[collect_index]
                state = get_state(chosen_pair,table_number_collect[collect_index][0],table_number_collect[collect_index][1],table_number_collect[collect_index][2],table_number_collect[collect_index][3])
                state_key = tuple(state)
                if state_key not in state_action_reward_collect_for_evaluate.keys():
                    state_action_reward_collect_for_evaluate[state_key] = {}
                state_action_reward_collect_for_evaluate[state_key][chosen_pair[7]] = chosen_pair[8]
            delete_keys = []
            for state_key in state_action_reward_collect_for_evaluate.keys():
                if len(state_action_reward_collect_for_evaluate[state_key]) < 2:
                    delete_keys.append(state_key)
            for state_key in delete_keys:
                del state_action_reward_collect_for_evaluate[state_key]
            if state_action_reward_collect_for_evaluate == {}:
                average_reward, deviation = -1000, -1000
            else:
                state_keys = list(state_action_reward_collect_for_evaluate.keys())
                number_of_state_key = 0
                for _ in range(1):
                    if algorithm == 'LBKLAC':
                        average_reward, deviation = _lbklac_eval(model, env, iteration_numbers_unit)
                    elif USE_LSTM:
                        average_reward, deviation = evaluate_recurrent_policy(model, env, n_eval_episodes=iteration_numbers_unit)
                    elif sb3_evaluate_policy is not None:
                        average_reward, deviation = sb3_evaluate_policy(model, env, n_eval_episodes=iteration_numbers_unit, render=False)
                    else:
                        average_reward, deviation = -1000, -1000
                    print('congestion_terminal_mean_list', congestion_terminal_mean_list, average_reward, deviation)
            rolling_avg = sum(recent_rewards) / len(recent_rewards) if recent_rewards else -1000
            eval_extra = {}
            try:
                if hasattr(model, "last_v5_metrics"):
                    eval_extra.update(dict(getattr(model, "last_v5_metrics", {}) or {}))
            except Exception:
                pass
            try:
                if hasattr(model, "last_v6_metrics"):
                    eval_extra.update(dict(getattr(model, "last_v6_metrics", {}) or {}))
            except Exception:
                pass
            try:
                if hasattr(model, "last_cql_metrics"):
                    eval_extra.update(dict(getattr(model, "last_cql_metrics", {}) or {}))
            except Exception:
                pass
            try:
                if hasattr(model, "last_tcr_metrics"):
                    eval_extra.update(dict(getattr(model, "last_tcr_metrics", {}) or {}))
            except Exception:
                pass
            try:
                if hasattr(model, "last_v8_metrics"):
                    eval_extra.update(dict(getattr(model, "last_v8_metrics", {}) or {}))
            except Exception:
                pass
            try:
                if hasattr(model, "last_v9_metrics"):
                    eval_extra.update(dict(getattr(model, "last_v9_metrics", {}) or {}))
            except Exception:
                pass
            try:
                if hasattr(model, "last_v8a2_metrics"):
                    eval_extra.update(dict(getattr(model, "last_v8a2_metrics", {}) or {}))
            except Exception:
                pass
            try:
                if hasattr(model, "last_v9a2_metrics"):
                    eval_extra.update(dict(getattr(model, "last_v9a2_metrics", {}) or {}))
            except Exception:
                pass
            _maybe_console_phase2(eval_extra, phase="eval")
            if not eval_extra:
                eval_extra = None
            log_training_row(
                "eval",
                step_idx=global_step,
                avg_reward=average_reward,
                std_reward=deviation,
                rolling_avg=rolling_avg,
                recent_count=len(recent_rewards),
                extra=eval_extra,
            )
            print('evaluation', 'average_reward', average_reward, 'deviation', deviation)# sys.exit('stop_it_in_testing')
            # Curriculum convergence: used for jumps only
            curriculum_last_avg_reward = rolling_avg
            threshold = CURRICULUM_REWARD_THRESHOLD
            if rolling_avg >= threshold:
                sucess_times += 1
            else:
                sucess_times = 0
            curriculum_converged = 1 if sucess_times >= CURRICULUM_SUCCESS_REQUIRED else 0

            wait_training_finish_last_iteration = 0
            evaluate = 0
            # record_results = record_results.append([travel_time, congestion_1_mean, congestion_2_mean, average_reward, deviation])
        if STAGE_MODE == "train_only":
            _maybe_save_model_checkpoint(model)
            return
        if implement == 1:
            stop_everything_in_learning_and_go_to_implementation_phase = 1
            _reset_model_inference_state("enter_implement")
            while True:
                if os.path.exists(get_stop_flag_path()):
                    _maybe_save_checkpoint_on_stop_exit()
                    sys.exit(78)
                if Dynamic_ALNS_RL34959.RL_can_start_implementation_phase_from_the_last_table == 1:
                    stop_everything_in_learning_and_go_to_implementation_phase = 0
                    break
            # if Intermodal_ALNS34959.used_interrupt == 1:
            #     print('I use interrupt here!!')
                # Intermodal_ALNS34959.interrupt_by_implement_is_one_and_assign_action_once_only == 1, the alns will be stopped because it is transferring training mode to implementation mode, then it will appear this, and then go to next iteration
            Intermodal_ALNS34959.state_reward_pairs = Intermodal_ALNS34959.state_reward_pairs.iloc[0:0]
            clear_pairs_done = 1
            Intermodal_ALNS34959.ALNS_implement_start_RL_can_move = 0
            # Intermodal_ALNS34959.used_interrupt = 0  # only use it as 1 once, then always be 0
            #check_RL_ALNS_iteraction_bug()
            # continue

            # Intermodal_ALNS34959.state_reward_pairs = Intermodal_ALNS34959.state_reward_pairs.iloc[0:0]
            # clear_pairs_done = 1

            if USE_LSTM:
                lstm_state = None
                lstm_episode_start = True
                lstm_impl_step = 0

            while True:
                while True:
                    if os.path.exists(get_stop_flag_path()):
                        _maybe_save_checkpoint_on_stop_exit()
                        sys.exit(78)
                    # if len(Intermodal_ALNS34959.state_reward_pairs) == 1 and implement == 1:
                    #     print('i should check this wrong')
                    if Intermodal_ALNS34959.ALNS_implement_start_RL_can_move == 1:
                        Intermodal_ALNS34959.ALNS_implement_start_RL_can_move = 0
                        break
                # if implement == 1 and Intermodal_ALNS34959.ALNS_implement_start_RL_can_move == 1:
                #     print('wrong...')
                #check_RL_ALNS_iteraction_bug()
                # if time_s == 22:
                #     print('c')
                # #check_RL_ALNS_iteraction_bug()
                # if implement == 1 and Intermodal_ALNS34959.ALNS_implement_start_RL_can_move == 1:
                #     print('wrong...')
                # if len(Intermodal_ALNS34959.state_reward_pairs) == 1 and implement == 1:
                #     print('i should check this wrong')
                obs = env.reset()
                # if implement == 1 and Intermodal_ALNS34959.ALNS_implement_start_RL_can_move == 1:
                #     print('wrong...')
                # #check_RL_ALNS_iteraction_bug()
                # if len(Intermodal_ALNS34959.state_reward_pairs) == 0:
                #     print('gesa')
                print('obs', obs)
                # while True:
                implementation_start_time = timeit.default_timer()
                impl_p_action1 = None
                if algorithm == 'LBKLAC':
                    act_info = model.act(obs, deterministic=True)
                    action_scalar = int(act_info.get("action", 0))
                else:
                    if USE_LSTM:
                        action, lstm_state = model.predict(
                            obs,
                            state=lstm_state,
                            episode_start=np.array([lstm_episode_start], dtype=bool),
                            deterministic=True,
                        )
                        try:
                            action_scalar = int(np.array(action).squeeze())
                        except Exception:
                            action_scalar = int(action)
                        lstm_impl_step += 1
                        if int(LSTM_CHAIN_LEN) > 1 and lstm_impl_step % int(LSTM_CHAIN_LEN) == 0:
                            lstm_episode_start = True
                            lstm_state = None
                        else:
                            lstm_episode_start = False
                        impl_p_action1 = _predict_action1_prob(model, obs)
                    elif _hat_is_active() and implement == 1 and algorithm in ("PPO", "A2C"):
                        action_scalar, _hat_info = _hat_select_action(model, obs)
                        impl_p_action1 = _hat_info.get("p1")
                    else:
                        if algorithm == "DRCB":
                            drcb_det = os.environ.get("DRCB_IMPL_DETERMINISTIC", "0").strip() == "1"
                            action, _states = model.predict(obs, deterministic=drcb_det)
                        elif algorithm == "BE_CVAR_DQN":
                            be_det = os.environ.get("BE_IMPL_DETERMINISTIC", "0").strip() == "1"
                            action, _states = model.predict(obs, deterministic=be_det)
                        elif algorithm == "QRDQN_CVAR":
                            action, _states = model.predict(obs, deterministic=True)
                        elif algorithm in {"CQL_DQN", "CQL"}:
                            cql_det = os.environ.get("CQL_IMPL_DETERMINISTIC", "1").strip() == "1"
                            action, _states = model.predict(obs, deterministic=cql_det)
                        else:
                            action, _states = model.predict(obs)
                        try:
                            action_scalar = int(np.array(action).squeeze())
                        except Exception:
                            action_scalar = action
                        impl_p_action1 = _predict_action1_prob(model, obs)
                # if implement == 1 and Intermodal_ALNS34959.ALNS_implement_start_RL_can_move == 1:
                #     print('wrong...')
                # #check_RL_ALNS_iteraction_bug()
                # if len(Intermodal_ALNS34959.state_reward_pairs) == 0:
                #     print('gesa')
                implementation_time = timeit.default_timer() - implementation_start_time
                log_training_row("implement", step_idx=global_step, implementation_time=implementation_time)
                try:
                    # Append one line to a file that does not exist
                    implementation_time_path = Intermodal_ALNS34959.path + "/finite_horizon_length" + str(
                        episode_length) + "_delay_reward_time_dependent" + str(
                        time_dependent) + "_tenterminal_" + algorithm + "_" + mode + "_" + str(
                        iteration_multiply) + "multiply" + 'implementation_time.txt'
                    append_new_line(implementation_time_path, str(implementation_time))
                except:
                    pass
                ALNS_got_action_in_implementation = 0
                #here i do not know why the Intermodal_ALNS34959.state_reward_pairs['uncertainty_type'] is finisih (maybe because implement_or_not is still 0 when it is the first implementation and the previous insertion/removal still unfinished), it should be begin because it is implement, so i set it as begin directly
                #Intermodal_ALNS34959.state_reward_pairs['uncertainty_type'] = 'begin'
                clear_pairs_done = 0
                # if len(Intermodal_ALNS34959.state_reward_pairs) == 0:
                #     print('gesa')
                #check_RL_ALNS_iteraction_bug()
                if algorithm == 'LBKLAC':
                    try:
                        n_state, reward, done, info = env.step(action_scalar)
                        _ = n_state, done
                        step_metrics = model.observe(
                            obs,
                            action_scalar,
                            reward,
                            n_state,
                            tokens=act_info.get("tokens"),
                            old_logp=float(act_info.get("logp", 0.0)),
                            record=False,
                            update=False,
                        )
                        payload = {
                            "action": action_scalar,
                            "reward": float(reward),
                            "action_prob": float(act_info.get("action_prob", 0.0)),
                            "entropy": float(act_info.get("entropy", 0.0)),
                            "info": info if isinstance(info, dict) else {},
                        }
                        payload.update(step_metrics)
                        _lbklac_on_step(payload)
                    except Exception as e:
                        print("LBKLAC implement step error", e)
                    try:
                        Intermodal_ALNS34959.state_reward_pairs = Intermodal_ALNS34959.state_reward_pairs.iloc[0:0]
                    except Exception:
                        pass
                    Intermodal_ALNS34959.ALNS_implement_start_RL_can_move = 0
                    clear_pairs_done = 1
                    continue
                trace_extra = None
                if impl_p_action1 is not None:
                    trace_extra = {"p_action1": float(impl_p_action1)}
                send_action(action_scalar, trace_extra=trace_extra, obs_snapshot=obs)
                #check_RL_ALNS_iteraction_bug()
                # if len(Intermodal_ALNS34959.state_reward_pairs) == 0:
                #     print('gesa')

                #check_RL_ALNS_iteraction_bug()
                while True:
                    # if implement == 1 and Intermodal_ALNS34959.ALNS_implement_start_RL_can_move == 1:
                    #     print('wrong...')
                    # print('main 1')
                    # if len(Intermodal_ALNS34959.state_reward_pairs) == 0:
                    #     print('gesa')
                    if Intermodal_ALNS34959.state_reward_pairs.iloc[0]['action'] == -10000000:
                        send_action(action_scalar, trace_extra=trace_extra, obs_snapshot=obs)
                    if ALNS_got_action_in_implementation == 1 or len(Intermodal_ALNS34959.state_reward_pairs) == 0:#danger donot know why in rare case Intermodal_ALNS34959.state_reward_pairs is empty when alns got action is 0, but i think i can let it go to next iteration
                        # clear all data in pairs
                        if os.path.exists(get_stop_flag_path()):
                            _maybe_save_checkpoint_on_stop_exit()
                            sys.exit(78)
                        _flush_impl_reward_lists(env)
                        if _hat_is_active() and implement == 1 and algorithm in ("PPO", "A2C"):
                            try:
                                if len(Intermodal_ALNS34959.state_reward_pairs) > 0:
                                    reward_hat = float(Intermodal_ALNS34959.state_reward_pairs.iloc[0]["reward"])
                                    if reward_hat != -10000000:
                                        _hat_update_stats(reward_hat, action_scalar)
                                        _hat_update_history_wrapper(env, action_scalar, reward_hat)
                            except Exception:
                                pass
                        #check_RL_ALNS_iteraction_bug()
                        ALNS_got_action_in_implementation = 0
                        Intermodal_ALNS34959.state_reward_pairs = Intermodal_ALNS34959.state_reward_pairs.iloc[0:0]
                        # if len(Intermodal_ALNS34959.state_reward_pairs) == 1 and implement == 1:
                        #     print('i should check this wrong')
                        Intermodal_ALNS34959.ALNS_implement_start_RL_can_move = 0
                        clear_pairs_done = 1
                        break
                    #check_RL_ALNS_iteraction_bug()
                    # n_state, reward, done, info = env.step(action)
                    # env.render()
                    # if done:
                        # print('n_state', n_state, 'reward', reward, 'info', info)
                    # break

        else:
            df_length = len(record_results)
            record_results.loc[df_length] = [congestion_terminal_mean_list, average_reward, deviation]

            #evaluate and print
            obs = env.reset()

            reward_all = 0
            all_action = 0
            evaluate_times = iteration_multiply
            for i in range(evaluate_times):
                # obs = np.array([random.choice([0,1])])
                obs = env.reset()
                print('obs', obs)
                while True:
                    # print('main 2')
                    if algorithm in ("DRCB", "QRDQN_CVAR", "BE_CVAR_DQN"):
                        action, _states = model.predict(obs, deterministic=True)
                    else:
                        action, _states = model.predict(obs)
                    n_state, reward, done, info = env.step(action)
                    # env.render()
                    # print('action', action, 'n_state', n_state, 'reward', reward, 'info', info)
                    all_action = all_action + action
                    reward_all = reward_all + reward
                    if done:
                        # print('n_state', n_state, 'reward', reward, 'info', info)
                        break

            print(mode, 'remove_proportion', all_action/evaluate_times)
            print('average_reward', reward_all/evaluate_times)

            #random
            reward_all = 0
            for i in range(iteration_multiply):
                env.reset()
                action = random.choice(range(0, 2))
                if env.step(action)[1] == 1:
                    reward_all += 1
            average_reward = reward_all / iteration_multiply
            print('congestion_terminal_mean_list',congestion_terminal_mean_list, average_reward)
            # record_results = record_results.append([travel_time, congestion_1_mean, congestion_2_mean, average_reward, '-'])
            df_length = len(record_results)
            record_results.loc[df_length] = [congestion_terminal_mean_list, average_reward,
                                             '-']
            compare_dir = os.path.join(
                ROOT_DIR,
                "Uncertainties Dynamic planning under unexpected events",
                "Average reward plots",
                "compare_algorithms_modes_episode_lenth2",
            )
            os.makedirs(compare_dir, exist_ok=True)
            compare_path = os.path.join(
                compare_dir,
                "finite_horizon_length"
                + str(episode_length)
                + "_delay_reward_time_dependent"
                + str(time_dependent)
                + "_tenterminal_"
                + algorithm
                + "_"
                + mode
                + "_"
                + str(iteration_multiply)
                + "multiply"
                + ".xlsx",
            )
            with pd.ExcelWriter(compare_path) as writer:  # doctest: +SKIP
                record_results.to_excel(writer, sheet_name='congestion')

if __name__ == '__main__':
    # ['A2C', 'DDPG', 'HER', 'SAC', 'TD3', 'PPO', 'DQN']
    # for algorithm in ['A2C', 'PPO', 'DQN']:
    for algorithm in ['DQN']:
        #DDPG AssertionError: The algorithm only supports <class 'gym.spaces.box.Box'> as action spaces but Discrete(2) was provided
        #Baselines 2.1.0, `HER` is now a replay buffer class `HerReplayBuffer`.\n "
# ImportError: Since Stable Baselines 2.1.0, `HER` is now a replay buffer class `HerReplayBuffer`.
#  Please check the documentation for more information: https://stable-baselines3.readthedocs.io/
#'TD3', SAC AssertionError: The algorithm only supports <class 'gym.spaces.box.Box'> as action spaces but Discrete(2) was provided

        for mode in ['all']:
            # , 'truck' 'barge'
            # mode = 'train'
            # mode = 'truck'
            main(algorithm, mode)

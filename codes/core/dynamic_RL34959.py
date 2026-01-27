import pandas as pd
import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
import numpy as np
import copy
import random
import os
import matplotlib.pyplot as plt
try:
    from stable_baselines3 import DQN, PPO, A2C, DDPG, HER, SAC, TD3
    from stable_baselines3.common.evaluation import evaluate_policy as sb3_evaluate_policy
    from stable_baselines3.common.vec_env import VecFrameStack
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.utils import explained_variance
    _SB3_AVAILABLE = True
except Exception:
    DQN = PPO = A2C = DDPG = HER = SAC = TD3 = None
    VecFrameStack = None
    sb3_evaluate_policy = None
    BaseCallback = None
    explained_variance = None
    _SB3_AVAILABLE = False
try:
    from sb3_contrib import RecurrentPPO
    _SB3_CONTRIB_AVAILABLE = True
except Exception:
    RecurrentPPO = None
    _SB3_CONTRIB_AVAILABLE = False
try:
    from robust_rl.lbklac import LBKLACAgent, LBKLACConfig
    _LBKLAC_AVAILABLE = True
except Exception:
    LBKLACAgent = None
    LBKLACConfig = None
    _LBKLAC_AVAILABLE = False
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


def _ema_update(prev, value, alpha):
    if prev is None:
        return float(value)
    return (1.0 - alpha) * float(prev) + alpha * float(value)


def _hat_is_active():
    try:
        return os.environ.get("RL_HAT", "0").strip() == "1" and algorithm in ("PPO", "A2C")
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
            log_trace_from_row(state_row, "receive_reward", action=action_val, reward=reward, source="RL")
        except Exception:
            pass
        if _hat_is_active() and implement == 1 and algorithm in ("PPO", "A2C"):
            try:
                _hat_update_stats(float(reward), float(action_val))
                _hat_update_history_wrapper(env, action_val, reward)
            except Exception:
                pass
        try:
            Intermodal_ALNS34959.log_impl_reward(reward)
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
            log_trace_from_row(state_row, "receive_reward", action=action_val, reward=reward, source="RL")
        except Exception:
            pass
        if _hat_is_active() and implement == 1 and algorithm in ("PPO", "A2C"):
            try:
                _hat_update_stats(float(reward), float(action_val))
                _hat_update_history_wrapper(env, action_val, reward)
            except Exception:
                pass
        try:
            Intermodal_ALNS34959.log_impl_reward(reward)
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

def resolve_dynamic_data_path(request_number_in_R, table_number, duration_type, add_event_types):
    dynamic_root = os.environ.get("DYNAMIC_DATA_ROOT", "").strip()
    if dynamic_root:
        return os.path.join(dynamic_root, f"R{request_number_in_R}", f"Intermodal_EGS_data_dynamic_congestion{table_number}.xlsx")
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
CURRICULUM_REWARD_THRESHOLD = 0.7
CURRICULUM_SUCCESS_REQUIRED = 3
curriculum_converged = 0
curriculum_last_avg_reward = ""
SCENARIO_NAME = os.environ.get("SCENARIO_NAME", "")

TRACE_FIELDS = [
    "ts", "phase", "stage", "uncertainty_index", "request", "vehicle",
    "table_number", "dynamic_t_begin", "duration_type", "gt_mean", "phase_label",
    "delay_tolerance", "severity", "passed_terminals", "current_time",
    "action", "reward", "action_meaning", "feasible", "source",
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
]

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
            "dynamic_t_begin": getattr(Intermodal_ALNS34959, "dynamic_t_begin", ""),
            "duration_type": getattr(Intermodal_ALNS34959, "duration_type", ""),
            "gt_mean": current_gt_mean,
            "phase_label": current_phase_label,
            "delay_tolerance": row.get("delay_tolerance", ""),
            "severity": globals().get("severity_level", ""),
            "passed_terminals": row.get("passed_terminals", ""),
            "current_time": row.get("current_time", ""),
            "action": action_val,
            "reward": reward if reward is not None else row.get("reward", ""),
            "action_meaning": action_meaning,
            "feasible": feasible,
            "source": source,
        }
        payload.update(_drift_snapshot())
        # Best-effort MoE stats from policy (no impact on non-MoE runs).
        try:
            if model is not None and hasattr(model, "policy") and hasattr(model.policy, "get_moe_log"):
                payload.update(model.policy.get_moe_log())
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
            sys.exit(78)
    except:
        if os.path.exists(get_stop_flag_path()):
            save_plot_reward_list()
            sys.exit(78)
#@profile()
def send_action(action):
    # global only_stop_once_by_implementation
    if stop_everything_in_learning_and_go_to_implementation_phase == 1:
        return
    # get the index first
    break_flag = 0
    wait_start = time.time()
    last_log = wait_start
    while True:
        if stop_everything_in_learning_and_go_to_implementation_phase == 1:
            return
        if len(Intermodal_ALNS34959.state_reward_pairs) != 0:
            break
        else:
            print('len(Intermodal_ALNS34959.state_reward_pairs) == 0 in send_action function')
            timed_out, last_log, wait_start = _wait_watchdog("send_action_wait_pairs", wait_start, last_log)
            if timed_out:
                return
            if WAIT_SLEEP_S > 0:
                time.sleep(WAIT_SLEEP_S)
    wait_start = time.time()
    last_log = wait_start
    while True:
        # print('send action 1')
        # if only_stop_once_by_implementation == 0:
        #     if Intermodal_ALNS34959.interrupt_by_implement_is_one_and_assign_action_once_only == 1:
        #         only_stop_once_by_implementation = 1
        #         break
        stop_wait()
        if stop_everything_in_learning_and_go_to_implementation_phase == 1:
            return
        for pair_index in Intermodal_ALNS34959.state_reward_pairs.index:
            try:
                check = Intermodal_ALNS34959.state_reward_pairs['uncertainty_index'][pair_index] == uncertainty_index and \
                    Intermodal_ALNS34959.state_reward_pairs['vehicle'][pair_index] == vehicle and \
                    Intermodal_ALNS34959.state_reward_pairs['request'][pair_index] == request and \
                    Intermodal_ALNS34959.state_reward_pairs['action'][pair_index] == -10000000
            except:
                break
            if implement == 0:
                if Intermodal_ALNS34959.after_action_review == 1:
                    check = check and Intermodal_ALNS34959.state_reward_pairs['uncertainty_type'][pair_index] == 'finish'
            else:
                check = check and Intermodal_ALNS34959.state_reward_pairs['uncertainty_type'][pair_index] == 'begin'
            if check:
                while True:
                    # print('send action 2')
                    stop_wait()
                    if stop_everything_in_learning_and_go_to_implementation_phase == 1:
                        return
                    try:
                        Intermodal_ALNS34959.state_reward_pairs['action'][pair_index] = action
                        try:
                            row_dict = dict(Intermodal_ALNS34959.state_reward_pairs.loc[pair_index])
                        except:
                            row_dict = {}
                        log_trace_from_row(row_dict, "send_action", action=action, source="RL")
                        break
                    except:
                        print("Intermodal_ALNS34959.state_reward_pairs['action'][pair_index] = action")
                        continue
                # parallel_save_excel(path + 'state_reward_pairs.xlsx', Intermodal_ALNS34959.state_reward_pairs, 'state_reward_pairs')
                break_flag = 1
                break
        if break_flag == 1:
            while True:
                # print('send action 3')
                stop_wait()
                if stop_everything_in_learning_and_go_to_implementation_phase == 1:
                    return
                if Intermodal_ALNS34959.state_reward_pairs['action'][pair_index] != -10000000:
                    break
                else:
                    print("Intermodal_ALNS34959.state_reward_pairs['action'][pair_index] != -10000000")
                    Intermodal_ALNS34959.state_reward_pairs['action'][pair_index] = action
            break
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
            base_low = [0, 0, 0]
            base_high = [200, 6, 6]
        else:
            base_low = [0, 0]
            base_high = [200, 6]
        if STAGE_IN_OBS:
            base_low = base_low + [0, 0]
            base_high = base_high + [1, 1]
        self.observation_space = Box(low=np.array(base_low), high=np.array(base_high))
        # self.state = [random.choice(range(0,24)), random.choice(range(0,11))]
        # Set coordination length
        self.horizon_length = 0
        # self.dis = 0

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
            else:
                if stop_everything_in_learning_and_go_to_implementation_phase == 1:
                    return self.state, 0, True, {}
                send_action(action)

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
                            recent_rewards.append(reward)
                            if _hat_is_active():
                                _hat_update_stats(reward, action)
                                _hat_update_train_params()
                            step_id = next_step()
                            try:
                                row_dict = dict(Intermodal_ALNS34959.state_reward_pairs.loc[pair_index])
                            except:
                                row_dict = {}
                            uncertainty_type = Intermodal_ALNS34959.state_reward_pairs['uncertainty_type'][pair_index]
                            info = {
                                "row_dict": row_dict,
                                "uncertainty_type": uncertainty_type,
                                "pair_index": pair_index,
                            }
                            if not LBKLAC_CUSTOM_LOGGING:
                                log_training_row("implement" if implement == 1 else "train", step_idx=step_id, reward=reward)
                                log_trace_from_row(row_dict, "receive_reward", action=row_dict.get('action', ''), reward=reward, source="RL")
                            # parallel_save_excel(path + 'state_reward_pairs.xlsx', state_reward_pairs, 'state_reward_pairs')
                            #drop the finish
                            # Intermodal_ALNS34959.state_reward_pairs = Intermodal_ALNS34959.state_reward_pairs.drop(
                            #     labels=pair_index,
                            #     axis=0)
                            break_flag = 1
                            break
                        elif Intermodal_ALNS34959.state_reward_pairs['action'][pair_index] == -10000000:
                            send_action(action)
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
    global wrong_severity_level_with_probability, add_event_types, stop_everything_in_learning_and_go_to_implementation_phase, clear_pairs_done, ALNS_got_action_in_implementation, table_number_collect, state_action_reward_collect, all_rewards_list, wait_training_finish_last_iteration, state_action_reward_collect_for_evaluate, number_of_state_key, state_keys, evaluate, implement, iteration_times, RL_drop_finish, non_stationary, algorithm, time_dependent, episode_length, next_state_reward_time_step, next_state_penalty_time_step, total_timesteps2, iteration_multiply, add_ALNS, iteration_numbers_unit, mode, travel_time_barge, travel_time_train, travel_time_truck, time_s, model, env, all_average_reward,all_deviation, timestamps, repeat, sucess_times, curriculum_converged, curriculum_last_avg_reward, LBKLAC_CUSTOM_LOGGING, USE_LSTM, STAGE_IN_OBS, LSTM_CHAIN_LEN, LSTM_CHAIN_STEP
    add_event_types =0 
    stop_everything_in_learning_and_go_to_implementation_phase = 0
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
        use_hat = os.environ.get("RL_HAT", "0").strip() == "1"
        if use_hat and algorithm in ("PPO", "A2C"):
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
            model = DriftRobustContextualBandit(
                env,
                seed=seed_val,
                decay=0.995,
                ridge=1.0,
                ucb_alpha=0.2,
                context_getter=lambda: {
                    "gt_mean": current_gt_mean,
                    "phase_label": current_phase_label,
                    "table_number": getattr(Dynamic_ALNS_RL34959, "table_number", ""),
                },
            )
        elif algorithm == 'DQN':
            if not _SB3_AVAILABLE or DQN is None:
                raise ImportError("stable_baselines3 is required for DQN. Please install stable-baselines3 + torch.")
            model = DQN('MlpPolicy', env, verbose=1, learning_starts=10, device='cpu', seed=seed_val)
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
        lstm_callback = None
        if USE_LSTM and BaseCallback is not None:
            try:
                lstm_callback = LstmStatsCallback()
            except Exception:
                lstm_callback = None
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
            if implement == 1:
                break
            if algorithm == 'LBKLAC':
                model.learn(total_timesteps=total_timesteps2, on_step=_lbklac_on_step)
            else:
                if lstm_callback is not None and USE_LSTM:
                    model.learn(total_timesteps=total_timesteps2, callback=lstm_callback)
                else:
                    model.learn(total_timesteps=total_timesteps2)
            training_time = timeit.default_timer() - start_time
            log_training_row("train", step_idx=global_step, training_time=training_time)
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
            log_training_row(
                "eval",
                step_idx=global_step,
                avg_reward=average_reward,
                std_reward=deviation,
                rolling_avg=rolling_avg,
                recent_count=len(recent_rewards),
            )
            print('evaluation', 'average_reward', average_reward, 'deviation', deviation)# sys.exit('stop_it_in_testing')
            # Curriculum convergence: used for jumps only
            curriculum_last_avg_reward = rolling_avg
            threshold = CURRICULUM_REWARD_THRESHOLD
            if SCENARIO_NAME == "S0_Debug":
                threshold = 0.3
            if rolling_avg >= threshold:
                sucess_times += 1
            else:
                sucess_times = 0
            curriculum_converged = 1 if sucess_times >= CURRICULUM_SUCCESS_REQUIRED else 0

            wait_training_finish_last_iteration = 0
            evaluate = 0
            # record_results = record_results.append([travel_time, congestion_1_mean, congestion_2_mean, average_reward, deviation])
        if implement == 1:
            stop_everything_in_learning_and_go_to_implementation_phase = 1
            while True:
                if os.path.exists(get_stop_flag_path()):
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
                    elif _hat_is_active() and implement == 1 and algorithm in ("PPO", "A2C"):
                        action_scalar, _hat_info = _hat_select_action(model, obs)
                    else:
                        action, _states = model.predict(obs)
                        try:
                            action_scalar = int(np.array(action).squeeze())
                        except Exception:
                            action_scalar = action
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
                send_action(action_scalar)
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
                        send_action(action_scalar)
                    if ALNS_got_action_in_implementation == 1 or len(Intermodal_ALNS34959.state_reward_pairs) == 0:#danger donot know why in rare case Intermodal_ALNS34959.state_reward_pairs is empty when alns got action is 0, but i think i can let it go to next iteration
                        # clear all data in pairs
                        if os.path.exists(get_stop_flag_path()):
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

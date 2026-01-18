import pandas as pd
import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
import numpy as np
import copy
import random
import os
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO, A2C, DDPG, HER, SAC, TD3
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
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
import timeit
import time
import Intermodal_ALNS34959
import sys
import Dynamic_ALNS_RL34959
import cProfile
import pstats
import io
from pathlib import Path
import rl_logging
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
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    "action", "reward", "action_meaning", "feasible", "source"
]

current_gt_mean = ""
current_phase_label = ""

TRAIN_FIELDS = [
    "ts", "phase", "step_idx", "reward", "avg_reward", "std_reward",
    "rolling_avg", "recent_count",
    "training_time", "implementation_time"
]

def log_trace_from_row(row, stage, action=None, reward=None, feasible="", source="RL"):
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
            "source": source
        }
        rl_logging.append_row("rl_trace.csv", TRACE_FIELDS, payload)
    except Exception as e:
        print("log_trace_from_row error", e)

def log_training_row(phase, step_idx="", reward=None, avg_reward=None, std_reward=None,
                     rolling_avg=None, recent_count=None, training_time=None, implementation_time=None):
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
        rl_logging.append_row("rl_training.csv", TRAIN_FIELDS, payload)
    except Exception as e:
        print("log_training_row error", e)

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
        if os.path.exists('34959.txt') and Intermodal_ALNS34959.ALNS_end_flag != 1:
            save_plot_reward_list()
            sys.exit(78)
    except:
        if os.path.exists('34959.txt'):
            save_plot_reward_list()
            sys.exit(78)
#@profile()
def send_action(action):
    # global only_stop_once_by_implementation
    if stop_everything_in_learning_and_go_to_implementation_phase == 1:
        return
    # get the index first
    break_flag = 0
    while True:
        if stop_everything_in_learning_and_go_to_implementation_phase == 1:
            return
        if len(Intermodal_ALNS34959.state_reward_pairs) != 0:
            break
        else:
            print('len(Intermodal_ALNS34959.state_reward_pairs) == 0 in send_action function')
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
            self.observation_space = Box(low=np.array([0, 0, 0]), high=np.array([200, 6, 6]))
        else:
            self.observation_space = Box(low=np.array([0, 0]),high=np.array([200, 6]))
        # self.state = [random.choice(range(0,24)), random.choice(range(0,11))]
        # Set coordination length
        self.horizon_length = 0
        # self.dis = 0

    #@profile()
    def step(self, action):
        global state_action_reward_collect, all_rewards_list, wait_training_finish_last_iteration, state_action_reward_collect_for_evaluate, number_of_state_key, state_keys, iteration_times, RL_drop_finish, episode_length, next_state_reward_time_step, next_state_penalty_time_step, time_s, all_average_reward, all_deviation, timestamps
        # 将动作转为标量
        try:
            if isinstance(action, np.ndarray):
                action = int(action.squeeze())
            else:
                action = int(action)
        except Exception:
            pass

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
            else:
                if stop_everything_in_learning_and_go_to_implementation_phase == 1:
                    return self.state, 0, True, {}
                send_action(action)

                #get the reward from ALNS
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
                            step_id = next_step()
                            log_training_row("implement" if implement == 1 else "train", step_idx=step_id, reward=reward)
                            try:
                                row_dict = dict(Intermodal_ALNS34959.state_reward_pairs.loc[pair_index])
                            except:
                                row_dict = {}
                            log_trace_from_row(row_dict, "receive_reward", action=row_dict.get('action', ''), reward=reward, source="RL")
                            # parallel_save_excel(path + 'state_reward_pairs.xlsx', state_reward_pairs, 'state_reward_pairs')
                            uncertainty_type = Intermodal_ALNS34959.state_reward_pairs['uncertainty_type'][pair_index]
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
                average_reward, deviation = evaluate_policy(model, env, n_eval_episodes=iteration_numbers_unit, render=False)
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
        if self.horizon_length == episode_length:
            done = True
        else:
            done = False

        # Apply temperature noise
        # self.state += random.randint(-1,1)
        # Set placeholder for info
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
    global wrong_severity_level_with_probability, add_event_types, stop_everything_in_learning_and_go_to_implementation_phase, clear_pairs_done, ALNS_got_action_in_implementation, table_number_collect, state_action_reward_collect, all_rewards_list, wait_training_finish_last_iteration, state_action_reward_collect_for_evaluate, number_of_state_key, state_keys, evaluate, implement, iteration_times, RL_drop_finish, non_stationary, algorithm, time_dependent, episode_length, next_state_reward_time_step, next_state_penalty_time_step, total_timesteps2, iteration_multiply, add_ALNS, iteration_numbers_unit, mode, travel_time_barge, travel_time_train, travel_time_truck, time_s, model, env, all_average_reward,all_deviation, timestamps, repeat, sucess_times, curriculum_converged, curriculum_last_avg_reward
    add_event_types =0 
    stop_everything_in_learning_and_go_to_implementation_phase = 0
    clear_pairs_done = 0
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
    episode_length = 1
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
        if algorithm == 'DQN':
            model = eval(algorithm + "('MlpPolicy', env, verbose=1, learning_starts=10, device='cpu')")
        else:
            model = eval(algorithm + "('MlpPolicy', env, n_steps=10, verbose=1, device='cpu')")
            #break
           # except:
            #    continue
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
                    average_reward, deviation = evaluate_policy(model, env, n_eval_episodes=iteration_numbers_unit, render=False)
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
                threshold = 0.55
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
                if os.path.exists('34959.txt'):
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

            while True:
                while True:
                    if os.path.exists('34959.txt'):
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
                        if os.path.exists('34959.txt'):
                            sys.exit(78)
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

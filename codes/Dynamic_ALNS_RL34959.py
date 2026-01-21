#!/usr/bin/env Python
# coding=utf-8
# import concurrent.futures
import Intermodal_ALNS34959
import dynamic_RL34959
#import dynamic_RL_online_insertion
import pandas as pd
import os
import time
import sys
import json
# haven't done: set the initial solution as original route, and detect which request is changed, and check which part can't be removed
SCENARIO_NAME = os.environ.get("SCENARIO_NAME", "")
RL_ALGORITHM = os.environ.get("RL_ALGORITHM", "DQN")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "distribution_config.json")

def load_distribution_patterns():
    patterns = {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data.get("distributions", []):
            name = str(item.get("name", "")).strip()
            pattern = str(item.get("pattern", "")).strip()
            if name and pattern:
                patterns[name.upper()] = pattern.lower()
    except Exception:
        pass
    return patterns

DISTRIBUTION_PATTERNS = load_distribution_patterns()

def Intermodal_ALNS_function(request_number_in_R):
    global dynamic_end
    Intermodal_ALNS34959.real_main(3, 0, request_number_in_R)

    # data_path = 'C:/Users/yimengzhang/OneDrive/桌面/Intermodal_EGS_data_dynamic_new_requests.xlsx'
    while True:
        try:
            request_number_in_R = Intermodal_ALNS34959.request_number_in_R
            data_path = Intermodal_ALNS34959.data_path
            break
        except Exception:
            continue
    Data = pd.ExcelFile(data_path)
    R = pd.read_excel(Data, 'R_' + str(request_number_in_R))
    # time_horizon is the maximum time of request delivery
    max_delivery = 0
    for r in R.index:
        if R['bd'][r] > max_delivery:
            max_delivery = R['bd'][r]
    time_horizon = range(0, max_delivery)
    # set unexpected events
    # two types:
    # uncertain events: generated stochastically
    # known events: know what will happen like a god
    # in the TRC paper, like a god, just set the events in excel
    # name of sheet: R_number_time
    # in the sheet, mark request number for changed request (OD/schedule/load), assign new request a number
    # only the changed R and new R in the sheet
    # for changed R, give the request number

    # unexpected delay is defined in ALNS by changing a vehicle's D
    # the line before if what == 'D':

    all_sheets = pd.read_excel(Data, None)

    unexpected_times = []
    for key in all_sheets.keys():
        prefix = 'R_' + str(request_number_in_R)
        if prefix in key and prefix != key and 'info' not in key:
            if ' ' in key:
                # pass
                blank_index = key.rfind(' ')
                unexpected_times.append(int(key.replace(prefix + '_', '')[0:blank_index - len(prefix) - 1]))
            else:
                unexpected_times.append(int(key.replace(prefix + '_', '')))
    unexpected_times = list(dict.fromkeys(unexpected_times))
    unexpected_times.sort()
    dynamic_end = 0

    for t in time_horizon:
        # handle unexpected events methods
        # rerun
        # predict what will happen
        # prepare in advance, optimize based on the worst situation
        #                                       an integrated way
        if t in unexpected_times:
            if t == unexpected_times[-1]:
                dynamic_end = 1
            Intermodal_ALNS34959.real_main(3, t, request_number_in_R)

    #another way of dynamic is optimizing only the urgent parts of requests, maybe better than this way
def main(approach, request_number_in_R = 5):
    global RL_can_start_implementation_phase_from_the_last_table, ALNS_calculates_average_duration_list, ALNS_reward_list_in_implementation, ALNS_removal_reward_list_in_implementation,  ALNS_removal_action_list_in_implementation, ALNS_insertion_reward_list_in_implementation, ALNS_insertion_action_list_in_implementation, table_number, reward_list_in_implementation, removal_reward_list_in_implementation, removal_state_list_in_implementation, removal_action_list_in_implementation, insertion_reward_list_in_implementation, insertion_state_list_in_implementation, insertion_action_list_in_implementation
    Intermodal_ALNS34959.request_number_in_R = request_number_in_R
    RL_can_start_implementation_phase_from_the_last_table = 0
    ALNS_calculates_average_duration_list = []
    combine_insertion_and_removal_operators = 1
    if combine_insertion_and_removal_operators == 0:
        if approach == 1:
            dynamic_RL34959.main(RL_ALGORITHM, 'barge')
        elif approach == 2:
            dynamic_RL_online_insertion.main('DQN', 'barge')
        else:
            Intermodal_ALNS_function(request_number_in_R)
    else:
        if approach == 1:
            dynamic_RL34959.main(RL_ALGORITHM, 'barge')
        else:
            reward_list_in_implementation, removal_reward_list_in_implementation, removal_state_list_in_implementation, removal_action_list_in_implementation, insertion_reward_list_in_implementation, insertion_state_list_in_implementation, insertion_action_list_in_implementation = [], [], [], [], [], [], []
            ALNS_reward_list_in_implementation, ALNS_removal_reward_list_in_implementation,  ALNS_removal_action_list_in_implementation, ALNS_insertion_reward_list_in_implementation, ALNS_insertion_action_list_in_implementation = [], [], [], [], []
            table_number = 0
            start_from_end_table = 0
            implement_start_synced = 0
            while True:
                # When switching to implementation/test, RL may briefly set a stop flag
                # to break the training loop and reset internal shared state. During that
                # window, ALNS returns early (no table processed). We must NOT advance
                # table_number, otherwise the "test" phase will fast-forward to the
                # boundary without actually running tables.
                # dynamic_RL34959.implement may not be initialized until the RL thread starts.
                if getattr(dynamic_RL34959, "implement", 0) == 1:
                    if RL_can_start_implementation_phase_from_the_last_table == 0:
                        RL_can_start_implementation_phase_from_the_last_table = 1
                    if implement_start_synced == 0:
                        # Ensure test phase starts from 499 once implement flips.
                        if table_number < 499:
                            table_number = 499
                        implement_start_synced = 1
                    if getattr(dynamic_RL34959, "stop_everything_in_learning_and_go_to_implementation_phase", 0) == 1:
                        time.sleep(0.05)
                        continue

                Intermodal_ALNS_function(request_number_in_R)
                try:
                    if getattr(dynamic_RL34959, "implement", 0) == 1:
                        table_number -= 1
                        if table_number < 350:
                            print(">>> TEST COMPLETE: Reached boundary (350). Saving data and exiting.")
                            try:
                                dynamic_RL34959.save_plot_reward_list()
                            except Exception:
                                pass
                            # Signal the RL thread to stop and return gracefully so the
                            # ThreadPoolExecutor can join cleanly.
                            try:
                                stop_flag = dynamic_RL34959.get_stop_flag_path()
                                os.makedirs(os.path.dirname(stop_flag), exist_ok=True)
                                with open(stop_flag, "a", encoding="utf-8"):
                                    pass
                            except Exception:
                                pass
                            return
                    else:
                        scenario_name = getattr(dynamic_RL34959, "SCENARIO_NAME", "") or SCENARIO_NAME or os.environ.get("SCENARIO_NAME", "")
                        scenario_name = str(scenario_name).upper()
                        scenario_pattern = DISTRIBUTION_PATTERNS.get(scenario_name, "")
                        converged = getattr(dynamic_RL34959, "curriculum_converged", 0) == 1
                        next_table_number = table_number + 1
                        if converged:
                            if scenario_pattern == "random_mix" or scenario_name.startswith("S1"):
                                min_train = 10
                                if table_number >= min_train:
                                    print(">>> [S1] Mastery of mixed stage. Jumping to Test (499)...")
                                    dynamic_RL34959.stop_everything_in_learning_and_go_to_implementation_phase = 1
                                    RL_can_start_implementation_phase_from_the_last_table = 1
                                    dynamic_RL34959.implement = 1
                                    next_table_number = 499
                                    dynamic_RL34959.sucess_times = 0
                                    dynamic_RL34959.curriculum_converged = 0
                            elif scenario_pattern == "aba" or scenario_name.startswith("S2"):
                                phase_a_end = 175
                                min_phase_b_train = 180
                                if table_number < phase_a_end:
                                    print(">>> [S2] Mastery of Phase A. Jumping to Phase B (175)...")
                                    next_table_number = phase_a_end
                                    dynamic_RL34959.sucess_times = 0
                                    dynamic_RL34959.curriculum_converged = 0
                                elif table_number >= min_phase_b_train:
                                    print(">>> [S2] Mastery of Phase B. Jumping to Test (499)...")
                                    dynamic_RL34959.stop_everything_in_learning_and_go_to_implementation_phase = 1
                                    RL_can_start_implementation_phase_from_the_last_table = 1
                                    dynamic_RL34959.implement = 1
                                    next_table_number = 499
                                    dynamic_RL34959.sucess_times = 0
                                    dynamic_RL34959.curriculum_converged = 0
                            elif scenario_pattern == "ab" or scenario_name.startswith("S3"):
                                min_train = 5
                                if table_number >= min_train:
                                    print(">>> [S3] Mastery of single stage. Jumping to Test (499)...")
                                    dynamic_RL34959.stop_everything_in_learning_and_go_to_implementation_phase = 1
                                    RL_can_start_implementation_phase_from_the_last_table = 1
                                    dynamic_RL34959.implement = 1
                                    next_table_number = 499
                                    dynamic_RL34959.sucess_times = 0
                                    dynamic_RL34959.curriculum_converged = 0
                            elif scenario_pattern == "recall" or scenario_name.startswith("S4"):
                                min_train = 5
                                if table_number >= min_train:
                                    print(">>> [S4] Mastery of single stage. Jumping to Test (499)...")
                                    dynamic_RL34959.stop_everything_in_learning_and_go_to_implementation_phase = 1
                                    RL_can_start_implementation_phase_from_the_last_table = 1
                                    dynamic_RL34959.implement = 1
                                    next_table_number = 499
                                    dynamic_RL34959.sucess_times = 0
                                    dynamic_RL34959.curriculum_converged = 0
                            elif scenario_pattern == "adaptation" or scenario_name.startswith("S5"):
                                phase_a_end = 100
                                min_phase_b_train = 105
                                if scenario_name.startswith("S0"):
                                    # Debug scenario: allow immediate test after entering Phase B.
                                    min_phase_b_train = phase_a_end
                                if table_number < phase_a_end:
                                    print(">>> [S5] Mastery of Phase A. Jumping to Phase B (100)...")
                                    next_table_number = phase_a_end
                                    dynamic_RL34959.sucess_times = 0
                                    dynamic_RL34959.curriculum_converged = 0
                                elif table_number >= min_phase_b_train:
                                    print(">>> [S5] Mastery of Phase B. Jumping to Test (499)...")
                                    dynamic_RL34959.stop_everything_in_learning_and_go_to_implementation_phase = 1
                                    RL_can_start_implementation_phase_from_the_last_table = 1
                                    dynamic_RL34959.implement = 1
                                    next_table_number = 499
                                    dynamic_RL34959.sucess_times = 0
                                    dynamic_RL34959.curriculum_converged = 0
                            elif scenario_pattern == "abc" or scenario_name.startswith("S6"):
                                phase_a_end = 175
                                min_phase_b_train = 180
                                if table_number < phase_a_end:
                                    print(">>> [S6] Mastery of Phase A. Jumping to Phase B (175)...")
                                    next_table_number = phase_a_end
                                    dynamic_RL34959.sucess_times = 0
                                    dynamic_RL34959.curriculum_converged = 0
                                elif table_number >= min_phase_b_train:
                                    print(">>> [S6] Mastery of Phase B. Jumping to Test (499)...")
                                    dynamic_RL34959.stop_everything_in_learning_and_go_to_implementation_phase = 1
                                    RL_can_start_implementation_phase_from_the_last_table = 1
                                    dynamic_RL34959.implement = 1
                                    next_table_number = 499
                                    dynamic_RL34959.sucess_times = 0
                                    dynamic_RL34959.curriculum_converged = 0
                        if getattr(dynamic_RL34959, "implement", 0) == 0 and table_number >= 349:
                            print(">>> FORCE SWITCH: Reached table_number 349. Jumping to Test (499)...")
                            dynamic_RL34959.stop_everything_in_learning_and_go_to_implementation_phase = 1
                            RL_can_start_implementation_phase_from_the_last_table = 1
                            dynamic_RL34959.implement = 1
                            next_table_number = 499
                            dynamic_RL34959.sucess_times = 0
                            dynamic_RL34959.curriculum_converged = 0
                        table_number = next_table_number
                        if getattr(dynamic_RL34959, "implement", 0) == 0 and table_number > 349:
                            table_number = 349
                except SystemExit:
                    raise
                except Exception:
                    if Intermodal_ALNS34959.add_RL == 0:
                        if Intermodal_ALNS34959.ALNS_greedy_under_unknown_duration_assume_duration == 0:
                            table_number -= 1
                        elif Intermodal_ALNS34959.ALNS_greedy_under_unknown_duration_assume_duration == 3 and len(
                                ALNS_reward_list_in_implementation) > Intermodal_ALNS34959.number_of_training:
                            if start_from_end_table == 0:
                                table_number = 999
                                start_from_end_table = 1
                            else:
                                table_number -= 1
                        else:
                            table_number += 1
if __name__ == '__main__':
    main(approach)

#!/usr/bin/env Python
# coding=utf-8
# import concurrent.futures
from core import Intermodal_ALNS34959
from core import dynamic_RL34959
#import dynamic_RL_online_insertion
import pandas as pd
import os
import time
import sys
import json
# haven't done: set the initial solution as original route, and detect which request is changed, and check which part can't be removed
SCENARIO_NAME = os.environ.get("SCENARIO_NAME", "")
RL_ALGORITHM = os.environ.get("RL_ALGORITHM", "DQN")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

TRAIN_TABLE_MIN = 0
TRAIN_TABLE_MAX = 799
TEST_TABLE_START = 999
TEST_TABLE_MIN = 800
PHASE_B_START_TABLE = 400
MIN_SINGLE_STAGE_TRAIN_RANDOM = 10
MIN_SINGLE_STAGE_TRAIN_OOD = 10
MIN_PHASE_B_TRAIN = 420


def _normalize_stage_mode(value):
    mode = str(value or "train_eval").strip().lower()
    if mode not in {"train_eval", "train_only", "eval_only"}:
        mode = "train_eval"
    return mode


def _touch_stop_flag():
    try:
        stop_flag = dynamic_RL34959.get_stop_flag_path()
        os.makedirs(os.path.dirname(stop_flag), exist_ok=True)
        with open(stop_flag, "a", encoding="utf-8"):
            pass
    except Exception:
        pass


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
            stage_mode = _normalize_stage_mode(os.environ.get("RL_STAGE_MODE", "train_eval"))
            train_only = stage_mode == "train_only"
            eval_only = stage_mode == "eval_only"
            train_only_early_stop = os.environ.get("RL_TRAIN_ONLY_EARLY_STOP", "1").strip() == "1"
            try:
                train_only_min_table = int(os.environ.get("RL_TRAIN_ONLY_MIN_TABLE", str(MIN_SINGLE_STAGE_TRAIN_RANDOM)))
            except Exception:
                train_only_min_table = MIN_SINGLE_STAGE_TRAIN_RANDOM
            train_only_min_table = max(TRAIN_TABLE_MIN, min(TRAIN_TABLE_MAX, int(train_only_min_table)))

            table_number = TEST_TABLE_START if eval_only else 0
            start_from_end_table = 0
            implement_start_synced = 0
            implement_reward_base = None
            f2_implement_jump_done = 0
            scenario_name = getattr(dynamic_RL34959, "SCENARIO_NAME", "") or SCENARIO_NAME or os.environ.get("SCENARIO_NAME", "")
            scenario_name = str(scenario_name).upper()
            scenario_pattern = DISTRIBUTION_PATTERNS.get(scenario_name, "")

            if eval_only:
                dynamic_RL34959.implement = 1
                dynamic_RL34959.stop_everything_in_learning_and_go_to_implementation_phase = 0

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
                        # Ensure test phase starts from TEST_TABLE_START once implement flips.
                        if table_number < TEST_TABLE_START:
                            table_number = TEST_TABLE_START
                        implement_start_synced = 1
                    if getattr(dynamic_RL34959, "stop_everything_in_learning_and_go_to_implementation_phase", 0) == 1:
                        time.sleep(0.05)
                        continue

                Intermodal_ALNS_function(request_number_in_R)
                try:
                    if getattr(dynamic_RL34959, "implement", 0) == 1:
                        if implement_reward_base is None:
                            implement_reward_base = len(reward_list_in_implementation)
                        table_number -= 1
                        # F2-only fast switch in implement phase:
                        # once implement reward count is high enough and we're still at 900..999,
                        # jump directly to 899 so test can cover both A/B blocks before the fixed 200-stop.
                        if scenario_pattern == "abba":
                            implement_reward_count = max(0, len(reward_list_in_implementation) - implement_reward_base)
                            if (
                                f2_implement_jump_done == 0
                                and 900 <= table_number <= TEST_TABLE_START
                                and implement_reward_count > 100
                            ):
                                print(
                                    f">>> [abba/F2] implement_reward_count={implement_reward_count} "
                                    f"at table={table_number}. Jumping to 899."
                                )
                                table_number = 899
                                f2_implement_jump_done = 1
                        if table_number < TEST_TABLE_MIN:
                            print(f">>> TEST COMPLETE: Reached boundary ({TEST_TABLE_MIN}). Saving data and exiting.")
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
                        converged = getattr(dynamic_RL34959, "curriculum_converged", 0) == 1
                        next_table_number = table_number + 1
                        if eval_only:
                            dynamic_RL34959.implement = 1
                            next_table_number = TEST_TABLE_START
                        elif train_only and train_only_early_stop and converged and table_number >= train_only_min_table:
                            print(
                                f">>> TRAIN_ONLY EARLY STOP: converged at table={table_number} "
                                f"(min={train_only_min_table})."
                            )
                            _touch_stop_flag()
                            return
                        elif converged and not train_only:
                            if scenario_pattern == "random_mix":
                                if table_number >= MIN_SINGLE_STAGE_TRAIN_RANDOM:
                                    print(f">>> [random_mix] Mastery reached. Jumping to Test ({TEST_TABLE_START})...")
                                    dynamic_RL34959.stop_everything_in_learning_and_go_to_implementation_phase = 1
                                    RL_can_start_implementation_phase_from_the_last_table = 1
                                    dynamic_RL34959.implement = 1
                                    next_table_number = TEST_TABLE_START
                                    dynamic_RL34959.sucess_times = 0
                                    dynamic_RL34959.curriculum_converged = 0
                            elif scenario_pattern == "ab":
                                if table_number >= MIN_SINGLE_STAGE_TRAIN_OOD:
                                    print(f">>> [ab] Mastery reached. Jumping to Test ({TEST_TABLE_START})...")
                                    dynamic_RL34959.stop_everything_in_learning_and_go_to_implementation_phase = 1
                                    RL_can_start_implementation_phase_from_the_last_table = 1
                                    dynamic_RL34959.implement = 1
                                    next_table_number = TEST_TABLE_START
                                    dynamic_RL34959.sucess_times = 0
                                    dynamic_RL34959.curriculum_converged = 0
                            elif scenario_pattern in {"aba", "abba", "abc"}:
                                if table_number < PHASE_B_START_TABLE:
                                    print(f">>> [{scenario_pattern}] Mastery of Phase A. Jumping to Phase B ({PHASE_B_START_TABLE})...")
                                    next_table_number = PHASE_B_START_TABLE
                                    dynamic_RL34959.sucess_times = 0
                                    dynamic_RL34959.curriculum_converged = 0
                                elif table_number >= MIN_PHASE_B_TRAIN:
                                    print(f">>> [{scenario_pattern}] Mastery of Phase B. Jumping to Test ({TEST_TABLE_START})...")
                                    dynamic_RL34959.stop_everything_in_learning_and_go_to_implementation_phase = 1
                                    RL_can_start_implementation_phase_from_the_last_table = 1
                                    dynamic_RL34959.implement = 1
                                    next_table_number = TEST_TABLE_START
                                    dynamic_RL34959.sucess_times = 0
                                    dynamic_RL34959.curriculum_converged = 0
                        if getattr(dynamic_RL34959, "implement", 0) == 0 and table_number >= TRAIN_TABLE_MAX:
                            if train_only:
                                print(f">>> TRAIN_ONLY COMPLETE: reached train boundary ({TRAIN_TABLE_MAX}).")
                                _touch_stop_flag()
                                return
                            print(f">>> FORCE SWITCH: Reached table_number {TRAIN_TABLE_MAX}. Jumping to Test ({TEST_TABLE_START})...")
                            dynamic_RL34959.stop_everything_in_learning_and_go_to_implementation_phase = 1
                            RL_can_start_implementation_phase_from_the_last_table = 1
                            dynamic_RL34959.implement = 1
                            next_table_number = TEST_TABLE_START
                            dynamic_RL34959.sucess_times = 0
                            dynamic_RL34959.curriculum_converged = 0
                        table_number = next_table_number
                        if getattr(dynamic_RL34959, "implement", 0) == 0 and table_number > TRAIN_TABLE_MAX:
                            table_number = TRAIN_TABLE_MAX
                except SystemExit:
                    raise
                except Exception:
                    if Intermodal_ALNS34959.add_RL == 0:
                        if Intermodal_ALNS34959.ALNS_greedy_under_unknown_duration_assume_duration == 0:
                            table_number -= 1
                        elif Intermodal_ALNS34959.ALNS_greedy_under_unknown_duration_assume_duration == 3 and len(
                                ALNS_reward_list_in_implementation) > Intermodal_ALNS34959.number_of_training:
                            if start_from_end_table == 0:
                                table_number = TEST_TABLE_START
                                start_from_end_table = 1
                            else:
                                table_number -= 1
                        else:
                            table_number += 1
if __name__ == '__main__':
    main(approach)

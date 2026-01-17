#!/usr/bin/env Python
# coding=utf-8
# import concurrent.futures
import Intermodal_ALNS34959
import dynamic_RL34959
#import dynamic_RL_online_insertion
import pandas as pd
import os
import time
# haven't done: set the initial solution as original route, and detect which request is changed, and check which part can't be removed

def Intermodal_ALNS_function(request_number_in_R):
    global dynamic_end
    Intermodal_ALNS34959.real_main(3, 0, request_number_in_R)

    # data_path = 'C:/Users/yimengzhang/OneDrive/桌面/Intermodal_EGS_data_dynamic_new_requests.xlsx'
    while True:
        try:
            request_number_in_R = Intermodal_ALNS34959.request_number_in_R
            data_path = Intermodal_ALNS34959.data_path
            break
        except:
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
            dynamic_RL34959.main('DQN', 'barge')
        elif approach == 2:
            dynamic_RL_online_insertion.main('DQN', 'barge')
        else:
            Intermodal_ALNS_function(request_number_in_R)
    else:
        if approach == 1:
            dynamic_RL34959.main('DQN', 'barge')
        else:
            reward_list_in_implementation, removal_reward_list_in_implementation, removal_state_list_in_implementation, removal_action_list_in_implementation, insertion_reward_list_in_implementation, insertion_state_list_in_implementation, insertion_action_list_in_implementation = [], [], [], [], [], [], []
            ALNS_reward_list_in_implementation, ALNS_removal_reward_list_in_implementation,  ALNS_removal_action_list_in_implementation, ALNS_insertion_reward_list_in_implementation, ALNS_insertion_action_list_in_implementation = [], [], [], [], []
            table_number = 0 
            start_from_end_table = 0
            while True:
                Intermodal_ALNS_function(request_number_in_R)
                try:
                    if dynamic_RL34959.implement == 1:
                        if start_from_end_table == 0:
                            table_number = 499
                            RL_can_start_implementation_phase_from_the_last_table = 1
                            start_from_end_table = 1
                        else:
                            table_number -= 1
                    else:
                        table_number += 1
                except:
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

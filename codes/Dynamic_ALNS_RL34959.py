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

def Intermodal_ALNS_function(distribution_name='default'):
    """
    ALNS函数包装器

    Args:
        distribution_name: 不确定性事件分布配置名称
    """
    global dynamic_end
    Intermodal_ALNS34959.real_main(3, 0, distribution_name)

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
            try:
                if ' ' in key:
                    # 旧格式：R_5_123 或 R_5_123 (2)
                    blank_index = key.rfind(' ')
                    time_str = key.replace(prefix + '_', '')[0:blank_index - len(prefix) - 1]
                    unexpected_times.append(int(time_str))
                elif key.endswith('_events'):
                    # 新格式：R_5_events - 从events表中提取时间信息
                    # 从事件表中读取第一个事件的时间
                    try:
                        events_df = pd.read_excel(Data, sheet_name=key)
                        if len(events_df) > 0 and 'duration' in events_df.columns:
                            # duration是列表格式 [start, end]，取start时间
                            first_duration = events_df.iloc[0]['duration']
                            if isinstance(first_duration, str) and first_duration.startswith('[') and first_duration.endswith(']'):
                                # 解析字符串格式的列表 "[start, end]"
                                import ast
                                duration_list = ast.literal_eval(first_duration)
                                if isinstance(duration_list, list) and len(duration_list) >= 1:
                                    unexpected_times.append(int(duration_list[0]))
                            elif isinstance(first_duration, list) and len(first_duration) >= 1:
                                # 已经是列表格式
                                unexpected_times.append(int(first_duration[0]))
                    except Exception as e:
                        print(f"警告: 无法从事件表 {key} 提取时间信息: {e}")
                        continue
                else:
                    # 尝试直接解析为时间（旧格式兼容）
                    time_str = key.replace(prefix + '_', '')
                    if time_str.isdigit():
                        unexpected_times.append(int(time_str))
                    else:
                        print(f"警告: 无法解析表名中的时间信息: {key}")
                        continue
            except ValueError as e:
                print(f"警告: 表名 {key} 解析失败: {e}")
                continue
            except Exception as e:
                print(f"警告: 处理表名 {key} 时发生未知错误: {e}")
                continue
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
            Intermodal_ALNS34959.real_main(3, t, distribution_name)

    #another way of dynamic is optimizing only the urgent parts of requests, maybe better than this way
def main(approach, distribution_name='default'):
    """
    ALNS-RL混合算法主函数

    Args:
        approach: 优化方法编号
        distribution_name: 不确定性事件分布配置名称
    """
    global RL_can_start_implementation_phase_from_the_last_table, ALNS_calculates_average_duration_list, ALNS_reward_list_in_implementation, ALNS_removal_reward_list_in_implementation,  ALNS_removal_action_list_in_implementation, ALNS_insertion_reward_list_in_implementation, ALNS_insertion_action_list_in_implementation, table_number, reward_list_in_implementation, removal_reward_list_in_implementation, removal_state_list_in_implementation, removal_action_list_in_implementation, insertion_reward_list_in_implementation, insertion_state_list_in_implementation, insertion_action_list_in_implementation
    RL_can_start_implementation_phase_from_the_last_table = 0
    ALNS_calculates_average_duration_list = []
    combine_insertion_and_removal_operators = 1
    if combine_insertion_and_removal_operators == 0:
        if approach == 1:
            dynamic_RL34959.main('DQN', 'barge')
        elif approach == 2:
            dynamic_RL_online_insertion.main('DQN', 'barge')
        else:
            Intermodal_ALNS_function(distribution_name)
    else:
        if approach == 1:
            dynamic_RL34959.main('DQN', 'barge')
        else:
            reward_list_in_implementation, removal_reward_list_in_implementation, removal_state_list_in_implementation, removal_action_list_in_implementation, insertion_reward_list_in_implementation, insertion_state_list_in_implementation, insertion_action_list_in_implementation = [], [], [], [], [], [], []
            ALNS_reward_list_in_implementation, ALNS_removal_reward_list_in_implementation,  ALNS_removal_action_list_in_implementation, ALNS_insertion_reward_list_in_implementation, ALNS_insertion_action_list_in_implementation = [], [], [], [], []
            table_number = 0 
            start_from_end_table = 0
            while True:
                print(f"调试: 准备执行 ALNS function，当前 table_number = {table_number}")
                Intermodal_ALNS_function(distribution_name)
                try:
                    if dynamic_RL34959.implement == 1:
                        if start_from_end_table == 0:
                            table_number = 499
                            RL_can_start_implementation_phase_from_the_last_table = 1
                            start_from_end_table = 1
                        else:
                            table_number -= 1
                            # 添加边界检查：确保table_number不小于0
                            if table_number < 0:
                                print(f"调试: table_number {table_number} 小于0，重置为99")
                                table_number = 99
                    else:
                        table_number += 1
                        # 添加边界检查：table_number范围应为0-99
                        if table_number >= 100:
                            print(f"调试: table_number {table_number} 超出范围[0, 99]，重置为0")
                            table_number = 0
                except:
                    if Intermodal_ALNS34959.add_RL == 0:
                        if Intermodal_ALNS34959.ALNS_greedy_under_unknown_duration_assume_duration == 0:
                            table_number -= 1
                            # 添加边界检查：确保table_number不小于0
                            if table_number < 0:
                                print(f"调试: table_number {table_number} 小于0，重置为99")
                                table_number = 99
                        elif Intermodal_ALNS34959.ALNS_greedy_under_unknown_duration_assume_duration == 3 and len(
                                ALNS_reward_list_in_implementation) > Intermodal_ALNS34959.number_of_training:
                            if start_from_end_table == 0:
                                table_number = 999
                                start_from_end_table = 1
                            else:
                                table_number -= 1
                                # 添加边界检查：确保table_number不小于0
                                if table_number < 0:
                                    print(f"调试: table_number {table_number} 小于0，重置为99")
                                    table_number = 99
                        else:
                            table_number += 1
                            # 添加边界检查：table_number范围应为0-99
                            if table_number >= 100:
                                print(f"调试: table_number {table_number} 超出范围[0, 99]，重置为0")
                                table_number = 0
if __name__ == '__main__':
    main(approach)



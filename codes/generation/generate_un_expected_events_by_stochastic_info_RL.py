import pandas as pd
import numpy as np
import random
import os.path
from openpyxl import load_workbook
import matplotlib.pyplot as plt
data_path = "/data/yimeng/Case study/Intermodal_EGS_data_all.xlsx"
Data = pd.ExcelFile(data_path)
wb = load_workbook(data_path, read_only=True)
N = pd.read_excel(Data, 'N')
T = pd.read_excel(Data, 'T')
K = pd.read_excel(Data, 'K')
o = pd.read_excel(Data, 'o')
target_initial_routes = 1
if target_initial_routes == 1:
    #first I need to get the initial routes
    for r in [5, 10, 20, 30, 50, 100]:
        #exp number 12793, 12792, 12794
        exp_numbers = {5:12793, 10:12792, 20:12794, 30: 12816, 50: 12817, 100: 12818}

        routes_path = "/data/yimeng/Figures/experiment" + str(exp_numbers[r]) + "/percentage0parallel_number9dynamic0/best_routespercentage0parallel_number9dynamic0_" + str(exp_numbers[r]) + ".xlsx"
        xls = pd.ExcelFile(routes_path)
        routes = pd.read_excel(xls, None, index_col=0)
        used_ks = []
        for k in routes.keys():
            if len(routes[k].loc[0]) > 2:
                used_ks.append(k)
        terminal_arrival_time = []
        for k in used_ks:
            route = routes[k]
            print(k)
            if 'Barge' in k:
                mode = 1

            elif 'Train' in k:
                mode = 2
            elif 'Truck' in k:
                mode = 3
            else:
                print('error')
            for column_index in route.columns[1:-2]:
                terminal_arrival_time.append([route[column_index][0],route[column_index][1], mode])
        terminal_arrival_time_array = np.array(terminal_arrival_time)
        terminal_arrival_time_array = terminal_arrival_time_array[np.argsort(terminal_arrival_time_array[:, 1])]
        R = pd.read_excel(Data, 'R_' + str(r))
        basic_basic_path = "/home/yimeng/Uncertainties Dynamic planning under unexpected events/plot_distribution_targetInstances_disruption_log_mu_1_1_not_time_dependent"
        isExist = os.path.exists(basic_basic_path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(basic_basic_path)
        basic_path = basic_basic_path + "/R" + str(
            r)
        isExist = os.path.exists(basic_path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(basic_path)

        for table in range(1000):
            end_time = 0
            uncertainty_index = 0
            output_path = basic_path + "/" + "Intermodal_EGS_data_dynamic_congestion" + str(
                table) + ".xlsx"
            with pd.ExcelWriter(output_path) as writer:  # doctest: +SKIP
                N.to_excel(writer, sheet_name='N', index=False)
                R.to_excel(writer, sheet_name='R_' + str(r), index=False)
                T.to_excel(writer, sheet_name='T', index=False)
                K.to_excel(writer, sheet_name='K', index=False)
                o.to_excel(writer, sheet_name='o', index=False)
                for _ in range(len(terminal_arrival_time_array)):
                    # start_time = int(random.choices(range(1, 3))[0])

                    unexpected_events = pd.DataFrame(
                        columns=['uncertainty_index', 'type', 'location_type', 'vehicle', 'location', 'duration',
                                 'mode'])
                    mode_terminal_pairs = []
                    location = terminal_arrival_time_array[_,0]
                    start_time = int(terminal_arrival_time_array[_,1])
                    if start_time < end_time:
                        #to make the durations of events not overlap, and here the end time is last event's end time
                        continue
                    mode = terminal_arrival_time_array[_,2]
                    if [location, mode] in mode_terminal_pairs:
                        continue
                    #low impact one is / 5
                    #high impact * 5
                    #medium impact nothing
                    #high_medium *3
                    # mu, sigma = start_time % 24 * 3, 1
                    terminal_dependent = 0
                    if terminal_dependent == 1:
                        if location < 5:
                            mu, sigma = 80, 20  # mean and standard deviation
                        else:
                            mu, sigma = 5, 1  # mean and standard deviation
                    else:
                        mu, sigma = 1, 1  # mean and standard deviation
                    duration_ = max(0, int(np.random.lognormal(mu, sigma)))
                    save_plot_path = basic_path + '/distribution_' + 'start_time' + str(start_time) + 'mu' + str(mu) + 'sigma' + str(
                        sigma) + '.pdf'
                    if not os.path.exists(save_plot_path):
                        s = np.random.lognormal(mu, sigma, 1000)
                        normal_plot = 0
                        if normal_plot == 1:
                            count, bins, ignored = plt.hist(s, 30, density=True)
                            plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
                                     np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
                                     linewidth=2, color='r')
                            plt.xlabel('Duration (h)')
                            plt.ylabel('Probability')
                            plt.title('Durations under normal distribution (mu=' + str(mu) + ', sigma=' + str(sigma) + ')')
                        else:
                            count, bins, ignored = plt.hist(s, 100, density=True, align='mid')

                            x = np.linspace(min(bins), max(bins), 10000)

                            pdf = (np.exp(-(np.log(x) - mu) ** 2 / (2 * sigma ** 2))

                                   / (x * sigma * np.sqrt(2 * np.pi)))

                            plt.plot(x, pdf, linewidth=2, color='r')

                            plt.axis('tight')
                            plt.xlabel('Duration (h)')
                            plt.ylabel('Probability')
                            plt.title(
                                'Durations under lognormal distribution (mu=' + str(mu) + ', sigma=' + str(sigma) + ')')
                        # plt.show()
                        plt.savefig(save_plot_path,
                            format='pdf', bbox_inches='tight')
                        plt.close()

                    if duration_ == 0:
                        continue

                    end_time = start_time + duration_
                    duration = [start_time, end_time]

                    mode_terminal_pairs.append([location, mode])
                    for type in ['congestion', 'congestion_finish']:
                        new_row = pd.Series(
                            data={'uncertainty_index': uncertainty_index, 'type': type, 'location_type': 'node',
                                  'vehicle': -1, 'location': location, 'duration': duration, 'mode': mode})
                        unexpected_events = unexpected_events.append(new_row, ignore_index=True)

                    uncertainty_index += 1
                    if _ < len(terminal_arrival_time_array)-1:
                        if int(terminal_arrival_time_array[_+1,1]) != start_time:
                            # with pd.ExcelWriter(output_path) as writer:  # doctest: +SKIP

                            unexpected_events.to_excel(writer, sheet_name='R_' + str(r) + '_' + str(start_time) + ' (2)',
                                                       index=False)
                        # start_time = end_time + random.choices(range(10))[0]

else:
    for r in [5, 10, 20, 30, 50, 100]:
        R = pd.read_excel(Data, 'R_' + str(r))

        for table in range(1000):

            start_time = int(random.choices(range(1,3))[0])
            uncertainty_index = 0
            output_path = "/home/yimeng/Uncertainties Dynamic planning under unexpected events/Instances/" + "R" + str(r) + "/" + "Intermodal_EGS_data_dynamic_congestion" + str(
                table) + ".xlsx"
            with pd.ExcelWriter(output_path) as writer:  # doctest: +SKIP
                N.to_excel(writer, sheet_name='N', index=False)
                R.to_excel(writer, sheet_name='R_' + str(r), index=False)
                T.to_excel(writer, sheet_name='T', index=False)
                K.to_excel(writer, sheet_name='K', index=False)
                o.to_excel(writer, sheet_name='o', index=False)
                for _ in range(10):
                    unexpected_events = pd.DataFrame(
                        columns=['uncertainty_index', 'type', 'location_type', 'vehicle', 'location', 'duration', 'mode'])
                    mode_terminal_pairs = []
                    for __ in range(10):
                        location = random.choices(range(10))[0]
                        mode = random.choices([0, 1, 2])[0]
                        if [location, mode] in mode_terminal_pairs:
                            continue
                        duration_ = max(0,int(np.random.lognormal(start_time % 24 / 5, 1)))
                        if duration_ == 0:

                            continue

                        end_time = start_time + duration_
                        duration = [start_time, end_time]

                        mode_terminal_pairs.append([location,mode])
                        for type in ['congestion', 'congestion_finish']:
                            new_row = pd.Series(data={'uncertainty_index': uncertainty_index, 'type': type, 'location_type': 'node', 'vehicle': -1, 'location': location, 'duration': duration, 'mode':mode})
                            unexpected_events = unexpected_events.append(new_row, ignore_index=True)

                        uncertainty_index += 1
                        # with pd.ExcelWriter(output_path) as writer:  # doctest: +SKIP

                    unexpected_events.to_excel(writer, sheet_name='R_' + str(r) + '_' + str(start_time) + ' (2)', index=False)
                    start_time = end_time + random.choices(range(10))[0]
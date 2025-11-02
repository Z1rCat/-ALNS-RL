#!/usr/bin/env Python
# coding=utf-8
import concurrent.futures
import threading
import Dynamic_ALNS_RL34959
import dynamic_RL34959
import pandas as pd
import os
import time
import warnings
import sys
import torch
from GPUtil import showUtilization as gpu_usage
from numba import cuda

def free_gpu_cache(GPU_number):
    print("Initial GPU Usage")
    gpu_usage()

    torch.cuda.empty_cache()

    cuda.select_device(GPU_number)
    cuda.close()
    cuda.select_device(GPU_number)

    print("GPU Usage after emptying the cache")
    gpu_usage()
#while True:
 #   try:

  #      free_gpu_cache(0); print('use GPU 0')
   #     break
    #except:
     #   try:
      #      free_gpu_cache(1); print('use GPU 1')
       #     break
        #except:
         #   continue


warnings.filterwarnings("ignore")
# from multiprocessing import Process
#coordinator is 0
add_RL =1 
combine_insertion_and_removal_operators = 1
if combine_insertion_and_removal_operators == 1:
    parallel_number = list(range(0,2))
else:
    parallel_number = list(range(0, 3))
# number_of_approachs = pd.DataFrame([len(parallel_number)-1])
# with pd.ExcelWriter(number_of_approachs_path) as writer:  # doctest: +SKIP
#     number_of_approachs.to_excel(writer, sheet_name='number_of_approachs', index=False)

# parallel_number.append(-1)
# def runInParallel(*fns):
#   proc = []
#   for fn in fns:
#     p = Process(target=fn)
#     p.start()
#     proc.append(p)
#   for p in proc:
#     p.join()


def main():
    # runInParallel(dynamic_RL34959.main(), Dynamic_ALNS_RL34959.main())
    # threading.Thread(target=dynamic_RL34959.main()).start()
    # threading.Thread(target=Dynamic_ALNS_RL34959.main()).start()
    #if I set max_workers=len(parallel_number), then child processors can't report error
    # with concurrent.futures.ProcessPoolExecutor(max_workers = 1) as executor:
    # with concurrent.futures.ProcessPoolExecutor(max_workers = 3) as executor:
    if os.path.exists('34959.txt'):
        os.remove('34959.txt')
    if add_RL == 0:
        Dynamic_ALNS_RL34959.main(0)
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # chunksize, extra = divmod(len(parallel_number), executor._max_workers * 4)
            #####map####
            # for number, output in zip(parallel_number, executor.map(Intermodal_ALNS.real_main, parallel_number)):
            #     print('%d is prime: %s' % (number, output))
            #####map####

            ####submit###
            # to_do = []

            futures = {executor.submit(Dynamic_ALNS_RL34959.main, approach): approach for approach in parallel_number}
            for future in concurrent.futures.as_completed(futures):
                print('finish')
                with open('34959.txt', 'w') as f:
                    f.write('do not wait anymore')
                future.result() #stop here, and type this line at commend line, then exception is raised

                # result = futures[future]
                # try:
                #     data = future.result()
                # except Exception as exc:
                #     print('%r generated an exception: %s' % (result, exc))
                #     sys.exit(-1)
                # else:
                #     print(data)
            ####submit###
            # for parallel_number in range(0,2):
            #     Intermodal_ALNS.parallel_number=parallel_number
            #     executor.submit(Intermodal_ALNS)

cprofile = 0
if cprofile == 1:
    import cProfile
    import pstats
    import io

    pr = cProfile.Profile()
    pr.enable()

if __name__ == '__main__':
    main()

if cprofile == 1:
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    ps.print_stats()

    with open('/data/yimeng/logs/RL34959_save_profile.txt', 'w+', encoding="utf-8") as f:
        f.write(s.getvalue())




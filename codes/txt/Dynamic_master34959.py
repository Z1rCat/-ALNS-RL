#!/usr/bin/env Python
# coding=utf-8
import concurrent.futures
import os
import time
import warnings
import sys
import argparse
import subprocess
import Dynamic_ALNS_RL34959
import dynamic_RL34959

# ================= USER CONFIGURATION =================
# 这是一个"特洛伊木马"文件夹名，必须与 ALNS 代码中写死的读取路径完全一致
LEGACY_FOLDER_NAME = "plot_distribution_targetInstances_disruption_mix_mu_5_40_terminal_dependent_not_time_dependent"
# ======================================================

warnings.filterwarnings("ignore")

# 参数定义
add_RL = 1 
combine_insertion_and_removal_operators = 1
if combine_insertion_and_removal_operators == 1:
    parallel_number = list(range(0,2))
else:
    parallel_number = list(range(0, 3))

def select_distribution_mode():
    """
    交互式菜单，让用户选择实验环境
    """
    print("\n" + "="*50)
    print(" ALNS-RL EXPERIMENT LAUNCHER ".center(50, "="))
    print("="*50)
    print("Please select the Uncertainty Distribution:")
    print("  [1] Mixed v1        (25% High Stress + 75% Chaos) -> Recommended for RL Training")
    print("  [2] Stress Test     (100% High Stress: Mean=120)  -> To verify RL triggering")
    print("  [3] Chaos Uniform   (100% Random: 10-100 min)     -> To test generalization")
    print("  [4] Curriculum Easy (50% Easy + 50% Medium)       -> Gentle start")
    print("  [5] Baseline Legacy (LogNormal: Mean~20)          -> Original behavior")
    print("="*50)
    
    mapping = {
        '1': 'mixed_v1',
        '2': 'stress_test',
        '3': 'chaos_uniform',
        '4': 'curriculum_easy',
        '5': 'baseline_legacy'
    }
    
    while True:
        choice = input("Enter choice [1-5] (Default 1): ").strip()
        if choice == "":
            return "mixed_v1"
        if choice in mapping:
            return mapping[choice]
        print("Invalid selection. Try again.")

def run_generator(dist_name):
    """
    调用生成器脚本覆盖旧数据
    """
    print(f"\n>>> [Master] Phase 1: Generating '{dist_name}' data...")
    print(f"    Target Legacy Folder: {LEGACY_FOLDER_NAME}")
    
    generator_script = os.path.join(os.path.dirname(__file__), "generate_mixed_parallel.py")
    if not os.path.exists(generator_script):
        print(f"!!! Error: Generator script not found at {generator_script}")
        sys.exit(1)
        
    cmd = [
        "python", generator_script,
        "--dist_name", dist_name,
        "--target_folder", LEGACY_FOLDER_NAME,
        "--total_files", "1000"
    ]
    
    try:
        # 调用子进程，实时输出会显示在控制台
        subprocess.run(cmd, check=True)
        print(">>> [Master] Generation Successful. Legacy files overwritten.")
    except subprocess.CalledProcessError:
        print("!!! [Master] Generator Failed. Please check the error messages above.")
        sys.exit(1)

def main():
    # 1. 交互式选择
    dist_name = select_distribution_mode()
    
    # 2. 生成数据 (鸠占鹊巢)
    run_generator(dist_name)
    
    # 3. 启动旧系统
    print("\n>>> [Master] Phase 2: Starting Legacy Simulation (ALNS + RL)...")
    print("="*50)
    
    if os.path.exists('34959.txt'):
        os.remove('34959.txt')
        
    if add_RL == 0:
        Dynamic_ALNS_RL34959.main(0)
    else:
        # 使用 ThreadPool 同时启动 ALNS 和 RL 线程
        # 注意：这里不传递 dist_name，因为 ALNS/RL 会去读已经被我们覆盖掉的默认路径
        with concurrent.futures.ThreadPoolExecutor() as executor:
            print(f"   Launching threads for approaches: {parallel_number}")
            futures = {executor.submit(Dynamic_ALNS_RL34959.main, approach): approach for approach in parallel_number}
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    data = future.result()
                except Exception as exc:
                    print(f'Thread generated an exception: {exc}')
                    # sys.exit(-1) # Optional: abort if one thread fails

if __name__ == '__main__':
    main()
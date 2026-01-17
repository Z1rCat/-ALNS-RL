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
import rl_logging
import datetime
import traceback

# ================= USER CONFIGURATION =================
# 建议与具体文件夹名称对齐，通常是 ALNS 能够识别的文件夹名
LEGACY_FOLDER_NAME = "plot_distribution_targetInstances_disruption_mix_mu_5_40_terminal_dependent_not_time_dependent"
# ======================================================

warnings.filterwarnings("ignore")
if os.name == "nt":
    try:
        os.system("chcp 65001 >nul")
    except Exception:
        pass
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# 默认 R 值
DEFAULT_REQUEST_NUMBER = 5

# 全局参数
add_RL = 1
combine_insertion_and_removal_operators = 1
if combine_insertion_and_removal_operators == 1:
    parallel_number = list(range(0, 2))
else:
    parallel_number = list(range(0, 3))

DIST_DISPLAY = {
    'mixed_v1': '混合分布 v1 (25% 正态 + 75% 指数)',
    'stress_test': '压力测试 (100% 拥堵时间大于120)',
    'chaos_uniform': '混沌测试 (100% 均匀 10-100)',
    'curriculum_easy': '课程学习-简单 (50% 正常 + 50% 轻微)',
    'baseline_legacy': '基准测试 (所有拥堵时间设为20)',
    'normal_mix_8_80_50_50': '正态混合 50/50 (均值8 / 80)',
    'lognormal_mix_8_80_50_50': '对数正态 50/50 (均值8 / 80)',
    'normal_mix_8_80_75_25': '正态混合 75/25 (均值8 / 80)',
    'lognormal_mix_8_80_75_25': '对数正态 75/25 (均值8 / 80)',
    'normal_mix_80_8_25_75': '正态混合 25/75 (均值80 -> 8)',
    'normal_mix_80_8_50_50': '正态混合 50/50 (均值80 -> 8)',
    'normal_mix_80_8_75_25': '正态混合 75/25 (均值80 -> 8)',
    'lognormal_mix_80_8_25_75': '对数正态 25/75 (均值80 -> 8)',
    'lognormal_mix_80_8_50_50': '对数正态 50/50 (均值80 -> 8)',
    'lognormal_mix_9_30_3_30_30_40': '复合分布 30/30/40 (均值9 / 30 / 3)'
}


def select_request_number():
    """
    交互式选择请求数 R 值
    """
    print("")
    print("=" * 50)
    print(" 交互式选择请求数 R ".center(50, "="))
    print("=" * 50)
    print("  [5]   极速测试")
    print("  [10]  标准测试")
    print("  [20]  中等复杂度")
    print("  [30]  高负载测试")
    print("  [50]  压力测试")
    print("  [100] 极限极限测试")
    print("=" * 50)
    while True:
        choice = input(f"请输入请求数 R (默认 {DEFAULT_REQUEST_NUMBER}): ").strip()
        if choice == "":
            return DEFAULT_REQUEST_NUMBER
        try:
            r_val = int(choice)
            if r_val in [5, 10, 20, 30, 50, 100]:
                return r_val
        except ValueError:
            pass
        print("输入无效，请重新输入。")


def select_distribution_mode():
    """
    交互式选择拥堵事件生成的分布模式
    """
    print("")
    print("=" * 50)
    print(" ALNS-RL 分布模式选择 ".center(50, "="))
    print("=" * 50)
    print(" 请选择生成的分布：")
    print("  [1] 混合分布 v1         (25% 正态 + 75% 指数)")
    print("  [2] 压力测试            (100% 拥堵时间大于120)")
    print("  [3] 混沌测试            (100% 均匀 10-100)")
    print("  [4] 课程学习-简单       (50% 正常 + 50% 轻微)")
    print("  [5] 基准测试            (所有拥堵时间设为20)")
    print("  [6] 正态混合 50/50      (均值8 / 80)")
    print("  [7] 对数正态 50/50      (均值8 / 80)")
    print("  [8] 正态混合 75/25      (均值8 / 80)")
    print("  [9] 对数正态 75/25      (均值8 / 80)")
    print("  [10] 正态混合 25/75     (均值80 -> 8)")
    print("  [11] 正态混合 50/50     (均值80 -> 8)")
    print("  [12] 正态混合 75/25     (均值80 -> 8)")
    print("  [13] 对数正态 25/75     (均值80 -> 8)")
    print("  [14] 对数正态 50/50     (均值80 -> 8)")
    print("  [15] 复合分布 30/30/40  (均值9 / 30 / 3)")
    print("=" * 50)

    mapping = {
        '1': 'mixed_v1',
        '2': 'stress_test',
        '3': 'chaos_uniform',
        '4': 'curriculum_easy',
        '5': 'baseline_legacy',
        '6': 'normal_mix_8_80_50_50',
        '7': 'lognormal_mix_8_80_50_50',
        '8': 'normal_mix_8_80_75_25',
        '9': 'lognormal_mix_8_80_75_25',
        '10': 'normal_mix_80_8_25_75',
        '11': 'normal_mix_80_8_50_50',
        '12': 'normal_mix_80_8_75_25',
        '13': 'lognormal_mix_80_8_25_75',
        '14': 'lognormal_mix_80_8_50_50',
        '15': 'lognormal_mix_9_30_3_30_30_40'
    }

    while True:
        choice = input("请输入编号 [1-15] (默认 1): ").strip()
        if choice == "":
            return "mixed_v1"
        if choice in mapping:
            return mapping[choice]
        print("输入无效，请重新输入。")


def select_run_count():
    print("")
    print("=" * 50)
    print(" 选择运行轮数 ".center(50, "="))
    print("=" * 50)
    while True:
        choice = input("请输入要运行的总轮数 (默认 1): ").strip()
        if choice == "":
            return 1
        try:
            count = int(choice)
            if count >= 1:
                return count
        except ValueError:
            pass
        print("输入无效，请重新输入。")


def collect_batch_plan(run_count):
    plan = []
    for idx in range(run_count):
        print("")
        print("-" * 50)
        print(f"配置第 {idx + 1} 轮运行")
        print("-" * 50)
        dist_name = select_distribution_mode()
        request_number = select_request_number()
        plan.append((dist_name, request_number))
    return plan


def run_generator(dist_name, request_number):
    """
    运行生成器生成分布
    """
    dist_label = DIST_DISPLAY.get(dist_name, "未知分布")
    print("")
    print(f">>> [阶段1] 正在生成随机分布事件 ({dist_label})")
    print(f"    目标文件夹: {LEGACY_FOLDER_NAME}")

    generator_script = os.path.join(os.path.dirname(__file__), "generate_mixed_parallel.py")
    if not os.path.exists(generator_script):
        print(f"错误：找不到生成器脚本 {generator_script}")
        sys.exit(1)

    cmd = [
        sys.executable, generator_script,
        "--dist_name", dist_name,
        "--target_folder", LEGACY_FOLDER_NAME,
        "--total_files", "1000",
        "--request_numbers", str(request_number)
    ]

    try:
        subprocess.run(cmd, check=True)
        print(">>> 生成器运行成功，旧数据已覆盖。")
    except subprocess.CalledProcessError:
        print(">>> 错误：分布生成器运行失败。")
        sys.exit(1)


def run_simulation(request_number):
    print("")
    print(">>> [阶段2] 正在启动主仿真程序 (ALNS + RL)...")
    print("=" * 50)

    if os.path.exists('34959.txt'):
        os.remove('34959.txt')

    if add_RL == 0:
        Dynamic_ALNS_RL34959.main(0)
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            print(f"   线程并行编号: {parallel_number}")
            futures = {executor.submit(Dynamic_ALNS_RL34959.main, approach, request_number): approach for approach in parallel_number}

            for future in concurrent.futures.as_completed(futures):
                try:
                    data = future.result()
                except Exception as exc:
                    print(f"线程任务产生异常: {exc}")
                    print(traceback.format_exc())


def run_single(dist_name, request_number):
    run_id = datetime.datetime.now().strftime(f"run_%Y%m%d_%H%M%S_R{request_number}_{dist_name}")
    rl_logging.set_run_dir(run_id)
    rl_logging.write_meta({"distribution": dist_name, "request_number": request_number})

    run_generator(dist_name, request_number)
    run_simulation(request_number)


def run_single_in_subprocess(dist_name, request_number):
    script_path = os.path.abspath(__file__)
    cmd = [
        sys.executable, script_path,
        "--dist_name", dist_name,
        "--request_number", str(request_number)
    ]
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dist_name", type=str)
    parser.add_argument("--request_number", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dist_name and args.request_number:
        run_single(args.dist_name, args.request_number)
        return

    run_count = select_run_count()
    if run_count <= 1:
        dist_name = select_distribution_mode()
        request_number = select_request_number()
        run_single(dist_name, request_number)
        return

    plan = collect_batch_plan(run_count)
    for idx, (dist_name, request_number) in enumerate(plan, start=1):
        dist_label = DIST_DISPLAY.get(dist_name, "未知分布")
        print("")
        print("=" * 50)
        print(f">>> [批处理] 第 {idx}/{run_count} 轮: 分布[{dist_label}] | R={request_number}")
        print("=" * 50)
        run_single_in_subprocess(dist_name, request_number)


if __name__ == '__main__':
    main()
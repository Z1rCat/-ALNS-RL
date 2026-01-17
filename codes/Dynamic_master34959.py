#!/usr/bin/env Python
# coding=utf-8
import concurrent.futures
import os
import time
import warnings
import sys
import argparse
import json
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

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "distribution_config.json")

DEFAULT_DISTRIBUTIONS = [
    {"name": "S1_1", "pattern": "random_mix", "means": {"A": 9, "B": 90}, "display": "S1_1 random mix A=9 B=90"},
    {"name": "S1_2", "pattern": "random_mix", "means": {"A": 3, "B": 30}, "display": "S1_2 random mix A=3 B=30"},
    {"name": "S1_3", "pattern": "random_mix", "means": {"A": 6, "B": 60}, "display": "S1_3 random mix A=6 B=60"},
    {"name": "S2_1", "pattern": "aba", "means": {"A": 9, "B": 90}, "display": "S2_1 ABA A=9 B=90"},
    {"name": "S2_2", "pattern": "aba", "means": {"A": 90, "B": 9}, "display": "S2_2 ABA A=90 B=9"},
    {"name": "S2_3", "pattern": "aba", "means": {"A": 3, "B": 30}, "display": "S2_3 ABA A=3 B=30"},
    {"name": "S2_4", "pattern": "aba", "means": {"A": 30, "B": 3}, "display": "S2_4 ABA A=30 B=3"},
    {"name": "S2_5", "pattern": "aba", "means": {"A": 6, "B": 60}, "display": "S2_5 ABA A=6 B=60"},
    {"name": "S2_6", "pattern": "aba", "means": {"A": 60, "B": 6}, "display": "S2_6 ABA A=60 B=6"},
    {"name": "S3_1", "pattern": "ab", "means": {"A": 9, "B": 90}, "display": "S3_1 OOD A=9 B=90"},
    {"name": "S3_2", "pattern": "ab", "means": {"A": 90, "B": 9}, "display": "S3_2 OOD A=90 B=9"},
    {"name": "S3_3", "pattern": "ab", "means": {"A": 3, "B": 30}, "display": "S3_3 OOD A=3 B=30"},
    {"name": "S3_4", "pattern": "ab", "means": {"A": 30, "B": 3}, "display": "S3_4 OOD A=30 B=3"},
    {"name": "S3_5", "pattern": "ab", "means": {"A": 6, "B": 60}, "display": "S3_5 OOD A=6 B=60"},
    {"name": "S3_6", "pattern": "ab", "means": {"A": 60, "B": 6}, "display": "S3_6 OOD A=60 B=6"},
    {"name": "S4_1", "pattern": "recall", "means": {"A": 9, "B": 90}, "display": "S4_1 recall A=9 B=90"},
    {"name": "S4_2", "pattern": "recall", "means": {"A": 90, "B": 9}, "display": "S4_2 recall A=90 B=9"},
    {"name": "S4_3", "pattern": "recall", "means": {"A": 3, "B": 30}, "display": "S4_3 recall A=3 B=30"},
    {"name": "S4_4", "pattern": "recall", "means": {"A": 30, "B": 3}, "display": "S4_4 recall A=30 B=3"},
    {"name": "S4_5", "pattern": "recall", "means": {"A": 6, "B": 60}, "display": "S4_5 recall A=6 B=60"},
    {"name": "S4_6", "pattern": "recall", "means": {"A": 60, "B": 6}, "display": "S4_6 recall A=60 B=6"},
    {"name": "S5_1", "pattern": "adaptation", "means": {"A": 9, "B": 90}, "display": "S5_1 adaptation A=9 B=90"},
    {"name": "S5_2", "pattern": "adaptation", "means": {"A": 90, "B": 9}, "display": "S5_2 adaptation A=90 B=9"},
    {"name": "S5_3", "pattern": "adaptation", "means": {"A": 3, "B": 30}, "display": "S5_3 adaptation A=3 B=30"},
    {"name": "S5_4", "pattern": "adaptation", "means": {"A": 30, "B": 3}, "display": "S5_4 adaptation A=30 B=3"},
    {"name": "S5_5", "pattern": "adaptation", "means": {"A": 6, "B": 60}, "display": "S5_5 adaptation A=6 B=60"},
    {"name": "S5_6", "pattern": "adaptation", "means": {"A": 60, "B": 6}, "display": "S5_6 adaptation A=60 B=6"},
    {"name": "S6_1", "pattern": "abc", "means": {"A": 9, "B": 90, "C": 30}, "display": "S6_1 ABC A=9 B=90 C=30"},
    {"name": "S6_2", "pattern": "abc", "means": {"A": 9, "B": 30, "C": 90}, "display": "S6_2 ABC A=9 B=30 C=90"},
    {"name": "S6_3", "pattern": "abc", "means": {"A": 90, "B": 30, "C": 9}, "display": "S6_3 ABC A=90 B=30 C=9"},
]

PHYSICAL_TOTAL_FILES = 500

def load_distribution_config():
    dist_entries = DEFAULT_DISTRIBUTIONS
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("distributions"), list):
            dist_entries = data["distributions"]
    except Exception:
        pass
    normalized = []
    for item in dist_entries:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        pattern = str(item.get("pattern", "")).strip()
        means = item.get("means", {})
        if not name or not pattern or not isinstance(means, dict):
            continue
        display = str(item.get("display", "")).strip()
        normalized.append({
            "name": name,
            "pattern": pattern,
            "means": means,
            "display": display,
        })
    return normalized or DEFAULT_DISTRIBUTIONS

def get_distribution_display_map():
    dist_map = {}
    for item in load_distribution_config():
        display = item.get("display") or item["name"]
        dist_map[item["name"]] = display
    return dist_map



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
    dist_entries = load_distribution_config()
    dist_display = get_distribution_display_map()
    print("")
    print("=" * 50)
    print(" ALNS-RL 分布模式选择 ".center(50, "="))
    print("=" * 50)
    print(" 请选择生成的分布：")
    for idx, item in enumerate(dist_entries, start=1):
        name = item["name"]
        label = dist_display.get(name, name)
        print(f"  [{idx}] {label}")
    print("=" * 50)

    mapping = {}
    for idx, item in enumerate(dist_entries, start=1):
        mapping[str(idx)] = item["name"]

    while True:
        choice = input(f"Choose [1-{len(dist_entries)}] or name (default 1): ").strip()
        if choice == "":
            return dist_entries[0]["name"] if dist_entries else "S1_1"
        if choice in mapping:
            return mapping[choice]
        choice_upper = choice.upper()
        if choice_upper in dist_display:
            return choice_upper
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


def select_worker_count():
    print("")
    print("=" * 50)
    print(" 设置 CPU 核心数 ".center(50, "="))
    print("=" * 50)
    print("  [Enter] 默认/自动")
    print("  [1]     单核")
    print("  [N]     N 核并行")
    print("=" * 50)
    while True:
        choice = input("请输入核心数(留空自动): ").strip()
        if choice == "":
            return None
        try:
            value = int(choice)
            if value >= 1:
                return value
        except ValueError:
            pass
        print("输入无效，请重新输入。")


def select_algorithm():
    print("")
    print("=" * 50)
    print(" 选择 RL 算法 ".center(50, "="))
    print("=" * 50)
    print("  [1] DQN")
    print("  [2] PPO")
    print("  [3] A2C")
    print("=" * 50)
    mapping = {"1": "DQN", "2": "PPO", "3": "A2C"}
    while True:
        choice = input("请选择算法 (默认 1=DQN): ").strip()
        if choice == "":
            return "DQN"
        if choice.upper() in mapping.values():
            return choice.upper()
        if choice in mapping:
            return mapping[choice]
        print("输入无效，请重新输入。")


def resolve_worker_count(args):
    if getattr(args, "workers", None) is not None:
        if args.workers < 1:
            print("workers 参数必须 >= 1，强制使用 1")
            return 1
        return args.workers
    if getattr(args, "single_core", False):
        return 1
    return None


def collect_batch_plan(run_count, algorithm):
    plan = []
    for idx in range(run_count):
        print("")
        print("-" * 50)
        print(f"配置第 {idx + 1} 轮运行")
        print("-" * 50)
        dist_name = select_distribution_mode()
        request_number = select_request_number()
        plan.append((dist_name, request_number, algorithm))
    return plan


def run_generator(dist_name, request_number, workers=None, target_folder=None):
    """
    运行生成器生成分布
    """
    dist_label = get_distribution_display_map().get(dist_name, dist_name)
    print("")
    print(f">>> [阶段1] 正在生成随机分布事件 ({dist_label})")
    if target_folder is None:
        target_folder = LEGACY_FOLDER_NAME
    print(f"    目标文件夹: {target_folder}")

    generator_script = os.path.join(os.path.dirname(__file__), "generate_mixed_parallel.py")
    if not os.path.exists(generator_script):
        print(f"错误：找不到生成器脚本 {generator_script}")
        sys.exit(1)

    cmd = [
        sys.executable, generator_script,
        "--dist_name", dist_name,
        "--target_folder", target_folder,
        "--total_files", str(PHYSICAL_TOTAL_FILES),
        "--request_numbers", str(request_number)
    ]
    if workers is not None:
        cmd.extend(["--workers", str(workers)])

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


def run_single(dist_name, request_number, workers=None, algorithm="DQN"):
    os.environ["SCENARIO_NAME"] = dist_name
    os.environ["RL_ALGORITHM"] = algorithm
    Dynamic_ALNS_RL34959.SCENARIO_NAME = dist_name
    Dynamic_ALNS_RL34959.RL_ALGORITHM = algorithm
    dynamic_RL34959.SCENARIO_NAME = dist_name
    run_id = datetime.datetime.now().strftime(f"run_%Y%m%d_%H%M%S_R{request_number}_{dist_name}")
    rl_logging.set_run_dir(run_id)
    run_data_dir = rl_logging.get_run_data_dir()
    os.environ["DYNAMIC_DATA_ROOT"] = str(run_data_dir)
    rl_logging.write_meta({
        "distribution": dist_name,
        "request_number": request_number,
        "generator_workers": workers if workers is not None else "auto",
        "algorithm": algorithm,
        "data_root": str(run_data_dir),
    })

    run_generator(dist_name, request_number, workers, str(run_data_dir))
    run_simulation(request_number)


def run_single_in_subprocess(dist_name, request_number, workers=None, algorithm="DQN"):
    script_path = os.path.abspath(__file__)
    cmd = [
        sys.executable, script_path,
        "--dist_name", dist_name,
        "--request_number", str(request_number),
        "--algorithm", algorithm,
    ]
    if workers is not None:
        cmd.extend(["--workers", str(workers)])
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dist_name", type=str)
    parser.add_argument("--request_number", type=int)
    parser.add_argument("--run_count", type=int)
    parser.add_argument("--algorithm", type=str, help="DQN/PPO/A2C")
    parser.add_argument("--workers", type=int, help="generator workers (1=single core)")
    parser.add_argument("--single_core", action="store_true", help="force generator single core")
    return parser.parse_args()


def main():
    args = parse_args()
    workers = resolve_worker_count(args)
    algorithm = args.algorithm.upper() if args.algorithm else None
    if algorithm is not None and algorithm not in {"DQN", "PPO", "A2C"}:
        print(f"未知算法 {algorithm}，回退为 DQN")
        algorithm = "DQN"

    if args.dist_name and args.request_number:
        run_count = args.run_count or 1
        if algorithm is None:
            algorithm = select_algorithm()
        if run_count <= 1:
            run_single(args.dist_name, args.request_number, workers, algorithm)
        else:
            for _ in range(run_count):
                run_single_in_subprocess(args.dist_name, args.request_number, workers, algorithm)
        return

    if workers is None:
        workers = select_worker_count()

    if algorithm is None:
        algorithm = select_algorithm()

    run_count = args.run_count if args.run_count is not None else select_run_count()
    if run_count <= 1:
        dist_name = select_distribution_mode()
        request_number = select_request_number()
        run_single(dist_name, request_number, workers, algorithm)
        return

    plan = collect_batch_plan(run_count, algorithm)
    for idx, (dist_name, request_number, algorithm) in enumerate(plan, start=1):
        dist_label = get_distribution_display_map().get(dist_name, dist_name)
        print("")
        print("=" * 50)
        print(f">>> [批量计划] 正在运行第 {idx}/{run_count} 轮: 分布模式[{dist_label}] | R={request_number}")
        print("=" * 50)
        run_single_in_subprocess(dist_name, request_number, workers, algorithm)



if __name__ == '__main__':
    main()

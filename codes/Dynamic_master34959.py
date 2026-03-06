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
from pathlib import Path
from core import Dynamic_ALNS_RL34959
from core import Intermodal_ALNS34959
from core import dynamic_RL34959
from core import rl_logging
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


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8") if streams else "utf-8"

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        self.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False


def stream_subprocess_output(cmd):
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    try:
        if proc.stdout is not None:
            for line in proc.stdout:
                sys.stdout.write(line)
        return proc.wait()
    finally:
        try:
            if proc.stdout is not None:
                proc.stdout.close()
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
    {"name": "O_10_90", "pattern": "ab", "means": {"A": 10, "B": 90}, "display": "O_10_90 OOD train A=10 test B=90"},
    {"name": "O_90_10", "pattern": "ab", "means": {"A": 90, "B": 10}, "display": "O_90_10 OOD train A=90 test B=10"},
    {"name": "O_30_80", "pattern": "ab", "means": {"A": 30, "B": 80}, "display": "O_30_80 OOD train A=30 test B=80"},
    {"name": "O_60_20", "pattern": "ab", "means": {"A": 60, "B": 20}, "display": "O_60_20 OOD train A=60 test B=20"},
    {"name": "O_10_120", "pattern": "ab", "means": {"A": 10, "B": 120}, "display": "O_10_120 OOD train A=10 test B=120"},
    {"name": "O_120_10", "pattern": "ab", "means": {"A": 120, "B": 10}, "display": "O_120_10 OOD train A=120 test B=10"},
    {"name": "G_10_90_50", "pattern": "abc", "means": {"A": 10, "B": 90, "C": 50}, "display": "G_10_90_50 generalization train A,B test C"},
    {"name": "G_10_40_90", "pattern": "abc", "means": {"A": 10, "B": 40, "C": 90}, "display": "G_10_40_90 generalization train A,B test C"},
    {"name": "G_40_80_10", "pattern": "abc", "means": {"A": 40, "B": 80, "C": 10}, "display": "G_40_80_10 generalization train A,B test C"},
    {"name": "G_30_60_90", "pattern": "abc", "means": {"A": 30, "B": 60, "C": 90}, "display": "G_30_60_90 generalization train A,B test C"},
    {"name": "F1_10_90", "pattern": "aba", "means": {"A": 10, "B": 90}, "display": "F1_10_90 forgetting train A,B test A"},
    {"name": "F1_90_10", "pattern": "aba", "means": {"A": 90, "B": 10}, "display": "F1_90_10 forgetting train A,B test A"},
    {"name": "F2_10_90", "pattern": "abba", "means": {"A": 10, "B": 90}, "display": "F2_10_90 forgetting train A,B test A,B"},
    {"name": "F2_30_80", "pattern": "abba", "means": {"A": 30, "B": 80}, "display": "F2_30_80 forgetting train A,B test A,B"},
    {"name": "R_10_90", "pattern": "random_mix", "means": {"A": 10, "B": 90}, "display": "R_10_90 random mix A=10 B=90"},
    {"name": "R_30_80", "pattern": "random_mix", "means": {"A": 30, "B": 80}, "display": "R_30_80 random mix A=30 B=80"},
]

PHYSICAL_TOTAL_FILES = 1000

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
            return dist_entries[0]["name"] if dist_entries else "R_10_30"
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
    print("  [4] DRCB (Drift-Robust Contextual Bandit)")
    print("  [5] LBKLAC (Latent Belief KL-Regularized Actor-Critic)")
    print("  [6] PPO_HAT (HAT: History-Attention Transform)")
    print("  [7] A2C_HAT (HAT: History-Attention Transform)")
    print("  [8] PPO_LSTM (RecurrentPPO + LSTM)")
    print("  [9] PPO_HAT_LSTM (HAT + RecurrentPPO + LSTM)")
    print("  [10] PPO_HAT_PDI (HAT + PPO + PDI)")
    print("  [11] PPO_HAT_MOE (HAT + MoE(K=2) + PPO)")
    print("  [12] A2C_HAT_MOE (HAT + MoE(K=2) + A2C)")
    print("  [13] QRDQN_CVAR (Distributional QRDQN + CVaR Inference)")
    print("  [14] BE_CVAR_DQN (Belief+Ensemble+CVaR DQN)")
    print("  [15] PPO_PROTOMEM (ProtoMem-PPO)")
    print("  [16] PPO_NEW (new PPO entry with versioning)")
    print("  [17] NOVA_EDRL (phase1->phase2->phase3->phase4 pipeline)")
    print("  [18] RARL (Robust Adversarial RL baseline)")
    print("  [19] PLR_UED (Prioritized Level Replay/UED baseline)")
    print("  [20] SABER_V0 (learnability-gated replay selector baseline)")
    print("  [21] CQL_DQN (Discrete Conservative Q-Learning)")
    print("  [22] CADM (Context-aware dynamics baseline)")
    print("=" * 50)
    mapping = {
        "1": "DQN",
        "2": "PPO",
        "3": "A2C",
        "4": "DRCB",
        "5": "LBKLAC",
        "6": "PPO_HAT",
        "7": "A2C_HAT",
        "8": "PPO_LSTM",
        "9": "PPO_HAT_LSTM",
        "10": "PPO_HAT_PDI",
        "11": "PPO_HAT_MOE",
        "12": "A2C_HAT_MOE",
        "13": "QRDQN_CVAR",
        "14": "BE_CVAR_DQN",
        "15": "PPO_PROTOMEM",
        "16": "PPO_NEW",
        "17": "NOVA_EDRL",
        "18": "RARL",
        "19": "PLR_UED",
        "20": "SABER_V0",
        "21": "CQL_DQN",
        "22": "CADM",
    }
    while True:
        choice = input("请选择算法 (默认 1=DQN): ").strip()
        if choice == "":
            return "DQN"
        choice_upper = choice.upper()
        if choice_upper == "HAT":
            return "PPO_HAT"
        if choice_upper == "CQL":
            return "CQL_DQN"
        if choice_upper == "SABER":
            return "SABER_V0"
        if choice_upper in {
            "PPO_HAT_MOE",
            "A2C_HAT_MOE",
            "PPO_LSTM",
            "PPO_HAT_LSTM",
            "PPO_HAT_PDI",
            "QRDQN_CVAR",
            "BE_CVAR_DQN",
            "PPO_PROTOMEM",
            "PPO_NEW",
            "NOVA_EDRL",
            "RARL",
            "PLR_UED",
            "PLR",
            "UED",
            "SABER_V0",
            "CQL_DQN",
            "CADM",
        }:
            return choice_upper
        if choice_upper in mapping.values():
            return choice_upper
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


def resolve_algorithm(algorithm):
    algo_label = (algorithm or "DQN").upper()
    if algo_label == "HAT":
        algo_label = "PPO_HAT"
    elif algo_label == "CQL":
        algo_label = "CQL_DQN"
    hat_enabled = algo_label in {"PPO_HAT", "A2C_HAT", "PPO_HAT_MOE", "A2C_HAT_MOE", "PPO_HAT_LSTM", "PPO_HAT_PDI"}
    if algo_label in {"PPO_HAT", "PPO_HAT_MOE"}:
        base_algo = "PPO"
    elif algo_label == "PPO_LSTM":
        base_algo = "PPO_LSTM"
    elif algo_label == "PPO_HAT_LSTM":
        base_algo = "PPO_LSTM"
    elif algo_label == "CADM":
        base_algo = "PPO_NEW"
    elif algo_label == "PPO_HAT_PDI":
        base_algo = "PPO_HAT_PDI"
    elif algo_label in {"A2C_HAT", "A2C_HAT_MOE"}:
        base_algo = "A2C"
    else:
        base_algo = algo_label
    return base_algo, hat_enabled, algo_label


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


def run_generator(dist_name, request_number, workers=None, target_folder=None, seed=None):
    """
    运行生成器生成分布
    """
    dist_label = get_distribution_display_map().get(dist_name, dist_name)
    print("")
    print(f">>> [阶段1] 正在生成随机分布事件 ({dist_label})")
    if target_folder is None:
        target_folder = LEGACY_FOLDER_NAME
    print(f"    目标文件夹: {target_folder}")

    generator_script = os.path.join(os.path.dirname(__file__), "generation", "generate_mixed_parallel.py")
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
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    cmd.append("--verify")

    try:
        return_code = stream_subprocess_output(cmd)
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)
        print(">>> 生成器运行成功，旧数据已覆盖。")
    except subprocess.CalledProcessError:
        print(">>> 错误：分布生成器运行失败。")
        sys.exit(1)


def run_simulation(request_number):
    print("")
    print(">>> [阶段2] 正在启动主仿真程序 (ALNS + RL)...")
    print("=" * 50)

    stop_flag = os.environ.get("STOP_FLAG_FILE", "34959.txt")
    if os.path.exists(stop_flag):
        os.remove(stop_flag)

    if add_RL == 0:
        Dynamic_ALNS_RL34959.main(0)
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            print(f"   线程并行编号: {parallel_number}")
            futures = {executor.submit(Dynamic_ALNS_RL34959.main, approach, request_number): approach for approach in parallel_number}

            for future in concurrent.futures.as_completed(futures):
                try:
                    data = future.result()
                except SystemExit as exc:
                    # Worker threads may call sys.exit() (e.g., stop flag / normal termination).
                    # Swallow it here so we can join other threads cleanly.
                    code = getattr(exc, "code", None)
                    print(f"线程任务请求退出: SystemExit({code})")
                except Exception as exc:
                    print(f"线程任务产生异常: {exc}")
                    print(traceback.format_exc())


def _normalize_stage_mode(value):
    mode = str(value or "train_eval").strip().lower()
    if mode not in {"train_eval", "train_only", "eval_only"}:
        mode = "train_eval"
    return mode


def run_nova_edrl_pipeline(
    dist_name,
    request_number,
    workers=None,
    seed=None,
    algo_version="v1",
    run_name=None,
    init_model_path=None,
    save_model_path=None,
    skip_generator=False,
    external_data_root=None,
):
    script_path = Path(__file__).resolve().parent / "outer_rl" / "run_edrl_pipeline.py"
    if not script_path.exists():
        raise FileNotFoundError(f"NOVA_EDRL pipeline script not found: {script_path}")

    seed_val = int(seed) if seed is not None else 42
    if run_name:
        run_id = str(run_name)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_id = f"run_{timestamp}_R{request_number}_{dist_name}_NOVA_EDRL_S{seed_val}"
    rl_log_root = str(os.environ.get("RL_LOG_ROOT", "") or "").strip()
    run_id_for_pipeline = run_id
    try:
        run_path = Path(run_id)
        if rl_log_root and (not run_path.is_absolute()):
            run_id_for_pipeline = str((Path(rl_log_root).resolve() / run_id).resolve())
    except Exception:
        run_id_for_pipeline = run_id

    nova_version = str(algo_version or "v1").strip().lower()
    outer_action_space_version = "v2" if nova_version in {"v2", "v3", "v4"} else "v1"
    if nova_version == "v4":
        outer_edrl_version = "v4"
    elif nova_version == "v3":
        outer_edrl_version = "v3"
    else:
        outer_edrl_version = "v1"

    cmd = [
        sys.executable,
        str(script_path),
        "--run-id",
        str(run_id_for_pipeline),
        "--python-bin",
        str(sys.executable),
        "--phase1-dist-name",
        str(dist_name),
        "--phase1-request-number",
        str(int(request_number)),
        "--phase1-algorithm",
        "PPO_NEW",
        "--phase1-algo-version",
        "v3",
        "--phase1-seed",
        str(seed_val),
        "--outer-phase",
        "auto",
        "--outer-dist-name",
        str(dist_name),
        "--outer-request-number",
        str(int(request_number)),
        "--outer-algorithm",
        "PPO_NEW",
        "--outer-algo-version",
        "v3",
        "--outer-seed",
        str(seed_val),
        "--outer-policy-mode",
        "ts",
        "--outer-warmup-iters",
        "1",
        "--outer-policy-decay",
        "1.0",
        "--outer-ts-prior-mean",
        "0.0",
        "--outer-ts-prior-std",
        "0.5",
        "--outer-ts-obs-std",
        "0.05",
        "--outer-edrl-version",
        str(outer_edrl_version),
        "--outer-mu-choices",
        "10,30,60,90",
        "--outer-ratio-choices",
        "0.2,0.3,0.5,0.7,0.8",
        "--outer-num-file-choices",
        "5,10,15",
        "--outer-pattern-choices",
        "ab,random_mix",
        "--outer-action-space-version",
        str(outer_action_space_version),
        "--outer-v2-fixed-ratio-a",
        "0.5",
        "--outer-v2-fixed-pattern",
        "ab",
        "--outer-v2-fixed-num-files",
        "0",
        "--outer-inner-stop-mode",
        "fixed_n",
        "--outer-inner-fixed-n",
        "0",
        "--phase2-fixed-num-files",
        "5",
        "--phase3-num-file-choices",
        "5,10,15",
        "--phase2-min-iters",
        "5",
        "--phase2-max-iters",
        "40",
        "--phase3-min-iters",
        "10",
        "--phase3-max-iters",
        "50",
        "--converge-patience",
        "2",
        "--phase3-converge-patience",
        "1",
        "--converge-max-abs-dj",
        "0.20",
        "--converge-max-obj-range",
        "0.50",
        "--phase2-converge-max-abs-dj",
        "0.80",
        "--phase2-converge-max-obj-range",
        "1.00",
        "--converge-minority-floor",
        "0.01",
        "--phase3-topk-k",
        "0",
        "--phase3-topk-warmup-iters",
        "0",
        "--phase3-topk-prior-count",
        "0.0",
        "--rho-target",
        "0.22",
        "--rho-floor",
        "0.10",
        "--eta-collapse",
        "1.4",
        "--rho-floor-weight",
        "4.0",
        "--rho-floor-hard-weight",
        "12.0",
        "--collapse-gap-power",
        "2.0",
        "--outer-curriculum-alpha-start",
        "0.7",
        "--outer-curriculum-alpha-end",
        "0.35",
        "--outer-curriculum-alpha-horizon",
        "25",
        "--outer-curriculum-replay-ratio",
        "0.2",
        "--outer-curriculum-replay-max-iters",
        "5",
        "--phase2-hard-weight",
        "1.00",
        "--phase2-minority-reward-weight",
        "1.20",
        "--phase2-too-hard-weight",
        "0.0",
        "--phase2-j-low",
        "0.20",
    ]
    if outer_edrl_version == "v3":
        cmd.extend(
            [
                "--outer-edrl-v3-dj-weight",
                "0.6",
                "--outer-edrl-v3-j-weight",
                "0.1",
                "--outer-edrl-v3-minority-abs-weight",
                "0.5",
                "--outer-edrl-v3-level-replay",
                "--outer-edrl-v3-replay-phase3-only",
            ]
        )
    if outer_edrl_version == "v4":
        cmd.extend(
            [
                "--outer-edrl-v4-challenge-weight",
                "0.40",
                "--outer-edrl-v4-lp-weight",
                "1.00",
                "--outer-edrl-v4-j-weight",
                "0.10",
                "--outer-edrl-v4-entropy-weight",
                "0.35",
                "--outer-edrl-v4-minority-weight",
                "1.20",
                "--outer-edrl-v4-minority-abs-weight",
                "0.80",
                "--outer-edrl-v4-novelty-weight",
                "0.20",
                "--outer-edrl-v4-j-center",
                "0.55",
                "--outer-edrl-v4-j-sigma",
                "0.20",
                "--outer-edrl-v4-p-new-k",
                "0.80",
                "--outer-edrl-v4-p-new-min",
                "0.20",
                "--outer-edrl-v4-p-new-max",
                "0.90",
                "--outer-edrl-v4-entropy-target",
                "0.25",
                "--outer-edrl-v4-level-replay",
                "--no-outer-edrl-v4-replay-phase3-only",
                "--outer-plr-p-new",
                "0.55",
                "--outer-plr-buffer-size",
                "300",
                "--outer-plr-priority-ema-alpha",
                "0.50",
                "--outer-plr-min-weight",
                "0.02",
            ]
        )
    if workers is not None and int(workers) > 0:
        cmd.extend(["--phase1-workers", str(int(workers))])
        cmd.extend(["--outer-workers", str(int(workers))])
    if init_model_path:
        cmd.extend(["--phase1-init-model-path", str(init_model_path)])
    if save_model_path:
        cmd.extend(["--phase1-save-model-path", str(save_model_path)])
    if bool(skip_generator):
        cmd.append("--phase1-skip-generator")
    if external_data_root:
        cmd.extend(["--phase1-external-data-root", str(external_data_root)])

    print("")
    print(f">>> [NOVA] orchestrating 4-phase pipeline for run_id={run_id_for_pipeline}")
    print(
        f">>> [NOVA] profile={nova_version} outer_action_space={outer_action_space_version} "
        f"outer_edrl={outer_edrl_version}"
    )
    print(">>> [NOVA] forcing phase flow: phase1(train_only) -> phase2/phase3(outer) -> phase4(eval_only implement)")
    subprocess.run(cmd, check=True)


def run_outer_reference_pipeline(
    profile_name,
    dist_name,
    request_number,
    workers=None,
    seed=None,
    run_name=None,
    init_model_path=None,
    save_model_path=None,
    skip_generator=False,
    external_data_root=None,
):
    script_path = Path(__file__).resolve().parent / "outer_rl" / "run_edrl_pipeline.py"
    if not script_path.exists():
        raise FileNotFoundError(f"outer reference pipeline script not found: {script_path}")

    seed_val = int(seed) if seed is not None else 42
    profile = str(profile_name or "").strip().upper()
    if run_name:
        run_id = str(run_name)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_id = f"run_{timestamp}_R{request_number}_{dist_name}_{profile}_S{seed_val}"
    rl_log_root = str(os.environ.get("RL_LOG_ROOT", "") or "").strip()
    run_id_for_pipeline = run_id
    try:
        run_path = Path(run_id)
        if rl_log_root and (not run_path.is_absolute()):
            run_id_for_pipeline = str((Path(rl_log_root).resolve() / run_id).resolve())
    except Exception:
        run_id_for_pipeline = run_id

    cmd = [
        sys.executable,
        str(script_path),
        "--run-id",
        str(run_id_for_pipeline),
        "--python-bin",
        str(sys.executable),
        "--phase1-dist-name",
        str(dist_name),
        "--phase1-request-number",
        str(int(request_number)),
        "--phase1-algorithm",
        "PPO_NEW",
        "--phase1-algo-version",
        "v3",
        "--phase1-seed",
        str(seed_val),
        "--outer-phase",
        "auto",
        "--outer-dist-name",
        str(dist_name),
        "--outer-request-number",
        str(int(request_number)),
        "--outer-algorithm",
        "PPO_NEW",
        "--outer-algo-version",
        "v3",
        "--outer-seed",
        str(seed_val),
        "--phase3-min-iters",
        "30",
        "--phase3-max-iters",
        "200",
    ]
    if workers is not None and int(workers) > 0:
        cmd.extend(["--phase1-workers", str(int(workers))])
        cmd.extend(["--outer-workers", str(int(workers))])
    if init_model_path:
        cmd.extend(["--phase1-init-model-path", str(init_model_path)])
    if save_model_path:
        cmd.extend(["--phase1-save-model-path", str(save_model_path)])
    if bool(skip_generator):
        cmd.append("--phase1-skip-generator")
    if external_data_root:
        cmd.extend(["--phase1-external-data-root", str(external_data_root)])

    if profile == "RARL":
        cmd.extend(
            [
                "--outer-policy-mode",
                "rarl_dqn",
                "--objective-mode",
                "rarl",
                "--rarl-k1",
                "1",
                "--rarl-k2",
                "4",
                "--rarl-replay-size",
                "4000",
                "--rarl-batch-size",
                "64",
                "--rarl-min-replay",
                "64",
                "--rarl-state-window",
                "5",
                "--rarl-zero-sum-strict",
                "1",
                "--rarl-force-objective",
                "1",
            ]
        )
    elif profile in {"PLR_UED", "PLR", "UED"}:
        cmd.extend(
            [
                "--outer-policy-mode",
                "ts",
                "--objective-mode",
                "plr",
                "--plr-level-replay",
                "--plr-p-new",
                "0.5",
                "--plr-buffer-size",
                "200",
                "--plr-priority-ema-alpha",
                "0.6",
                "--plr-min-weight",
                "0.05",
                "--outer-curriculum-enable",
            ]
        )
    elif profile in {"SABER_V0", "SABER", "SABER-PLR-V0"}:
        cmd.extend(
            [
                "--outer-policy-mode",
                "ts",
                "--objective-mode",
                "saber_v0",
                "--plr-level-replay",
                "--plr-p-new",
                "0.55",
                "--plr-buffer-size",
                "200",
                "--plr-priority-ema-alpha",
                "0.50",
                "--plr-min-weight",
                "0.02",
                "--saber-v0-dj-weight",
                "0.45",
                "--saber-v0-j-weight",
                "0.25",
                "--saber-v0-novelty-weight",
                "0.15",
                "--saber-v0-j-center",
                "0.55",
                "--saber-v0-j-sigma",
                "0.18",
                "--outer-curriculum-enable",
            ]
        )
    else:
        raise ValueError(f"unsupported outer reference profile: {profile}")

    print("")
    print(f">>> [{profile}] orchestrating reference 4-phase pipeline for run_id={run_id_for_pipeline}")
    subprocess.run(cmd, check=True)


def run_single(
    dist_name,
    request_number,
    workers=None,
    algorithm="DQN",
    seed=None,
    run_name=None,
    algo_version="v1",
    stage_mode="train_eval",
    init_model_path=None,
    save_model_path=None,
    skip_generator=False,
    external_data_root=None,
):
    algorithm_label = str(algorithm or "DQN").strip().upper()
    if algorithm_label == "NOVA_EDRL":
        run_nova_edrl_pipeline(
            dist_name=dist_name,
            request_number=request_number,
            workers=workers,
            seed=seed,
            algo_version=algo_version,
            run_name=run_name,
            init_model_path=init_model_path,
            save_model_path=save_model_path,
            skip_generator=skip_generator,
            external_data_root=external_data_root,
        )
        return
    if algorithm_label == "RARL":
        run_outer_reference_pipeline(
            profile_name="RARL",
            dist_name=dist_name,
            request_number=request_number,
            workers=workers,
            seed=seed,
            run_name=run_name,
            init_model_path=init_model_path,
            save_model_path=save_model_path,
            skip_generator=skip_generator,
            external_data_root=external_data_root,
        )
        return
    if algorithm_label in {"PLR_UED", "PLR", "UED"}:
        run_outer_reference_pipeline(
            profile_name="PLR_UED",
            dist_name=dist_name,
            request_number=request_number,
            workers=workers,
            seed=seed,
            run_name=run_name,
            init_model_path=init_model_path,
            save_model_path=save_model_path,
            skip_generator=skip_generator,
            external_data_root=external_data_root,
        )
        return
    if algorithm_label in {"SABER_V0", "SABER", "SABER-PLR-V0"}:
        run_outer_reference_pipeline(
            profile_name="SABER_V0",
            dist_name=dist_name,
            request_number=request_number,
            workers=workers,
            seed=seed,
            run_name=run_name,
            init_model_path=init_model_path,
            save_model_path=save_model_path,
            skip_generator=skip_generator,
            external_data_root=external_data_root,
        )
        return

    base_algo, hat_enabled, algo_label = resolve_algorithm(algorithm)
    algo_version = str(algo_version or "v1").strip().lower()
    if algo_label == "CADM" and algo_version in {"", "v1"}:
        # CADM baseline is implemented on PPO_NEW as v6.3_cadm.
        algo_version = "v6.3_cadm"
    stage_mode = _normalize_stage_mode(stage_mode)
    os.environ["SCENARIO_NAME"] = dist_name
    os.environ["RL_ALGORITHM"] = base_algo
    os.environ["RL_ALGO_VERSION"] = algo_version
    os.environ["RL_STAGE_MODE"] = stage_mode
    os.environ["RL_HAT"] = "1" if hat_enabled else "0"
    os.environ["RL_MOE"] = "1" if algo_label in {"PPO_HAT_MOE", "A2C_HAT_MOE"} else "0"
    if algo_label == "CADM":
        os.environ["RL_CADM_PROFILE"] = "1"
        os.environ["RL_STAGE_IN_OBS"] = "1"
        os.environ["RL_USE_AUGMENTED_OBS"] = "1"
        os.environ["LSTM_HIDDEN_SIZE"] = os.environ.get("LSTM_HIDDEN_SIZE", "64")
        os.environ["LSTM_LAYERS"] = os.environ.get("LSTM_LAYERS", "1")
    else:
        os.environ.pop("RL_CADM_PROFILE", None)
    if algo_label in {"PPO_LSTM", "PPO_HAT_LSTM", "CADM"}:
        os.environ["RL_STAGE_IN_OBS"] = os.environ.get("RL_STAGE_IN_OBS", "1")
    else:
        os.environ["RL_STAGE_IN_OBS"] = os.environ.get("RL_STAGE_IN_OBS", "0")
    if seed is None:
        os.environ.pop("RL_SEED", None)
    else:
        os.environ["RL_SEED"] = str(seed)
    if init_model_path:
        os.environ["RL_INIT_MODEL_PATH"] = str(init_model_path)
    else:
        os.environ.pop("RL_INIT_MODEL_PATH", None)
    if save_model_path:
        os.environ["RL_SAVE_MODEL_PATH"] = str(save_model_path)
    else:
        os.environ.pop("RL_SAVE_MODEL_PATH", None)
    Dynamic_ALNS_RL34959.SCENARIO_NAME = dist_name
    Dynamic_ALNS_RL34959.RL_ALGORITHM = base_algo
    dynamic_RL34959.SCENARIO_NAME = dist_name
    if run_name:
        run_id = run_name
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        seed_tag = f"S{seed}" if seed is not None else "SNA"
        run_id = f"run_{timestamp}_R{request_number}_{dist_name}_{algo_label}_{seed_tag}"
    rl_logging.set_run_dir(run_id)
    stop_flag_path = str(rl_logging.get_run_dir() / "34959.txt")
    os.environ["STOP_FLAG_FILE"] = stop_flag_path
    run_data_dir = rl_logging.get_run_data_dir()
    dynamic_data_root = Path(external_data_root).resolve() if external_data_root else Path(run_data_dir).resolve()
    if skip_generator and (not dynamic_data_root.exists()):
        raise FileNotFoundError(f"skip_generator=1 but external data root does not exist: {dynamic_data_root}")
    os.environ["DYNAMIC_DATA_ROOT"] = str(dynamic_data_root)
    os.environ["ALNS_OUTPUT_ROOT"] = str(rl_logging.get_run_dir())
    Intermodal_ALNS34959.refresh_figures_dir()
    curriculum_threshold = dynamic_RL34959.CURRICULUM_REWARD_THRESHOLD
    if dist_name == "S0_Debug":
        curriculum_threshold = 0.3
    rl_logging.write_meta({
        "distribution": dist_name,
        "request_number": request_number,
        "generator_workers": workers if workers is not None else "auto",
        "algorithm": algo_label,
        "algo_version": algo_version,
        "hat_enabled": int(hat_enabled),
        "hat_base": base_algo if hat_enabled else "",
        "seed": seed,
        "run_name": run_id,
        "stage_mode": stage_mode,
        "init_model_path": str(init_model_path or ""),
        "save_model_path": str(save_model_path or ""),
        "curriculum_reward_threshold": curriculum_threshold,
        "curriculum_success_required": getattr(dynamic_RL34959, "CURRICULUM_SUCCESS_REQUIRED", None),
        "stop_flag_file": stop_flag_path,
        "data_root": str(dynamic_data_root),
        "run_data_dir": str(run_data_dir),
        "alns_output_root": str(rl_logging.get_run_dir()),
        "skip_generator": int(bool(skip_generator)),
        "external_data_root": str(external_data_root or ""),
    })
    log_file = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        log_path = os.path.join(str(rl_logging.get_run_dir()), "console_output.txt")
        log_file = open(log_path, "a", encoding="utf-8")
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        if skip_generator:
            print(f">>> [stage1] skip generator, using external data root: {dynamic_data_root}")
        else:
            run_generator(dist_name, request_number, workers, str(dynamic_data_root), seed)
        run_simulation(request_number)
    finally:
        if log_file is not None:
            log_file.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def run_single_in_subprocess(
    dist_name,
    request_number,
    workers=None,
    algorithm="DQN",
    seed=None,
    run_name=None,
    algo_version="v1",
    stage_mode="train_eval",
    init_model_path=None,
    save_model_path=None,
    skip_generator=False,
    external_data_root=None,
):
    script_path = os.path.abspath(__file__)
    cmd = [
        sys.executable, script_path,
        "--dist_name", dist_name,
        "--request_number", str(request_number),
        "--algorithm", algorithm,
        "--algo_version", str(algo_version),
        "--stage-mode", str(_normalize_stage_mode(stage_mode)),
    ]
    if workers is not None:
        cmd.extend(["--workers", str(workers)])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if run_name:
        cmd.extend(["--run-name", run_name])
    if init_model_path:
        cmd.extend(["--init-model-path", str(init_model_path)])
    if save_model_path:
        cmd.extend(["--save-model-path", str(save_model_path)])
    if skip_generator:
        cmd.append("--skip-generator")
    if external_data_root:
        cmd.extend(["--external-data-root", str(external_data_root)])
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dist_name", type=str)
    parser.add_argument("--request_number", type=int)
    parser.add_argument("--run_count", type=int)
    parser.add_argument("--algorithm", type=str, help="DQN/PPO/PPO_NEW/A2C/DRCB/LBKLAC/PPO_HAT/A2C_HAT/PPO_LSTM/PPO_HAT_LSTM/PPO_HAT_PDI/PPO_HAT_MOE/A2C_HAT_MOE/QRDQN_CVAR/BE_CVAR_DQN/PPO_PROTOMEM/NOVA_EDRL/RARL/PLR_UED/SABER_V0/CQL_DQN/CADM/HAT")
    parser.add_argument("--algo_version", type=str, default="v1", help="algorithm version tag for extensible entries such as PPO_NEW")
    parser.add_argument("--workers", type=int, help="generator workers (1=single core)")
    parser.add_argument("--single_core", action="store_true", help="force generator single core")
    parser.add_argument("--parallel-runs", type=int, help="parallel run count when run_count > 1")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--run-name", type=str, help="override run folder name")
    parser.add_argument("--stage-mode", type=str, default="train_eval", help="train_eval/train_only/eval_only")
    parser.add_argument("--init-model-path", type=str, default="", help="optional checkpoint to load before run")
    parser.add_argument("--save-model-path", type=str, default="", help="optional checkpoint path to save after run")
    parser.add_argument("--skip-generator", action="store_true", help="skip stage1 generator and use external data root")
    parser.add_argument("--external-data-root", type=str, default="", help="override DYNAMIC_DATA_ROOT for training/eval")
    return parser.parse_args()


def main():
    args = parse_args()
    workers = resolve_worker_count(args)
    algorithm = args.algorithm.upper() if args.algorithm else None
    algo_version = str(args.algo_version or "v1").strip().lower()
    stage_mode = _normalize_stage_mode(args.stage_mode)
    init_model_path = str(args.init_model_path or "").strip() or None
    save_model_path = str(args.save_model_path or "").strip() or None
    skip_generator = bool(args.skip_generator)
    external_data_root = str(args.external_data_root or "").strip() or None
    seed = args.seed
    run_name = args.run_name
    parallel_runs = int(args.parallel_runs) if args.parallel_runs else 1
    if parallel_runs < 1:
        parallel_runs = 1
    if algorithm is not None and algorithm not in {"DQN", "PPO", "PPO_NEW", "A2C", "DRCB", "LBKLAC", "PPO_HAT", "A2C_HAT", "PPO_HAT_MOE", "A2C_HAT_MOE", "PPO_LSTM", "PPO_HAT_LSTM", "PPO_HAT_PDI", "QRDQN_CVAR", "BE_CVAR_DQN", "PPO_PROTOMEM", "NOVA_EDRL", "RARL", "PLR_UED", "PLR", "UED", "SABER_V0", "SABER", "SABER-PLR-V0", "CQL_DQN", "CQL", "CADM", "HAT"}:
        print(f"未知算法 {algorithm}，回退为 DQN")
        algorithm = "DQN"

    if args.dist_name and args.request_number:
        run_count = args.run_count or 1
        if algorithm is None:
            algorithm = select_algorithm()
        if run_count <= 1:
            run_single(
                args.dist_name,
                args.request_number,
                workers,
                algorithm,
                seed,
                run_name,
                algo_version,
                stage_mode=stage_mode,
                init_model_path=init_model_path,
                save_model_path=save_model_path,
                skip_generator=skip_generator,
                external_data_root=external_data_root,
            )
        else:
            max_workers = min(run_count, parallel_runs)
            if max_workers <= 1:
                for idx in range(run_count):
                    child_run_name = f"{run_name}_{idx + 1}" if run_name else None
                    run_single_in_subprocess(
                        args.dist_name,
                        args.request_number,
                        workers,
                        algorithm,
                        seed,
                        child_run_name,
                        algo_version,
                        stage_mode=stage_mode,
                        init_model_path=init_model_path,
                        save_model_path=save_model_path,
                        skip_generator=skip_generator,
                        external_data_root=external_data_root,
                    )
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    for idx in range(run_count):
                        child_run_name = f"{run_name}_{idx + 1}" if run_name else None
                        futures.append(executor.submit(
                            run_single_in_subprocess,
                            args.dist_name,
                            args.request_number,
                            workers,
                            algorithm,
                            seed,
                            child_run_name,
                            algo_version,
                            stage_mode,
                            init_model_path,
                            save_model_path,
                            skip_generator,
                            external_data_root,
                        ))
                    for future in concurrent.futures.as_completed(futures):
                        future.result()
        return

    if workers is None:
        workers = select_worker_count()

    if algorithm is None:
        algorithm = select_algorithm()

    run_count = args.run_count if args.run_count is not None else select_run_count()
    if run_count <= 1:
        dist_name = select_distribution_mode()
        request_number = select_request_number()
        run_single(
            dist_name,
            request_number,
            workers,
            algorithm,
            seed,
            run_name,
            algo_version,
            stage_mode=stage_mode,
            init_model_path=init_model_path,
            save_model_path=save_model_path,
            skip_generator=skip_generator,
            external_data_root=external_data_root,
        )
        return

    plan = collect_batch_plan(run_count, algorithm)
    max_workers = min(run_count, parallel_runs)
    if max_workers <= 1:
        for idx, (dist_name, request_number, algorithm) in enumerate(plan, start=1):
            dist_label = get_distribution_display_map().get(dist_name, dist_name)
            print("")
            print("=" * 50)
            print(f">>> [batch] running {idx}/{run_count} dist[{dist_label}] | R={request_number}")
            print("=" * 50)
            run_single_in_subprocess(
                dist_name,
                request_number,
                workers,
                algorithm,
                seed,
                algo_version=algo_version,
                stage_mode=stage_mode,
                init_model_path=init_model_path,
                save_model_path=save_model_path,
                skip_generator=skip_generator,
                external_data_root=external_data_root,
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, (dist_name, request_number, algorithm) in enumerate(plan, start=1):
                dist_label = get_distribution_display_map().get(dist_name, dist_name)
                print("")
                print("=" * 50)
                print(f">>> [batch] scheduling {idx}/{run_count} dist[{dist_label}] | R={request_number}")
                print("=" * 50)
                futures.append(executor.submit(
                    run_single_in_subprocess,
                    dist_name,
                    request_number,
                    workers,
                    algorithm,
                    seed,
                    None,
                    algo_version,
                    stage_mode,
                    init_model_path,
                    save_model_path,
                    skip_generator,
                    external_data_root,
                ))
            for future in concurrent.futures.as_completed(futures):
                future.result()
if __name__ == '__main__':
    main()

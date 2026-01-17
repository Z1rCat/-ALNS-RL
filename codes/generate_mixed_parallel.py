import os
import time
import argparse
import json
import math
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import sys

# 尝试导入 tqdm 用于显示进度条，如果没有则使用简易打印
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# 忽略 FutureWarning
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

# ================= CONFIGURATION =================
# 自动获取项目根目录 (假设脚本在 codes/ 下)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(ROOT_DIR, "Intermodal_EGS_data_all.xlsx")
OUTPUT_ROOT = os.path.join(ROOT_DIR, "Uncertainties Dynamic planning under unexpected events")
FIGURES_DIR = os.path.join(ROOT_DIR, "Figures")

# 实验映射 (用于查找 Best Routes 以确定事件发生时间)
EXP_NUMBERS = {5: 12793, 10: 12792, 20: 12794, 30: 12816, 50: 12817, 100: 12818}

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

DIST_DISPLAY = {}

# 子进程全局缓存
GLOBAL_DATA = {}

EXPECTED_TOTAL_FILES = 500

SCENARIO_CONFIGS = {}

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

def build_display(name, pattern, means):
    parts = []
    for key in ("A", "B", "C"):
        if key in means:
            value = means[key]
            if isinstance(value, dict):
                mean_val = value.get("mean")
                if mean_val is None:
                    mean_val = value.get("mu")
                extra = ""
                if "var" in value:
                    extra = f",var={value['var']}"
                elif "std" in value:
                    extra = f",std={value['std']}"
                parts.append(f"{key}={mean_val}{extra}")
            else:
                parts.append(f"{key}={value}")
    suffix = " ".join(parts)
    if suffix:
        return f"{name} {pattern} {suffix}"
    return f"{name} {pattern}"

def build_scenario_configs():
    SCENARIO_CONFIGS.clear()
    DIST_DISPLAY.clear()
    for item in load_distribution_config():
        name = item["name"]
        pattern = item["pattern"]
        means = item["means"]
        display_str = item["display"] or build_display(name, pattern, means)
        SCENARIO_CONFIGS[name] = {"pattern": pattern, "means": means}
        DIST_DISPLAY[name] = display_str

build_scenario_configs()

def sample_durations(mean_val, max_events, std=None, dist="normal"):
    if dist == "normal":
        sigma = std if std is not None else max(1.0, mean_val * 0.25)
        samples = np.random.normal(mean_val, sigma, size=max_events)
    elif dist == "lognormal":
        if std is None:
            sigma = 0.5
            mu = math.log(max(mean_val, 1.0)) - 0.5 * sigma * sigma
        else:
            variance = std * std
            mean_val = max(mean_val, 1.0)
            mu = math.log((mean_val * mean_val) / math.sqrt(variance + mean_val * mean_val))
            sigma = math.sqrt(max(1e-6, math.log(1 + variance / (mean_val * mean_val))))
        samples = np.random.lognormal(mean=mu, sigma=sigma, size=max_events)
    else:
        raise ValueError(f"Unsupported dist '{dist}'")
    samples = np.maximum(samples, 1)
    return samples.astype(int)

def build_phase_labels(pattern, total_files):
    if pattern == "random_mix":
        return list(np.random.choice(["A", "B"], size=total_files, p=[0.5, 0.5]))
    segments = {
        "aba": [(0, 174, "A"), (175, 349, "B"), (350, 499, "A")],
        "ab": [(0, 349, "A"), (350, 499, "B")],
        "recall": [(0, 424, "A"), (425, 499, "B")],
        "adaptation": [(0, 99, "A"), (100, 499, "B")],
        "abc": [(0, 174, "A"), (175, 349, "B"), (350, 499, "C")],
    }
    labels = [""] * total_files
    for start, end, label in segments[pattern]:
        end = min(end, total_files - 1)
        for idx in range(start, end + 1):
            if 0 <= idx < total_files:
                labels[idx] = label
    if any(lbl == "" for lbl in labels):
        last = "A"
        for i in range(total_files):
            if labels[i] == "":
                labels[i] = last
            else:
                last = labels[i]
    return labels

def parse_phase_spec(spec):
    if isinstance(spec, dict):
        mean_val = spec.get("mean")
        if mean_val is None:
            mean_val = spec.get("mu")
        if mean_val is None:
            raise ValueError("Phase spec must include mean")
        std = spec.get("std")
        var = spec.get("var")
        if std is None and var is not None:
            std = math.sqrt(var)
        dist = spec.get("dist", "normal")
        return {"mean": float(mean_val), "std": float(std) if std is not None else None, "dist": dist}
    return {"mean": float(spec), "std": None, "dist": "normal"}

def build_scenario_matrix_and_meta(dist_name, total_files, max_events):
    config = SCENARIO_CONFIGS[dist_name]
    labels = build_phase_labels(config["pattern"], total_files)
    matrix = np.zeros((total_files, max_events), dtype=int)
    meta_rows = []
    for i, label in enumerate(labels):
        phase_spec = config["means"][label]
        phase_params = parse_phase_spec(phase_spec)
        mean_val = phase_params["mean"]
        matrix[i] = sample_durations(mean_val, max_events, std=phase_params["std"], dist=phase_params["dist"])
        meta_rows.append({"gt_mean": mean_val, "phase_label": label})
    return matrix, meta_rows

def build_default_meta(matrix, phase_label):
    meta_rows = []
    for row in matrix:
        meta_rows.append({"gt_mean": float(np.mean(row)), "phase_label": phase_label})
    return meta_rows

def get_distribution_matrix(dist_name, total_files, max_events):
    """
    【兵工厂核心】根据策略生成随机数矩阵 [Files, Events]
    """
    dist_label = DIST_DISPLAY.get(dist_name, dist_name)
    print(f"   -> distribution: {dist_label}")

    if dist_name in SCENARIO_CONFIGS:
        return build_scenario_matrix_and_meta(dist_name, total_files, max_events)
    available = ", ".join(sorted(SCENARIO_CONFIGS.keys()))
    raise ValueError(f"Unknown dist_name '{dist_name}'. Available: {available}")

def init_worker(base_data_path, exp_mapping, figures_dir):
    """子进程初始化：加载一次大文件"""
    try:
        # 1. Load Base Data
        xls = pd.ExcelFile(base_data_path)
        GLOBAL_DATA['N'] = pd.read_excel(xls, 'N')
        GLOBAL_DATA['T'] = pd.read_excel(xls, 'T')
        GLOBAL_DATA['K'] = pd.read_excel(xls, 'K')
        GLOBAL_DATA['o'] = pd.read_excel(xls, 'o')
        GLOBAL_DATA['R_sheets'] = {}
        for r in exp_mapping.keys():
            GLOBAL_DATA['R_sheets'][r] = pd.read_excel(xls, f'R_{r}')
            
        # 2. Load Best Routes Triggers
        GLOBAL_DATA['triggers'] = {}
        for r, exp_num in exp_mapping.items():
            routes_path = os.path.join(figures_dir, f"experiment{exp_num}", "percentage0parallel_number9dynamic0", f"best_routespercentage0parallel_number9dynamic0_{exp_num}.xlsx")
            triggers = []
            if os.path.exists(routes_path):
                try:
                    route_xls = pd.ExcelFile(routes_path)
                    # 简易解析逻辑：遍历 Sheet，提取 Location 和 Time
                    # 这里为了健壮性，使用 try-except 包裹读取逻辑
                    sheet_map = pd.read_excel(route_xls, None)
                    for k, df in sheet_map.items():
                        if len(df.columns) > 2:
                            mode = 1 if 'Barge' in k else (2 if 'Train' in k else 3)
                            # 假设格式：Col[i] -> Row 0: Loc, Row 1: Time
                            # 跳过第一列和最后一列
                            cols = df.columns[1:-1]
                            for col in cols:
                                try:
                                    loc = df[col].iloc[0]
                                    t_val = df[col].iloc[1]
                                    triggers.append([loc, t_val, mode])
                                except: pass
                except: pass
            
            if triggers:
                arr = np.array(triggers)
                GLOBAL_DATA['triggers'][r] = arr[arr[:, 1].argsort()] # 按时间排序
            else:
                GLOBAL_DATA['triggers'][r] = None
                
    except Exception as e:
        print(f"?????????{e}")

def generate_single_file(args):
    """写入单个 Excel"""
    idx, r, duration_row, out_dir, meta_row = args
    
    fname = f"Intermodal_EGS_data_dynamic_congestion{idx}.xlsx"
    fpath = os.path.join(out_dir, fname)
    
    # 获取缓存
    N, T, K, o = GLOBAL_DATA['N'], GLOBAL_DATA['T'], GLOBAL_DATA['K'], GLOBAL_DATA['o']
    R_df = GLOBAL_DATA['R_sheets'][r]
    triggers = GLOBAL_DATA['triggers'][r]
    
    try:
        with pd.ExcelWriter(fpath) as writer:
            # 基础 Sheets
            N.to_excel(writer, 'N', index=False)
            R_df.to_excel(writer, f'R_{r}', index=False)
            T.to_excel(writer, 'T', index=False)
            K.to_excel(writer, 'K', index=False)
            o.to_excel(writer, 'o', index=False)
            current_mean = meta_row.get("gt_mean", "")
            current_label = meta_row.get("phase_label", "")
            meta_df = pd.DataFrame(
                {
                    "Property": ["gt_mean", "phase_label"],
                    "Value": [current_mean, current_label],
                }
            )
            meta_df.to_excel(writer, sheet_name="__meta__", index=False)
            
            # 动态事件 Sheets
            limit = min(len(duration_row), 50) # 限制每个文件最多50个事件
            used_pairs = []
            last_end = 0
            u_idx = 0
            
            # 确定事件源
            loop_range = range(limit)
            
            for i in loop_range:
                # 1. 确定 Location, StartTime, Mode
                if triggers is not None and i < len(triggers):
                    loc, start_t, mode = int(triggers[i][0]), int(triggers[i][1]), int(triggers[i][2])
                else:
                    # Fallback Random
                    loc = np.random.randint(0, 10)
                    start_t = last_end + np.random.randint(1, 5)
                    mode = np.random.choice([1,2,3])
                
                if start_t < last_end: continue
                if [loc, mode] in used_pairs: continue
                
                # 2. 获取 Duration (向量化数据)
                dur = int(duration_row[i])
                end_t = start_t + dur
                last_end = end_t
                
                # 3. 格式化为 List String (关键!)
                dur_str = str([start_t, end_t])
                used_pairs.append([loc, mode])
                
                # 4. DataFrame
                df = pd.DataFrame({
                    'uncertainty_index': [u_idx, u_idx],
                    'type': ['congestion', 'congestion_finish'],
                    'location_type': ['node', 'node'],
                    'vehicle': [-1, -1],
                    'location': [loc, loc],
                    'duration': [dur_str, dur_str],
                    'mode': [mode, mode]
                })
                
                sheet_name = f"R_{r}_{start_t} (2)"
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                u_idx += 1
                
        return True
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist_name", required=True)
    parser.add_argument("--target_folder", required=True)
    parser.add_argument("--total_files", type=int, default=EXPECTED_TOTAL_FILES)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--request_numbers", type=str, default="5",
                        help="指定要生成的R数量，单个值如'5'或逗号分隔如'5,10'")
    args = parser.parse_args()

    try:
        target_rs = [int(x) for x in args.request_numbers.split(",") if x.strip()]
    except ValueError:
        print(f"?? 无效的 --request_numbers 参数: {args.request_numbers}")
        sys.exit(1)
    target_rs = [r for r in target_rs if r in EXP_NUMBERS]
    if not target_rs:
        print("?? --request_numbers 为空或不在预设列表 {5,10,20,30,50,100} 中")
        sys.exit(1)

    if args.total_files != EXPECTED_TOTAL_FILES:
        print(f"?? total_files overridden to {EXPECTED_TOTAL_FILES} for physical isolation")
        args.total_files = EXPECTED_TOTAL_FILES

    dist_label = DIST_DISPLAY.get(args.dist_name, "????")
    print(f"=== ?????: {dist_label} ===")
    print(f"   ????: .../{os.path.basename(args.target_folder)}")
    print(f"   ?? R ???: {args.total_files}")
    print(f"   ????: {args.workers}")
    print(f"   R 集合: {target_rs}")

    start_all = time.time()
    
    # 1. 生成数据矩阵
    MAX_EVT = 60
    full_matrix, meta_rows = get_distribution_matrix(args.dist_name, args.total_files, MAX_EVT)
    
    # 2. 准备输出路径
    base_out = os.path.join(OUTPUT_ROOT, args.target_folder)
    if not os.path.exists(base_out):
        os.makedirs(base_out, exist_ok=True)
        
    # 3. 并行处理
    with ProcessPoolExecutor(max_workers=args.workers, 
                             initializer=init_worker, 
                             initargs=(DATA_FILE, EXP_NUMBERS, FIGURES_DIR)) as executor:
        
        for r in target_rs:
            print(f"\n>> Generating R_{r}...")
            r_dir = os.path.join(base_out, f"R{r}")
            os.makedirs(r_dir, exist_ok=True)
            
            # 准备任务
            tasks = []
            for i in range(args.total_files):
                tasks.append((i, r, full_matrix[i], r_dir, meta_rows[i]))
            
            # 提交并监控进度
            futures = [executor.submit(generate_single_file, t) for t in tasks]
            
            if HAS_TQDM:
                for _ in tqdm(as_completed(futures), total=len(futures), unit="file", ncols=80):
                    pass
            else:
                # 简易进度条
                done = 0
                for _ in as_completed(futures):
                    done += 1
                    if done % 100 == 0:
                        sys.stdout.write(f"\r   ??: {done}/{len(futures)}")
                        sys.stdout.flush()
                print("")
                
    print(f"\n=== ??????? {time.time() - start_all:.2f} ? ===")

if __name__ == "__main__":
    main()

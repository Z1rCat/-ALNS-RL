import os
import time
import argparse
import json
import math
import pandas as pd
import numpy as np
import zipfile
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

_ANSI_ENABLED = bool(sys.stdout and sys.stdout.isatty())


def _enable_ansi_on_windows():
    global _ANSI_ENABLED
    if os.name != "nt" or not _ANSI_ENABLED:
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        _ANSI_ENABLED = False


def _c(text, color="", bold=False):
    if not _ANSI_ENABLED:
        return text
    palette = {
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
        "white": "37",
    }
    codes = []
    if bold:
        codes.append("1")
    if color and color in palette:
        codes.append(palette[color])
    if not codes:
        return text
    return f"\033[{';'.join(codes)}m{text}\033[0m"


_enable_ansi_on_windows()

# ================= CONFIGURATION =================
# 自动获取项目根目录 (假设脚本在 codes/ 下)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FILE = os.path.join(ROOT_DIR, "Intermodal_EGS_data_all.xlsx")
OUTPUT_ROOT = os.path.join(ROOT_DIR, "Uncertainties Dynamic planning under unexpected events")
FIGURES_DIR = os.path.join(ROOT_DIR, "Figures")

# 实验映射 (用于查找 Best Routes 以确定事件发生时间)
EXP_NUMBERS = {5: 12793, 10: 12792, 20: 12794, 30: 12816, 50: 12817, 100: 12818}

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

DIST_DISPLAY = {}

# 子进程全局缓存
GLOBAL_DATA = {}

EXPECTED_TOTAL_FILES = 1000

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
        variance = item.get("variance")
        if not name or not pattern or not isinstance(means, dict):
            continue
        display = str(item.get("display", "")).strip()
        normalized.append({
            "name": name,
            "pattern": pattern,
            "means": means,
            "variance": variance,
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
        variance = item.get("variance")
        display_str = item["display"] or build_display(name, pattern, means)
        SCENARIO_CONFIGS[name] = {"pattern": pattern, "means": means, "variance": variance}
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
    if pattern in {"single_mean", "single", "constant", "aa"}:
        return ["A"] * total_files
    if pattern == "random_mix":
        return list(np.random.choice(["A", "B"], size=total_files, p=[0.5, 0.5]))
    train_files = int(total_files * 0.8)
    test_start = train_files
    half_train = train_files // 2
    half_test = (total_files - train_files) // 2
    segments = {
        # OOD: train A, test B
        "ab": [(0, train_files - 1, "A"), (test_start, total_files - 1, "B")],
        # Forgetting type-1: train A,B; test A
        "aba": [(0, half_train - 1, "A"), (half_train, train_files - 1, "B"), (test_start, total_files - 1, "A")],
        # Forgetting type-2: train A,B; test A,B (reverse read should be A then B)
        "abba": [
            (0, half_train - 1, "A"),
            (half_train, train_files - 1, "B"),
            (test_start, test_start + half_test - 1, "B"),
            (test_start + half_test, total_files - 1, "A"),
        ],
        # Generalization: train A,B; test C
        "abc": [(0, half_train - 1, "A"), (half_train, train_files - 1, "B"), (test_start, total_files - 1, "C")],
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

def resolve_phase_variance(variance_spec, phase_label):
    if variance_spec is None:
        return None
    if isinstance(variance_spec, dict):
        value = variance_spec.get(phase_label)
    else:
        value = variance_spec
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def build_scenario_matrix_and_meta(dist_name, total_files, max_events):
    config = SCENARIO_CONFIGS[dist_name]
    labels = build_phase_labels(config["pattern"], total_files)
    matrix = np.zeros((total_files, max_events), dtype=int)
    meta_rows = []
    variance_spec = config.get("variance")
    means_map = dict(config.get("means", {}) or {})
    fallback_label = "A" if "A" in means_map else (next(iter(means_map.keys())) if means_map else "")
    if not fallback_label:
        raise ValueError(f"distribution '{dist_name}' has empty means map")
    for i, label in enumerate(labels):
        phase_spec = means_map.get(label, means_map[fallback_label])
        phase_params = parse_phase_spec(phase_spec)
        mean_val = phase_params["mean"]
        std_val = phase_params["std"]
        if std_val is None:
            var_val = resolve_phase_variance(variance_spec, label)
            if var_val is not None:
                std_val = math.sqrt(var_val)
        matrix[i] = sample_durations(mean_val, max_events, std=std_val, dist=phase_params["dist"])
        meta_rows.append({"gt_mean": mean_val, "phase_label": label})
    return matrix, meta_rows

def verify_excel_file(path):
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            return bad is None
    except (zipfile.BadZipFile, EOFError, OSError):
        return False

def verify_output_dir(r_dir, total_files):
    failures = 0
    for idx in range(total_files):
        fpath = os.path.join(r_dir, f"Intermodal_EGS_data_dynamic_congestion{idx}.xlsx")
        if not os.path.exists(fpath) or not verify_excel_file(fpath):
            failures += 1
            print(f"{_c('[VERIFY]', 'yellow', True)} corrupted or missing file: {fpath}")
    return failures

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
    print(f"{_c('[DIST]', 'cyan', True)} {dist_label}")

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
        print(f"{_c('[WORKER-INIT]', 'red', True)} failed: {e}")

def generate_single_file(args):
    """写入单个 Excel"""
    idx, r, duration_row, out_dir, meta_row, seed = args
    
    fname = f"Intermodal_EGS_data_dynamic_congestion{idx}.xlsx"
    fpath = os.path.join(out_dir, fname)
    tmp_path = f"{fpath}.tmp.xlsx"
    
    # 获取缓存
    N, T, K, o = GLOBAL_DATA['N'], GLOBAL_DATA['T'], GLOBAL_DATA['K'], GLOBAL_DATA['o']
    R_df = GLOBAL_DATA['R_sheets'][r]
    triggers = GLOBAL_DATA['triggers'][r]
    
    try:
        with pd.ExcelWriter(tmp_path) as writer:
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
            rng = np.random.RandomState(seed + idx) if seed is not None else np.random
            
            for i in loop_range:
                # 1. 确定 Location, StartTime, Mode
                if triggers is not None and i < len(triggers):
                    loc, start_t, mode = int(triggers[i][0]), int(triggers[i][1]), int(triggers[i][2])
                else:
                    # Fallback Random
                    loc = rng.randint(0, 10)
                    start_t = last_end + rng.randint(1, 5)
                    mode = rng.choice([1, 2, 3])
                
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

        os.replace(tmp_path, fpath)
        return True
    except Exception as exc:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        print(f"{_c('[WRITE]', 'red', True)} failed: {fpath} | {exc}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist_name", required=True)
    parser.add_argument("--target_folder", required=True)
    parser.add_argument("--total_files", type=int, default=EXPECTED_TOTAL_FILES)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument(
        "--request_numbers",
        type=str,
        default="5",
        help="Target request numbers, e.g. '5' or '5,10'",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed for generator")
    parser.add_argument("--verify", action="store_true", help="verify generated excel files")
    args = parser.parse_args()

    try:
        target_rs = [int(x) for x in args.request_numbers.split(",") if x.strip()]
    except ValueError:
        print(f"{_c('[ARG]', 'red', True)} invalid --request_numbers: {args.request_numbers}")
        sys.exit(1)
    target_rs = [r for r in target_rs if r in EXP_NUMBERS]
    if not target_rs:
        print(f"{_c('[ARG]', 'red', True)} --request_numbers is empty or out of allowed set: 5,10,20,30,50,100")
        sys.exit(1)

    if args.total_files != EXPECTED_TOTAL_FILES:
        print(f"{_c('[INFO]', 'yellow', True)} total_files overridden to {EXPECTED_TOTAL_FILES} for physical isolation")
        args.total_files = EXPECTED_TOTAL_FILES

    dist_label = DIST_DISPLAY.get(args.dist_name, "<unknown_distribution>")
    print(_c("=" * 72, "blue", True))
    print(f"{_c('[GENERATOR]', 'blue', True)} distribution: {dist_label}")
    print(f"{_c('[TARGET]', 'cyan', True)} {args.target_folder}")
    print(f"{_c('[FILES]', 'cyan', True)} total_files={args.total_files}")
    print(f"{_c('[WORKERS]', 'cyan', True)} {args.workers}")
    print(f"{_c('[REQUESTS]', 'cyan', True)} {target_rs}")
    if args.seed is not None:
        print(f"{_c('[SEED]', 'cyan', True)} {args.seed}")
    print(_c("=" * 72, "blue", True))

    start_all = time.time()
    if args.seed is not None:
        np.random.seed(args.seed)

    MAX_EVT = 60
    full_matrix, meta_rows = get_distribution_matrix(args.dist_name, args.total_files, MAX_EVT)

    if os.path.isabs(args.target_folder):
        base_out = args.target_folder
    else:
        base_out = os.path.join(OUTPUT_ROOT, args.target_folder)
    if not os.path.exists(base_out):
        os.makedirs(base_out, exist_ok=True)

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(DATA_FILE, EXP_NUMBERS, FIGURES_DIR),
    ) as executor:
        for r in target_rs:
            print(f"\n{_c('[RUN]', 'green', True)} generating R_{r} ...")
            r_dir = os.path.join(base_out, f"R{r}")
            os.makedirs(r_dir, exist_ok=True)

            tasks = []
            for i in range(args.total_files):
                tasks.append((i, r, full_matrix[i], r_dir, meta_rows[i], args.seed))

            futures = [executor.submit(generate_single_file, t) for t in tasks]
            failures = 0
            if HAS_TQDM:
                iterator = tqdm(as_completed(futures), total=len(futures), unit="file", ncols=80)
            else:
                iterator = as_completed(futures)
            done = 0
            for future in iterator:
                done += 1
                try:
                    ok = future.result()
                except Exception:
                    ok = False
                if not ok:
                    failures += 1
                if not HAS_TQDM and done % 100 == 0:
                    sys.stdout.write(f"\r{_c('[PROGRESS]', 'magenta', True)} {done}/{len(futures)}")
                    sys.stdout.flush()
            if not HAS_TQDM:
                print("")
            if failures:
                print(f"{_c('[ERROR]', 'red', True)} generator failed on {failures} files.")
                sys.exit(1)

            if args.verify:
                print(f"{_c('[VERIFY]', 'yellow', True)} verifying generated Excel files...")
                verify_failures = verify_output_dir(r_dir, args.total_files)
                if verify_failures:
                    print(f"{_c('[ERROR]', 'red', True)} verification failed on {verify_failures} files.")
                    sys.exit(1)

    print(f"\n{_c('[DONE]', 'green', True)} generation finished in {time.time() - start_all:.2f}s")
if __name__ == "__main__":
    main()


import os
import time
import argparse
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

DIST_DISPLAY = {
    'mixed_v1': '?? v1 (25% ?? + 75% ??)',
    'stress_test': '???? (100% ?????120)',
    'chaos_uniform': '???? (100% ?? 10-100)',
    'curriculum_easy': '???? (50% ?? + 50% ??)',
    'baseline_legacy': '???? (????????20)',
    'normal_mix_8_80_50_50': '?? 50/50 (??8 / 80)',
    'lognormal_mix_8_80_50_50': '???? 50/50 (??8 / 80)',
    'normal_mix_8_80_75_25': '?? 75/25 (??8 / 80)',
    'lognormal_mix_8_80_75_25': '???? 75/25 (??8 / 80)',
    'normal_mix_80_8_25_75': '?? 25/75 (??80 -> 8)',
    'normal_mix_80_8_50_50': '?? 50/50 (??80 -> 8)',
    'normal_mix_80_8_75_25': '?? 75/25 (??80 -> 8)',
    'lognormal_mix_80_8_25_75': '???? 25/75 (??80 -> 8)',
    'lognormal_mix_80_8_50_50': '???? 50/50 (??80 -> 8)',
    'lognormal_mix_9_30_3_30_30_40': '???? 30/30/40 (??9 / 30 / 3)'
}

# 子进程全局缓存
GLOBAL_DATA = {}

def get_distribution_matrix(dist_name, total_files, max_events):
    """
    【兵工厂核心】根据策略生成随机数矩阵 [Files, Events]
    """
    dist_label = DIST_DISPLAY.get(dist_name, "????")
    print(f"   -> ????????: {dist_label}")
    
    if dist_name == "mixed_v1":
        # 混合 V1: 25% 高压 (RL触发区), 75% 混沌
        n_stress = int(total_files * 0.25)
        n_chaos = total_files - n_stress
        # Stress: Mean=120, Std=30
        mat_1 = np.random.normal(120, 30, (n_stress, max_events))
        # Chaos: Uniform 10-100
        mat_2 = np.random.uniform(10, 100, (n_chaos, max_events))
        matrix = np.vstack([mat_1, mat_2])
        
    elif dist_name == "stress_test":
        # 全高压: 模拟严重拥堵
        matrix = np.random.normal(120, 30, (total_files, max_events))
        
    elif dist_name == "chaos_uniform":
        # 全混沌: 完全随机
        matrix = np.random.uniform(10, 100, (total_files, max_events))
        
    elif dist_name == "curriculum_easy":
        # 课程学习: 前50%简单，后50%逐渐变难
        n_easy = int(total_files * 0.5)
        n_hard = total_files - n_easy
        # LogNormal (Legacy-like)
        mat_1 = np.random.lognormal(3, 0.5, (n_easy, max_events)) 
        # Normal (Medium)
        mat_2 = np.random.normal(60, 15, (n_hard, max_events))
        matrix = np.vstack([mat_1, mat_2])
        
    elif dist_name == "baseline_legacy":
        # 模仿原始数据分布 (LogNormal)
        matrix = np.random.lognormal(3, 0.5, (total_files, max_events))
    elif dist_name == "normal_mix_8_80_50_50":
        n1 = int(total_files * 0.5)
        n2 = total_files - n1
        mat_1 = np.random.normal(8, 2, (n1, max_events))
        mat_2 = np.random.normal(80, 20, (n2, max_events))
        matrix = np.vstack([mat_1, mat_2])
    elif dist_name == "lognormal_mix_8_80_50_50":
        n1 = int(total_files * 0.5)
        n2 = total_files - n1
        mat_1 = np.random.lognormal(mean=2, sigma=0.5, size=(n1, max_events))
        mat_2 = np.random.lognormal(mean=4.4, sigma=0.5, size=(n2, max_events))
        matrix = np.vstack([mat_1, mat_2])
    elif dist_name == "normal_mix_8_80_75_25":
        n1 = int(total_files * 0.75)
        n2 = total_files - n1
        mat_1 = np.random.normal(8, 2, (n1, max_events))
        mat_2 = np.random.normal(80, 20, (n2, max_events))
        matrix = np.vstack([mat_1, mat_2])
    elif dist_name == "lognormal_mix_8_80_75_25":
        n1 = int(total_files * 0.75)
        n2 = total_files - n1
        mat_1 = np.random.lognormal(mean=2, sigma=0.5, size=(n1, max_events))
        mat_2 = np.random.lognormal(mean=4.4, sigma=0.5, size=(n2, max_events))
        matrix = np.vstack([mat_1, mat_2])
    elif dist_name == "normal_mix_80_8_25_75":
        n1 = int(total_files * 0.25)
        n2 = total_files - n1
        mat_1 = np.random.normal(80, 20, (n1, max_events))
        mat_2 = np.random.normal(8, 2, (n2, max_events))
        matrix = np.vstack([mat_1, mat_2])
    elif dist_name == "normal_mix_80_8_50_50":
        n1 = int(total_files * 0.5)
        n2 = total_files - n1
        mat_1 = np.random.normal(80, 20, (n1, max_events))
        mat_2 = np.random.normal(8, 2, (n2, max_events))
        matrix = np.vstack([mat_1, mat_2])
    elif dist_name == "normal_mix_80_8_75_25":
        n1 = int(total_files * 0.75)
        n2 = total_files - n1
        mat_1 = np.random.normal(80, 20, (n1, max_events))
        mat_2 = np.random.normal(8, 2, (n2, max_events))
        matrix = np.vstack([mat_1, mat_2])
    elif dist_name == "lognormal_mix_80_8_25_75":
        n1 = int(total_files * 0.25)
        n2 = total_files - n1
        mat_1 = np.random.lognormal(mean=4.4, sigma=0.5, size=(n1, max_events))
        mat_2 = np.random.lognormal(mean=2, sigma=0.5, size=(n2, max_events))
        matrix = np.vstack([mat_1, mat_2])
    elif dist_name == "lognormal_mix_80_8_50_50":
        n1 = int(total_files * 0.5)
        n2 = total_files - n1
        mat_1 = np.random.lognormal(mean=4.4, sigma=0.5, size=(n1, max_events))
        mat_2 = np.random.lognormal(mean=2, sigma=0.5, size=(n2, max_events))
        matrix = np.vstack([mat_1, mat_2])
    elif dist_name == "lognormal_mix_9_30_3_30_30_40":
        n1 = int(total_files * 0.3)
        n2 = int(total_files * 0.3)
        n3 = total_files - n1 - n2
        mu_9 = np.log(9) - 0.125
        mu_30 = np.log(30) - 0.125
        mu_3 = np.log(3) - 0.125
        mat_1 = np.random.lognormal(mean=mu_9, sigma=0.5, size=(n1, max_events))
        mat_2 = np.random.lognormal(mean=mu_30, sigma=0.5, size=(n2, max_events))
        mat_3 = np.random.lognormal(mean=mu_3, sigma=0.5, size=(n3, max_events))
        matrix = np.vstack([mat_1, mat_2, mat_3])
        
    else:
        print("?????????????????")
        matrix = np.random.lognormal(3, 0.5, (total_files, max_events))

    # 截断: 保证最小 Duration 为 1 分钟，防止负数
    matrix = np.maximum(matrix, 1)
    # 取整
    return matrix.astype(int)

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
    idx, r, duration_row, out_dir = args
    
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
    parser.add_argument("--total_files", type=int, default=1000)
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

    dist_label = DIST_DISPLAY.get(args.dist_name, "????")
    print(f"=== ?????: {dist_label} ===")
    print(f"   ????: .../{os.path.basename(args.target_folder)}")
    print(f"   ?? R ???: {args.total_files}")
    print(f"   ????: {args.workers}")
    print(f"   R 集合: {target_rs}")

    start_all = time.time()
    
    # 1. 生成数据矩阵
    MAX_EVT = 60
    full_matrix = get_distribution_matrix(args.dist_name, args.total_files, MAX_EVT)
    
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
                tasks.append((i, r, full_matrix[i], r_dir))
            
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

#!/usr/bin/env Python
# coding=utf-8
"""
结果验证脚本 - 检查实验结果文件是否正确生成
"""

import os
import pandas as pd
import glob
from pathlib import Path

def check_obj_record_files(base_path):
    """
    检查obj_record文件是否生成并验证其内容

    Args:
        base_path: 实验结果基础路径
    """
    print("="*80)
    print("检查obj_record文件生成情况")
    print("="*80)

    # 查找所有实验目录
    experiment_dirs = glob.glob(os.path.join(base_path, "experiment34959", "percentage*parallel*dynamic*"))

    if not experiment_dirs:
        print("[ERROR] 未找到实验目录")
        return

    total_files = 0
    missing_files = 0
    empty_files = 0
    valid_files = 0

    for exp_dir in sorted(experiment_dirs):
        exp_name = os.path.basename(exp_dir)
        print(f"\n[DIR] 检查实验目录: {exp_name}")

        # 查找obj_record文件
        obj_record_pattern = os.path.join(exp_dir, "obj_record*.xlsx")
        obj_record_files = glob.glob(obj_record_pattern)

        if not obj_record_files:
            print(f"  [ERROR] 未找到obj_record文件")
            missing_files += 1
            continue

        for obj_file in obj_record_files:
            total_files += 1
            file_name = os.path.basename(obj_file)

            try:
                # 检查文件大小
                file_size = os.path.getsize(obj_file)
                if file_size == 0:
                    print(f"  [WARNING] 文件为空: {file_name}")
                    empty_files += 1
                    continue

                # 尝试读取Excel文件
                df = pd.read_excel(obj_file, sheet_name='obj_record')

                if df.empty:
                    print(f"  [WARNING] Excel表格为空: {file_name}")
                    empty_files += 1
                else:
                    print(f"  [OK] 文件有效: {file_name} ({len(df)} 行, {len(df.columns)} 列)")
                    valid_files += 1

                    # 显示关键指标
                    if 'overall_cost' in df.columns and 'served_requests' in df.columns:
                        latest_cost = df['overall_cost'].iloc[-1]
                        latest_served = df['served_requests'].iloc[-1]
                        print(f"     [DATA] 最新成本: {latest_cost:.2f}, 服务请求: {latest_served}")

            except Exception as e:
                print(f"  [ERROR] 文件读取失败: {file_name} - {str(e)}")
                missing_files += 1

    print(f"\n" + "="*80)
    print("检查结果汇总")
    print("="*80)
    print(f"实验目录总数: {len(experiment_dirs)}")
    print(f"obj_record文件总数: {total_files}")
    print(f"[OK] 有效文件: {valid_files}")
    print(f"[WARNING] 空文件: {empty_files}")
    print(f"[ERROR] 缺失/损坏文件: {missing_files}")

    if missing_files == 0 and empty_files == 0:
        print("\n[SUCCESS] 所有obj_record文件都已正确生成！")
    else:
        print(f"\n[WARNING] 发现 {missing_files + empty_files} 个问题文件需要检查")

def check_other_result_files(base_path):
    """
    检查其他结果文件

    Args:
        base_path: 实验结果基础路径
    """
    print("\n" + "="*80)
    print("检查其他结果文件")
    print("="*80)

    experiment_dirs = glob.glob(os.path.join(base_path, "experiment34959", "percentage*parallel*dynamic*"))

    for exp_dir in sorted(experiment_dirs):
        exp_name = os.path.basename(exp_dir)
        print(f"\n[DIR] {exp_name}")

        # 检查best_routes文件
        best_routes_files = glob.glob(os.path.join(exp_dir, "best_routes*.xlsx"))
        print(f"  best_routes文件: {len(best_routes_files)} 个")

        # 检查routes_match文件
        routes_match_files = glob.glob(os.path.join(exp_dir, "routes_match*.xlsx"))
        print(f"  routes_match文件: {len(routes_match_files)} 个")

        # 检查PDF文件
        pdf_files = glob.glob(os.path.join(exp_dir, "*.pdf"))
        print(f"  PDF图表文件: {len(pdf_files)} 个")

def main():
    """
    主函数
    """
    # 默认路径 - 修正路径中的特殊字符
    base_path = os.path.join(os.getcwd().replace("codes", ""), "Uncertainties Dynamic planning under unexpected events", "Figures")

    if not os.path.exists(base_path):
        print(f"[ERROR] 结果目录不存在: {base_path}")
        return

    print("[INFO] 开始验证实验结果文件...")
    print(f"[INFO] 检查路径: {base_path}")

    # 检查obj_record文件
    check_obj_record_files(base_path)

    # 检查其他结果文件
    check_other_result_files(base_path)

    print("\n" + "="*80)
    print("[SUCCESS] 验证完成")
    print("="*80)

if __name__ == "__main__":
    main()
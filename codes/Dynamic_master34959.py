#!/usr/bin/env Python
# coding=utf-8

# 抑制tkinter线程退出错误
import os
os.environ['TK_SILENCE_DEPRECATION'] = '1'

import concurrent.futures
import threading
import argparse
import Dynamic_ALNS_RL34959
import dynamic_RL34959
import pandas as pd
import os
import time
import warnings
import sys
# 尝试导入GPU相关模块
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("警告: PyTorch不可用，GPU功能将被禁用")
    TORCH_AVAILABLE = False
    torch = None

try:
    from GPUtil import showUtilization as gpu_usage
    GPUTIL_AVAILABLE = True
except ImportError:
    print("警告: GPUtil不可用，GPU监控功能将被禁用")
    GPUTIL_AVAILABLE = False
    def gpu_usage():
        print("GPU监控不可用")

try:
    from numba import cuda
    NUMBA_AVAILABLE = True
except ImportError:
    print("警告: Numba不可用，CUDA功能将被禁用")
    NUMBA_AVAILABLE = False
    cuda = None

# 导入分布配置
try:
    from distribution_config import DistributionConfig
except ImportError:
    print("警告: 无法导入distribution_config模块，将使用默认配置")
    DistributionConfig = None

def free_gpu_cache(GPU_number):
    """
    清理GPU缓存
    """
    if not TORCH_AVAILABLE or not NUMBA_AVAILABLE:
        print("GPU功能不可用，跳过缓存清理")
        return

    try:
        print("Initial GPU Usage")
        if GPUTIL_AVAILABLE:
            gpu_usage()

        torch.cuda.empty_cache()

        if NUMBA_AVAILABLE and cuda:
            cuda.select_device(GPU_number)
            cuda.close()
            cuda.select_device(GPU_number)

        print("GPU Usage after emptying the cache")
        if GPUTIL_AVAILABLE:
            gpu_usage()

    except Exception as e:
        print(f"GPU缓存清理失败: {e}")
        print("继续执行程序...")
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


def parse_arguments():
    """
    解析命令行参数

    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description='ALNS-RL动态多式联运优化系统主程序',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用默认分布运行
  python Dynamic_master34959.py

  # 使用指定分布运行
  python Dynamic_master34959.py --distribution normal_mean80_std20

  # 列出所有可用分布
  python Dynamic_master34959.py --list-distributions

  # 不使用强化学习
  python Dynamic_master34959.py --no-rl

  # 设置并行工作进程数
  python Dynamic_master34959.py --workers 4

  # 设置最大处理文件数
  python Dynamic_master34959.py --max-tables 500

  # 无限循环模式（0）
  python Dynamic_master34959.py --max-tables 0
        """
    )

    # 分布参数
    distribution_choices = ['default']
    if DistributionConfig:
        distribution_choices = DistributionConfig.get_distribution_names()

    parser.add_argument(
        '--distribution', '-d',
        type=str,
        choices=distribution_choices,
        default='default',
        help='指定不确定性事件的分布类型 (默认: default)'
    )

    # 系统参数
    parser.add_argument(
        '--no-rl',
        action='store_true',
        help='不使用强化学习，仅运行ALNS算法'
    )

    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=2,
        help='并行工作进程数 (默认: 2)'
    )

    parser.add_argument(
        '--max-tables', '-m',
        type=int,
        default=100,
        help='最大处理的数据文件数量 (默认: 100, 0=无限循环)'
    )

    parser.add_argument(
        '--gpu',
        type=int,
        choices=[0, 1],
        help='指定使用的GPU编号 (0或1)'
    )

    # 信息参数
    parser.add_argument(
        '--list-distributions', '-l',
        action='store_true',
        help='列出所有可用的分布配置并退出'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细运行信息'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='ALNS-RL Dynamic Optimization System v2.0'
    )

    return parser.parse_args()


def list_available_distributions():
    """
    列出所有可用的分布配置
    """
    if not DistributionConfig:
        print("分布配置模块不可用")
        return

    print("\n" + "="*80)
    print("可用的不确定性事件分布配置")
    print("="*80)

    categories = DistributionConfig.get_categories()
    for category in categories:
        configs = DistributionConfig.get_configs_by_category(category)
        print(f"\n{category.upper()} 分布系列 ({len(configs)} 个)")
        print("-" * 60)

        for name, config in configs.items():
            print(f"  * {name}")
            print(f"    描述: {config['description']}")
            print(f"    参数: {config['params']}")
            print(f"    用途: {config['use_case']}")

    print(f"\n总计: {len(DistributionConfig.get_all_configs())} 个分布配置")
    print("="*80)


def main():
    """
    主函数：解析参数并执行优化任务
    """
    # 解析命令行参数
    args = parse_arguments()

    # 处理信息类参数
    if args.list_distributions:
        list_available_distributions()
        return

    # 显示配置信息
    if args.verbose:
        print("="*80)
        print("ALNS-RL 动态多式联运优化系统 v2.0")
        print("="*80)
        print(f"分布配置: {args.distribution}")
        print(f"强化学习: {'禁用' if args.no_rl else '启用'}")
        print(f"并行进程数: {args.workers}")
        print(f"最大数据文件数: {'无限循环' if args.max_tables == 0 else args.max_tables}")
        if args.gpu is not None:
            print(f"使用GPU: {args.gpu}")
        print("="*80)

    # 设置全局变量
    global add_RL, parallel_number

    # 根据参数调整配置
    add_RL = 0 if args.no_rl else 1

    # 设置并行进程数
    if combine_insertion_and_removal_operators == 1:
        max_workers = min(args.workers, 2)
        parallel_number = list(range(0, max_workers))
    else:
        max_workers = min(args.workers, 3)
        parallel_number = list(range(0, max_workers))

    if args.verbose:
        print(f"实际并行进程数: {len(parallel_number)}")
        print(f"进程列表: {parallel_number}")

    # GPU设置
    if args.gpu is not None:
        try:
            free_gpu_cache(args.gpu)
            print(f"使用GPU {args.gpu}")
        except Exception as e:
            print(f"GPU设置失败: {e}")

    # 清理临时文件
    if os.path.exists('34959.txt'):
        os.remove('34959.txt')

    try:
        # 运行优化算法
        if add_RL == 0:
            print("运行ALNS算法...")
            Dynamic_ALNS_RL34959.main(0, distribution_name=args.distribution, max_tables=args.max_tables)
        else:
            print("运行ALNS-RL混合算法...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(parallel_number)) as executor:
                # 提交任务
                futures = {
                    executor.submit(Dynamic_ALNS_RL34959.main, approach, args.distribution, args.max_tables): approach
                    for approach in parallel_number
                }

                # 等待任务完成
                for future in concurrent.futures.as_completed(futures):
                    approach = futures[future]
                    try:
                        result = future.result()
                        if args.verbose:
                            print(f"进程 {approach} 完成")
                    except Exception as exc:
                        print(f"进程 {approach} 发生异常: {exc}")
                        if not args.verbose:
                            raise

                    # 标记完成状态
                    with open('34959.txt', 'w', encoding='utf-8') as f:
                        f.write('do not wait anymore')

        print("优化任务完成!")

    except KeyboardInterrupt:
        print("\n用户中断了优化过程")
        sys.exit(1)
    except Exception as e:
        print(f"\n优化过程中发生错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

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




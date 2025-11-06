#!/usr/bin/env Python
# -*- coding: utf-8 -*-
"""
自动化不确定性事件数据生成脚本
一键生成所有分布配置的数据文件

Author: 资深后端工程师
Date: 2025-11-06
Description: 批量生成所有分布配置的不确定性事件数据
"""

import argparse
import sys
import os
import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from tqdm import tqdm

# 导入配置和生成器
from distribution_config import DistributionConfig
from generate_un_expected_events_by_stochastic_info_RL_v2 import UncertaintyEventGenerator


class BatchDistributionGenerator:
    """
    批量分布生成器
    负责管理多个分布配置的批量生成
    """

    def __init__(self, base_output_dir: str = "Uncertainties_Dynamic_Planning",
                 base_data_path: str = "A:/MYpython/34959_RL/Intermodal_EGS_data_all.xlsx"):
        """
        初始化批量生成器

        Args:
            base_output_dir (str): 基础输出目录
            base_data_path (str): 基础数据文件路径
        """
        self.base_output_dir = base_output_dir
        self.base_data_path = base_data_path
        self.logger = self._setup_logger()
        self.generator = UncertaintyEventGenerator(base_data_path)

    def _setup_logger(self) -> logging.Logger:
        """
        设置批量生成专用日志记录器

        Returns:
            logging.Logger: 配置好的日志记录器
        """
        logger = logging.getLogger('BatchDistributionGenerator')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            # 文件handler
            file_handler = logging.FileHandler(
                'batch_distribution_generation.log', encoding='utf-8'
            )
            file_handler.setLevel(logging.INFO)

            # 控制台handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # 格式化器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        return logger

    def generate_all_distributions(self, verbose: bool = True,
                                 specific_distribution: Optional[str] = None) -> Dict[str, Any]:
        """
        生成所有分布配置的数据

        Args:
            verbose (bool): 是否显示详细进度信息
            specific_distribution (Optional[str]): 指定生成特定分布，None表示生成所有分布

        Returns:
            Dict[str, Any]: 批量生成统计信息
        """
        start_time = datetime.now()
        self.logger.info("="*80)
        self.logger.info("开始批量生成不确定性事件数据")
        self.logger.info(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("="*80)

        # 获取要生成的分布配置
        if specific_distribution:
            try:
                distributions = {
                    specific_distribution: DistributionConfig.get_config(specific_distribution)
                }
                self.logger.info(f"生成指定分布: {specific_distribution}")
            except ValueError as e:
                self.logger.error(f"指定的分布配置不存在: {e}")
                return {"error": str(e)}
        else:
            distributions = DistributionConfig.get_all_configs()
            self.logger.info(f"生成所有分布配置，共 {len(distributions)} 个")

        # 批量生成统计
        batch_stats = {
            'start_time': start_time.isoformat(),
            'total_distributions': len(distributions),
            'successful_generations': 0,
            'failed_generations': 0,
            'total_files_created': 0,
            'total_events_generated': 0,
            'distribution_results': {},
            'errors': [],
            'processing_time_seconds': 0
        }

        # 进度条配置
        distributions_items = list(distributions.items())
        if verbose:
            pbar = tqdm(distributions_items, desc="生成分布数据", unit="distribution")
        else:
            pbar = distributions_items

        # 逐个生成分布数据
        for config_name, distribution_details in pbar:
            try:
                if verbose:
                    pbar.set_description(f"生成 {config_name}")

                self.logger.info(f"正在生成分布: {config_name}")

                # 生成单个分布的数据
                distribution_stats = self.generator.generate_events(
                    config_name,
                    distribution_details,
                    self.base_output_dir
                )

                # 更新统计信息
                batch_stats['successful_generations'] += 1
                batch_stats['total_files_created'] += distribution_stats.get('total_files_created', 0)
                batch_stats['total_events_generated'] += distribution_stats.get('total_events_generated', 0)
                batch_stats['distribution_results'][config_name] = distribution_stats

                self.logger.info(f"[SUCCESS] {config_name} 生成成功: "
                               f"{distribution_stats.get('total_files_created', 0)} 个文件, "
                               f"{distribution_stats.get('total_events_generated', 0)} 个事件")

            except Exception as e:
                error_msg = f"[ERROR] 生成分布 {config_name} 失败: {e}"
                self.logger.error(error_msg)
                batch_stats['failed_generations'] += 1
                batch_stats['errors'].append(error_msg)

        # 计算处理时间
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        batch_stats['end_time'] = end_time.isoformat()
        batch_stats['processing_time_seconds'] = processing_time

        # 记录最终统计
        self.logger.info("="*80)
        self.logger.info("批量生成完成")
        self.logger.info(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"总处理时间: {processing_time:.2f} 秒")
        self.logger.info(f"成功生成: {batch_stats['successful_generations']}/{batch_stats['total_distributions']} 个分布")
        self.logger.info(f"总文件数: {batch_stats['total_files_created']}")
        self.logger.info(f"总事件数: {batch_stats['total_events_generated']}")
        if batch_stats['errors']:
            self.logger.warning(f"错误数量: {len(batch_stats['errors'])}")
            for error in batch_stats['errors']:
                self.logger.warning(f"  - {error}")
        self.logger.info("="*80)

        return batch_stats

    def generate_by_category(self, category: str, verbose: bool = True) -> Dict[str, Any]:
        """
        按类别生成分布数据

        Args:
            category (str): 分布类别 (lognormal, normal, uniform, exponential)
            verbose (bool): 是否显示详细进度信息

        Returns:
            Dict[str, Any]: 生成统计信息
        """
        distributions = DistributionConfig.get_configs_by_category(category)
        if not distributions:
            self.logger.error(f"未找到类别 '{category}' 的分布配置")
            return {"error": f"类别 '{category}' 不存在"}

        self.logger.info(f"按类别生成: {category}, 包含 {len(distributions)} 个分布")

        # 临时修改生成器为按类别生成
        batch_stats = {
            'category': category,
            'start_time': datetime.now().isoformat(),
            'distributions_in_category': len(distributions),
            'distribution_results': {}
        }

        for config_name, distribution_details in distributions.items():
            try:
                self.logger.info(f"生成 {config_name}")
                stats = self.generator.generate_events(
                    config_name, distribution_details, self.base_output_dir
                )
                batch_stats['distribution_results'][config_name] = stats
                batch_stats['total_files_created'] = batch_stats.get('total_files_created', 0) + stats.get('total_files_created', 0)
                batch_stats['total_events_generated'] = batch_stats.get('total_events_generated', 0) + stats.get('total_events_generated', 0)

            except Exception as e:
                self.logger.error(f"生成 {config_name} 失败: {e}")
                batch_stats['errors'] = batch_stats.get('errors', []) + [str(e)]

        batch_stats['end_time'] = datetime.now().isoformat()
        return batch_stats

    def list_available_distributions(self) -> None:
        """
        列出所有可用的分布配置
        """
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

    def validate_environment(self) -> bool:
        """
        验证生成环境

        Returns:
            bool: 环境是否有效
        """
        self.logger.info("验证生成环境...")

        # 检查基础数据文件
        if not os.path.exists(self.base_data_path):
            self.logger.error(f"基础数据文件不存在: {self.base_data_path}")
            return False

        # 检查输出目录权限
        try:
            test_dir = os.path.join(self.base_output_dir, "test")
            os.makedirs(test_dir, exist_ok=True)
            test_file = os.path.join(test_dir, "test.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            os.rmdir(test_dir)
        except Exception as e:
            self.logger.error(f"输出目录权限检查失败: {e}")
            return False

        # 检查分布配置
        try:
            configs = DistributionConfig.get_all_configs()
            if not configs:
                self.logger.error("没有找到任何分布配置")
                return False
        except Exception as e:
            self.logger.error(f"分布配置检查失败: {e}")
            return False

        self.logger.info("[SUCCESS] 环境验证通过")
        return True

    def save_batch_report(self, batch_stats: Dict[str, Any]) -> None:
        """
        保存批量生成报告

        Args:
            batch_stats (Dict[str, Any]): 批量生成统计信息
        """
        try:
            import json

            report_path = os.path.join(
                self.base_output_dir,
                f"batch_generation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(batch_stats, f, indent=4, ensure_ascii=False)

            self.logger.info(f"[INFO] 批量生成报告已保存到: {report_path}")

        except Exception as e:
            self.logger.error(f"保存批量报告失败: {e}")


def main():
    """
    主函数：处理命令行参数并执行生成任务
    """
    parser = argparse.ArgumentParser(
        description="批量生成不确定性事件数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 生成所有分布数据
  python generate_all_distributions.py

  # 生成特定分布
  python generate_all_distributions.py --distribution normal_mean80_std20

  # 按类别生成
  python generate_all_distributions.py --category normal

  # 静默模式（无进度条）
  python generate_all_distributions.py --quiet

  # 列出所有可用分布
  python generate_all_distributions.py --list

  # 验证环境
  python generate_all_distributions.py --validate
        """
    )

    parser.add_argument(
        '--distribution', '-d',
        type=str,
        help='生成特定的分布配置'
    )

    parser.add_argument(
        '--category', '-c',
        type=str,
        choices=['lognormal', 'normal', 'uniform', 'exponential'],
        help='按类别生成分布数据'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='Uncertainties_Dynamic_Planning',
        help='输出目录 (默认: Uncertainties_Dynamic_Planning)'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default="A:/MYpython/34959_RL/Intermodal_EGS_data_all.xlsx",
        help='基础数据文件路径'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='静默模式，不显示进度条'
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='列出所有可用的分布配置'
    )

    parser.add_argument(
        '--validate', '-v',
        action='store_true',
        help='验证生成环境'
    )

    args = parser.parse_args()

    # 创建批量生成器
    generator = BatchDistributionGenerator(
        base_output_dir=args.output_dir,
        base_data_path=args.data_path
    )

    # 处理不同的命令
    if args.list:
        generator.list_available_distributions()
        return

    if args.validate:
        if generator.validate_environment():
            print("[SUCCESS] 环境验证通过，可以开始生成数据")
        else:
            print("[ERROR] 环境验证失败，请检查配置")
            sys.exit(1)
        return

    # 执行生成任务
    try:
        if args.category:
            # 按类别生成
            stats = generator.generate_by_category(args.category, not args.quiet)
        elif args.distribution:
            # 生成特定分布
            stats = generator.generate_all_distributions(
                not args.quiet, args.distribution
            )
        else:
            # 生成所有分布
            stats = generator.generate_all_distributions(not args.quiet)

        # 保存批量报告
        generator.save_batch_report(stats)

        # 显示摘要
        print(f"\n[COMPLETE] 生成完成!")
        print(f"成功: {stats.get('successful_generations', 0)}/{stats.get('total_distributions', 0)} 个分布")
        print(f"文件: {stats.get('total_files_created', 0)} 个")
        print(f"事件: {stats.get('total_events_generated', 0)} 个")
        if stats.get('errors'):
            print(f"[WARNING] 错误: {len(stats['errors'])} 个")

    except KeyboardInterrupt:
        print("\n[WARNING] 用户中断了生成过程")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] 生成过程中发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
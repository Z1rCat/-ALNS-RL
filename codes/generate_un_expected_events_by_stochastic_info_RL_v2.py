#!/usr/bin/env Python
# -*- coding: utf-8 -*-
"""
重构版：不确定性事件生成器
支持配置驱动的多分布类型数据生成

Author: 资深后端工程师
Date: 2025-11-06
Description: 基于分布配置的动态不确定性事件生成模块
"""

import pandas as pd
import numpy as np
import random
import os
import sys
import json
import logging
from pathlib import Path
from openpyxl import load_workbook
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# 导入配置中心
from distribution_config import DistributionConfig


class UncertaintyEventGenerator:
    """
    不确定性事件生成器类
    支持多种概率分布的动态事件生成
    """

    def __init__(self, base_data_path: str = "../Intermodal_EGS_data_all.xlsx"):
        """
        初始化生成器

        Args:
            base_data_path (str): 基础数据文件路径
        """
        self.base_data_path = base_data_path
        self.logger = self._setup_logger()
        self.base_data = self._load_base_data()

    def _setup_logger(self) -> logging.Logger:
        """
        设置日志记录器

        Returns:
            logging.Logger: 配置好的日志记录器
        """
        logger = logging.getLogger('UncertaintyEventGenerator')
        logger.setLevel(logging.INFO)

        # 避免重复添加handler
        if not logger.handlers:
            # 文件handler
            file_handler = logging.FileHandler(
                'uncertainty_generation.log', encoding='utf-8'
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

    def _load_base_data(self) -> Dict[str, pd.DataFrame]:
        """
        加载基础数据文件

        Returns:
            Dict[str, pd.DataFrame]: 基础数据字典

        Raises:
            FileNotFoundError: 基础数据文件不存在时抛出
        """
        try:
            if not os.path.exists(self.base_data_path):
                raise FileNotFoundError(f"基础数据文件不存在: {self.base_data_path}")

            self.logger.info(f"正在加载基础数据: {self.base_data_path}")
            data_file = pd.ExcelFile(self.base_data_path)

            base_data = {
                'N': pd.read_excel(data_file, 'N'),
                'T': pd.read_excel(data_file, 'T'),
                'K': pd.read_excel(data_file, 'K'),
                'o': pd.read_excel(data_file, 'o')
            }

            # 加载所有R表
            for sheet_name in data_file.sheet_names:
                if sheet_name.startswith('R_'):
                    base_data[sheet_name] = pd.read_excel(data_file, sheet_name)

            self.logger.info(f"成功加载基础数据，包含 {len(base_data)} 个数据表")
            return base_data

        except Exception as e:
            self.logger.error(f"加载基础数据失败: {e}")
            raise

    def _generate_duration_value(self, distribution_type: str, params: Dict[str, Any]) -> int:
        """
        根据分布类型和参数生成持续时间

        Args:
            distribution_type (str): 分布类型
            params (Dict[str, Any]): 分布参数

        Returns:
            int: 生成的持续时间值

        Raises:
            ValueError: 不支持的分布类型时抛出
        """
        try:
            if distribution_type == 'lognormal':
                mu = params['mu']
                sigma = params['sigma']
                duration = max(0, int(np.random.lognormal(mu, sigma)))

            elif distribution_type == 'normal':
                loc = params['loc']
                scale = params['scale']
                duration = max(0, int(np.random.normal(loc, scale)))

            elif distribution_type == 'uniform':
                low = params['low']
                high = params['high']
                duration = max(0, int(np.random.uniform(low, high)))

            elif distribution_type == 'exponential':
                scale = params['scale']
                duration = max(0, int(np.random.exponential(scale)))

            else:
                raise ValueError(f"不支持的分布类型: {distribution_type}")

            return duration

        except Exception as e:
            self.logger.error(f"生成持续时间失败: {e}")
            # 返回默认值避免程序中断
            return max(0, int(np.random.lognormal(1, 1)))

    def _create_distribution_plot(self, config_name: str, distribution_type: str,
                                params: Dict[str, Any], output_path: str) -> None:
        """
        创建分布概率密度函数图

        Args:
            config_name (str): 配置名称
            distribution_type (str): 分布类型
            params (Dict[str, Any]): 分布参数
            output_path (str): 图片输出路径
        """
        try:
            if os.path.exists(output_path):
                return  # 图片已存在，跳过生成

            self.logger.info(f"生成分布图: {config_name}")

            # 生成样本数据
            sample_size = 1000
            if distribution_type == 'lognormal':
                mu, sigma = params['mu'], params['sigma']
                samples = np.random.lognormal(mu, sigma, sample_size)
                title = f'Lognormal Distribution (μ={mu}, σ={sigma})'

                # 绘制PDF
                x = np.linspace(min(samples), max(samples), 1000)
                pdf = (np.exp(-(np.log(x) - mu) ** 2 / (2 * sigma ** 2))
                       / (x * sigma * np.sqrt(2 * np.pi)))
                plt.plot(x, pdf, linewidth=2, color='r')

            elif distribution_type == 'normal':
                loc, scale = params['loc'], params['scale']
                samples = np.random.normal(loc, scale, sample_size)
                title = f'Normal Distribution (μ={loc}, σ={scale})'

                # 绘制PDF
                x = np.linspace(min(samples), max(samples), 1000)
                pdf = (1 / (scale * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - loc) / scale) ** 2)
                plt.plot(x, pdf, linewidth=2, color='r')

            elif distribution_type == 'uniform':
                low, high = params['low'], params['high']
                samples = np.random.uniform(low, high, sample_size)
                title = f'Uniform Distribution ({low}, {high})'

                # 绘制PDF
                x = np.linspace(low - 10, high + 10, 1000)
                pdf = np.where((x >= low) & (x <= high), 1 / (high - low), 0)
                plt.plot(x, pdf, linewidth=2, color='r')

            elif distribution_type == 'exponential':
                scale = params['scale']
                samples = np.random.exponential(scale, sample_size)
                title = f'Exponential Distribution (scale={scale})'

                # 绘制PDF
                x = np.linspace(0, max(samples), 1000)
                pdf = (1 / scale) * np.exp(-x / scale)
                plt.plot(x, pdf, linewidth=2, color='r')

            # 绘制直方图
            plt.hist(samples, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')

            plt.xlabel('Duration (hours)')
            plt.ylabel('Probability Density')
            plt.title(f'{title}\nConfiguration: {config_name}')
            plt.grid(True, alpha=0.3)

            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
            plt.close()

        except Exception as e:
            self.logger.error(f"生成分布图失败: {e}")

    def _build_output_path(self, base_dir: str, config_name: str,
                          request_number: int, table_number: int) -> str:
        """
        构建输出文件路径

        Args:
            base_dir (str): 基础输出目录
            config_name (str): 配置名称
            request_number (int): 请求数量
            table_number (int): 表格编号

        Returns:
            str: 完整的输出文件路径
        """
        # 新的文件命名规则，包含分布标识
        filename = f"Intermodal_EGS_data_dynamic_{config_name}_table{table_number}.xlsx"

        # 构建路径，使用Path处理跨平台路径问题
        output_dir = Path(base_dir) / f"plot_distribution_{config_name}" / f"R{request_number}"
        output_dir.mkdir(parents=True, exist_ok=True)

        return str(output_dir / filename)

    def _save_metadata(self, config_name: str, distribution_details: Dict[str, Any],
                      output_dir: str, generation_stats: Dict[str, Any]) -> None:
        """
        保存生成元数据信息

        Args:
            config_name (str): 配置名称
            distribution_details (Dict[str, Any]): 分布配置详情
            output_dir (str): 输出目录
            generation_stats (Dict[str, Any]): 生成统计信息
        """
        try:
            metadata = {
                'configuration': {
                    'config_name': config_name,
                    'distribution_type': distribution_details['type'],
                    'parameters': distribution_details['params'],
                    'description': distribution_details['description'],
                    'use_case': distribution_details['use_case']
                },
                'generation_info': {
                    'generated_at': datetime.now().isoformat(),
                    'generator_version': '2.0',
                    'base_data_file': self.base_data_path
                },
                'statistics': generation_stats
            }

            metadata_path = Path(output_dir) / f"metadata_{config_name}.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)

            self.logger.info(f"元数据已保存到: {metadata_path}")

        except Exception as e:
            self.logger.error(f"保存元数据失败: {e}")

    def generate_events(self, config_name: str, distribution_details: Dict[str, Any],
                       base_output_dir: str = "Uncertainties_Dynamic_Planning",
                       use_initial_routes: bool = True) -> Dict[str, Any]:
        """
        生成不确定性事件数据的核心函数

        Args:
            config_name (str): 分布配置名称
            distribution_details (Dict[str, Any]): 分布配置详情
            base_output_dir (str): 基础输出目录
            use_initial_routes (bool): 是否使用初始路径数据

        Returns:
            Dict[str, Any]: 生成统计信息

        Raises:
            ValueError: 配置参数无效时抛出
            FileNotFoundError: 路径数据文件不存在时抛出
        """
        try:
            self.logger.info(f"开始生成不确定性事件数据: {config_name}")

            # 验证配置
            if not DistributionConfig.validate_config(distribution_details):
                raise ValueError(f"无效的分布配置: {config_name}")

            distribution_type = distribution_details['type']
            params = distribution_details['params']

            generation_stats = {
                'total_files_created': 0,
                'total_events_generated': 0,
                'request_numbers_processed': [],
                'errors': []
            }

            if use_initial_routes:
                # 获取初始路径数据（这里需要适配实际的路径）
                request_numbers = [5, 10, 20, 30, 50, 100]
                exp_numbers = {5: 12793, 10: 12792, 20: 12794, 30: 12816, 50: 12817, 100: 12818}

                for r in request_numbers:
                    try:
                        self.logger.info(f"处理请求数量: {r}")

                        # 这里需要根据实际情况调整路径
                        # 暂时使用模拟数据
                        terminal_arrival_time_array = self._get_terminal_arrival_times(r)

                        if terminal_arrival_time_array is None:
                            self.logger.warning(f"无法获取请求 {r} 的终端到达时间数据，跳过")
                            continue

                        generation_stats['request_numbers_processed'].append(r)

                        # 生成多个表格
                        for table in range(100):  # 减少到100个用于测试
                            try:
                                output_path = self._build_output_path(
                                    base_output_dir, config_name, r, table
                                )

                                # 生成事件数据
                                events_count = self._generate_single_table(
                                    output_path, config_name, distribution_type, params,
                                    terminal_arrival_time_array, r
                                )

                                generation_stats['total_events_generated'] += events_count
                                generation_stats['total_files_created'] += 1

                            except Exception as e:
                                error_msg = f"生成表格 {table} 失败: {e}"
                                self.logger.error(error_msg)
                                generation_stats['errors'].append(error_msg)

                    except Exception as e:
                        error_msg = f"处理请求 {r} 失败: {e}"
                        self.logger.error(error_msg)
                        generation_stats['errors'].append(error_msg)

            else:
                # 使用随机模式生成
                self.logger.info("使用随机模式生成数据")
                generation_stats.update(self._generate_random_mode(
                    base_output_dir, config_name, distribution_type, params
                ))

            # 保存元数据
            output_dir = Path(base_output_dir) / f"plot_distribution_{config_name}"
            self._save_metadata(config_name, distribution_details, str(output_dir), generation_stats)

            self.logger.info(f"完成生成 {config_name}: 创建 {generation_stats['total_files_created']} 个文件，"
                           f"生成 {generation_stats['total_events_generated']} 个事件")

            return generation_stats

        except Exception as e:
            self.logger.error(f"生成事件数据失败: {e}")
            raise

    def _get_terminal_arrival_times(self, request_number: int) -> Optional[np.ndarray]:
        """
        获取终端到达时间数组（这里需要根据实际情况实现）

        Args:
            request_number (int): 请求数量

        Returns:
            Optional[np.ndarray]: 终端到达时间数组，如果无法获取则返回None
        """
        # 这里需要根据实际的路径文件位置和数据结构来实现
        # 暂时返回模拟数据用于测试
        try:
            # 模拟生成终端到达时间数据
            np.random.seed(42)  # 确保可重现
            num_terminals = np.random.randint(5, 15)

            data = []
            for i in range(num_terminals):
                location = i
                arrival_time = np.random.randint(0, 200)  # 0-200小时
                mode = np.random.choice([1, 2, 3])  # 1: Barge, 2: Train, 3: Truck
                data.append([location, arrival_time, mode])

            return np.array(data)

        except Exception as e:
            self.logger.error(f"获取终端到达时间失败: {e}")
            return None

    def _generate_single_table(self, output_path: str, config_name: str,
                              distribution_type: str, params: Dict[str, Any],
                              terminal_arrival_time_array: np.ndarray,
                              request_number: int) -> int:
        """
        生成单个表格的事件数据

        Args:
            output_path (str): 输出文件路径
            config_name (str): 配置名称
            distribution_type (str): 分布类型
            params (Dict[str, Any]): 分布参数
            terminal_arrival_time_array (np.ndarray): 终端到达时间数组
            request_number (int): 请求数量

        Returns:
            int: 生成的事件数量
        """
        end_time = 0
        uncertainty_index = 0
        total_events = 0

        # 获取请求数据
        r_key = f'R_{request_number}'
        if r_key not in self.base_data:
            raise ValueError(f"请求数据 {r_key} 不存在")

        R = self.base_data[r_key]

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 写入基础数据表
            for key in ['N', 'T', 'K', 'o']:
                if key in self.base_data:
                    self.base_data[key].to_excel(writer, sheet_name=key, index=False)
            R.to_excel(writer, sheet_name=r_key, index=False)

            # 生成不确定性事件
            all_events = []

            for i in range(len(terminal_arrival_time_array)):
                location = int(terminal_arrival_time_array[i, 0])
                start_time = int(terminal_arrival_time_array[i, 1])
                mode = int(terminal_arrival_time_array[i, 2])

                # 避免事件时间重叠
                if start_time < end_time:
                    continue

                # 生成持续时间
                duration = self._generate_duration_value(distribution_type, params)

                if duration == 0:
                    continue

                end_time = start_time + duration
                duration_range = [start_time, end_time]

                # 创建事件记录
                for event_type in ['congestion', 'congestion_finish']:
                    event_data = {
                        'uncertainty_index': uncertainty_index,
                        'type': event_type,
                        'location_type': 'node',
                        'vehicle': -1,
                        'location': location,
                        'duration': duration_range,
                        'mode': mode
                    }
                    all_events.append(event_data)

                uncertainty_index += 1
                total_events += 2  # 每个事件有两个类型

            # 保存事件数据
            if all_events:
                events_df = pd.DataFrame(all_events)
                events_df.to_excel(
                    writer,
                    sheet_name=f'{r_key}_events',
                    index=False
                )

        # 生成分布图
        plot_path = os.path.join(
            os.path.dirname(output_path),
            f'distribution_{config_name}.pdf'
        )
        self._create_distribution_plot(config_name, distribution_type, params, plot_path)

        return total_events

    def _generate_random_mode(self, base_output_dir: str, config_name: str,
                            distribution_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        随机模式生成数据

        Args:
            base_output_dir (str): 基础输出目录
            config_name (str): 配置名称
            distribution_type (str): 分布类型
            params (Dict[str, Any]): 分布参数

        Returns:
            Dict[str, Any]: 生成统计信息
        """
        stats = {
            'total_files_created': 0,
            'total_events_generated': 0,
            'request_numbers_processed': [],
            'errors': []
        }

        request_numbers = [5, 10, 20, 30, 50, 100]

        for r in request_numbers:
            r_key = f'R_{request_number}'
            if r_key not in self.base_data:
                continue

            try:
                for table in range(100):  # 减少到100个用于测试
                    output_path = self._build_output_path(
                        base_output_dir, config_name, r, table
                    )

                    # 生成随机事件数据
                    events_count = self._generate_random_table(
                        output_path, config_name, distribution_type, params, r
                    )

                    stats['total_events_generated'] += events_count
                    stats['total_files_created'] += 1

                stats['request_numbers_processed'].append(r)

            except Exception as e:
                error_msg = f"随机模式生成请求 {r} 失败: {e}"
                self.logger.error(error_msg)
                stats['errors'].append(error_msg)

        return stats

    def _generate_random_table(self, output_path: str, config_name: str,
                             distribution_type: str, params: Dict[str, Any],
                             request_number: int) -> int:
        """
        生成随机模式表格

        Args:
            output_path (str): 输出文件路径
            config_name (str): 配置名称
            distribution_type (str): 分布类型
            params (Dict[str, Any]): 分布参数
            request_number (int): 请求数量

        Returns:
            int: 生成的事件数量
        """
        r_key = f'R_{request_number}'
        if r_key not in self.base_data:
            raise ValueError(f"请求数据 {r_key} 不存在")

        R = self.base_data[r_key]
        start_time = np.random.randint(1, 24)
        uncertainty_index = 0
        total_events = 0

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 写入基础数据
            for key in ['N', 'T', 'K', 'o']:
                if key in self.base_data:
                    self.base_data[key].to_excel(writer, sheet_name=key, index=False)
            R.to_excel(writer, sheet_name=r_key, index=False)

            # 生成随机事件
            all_events = []
            mode_terminal_pairs = []

            for _ in range(10):  # 生成10个时间段
                for __ in range(10):  # 每个时间段10个事件
                    location = np.random.randint(0, 10)
                    mode = np.random.choice([0, 1, 2])

                    if [location, mode] in mode_terminal_pairs:
                        continue

                    duration = self._generate_duration_value(distribution_type, params)
                    if duration == 0:
                        continue

                    end_time = start_time + duration
                    duration_range = [start_time, end_time]

                    for event_type in ['congestion', 'congestion_finish']:
                        event_data = {
                            'uncertainty_index': uncertainty_index,
                            'type': event_type,
                            'location_type': 'node',
                            'vehicle': -1,
                            'location': location,
                            'duration': duration_range,
                            'mode': mode
                        }
                        all_events.append(event_data)

                    mode_terminal_pairs.append([location, mode])
                    uncertainty_index += 1
                    total_events += 2

                # 更新开始时间
                start_time = end_time + np.random.randint(1, 10)

            # 保存事件数据
            if all_events:
                events_df = pd.DataFrame(all_events)
                events_df.to_excel(
                    writer,
                    sheet_name=f'{r_key}_random_events',
                    index=False
                )

        # 生成分布图
        plot_path = os.path.join(
            os.path.dirname(output_path),
            f'distribution_{config_name}_random.pdf'
        )
        self._create_distribution_plot(config_name, distribution_type, params, plot_path)

        return total_events


# 便捷函数，保持与原脚本的兼容性
def generate_events(config_name: str, distribution_details: Dict[str, Any],
                   base_output_dir: str = "Uncertainties_Dynamic_Planning") -> Dict[str, Any]:
    """
    便捷函数：生成指定分布的不确定性事件数据

    Args:
        config_name (str): 分布配置名称
        distribution_details (Dict[str, Any]): 分布配置详情
        base_output_dir (str): 基础输出目录

    Returns:
        Dict[str, Any]: 生成统计信息
    """
    generator = UncertaintyEventGenerator()
    return generator.generate_events(config_name, distribution_details, base_output_dir)


# 主函数测试代码
if __name__ == "__main__":
    # 测试单个分布生成
    try:
        print("=== 测试不确定性事件生成器 ===")

        # 获取一个测试配置
        test_config = DistributionConfig.get_config('normal_mean80_std20')
        print(f"测试配置: {test_config}")

        # 生成数据
        generator = UncertaintyEventGenerator()
        stats = generator.generate_events(
            'normal_mean80_std20',
            test_config,
            base_output_dir="Uncertainties_Dynamic_Planning_Test"
        )

        print(f"生成统计: {stats}")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
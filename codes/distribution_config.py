#!/usr/bin/env Python
# -*- coding: utf-8 -*-
"""
不确定性事件分布配置中心
作为整个系统的"不确定性事件菜单"，集中管理所有概率分布及其参数

Author: 资深后端工程师
Date: 2025-11-06
Description: 配置驱动的分布管理模块，支持可扩展的概率分布配置
"""

import json
import os
from typing import Dict, Any, List


class DistributionConfig:
    """
    分布配置管理类
    作为单一事实来源(Single Source of Truth)管理所有不确定性分布参数
    """

    # 静态分布配置字典 - 系统的核心配置数据
    DISTRIBUTIONS = {
        # 对数正态分布系列 - 保持与现有系统兼容
        'lognormal_mu1_sigma1': {
            'type': 'lognormal',
            'params': {'mu': 1, 'sigma': 1},
            'description': '基准对数正态分布',
            'use_case': '轻度影响事件，基准测试场景',
            'category': 'lognormal'
        },
        'lognormal_mu5_sigma1': {
            'type': 'lognormal',
            'params': {'mu': 5, 'sigma': 1},
            'description': '中度影响对数正态分布',
            'use_case': '中等影响事件，标准场景',
            'category': 'lognormal'
        },
        'lognormal_mu80_sigma20': {
            'type': 'lognormal',
            'params': {'mu': 80, 'sigma': 20},
            'description': '高影响对数正态分布',
            'use_case': '严重影响事件，极端场景测试',
            'category': 'lognormal'
        },

        # 正态分布系列 - 新增
        'normal_mean80_std20': {
            'type': 'normal',
            'params': {'loc': 80, 'scale': 20},
            'description': '高均值正态分布',
            'use_case': '长时间延误场景',
            'category': 'normal'
        },
        'normal_mean50_std10': {
            'type': 'normal',
            'params': {'loc': 50, 'scale': 10},
            'description': '中等正态分布',
            'use_case': '标准延误场景',
            'category': 'normal'
        },
        'normal_mean30_std5': {
            'type': 'normal',
            'params': {'loc': 30, 'scale': 5},
            'description': '低均值正态分布',
            'use_case': '短时间延误场景',
            'category': 'normal'
        },

        # 均匀分布系列 - 新增
        'uniform_low10_high150': {
            'type': 'uniform',
            'params': {'low': 10, 'high': 150},
            'description': '宽范围均匀分布',
            'use_case': '随机延误场景，覆盖范围广',
            'category': 'uniform'
        },
        'uniform_low0_high100': {
            'type': 'uniform',
            'params': {'low': 0, 'high': 100},
            'description': '标准均匀分布',
            'use_case': '基础随机场景测试',
            'category': 'uniform'
        },

        # 指数分布系列 - 新增
        'exponential_scale20': {
            'type': 'exponential',
            'params': {'scale': 20},
            'description': '长尾指数分布',
            'use_case': '偶发长时间延误场景',
            'category': 'exponential'
        },
        'exponential_scale10': {
            'type': 'exponential',
            'params': {'scale': 10},
            'description': '标准指数分布',
            'use_case': '常见短时间延误场景',
            'category': 'exponential'
        }
    }

    @staticmethod
    def get_all_configs() -> Dict[str, Dict[str, Any]]:
        """
        获取所有分布配置

        Returns:
            Dict[str, Dict[str, Any]]: 完整的分布配置字典
        """
        return DistributionConfig.DISTRIBUTIONS.copy()

    @staticmethod
    def get_config(name: str) -> Dict[str, Any]:
        """
        获取指定分布的配置

        Args:
            name (str): 分布配置名称

        Returns:
            Dict[str, Any]: 分布配置字典

        Raises:
            ValueError: 当分布名称不存在时抛出异常
        """
        if name not in DistributionConfig.DISTRIBUTIONS:
            available_configs = list(DistributionConfig.DISTRIBUTIONS.keys())
            raise ValueError(f"分布配置 '{name}' 不存在。可用配置: {available_configs}")

        return DistributionConfig.DISTRIBUTIONS[name].copy()

    @staticmethod
    def get_configs_by_category(category: str) -> Dict[str, Dict[str, Any]]:
        """
        按类别获取分布配置

        Args:
            category (str): 分布类别 (lognormal, normal, uniform, exponential)

        Returns:
            Dict[str, Dict[str, Any]]: 该类别的所有分布配置
        """
        filtered_configs = {}
        for name, config in DistributionConfig.DISTRIBUTIONS.items():
            if config.get('category') == category:
                filtered_configs[name] = config.copy()

        return filtered_configs

    @staticmethod
    def get_distribution_names() -> List[str]:
        """
        获取所有分布配置名称列表

        Returns:
            List[str]: 分布配置名称列表
        """
        return list(DistributionConfig.DISTRIBUTIONS.keys())

    @staticmethod
    def get_categories() -> List[str]:
        """
        获取所有分布类别

        Returns:
            List[str]: 分布类别列表
        """
        categories = set()
        for config in DistributionConfig.DISTRIBUTIONS.values():
            categories.add(config.get('category', 'unknown'))
        return sorted(list(categories))

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """
        验证分布配置的有效性

        Args:
            config (Dict[str, Any]): 分布配置字典

        Returns:
            bool: 配置是否有效
        """
        required_fields = ['type', 'params', 'description']
        for field in required_fields:
            if field not in config:
                return False

        # 验证分布类型
        valid_types = ['lognormal', 'normal', 'uniform', 'exponential']
        if config['type'] not in valid_types:
            return False

        # 验证参数
        if not isinstance(config['params'], dict):
            return False

        return True

    @staticmethod
    def export_to_json(filepath: str) -> None:
        """
        导出配置到JSON文件

        Args:
            filepath (str): JSON文件路径
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(DistributionConfig.DISTRIBUTIONS, f,
                         indent=4, ensure_ascii=False)
            print(f"配置已导出到: {filepath}")
        except Exception as e:
            print(f"导出配置失败: {e}")
            raise

    @staticmethod
    def print_all_configs() -> None:
        """
        打印所有分布配置信息（用于调试和展示）
        """
        print("=" * 80)
        print("可用的不确定性事件分布配置")
        print("=" * 80)

        for category in DistributionConfig.get_categories():
            configs = DistributionConfig.get_configs_by_category(category)
            print(f"\n【{category.upper()}分布系列】")
            print("-" * 50)

            for name, config in configs.items():
                print(f"配置名称: {name}")
                print(f"描述: {config['description']}")
                print(f"用途: {config['use_case']}")
                print(f"参数: {config['params']}")
                print("-" * 30)


# 示例使用和测试代码
if __name__ == "__main__":
    # 测试配置功能
    print("=== 分布配置中心测试 ===")

    # 显示所有配置
    DistributionConfig.print_all_configs()

    # 测试获取单个配置
    try:
        config = DistributionConfig.get_config('normal_mean80_std20')
        print(f"\n获取单个配置测试: {config}")
    except ValueError as e:
        print(f"配置获取错误: {e}")

    # 测试按类别获取
    normal_configs = DistributionConfig.get_configs_by_category('normal')
    print(f"\n正态分布配置: {list(normal_configs.keys())}")

    # 导出配置到JSON
    output_dir = "A:/MYpython/34959_RL/codes"
    os.makedirs(output_dir, exist_ok=True)
    DistributionConfig.export_to_json(f"{output_dir}/distribution_configs.json")
"""
ALNS-RL 实验配置（Phase 1）

目标：
1) 把分布/场景/训练-测试窗口/早停参数集中管理（配置驱动）
2) 为生成器与后续日志透传提供统一的 Ground Truth 规格

约束（来自 Master Protocol）：
- 分布类型：对数正态分布（Log-Normal）
- 固定 sigma = 0.5
- 训练窗口：ID 0 → 499（强制）
- 测试窗口：ID 999 → 800（倒序）
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Tuple

# =========================
# 全局控制开关
# =========================

AGENT_MODE: Literal["RL", "ALWAYS_WAIT", "ALWAYS_REROUTE", "RANDOM"] = "RL"

# =========================
# 数据集切分（固定 1000）
# =========================

TOTAL_FILES: int = 1000

TRAIN_START: int = 0
TRAIN_END: int = 5  # inclusive

TEST_START: int = 999
TEST_END: int = 996  # inclusive, 倒序遍历

# =========================
# 分布基础定义
# =========================

LOGNORMAL_SIGMA: float = 0.5


def lognormal_mu_from_target_mean(target_mean: float, sigma: float = LOGNORMAL_SIGMA) -> float:
    """
    给定对数正态分布的目标均值 E[X]=m 与 sigma，计算 mu：
      m = exp(mu + sigma^2/2)  =>  mu = ln(m) - sigma^2/2
    """
    if target_mean <= 0:
        raise ValueError("target_mean 必须为正数")
    return math.log(target_mean) - (sigma * sigma) / 2.0


# =========================
# 早停参数（Phase-Aware Early Stopping 会在后续实现）
# =========================

EARLY_STOP = {
    "TARGET_ACC": 0,
    "MIN_TABLES": 3,
    "MAX_STD": 1,
    "PATIENCE": 5,

}


# =========================
# 分布配置数据结构
# =========================


@dataclass(frozen=True)
class ComponentSpec:
    label: str  # "A" / "B" / "C"
    target_mean: float
    sigma: float = LOGNORMAL_SIGMA

    @property
    def mu(self) -> float:
        return lognormal_mu_from_target_mean(self.target_mean, self.sigma)


@dataclass(frozen=True)
class DistributionSpec:
    """
    一个 dist_key 对应一个“生成剧本”：
    - components: 组件集合（A/B/C）
    - segments:  按比例的顺序段（如 [("A",0.25),("B",0.25),("A",0.50)]）
                若 shuffle=True，则先按 segments 生成，再全局打乱（用于 S1）
    """

    key: str
    scenario: str  # "S1"..."S6"
    description: str
    components: Dict[str, ComponentSpec]
    segments: List[Tuple[str, float]]
    shuffle: bool = False


def _normalize_segments(segments: Iterable[Tuple[str, float]]) -> List[Tuple[str, float]]:
    seg_list = list(segments)
    total = sum(r for _, r in seg_list)
    if total <= 0:
        raise ValueError("segments 的比例之和必须为正数")
    return [(label, r / total) for label, r in seg_list]


def build_component_plan(
    spec: DistributionSpec,
    total_files: int = TOTAL_FILES,
    seed: Optional[int] = None,
) -> List[Dict[str, object]]:
    """
    生成长度为 total_files 的 Ground Truth 映射表（每个 sample_id 一行）。
    返回列表中的第 i 个元素对应 table_number=i。

    注意：
    - segmented：按段顺序排列
    - shuffle：按段生成后做全局随机打乱（S1 随机混合基准）
    """
    segs = _normalize_segments(spec.segments)

    labels: List[str] = []
    remaining = total_files
    for idx, (label, ratio) in enumerate(segs):
        if idx == len(segs) - 1:
            count = remaining
        else:
            count = int(round(total_files * ratio))
            count = max(0, min(count, remaining))
        labels.extend([label] * count)
        remaining -= count
        if remaining <= 0:
            break

    if len(labels) < total_files:
        labels.extend([segs[-1][0]] * (total_files - len(labels)))
    if len(labels) > total_files:
        labels = labels[:total_files]

    if spec.shuffle:
        rng = random.Random(seed)
        rng.shuffle(labels)

    plan: List[Dict[str, object]] = []
    for sample_id, label in enumerate(labels):
        comp = spec.components[label]
        plan.append(
            {
                "sample_id": sample_id,
                "dist_key": spec.key,
                "scenario": spec.scenario,
                "dist_phase": label,
                "gt_mean": float(comp.target_mean),
                "gt_mu": float(comp.mu),
                "gt_sigma": float(comp.sigma),
            }
        )
    return plan


def early_stop_unlock_table_id(
    spec: DistributionSpec,
    train_end: int = TRAIN_END,
    total_files: int = TOTAL_FILES,
) -> int:
    """
    Phase-Aware Early Stopping 的“锁”：
    - 若为 segmented：必须进入训练窗口最后所在阶段（例如 S5 的后段 B），才能允许早停检查。
    - 若为 shuffle（S1）：从一开始就处于混合环境，锁直接打开（返回 0）。
    """
    if spec.shuffle:
        return 0

    segs = _normalize_segments(spec.segments)
    # 计算每段的 index 范围
    start = 0
    boundaries: List[Tuple[str, int, int]] = []
    remaining = total_files
    for idx, (label, ratio) in enumerate(segs):
        if idx == len(segs) - 1:
            count = remaining
        else:
            count = int(round(total_files * ratio))
            count = max(0, min(count, remaining))
        end = start + count
        boundaries.append((label, start, end))  # [start,end)
        start = end
        remaining -= count
        if remaining <= 0:
            break

    # 查 train_end 落在哪一段
    for _label, seg_start, seg_end in boundaries:
        if seg_start <= train_end < seg_end:
            return seg_start
    return 0


# =========================
# 30 种分布配置（S1~S6）
# =========================


def _binary_components(mean_a: float, mean_b: float) -> Dict[str, ComponentSpec]:
    return {
        "A": ComponentSpec(label="A", target_mean=mean_a),
        "B": ComponentSpec(label="B", target_mean=mean_b),
    }


def _triple_components(mean_a: float, mean_b: float, mean_c: float) -> Dict[str, ComponentSpec]:
    return {
        "A": ComponentSpec(label="A", target_mean=mean_a),
        "B": ComponentSpec(label="B", target_mean=mean_b),
        "C": ComponentSpec(label="C", target_mean=mean_c),
    }


def _make_s1(key: str, mean_a: float, mean_b: float, description: str) -> DistributionSpec:
    # S1：随机混合 50/50，全局 shuffle
    return DistributionSpec(
        key=key,
        scenario="S1",
        description=description,
        components=_binary_components(mean_a, mean_b),
        segments=[("A", 0.5), ("B", 0.5)],
        shuffle=True,
    )


def _make_s2(key: str, mean_a: float, mean_b: float, description: str) -> DistributionSpec:
    # S2：ABA，25% A → 25% B → 50% A
    return DistributionSpec(
        key=key,
        scenario="S2",
        description=description,
        components=_binary_components(mean_a, mean_b),
        segments=[("A", 0.25), ("B", 0.25), ("A", 0.50)],
        shuffle=False,
    )


def _make_s3(key: str, mean_a: float, mean_b: float, description: str) -> DistributionSpec:
    # S3：隔离 50/50，50% A → 50% B
    return DistributionSpec(
        key=key,
        scenario="S3",
        description=description,
        components=_binary_components(mean_a, mean_b),
        segments=[("A", 0.50), ("B", 0.50)],
        shuffle=False,
    )


def _make_s4(key: str, mean_a: float, mean_b: float, description: str) -> DistributionSpec:
    # S4：召回 85/15，85% A → 15% B
    return DistributionSpec(
        key=key,
        scenario="S4",
        description=description,
        components=_binary_components(mean_a, mean_b),
        segments=[("A", 0.85), ("B", 0.15)],
        shuffle=False,
    )


def _make_s5(key: str, mean_a: float, mean_b: float, description: str) -> DistributionSpec:
    # S5：适应 15/85，15% A → 85% B
    return DistributionSpec(
        key=key,
        scenario="S5",
        description=description,
        components=_binary_components(mean_a, mean_b),
        segments=[("A", 0.15), ("B", 0.85)],
        shuffle=False,
    )


def _make_s6(key: str, mean_a: float, mean_b: float, mean_c: float, description: str) -> DistributionSpec:
    # S6：三重分布 20/65/15，20% A → 65% B → 15% C
    return DistributionSpec(
        key=key,
        scenario="S6",
        description=description,
        components=_triple_components(mean_a, mean_b, mean_c),
        segments=[("A", 0.20), ("B", 0.65), ("C", 0.15)],
        shuffle=False,
    )


DISTRIBUTIONS: Dict[str, DistributionSpec] = {
    # 场景一：随机混合 50-50（Benchmark）
    "S1_1": _make_s1("S1_1", 9, 90, "S1_1 强对比：均值 9 ↔ 90（随机 50/50 全局打乱）"),
    "S1_2": _make_s1("S1_2", 3, 30, "S1_2 中对比：均值 3 ↔ 30（随机 50/50 全局打乱）"),
    "S1_3": _make_s1("S1_3", 6, 60, "S1_3 弱对比：均值 6 ↔ 60（随机 50/50 全局打乱）"),

    # 场景二：ABA 回马枪 / 遗忘测试（Memory）
    "S2_1": _make_s2("S2_1", 9, 90, "S2_1 ABA：9 → 90 → 9（低→高→低）"),
    "S2_2": _make_s2("S2_2", 90, 9, "S2_2 ABA：90 → 9 → 90（高→低→高）"),
    "S2_3": _make_s2("S2_3", 3, 30, "S2_3 ABA：3 → 30 → 3"),
    "S2_4": _make_s2("S2_4", 30, 3, "S2_4 ABA：30 → 3 → 30"),
    "S2_5": _make_s2("S2_5", 6, 60, "S2_5 ABA：6 → 60 → 6"),
    "S2_6": _make_s2("S2_6", 60, 6, "S2_6 ABA：60 → 6 → 60"),

    # 场景三：50-50 隔离 / 过拟合测试（Overfitting）
    "S3_1": _make_s3("S3_1", 9, 90, "S3_1 练低考高：9 → 90（预期崩盘）"),
    "S3_2": _make_s3("S3_2", 90, 9, "S3_2 练高考低：90 → 9（预期过敏）"),
    "S3_3": _make_s3("S3_3", 3, 30, "S3_3 练低考高：3 → 30"),
    "S3_4": _make_s3("S3_4", 30, 3, "S3_4 练高考低：30 → 3"),
    "S3_5": _make_s3("S3_5", 6, 60, "S3_5 练低考高：6 → 60"),
    "S3_6": _make_s3("S3_6", 60, 6, "S3_6 练高考低：60 → 6"),

    # 场景四：85-15 召回 / 记忆唤醒（Recall）
    "S4_1": _make_s4("S4_1", 9, 90, "S4_1 召回：85% 9 → 15% 90"),
    "S4_2": _make_s4("S4_2", 90, 9, "S4_2 召回：85% 90 → 15% 9"),
    "S4_3": _make_s4("S4_3", 3, 30, "S4_3 召回：85% 3 → 15% 30"),
    "S4_4": _make_s4("S4_4", 30, 3, "S4_4 召回：85% 30 → 15% 3"),
    "S4_5": _make_s4("S4_5", 6, 60, "S4_5 召回：85% 6 → 15% 60"),
    "S4_6": _make_s4("S4_6", 60, 6, "S4_6 召回：85% 60 → 15% 6"),

    # 场景五：15-85 适应 / 在线纠错（Adaptation）
    "S5_1": _make_s5("S5_1", 9, 90, "S5_1 适应：15% 9 → 85% 90"),
    "S5_2": _make_s5("S5_2", 90, 9, "S5_2 适应：15% 90 → 85% 9"),
    "S5_3": _make_s5("S5_3", 3, 30, "S5_3 适应：15% 3 → 85% 30"),
    "S5_4": _make_s5("S5_4", 30, 3, "S5_4 适应：15% 30 → 85% 3"),
    "S5_5": _make_s5("S5_5", 6, 60, "S5_5 适应：15% 6 → 85% 60"),
    "S5_6": _make_s5("S5_6", 60, 6, "S5_6 适应：15% 60 → 85% 6"),

    # 场景六：三重分布 / 复杂动态（Complexity）
    "S6_1": _make_s6("S6_1", 9, 90, 30, "S6_1 消散型：9 → 90 → 30（Low→High→Mid）"),
    "S6_2": _make_s6("S6_2", 9, 30, 90, "S6_2 阶梯型：9 → 30 → 90（Low→Mid→High）"),
    "S6_3": _make_s6("S6_3", 90, 30, 9, "S6_3 缓释型：90 → 30 → 9（High→Mid→Low）"),
}


def get_distribution_spec(dist_key: str) -> DistributionSpec:
    if dist_key not in DISTRIBUTIONS:
        raise KeyError(f"未知分布配置: {dist_key}")
    return DISTRIBUTIONS[dist_key]


def list_distribution_keys() -> List[str]:
    return sorted(DISTRIBUTIONS.keys())

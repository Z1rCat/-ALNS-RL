# ALNS-RL: 基于强化学习和自适应大邻域搜索的动态多式联运优化系统

## 📖 项目简介

本项目实现了一个创新的**ALNS-RL混合算法**，用于解决动态多式联运物流网络优化问题。系统结合了**自适应大邻域搜索(ALNS)**元启发式算法和**强化学习(RL)**智能体，能够在不确定环境下进行实时的运输路径优化和调度决策。

### 🎯 研究目标
- 处理动态需求变化和不确定事件
- 优化多式联运网络中的路径选择和资源配置
- 通过强化学习实现实时决策和自适应调整
- 评估环境影响和运输成本

## 🏗️ 系统架构

```
ALNS-RL动态优化系统
├── 强化学习层 (dynamic_RL34959.py)
│   ├── DQN智能体：实时决策制定
│   ├── PPO智能体：策略优化
│   └── 环境建模：Gym接口
├── 优化算法层 (Intermodal_ALNS34959.py)
│   ├── ALNS核心：大规模优化求解
│   ├── 邻域操作：破坏与重建算子
│   └── 并行计算：加速求解过程
├── 集成控制层 (Dynamic_ALNS_RL34959.py)
│   ├── 算法协调：ALNS与RL协同
│   ├── 动态事件处理：不确定事件响应
│   └── 解空间管理：最优解维护
├── 不确定性处理 (fuzzy_HP.py)
│   ├── 模糊逻辑：不确定性建模
│   └── 隶属度函数：事件严重程度评估
└── 环境评估 (emission_models.py)
    ├── 碳排放计算：运输环境影响
    └── 可持续性指标：绿色物流评估
```
3. 参数配置
若需修改实验设置（如订单规模 R 或不确定性分布），目前需要直接修改代码文件：
修改 R 值: 打开 codes/Intermodal_ALNS34959.py，修改全局变量 request_number_in_R。
修改分布: 需使用数据生成脚本重新生成 Excel 文件并替换目标文件夹。


📈 实验结果
系统在 Uncertainties... 目录下输出实验日志：
obj_record*.xlsx: 记录每一步的目标函数值、成本、排放等指标。
best_routes*.xlsx: 记录找到的最优路径方案。
## 🔧 核心算法

### ALNS算法特性
- **自适应机制**：动态调整算子选择概率
- **多邻域搜索**：破坏与重建操作组合
- **并行优化**：多进程加速求解
- **解空间管理**：哈希表避免重复计算

### 强化学习集成
- **状态空间**：网络状态、车辆位置、需求信息
- **动作空间**：路径选择、调度决策、资源分配
- **奖励函数**：成本最小化 + 服务质量最大化
- **训练策略**：经验回放 + 目标网络更新

### 动态事件处理
- **需求变化**：新订单插入、订单取消
- **运输延误**：车辆故障、交通拥堵
- **网络中断**：节点失效、弧段不可用
- **容量限制**：节点拥堵、运力约束

## 📋 环境要求

### 系统环境
- **Python**: 3.9.19
- **操作系统**: Windows 10/11
- **GPU**: 支持CUDA的NVIDIA显卡（推荐）

### Conda环境配置

```bash
# 创建新的conda环境
conda create -n alns-rl python=3.9.19
conda activate alns-rl

# 安装PyTorch (CUDA版本)
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html

# 安装其他依赖
pip install pandas numpy matplotlib networkx openpyxl scikit-fuzzy stable-baselines3 gym gymnasium
```

### 完整依赖清单
详见 `codes/environment.yml` 文件，包含所有必需的Python包及版本号。

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/Z1rCat/-ALNS-RL.git
cd -ALNS-RL
```

### 2. 环境配置
```bash
# 使用项目提供的conda环境文件
conda env create -f codes/environment.yml
conda activate pytorch
```

### 3. 数据准备
将数据文件放置在项目根目录：
- `Intermodal_EGS_data_all.xlsx`: 主要数据集
- `Intermodal_EGS_data_all_heterogeneous_6.xlsx`: 异构场景数据
- 其他Excel文件：不同实验场景的数据

### 4. 运行实验

#### 基础ALNS优化
```python
python codes/Intermodal_ALNS34959.py
```

#### ALNS-RL混合优化
```python
python codes/Dynamic_ALNS_RL34959.py
```

#### 强化学习训练
```python
python codes/dynamic_RL34959.py
```

## 📊 实验配置

### 参数设置
```python
# ALNS参数
max_iterations = 1000
reaction_factor = 0.8
destruction_size = [0.1, 0.3]  # 破坏比例范围

# RL参数
learning_rate = 0.001
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995

# 动态参数
time_horizon = 24  # 小时
update_frequency = 1  # 小时
uncertainty_level = 0.3  # 不确定性程度
```

### 场景设置
- **确定性场景**: 已知所有需求和运输时间
- **随机需求场景**: 需求按概率分布生成
- **混合事件场景**: 多种不确定事件组合
- **时间依赖场景**: 事件概率随时间变化

## 📁 项目结构

```
34959_RL/
├── codes/                          # 核心代码目录
│   ├── dynamic_RL34959.py          # 强化学习主模块
│   ├── Intermodal_ALNS34959.py     # ALNS算法核心
│   ├── Dynamic_ALNS_RL34959.py     # ALNS-RL集成模块
│   ├── fuzzy_HP.py                 # 模糊逻辑处理
│   ├── emission_models.py          # 环境影响模型
│   └── environment.yml             # Conda环境配置
├── Uncertainties Dynamic planning under unexpected events/  # 实验结果
│   ├── Figures/                    # 图表结果
│   ├── Instances/                  # 实例数据
│   └── plot_distribution_*/       # 分布分析结果
├── analysis_results/               # 分析结果
├── *.xlsx                          # 数据文件
└── README.md                       # 项目说明文档
```

## 📈 实验结果分析

### 性能指标
- **总成本**: 运输成本 + 转运成本 + 延误惩罚
- **服务水平**: 需求满足率 + 准时交付率
- **计算效率**: 求解时间 + 收敛速度
- **环境影响**: 碳排放量 + 能源消耗

### 结果文件说明
- `best_routes*.xlsx`: 最优路径方案
- `obj_record*.xlsx`: 目标函数值变化记录
- `routes_match*.xlsx`: 路径匹配分析
- `exps_record*.xlsx`: 实验统计记录

## 🎯 主要创新点

1. **算法创新**: 首次将ALNS与RL结合用于动态多式联运优化
2. **实时决策**: RL智能体实现毫秒级响应动态变化
3. **不确定性建模**: 模糊逻辑处理复杂不确定事件
4. **并行计算**: 多进程加速大规模问题求解
5. **环境友好**: 集成碳排放评估支持绿色物流

## 📚 相关文献

- Adaptive Large Neighborhood Search for Dynamic Vehicle Routing Problems
- Deep Reinforcement Learning for Real-time Transportation Optimization
- Fuzzy Logic Applications in Uncertain Supply Chain Management
- Sustainable Intermodal Freight Network Design under Uncertainty

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进本项目：

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- GitHub Issues: [项目Issues页面]
- 邮箱: [您的邮箱]

## 🙏 致谢

感谢所有为本项目做出贡献的研究人员和开发者。

---

**注意**: 这是一个研究项目，主要用于学术研究目的。如需商业应用，请确保遵守相关许可证要求。
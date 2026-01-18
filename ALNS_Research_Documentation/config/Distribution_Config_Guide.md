# 分布配置说明（distribution_config.json）

本文档说明如何使用 `distribution_config.json` 管理分布定义，并与生成器/主控面板联动。

## 1. 文件位置
- 配置文件：`A:/MYpython/34959_RL/distribution_config.json`
- 生成器读取：`codes/generate_mixed_parallel.py`
- Master 菜单读取：`codes/Dynamic_master34959.py`

只要修改 config，master 菜单会自动刷新分布列表，无需再手改代码。

## 2. 配置结构
```json
{
  "version": 1,
  "distributions": [
    {
      "name": "S1_1",
      "pattern": "random_mix",
      "means": {"A": 9, "B": 90},
      "display": "S1_1 random mix A=9 B=90"
    }
  ]
}
```

字段说明：
- `name`：分布名称，需唯一。
- `pattern`：分布模式（见第 3 节）。
- `means`：阶段参数，支持 **仅均值** 或 **均值 + 方差/标准差**。
- `display`：控制台显示文本（可省略，系统会自动生成）。

## 3. 支持的 pattern
当前支持以下模式：
- `random_mix`：全区间随机 A/B 50/50
- `aba`：A(0-174) → B(175-349) → A(350-499)
- `ab`：A(0-349) → B(350-499)
- `recall`：A(0-424) → B(425-499)
- `adaptation`：A(0-99) → B(100-499)
- `abc`：A(0-174) → B(175-349) → C(350-499)

## 4. 均值/方差扩展写法
你可以写：

### 方式 1：仅均值（默认 std=mean*0.25）
```json
"means": {"A": 9, "B": 90}
```

### 方式 2：均值 + 方差
```json
"means": {
  "A": {"mean": 3, "var": 3},
  "B": {"mean": 3, "var": 5}
}
```

### 方式 3：均值 + 标准差
```json
"means": {
  "A": {"mean": 3, "std": 1.7},
  "B": {"mean": 3, "std": 2.2}
}
```

### 方式 4：指定分布类型（可选）
```json
"means": {
  "A": {"mean": 9, "std": 3, "dist": "normal"},
  "B": {"mean": 90, "std": 10, "dist": "normal"}
}
```
目前 `dist` 支持：`normal`、`lognormal`。

## 5. 新增分布示例
```json
{
  "name": "S7_1",
  "pattern": "ab",
  "means": {
    "A": {"mean": 3, "var": 3},
    "B": {"mean": 3, "var": 5}
  },
  "display": "S7_1 custom A(mean=3,var=3) B(mean=3,var=5)"
}
```

新增后，运行 `Dynamic_master34959.py` 时会自动出现该条目，无需改动菜单代码。

## 6. 注意事项
- `pattern` 必须是支持的模式之一，否则生成器会报错并退出。
- `means` 必须包含该 pattern 所需的阶段：  
  - `random_mix/ab/aba/recall/adaptation` 需要 A 和 B  
  - `abc` 需要 A、B、C
- 生成数据总量固定为 500（0–499），训练/测试切割逻辑由调度器控制。

# 服务器部署待确认信息清单（用于后续 Dockerfiles / 实验脚本落地）

目的：在不反复试错的前提下，确认服务器环境约束，决定 **Docker / CUDA / 存储路径 / 并发策略**，并最终完成可复现的一键实验部署。

---

## 1) 基础环境

- 服务器系统：Linux 发行版 + 版本（例如 Ubuntu 22.04 / CentOS 7 / Rocky 9）
- 内核版本：`uname -a`
- 是否有外网访问（用于 `git clone` / `pip/conda` 拉包）：
  - 能否访问 GitHub
  - 是否需要代理（HTTP(S) proxy）或内网镜像源
- 是否有作业调度系统：Slurm / PBS / 无（直接跑）
  - 若有：单作业可用的 CPU 核数 / 内存 / 时长限制

---

## 2) 硬件资源

- CPU：
  - 物理核数（physical cores）
  - 逻辑线程数（logical cores）
  - 型号（可选）
- 内存（RAM）：总量 + 单作业限制（若有）
- GPU（如有）：
  - 型号、数量、显存（VRAM）
  - CUDA 驱动版本（`nvidia-smi`）
  - 是否允许使用 GPU（有些服务器虽有 GPU 但默认不开放）

说明：当前项目训练负载以 **CPU（ALNS）** 为主，GPU 只在 SB3 模型更新上可能收益；如果 GPU 不可用也能跑，只是训练更慢。

---

## 3) Docker/容器能力（决定我们用 Dockerfile 还是替代方案）

- 服务器是否已安装 Docker？
  - `docker --version`
  - 当前用户是否在 `docker` 组（能否无 sudo 运行）
- 是否允许 rootless docker？
- 若 GPU 需要：
  - 是否支持 `nvidia-container-toolkit`（NVIDIA Docker runtime）
  - 容器内是否能用 `nvidia-smi`
- 若不允许 Docker：
  - 是否支持 Apptainer/Singularity（HPC 常用）

---

## 4) 存储与路径（最关键：磁盘是否扛得住 + 路径权限）

项目运行会在每个 `run_dir` 里生成大体量文件（`data/` 500 个 Excel + `alns_outputs/`），我们已实现 **run 完整结束后自动清理**，但仍需要确认：

- 推荐的工作目录（建议放在高速盘/本地盘而非 NFS）：
  - 例如：`/scratch/<user>/34959_RL` 或 `/data/<user>/34959_RL`
- 单用户/单作业磁盘配额（quota）
- 文件系统类型：
  - NFS/共享盘（并发写 Excel 容易遇到锁/性能问题）
  - 本地 SSD（更适合）
- 是否允许大量小文件写入（Excel + 日志）
- 运行目录是否支持原子重命名（`os.replace`）：
  - Windows 上我们已遇到过锁冲突；Linux 一般更稳定，但 NFS 仍可能异常

---

## 5) 并发策略约束（决定 `--max-workers`、一次跑多少个 run）

我们当前的并发是：**一个 run 作为一个任务**（串行执行 master → baseline → plot → metrics → cleanup），多个 run 之间并发。

待确认：

- 服务器允许同时跑多少个 Python 进程（或 Slurm 单作业允许多少核）
- 是否限制线程/进程数（ulimit）
- 是否限制单目录并发 IO（特别是 NFS）
- 推荐的并发策略：
  - 多 run 并发（推荐）
  - 单 run 内多线程（我们尽量避免在生成 Excel 时开多核，减少写冲突）

---

## 6) 依赖安装方案（决定 Dockerfile 基于 conda 还是 pip）

当前本地使用 `codes/environment.yml`（conda env）：

- 服务器上是否允许安装 Miniconda/Anaconda？
- 是否必须使用系统 Python + venv？
- 是否需要指定 pip/conda 镜像源（例如清华源/中科大源）

---

## 7) 运行/产物管理（实验可复现与成果收集）

- 产物保留策略：
  - 我们会保留：`meta.json`、`rl_trace.csv`、`rl_training.csv`、`baseline_*.csv`、`metrics.json`、`paper_figures/`
  - 会删除：`data/`、`alns_outputs/`
- 需要确认：清理后是否还需要额外保留某些 ALNS 中间文件（例如为了审计/复盘）
- 结果汇总文件写入位置（全局）：
  - 默认：`codes/logs/summary/metrics_summary.csv`（并发写入已加锁）

---

## 8) 你可以直接问学长的“最短问题列表”

1. 服务器 OS/版本 + 是否有 Slurm（单作业资源上限是多少）？
2. 物理核数/内存总量（以及单作业限制）？
3. 是否有 GPU？能否在容器/非容器环境使用？CUDA/驱动版本？
4. 是否允许 Docker？若不允许，是否支持 Apptainer/Singularity？
5. 推荐的高速写盘路径是什么？磁盘配额多少？（NFS 还是本地 SSD）
6. 是否允许访问 GitHub 和 pip/conda 外网源？是否需要代理/镜像？


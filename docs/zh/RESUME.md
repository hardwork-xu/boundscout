# 简历素材

[English](../en/RESUME.md) · [项目讲解](PROJECT_TALK.md) · [实验结果](RESULTS.md)

以下是有项目证据支撑的表述草稿。请根据自己实际参与、能够解释的工作调整措辞；拥有这份仓库并不能证明个人已经掌握全部内容、具备工作经历或完成生产部署。

## 四条对应表述

- 实现可检查的 CPU 向量检索原型，采用空间中位数分块、保守包围盒剪枝、float64 平方 L2 距离和原始 ID 确定性并列排序，并提供 Python API 与离线 CLI。
- 实现不可变 NPY 索引持久化、只读 mmap 加载、SHA256 与几何验证、显式内存预算和生命周期管理；通过 **111 项离线测试**验证引擎、存储、并发读取、CLI 与基准行为。
- 使用六类工作负载、六种方法、两个输入种子、每种方法七次正式重复测量，完成 **72 个工作负载／种子／方法组合及 504 个计时查询批次**，保留原始样本、消融、源码摘要及独立进程峰值 RSS。
- 在预先指定的 **32,768 × 32 聚类工作负载、64 个查询、k=10** 上，相比更强的 `cdist-4096` 穷举基线获得 **3.50–3.62 倍**加速，距离计算向量数减少 **92.72–92.74%**；同时披露 `cKDTree` 仍快 **2.01–2.71 倍**，归一化 GloVe25 上的速度仅为 `cdist-4096` 的 **0.30 倍**。

## 必须保留的测量范围

主要工作负载相比 `cdist-256` 还获得了 8.18 倍加速，但只引用这一数字会隐藏更强的穷举基线。上述范围是两个种子各自七次重复的批次中位数之比，并非置信区间。实验使用 Apple M1 Pro、16 GiB 内存、单原生线程，运行环境为有其他交互负载的共享主机。稳态查询耗时不含数据准备和索引构建。这是有明确边界的 CPU 实验，不代表普适加速、p99 表现、新的最近邻算法或生产结果。

## 证据索引

| 表述 | 可检查的证据 |
| --- | --- |
| 空间分块、保守剪枝和并列语义 | [`engine.py`](../../src/boundscout/engine.py)：`BoundIndex.build`、`BoundIndex.search`、`_topk`；[数值契约](ARCHITECTURE.md#数值契约) |
| 索引持久化与文件工作流 | [`storage.py`](../../src/boundscout/storage.py)：`save_index`、`load_index`；[`cli.py`](../../src/boundscout/cli.py)：`main` |
| 测试覆盖和可执行检查 | [`test_engine.py`](../../tests/test_engine.py)、[`test_storage.py`](../../tests/test_storage.py)、[`test_cli.py`](../../tests/test_cli.py)、[`test_benchmark.py`](../../tests/test_benchmark.py)；在已安装的开发环境运行 `python -m pytest -q` |
| 冻结的工作负载矩阵与计时过程 | [`configs/full.json`](../../configs/full.json)、[实验协议](PROTOCOL.md)、[`benchmark.py`](../../benchmark.py)：`cdist_search`、`run_workload` |
| 所有数值结论 | [`results/raw/full.json`](../../results/raw/full.json)：按 `workload_id` 和 `seed` 选择 `workloads`，读取 `methods[method].median_seconds`、`evaluated_vectors`、`correctness` 和 `samples_seconds`；[`summary.json`](../../results/summary.json) 包含推导比例 |
| 硬件、环境与被测源码 | [`environment.json`](../../results/environment.json)；原始 JSON 中的 `machine`、`config`、`source.files_sha256` 和 `source.sha256` |

证据运行 ID：`99f9c3f6-cfb9-4e8b-a393-2d38033b24b5`，完成时间 `2026-09-21T12:17:37.935739+00:00`，状态 `ok`。源码指纹为 `fafc0b607164421d889499390d160ca719fa43502f3355a881949bb4d6108097`。该运行记录了 `git_dirty: true`；逐文件摘要标识实际被测内容，仅凭记录中的 Git 提交号不足以确定源码版本。

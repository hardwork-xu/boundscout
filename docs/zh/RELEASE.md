# 发布准备

[English](../en/RELEASE.md)

我将此版本维护为可审阅的 CPU 研究项目。0.1.0 是独立复现与后续研究的起点，不是生产服务，也不是新的最近邻算法。

## 0.1.0 发布说明草稿

- 确定性 float64 平方欧氏 top-k，采用空间中位数分块与保守距离下界剪枝。
- 剪枝与布局消融、独立编译穷举基线，以及 SciPy cKDTree 对照。
- 只读 NPY 索引持久化，验证格式、哈希、ID 与几何；显式关闭与资源限制。
- 离线双语 CLI、回归测试、带哈希的 Python 3.12 依赖锁、CI 配置与原始数据报告工具。
- 真实 GloVe25 和合成负载；完整测量、失败与适用范围见[实验记录](EXPERIMENTS.md)。

已知限制：不可变索引、在内存中构建；高维剪枝可能更慢；Python 块遍历有额外开销；不涉及 GPU、近似检索、流式插入或服务吞吐声明。逻辑内存预算不等于硬性进程 RSS 限额。提供 Docker 配置，但本地缺少 Docker，未执行容器验证。远端 CI 与公开发布状态应以生成后的 `results/publication.json` 为准，不能从工作流配置推断成功。

## 构建与演示

在具备 Python 3.12 的检出目录中执行：

```bash
python3 -m venv .venv
make install
make demo
make check
```

构建产物为 `dist/boundscout-0.1.0-py3-none-any.whl` 与 `dist/boundscout-0.1.0.tar.gz`。`python -m build --no-isolation` 使用 `make install` 安装的固定构建后端。

```bash
docker build -t boundscout .
docker run --rm boundscout
```

这些 Docker 命令已在 `d7cfd6e` 的 Ubuntu CI 运行 `35601617064` 中通过；本机仍无 Docker。参见[发布证据](../../results/publication.json)。镜像固定 Python 3.12.10，但基础镜像标签未绑定摘要，未来重建的操作系统层可能变化。本地证据使用 Python 3.12.2，不能等同于容器证据。

## 公开元数据

一句话 / GitHub About：**可审阅的 CPU 精确向量检索，提供块剪枝、确定性等距排序与可复现实验。** 英文 About 使用对应英文文案。

Topics：`nearest-neighbor-search`、`vector-search`、`cpu`、`numpy`、`scipy`、`benchmark`、`reproducible-research`、`python`。

许可证：[MIT](../../LICENSE)。依赖与 GloVe 归属见 [NOTICE](../../NOTICE_zh.md)。`CITATION.cff` 使用合法 YAML 子集 JSON，只列出已核验的公开维护者账号。引用时保留版本、源码提交和实验 run ID；本项目没有 DOI 或对应论文。

发布范围为 GitHub 仓库及其发布附件。不包含 PyPI 上传、托管服务、付费资源、外部宣传或私人数据集发布。发布附件在本地构建，发布记录区分实际执行的外部操作与准备好的命令。

## 实际发布状态

公开仓库与 GitHub `v0.1.0` 已发布。附件由干净 Git 归档 `d7cfd6e` 构建，GitHub 附件哈希及匿名下载的 wheel 均与本地文件一致。随后加入的发布记录只保存这些实际观察，不修改被测实现。未执行 PyPI 上传或公网服务部署。

# 完整本地使用导览

[English](../en/WALKTHROUGH.md) · [架构](ARCHITECTURE.md) · [实验协议](PROTOCOL.md)

以下命令在仓库根目录运行，使用 Python 3.12。首次安装依赖需要访问包源；之后的合成演示、建索引、查询和正确性测试均在本地运行，无需模型、服务或下载语料。

## 安装并检查演示

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements.lock
python -m pip install --no-deps --no-build-isolation -e .
python -m boundscout demo
```

演示生成固定种子的聚类向量，构建索引，并与穷举参照比较。JSON 输出包含 `ok`、结果形状、最大绝对距离误差、工作量计数和索引逻辑字节数。`ok: true` 是该演示的成功条件，不是基准加速比，也不承诺其他输入的表现。正确性失败退出码为 1，普通 CLI 错误退出码为 2。

## 准备小型合成数据

```sh
python - <<'PY'
from pathlib import Path

import numpy as np

folder = Path("work/tutorial")
folder.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20260921)
centers = rng.normal(size=(8, 16)) * 8.0
points = centers[rng.integers(8, size=4096)] + rng.normal(size=(4096, 16)) * 0.5
queries = centers[:4] + rng.normal(size=(4, 16)) * 0.5
np.save(folder / "points.npy", points, allow_pickle=False)
np.save(folder / "queries.npy", queries, allow_pickle=False)
PY
```

每行是一个向量，每列是一个坐标。数据库形状为 `(N, D)`，查询形状为 `(Q, D)`；单个查询也必须保留 `(1, D)` 二维形状。返回 ID 是原始数据库从零开始的行号。这个示例是合成数据，并且有意设置了有利的空间结构。

## 通过 CLI 构建、保存与查询

```sh
python -m boundscout build --input work/tutorial/points.npy --output work/tutorial/index --leaf-size 256
python -m boundscout query --index work/tutorial/index --input work/tutorial/queries.npy --output work/tutorial/results.npz --k 10
```

两个输出路径都必须尚不存在。重复运行时选用新输出名称；CLI 拒绝覆盖索引或查询结果。构建命令输出形状和逻辑数组字节数。查询命令验证保存的索引，写入包含 `indices` 与 `distances` 的 NPZ，二者形状均为 `(Q, k)`。距离是 **平方 L2**，不是欧氏长度。工作量计数以 JSON 打印到标准输出。

```sh
python - <<'PY'
import numpy as np

with np.load("work/tutorial/results.npz", allow_pickle=False) as result:
    print("IDs:", result["indices"][0])
    print("Squared L2:", result["distances"][0])
PY
```

使用 `build --layout input` 按连续输入行分块，或使用 `query --no-prune` 访问所有已存储块。这些是消融控制项，保留相同的 top-k 任务。完整 CLI 参数见 `build --help` 与 `query --help`。

## 在 Python 中比较，并管理索引生命周期

```python
from dataclasses import asdict

import numpy as np

from boundscout import BoundIndex, exhaustive_search

points = np.load("work/tutorial/points.npy", allow_pickle=False)
queries = np.load("work/tutorial/queries.npy", allow_pickle=False)

with BoundIndex.build(points, leaf_size=256) as index:
    actual = index.search(queries, k=10)
    expected = exhaustive_search(points, queries, k=10, block_size=256)
    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_allclose(actual.distances, expected.distances, rtol=1e-12, atol=1e-12)
    print(asdict(actual.stats))
    index.save("work/tutorial/python-index")

with BoundIndex.load("work/tutorial/python-index") as loaded:
    again = loaded.search(queries, k=10)
    np.testing.assert_array_equal(again.indices, actual.indices)
```

这里使用新的 `python-index` 路径，与 CLI 索引区分。上下文管理器在退出时关闭索引。搜索结果拥有独立数组，关闭索引后仍可使用；私有的索引映射数组则不可以。不要在其他操作正在搜索时关闭索引。默认加载检查哈希和几何，使用 mmap 也不意味着常数时间加载。

`evaluated_vectors` 汇总所有查询实际执行的向量距离计算次数；`visited_blocks` 统计计算过的块；`total_blocks` 是可用块数量乘以 `query_count`。对于非空查询批次，相对穷举的向量计算减少比例为 `1 - evaluated_vectors / (N * Q)`。这个比例衡量省下的工作量，不是加速比。

## 复现实验

```sh
python -m pytest
python benchmark.py --config configs/smoke.json --output results/raw/tutorial-smoke.json
```

可选真实数据负载与完整冻结实验：

```sh
python scripts/download_data.py --output data/glove-25-angular.hdf5
python benchmark.py --config configs/full.json --output results/raw/tutorial-reproduction.json --measure-rss
```

下载大小约 121 MB。获取脚本在文件旁记录来源与 SHA256。来源和条款见[研究依据](RESEARCH.md#真实数据来源)及[第三方声明](../../NOTICE_zh.md)。完整配置包含合成对照与可选的归一化 GloVe 子集；获取失败必须如实保留，不能换成合成数据后仍标为真实数据。

阅读[实验结果](RESULTS.md)时，应同时查看[冻结协议](PROTOCOL.md)。比较自己的运行时，预热、重复批次样本、原生线程设置和源码指纹都很重要。smoke 原始结果用于功能检查，不能当作完整评估。

## 常见输入错误

| 现象 | 检查方式 |
| --- | --- |
| 维度错误 | 两个数组都必须为二维，D 相同，且 `1 <= D <= 4096`。 |
| k 无效 | 使用满足 `1 <= k <= N` 的整数；布尔值不属于可接受的整数参数。 |
| 非有限值或数值过大 | 排除 NaN/Inf，坐标绝对值不超过 `1e100`，并检查生成这些数值的预处理。 |
| 索引或输出预算错误 | 默认索引数组预算为 1 GiB，搜索输出预算为 64 MiB。输出过大时按较小查询批次处理。 |
| 输出已存在 | 选择新路径。替代索引通过显式新构建完成，不原地更新。 |
| 校验和或几何错误 | 停止使用该索引，从原始可信向量重新构建。 |

项目不生成嵌入、不分配语义标签，也不将检索 ID 转为应用记录。应用应自行保留原始行号到记录的映射，并确保它与该索引版本使用的向量一致。

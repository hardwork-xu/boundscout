# 研究动机与依据

[English](../en/RESEARCH.md) · [实验协议](PROTOCOL.md) · [实验结果](RESULTS.md)

## 我为什么做这个项目

我想弄清楚：一个精确 CPU 向量索引，什么时候值得引入额外复杂度？基准测试可能只展示距离计算次数下降，却没有说明包围盒计算、排序、内存复制和候选维护的成本。BoundScout 把这些成本放到可检查的位置，并保留一个穷举实现作为参照。

预期使用者是探索离线特征检索的维护者：向量集合固定、查询重复发生，并且需要保留精确搜索语义。项目贡献是小型实现、明确的数值契约，以及有利和不利场景的可复现实证；不宣称提出新的最近邻算法，也不宣称已经是生产搜索服务。

## 选题过程

在仅使用 CPU、以 16 GiB 内存作为规划上限的条件下，项目比较了三个方向。以下是可行性判断，不是横向性能实测。

| 方向 | 有价值的问题 | 决策 |
| --- | --- | --- |
| 精确向量剪枝 | 空间分块节省的工作，何时足以抵消建索引与查询开销？ | 选用：无需模型权重；编译实现的穷举搜索可提供实用的正确性和性能基线。 |
| KV-cache 管理 | 碎片和共享前缀如何影响服务容量？ | 暂缓：有说服力的证据需要推理运行时、请求轨迹和 prefill/decode 测量。[PagedAttention](https://arxiv.org/abs/2309.06180) 是已有系统研究的参考。 |
| 模型量化 | 通过质量评估后，还能保留多少内存和延迟收益？ | 暂缓：校准、模型权重和下游评估会扩大工作范围。[GPTQ](https://arxiv.org/abs/2210.17323) 展示了这类研究的评估规模。 |

冻结的[实验协议](PROTOCOL.md) 在计时前记录了初始目标。未达到目标也是结果；实测结论放在[实验结果](RESULTS.md)，不写成这里的先验事实。

## 已有基础与项目贡献

包围矩形、平衡空间划分和按距离排序的分支限界搜索，早于本项目。[ANN 手册第 2.4.1–2.4.2 节](https://www.cs.umd.edu/~mount/ANN/Files/1.1.1/ANNmanual_1.1.1.pdf) 已介绍精确搜索与优先遍历。[SciPy cKDTree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html) 已支持紧致包围盒、中位数平衡划分和叶节点穷举。

BoundScout 有意把叶节点展开为数据块：先计算所有块的下界，再排序，通过 SciPy 编译实现计算保留下来的块。这让成本容易记录，但每次查询都必须先支付下界计算与排序开销。持久化不可变索引、确定性并列规则、工作量计数、消融与来源记录属于工程设计，不构成算法原创性声明。

## 实数运算中剪枝为何成立

设一个块的所有点都落在下角点为 $l$、上角点为 $u$ 的轴对齐包围盒内，定义

$$
\delta_j(q)=\max\{l_j-q_j,\;0,\;q_j-u_j\},\qquad
L_B(q)=\sum_{j=1}^{D}\delta_j(q)^2.
$$

对块中任意点 $x$，都有 $\delta_j(q)\le |x_j-q_j|$，因此 $L_B(q)\le\|x-q\|_2^2$。当候选集合已有 $k$ 个点，第 k 个点的平方距离为 $\tau$ 时，满足 $L_B(q)>\tau$ 的块不可能改善结果。按下界递增访问后，只要剩余块的最小下界超过阈值，就可以终止搜索。

下界等于阈值时仍须搜索，因为等距离点可能具有更小的原始行号。结果按 `(平方距离, 原始行号)` 字典序排列。候选点不足 $k$ 个时，阈值视为无穷大。

以上论证针对实数运算。实现采用 float64、保守舍入余量和受限输入范围，不提供任意近距离浮点输入在实数意义下排序的机器验证证明。参见[数值契约](ARCHITECTURE.md#数值契约)；[Goldberg 的论文](https://docs.oracle.com/cd/E19957-01/816-2464/ncg_goldberg.html) 解释了为何不能用最后一次舍入修正代替多步运算的误差分析。

## 实验能支持哪些结论

主要基线是编译实现的 [`cdist(..., "sqeuclidean")`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)，限制工作内存并采用相同的 top-k 并列规则。额外的大块穷举运行公开 Python 循环开销的影响。[`cKDTree.query(..., eps=0, p=2, workers=1)`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query.html) 作为成熟精确索引基线；它的等距离行号顺序不属于本项目契约。

空间布局关闭剪枝的消融保留相同的聚集数据块，但计算所有块，同时跳过下界计算与排序。因此，其耗时差同时包含距离计算量变化和这些阶段自身的成本，不能隔离某个操作的开销。保持输入顺序分块但启用剪枝，用于检验空间聚集的作用。冻结矩阵改变负载规模、维度和 k，叶块大小固定为 256。扫描不同叶块大小可以进一步展示下界紧致程度与块管理、候选维护成本的竞争；这是后续实验，不是当前矩阵的结果。构建时间与稳态查询时间分开报告。一个实用的摊销估计是

$$
Q_{\mathrm{break-even}}=\left\lceil\frac{T_{\mathrm{build}}}{T_{\mathrm{scan}}-T_{\mathrm{index}}}\right\rceil,
$$

前提是分母为正，而且两个查询时间使用相同单位。分母为零或负数时，该实测负载下不存在有限的回本查询量。估计不涵盖缓存状态变化、数据加载和未来查询分布变化。

预期失效因素包括包围盒大量重叠、k 较大、空间结构较弱和维度较高。轴对齐还会导致对旋转敏感，即使旋转保持欧氏距离不变。旋转实验是有价值的扩展；除非明确追加协议修订，否则不属于冻结矩阵。计算量下降不等于延迟下降，合成聚类数据上的成功也不等于真实嵌入检索性能。

## 真实数据来源

可选真实负载使用 [ANN-Benchmarks GloVe25](https://github.com/erikbern/ann-benchmarks/blob/main/README.md)，以约 121 MB 的 [HDF5 文件](https://ann-benchmarks.com/glove-25-angular.hdf5) 发布。[生成代码](https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/datasets.py) 将其来源关联到 Stanford Twitter27B 向量。[Stanford 明确以 PDDL 1.0 发布预训练 GloVe 向量](https://nlp.stanford.edu/projects/glove/)，与其代码许可证不同。

上游任务是 angular search。本地负载对选定向量进行归一化后计算平方 L2；对于单位向量，$\|x-q\|_2^2=2(1-x^\top q)$。抽样改变了候选集合，因此必须重新计算本地穷举真值；归一化也会引入舍入。报告应记录子集、种子、来源哈希和预处理，不得把上游全量语料的邻居行号直接当作子集真值。离线测试不依赖下载数据，下载向量排除在 Git 之外。参见[第三方声明](../../NOTICE_zh.md)。

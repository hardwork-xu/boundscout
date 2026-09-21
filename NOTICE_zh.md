# 第三方声明与研究归属

[English](NOTICE.md)

BoundScout 原创代码使用仓库中的 [MIT 许可证](LICENSE)。该许可证不会改变依赖、数据集或引用论文的许可条款。

实现建立在已有最近邻方法上，包括空间中位数划分、轴对齐包围盒下界和有序分支限界搜索。参见 [Arya 与 Mount 的 ANN 手册](https://www.cs.umd.edu/~mount/ANN/Files/1.1.1/ANNmanual_1.1.1.pdf) 和 [SciPy cKDTree 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html)。BoundScout 不将这些算法宣称为原创研究。引用说明思想来源，不代表被引用作者认可本项目。

Python 依赖单独安装并保留各自许可证。数值计算使用 NumPy 与 SciPy；包元数据和依赖锁文件标识所用版本。原创项目许可证不会覆盖任何依赖的源代码发行物。

可选 GloVe25 负载从 [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks) 下载，其[数据生成器](https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/datasets.py) 标明原始来源为 Stanford Twitter27B 向量。[Stanford GloVe 项目](https://nlp.stanford.edu/projects/glove/) 以 [Open Data Commons Public Domain Dedication and License 1.0](https://opendatacommons.org/licenses/pddl/1-0/) 发布预训练词向量。Stanford 代码另用 Apache 2.0，不能将其与向量数据许可证混淆。

下载语料和本地生成索引排除在 Git 之外。获取记录保存下载地址和 SHA256；基准记录说明选定负载与预处理。归一化或抽样后的 GloVe 结果属于本地转换，不是上游全量语料基准。未来新增数据集时，应明确记录来源和条款，不能假设仓库的 MIT 许可证适用于数据。

GloVe 的主要研究引用为 Jeffrey Pennington、Richard Socher 与 Christopher D. Manning 的 [“GloVe: Global Vectors for Word Representation”](https://nlp.stanford.edu/pubs/glove.pdf)，EMNLP 2014。更多一手研究来源和贡献边界见[研究依据](docs/zh/RESEARCH.md)。

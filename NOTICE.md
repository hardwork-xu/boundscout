# Third-party notices and research attribution

[简体中文](NOTICE_zh.md)

Original BoundScout code is released under the repository's [MIT license](LICENSE). That license does not relicense dependencies, datasets or referenced publications.

The implementation builds on established nearest-neighbor ideas: spatial median partitioning, axis-aligned box lower bounds and ordered branch-and-bound search. See [Arya and Mount's ANN manual](https://www.cs.umd.edu/~mount/ANN/Files/1.1.1/ANNmanual_1.1.1.pdf) and [SciPy cKDTree documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html). BoundScout does not claim those algorithms as original research. References identify intellectual provenance; they do not imply endorsement.

Python dependencies are installed separately and retain their own licenses. Numerical execution uses NumPy and SciPy; package metadata and dependency locks identify the versions used. No dependency's source distribution is incorporated into the original-project license.

The optional GloVe25 workload is downloaded from [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks), whose [dataset generator](https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/datasets.py) identifies Stanford Twitter27B vectors as the source. The [Stanford GloVe project](https://nlp.stanford.edu/projects/glove/) releases its pretrained word vectors under the [Open Data Commons Public Domain Dedication and License 1.0](https://opendatacommons.org/licenses/pddl/1-0/). Stanford's separate Apache 2.0 code license must not be confused with the vector-data license.

Downloaded corpus files and generated local indexes are excluded from Git. The acquisition record stores the download URL and SHA256; benchmark records document the selected workload and preprocessing. Normalized or sampled GloVe results are local transformations, not the upstream full-corpus benchmark. If a future contributor adds another dataset, record its provenance and terms explicitly instead of assuming the repository's MIT license applies.

The primary GloVe research reference is Jeffrey Pennington, Richard Socher and Christopher D. Manning, [“GloVe: Global Vectors for Word Representation”](https://nlp.stanford.edu/pubs/glove.pdf), EMNLP 2014. Additional primary research sources and the contribution boundary are documented in [Research](docs/en/RESEARCH.md).

# Research rationale

[简体中文](../zh/RESEARCH.md) · [Protocol](PROTOCOL.md) · [Results](RESULTS.md)

## Why I built this

I want to understand when an exact CPU vector index earns its complexity. A benchmark can show fewer distance calculations while hiding the cost of bounds, ordering, memory copies and maintaining the best candidates. BoundScout makes those costs inspectable and keeps an exhaustive implementation beside the index.

The intended user is a maintainer experimenting with offline feature retrieval: a fixed collection of vectors, repeat queries, and a requirement to retain exact-search semantics. The contribution is a small implementation, an explicit numerical contract and reproducible evidence about useful and adverse cases. This is not a claim of a new nearest-neighbor algorithm or a production search service.

## Choosing the question

The project considered three directions under a CPU-only execution budget and a 16 GiB memory planning envelope. These are feasibility judgments, not comparative benchmark results.

| Direction | Useful question | Decision |
| --- | --- | --- |
| Exact vector pruning | When do spatial blocks save enough work to repay indexing and query overhead? | Selected: no model weights; a compiled exhaustive reference gives a practical correctness and timing baseline. |
| KV-cache management | How do fragmentation and shared prefixes affect serving capacity? | Deferred: convincing evidence needs an inference runtime, request traces and prefill/decode measurements. [PagedAttention](https://arxiv.org/abs/2309.06180) provides an established systems reference. |
| Model quantization | What memory and latency savings survive quality evaluation? | Deferred: calibration, model weights and downstream evaluation broaden the workload. [GPTQ](https://arxiv.org/abs/2210.17323) illustrates that scope. |

The frozen [protocol](PROTOCOL.md) records the initial target before timing. A missed target remains a result. Measured outcomes belong in [Results](RESULTS.md), not in this rationale.

## Established foundation and local contribution

Bounding rectangles, balanced spatial partitioning and distance-ordered branch-and-bound traversal predate this project. The [ANN manual, sections 2.4.1–2.4.2](https://www.cs.umd.edu/~mount/ANN/Files/1.1.1/ANNmanual_1.1.1.pdf) describes exact search and priority traversal. [SciPy cKDTree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html) already provides compact bounding boxes, median-balanced construction and brute-force leaves.

BoundScout deliberately flattens the leaves into blocks. It computes all block bounds before searching, sorts them, and evaluates surviving blocks through compiled SciPy distance calculations. That makes the tradeoff easy to instrument, but gives every query an up-front bound and sorting cost. The persistent immutable index, deterministic tie policy, work counters, ablations and provenance are engineering choices to study, not algorithmic novelty claims.

## Why pruning is valid in real arithmetic

For a block containing points inside the axis-aligned box with lower corner $l$ and upper corner $u$, define

$$
\delta_j(q)=\max\{l_j-q_j,\;0,\;q_j-u_j\},\qquad
L_B(q)=\sum_{j=1}^{D}\delta_j(q)^2.
$$

For every point $x$ in the block, $\delta_j(q)\le |x_j-q_j|$, hence $L_B(q)\le\|x-q\|_2^2$. Once the current candidate set contains $k$ points with kth squared distance $\tau$, a block with $L_B(q)>\tau$ cannot improve the result. Visiting blocks in increasing bound order permits termination when the smallest remaining bound exceeds the threshold.

Equality must remain searchable: a point at distance $\tau$ may have a smaller original row ID. Results use lexicographic `(squared distance, original row ID)` ordering. Before there are $k$ candidates, the threshold is effectively infinite.

This argument is about real arithmetic. The implementation uses float64, a conservative rounding allowance and a restricted input domain. It does not provide a machine-checked proof of real-number ranking for arbitrarily close floating-point inputs. See the [numerical contract](ARCHITECTURE.md#numerical-contract); [Goldberg](https://docs.oracle.com/cd/E19957-01/816-2464/ncg_goldberg.html) explains why multi-operation rounding cannot be dismissed with a single final rounding adjustment.

## What the experiments can establish

The primary comparison uses compiled [`cdist(..., "sqeuclidean")`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html) with bounded working memory and the same top-k tie policy. A larger-block exhaustive run discloses sensitivity to Python loop overhead. An additional [`cKDTree.query(..., eps=0, p=2, workers=1)`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query.html) baseline represents an established exact index; its equal-distance ID ordering is not our contract.

The spatial-unpruned ablation retains the same packed blocks but evaluates every block and bypasses bound calculation and sorting. Its timing difference therefore combines changed distance work with the cost of those stages; it does not isolate an individual operation's overhead. Input-order blocks with pruning enabled test the value of spatial packing. The frozen matrix varies workload size, dimension and k while holding leaf size at 256. A leaf-size sweep would further expose the tradeoff between bound tightness and block or candidate costs; it is a follow-up experiment, not a result of this matrix. Build time and steady-state query time are reported separately. A useful break-even estimate is

$$
Q_{\mathrm{break-even}}=\left\lceil\frac{T_{\mathrm{build}}}{T_{\mathrm{scan}}-T_{\mathrm{index}}}\right\rceil,
$$

when the denominator is positive and both query timings use the same unit. A zero or negative denominator provides no finite break-even under that measured workload. This estimate excludes changes in cache state, data loading and future query distributions.

Expected failure mechanisms include overlapping boxes, large k, weak spatial structure and high ambient dimension. Axis alignment also makes the method sensitive to rotations that preserve Euclidean distances. A rotation study is a useful extension; it is not part of the frozen matrix unless explicitly recorded as an amendment. Work reduction does not establish latency reduction, and a synthetic clustered success does not establish embedding-search performance.

## Real-data provenance

The optional real workload uses [ANN-Benchmarks GloVe25](https://github.com/erikbern/ann-benchmarks/blob/main/README.md), distributed as an approximately 121 MB [HDF5 file](https://ann-benchmarks.com/glove-25-angular.hdf5). Its [generator](https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/datasets.py) traces it to Stanford's Twitter27B vectors. [Stanford licenses pretrained GloVe vectors under PDDL 1.0](https://nlp.stanford.edu/projects/glove/); this is distinct from its code license.

The upstream task is angular search. The local workload normalizes selected vectors and evaluates squared L2: on unit vectors, $\|x-q\|_2^2=2(1-x^\top q)$. Sampling changes the candidate universe, so local exhaustive ground truth must be recomputed. Normalization also introduces rounding. Report the subset, seed, source hash and preprocessing; do not reuse upstream full-corpus neighbor IDs as subset ground truth. Dataset acquisition is optional for offline tests, and downloaded vectors are excluded from Git. See [NOTICE](../../NOTICE.md).

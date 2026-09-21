# Preregistered experiment protocol

Frozen before engine implementation and timing on 2026-09-21. Any change must be appended with a reason; original targets remain visible.

## Question and scope

Can spatially packed blocks and conservative bounding-box lower bounds reduce exact CPU top-k search work at modest index and query overhead? The engineering contribution is an inspectable implementation and auditable evidence, not a new nearest-neighbor algorithm. Target users are maintainers exploring exact retrieval for offline feature vectors. No GPU, approximate recall tradeoff, embedding model, mutable index, distributed service or production claim.

## Target (not measured)

Primary target: at least **1.50x** steady-state median query-batch speedup over a bounded-memory SciPy `cdist` exhaustive scan, and at least **75%** fewer evaluated vectors, for 32,768 clustered 32-dimensional vectors, 64 held-out queries, k=10, leaf size 256, 16 clusters, cluster-center scale 8 and within-cluster standard deviation 0.5. Both implementations use float64 and one native thread. Targets are not acceptance requirements for truthful delivery.

Correctness: identical ordered IDs to exhaustive search using `(squared distance, original row ID)` tie ordering; distance tolerance `rtol=1e-12, atol=1e-12`. No nonfinite inputs. Auxiliary metrics: index build/save/load time, query batch median and min/max, throughput, vector evaluations, index array bytes, isolated-process peak RSS, and build-amortization break-even. Logical array bytes and process RSS are separate metrics.

## Baselines and ablations

Practical baseline: compiled SciPy squared-Euclidean `cdist` evaluated in 256-row blocks with partial top-k selection. Additional baseline: SciPy `cKDTree`, eps=0, workers=1 (equal-distance identity ordering may differ; compare sorted distances). Ablations: same spatial index with pruning disabled; input-order blocks with pruning enabled. All complete the same top-k work. Include a large-block exhaustive baseline to disclose small-block loop overhead.

## Workloads and timing

Synthetic clustered and isotropic Gaussian vectors at N=4,096 and 32,768, dimensions 8/32/128, k=1/10/100 where configured, plus openly sourced real embedding vectors if downloaded successfully. Include a high-dimensional adverse case, identical points and queries outside the training distribution in correctness tests. Seeds 20260921 and 20260922 for independent input generation; full matrix fixed in configs before timing. One untimed warmup and seven measured batch repetitions per method; seeded randomized method order each repetition. First query after build is recorded separately; no JIT compilation. Dataset acquisition/preparation and index building are timed separately, excluded from steady-state query time. No model or service end-to-end claim. Seven batches do not support a p99 claim. Preserve all valid samples and record failures separately.

## Acceptance

Offline pytest tests, CLI and storage round-trip, clean isolated install, Ruff and mypy, package build, machine-readable raw benchmark data with source fingerprints and commands, generated bilingual tables/plots, bilingual maintenance/release material, privacy review and local commits. Docker must be executed if available; otherwise explicitly not run. Remote CI counts only after actual execution. Public publication is authorized by the final user instruction and follows a privacy audit.

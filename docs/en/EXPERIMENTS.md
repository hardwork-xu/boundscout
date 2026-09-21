# Experiments and interpretation

[简体中文](../zh/EXPERIMENTS.md) · [Frozen protocol](PROTOCOL.md) · [All generated tables](RESULTS.md)

The primary synthetic target was met on both seeds. The stronger large-block exhaustive comparison shows 3.50–3.62× speedup on that workload, while cKDTree remains faster. The real GloVe workload and both isotropic 128-dimensional workloads expose substantial slowdowns. I use this run to document when the mechanism helps and when its overhead is wasted.

## Evidence and provenance

The full run is [raw/full.json](../../results/raw/full.json), run ID `99f9c3f6-cfb9-4e8b-a393-2d38033b24b5`, executed from `2026-09-21T12:16:14.404470+00:00` to `2026-09-21T12:17:37.935739+00:00`. It contains 12 workload/seed cases, six methods per case and **504 original measured batch samples**. All cases completed successfully; no failed runs or valid samples were removed. The per-method `samples_seconds` arrays and per-round `measurement_orders` preserve the original evidence. [RESULTS.md](RESULTS.md) contains every method's median, observed minimum/maximum, setup times and memory observations; [summary.json](../../results/summary.json) contains derived metrics.

The measured Git revision is `6b730521339dad01b7bfc9221d5d890e97bf7fc8`; `git_dirty=true` is retained. The source manifest digest is `fafc0b607164421d889499390d160ca719fa43502f3355a881949bb4d6108097`. After the run, every file listed in `source.files_sha256`, including the engine, storage, CLI, benchmark, frozen configs and then-existing scripts, was independently rehashed and matched. Later documentation-generation files are outside that recorded manifest. A later reporting-only amendment is recorded in [post-experiment-changes.json](../../results/post-experiment-changes.json); it clarifies labels and output path handling without changing measured engine/harness code or raw samples. Do not substitute a later repository revision for the recorded measured source.

## Machine and measurement conditions

The host was an Apple M1 Pro, arm64, 10 logical CPUs and 16 GiB physical memory, running Darwin kernel 25.6.0. Python was 3.12.2; BoundScout 0.1.0, NumPy 2.2.6, SciPy 1.15.3, threadpoolctl 3.6.0 and h5py 3.13.0 were used. [environment.json](../../results/environment.json) records sanitized environmental observations.

The benchmark requested one native thread with `threadpool_limits(1)` and `cKDTree.query(workers=1)`. The raw threadpool inventory is empty, so it records no detected controllable native pools. This was a shared interactive host with unrelated sustained CPU activity. No CPU affinity, exclusive-host reservation, thermal control, energy measurement or device synchronization claim is made. All execution was CPU work; GPU and container performance were not measured.

Each method was built separately, performed one postbuild single query, then one full warmup. Seven full-query-batch measurements followed, with a seeded random permutation of the six methods each round. Wall-clock timing used `perf_counter`; correctness was checked after each measured call, outside its timer. Preparation, construction, save/load, correctness checks and isolated RSS subprocesses are excluded from steady-state query latency. Save/load and first-query times are single observations, not seven-repeat distributions. No cache flushing was performed: “first query” does not mean a cold cache or application startup.

The shared-host variability remains visible. For example, the valid `clustered-large-k`/20260922 cdist-256 samples range from 17.986 to 203.780 ms, with a 19.150 ms median. All seven samples remain in the raw file. Randomizing order limits systematic ordering bias but cannot eliminate contention. Seven repeats support descriptive medians and ranges, not p99, confidence intervals or statistical significance claims. The two seeds are reported separately.

## Frozen workloads and data

The [original protocol](PROTOCOL.md) and [full.json config](../../configs/full.json) were frozen before performance timing. The primary target remains **at least 1.50× median batch speedup versus cdist-256 and at least 75% fewer evaluated vectors**. It was not changed after observing results. All full workloads use 64 held-out queries, float64, leaf size 256, and seeds 20260921 and 20260922.

| Workload | N | D | k | Source and construction |
|---|---:|---:|---:|---|
| clustered-small | 4,096 | 8 | 1 | Synthetic, 16 Gaussian clusters |
| clustered-primary | 32,768 | 32 | 10 | Synthetic, 16 Gaussian clusters |
| clustered-large-k | 4,096 | 32 | 100 | Synthetic, 16 Gaussian clusters |
| isotropic-high-d-small | 4,096 | 128 | 100 | Synthetic independent standard Gaussian vectors |
| isotropic-high-d-large | 32,768 | 128 | 10 | Synthetic independent standard Gaussian vectors |
| glove25-real | 32,768 | 25 | 10 | Sampled public GloVe train/test vectors, normalized in float64 |

For clustered data, centers have standard deviation 8 and within-cluster noise has standard deviation 0.5. Training and query cluster assignments and noise are independently drawn from the same centers. These favorable synthetic cases differ from the real embedding workload. The frozen matrix jointly varies dimensions and k; it is not a complete factorial study isolating each parameter's effect.

GloVe comes from [ANN-Benchmarks](https://ann-benchmarks.com/glove-25-angular.hdf5), with original pretrained-vector licensing described in [NOTICE](../../NOTICE.md). The downloaded file has 127,359,688 bytes and SHA256 `51004cb0ae962159f0db507a51fec2b395de14b166f55976c89f16bd2f8b6391`; it contains 1,183,514 train rows and 10,000 test rows with 25 dimensions. This is an observed file digest, not a claim that upstream publishes a checksum. Raw data records every sampled train/test row ID, seed, normalization choice, source-file hash and prepared-array hashes. Returned local IDs index the sampled training subset; `train_rows` maps them back to the source file.

Vectors are L2-normalized in float64, then evaluated with squared Euclidean distance. Upstream full-corpus neighbors are not reused because the candidate universe has changed. Data acquisition preceded the benchmark; no download-latency measurement is reported. Real-data preparation, including checksum verification, reading, sampling and normalization, took 253.141 and 117.271 ms for the two seeds and is excluded from query latency. Vectors remain outside Git; only provenance and measurements are published.

## Methods, resources and fairness

| Method | What is timed | Purpose and resource differences |
|---|---|---|
| cdist-256 | Compiled SciPy squared-L2 batches plus tie-safe partial top-k | Preregistered practical exhaustive baseline; 256 training rows per distance block |
| cdist-4096 | Identical baseline with 4,096-row blocks | Stronger measured exhaustive comparator; discloses small-block Python overhead |
| ckdtree | SciPy exact `query`, `eps=0`, `workers=1`, Euclidean output squared | Established compiled spatial index; `leafsize=256` (SciPy default: 16), no leaf-size tuning sweep |
| spatial-pruned | BoundIndex with spatial packing and conservative box bounds | Proposed inspectable mechanism; actual evaluated vectors and visited blocks recorded |
| spatial-unpruned | Same spatial layout algorithm with `prune=False` | Disables bound calculation/pruning; independently built copy for first-query timing |
| input-pruned | BoundIndex with original input-order blocks and pruning | Tests whether pruning alone helps without spatial packing |

All methods return the same k results for the same data/query batches and include query safety checks. The independent benchmark baseline does not repeatedly validate the full training matrix; indexed data validation is part of build time. It is separate from the package's general-purpose `exhaustive_search` API, which validates its database on each call. Benchmark numbers must not be presented as timings of that API.

The cdist baselines borrow the prepared input and batch all 64 queries. Their distance matrices use 128 KiB for block 256 or 2 MiB for block 4096; candidate-distance/ID merge buffers require additional memory. The core processes one query at a time, using a 2 KiB distance vector per 256-row leaf, plus all-block bounds, ordering and top-k workspace. BoundIndex owns a packed data copy while the original input stays alive. Its 1 GiB logical-index guard and 64 MiB default output guard are not whole-process memory caps. No operating-system RSS cap was imposed, and complete cKDTree logical allocation accounting is unavailable. There is no tuning sweep over tree leaf size, vectorized exhaustive block sizes beyond the two configured choices, or BLAS/GPU exact search.

## Target and complete workload outcomes

Both primary seeds met both original target thresholds. The summary below reports the spatial-pruned median, both exhaustive speedups, the cKDTree median and evaluated-vector reduction. Ratios below 1 mean slower. All six-method results and all min/max ranges remain in [RESULTS.md](RESULTS.md).

| Workload | Seed | Spatial-pruned ms | × vs 256 | × vs 4096 | cKDTree ms | Fewer vectors |
|---|---:|---:|---:|---:|---:|---:|
| clustered-small | 20260921 | 2.115 | 3.66× | 1.16× | 0.209 | 89.94% |
| clustered-small | 20260922 | 1.925 | 4.05× | 1.08× | 0.217 | 88.87% |
| clustered-primary | 20260921 | 11.195 | 8.18× | 3.62× | 5.570 | 92.72% |
| clustered-primary | 20260922 | 11.288 | 8.18× | 3.50× | 4.160 | 92.74% |
| clustered-large-k | 20260921 | 3.858 | 4.45× | 1.58× | 1.590 | 85.55% |
| clustered-large-k | 20260922 | 4.336 | 4.42× | 1.59× | 1.895 | 85.45% |
| isotropic-high-d-small | 20260921 | 30.297 | 0.81× | 0.44× | 14.852 | 0.00% |
| isotropic-high-d-small | 20260922 | 30.087 | 0.80× | 0.41× | 13.638 | 0.00% |
| isotropic-high-d-large | 20260921 | 210.032 | 0.79× | 0.47× | 463.059 | 0.00% |
| isotropic-high-d-large | 20260922 | 214.641 | 0.71× | 0.46× | 520.249 | 0.00% |
| glove25-real | 20260921 | 120.734 | 0.72× | 0.30× | 30.725 | 2.28% |
| glove25-real | 20260922 | 122.022 | 0.70× | 0.30× | 26.826 | 1.23% |

On `clustered-primary`, spatial-pruned evaluated 152,576 and 152,320 vectors per batch instead of 2,097,152. This is 92.7246% and 92.7368% fewer evaluations. Its 11.195 and 11.288 ms medians compare with cdist-256 at 91.535 and 92.389 ms and cdist-4096 at 40.504 and 39.464 ms. The primary 8.18× headline depends on the preregistered small-block baseline; the 3.50–3.62× comparison is the more conservative measured exhaustive result. cKDTree still takes only 5.570 and 4.160 ms: BoundScout is 2.01× and 2.71× slower than that exact index.

The small clustered workload gains only 1.08–1.16× against cdist-4096, despite 88.87–89.94% fewer vector evaluations. Bounds and Python candidate maintenance can absorb much of the avoided distance work. The large-k clustered workload retains 1.58–1.59× gains against cdist-4096. These observations do not establish a general relation between k, N and speedup because the matrix is not factorial.

## Ablations and adverse cases

For the two primary seeds, disabling pruning gives 124.901 and 131.174 ms and evaluates every vector; retaining pruning with input-order blocks gives 129.859 and 132.390 ms and also evaluates every vector. The spatial-pruned path takes 11.195 and 11.288 ms. On these clustered inputs, useful geometry in the packed blocks and actual skipped distance work are both necessary. The unpruned path being slower than either cdist baseline also shows that Python per-query/block bookkeeping has material cost. This ablation does not isolate a pure cache-locality effect: traversal order, bound work and pruning differ together.

All four isotropic 128-dimensional cases prune **zero vectors**. Spatial-pruned is 2.14–2.42× slower than cdist-4096. At N=32,768, it takes 210.032 and 214.641 ms versus 97.975 and 98.486 ms for large-block exhaustive search. cKDTree is even slower there, at 463.059 and 520.249 ms; winning against this tree would therefore be a misleading claim of overall advantage.

The real GloVe cases prune only 2.2827% and 1.2329% of vectors. Spatial-pruned takes 120.734 and 122.022 ms, **3.35× and 3.32× slower** than cdist-4096 at 36.001 and 36.710 ms. cKDTree takes 30.725 and 26.826 ms. The second seed is also slower with pruning than without it: 122.022 versus 119.302 ms. The observation is consistent with overlapping boxes offering weak bounds, but it is not a separately measured causal decomposition. The current implementation does not automatically select a flat scan when geometry is unfavorable.

## Build, storage, memory and break-even

Primary spatial-pruned setup observations:

| Seed | Build ms | First single query µs | Save ms | Load ms | Logical index MiB | Isolated peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| 20260921 | 46.511 | 1593.5 | 12.583 | 22.505 | 8.313 | 122.547 |
| 20260922 | 40.650 | 356.1 | 11.745 | 32.359 | 8.313 | 128.484 |

Each primary packed index occupies 8,717,320 logical array bytes; persisted files total 8,718,535 bytes. RSS measures a new process from imports through dataset preparation, construction and one query batch. It includes temporary preparation peaks and cannot isolate persistent-index overhead. For the primary seeds, cdist-256 peaks at 90.422/91.000 MiB, cdist-4096 at 126.859/118.297 MiB and cKDTree at 88.812/89.359 MiB. Spatial-pruned peaks at 122.547/128.484 MiB. These single subprocess observations must not be subtracted from logical bytes or interpreted as a controlled allocation delta.

Using the measured build cost and ratio-of-median batch savings, break-even is `ceil(build_seconds / (baseline_median_seconds - spatial_median_seconds))` when the denominator is positive. Both primary seeds repay construction after **one 64-query batch versus cdist-256**, or **two batches versus cdist-4096**. This estimate excludes save/load, acquisition and preparation, and assumes the future query distribution and timings remain similar. There is no query-time break-even against the faster primary cKDTree result. There is no finite build-amortization break-even against either exhaustive baseline for the isotropic or real GloVe cases in this run.

## Correctness, limits and next experiments

All 72 method/workload/seed records passed complete-batch distance checks at `rtol=1e-12, atol=1e-12`, including every measured repetition. All BoundIndex and cdist outputs matched ordered reference IDs and distances exactly in this matrix. The largest distance discrepancy across all methods was `2.8421709430404007e-13`, from the cKDTree Euclidean-to-squared-distance comparison. cKDTree identity tie ordering is not an acceptance requirement. All 36 BoundIndex storage round-trips and all 72 isolated RSS measurements succeeded. These are measured-input observations, not a proof of ranking for all representable floating-point inputs.

The evidence supports a narrow geometric optimization with an explicit negative real-data result. It does not establish a new nearest-neighbor algorithm, superiority to existing indexes, production latency, a serving-system improvement, or results on other machines. Axis alignment, query distribution, leaf size, larger k and high-dimensional box overlap remain important restrictions. Rotation tests, leaf-size sweeps, a measured adaptive flat fallback, larger real datasets and repeated controlled-host runs are follow-up experiments, not completed capabilities.

## Reproduce without replacing evidence

Run from the repository root after the locked installation in [README](../../README.md). Choose a new output filename; the harness refuses to overwrite existing raw evidence.

```bash
python scripts/download_data.py --output data/glove-25-angular.hdf5 --expected-sha256 51004cb0ae962159f0db507a51fec2b395de14b166f55976c89f16bd2f8b6391
python benchmark.py --config configs/full.json --output results/raw/full-reproduction.json --measure-rss
python scripts/analyze.py --input 'results/raw/*.json'
```

The default [smoke config](../../configs/smoke.json) is offline and uses small synthetic fixtures. It is an execution check, not a substitute for the full matrix. Supplying all raw JSON files to the analyzer preserves every supplied run as separate records; it does not silently select the fastest rerun.

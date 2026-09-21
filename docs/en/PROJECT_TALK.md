# Project explanation and discussion guide

[简体中文](../zh/PROJECT_TALK.md) · [Résumé material](RESUME.md) · [Results](RESULTS.md)

This guide describes the project and its evidence. Use it to prepare an explanation after inspecting the implementation; it does not certify personal mastery or prescribe an invented personal history.

## Opening: the problem and the useful result

BoundScout asks when a small exact CPU vector index repays the cost of building and querying it. The use case is a fixed set of local feature vectors with repeated top-k queries. Fewer distance calculations can be misleading if box evaluation, sorting or Python candidate maintenance costs more than the saved work. The project therefore exposes work counters, keeps compiled exhaustive baselines and records adverse cases.

Its strongest useful result is a scoped one: on the prescribed 32,768-point, 32-dimensional clustered case, it was 3.50–3.62× faster than the stronger `cdist-4096` scan, while `cKDTree` was still 2.01–2.71× faster than BoundScout. On normalized real GloVe25 data, BoundScout lost to both. The contribution is an inspectable implementation and reproducible comparison, not a new nearest-neighbor algorithm.

## How a query works

Construction splits points at the median of the coordinate with the widest range, packs leaves into contiguous blocks and stores each block's coordinate-wise minimum and maximum. Original row IDs are retained even though the stored point order changes.

For a query q and a block box [l,u], the squared distance to the box is

$$
L_B(q)=\sum_j \max(l_j-q_j,\;0,\;q_j-u_j)^2.
$$

In real arithmetic this is no larger than the squared distance to any point in the block. Search visits blocks in increasing bound order. Once k candidates exist, it can stop when the smallest remaining bound is strictly greater than the worst retained squared distance. Equality must remain searchable because an equal-distance point may have a smaller original ID. Candidate selection retains boundary ties and orders results by `(computed squared distance, original row ID)`.

Implementation uses float64 differences and SciPy's compiled squared-distance kernel. The bound is reduced by `1 - 64 * eps * (D + 1)`, moved one representable step downward and clamped to zero; the stopping threshold moves one step upward. This conservative guard and adversarial tests support the stated computed-distance contract. They are not a formal proof of exact-real ranking for arbitrarily close inputs. See [`engine.py`](../../src/boundscout/engine.py) and the [numerical contract](ARCHITECTURE.md#numerical-contract).

## Engineering tradeoffs

Flattening the leaves makes the code easy to inspect, but every query pays for all box bounds and their ordering. Weakly separated boxes can leave every point to be evaluated. Stored arrays take O(ND + N + BD) space; build copies, validation and candidate work add temporary memory. `max_bytes` limits logical index arrays rather than process RSS.

The primary-case index occupied **8.31 MiB** of logical arrays; measured fresh-process peaks were **122.55–128.48 MiB**, including imports, data preparation, build and one query batch. These are different measurements. Build took **40.65–46.51 ms**. Dividing build time by the measured batch saving against `cdist-4096` gives an estimated **two 64-query batches** to amortize construction for each seed, excluding file I/O and distribution changes.

[`storage.py`](../../src/boundscout/storage.py) saves arrays and metadata in a temporary sibling directory before publishing a new index path. Loading checks hashes, shapes, original-ID permutation and actual block geometry, then uses read-only memory maps. Hashes detect changed content relative to the manifest, not author authenticity. Another process can still modify mapped files. Context managers release mappings, and callers must not close an index during a search. The [security notes](../../SECURITY.md) define the trust boundary.

## The experiment to explain in detail

The frozen matrix contains six workloads, two seeds, six methods and seven measured repetitions after one warmup. Seeded randomized method order and all **504 timed batches** are retained. Timing used one native thread on an Apple M1 Pro with 16 GiB RAM, on a shared interactive host without exclusive CPU or thermal control. The **111 offline tests** include an independent direct-distance oracle, ties, duplicates, large offsets, numeric limits, concurrent reads, file corruption and real CLI subprocesses. Every one of the **72 measured method cases** passed its correctness check; `cKDTree` checks sorted distances because equal-distance ID order is outside its contract.

On `clustered-primary`, the two seeds gave these median times for a complete 64-query batch:

| Method | Seed 20260921 | Seed 20260922 |
| --- | ---: | ---: |
| `cdist-256` | 91.535 ms | 92.389 ms |
| `cdist-4096` | 40.504 ms | 39.464 ms |
| `spatial-pruned` | 11.195 ms | 11.288 ms |
| `ckdtree` | 5.570 ms | 4.160 ms |

BoundScout reduced evaluated vectors by **92.72–92.74%**, meeting the frozen 75% work-reduction and 1.50× `cdist-256` speed targets. Its 8.18× result against `cdist-256` becomes 3.50–3.62× against the larger-block scan. That difference exposes baseline loop overhead. Input-order blocks eliminated **0%** of vectors on this case; turning pruning off also lost the speed advantage. The latter ablation removes bound computation and sorting as well, so it does not measure a distance-kernel-only effect.

## Negative cases and the next question

Both 128-dimensional isotropic cases eliminated **0%** of vectors. On the 32,768-point case, BoundScout took **210.03–214.64 ms** versus **97.98–98.49 ms** for `cdist-4096`. `cKDTree` was slower still there, so no tested index won universally.

On the normalized 32,768-point GloVe25 subset, only **1.23–2.28%** of vector evaluations disappeared. BoundScout took **120.73–122.02 ms**, compared with **36.00–36.71 ms** for `cdist-4096` and **26.83–30.72 ms** for `cKDTree`. Synthetic cluster separation did not transfer to this real workload. Explain this result before proposing deployment or further optimization.

A useful next experiment is to rotate the same vectors with an orthogonal transform and sweep leaf size while retaining identical Euclidean distances, inputs and baseline methods. This would test how much the observed pruning depends on axis alignment and block granularity. It is future work; this repository's frozen matrix does not establish its outcome. With only seven timed batches per method and a shared host, the current evidence supports median comparisons within the recorded setup, not p99 or statistical-significance claims.

## Evidence to have open

Open [`engine.py`](../../src/boundscout/engine.py), [`storage.py`](../../src/boundscout/storage.py), [`tests`](../../tests), [`configs/full.json`](../../configs/full.json), [`benchmark.py`](../../benchmark.py) and [`results/raw/full.json`](../../results/raw/full.json). The [résumé evidence index](RESUME.md#evidence-index) maps claims to functions and raw fields. The complete run is `99f9c3f6-cfb9-4e8b-a393-2d38033b24b5`, finished `2026-09-21T12:17:37.935739+00:00`; use its `source.files_sha256` values to identify measured code. Read [Results](RESULTS.md) for all cases rather than showing only the primary row.

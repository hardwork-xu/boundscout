# Résumé material

[简体中文](../zh/RESUME.md) · [Project talk](PROJECT_TALK.md) · [Results](RESULTS.md)

These are project-grounded drafting options. Adapt the wording to the work you personally contributed and can explain; the existence of this repository does not establish personal mastery, employment experience or a production deployment.

## Four matching bullets

- Implemented an inspectable CPU vector-search prototype with spatial median blocks, conservative bounding-box pruning, float64 squared-L2 distances and deterministic original-ID tie ordering; exposed a Python API and offline CLI.
- Added immutable NPY index persistence, read-only mmap loading, SHA256 and geometry validation, explicit memory budgets and lifecycle handling; validated engine, storage, concurrent reads, CLI and benchmark behavior with **111 passing offline tests**.
- Evaluated six workloads with six methods, two input seeds and seven measured repetitions per method: **72 workload/seed/method cases and 504 timed query batches**, retaining raw samples, ablations, source hashes and isolated-process peak RSS.
- On the prescribed **32,768 × 32 clustered workload, 64 queries, k=10**, measured **3.50–3.62×** speedup over the stronger `cdist-4096` exhaustive baseline and **92.72–92.74%** fewer evaluated vectors; reported that `cKDTree` remained **2.01–2.71× faster** and that normalized GloVe25 achieved only **0.30×** the `cdist-4096` speed.

## Keep the measurement scope attached

The primary workload also achieved 8.18× versus `cdist-256`; citing that number alone hides the stronger exhaustive baseline. Ranges above are the two seeds' ratios of seven-repetition median batch times, not confidence intervals. Runs used one native thread on an Apple M1 Pro with 16 GiB RAM and a shared interactive workload. Preparation and index building are excluded from steady-state query latency. This is a bounded CPU experiment, not a universal speedup, p99 claim, novel nearest-neighbor algorithm or production result.

## Evidence index

| Claim | Inspectable evidence |
| --- | --- |
| Spatial partitioning, guarded pruning and tie semantics | [`engine.py`](../../src/boundscout/engine.py): `BoundIndex.build`, `BoundIndex.search`, `_topk`; [numerical contract](ARCHITECTURE.md#numerical-contract) |
| Persistent index and file-based workflow | [`storage.py`](../../src/boundscout/storage.py): `save_index`, `load_index`; [`cli.py`](../../src/boundscout/cli.py): `main` |
| Test coverage and executable checks | [`test_engine.py`](../../tests/test_engine.py), [`test_storage.py`](../../tests/test_storage.py), [`test_cli.py`](../../tests/test_cli.py), [`test_benchmark.py`](../../tests/test_benchmark.py); run `python -m pytest -q` in the installed development environment |
| Frozen workload matrix and timing procedure | [`configs/full.json`](../../configs/full.json), [protocol](PROTOCOL.md), [`benchmark.py`](../../benchmark.py): `cdist_search`, `run_workload` |
| Every numerical claim | [`results/raw/full.json`](../../results/raw/full.json): select `workloads` by `workload_id` and `seed`, then read `methods[method].median_seconds`, `evaluated_vectors`, `correctness` and `samples_seconds`; [`summary.json`](../../results/summary.json) contains derived ratios |
| Hardware, conditions and measured source | [`environment.json`](../../results/environment.json); raw JSON `machine`, `config`, `source.files_sha256` and `source.sha256` |

Evidence run: `99f9c3f6-cfb9-4e8b-a393-2d38033b24b5`, finished `2026-09-21T12:17:37.935739+00:00`, status `ok`. Its source fingerprint is `fafc0b607164421d889499390d160ca719fa43502f3355a881949bb4d6108097`. The run records `git_dirty: true`; the per-file hashes identify the measured contents, so the recorded Git revision alone is insufficient provenance.

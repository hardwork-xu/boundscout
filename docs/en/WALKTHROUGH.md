# A complete local walkthrough

[简体中文](../zh/WALKTHROUGH.md) · [Architecture](ARCHITECTURE.md) · [Protocol](PROTOCOL.md)

The commands below run from the repository root with Python 3.12. Initial dependency installation needs access to a package source. The synthetic demo, index build, query and correctness tests then run locally without a model, service or downloaded corpus.

## Install and check the demo

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements.lock
python -m pip install --no-deps --no-build-isolation -e .
python -m boundscout demo
```

The demo creates seeded clustered vectors, builds an index and compares search with the exhaustive reference. Its JSON reports `ok`, result shape, maximum absolute distance error, work counters and logical index bytes. `ok: true` is the demo's success criterion, not a benchmark speedup or a promise about other inputs. A correctness failure exits with status 1; ordinary CLI errors exit with status 2.

## Prepare a small synthetic collection

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

Rows are vectors and columns are coordinates. The database has shape `(N, D)`; queries have shape `(Q, D)`, including `(1, D)` for one query. The returned ID is the zero-based original database row number. This example is synthetic and intentionally favors spatial structure.

## Build, save and query from the CLI

```sh
python -m boundscout build --input work/tutorial/points.npy --output work/tutorial/index --leaf-size 256
python -m boundscout query --index work/tutorial/index --input work/tutorial/queries.npy --output work/tutorial/results.npz --k 10
```

Both output paths must be new. To repeat the walkthrough, select new output names; the CLI refuses to overwrite an index or query result. Build emits shape and logical array bytes. Query verifies the saved index and writes an NPZ containing `indices` and `distances`, each shaped `(Q, k)`. Distances are **squared L2**, not Euclidean lengths. Work counters are printed as JSON on standard output.

```sh
python - <<'PY'
import numpy as np

with np.load("work/tutorial/results.npz", allow_pickle=False) as result:
    print("IDs:", result["indices"][0])
    print("Squared L2:", result["distances"][0])
PY
```

Use `build --layout input` for consecutive input blocks, or `query --no-prune` to visit every stored block. These are ablation controls; they retain the requested top-k task. Use `build --help` and `query --help` for the complete CLI surface.

## Compare in Python and manage the index lifetime

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

The new `python-index` path keeps this example separate from the CLI index. Context managers close indexes on exit. Search results own their arrays and remain usable after index closure; private mapped index arrays do not. Never close an index while another operation is searching it. Default loading checks hashes and geometry; it is not constant-time despite mmap.

`evaluated_vectors` sums actual vector distance evaluations across all queries. `visited_blocks` counts evaluated blocks, and `total_blocks` is the number of available blocks multiplied by `query_count`. For a nonempty query batch, vector reduction relative to exhaustive work is `1 - evaluated_vectors / (N * Q)`. This fraction measures work saved, not speedup.

## Reproduce experiments

```sh
python -m pytest
python benchmark.py --config configs/smoke.json --output results/raw/tutorial-smoke.json
```

For the optional real-data workload and full frozen experiment:

```sh
python scripts/download_data.py --output data/glove-25-angular.hdf5
python benchmark.py --config configs/full.json --output results/raw/tutorial-reproduction.json --measure-rss
```

The download is approximately 121 MB. Acquisition records provenance and SHA256 beside the file. Consult [Research](RESEARCH.md#real-data-provenance) and [NOTICE](../../NOTICE.md) for source and terms. The full configuration includes synthetic controls and an optional normalized GloVe subset; acquisition failure must remain visible rather than becoming a synthetic substitute labeled as real data.

Read [Results](RESULTS.md) together with the [frozen protocol](PROTOCOL.md). Warmup, repeated batch samples, native thread settings and source fingerprints matter when comparing your run. A raw smoke result is a functional check; do not interpret it as the full evaluation.

## Common input errors

| Symptom | Check |
| --- | --- |
| Dimension error | Both arrays must be two-dimensional with the same D, where `1 <= D <= 4096`. |
| Invalid k | Use an integer with `1 <= k <= N`; a Boolean is not an accepted integer parameter. |
| Nonfinite or magnitude error | Remove NaN/Inf and keep absolute coordinates at most `1e100`; review the preprocessing that created them. |
| Index or output budget error | The default index array budget is 1 GiB and search output budget is 64 MiB. Query in smaller batches when output is too large. |
| Existing output | Choose a new path. Index replacement is an explicit new build, not an in-place update. |
| Checksum or geometry error | Stop using that index and rebuild from the original trusted vectors. |

This project does not create embeddings, assign semantic labels or translate retrieved IDs into application records. Keep the original row-to-record mapping in your application and align it with the vectors used for that index version.

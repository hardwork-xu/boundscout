# Release preparation

[简体中文](../zh/RELEASE.md)

I maintain this release as an inspectable CPU research artifact. Version 0.1.0 is a starting point for independent reproduction and further study; it is not a production service or a new nearest-neighbor algorithm.

## Draft release notes: 0.1.0

- Deterministic float64 squared-L2 top-k with spatial median blocks and conservative lower-bound pruning.
- Pruning and layout ablations, independent compiled exhaustive baselines, and a SciPy cKDTree comparator.
- Read-only NPY index persistence with schema, hashes, ID and geometric validation; explicit close and resource guards.
- Offline bilingual CLI, regression tests, hash-locked Python 3.12 environment, CI workflow, and raw-to-report benchmark tooling.
- Real GloVe25 and synthetic workloads; full measurements, failures and scope belong to the [experiment record](EXPERIMENTS.md).

Known limits: immutable in-memory construction; high-dimensional pruning may be slower; Python block traversal overhead; no GPU, approximate mode, streaming insertion or service throughput claim. Logical memory budgets are not hard process-RSS limits. Docker configuration is supplied but was not executed locally because Docker is absent. Remote CI and public release status must be checked in `results/publication.json` when that record is available, rather than inferred from the workflow file.

## Build and demo

From a checkout with Python 3.12:

```bash
python3 -m venv .venv
make install
make demo
make check
```

Artifacts are `dist/boundscout-0.1.0-py3-none-any.whl` and `dist/boundscout-0.1.0.tar.gz`. `python -m build --no-isolation` runs with the pinned build backend installed by `make install`.

```bash
docker build -t boundscout .
docker run --rm boundscout
```

These Docker commands are a reproduction path, not evidence of a successful container run. The image uses fixed Python 3.12.10; the base tag is not digest-pinned, so future OS-layer rebuilds can differ. The local evidence uses Python 3.12.2 and must not be equated with container evidence.

## Public metadata

One-line / GitHub About: **Inspectable exact CPU vector search with block pruning, deterministic ties, and reproducible benchmarks.**

Topics: `nearest-neighbor-search`, `vector-search`, `cpu`, `numpy`, `scipy`, `benchmark`, `reproducible-research`, `python`.

License: [MIT](../../LICENSE). Dependencies and GloVe attribution: [NOTICE](../../NOTICE.md). `CITATION.cff` is JSON, which is valid YAML, and names only the verified public maintainer account. Cite the version, source revision and benchmark run ID; there is no DOI or associated publication.

Publication scope is the GitHub repository and its release assets. No PyPI upload, hosted service, paid resource, external announcement, or private dataset publication is part of this release. Release assets are built locally; the publication receipt distinguishes actual external actions from prepared commands.

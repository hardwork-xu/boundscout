"""Offline benchmark/schema tests; 离线基准与证据格式测试。"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("boundscout_benchmark", ROOT / "benchmark.py")
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_independent_baseline_preserves_ties_across_blocks() -> None:
    data = np.array([[1.0], [-1.0], [0.0], [1.0], [-1.0], [0.0]])
    queries = np.array([[0.0], [0.5]])
    distances = cdist(queries, data, metric="sqeuclidean")
    expected = np.array([np.lexsort((np.arange(len(data)), row))[:4] for row in distances])
    for block in (1, 2, 256, 4096):
        actual = benchmark.cdist_search(data, queries, 4, block)
        np.testing.assert_array_equal(actual.indices, expected)
        np.testing.assert_allclose(
            actual.distances, np.take_along_axis(distances, expected, axis=1)
        )
        assert actual.evaluated_vectors == len(data) * len(queries)


def test_offline_benchmark_and_portable_raw_schema(tmp_path: Path) -> None:
    config = benchmark.load_config(ROOT / "configs/smoke.json")
    config["workloads"] = [dict(config["workloads"][0], n=48, queries=3)]
    config["leaf_size"] = 8
    raw = benchmark.run_benchmark(config)
    assert raw["status"] == "ok"
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(raw, allow_nan=False))
    restored = json.loads(raw_path.read_text())
    benchmark.validate_raw(restored)
    assert len(restored["workloads"][0]["measurement_orders"]) == 7
    assert all(
        method["correctness"]["passed"] and len(method["samples_seconds"]) == 7
        for method in restored["workloads"][0]["methods"].values()
    )
    assert str(Path.home()) not in raw_path.read_text()
    broken = copy.deepcopy(restored)
    broken["workloads"][0]["methods"]["cdist-256"]["samples_seconds"][0] = 0.0
    with pytest.raises(ValueError, match="timing"):
        benchmark.validate_raw(broken)


def test_dataset_failure_remains_in_raw_evidence(tmp_path: Path) -> None:
    config = benchmark.load_config(ROOT / "configs/smoke.json")
    config["workloads"] = [
        {
            "id": "missing-real-data",
            "kind": "glove",
            "n": 32,
            "dimensions": 25,
            "queries": 2,
            "k": 1,
            "dataset_path": str(tmp_path / "absent.hdf5"),
            "seeds": [1],
        }
    ]
    raw = benchmark.run_benchmark(config)
    assert raw["status"] == "partial_failure"
    assert raw["workloads"][0]["status"] == "failed"
    assert "FileNotFoundError" in raw["workloads"][0]["error"]
    assert raw["workloads"][0]["methods"] == {}
    benchmark.validate_raw(json.loads(json.dumps(raw)))


def test_config_rejects_invalid_matrix(tmp_path: Path) -> None:
    config = json.loads((ROOT / "configs/smoke.json").read_text())
    config["workloads"][0]["k"] = config["workloads"][0]["n"] + 1
    path = tmp_path / "bad-config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="k"):
        benchmark.load_config(path)

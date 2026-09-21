"""Exercise the installed module through real, offline subprocess calls."""

import json
import re
import subprocess
import sys

import numpy as np
import pytest


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "boundscout", *map(str, args)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


@pytest.mark.integration
def test_offline_demo_returns_successful_json():
    result = run_cli("demo")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["ok"] is True


@pytest.mark.integration
def test_build_and_query_round_trip(tmp_path):
    rng = np.random.default_rng(102)
    data = rng.normal(size=(51, 3))
    queries = rng.normal(size=(4, 3))
    data_path = tmp_path / "points.npy"
    query_path = tmp_path / "queries.npy"
    index_path = tmp_path / "index"
    np.save(data_path, data, allow_pickle=False)
    np.save(query_path, queries, allow_pickle=False)
    built = run_cli("build", "--input", data_path, "--output", index_path, "--leaf-size", 16)
    assert built.returncode == 0, built.stderr
    distances = np.sum((queries[:, None] - data[None]) ** 2, axis=2)
    expected_ids = np.argsort(distances, axis=1, kind="stable")[:, :3]
    expected_distances = np.take_along_axis(distances, expected_ids, axis=1)
    for suffix, extra in [("pruned", []), ("flat", ["--no-prune"])]:
        output = tmp_path / f"result-{suffix}.npz"
        queried = run_cli(
            "query",
            "--index",
            index_path,
            "--input",
            query_path,
            "--output",
            output,
            "--k",
            3,
            *extra,
        )
        assert queried.returncode == 0, queried.stderr
        with np.load(output, allow_pickle=False) as result:
            np.testing.assert_array_equal(result["indices"], expected_ids)
            np.testing.assert_allclose(result["distances"], expected_distances, rtol=1e-13)


@pytest.mark.integration
def test_missing_input_reports_bilingual_error(tmp_path):
    result = run_cli("build", "--input", tmp_path / "missing.npy", "--output", tmp_path / "index")
    assert result.returncode == 2
    assert re.search(r"[A-Za-z]", result.stderr)
    assert re.search(r"[\u4e00-\u9fff]", result.stderr)
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "index").exists()


@pytest.mark.integration
def test_invalid_dataset_does_not_create_index(tmp_path):
    source = tmp_path / "invalid.npy"
    np.save(source, np.array([[np.nan]]), allow_pickle=False)
    result = run_cli("build", "--input", source, "--output", tmp_path / "index")
    assert result.returncode == 2
    assert re.search(r"[\u4e00-\u9fff]", result.stderr)
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "index").exists()


@pytest.mark.integration
def test_cli_refuses_to_replace_index(tmp_path):
    source = tmp_path / "points.npy"
    np.save(source, np.zeros((8, 2)), allow_pickle=False)
    target = tmp_path / "index"
    first = run_cli("build", "--input", source, "--output", target)
    assert first.returncode == 0, first.stderr
    before = (target / "metadata.json").read_bytes()
    second = run_cli("build", "--input", source, "--output", target)
    assert second.returncode == 2
    assert (target / "metadata.json").read_bytes() == before


@pytest.mark.integration
def test_invalid_argument_type_reports_bilingual_error(tmp_path):
    result = run_cli(
        "build",
        "--input",
        tmp_path / "points.npy",
        "--output",
        tmp_path / "index",
        "--leaf-size",
        "not-an-integer",
    )
    assert result.returncode == 2
    assert re.search(r"[A-Za-z]", result.stderr)
    assert re.search(r"[\u4e00-\u9fff]", result.stderr)
    assert "Traceback" not in result.stderr


@pytest.mark.integration
def test_invalid_k_and_existing_result_are_not_written(tmp_path):
    source = tmp_path / "points.npy"
    queries = tmp_path / "queries.npy"
    target = tmp_path / "index"
    np.save(source, np.zeros((8, 2)), allow_pickle=False)
    np.save(queries, np.zeros((1, 2)), allow_pickle=False)
    built = run_cli("build", "--input", source, "--output", target)
    assert built.returncode == 0, built.stderr
    output = tmp_path / "result.npz"
    invalid = run_cli("query", "--index", target, "--input", queries, "--output", output, "--k", 9)
    assert invalid.returncode == 2
    assert re.search(r"[\u4e00-\u9fff]", invalid.stderr)
    assert not output.exists()
    output.write_bytes(b"existing result must be retained")
    refused = run_cli("query", "--index", target, "--input", queries, "--output", output, "--k", 1)
    assert refused.returncode == 2
    assert output.read_bytes() == b"existing result must be retained"

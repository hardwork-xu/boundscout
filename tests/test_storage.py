"""Real filesystem round trips and persistence integrity failures."""

import hashlib
import json

import numpy as np
import pytest

from boundscout import BoundIndex


@pytest.fixture
def saved_index(tmp_path):
    rng = np.random.default_rng(42)
    data = rng.normal(size=(65, 4))
    path = tmp_path / "index"
    with BoundIndex.build(data, leaf_size=7) as index:
        index.save(path)
    return path, data


@pytest.mark.integration
@pytest.mark.parametrize("layout", ["input", "spatial"])
def test_save_load_preserves_ids_distances_and_size(tmp_path, layout):
    rng = np.random.default_rng(11)
    data = rng.integers(-3, 4, size=(63, 5)).astype(np.float64)
    queries = rng.integers(-3, 4, size=(6, 5)).astype(np.float64)
    path = tmp_path / "saved"
    with BoundIndex.build(data, leaf_size=8, layout=layout) as index:
        expected = index.search(queries, k=9)
        nbytes = index.nbytes
        index.save(path)
    assert (path / "metadata.json").is_file()
    assert list(path.glob("*.npy"))
    for array_file in path.glob("*.npy"):
        assert np.load(array_file, allow_pickle=False).dtype != object
    with BoundIndex.load(path) as loaded:
        actual = loaded.search(queries, k=9)
        assert loaded.nbytes == nbytes
        np.testing.assert_array_equal(actual.indices, expected.indices)
        np.testing.assert_array_equal(actual.distances, expected.distances)


def test_save_refuses_existing_directory_without_modifying_it(saved_index):
    path, data = saved_index
    before = {file.name: file.read_bytes() for file in path.iterdir() if file.is_file()}
    with BoundIndex.build(data) as index:
        with pytest.raises((ValueError, FileExistsError)):
            index.save(path)
    after = {file.name: file.read_bytes() for file in path.iterdir() if file.is_file()}
    assert before == after


def test_save_refuses_existing_empty_directory(tmp_path):
    path = tmp_path / "existing"
    path.mkdir()
    with BoundIndex.build(np.ones((2, 2))) as index:
        with pytest.raises((ValueError, FileExistsError)):
            index.save(path)
    assert not list(path.iterdir())


def test_load_rejects_corrupted_array_payload(saved_index):
    path, _ = saved_index
    array_path = sorted(path.glob("*.npy"))[0]
    payload = bytearray(array_path.read_bytes())
    payload[-1] ^= 1
    array_path.write_bytes(payload)
    with pytest.raises((ValueError, OSError)):
        BoundIndex.load(path)


def test_load_rejects_missing_array(saved_index):
    path, _ = saved_index
    sorted(path.glob("*.npy"))[0].unlink()
    with pytest.raises((ValueError, OSError)):
        BoundIndex.load(path)


@pytest.mark.parametrize("payload_kind", ["empty", "npz"])
def test_load_rejects_non_npy_payload_with_matching_digest(saved_index, payload_kind):
    path, _ = saved_index
    array_path = path / "ids.npy"
    if payload_kind == "empty":
        array_path.write_bytes(b"")
    else:
        with array_path.open("wb") as stream:
            np.savez(stream, ids=np.array([0, 1], dtype=np.int64))
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["sha256"][array_path.name] = hashlib.sha256(array_path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        BoundIndex.load(path)


@pytest.mark.parametrize("payload", ["not json", "[]", "null", "{}"])
def test_load_rejects_invalid_metadata(saved_index, payload):
    path, _ = saved_index
    (path / "metadata.json").write_text(payload, encoding="utf-8")
    with pytest.raises((ValueError, OSError)):
        BoundIndex.load(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("format_version", 2),
        ("format_version", True),
        ("leaf_size", 0),
        ("leaf_size", True),
        ("layout", "unknown"),
        ("shape", [1, 4]),
        ("array_bytes", 0),
        ("sha256", None),
        ("sha256", []),
        ("sha256", "invalid"),
        ("sha256", {}),
    ],
)
def test_load_rejects_tampered_metadata_fields(saved_index, field, value):
    path, _ = saved_index
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        BoundIndex.load(path)


@pytest.mark.parametrize("array_name", ["lower", "upper", "ids", "offsets"])
def test_verify_false_still_rejects_invalid_geometry_or_ids(saved_index, array_name):
    path, _ = saved_index
    array_path = path / f"{array_name}.npy"
    values = np.load(array_path, allow_pickle=False)
    if array_name in ("lower", "upper"):
        values[0, 0] += 1
    elif array_name == "ids":
        values[0] = values[1]
    else:
        values[1] = 0
    np.save(array_path, values, allow_pickle=False)
    with pytest.raises(ValueError):
        BoundIndex.load(path, verify=False)


def test_load_enforces_memory_budget(saved_index):
    path, _ = saved_index
    with pytest.raises((ValueError, MemoryError)):
        BoundIndex.load(path, max_bytes=16)


def test_verify_false_still_loads_valid_index(saved_index):
    path, data = saved_index
    with BoundIndex.load(path, verify=False) as loaded:
        result = loaded.search(data[:3], k=1)
    np.testing.assert_array_equal(result.indices[:, 0], np.arange(3))
    np.testing.assert_array_equal(result.distances, np.zeros((3, 1)))


def test_metadata_is_plain_json(saved_index):
    path, _ = saved_index
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    assert isinstance(metadata, dict)
    assert metadata

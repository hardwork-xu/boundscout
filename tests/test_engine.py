"""Offline correctness tests using an independent direct-distance oracle."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from boundscout import BoundIndex, exhaustive_search


def oracle(data, queries, k):
    """Compute the documented (squared distance, original ID) ordering."""
    points = np.asarray(data, dtype=np.float64)
    query_points = np.asarray(queries, dtype=np.float64)
    distances = np.sum((query_points[:, None, :] - points[None, :, :]) ** 2, axis=2)
    ids = np.arange(len(points), dtype=np.int64)
    ordered = np.stack([np.lexsort((ids, row))[:k] for row in distances])
    return ordered, np.take_along_axis(distances, ordered, axis=1)


@pytest.mark.parametrize("layout", ["input", "spatial"])
@pytest.mark.parametrize("seed", [0, 7, 42])
@pytest.mark.parametrize("k", [1, 9, 70])
def test_search_matches_independent_oracle(layout, seed, k):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(70, 5))
    queries = rng.normal(size=(11, 5))
    expected_ids, expected_distances = oracle(data, queries, k)
    with BoundIndex.build(data, leaf_size=9, layout=layout) as index:
        for prune in (False, True):
            result = index.search(queries, k=k, prune=prune)
            np.testing.assert_array_equal(result.indices, expected_ids)
            np.testing.assert_allclose(result.distances, expected_distances, rtol=1e-13)
            assert result.indices.dtype == np.int64
            assert result.distances.dtype == np.float64
            assert result.stats.query_count == len(queries)
            assert result.stats.evaluated_vectors <= len(data) * len(queries)
            if not prune:
                assert result.stats.evaluated_vectors == len(data) * len(queries)


@pytest.mark.parametrize("leaf_size", [1, 3, 64])
@pytest.mark.parametrize("layout", ["input", "spatial"])
def test_ties_use_original_ids_across_blocks(leaf_size, layout):
    data = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0], [0, 0]])
    queries = np.array([[0, 0], [1, 0]])
    expected_ids, expected_distances = oracle(data, queries, 4)
    with BoundIndex.build(data, leaf_size=leaf_size, layout=layout) as index:
        for _ in range(3):
            result = index.search(queries, k=4)
            np.testing.assert_array_equal(result.indices, expected_ids)
            np.testing.assert_array_equal(result.distances, expected_distances)


def test_all_duplicate_vectors_are_stably_ordered():
    data = np.ones((33, 4))
    with BoundIndex.build(data, leaf_size=2) as index:
        result = index.search(np.ones((2, 4)), k=11)
    np.testing.assert_array_equal(result.indices, np.tile(np.arange(11), (2, 1)))
    np.testing.assert_array_equal(result.distances, np.zeros((2, 11)))


def test_nearby_points_with_large_offset_avoid_cancellation():
    data = 1e12 + np.array([[0, 0], [0.25, 0], [0, 0.5], [1, 1], [-1, 0]])
    queries = 1e12 + np.array([[0.125, 0.125], [0.375, 0]])
    expected_ids, expected_distances = oracle(data, queries, 3)
    with BoundIndex.build(data, leaf_size=1) as index:
        result = index.search(queries, k=3)
    np.testing.assert_array_equal(result.indices, expected_ids)
    np.testing.assert_array_equal(result.distances, expected_distances)


def test_documented_maximum_magnitude_stays_finite():
    data = np.array([[1e100, 1e100], [-1e100, -1e100], [0, 0]])
    queries = np.array([[-1e100, 1e100]])
    expected_ids, expected_distances = oracle(data, queries, 3)
    with BoundIndex.build(data, leaf_size=1) as index:
        result = index.search(queries, k=3)
    assert np.isfinite(result.distances).all()
    np.testing.assert_array_equal(result.indices, expected_ids)
    np.testing.assert_allclose(result.distances, expected_distances, rtol=1e-13)


def test_pruning_avoids_visiting_distant_blocks():
    rng = np.random.default_rng(123)
    data = np.concatenate(
        [rng.normal(loc=center, scale=0.01, size=(40, 3)) for center in (-1000, 0, 1000)]
    )
    queries = np.zeros((1, 3))
    with BoundIndex.build(data, leaf_size=10) as index:
        pruned = index.search(queries, k=3)
        flat = index.search(queries, k=3, prune=False)
    np.testing.assert_array_equal(pruned.indices, flat.indices)
    np.testing.assert_array_equal(pruned.distances, flat.distances)
    assert 0 < pruned.stats.evaluated_vectors < flat.stats.evaluated_vectors
    assert 0 < pruned.stats.visited_blocks < pruned.stats.total_blocks


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_build_copies_numeric_source_and_handles_noncontiguous_queries(dtype):
    source = np.arange(60, dtype=dtype).reshape(10, 6)[:, ::2]
    original = source.copy()
    queries = np.arange(12, dtype=dtype).reshape(2, 6)[:, ::2]
    expected_ids, expected_distances = oracle(original, queries, 3)
    with BoundIndex.build(source, leaf_size=3) as index:
        source[:] = -999
        result = index.search(queries, k=3)
        assert index.nbytes >= original.size * 8
    np.testing.assert_array_equal(result.indices, expected_ids)
    np.testing.assert_array_equal(result.distances, expected_distances)


def test_empty_query_batch_has_correct_shapes_and_zero_work():
    with BoundIndex.build(np.ones((5, 3)), leaf_size=2) as index:
        result = index.search(np.empty((0, 3)), k=4)
    assert result.indices.shape == (0, 4)
    assert result.distances.shape == (0, 4)
    assert result.stats.evaluated_vectors == 0
    assert result.stats.visited_blocks == 0
    assert result.stats.query_count == 0


def test_concurrent_readonly_searches_match_serial_results():
    rng = np.random.default_rng(17)
    data = rng.normal(size=(127, 7))
    batches = [rng.normal(size=(5, 7)) for _ in range(8)]
    with BoundIndex.build(data, leaf_size=11) as index:
        expected = [index.search(batch, k=7) for batch in batches]
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual = list(pool.map(lambda batch: index.search(batch, k=7), batches))
    for serial, concurrent in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(concurrent.indices, serial.indices)
        np.testing.assert_array_equal(concurrent.distances, serial.distances)
        assert concurrent.stats == serial.stats


@pytest.mark.parametrize("block_size", [1, 7, 100])
def test_exhaustive_reference_matches_independent_oracle(block_size):
    rng = np.random.default_rng(991)
    data = rng.integers(-2, 3, size=(47, 4))
    queries = rng.integers(-2, 3, size=(7, 4))
    expected_ids, expected_distances = oracle(data, queries, 13)
    result = exhaustive_search(data, queries, k=13, block_size=block_size)
    np.testing.assert_array_equal(result.indices, expected_ids)
    np.testing.assert_array_equal(result.distances, expected_distances)


@pytest.mark.parametrize(
    "data",
    [
        np.empty((0, 3)),
        np.empty((3, 0)),
        np.ones(3),
        np.ones((2, 3, 4)),
        np.ones((1, 4097)),
        np.array([[np.nan]]),
        np.array([[np.inf]]),
        np.array([[-np.inf]]),
        np.array([[1.01e100]]),
        np.array([[1 + 0j]]),
        np.array([[True]]),
        np.array([[1]], dtype=object),
        np.array([["1"]]),
    ],
)
def test_invalid_database_is_rejected(data):
    with pytest.raises((ValueError, TypeError)):
        BoundIndex.build(data)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"leaf_size": 0},
        {"leaf_size": -1},
        {"layout": "unknown"},
        {"max_bytes": 1},
    ],
)
def test_invalid_build_options_are_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        BoundIndex.build(np.zeros((8, 2)), **kwargs)


@pytest.mark.parametrize(
    "queries",
    [
        np.ones(2),
        np.ones((2, 3)),
        np.ones((2, 2, 1)),
        np.array([[np.nan, 0]]),
        np.array([[0, np.inf]]),
        np.array([[0, 1.01e100]]),
        np.array([[1 + 0j, 0]]),
        np.array([[True, False]]),
        np.array([[0, 0]], dtype=object),
    ],
)
def test_invalid_queries_are_rejected(queries):
    with BoundIndex.build(np.zeros((8, 2))) as index:
        with pytest.raises((ValueError, TypeError)):
            index.search(queries, k=1)


@pytest.mark.parametrize("k", [0, -1, 9])
def test_invalid_neighbor_count_is_rejected(k):
    with BoundIndex.build(np.zeros((8, 2))) as index:
        with pytest.raises((ValueError, TypeError)):
            index.search(np.zeros((1, 2)), k=k)


def test_output_memory_budget_is_enforced():
    with BoundIndex.build(np.zeros((8, 2))) as index:
        with pytest.raises((ValueError, MemoryError)):
            index.search(np.zeros((5, 2)), k=8, max_output_bytes=16)


def test_close_is_idempotent_and_closed_search_fails():
    index = BoundIndex.build(np.zeros((8, 2)))
    index.close()
    index.close()
    with pytest.raises(ValueError):
        index.search(np.zeros((1, 2)), k=1)


def test_context_manager_closes_index():
    with BoundIndex.build(np.zeros((8, 2))) as index:
        index.search(np.zeros((1, 2)), k=1)
    with pytest.raises(ValueError):
        index.search(np.zeros((1, 2)), k=1)

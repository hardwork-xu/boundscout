"""Exact squared-L2 search and conservative block pruning. / 精确平方距离检索与块剪枝。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
MAX_DIMENSIONS = 4096
MAX_MAGNITUDE = 1e100


def _integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be an integer / 必须为整数")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum} / 数值过小")
    return int(value)


def _matrix(value: Any, name: str, *, max_bytes: int = 1 << 30) -> FloatArray:
    raw = np.asarray(value)
    if raw.ndim != 2 or not 1 <= raw.shape[1] <= MAX_DIMENSIONS:
        raise ValueError(f"{name}: expected 2D array with 1..4096 columns / 需要二维数组")
    if raw.dtype.kind not in "fiu":
        raise ValueError(f"{name}: expected real numeric values / 需要实数数值")
    if raw.size * 8 > max_bytes:
        raise ValueError(f"{name}: array exceeds memory limit / 数组超过内存限制")
    array = np.asarray(raw, dtype=np.float64, order="C")
    if not np.isfinite(array).all() or (array.size and np.max(np.abs(array)) > MAX_MAGNITUDE):
        raise ValueError(f"{name}: nonfinite or magnitude > 1e100 / 非有限数或数值过大")
    return array


def _topk(distances: FloatArray, ids: IntArray, k: int) -> tuple[FloatArray, IntArray]:
    # Keep all boundary ties before ID selection. / 保留边界等距点，再按原始 ID 取舍。
    if len(distances) > k:
        threshold = np.partition(distances, k - 1)[k - 1]
        better = np.flatnonzero(distances < threshold)
        ties = np.flatnonzero(distances == threshold)
        remaining = k - len(better)
        if len(ties) > remaining:
            ties = ties[np.argpartition(ids[ties], remaining - 1)[:remaining]]
        keep = np.concatenate((better, ties))
        distances, ids = distances[keep], ids[keep]
    order = np.lexsort((ids, distances))
    return distances[order], ids[order]


@dataclass(frozen=True)
class SearchStats:
    """Actual work counters for all queries. / 全部查询的实际工作量统计。"""

    evaluated_vectors: int
    visited_blocks: int
    total_blocks: int
    query_count: int


@dataclass(frozen=True)
class SearchResult:
    """IDs and squared distances, shape (queries, k). / ID 与平方距离，形状为查询数乘 k。"""

    indices: IntArray
    distances: FloatArray
    stats: SearchStats


class BoundIndex:
    """Immutable spatial block index; use build/load. / 不可变空间块索引，使用 build/load。"""

    def __init__(
        self,
        points: FloatArray,
        ids: IntArray,
        offsets: IntArray,
        lower: FloatArray,
        upper: FloatArray,
        leaf_size: int,
        layout: str,
    ) -> None:
        self._points = points
        self._ids = ids
        self._offsets = offsets
        self._lower = lower
        self._upper = upper
        self.leaf_size = leaf_size
        self.layout = layout
        self._closed = False
        for array in self._arrays():
            array.flags.writeable = False

    def _arrays(self) -> tuple[NDArray[Any], ...]:
        return self._points, self._ids, self._offsets, self._lower, self._upper

    @property
    def nbytes(self) -> int:
        """Logical index array bytes; not process RSS. / 索引数组字节数，不等于进程 RSS。"""
        self._check_open()
        return sum(a.nbytes for a in self._arrays())

    @property
    def shape(self) -> tuple[int, int]:
        """Database rows and dimensions. / 数据库行数与维数。"""
        self._check_open()
        return self._points.shape[0], self._points.shape[1]

    def _check_open(self) -> None:
        if self._closed:
            raise ValueError("index is closed / 索引已关闭")

    @classmethod
    def build(
        cls,
        data: Any,
        *,
        leaf_size: int = 256,
        layout: str = "spatial",
        max_bytes: int = 1 << 30,
    ) -> BoundIndex:
        """Copy real finite (N,D) data and partition median leaves. / 复制实数矩阵并按中位数分叶。

        max_bytes bounds stored index arrays, not peak build RSS. / 限制索引数组而非构建峰值 RSS。
        """
        leaf_size = _integer(leaf_size, "leaf_size")
        max_bytes = _integer(max_bytes, "max_bytes")
        if layout not in ("spatial", "input"):
            raise ValueError("layout must be spatial or input / 布局参数无效")
        points = _matrix(data, "data", max_bytes=max_bytes)
        n, d = points.shape
        if n == 0:
            raise ValueError("database must not be empty / 数据库不能为空")
        # At most twice ceil(N / leaf_size) leaves after balanced splits. / 平衡分割叶数上界。
        max_leaves = min(n, 2 * ((n + leaf_size - 1) // leaf_size))
        estimate = n * (d + 1) * 8 + max_leaves * (2 * d + 1) * 8 + 8
        if estimate > max_bytes:
            raise ValueError("index exceeds memory limit / 索引超过内存限制")
        points = points.copy()
        ids = np.arange(n, dtype=np.int64)
        leaves: list[tuple[int, int]] = []

        def split(start: int, stop: int) -> None:
            if stop - start <= leaf_size:
                leaves.append((start, stop))
                return
            subset = points[ids[start:stop]]
            axis = int(np.argmax(np.ptp(subset, axis=0)))
            middle = (stop - start) // 2
            partition = np.argpartition(subset[:, axis], middle)
            ids[start:stop] = ids[start:stop][partition]
            del subset, partition
            split(start, start + middle)
            split(start + middle, stop)

        if layout == "input":
            leaves = [(s, min(s + leaf_size, n)) for s in range(0, n, leaf_size)]
        else:
            split(0, n)
        packed = points[ids]
        lower = np.empty((len(leaves), d), dtype=np.float64)
        upper = np.empty_like(lower)
        offsets = np.array([s for s, _ in leaves] + [n], dtype=np.int64)
        for b, (start, stop) in enumerate(leaves):
            lower[b] = packed[start:stop].min(axis=0)
            upper[b] = packed[start:stop].max(axis=0)
        return cls(packed, ids, offsets, lower, upper, leaf_size, layout)

    def search(
        self,
        queries: Any,
        k: int = 10,
        *,
        prune: bool = True,
        max_output_bytes: int = 64 << 20,
    ) -> SearchResult:
        """Return top-k by squared L2 then row ID. / 按平方距离和行 ID 返回 top-k。

        prune=False scans the same layout for ablation. / 关闭剪枝用于相同布局的消融。
        """
        self._check_open()
        k = _integer(k, "k")
        max_output_bytes = _integer(max_output_bytes, "max_output_bytes")
        q = _matrix(queries, "queries")
        n, d = self.shape
        if q.shape[1] != d or k > n:
            raise ValueError("dimension mismatch or k > database size / 维数不匹配或 k 过大")
        if not isinstance(prune, bool):
            raise ValueError("prune must be bool / 剪枝开关必须为布尔值")
        if len(q) * k * 16 > max_output_bytes:
            raise ValueError("results exceed output memory limit / 结果超过输出内存限制")
        result_ids = np.empty((len(q), k), dtype=np.int64)
        result_distances = np.empty((len(q), k), dtype=np.float64)
        evaluated, visited = 0, 0
        blocks = len(self._offsets) - 1
        for row in range(len(q)):
            query: FloatArray = q[row]
            if prune:
                gap = np.maximum(np.maximum(self._lower - query, query - self._upper), 0.0)
                bounds = np.sum(gap * gap, axis=1)
                # Dimension-scaled downward guard for float64 arithmetic. / 按维数缩减浮点下界。
                bounds *= 1.0 - 64.0 * np.finfo(np.float64).eps * (d + 1)
                bounds = np.maximum(0.0, np.nextafter(bounds, -np.inf))
                order = np.argsort(bounds, kind="stable")
            else:
                bounds = np.zeros(blocks, dtype=np.float64)
                order = np.arange(blocks)
            best_d: FloatArray = np.empty(0, dtype=np.float64)
            best_i: IntArray = np.empty(0, dtype=np.int64)
            for block in order:
                # Strict > preserves ties with a smaller original ID. / 严格大于才能跳过等距小 ID。
                if len(best_d) == k and bounds[block] > np.nextafter(best_d[-1], np.inf):
                    break
                start, stop = self._offsets[block : block + 2]
                distances = cdist(query[None, :], self._points[start:stop], "sqeuclidean")[0]
                best_d, best_i = _topk(
                    np.concatenate((best_d, distances)),
                    np.concatenate((best_i, self._ids[start:stop])),
                    k,
                )
                evaluated += int(stop - start)
                visited += 1
            result_ids[row], result_distances[row] = best_i, best_d
        return SearchResult(
            result_ids,
            result_distances,
            SearchStats(
                evaluated,
                visited,
                blocks * len(q),
                len(q),
            ),
        )

    def save(self, path: str | Path) -> None:
        """Save to a new directory atomically. / 原子保存，拒绝已有目录。"""
        from .storage import save_index

        self._check_open()
        save_index(self, Path(path))

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        max_bytes: int = 1 << 30,
        verify: bool = True,
    ) -> BoundIndex:
        """Load validated read-only NPY mappings. / 加载经过验证的只读 NPY 映射。"""
        from .storage import load_index

        return load_index(Path(path), max_bytes=max_bytes, verify=verify)

    def close(self) -> None:
        """Release mappings; no concurrent close/search. / 释放映射，可重复关闭。"""
        if not self._closed:
            for array in self._arrays():
                mapping = getattr(array, "_mmap", None)
                if mapping is not None:
                    mapping.close()
            self._closed = True
            self._points = np.empty((0, 0), dtype=np.float64)
            self._ids = np.empty(0, dtype=np.int64)
            self._offsets = np.empty(0, dtype=np.int64)
            self._lower = np.empty((0, 0), dtype=np.float64)
            self._upper = np.empty((0, 0), dtype=np.float64)

    def __enter__(self) -> BoundIndex:
        self._check_open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def exhaustive_search(
    data: Any,
    queries: Any,
    k: int = 10,
    *,
    block_size: int = 256,
) -> SearchResult:
    """Bounded-distance-workspace cdist baseline. / 限制距离工作区的 cdist 穷举基线。

    Includes validation, excludes index construction. / 包含验证，不需要构建索引。
    """
    points = _matrix(data, "data")
    q = _matrix(queries, "queries")
    k = _integer(k, "k")
    block_size = _integer(block_size, "block_size")
    if len(points) == 0 or q.shape[1] != points.shape[1] or k > len(points):
        raise ValueError("invalid database, dimensions or k / 数据库、维数或 k 无效")
    if len(q) * k * 16 > 64 << 20:
        raise ValueError("results exceed output memory limit / 结果超过输出内存限制")
    out_d = np.empty((len(q), k), dtype=np.float64)
    out_i = np.empty((len(q), k), dtype=np.int64)
    for row in range(len(q)):
        query: FloatArray = q[row]
        best_d: FloatArray = np.empty(0, dtype=np.float64)
        best_i: IntArray = np.empty(0, dtype=np.int64)
        for start in range(0, len(points), block_size):
            stop = min(start + block_size, len(points))
            distances = cdist(query[None, :], points[start:stop], "sqeuclidean")[0]
            best_d, best_i = _topk(
                np.concatenate((best_d, distances)),
                np.concatenate((best_i, np.arange(start, stop, dtype=np.int64))),
                k,
            )
        out_d[row], out_i[row] = best_d, best_i
    blocks = (len(points) + block_size - 1) // block_size
    return SearchResult(
        out_i,
        out_d,
        SearchStats(
            len(points) * len(q),
            blocks * len(q),
            blocks * len(q),
            len(q),
        ),
    )

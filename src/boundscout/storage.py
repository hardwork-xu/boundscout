"""Versioned, validated NPY persistence. / 版本化、经过验证的 NPY 持久化。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .engine import BoundIndex, _integer, _matrix

NAMES = ("points", "ids", "offsets", "lower", "upper")


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_index(index: BoundIndex, path: Path) -> None:
    """Publish a complete directory after writing all arrays. / 写完所有数组后发布完整目录。"""
    if path.exists():
        raise FileExistsError("output path already exists / 输出路径已存在")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".boundscout-", dir=path.parent))
    try:
        files: dict[str, str] = {}
        for name, array in zip(NAMES, index._arrays(), strict=True):
            dest = temporary / f"{name}.npy"
            with dest.open("wb") as stream:
                np.save(stream, array, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            files[dest.name] = _digest(dest)
        metadata = {
            "format_version": 1,
            "leaf_size": index.leaf_size,
            "layout": index.layout,
            "shape": list(index.shape),
            "array_bytes": index.nbytes,
            "sha256": files,
        }
        with (temporary / "metadata.json").open("w") as stream:
            json.dump(metadata, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError("output path already exists / 输出路径已存在")
        temporary.rename(path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_index(path: Path, *, max_bytes: int = 1 << 30, verify: bool = True) -> BoundIndex:
    """Validate schema, geometry, IDs and hashes. / 验证结构、几何、ID 与校验和。"""
    max_bytes = _integer(max_bytes, "max_bytes")
    if not isinstance(verify, bool):
        raise ValueError("verify must be bool / 校验开关须为布尔值")
    arrays: list[NDArray[Any]] = []
    try:
        metadata_path = path / "metadata.json"
        if metadata_path.is_symlink() or metadata_path.stat().st_size > 65536:
            raise ValueError("invalid metadata file / 元数据文件无效")
        metadata = json.loads(metadata_path.read_text())
        if (
            not isinstance(metadata, dict)
            or type(metadata.get("format_version")) is not int
            or metadata.get("format_version") != 1
        ):
            raise ValueError("unsupported index format / 不支持的索引格式")
        hashes = metadata.get("sha256")
        if not isinstance(hashes, dict) or set(hashes) != {f"{n}.npy" for n in NAMES}:
            raise ValueError("invalid checksum metadata / 校验和元数据无效")
        leaf_size = _integer(metadata["leaf_size"], "leaf_size")
        layout = metadata["layout"]
        if layout not in ("spatial", "input"):
            raise ValueError("invalid index layout / 索引布局无效")
        paths = [path / f"{name}.npy" for name in NAMES]
        if any(p.is_symlink() or not p.is_file() for p in paths):
            raise ValueError("missing file or symlink / 缺失文件或符号链接")
        if sum(p.stat().st_size for p in paths) > max_bytes + 65536:
            raise ValueError("index exceeds memory limit / 索引超过内存限制")
        for p in paths:
            if verify and _digest(p) != hashes.get(p.name):
                raise ValueError("index checksum mismatch / 索引校验和不匹配")
            loaded = np.load(p, mmap_mode="r", allow_pickle=False)
            if not isinstance(loaded, np.ndarray):
                loaded.close()
                raise ValueError("expected NPY array / 需要 NPY 数组")
            arrays.append(loaded)
        points, ids, offsets, lower, upper = arrays
        if points.dtype != np.float64 or ids.dtype != np.int64 or offsets.dtype != np.int64:
            raise ValueError("invalid index dtype / 索引类型无效")
        if lower.dtype != np.float64 or upper.dtype != np.float64:
            raise ValueError("invalid bounds dtype / 包围盒类型无效")
        _matrix(points, "points", max_bytes=max_bytes)
        n, d = points.shape
        if n == 0 or list(points.shape) != metadata["shape"]:
            raise ValueError("invalid index shape / 索引形状无效")
        if sum(a.nbytes for a in arrays) > max_bytes:
            raise ValueError("index exceeds memory limit / 索引超过内存限制")
        if sum(a.nbytes for a in arrays) != metadata["array_bytes"]:
            raise ValueError("invalid array size metadata / 数组大小元数据无效")
        if ids.shape != (n,) or not np.array_equal(np.sort(ids), np.arange(n)):
            raise ValueError("IDs must be a permutation / ID 必须为完整排列")
        if offsets.ndim != 1 or len(offsets) < 2:
            raise ValueError("invalid offsets / 块偏移无效")
        sizes = np.diff(offsets)
        if offsets[0] != 0 or offsets[-1] != n or np.any(sizes <= 0):
            raise ValueError("invalid block ranges / 块范围无效")
        if np.any(sizes > leaf_size) or lower.shape != (len(sizes), d):
            raise ValueError("invalid block geometry / 块几何无效")
        if upper.shape != lower.shape:
            raise ValueError("invalid bounds shape / 包围盒形状无效")
        # Hashes detect corruption; geometry protects pruning. / 校验和检测损坏，几何检查保护剪枝。
        for block, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
            if not np.array_equal(lower[block], points[start:stop].min(axis=0)):
                raise ValueError("invalid lower bounds / 下界包围盒无效")
            if not np.array_equal(upper[block], points[start:stop].max(axis=0)):
                raise ValueError("invalid upper bounds / 上界包围盒无效")
        return BoundIndex(points, ids, offsets, lower, upper, leaf_size, layout)
    except (OSError, ValueError, KeyError, TypeError, OverflowError, EOFError) as error:
        for array in arrays:
            mapping = getattr(array, "_mmap", None)
            if mapping is not None:
                mapping.close()
        raise ValueError(f"cannot load index / 无法加载索引: {error}") from error

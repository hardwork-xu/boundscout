"""Bilingual CLI for local NPY vectors. / 本地 NPY 向量的双语命令行。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import NoReturn

import numpy as np

from . import BoundIndex, __version__, exhaustive_search


class BilingualParser(argparse.ArgumentParser):
    """Attach Chinese context to argparse diagnostics. / 为参数诊断补充中文提示。"""

    def error(self, message: str) -> NoReturn:
        super().error(f"{message} / 参数错误")


def main(argv: list[str] | None = None) -> int:
    """Run demo, build or query; errors exit 2. / 运行演示、建索引或查询，错误退出码为 2。"""
    parser = BilingualParser(description="Exact CPU vector search / CPU 精确向量检索")
    parser.add_argument("--version", action="version", version=__version__)
    subs = parser.add_subparsers(dest="command", required=True, title="commands / 命令")
    subs.add_parser("demo", help="run an offline complete example / 离线完整示例")
    build = subs.add_parser("build", help="create an immutable index / 创建不可变索引")
    build.add_argument("--input", required=True, type=Path, help="(N,D) NPY vectors / 向量文件")
    build.add_argument(
        "--output", required=True, type=Path, help="new index directory / 新索引目录"
    )
    build.add_argument(
        "--leaf-size", type=int, default=256, help="maximum leaf rows / 叶块最大行数"
    )
    build.add_argument(
        "--max-bytes", type=int, default=1 << 30, help="index byte limit / 索引字节限制"
    )
    build.add_argument(
        "--layout", choices=["spatial", "input"], default="spatial", help="block layout / 分块布局"
    )
    query = subs.add_parser("query", help="search a saved index / 查询已保存索引")
    query.add_argument("--index", required=True, type=Path, help="index directory / 索引目录")
    query.add_argument("--input", required=True, type=Path, help="(Q,D) query NPY / 查询文件")
    query.add_argument(
        "--output", required=True, type=Path, help="new NPZ result / 新 NPZ 结果文件"
    )
    query.add_argument("--k", type=int, default=10, help="neighbors per query / 每次查询邻居数")
    query.add_argument("--no-prune", action="store_true", help="scan every block / 扫描全部块")
    try:
        args = parser.parse_args(argv)
        if args.command == "demo":
            rng = np.random.default_rng(20260921)
            centers = rng.normal(size=(8, 16)) * 8
            points = centers[rng.integers(8, size=4096)] + rng.normal(size=(4096, 16)) * 0.5
            queries = centers[:4] + rng.normal(size=(4, 16)) * 0.5
            with BoundIndex.build(points, leaf_size=128) as index:
                actual = index.search(queries, k=10)
                reference = exhaustive_search(points, queries, k=10)
                ok = bool(
                    np.array_equal(actual.indices, reference.indices)
                    and np.allclose(actual.distances, reference.distances, rtol=1e-12, atol=1e-12)
                )
                print(
                    json.dumps(
                        {
                            "ok": ok,
                            "shape": list(actual.indices.shape),
                            "max_abs_error": float(
                                np.max(np.abs(actual.distances - reference.distances))
                            ),
                            "stats": asdict(actual.stats),
                            "index_array_bytes": index.nbytes,
                        }
                    )
                )
                return 0 if ok else 1
        if args.command == "build":
            points = np.load(args.input, mmap_mode="r", allow_pickle=False)
            with BoundIndex.build(
                points, leaf_size=args.leaf_size, layout=args.layout, max_bytes=args.max_bytes
            ) as index:
                index.save(args.output)
                print(
                    json.dumps(
                        {"ok": True, "shape": list(index.shape), "index_array_bytes": index.nbytes}
                    )
                )
        else:
            if args.output.exists():
                raise FileExistsError("output already exists / 输出已存在")
            queries = np.load(args.input, mmap_mode="r", allow_pickle=False)
            with BoundIndex.load(args.index) as index:
                result = index.search(queries, k=args.k, prune=not args.no_prune)
                # Exclusive creation prevents overwrites. / 独占创建防止覆盖。
                with args.output.open("xb") as stream:
                    np.savez(stream, indices=result.indices, distances=result.distances)
                print(json.dumps({"ok": True, "stats": asdict(result.stats)}))
        return 0
    except (OSError, ValueError, TypeError) as error:
        print(f"error / 错误: {error}", file=sys.stderr)
        return 2

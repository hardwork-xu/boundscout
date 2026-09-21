"""Auditable CPU benchmark; 可审计的 CPU 基准实验。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_info, threadpool_limits

from boundscout import BoundIndex, __version__

ROOT = Path(__file__).resolve().parent
METHOD_NAMES = (
    "cdist-256",
    "cdist-4096",
    "ckdtree",
    "spatial-pruned",
    "spatial-unpruned",
    "input-pruned",
)


@dataclass
class Output:
    """Comparable squared-distance outputs; 可比较的平方距离结果。"""

    indices: NDArray[np.int64]
    distances: NDArray[np.float64]
    evaluated_vectors: int | None
    visited_blocks: int | None = None
    total_blocks: int | None = None


def validate_queries(queries: NDArray[np.float64], dimensions: int) -> NDArray[np.float64]:
    """Match the public query safety checks; 对齐公共查询输入校验。"""
    array = np.asarray(queries, dtype=np.float64, order="C")
    if array.ndim != 2 or array.shape[1] != dimensions or array.nbytes > 1 << 30:
        raise ValueError("invalid query shape or size / 查询形状或大小无效")
    if not np.isfinite(array).all() or (array.size and np.max(np.abs(array)) > 1e100):
        raise ValueError("invalid query values / 查询数值无效")
    return array


def cdist_search(
    data: NDArray[np.float64], queries: NDArray[np.float64], k: int, block_size: int
) -> Output:
    """Compiled, bounded-memory exhaustive scan; 编译距离核与有界内存穷举。

    Threshold selection retains all equal-distance candidates before row-ID sorting.
    阈值选择保留距离相等的候选，随后按原始行号打破平局。
    """
    queries = validate_queries(queries, data.shape[1])
    if block_size < 1 or not 1 <= k <= len(data):
        raise ValueError("invalid k or block_size / k 或块大小无效")
    best_ids = np.empty((len(queries), 0), dtype=np.int64)
    best_dist = np.empty((len(queries), 0), dtype=np.float64)
    for start in range(0, len(data), block_size):
        stop = min(start + block_size, len(data))
        distances = cdist(queries, data[start:stop], metric="sqeuclidean")
        candidate_ids = np.concatenate(
            (best_ids, np.broadcast_to(np.arange(start, stop, dtype=np.int64), distances.shape)),
            axis=1,
        )
        candidate_dist = np.concatenate((best_dist, distances), axis=1)
        keep = min(k, candidate_dist.shape[1])
        best_ids = np.empty((len(queries), keep), dtype=np.int64)
        best_dist = np.empty((len(queries), keep), dtype=np.float64)
        for row in range(len(queries)):
            threshold = np.partition(candidate_dist[row], keep - 1)[keep - 1]
            selected = np.flatnonzero(candidate_dist[row] <= threshold)
            order = np.lexsort((candidate_ids[row, selected], candidate_dist[row, selected]))
            selected = selected[order[:keep]]
            best_ids[row] = candidate_ids[row, selected]
            best_dist[row] = candidate_dist[row, selected]
    return Output(best_ids, best_dist, len(data) * len(queries))


def file_sha256(path: Path) -> str:
    """Streaming file digest; 流式文件校验。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint(root: Path = ROOT) -> dict[str, Any]:
    """Fingerprint measured source without private paths; 记录源码且不含私人路径。"""
    paths = [root / "benchmark.py", root / "pyproject.toml"]
    for directory in ("src", "configs", "scripts"):
        if (root / directory).exists():
            paths.extend(
                path
                for path in (root / directory).rglob("*")
                if path.suffix in {".py", ".json"} and "__pycache__" not in path.parts
            )
    manifest = {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(set(paths))
        if path.is_file()
    }
    revision: str | None = None
    dirty: bool | None = None
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "git_revision": revision,
        "git_dirty": dirty,
        "sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "files_sha256": manifest,
    }


def machine_info() -> dict[str, Any]:
    """Sanitized hardware/software metadata; 去除身份与路径的环境记录。"""
    versions: dict[str, str | None] = {"boundscout": __version__}
    for name in ("numpy", "scipy", "threadpoolctl", "h5py"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        try:
            cpu = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    memory: int | None = None
    try:
        memory = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        pass
    pools = [
        {key: pool.get(key) for key in ("internal_api", "user_api", "num_threads", "version")}
        for pool in threadpool_info()
    ]
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "cpu": cpu,
        "logical_cpu_count": os.cpu_count(),
        "physical_memory_bytes": memory,
        "python": platform.python_version(),
        "dependencies": versions,
        "threadpools": pools,
    }


def load_config(path: Path) -> dict[str, Any]:
    """Validate the frozen workload matrix; 校验冻结的实验配置。"""
    config: dict[str, Any] = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("unsupported config schema / 不支持的配置版本")
    for key in ("repeats", "leaf_size"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer / 必须为正整数")
    if config.get("warmup") != 1 or config.get("native_threads") != 1:
        raise ValueError("one warmup and one native thread required / 要求一次预热和单线程")
    workloads = config.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        raise ValueError("workloads must be nonempty / 工作负载不能为空")
    ids: set[str] = set()
    for workload in workloads:
        if workload["id"] in ids:
            raise ValueError("duplicate workload id / 工作负载 ID 重复")
        ids.add(workload["id"])
        if workload["kind"] not in {"clustered", "isotropic", "glove"}:
            raise ValueError("unknown dataset kind / 未知数据类型")
        for key in ("n", "dimensions", "queries", "k"):
            if type(workload.get(key)) is not int or workload[key] < 1:
                raise ValueError(f"{key} must be positive / 必须为正数")
        if workload["k"] > workload["n"]:
            raise ValueError("k exceeds n / k 超过数据行数")
        if not workload.get("seeds") or not all(
            type(seed) is int and seed >= 0 for seed in workload["seeds"]
        ):
            raise ValueError("seeds must be nonnegative integers / 种子须为非负整数")
    return config


def prepare_data(
    spec: dict[str, Any], seed: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
    """Create independent held-out queries; 生成独立查询或采样真实测试集。"""
    rng = np.random.default_rng(seed)
    n, d, q = spec["n"], spec["dimensions"], spec["queries"]
    info: dict[str, Any] = {"kind": spec["kind"], "seed": seed, "dtype": "float64"}
    if spec["kind"] == "clustered":
        centers = rng.normal(0, spec["center_scale"], (spec["clusters"], d))
        data = centers[rng.integers(0, len(centers), n)] + rng.normal(
            0, spec["cluster_std"], (n, d)
        )
        queries = centers[rng.integers(0, len(centers), q)] + rng.normal(
            0, spec["cluster_std"], (q, d)
        )
        info["generator"] = "numpy.default_rng.normal; independent data/query cluster labels"
    elif spec["kind"] == "isotropic":
        data = rng.normal(size=(n, d))
        queries = rng.normal(size=(q, d))
        info["generator"] = "numpy.default_rng.normal; independent data and queries"
    else:
        import h5py

        path = Path(spec["dataset_path"])
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(
                "GloVe file missing; run python scripts/download_data.py / GloVe 文件缺失，请先下载"
            )
        metadata_path = path.with_suffix(path.suffix + ".metadata.json")
        if not metadata_path.is_file():
            raise FileNotFoundError("dataset checksum metadata missing / 数据校验元数据缺失")
        metadata = json.loads(metadata_path.read_text())
        digest = file_sha256(path)
        if metadata["sha256"] != digest:
            raise ValueError("dataset SHA256 mismatch / 数据文件校验失败")
        with h5py.File(path, "r") as handle:
            train, test = handle["train"], handle["test"]
            if train.shape[1] != d or test.shape[1] != d:
                raise ValueError("dataset dimensions mismatch / 数据维数不匹配")
            train_ids = np.sort(rng.choice(len(train), n, replace=False))
            test_ids = np.sort(rng.choice(len(test), q, replace=False))
            # Chunked reads avoid HDF5's quadratic large fancy-index selection.
            # 分块读取避免 HDF5 大规模花式索引的二次开销。
            data = np.empty((n, d), dtype=np.float64)
            for start in range(0, len(train), 65536):
                lo, hi = np.searchsorted(train_ids, [start, start + 65536])
                if hi > lo:
                    block = np.asarray(train[start : start + 65536])
                    data[lo:hi] = block[train_ids[lo:hi] - start]
            queries = np.asarray(test[test_ids], dtype=np.float64)
        # Unit vectors turn angular ranking into squared-L2 ranking.
        # 单位向量使角距离排序与平方欧氏距离排序一致。
        data_norm = np.linalg.norm(data, axis=1, keepdims=True)
        query_norm = np.linalg.norm(queries, axis=1, keepdims=True)
        if np.any(data_norm == 0) or np.any(query_norm == 0):
            raise ValueError("zero norm in real data / 真实数据包含零范数向量")
        data /= data_norm
        queries /= query_norm
        info.update(
            {
                "source_url": metadata["source_url"],
                "file_sha256": digest,
                "normalization": "l2-float64",
                "train_rows": train_ids.tolist(),
                "test_rows": test_ids.tolist(),
                "license": metadata["license"],
            }
        )
    data = np.ascontiguousarray(data, dtype=np.float64)
    queries = np.ascontiguousarray(queries, dtype=np.float64)
    info["data_sha256"] = hashlib.sha256(data.tobytes()).hexdigest()
    info["queries_sha256"] = hashlib.sha256(queries.tobytes()).hexdigest()
    return data, queries, info


@dataclass
class Method:
    name: str
    search: Callable[[NDArray[np.float64]], Output]
    build_seconds: float
    index_bytes: int | None
    index: BoundIndex | None = None


def make_method(name: str, data: NDArray[np.float64], k: int, leaf_size: int) -> Method:
    """Construct one separately timed implementation; 独立计时构建各实现。"""
    started = time.perf_counter()
    if name.startswith("cdist-"):
        block = int(name.split("-")[1])
        return Method(name, lambda queries: cdist_search(data, queries, k, block), 0.0, 0)
    if name == "ckdtree":
        tree = cKDTree(data, leafsize=leaf_size)
        build_seconds = time.perf_counter() - started

        def tree_search(queries: NDArray[np.float64]) -> Output:
            queries = validate_queries(queries, data.shape[1])
            distances, ids = tree.query(queries, k=k, eps=0, workers=1)
            return Output(
                np.asarray(ids, dtype=np.int64).reshape(len(queries), k),
                np.asarray(distances, dtype=np.float64).reshape(len(queries), k) ** 2,
                None,
            )

        # SciPy exposes no complete array-byte accounting; do not guess.
        # SciPy 没有公开完整索引字节统计，保留空值。
        return Method(name, tree_search, build_seconds, None)
    index = BoundIndex.build(
        data, leaf_size=leaf_size, layout="input" if name == "input-pruned" else "spatial"
    )
    build_seconds = time.perf_counter() - started

    def index_search(queries: NDArray[np.float64]) -> Output:
        result = index.search(queries, k=k, prune=name != "spatial-unpruned")
        return Output(
            result.indices,
            result.distances,
            int(result.stats.evaluated_vectors),
            int(result.stats.visited_blocks),
            int(result.stats.total_blocks),
        )

    return Method(name, index_search, build_seconds, index.nbytes, index)


def correctness(actual: Output, expected: Output, *, ids_required: bool) -> dict[str, Any]:
    """Check every full workload, preserving cKDTree tie semantics; 校验完整工作负载。"""
    ids_equal = bool(np.array_equal(actual.indices, expected.indices))
    distances_equal = bool(
        np.allclose(actual.distances, expected.distances, rtol=1e-12, atol=1e-12)
    )
    return {
        "passed": distances_equal and (ids_equal or not ids_required),
        "ordered_ids_equal": ids_equal,
        "ids_required": ids_required,
        "squared_distances_allclose": distances_equal,
        "rtol": 1e-12,
        "atol": 1e-12,
        "max_absolute_distance_error": float(np.max(np.abs(actual.distances - expected.distances))),
    }


def safe_error(error: Exception) -> str:
    """Remove local identity-bearing paths from recorded exceptions; 清除异常中的私人路径。"""
    message = str(error).replace(str(ROOT), "<project>")
    message = message.replace(str(Path.home()), "<home>")
    return f"{type(error).__name__}: {message[:1000]}"


def isolated_rss(spec: dict[str, Any], seed: int, leaf_size: int, name: str) -> dict[str, Any]:
    """Measure a fresh interpreter peak, including setup; 测量包含初始化的独立进程峰值。"""
    payload = {"spec": spec, "seed": seed, "leaf_size": leaf_size, "method": name}
    completed = subprocess.run(
        [sys.executable, str(ROOT / "benchmark.py"), "--rss-worker"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
        timeout=300,
        cwd=ROOT,
    )
    result: dict[str, Any] = json.loads(completed.stdout)
    return result


def rss_worker() -> None:
    payload = json.load(sys.stdin)
    with threadpool_limits(limits=1):
        data, queries, _ = prepare_data(payload["spec"], payload["seed"])
        method = make_method(payload["method"], data, payload["spec"]["k"], payload["leaf_size"])
        method.search(queries)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        json.dumps(
            {
                "status": "ok",
                "isolated_process_peak_rss_bytes": int(
                    peak if sys.platform == "darwin" else peak * 1024
                ),
                "scope": (
                    "fresh interpreter, imports, data preparation, method build and one query batch"
                ),
                "method": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
            }
        )
    )


def run_workload(
    spec: dict[str, Any], seed: int, config: dict[str, Any], measure_rss: bool
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "workload_id": spec["id"],
        "seed": seed,
        "spec": spec,
        "status": "running",
        "methods": {},
    }
    started = time.perf_counter()
    try:
        data, queries, dataset = prepare_data(spec, seed)
        record["preparation_seconds"] = time.perf_counter() - started
        record["dataset"] = dataset
        expected = cdist_search(data, queries, spec["k"], 4096)
    except Exception as error:
        record.update(status="failed", error=safe_error(error))
        return record
    methods: dict[str, Method] = {}
    for name in METHOD_NAMES:
        item: dict[str, Any] = {"status": "running", "samples_seconds": []}
        record["methods"][name] = item
        try:
            method = make_method(name, data, spec["k"], config["leaf_size"])
            methods[name] = method
            item.update(build_seconds=method.build_seconds, index_array_bytes=method.index_bytes)
            before = time.perf_counter()
            method.search(queries[:1])
            item["first_single_query_seconds"] = time.perf_counter() - before
            # Exactly one full-batch warmup; correctness is checked on its output.
            # 恰好一次完整批次预热，并使用其结果检查正确性。
            actual = method.search(queries)
            item["correctness"] = correctness(actual, expected, ids_required=name != "ckdtree")
            item["evaluated_vectors"] = actual.evaluated_vectors
            item["visited_blocks"] = actual.visited_blocks
            item["total_blocks"] = actual.total_blocks
            if not item["correctness"]["passed"]:
                raise AssertionError("correctness comparison failed / 正确性校验失败")
            item["status"] = "ready"
        except Exception as error:
            item.update(status="failed", error=safe_error(error))
    order_rng = np.random.default_rng(config["order_seed"] + seed)
    orders: list[list[str]] = []
    for _ in range(config["repeats"]):
        order = list(order_rng.permutation(list(METHOD_NAMES)))
        orders.append(order)
        for name in order:
            item = record["methods"][name]
            if item["status"] == "failed":
                continue
            try:
                before = time.perf_counter()
                actual = methods[name].search(queries)
                elapsed = time.perf_counter() - before
                # Keep the sample even if a later check or repetition fails.
                # 即使后续校验或重复失败，仍保留已测得的样本。
                item["samples_seconds"].append(elapsed)
                check = correctness(actual, expected, ids_required=name != "ckdtree")
                if not check["passed"]:
                    item["correctness"] = check
                    raise AssertionError("timed result differs from reference / 计时结果偏离参考")
            except Exception as error:
                item.update(status="failed", error=safe_error(error))
    record["measurement_orders"] = orders
    for name, item in record["methods"].items():
        if item["status"] == "failed":
            continue
        item["status"] = "ok"
        samples = np.asarray(item["samples_seconds"], dtype=np.float64)
        median = float(np.median(samples))
        item.update(
            median_seconds=median,
            min_seconds=float(np.min(samples)),
            max_seconds=float(np.max(samples)),
            queries_per_second=len(queries) / median,
        )
        method = methods[name]
        if method.index is not None:
            try:
                with tempfile.TemporaryDirectory(prefix="boundscout-storage-") as directory:
                    path = Path(directory) / "index"
                    before = time.perf_counter()
                    method.index.save(path)
                    item["save_seconds"] = time.perf_counter() - before
                    item["saved_file_bytes"] = sum(
                        member.stat().st_size for member in path.rglob("*") if member.is_file()
                    )
                    before = time.perf_counter()
                    restored = BoundIndex.load(path)
                    item["load_seconds"] = time.perf_counter() - before
                    try:
                        restored_result = restored.search(queries, k=spec["k"])
                        storage_output = Output(
                            restored_result.indices, restored_result.distances, None
                        )
                        item["storage_correctness"] = correctness(
                            storage_output, expected, ids_required=True
                        )
                        if not item["storage_correctness"]["passed"]:
                            raise AssertionError("storage roundtrip differs / 存储往返结果不一致")
                    finally:
                        restored.close()
            except Exception as error:
                item["storage_status"] = "failed"
                item["storage_error"] = safe_error(error)
            else:
                item["storage_status"] = "ok"
        if measure_rss:
            try:
                item["memory"] = isolated_rss(spec, seed, config["leaf_size"], name)
            except Exception as error:
                item["memory"] = {"status": "failed", "error": safe_error(error)}
        else:
            item["memory"] = {"status": "not_run", "reason": "--measure-rss not requested"}
    for method in methods.values():
        if method.index is not None:
            method.index.close()
    record["status"] = (
        "ok"
        if all(
            item["status"] == "ok"
            and item.get("storage_status", "ok") == "ok"
            and item["memory"]["status"] != "failed"
            for item in record["methods"].values()
        )
        else "partial_failure"
    )
    return record


def validate_raw(raw: dict[str, Any]) -> None:
    """Portable raw-evidence validation; 可移植的原始证据格式校验。"""
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported raw schema / 不支持的结果格式")
    for key in (
        "run_id",
        "started_at_utc",
        "finished_at_utc",
        "command",
        "config",
        "source",
        "machine",
    ):
        if key not in raw:
            raise ValueError(f"missing field / 缺失字段: {key}")
    uuid.UUID(raw["run_id"])
    for key in ("started_at_utc", "finished_at_utc"):
        if datetime.fromisoformat(raw[key]).tzinfo is None:
            raise ValueError("UTC-aware timestamps required / 时间须包含时区")
    if len(raw["source"]["sha256"]) != 64:
        raise ValueError("invalid source digest / 源码摘要无效")
    if not isinstance(raw.get("workloads"), list) or not raw["workloads"]:
        raise ValueError("missing workload results / 缺少工作负载结果")
    for workload in raw["workloads"]:
        if workload.get("status") not in {"ok", "failed", "partial_failure"}:
            raise ValueError("invalid workload status / 工作负载状态无效")
        if workload["status"] == "failed":
            if not workload.get("error"):
                raise ValueError("failure needs an error / 失败须记录原因")
            continue
        for name in METHOD_NAMES:
            method = workload["methods"][name]
            samples = method["samples_seconds"]
            if not all(
                isinstance(value, int | float) and math.isfinite(value) and value > 0
                for value in samples
            ):
                raise ValueError("invalid timing samples / 时间样本无效")
            if method["status"] == "ok":
                if len(samples) != raw["config"]["repeats"]:
                    raise ValueError("incomplete repetitions / 重复次数不足")
                if not method["correctness"]["passed"]:
                    raise ValueError("successful method lacks correctness / 成功方法缺少正确性验证")
            elif method["status"] != "failed" or not method.get("error"):
                raise ValueError("invalid method status / 方法状态无效")


def run_benchmark(config: dict[str, Any], *, measure_rss: bool = False) -> dict[str, Any]:
    """Run the frozen experiment; 执行冻结的实验。"""
    raw: dict[str, Any] = {
        "schema_version": 1,
        "run_id": str(uuid.uuid4()),
        "started_at_utc": datetime.now(UTC).isoformat(),
        "command": ["python", "benchmark.py"],
        "config": config,
        "source": source_fingerprint(),
        "workloads": [],
        "timing_scope": "steady-state query batches; build, preparation and serialization excluded",
        "correctness_reference": (
            "independent scipy cdist sqeuclidean, block 4096, (distance,row ID) ordering"
        ),
        "memory_scope": (
            "index_array_bytes is logical; "
            "isolated_process_peak_rss_bytes includes imports and setup"
        ),
    }
    with threadpool_limits(limits=config["native_threads"]):
        raw["machine"] = machine_info()
        for spec in config["workloads"]:
            for seed in spec["seeds"]:
                raw["workloads"].append(run_workload(spec, seed, config, measure_rss))
    raw["finished_at_utc"] = datetime.now(UTC).isoformat()
    raw["status"] = (
        "ok" if all(item["status"] == "ok" for item in raw["workloads"]) else "partial_failure"
    )
    validate_raw(raw)
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproducible exact-search benchmark / 可复现精确检索实验"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/smoke.json"),
        help="Frozen JSON config / 冻结配置",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/raw/smoke.json"),
        help="Raw evidence JSON / 原始证据",
    )
    parser.add_argument(
        "--measure-rss", action="store_true", help="Measure isolated process RSS / 测量独立进程内存"
    )
    parser.add_argument("--rss-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.rss_worker:
        rss_worker()
        return 0
    if args.output.exists():
        parser.error(
            "output exists; choose a new evidence filename / 输出已存在，请选择新文件名以保留证据"
        )
    config = load_config(args.config)
    raw = run_benchmark(config, measure_rss=args.measure_rss)

    def public_path(path: Path) -> str:
        try:
            return path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            return path.name

    raw["command"] = [
        "python",
        "benchmark.py",
        "--config",
        public_path(args.config),
        "--output",
        public_path(args.output),
    ]
    if args.measure_rss:
        raw["command"].append("--measure-rss")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(raw, indent=2, allow_nan=False) + "\n")
    print(f"{raw['status']}: {public_path(args.output)} / 原始结果已保存")
    return 0 if raw["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

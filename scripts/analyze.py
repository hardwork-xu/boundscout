"""Generate bilingual tables and plots solely from raw runs; 仅由原始运行生成双语表图。"""

# Long bilingual report paragraphs intentionally remain single literals.
# ruff: noqa: E501
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = (
    "cdist-256",
    "cdist-4096",
    "ckdtree",
    "spatial-pruned",
    "spatial-unpruned",
    "input-pruned",
)
COLORS = ("#718096", "#394b59", "#c26d42", "#178074", "#74aab2", "#bc9e44")


def number(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    return (
        None
        if numerator is None or denominator is None or denominator <= 0
        else numerator / denominator
    )


def summarize(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Derive metrics without discarding failures; 派生指标并保留失败。"""
    rows: list[dict[str, Any]] = []
    for workload in raw["workloads"]:
        spec = workload["spec"]
        baselines = workload.get("methods", {})
        practical = baselines.get("cdist-256", {}).get("median_seconds")
        large = baselines.get("cdist-4096", {}).get("median_seconds")
        if workload["status"] == "failed":
            rows.append(
                {
                    "run_id": raw["run_id"],
                    "workload_id": workload["workload_id"],
                    "seed": workload["seed"],
                    "status": "failed",
                    "error": workload["error"],
                }
            )
            continue
        for name in METHODS:
            item = baselines[name]
            row: dict[str, Any] = {
                "run_id": raw["run_id"],
                "workload_id": workload["workload_id"],
                "seed": workload["seed"],
                "method": name,
                "status": item["status"],
                "kind": spec["kind"],
                "n": spec["n"],
                "dimensions": spec["dimensions"],
                "queries": spec["queries"],
                "k": spec["k"],
            }
            if item["status"] != "ok":
                row.update(error=item.get("error"), samples_seconds=item["samples_seconds"])
                rows.append(row)
                continue
            samples = item["samples_seconds"]
            median = float(np.median(samples))
            evaluated = item.get("evaluated_vectors")
            saved_seconds = None if practical is None else practical - median
            build = item["build_seconds"]
            reduction = None if evaluated is None else 1 - evaluated / (spec["n"] * spec["queries"])
            row.update(
                {
                    "samples_seconds": samples,
                    "median_seconds": median,
                    "min_seconds": min(samples),
                    "max_seconds": max(samples),
                    "speedup_vs_cdist256": ratio(practical, median),
                    "speedup_vs_cdist4096": ratio(large, median),
                    "queries_per_second": spec["queries"] / median,
                    "evaluated_vectors": evaluated,
                    "evaluated_vector_reduction": reduction,
                    "build_seconds": build,
                    "first_single_query_seconds": item["first_single_query_seconds"],
                    "index_array_bytes": item["index_array_bytes"],
                    "isolated_process_peak_rss_bytes": item.get("memory", {}).get(
                        "isolated_process_peak_rss_bytes"
                    ),
                    "memory_status": item.get("memory", {}).get("status", "not_run"),
                    "save_seconds": item.get("save_seconds"),
                    "load_seconds": item.get("load_seconds"),
                    "storage_status": item.get("storage_status"),
                    "break_even_batches_vs_cdist256": None
                    if saved_seconds is None or saved_seconds <= 0
                    else math.ceil(build / saved_seconds),
                    "correctness": item["correctness"],
                }
            )
            target = raw["config"].get("primary_target", {})
            if workload["workload_id"] == target.get("workload_id") and name == target.get(
                "method"
            ):
                row["preregistered_target"] = {
                    "minimum_speedup": target["minimum_speedup"],
                    "minimum_evaluated_vector_reduction": target[
                        "minimum_evaluated_vector_reduction"
                    ],
                    "met": row["speedup_vs_cdist256"] is not None
                    and row["speedup_vs_cdist256"] >= target["minimum_speedup"]
                    and reduction is not None
                    and reduction >= target["minimum_evaluated_vector_reduction"],
                }
            rows.append(row)
    return rows


def render_report(runs: list[dict[str, Any]], rows: list[dict[str, Any]], *, chinese: bool) -> str:
    """Both language versions share numeric derivation; 两种语言共用数值推导。"""
    if chinese:
        lines = [
            "# 自动生成的实测结果",
            "",
            "由 `python scripts/analyze.py` 从原始 JSON 生成。下表保留所有输入运行与有效样本；失败另行列出。",
            "目标与实测分开：预注册主要目标为 clustered-primary 对 cdist-256 至少 1.50× 加速，且向量计算量至少减少 75%。每个独立种子单独判断。",
            "完整批次计时排除数据准备、构建和序列化。首次查询列是构建后的单个查询，不是冷缓存或应用端到端延迟。七次重复的最小/最大值不构成 p99 或置信区间。",
            "",
            "## 运行来源",
            "",
        ]
    else:
        lines = [
            "# Generated measured results",
            "",
            "Generated by `python scripts/analyze.py` from raw JSON. All supplied runs and valid samples are retained; failures are listed separately.",
            "Target is separate from measurement: the preregistered clustered-primary target is at least 1.50× speedup versus cdist-256 and at least 75% fewer vector evaluations. Each independent seed is judged separately.",
            "Batch timing excludes preparation, build and serialization. First query means one query immediately after build, not a cold-cache or application end-to-end claim. Min/max from seven repetitions are neither p99 nor confidence intervals.",
            "",
            "## Run provenance",
            "",
        ]
    for run in runs:
        machine = run["machine"]
        lines += [
            f"- `{run['run_id']}` · `{run['config']['name']}` · {run['started_at_utc']}",
            f"  - {machine['os']} {machine['architecture']} · {machine['cpu']} · Python {machine['python']} · native threads = {run['config']['native_threads']}",
            f"  - Git `{run['source'].get('git_revision')}`; dirty = `{run['source'].get('git_dirty')}`; source SHA256 `{run['source']['sha256']}`.",
            f"  - Raw / 原始证据: [JSON](../../{run['_input_path']}); command / 命令: `{' '.join(run['command'])}`.",
            f"  - Dependencies / 依赖: `{json.dumps(machine['dependencies'], sort_keys=True)}`.",
        ]
    lines += ["", "## 主要目标对照" if chinese else "## Primary target comparison", ""]
    targets = [row for row in rows if "preregistered_target" in row]
    if not targets:
        lines += [
            "本批输入未包含主要目标的有效测量。"
            if chinese
            else "The supplied runs contain no valid primary-target measurement."
        ]
    else:
        lines += [
            "| Run | Seed | Target speedup | Measured speedup | Target reduction | Measured reduction | Both met |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for row in targets:
            target = row["preregistered_target"]
            lines.append(
                f"| {row['run_id'][:8]} | {row['seed']} | {target['minimum_speedup']:.2f}× | "
                f"{row['speedup_vs_cdist256']:.2f}× | {target['minimum_evaluated_vector_reduction']:.0%} | "
                f"{row['evaluated_vector_reduction']:.2%} | {target['met']} |"
            )
    lines += [
        "",
        "## 完整稳态结果" if chinese else "## Complete steady-state results",
        "",
        "每行是一个运行、工作负载、种子和方法。加速比是同一输入的中位耗时之比，低于 1 表示更慢。"
        if chinese
        else "Each row is one run, workload, seed and method. Speedup is the ratio of medians on identical input; below 1 means slower.",
        "",
        "| Run/workload/seed | N/d/Q/k | Method | Median [min,max] ms | Query/s | × vs 256 | × vs 4096 | Fewer vectors |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["status"] != "ok":
            continue
        label = f"{row['run_id'][:8]}/{row['workload_id']}/{row['seed']}"
        reduction = row["evaluated_vector_reduction"]
        reduction_text = "—" if reduction is None else f"{reduction:.2%}"
        lines.append(
            f"| {label} | {row['n']}/{row['dimensions']}/{row['queries']}/{row['k']} | {row['method']} | "
            f"{row['median_seconds'] * 1000:.3f} [{row['min_seconds'] * 1000:.3f},{row['max_seconds'] * 1000:.3f}] | "
            f"{row['queries_per_second']:.1f} | {number(row['speedup_vs_cdist256'])} | "
            f"{number(row['speedup_vs_cdist4096'])} | {reduction_text} |"
        )
    lines += [
        "",
        "## 构建、首次查询与内存" if chinese else "## Build, first query and memory",
        "",
        "索引 MiB 是数组逻辑字节数；RSS MiB 是独立新进程的实测峰值，包含解释器、导入、数据准备、构建和一个查询批次。两者不能相减或视作同一统计口径。cKDTree 完整逻辑字节数未公开，保留空值。"
        if chinese
        else "Index MiB is logical array bytes. RSS MiB is the measured peak of a fresh isolated process, including interpreter, imports, data preparation, build and one query batch. These are different scopes and must not be subtracted. Complete cKDTree logical bytes are unavailable and remain blank.",
        "回本批次数 = ceil(构建耗时 / (基线批次耗时 − 方法批次耗时))；没有稳态收益时记为 —。每批次 Q 次查询。输入数组仍存活；BoundIndex 自有数据副本，cdist 借用输入。"
        if chinese
        else "Break-even batches = ceil(build time / (baseline batch time − method batch time)); no steady-state benefit is shown as —. Each batch has Q queries. Input arrays remain alive: BoundIndex owns a copy, while cdist borrows input.",
        "",
        "| Run/workload/seed | Method | Build ms | First query µs | Save/load ms | Index MiB | Peak RSS MiB | Break-even batches |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["status"] != "ok":
            continue
        label = f"{row['run_id'][:8]}/{row['workload_id']}/{row['seed']}"
        saved = None if row["save_seconds"] is None else row["save_seconds"] * 1000
        loaded = None if row["load_seconds"] is None else row["load_seconds"] * 1000
        index = None if row["index_array_bytes"] is None else row["index_array_bytes"] / 2**20
        rss = (
            None
            if row["isolated_process_peak_rss_bytes"] is None
            else row["isolated_process_peak_rss_bytes"] / 2**20
        )
        break_even = row["break_even_batches_vs_cdist256"]
        lines.append(
            f"| {label} | {row['method']} | {row['build_seconds'] * 1000:.3f} | "
            f"{row['first_single_query_seconds'] * 1e6:.1f} | {number(saved)}/{number(loaded)} | "
            f"{number(index)} | {number(rss)} | {'—' if break_even is None else break_even} |"
        )
    lines += ["", "## 正确性与失败记录" if chinese else "## Correctness and failures", ""]
    valid = [row for row in rows if row["status"] == "ok"]
    lines.append(
        f"有效方法/工作负载/种子记录数：{len(valid)}；完整批次输出均检查平方距离，BoundIndex/cdist 还检查有序 ID。cKDTree 允许等距离 ID 顺序不同。原始数据保存每次重复，失败前已产生的计时样本仍保留。"
        if chinese
        else f"Valid method/workload/seed records: {len(valid)}. Full-batch squared distances were checked for every method; ordered IDs were also checked for BoundIndex/cdist. cKDTree may order equal-distance IDs differently. Raw evidence retains every repetition, including samples recorded before failures."
    )
    failures: list[str] = []
    for run in runs:
        for workload in run["workloads"]:
            label = f"{run['run_id'][:8]}/{workload['workload_id']}/{workload['seed']}"
            if workload["status"] == "failed":
                failures.append(f"- {label}: {workload['error']}")
            for name, method in workload.get("methods", {}).items():
                if method["status"] == "failed":
                    failures.append(f"- {label}/{name}: {method['error']}")
                if method.get("storage_status") == "failed":
                    failures.append(f"- {label}/{name}/storage: {method['storage_error']}")
                memory = method.get("memory", {})
                if memory.get("status") == "failed":
                    failures.append(f"- {label}/{name}/RSS: {memory['error']}")
    lines += [""] + (
        failures
        if failures
        else [
            "输入运行没有记录失败。"
            if chinese
            else "No failures were recorded in the supplied runs."
        ]
    )
    unmeasured = sum(row.get("memory_status") == "not_run" for row in valid)
    if unmeasured:
        lines.append(
            f"RSS 未执行记录：{unmeasured}。"
            if chinese
            else f"RSS was not run for {unmeasured} records."
        )
    lines += [
        "",
        "## 解释边界" if chinese else "## Interpretation limits",
        "",
        "两个 cdist 基线共享独立于核心引擎的 SciPy 编译距离核与精确 top-k 选择，只改变块大小。大块基线用于披露 Python 小块循环开销。所有方法均为 float64、单原生线程、相同数据与查询，包含查询合法性校验。cdist 同时处理整个查询批次，距离临时数组为 Q×块大小；核心逐查询处理，距离临时数组为 1×叶大小。两者均处于同一资源预算内。各方法先执行构建后单查询，再一次完整预热，然后每轮随机方法顺序，七次完整批次计时。"
        if chinese
        else "Both cdist baselines use the same compiled SciPy distance kernel and exact top-k selection independent of the core engine; only block size changes. The large-block baseline discloses Python small-block loop overhead. All methods use float64, one native thread and identical data/queries with query safety checks. cdist batches all queries, using a Q-by-block distance workspace; the core processes one query at a time, using a one-by-leaf workspace. Both remain within the same resource budget. Each method performs a postbuild single query, one full warmup, then seven timed batches with a randomized method order per round.",
        "高维各向同性与真实 GloVe 结果用于检验适用范围。合成簇结构可能有利于边界盒剪枝；不能由此推断其他嵌入、近似检索、生产服务或其他硬件的收益。不同种子的结果分别呈现，未合并为总体显著性结论。"
        if chinese
        else "High-dimensional isotropic and real GloVe workloads probe applicability. Synthetic cluster structure may favor bounding-box pruning. Results do not establish benefits for other embeddings, approximate retrieval, production services or other hardware. Independent seeds remain separate; no pooled significance claim is made.",
        "",
        "![Query batch latency / 查询批次延迟](../../results/figures/query_latency.png)",
        "",
        "图中误差线为所保存样本的最小与最大值，纵轴为对数尺度。"
        if chinese
        else "Error bars show the minimum and maximum saved samples; the vertical axis is logarithmic.",
        "",
    ]
    return "\n".join(lines)


def plot_rows(rows: list[dict[str, Any]], output: Path) -> None:
    valid = [row for row in rows if row["status"] == "ok"]
    keys = list(dict.fromkeys((row["run_id"], row["workload_id"], row["seed"]) for row in valid))
    if not keys:
        return
    columns = min(3, len(keys))
    nrows = math.ceil(len(keys) / columns)
    figure, axes = plt.subplots(nrows, columns, figsize=(5.8 * columns, 3.8 * nrows), squeeze=False)
    for axis, key in zip(axes.flat, keys, strict=False):
        selection = {
            row["method"]: row
            for row in valid
            if (row["run_id"], row["workload_id"], row["seed"]) == key
        }
        names = [name for name in METHODS if name in selection]
        medians = np.array([selection[name]["median_seconds"] * 1000 for name in names])
        low = np.array([selection[name]["min_seconds"] * 1000 for name in names])
        high = np.array([selection[name]["max_seconds"] * 1000 for name in names])
        axis.bar(
            np.arange(len(names)),
            medians,
            color=[COLORS[METHODS.index(name)] for name in names],
            yerr=np.vstack((medians - low, high - medians)),
            capsize=3,
        )
        axis.set_xticks(np.arange(len(names)), names, rotation=28, ha="right", fontsize=8)
        axis.set_yscale("log")
        axis.set_ylabel("Batch latency (ms, log scale)")
        axis.set_title(f"{key[1]} · seed {key[2]}\nrun {key[0][:8]}", fontsize=10)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    for axis in list(axes.flat)[len(keys) :]:
        axis.set_visible(False)
    figure.suptitle("Measured exact-search latency · median with observed min/max", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate bilingual measured reports / 生成双语实测报告"
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=["results/raw/*.json"],
        help="Raw JSON paths or globs / 原始 JSON 路径或匹配式",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Summary/figure directory / 汇总与图表目录",
    )
    parser.add_argument(
        "--docs-root",
        type=Path,
        default=Path("docs"),
        help="Bilingual documentation root / 双语文档根目录",
    )
    args = parser.parse_args()
    paths = sorted({Path(path) for pattern in args.input for path in glob.glob(pattern)})
    if not paths:
        parser.error("no raw runs matched / 未匹配到原始运行")
    root = Path(__file__).resolve().parents[1]
    runs: list[dict[str, Any]] = []
    for path in paths:
        raw = json.loads(path.read_text())
        if raw.get("schema_version") != 1 or "workloads" not in raw:
            parser.error(f"not a benchmark raw file / 不是基准原始文件: {path.name}")
        try:
            raw["_input_path"] = path.resolve().relative_to(root).as_posix()
        except ValueError:
            raw["_input_path"] = path.name
        runs.append(raw)
    rows = [row for raw in runs for row in summarize(raw)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"schema_version": 1, "rows": rows}, indent=2, allow_nan=False) + "\n"
    )
    plot_rows(rows, args.output_dir / "figures" / "query_latency.png")
    for language in ("en", "zh"):
        directory = args.docs_root / language
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "RESULTS.md").write_text(render_report(runs, rows, chinese=language == "zh"))
    print(f"Analyzed {len(runs)} runs, {len(rows)} records / 已分析运行与记录")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

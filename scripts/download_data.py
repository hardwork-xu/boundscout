"""Fetch public GloVe benchmark vectors with provenance; 下载公开向量并保存来源校验。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import h5py

SOURCE_URL = "https://ann-benchmarks.com/glove-25-angular.hdf5"
LICENSE_URL = "https://nlp.stanford.edu/projects/glove/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(output: Path, expected_sha256: str | None = None) -> dict[str, object]:
    """Never silently replace an existing dataset; 不静默覆盖已有数据。"""
    output.parent.mkdir(parents=True, exist_ok=True)
    downloaded = not output.exists()
    if downloaded:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=output.parent, suffix=".partial", delete=False
            ) as handle:
                temporary = Path(handle.name)
                with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
                    size = 0
                    while block := response.read(1024 * 1024):
                        size += len(block)
                        if size > 600 * 1024 * 1024:
                            raise ValueError("download exceeds 600 MiB / 下载超过 600 MiB 限额")
                        handle.write(block)
            digest = sha256(temporary)
            if expected_sha256 is not None and digest != expected_sha256.lower():
                raise ValueError("SHA256 mismatch / SHA256 校验失败")
            # Verify the expected public HDF5 schema before publishing the local file.
            # 保存前验证公开数据集的 HDF5 结构。
            with h5py.File(temporary, "r") as handle:
                if handle["train"].shape[1] != 25 or handle["test"].shape[1] != 25:
                    raise ValueError("unexpected GloVe dimensions / GloVe 维数异常")
            os.replace(temporary, output)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    digest = sha256(output)
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise ValueError("SHA256 mismatch / SHA256 校验失败")
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        if existing["sha256"] != digest:
            raise ValueError("existing metadata digest mismatch / 已有元数据校验不匹配")
        return dict(existing)
    with h5py.File(output, "r") as handle:
        train_shape, test_shape = list(handle["train"].shape), list(handle["test"].shape)
    metadata: dict[str, object] = {
        "schema_version": 1,
        "source_url": SOURCE_URL,
        "upstream_project": "https://github.com/erikbern/ann-benchmarks",
        "original_vectors": LICENSE_URL,
        "license": "PDDL-1.0 (Stanford GloVe pretrained vectors)",
        "license_source": LICENSE_URL,
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "sha256": digest,
        "file_bytes": output.stat().st_size,
        "train_shape": train_shape,
        "test_shape": test_shape,
        "acquisition": "downloaded" if downloaded else "existing file inspected",
        "checksum_trust": "verified against supplied digest"
        if expected_sha256
        else "first-observed local digest; no upstream checksum claimed",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Download public GloVe data / 下载公开 GloVe 数据")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/glove-25-angular.hdf5"),
        help="Local dataset path / 本地数据路径",
    )
    parser.add_argument(
        "--expected-sha256",
        help="Previously recorded SHA256 for strict replication / 严格复现用已记录校验值",
    )
    args = parser.parse_args()
    try:
        metadata = download(args.output, args.expected_sha256)
    except (OSError, ValueError, KeyError) as error:
        message = str(error).replace(str(Path.home()), "<home>")
        parser.exit(1, f"Dataset acquisition failed / 数据获取失败: {message}\n")
    print(
        json.dumps(
            {key: metadata[key] for key in ("sha256", "file_bytes", "train_shape", "test_shape")},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

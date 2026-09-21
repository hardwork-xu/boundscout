"""Execute and retain sanitized acceptance evidence. / 执行验收并保存去敏证据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fingerprint() -> dict[str, object]:
    """Hash current package and measured benchmark. / 对当前包与基准源码生成哈希。"""
    paths = [
        *ROOT.glob("src/**/*.py"),
        ROOT / "benchmark.py",
        ROOT / "pyproject.toml",
        ROOT / "requirements.lock",
        *ROOT.glob("tests/*.py"),
        *ROOT.glob("scripts/*.py"),
    ]
    files = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
        if p.is_file()
    }
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return {
        "git_revision": rev,
        "files_sha256": files,
        "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
    }


def main() -> int:
    """Run local reproducible checks, never network publication. / 运行本地验收，不执行网络发布。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="results/acceptance.json", help="acceptance JSON / 验收记录"
    )
    args = parser.parse_args()
    output = ROOT / args.output
    logs = output.parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    checks: list[dict[str, object]] = []
    version = fingerprint()
    specs = [
        (
            "public-api",
            [
                python,
                "-c",
                "import numpy as np; from boundscout import BoundIndex; "
                "i=BoundIndex.build(np.eye(4)); "
                "print(i.search(np.eye(4),k=1).indices.tolist()); i.close()",
            ],
        ),
        ("default-demo", [python, "-m", "boundscout", "demo"]),
        ("storage-example", [python, "examples/local_vectors.py"]),
        ("unit-tests", [python, "-m", "pytest", "-m", "not integration", "-q"]),
        ("integration-tests", [python, "-m", "pytest", "-m", "integration", "-q"]),
        ("ruff", [python, "-m", "ruff", "check", "."]),
        ("format", [python, "-m", "ruff", "format", "--check", "."]),
        ("mypy", [python, "-m", "mypy", "src/boundscout"]),
        ("documentation", [python, "scripts/check_docs.py"]),
        ("build", [python, "-m", "build", "--no-isolation"]),
        ("diff-whitespace", ["git", "diff", "--check"]),
    ]
    for name, command in specs:
        started = datetime.now(UTC).isoformat()
        run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        text = run.stdout + run.stderr
        for path, replacement in (
            (str(ROOT), "$PROJECT"),
            (str(Path(python).parent.parent), "$VENV"),
            (str(Path.home()), "$HOME"),
        ):
            text = text.replace(path, replacement)
        log = logs / f"{name}.log"
        log.write_text(text)
        checks.append(
            {
                "name": name,
                "command": ["python" if x == python else x for x in command],
                "started_at": started,
                "exit_code": run.returncode,
                "status": "passed" if run.returncode == 0 else "failed",
                "summary": text[-1400:],
                "log": str(log.relative_to(ROOT)),
                "source_sha256": version["sha256"],
            }
        )
        print(f"{name}: {checks[-1]['status']}", flush=True)
    if shutil.which("docker") is None:
        checks.append(
            {
                "name": "docker-build-run",
                "command": ["docker build -t boundscout .", "docker run --rm boundscout"],
                "status": "not_run",
                "exit_code": None,
                "reason": "Docker executable absent / 未安装 Docker 可执行程序",
                "source_sha256": version["sha256"],
            }
        )
    else:
        for name, cmd in [
            ("docker-build", ["docker", "build", "-t", "boundscout", "."]),
            ("docker-run", ["docker", "run", "--rm", "boundscout"]),
        ]:
            run = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
            log = logs / f"{name}.log"
            log.write_text((run.stdout + run.stderr).replace(str(Path.home()), "$HOME"))
            checks.append(
                {
                    "name": name,
                    "command": cmd,
                    "exit_code": run.returncode,
                    "status": "passed" if run.returncode == 0 else "failed",
                    "log": str(log.relative_to(ROOT)),
                    "source_sha256": version["sha256"],
                }
            )
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at": datetime.now(UTC).isoformat(),
                "source": version,
                "platform": platform.system(),
                "checks": checks,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    return int(any(check["status"] == "failed" for check in checks))


if __name__ == "__main__":
    raise SystemExit(main())

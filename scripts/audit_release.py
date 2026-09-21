"""Review publishable content and measured-source identity. / 检查发布内容与被测源码一致性。"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "private-home-path": re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+/"),
    "private-temporary-path": re.compile(r"/private/var/folders/[A-Za-z0-9]{2}/[^\s]+"),
    "token": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    "private-email": re.compile(
        r"(?=[A-Za-z0-9_.+%-]*[A-Za-z0-9])[A-Za-z0-9_.+%-]+@(?!users\.noreply\.github\.com)[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    ),
}


def main() -> int:
    """Report locations and types, never leaked values. / 只报告位置与类型，不输出泄漏内容。"""
    names = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
        )
        .decode()
        .split("\0")
    )
    findings = []
    for name in filter(None, names):
        path = ROOT / name
        if not path.is_file() or path.suffix == ".png":
            continue
        content = path.read_text(errors="replace")
        for label, pattern in PATTERNS.items():
            if pattern.search(content):
                findings.append({"path": name, "type": label})
    history = subprocess.check_output(["git", "log", "--all", "-p"], cwd=ROOT, text=True)
    for label, pattern in PATTERNS.items():
        if pattern.search(history):
            findings.append({"path": "git-history", "type": label})
    raw = json.loads((ROOT / "results/raw/full.json").read_text())
    changed = []
    reporting_changes = []
    notes = ROOT / "results/post-experiment-changes.json"
    amendments = json.loads(notes.read_text())["changes"] if notes.exists() else []
    for name, expected in raw["source"]["files_sha256"].items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            current = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            approved = any(
                n["path"] == name
                and n["before_sha256"] == expected
                and n["after_sha256"] == current
                and n["kind"] == "reporting-only"
                for n in amendments
            )
            if name == "scripts/analyze.py" and approved:
                reporting_changes.append(name)
            else:
                changed.append(name)
    artifacts = []
    for path in (ROOT / "dist").glob("*.whl"):
        with zipfile.ZipFile(path) as wheel:
            for name in wheel.namelist():
                content = wheel.read(name).decode(errors="replace")
                for label, pattern in PATTERNS.items():
                    if pattern.search(content):
                        findings.append({"path": f"dist/{path.name}:{name}", "type": label})
        artifacts.append(path.name)
    payload = {
        "ok": not findings and not changed,
        "files_scanned": len(names) - 1,
        "findings": findings,
        "changed_recorded_benchmark_files": changed,
        "documented_reporting_only_changes": reporting_changes,
        "wheel_artifacts_scanned": artifacts,
        "scope": "content/history patterns and recorded source digests; not a security guarantee",
    }
    print(json.dumps(payload, indent=2))
    return int(not payload["ok"])


if __name__ == "__main__":
    raise SystemExit(main())

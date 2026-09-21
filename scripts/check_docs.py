"""Check document pairs and repository links. / 检查双语文件配对与仓库链接。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Return nonzero for missing pairs/links. / 缺失配对或链接时返回非零。"""
    errors: list[str] = []
    en = {p.name for p in (ROOT / "docs/en").glob("*.md")}
    zh = {p.name for p in (ROOT / "docs/zh").glob("*.md")}
    if en != zh:
        errors.append(f"document pairs differ / 双语文件不匹配: {sorted(en ^ zh)}")
    paths = list(ROOT.glob("*.md")) + list((ROOT / "docs").rglob("*.md"))
    links = 0
    for path in paths:
        content = path.read_text()
        if content.count("```") % 2:
            errors.append(f"unclosed code fence / 代码围栏未闭合: {path.relative_to(ROOT)}")
        for target in re.findall(r"\]\(([^\s)]+)(?:\s+[^)]*)?\)", content):
            target = unquote(target.strip("<>").split("#", 1)[0])
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            links += 1
            if not (path.parent / target).exists():
                errors.append(f"broken link / 链接失效: {path.relative_to(ROOT)} -> {target}")
    for filename in ("README", "CONTRIBUTING", "SECURITY", "NOTICE"):
        if not (ROOT / f"{filename}.md").exists() or not (ROOT / f"{filename}_zh.md").exists():
            errors.append(f"missing root pair / 根目录缺少双语文件: {filename}")
    print(
        json.dumps(
            {
                "ok": not errors,
                "document_pairs": len(en),
                "local_links": links,
                "errors": errors,
                "scope": "file pairing and relative links; semantic parity reviewed manually",
            },
            ensure_ascii=False,
        )
    )
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())

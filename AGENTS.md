# Repository conventions / 仓库约定

- `src/boundscout`: engine, persistence and CLI. `tests`: offline tests. `configs`, `benchmark.py`, `scripts`, `results`: reproducible evidence. / 核心引擎、存储、CLI、离线测试与实验分别维护。
- Python 3.11–3.12. Use `python3 -m venv .venv`, then `make install`; `make demo test lint typecheck build`. / 使用隔离环境与统一命令。
- Update English and simplified Chinese documentation together; machine fields remain English. / 双语同步，机器字段保持英文。
- Preserve frozen targets, raw measurements, failures and source hashes. Never invent numbers or treat skipped checks as passed. / 保留原目标、全部样本、失败与源码指纹，禁止伪造。
- No private data, personal email, tokens, home paths or datasets in Git. No global Git changes, destructive cleanup, paid resources or unsolicited external publication. / Git 不收私人数据、邮箱、令牌、主目录或数据集；禁止全局 Git 修改、破坏性清理、付费资源及未经授权发布。
- Finish changes with relevant tests, static checks, bilingual docs and updated evidence; rerun affected benchmarks after engine changes. / 改动须完成相关验证、文档与证据更新。

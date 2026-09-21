# Contributing

[简体中文](CONTRIBUTING_zh.md)

I welcome small changes that make exact search easier to verify or explain. Start with the [walkthrough](docs/en/WALKTHROUGH.md) and [repository conventions](AGENTS.md). Set up Python 3.12 with `python3 -m venv .venv` and `make install`, then run `make check`.

Describe the problem and the workload before changing a mechanism. Preserve distance accuracy and ID tie order. Add a regression for a demonstrated defect; keep tests offline. Engine changes need affected benchmark reruns with raw samples and source hashes. Never overwrite a published result file to hide a regression: use a new run file. A negative performance result is useful evidence.

Update English and simplified Chinese together, retain third-party attribution, and exclude private vectors and local paths from commits. Use Conventional Commits with an English subject and a Chinese explanation plus validation in the body. Contributions are under the MIT license. There is no contributor agreement or promise of review time.

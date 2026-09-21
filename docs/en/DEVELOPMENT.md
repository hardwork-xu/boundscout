# Development record

[简体中文](../zh/DEVELOPMENT.md) · [Protocol](PROTOCOL.md) · [Architecture](ARCHITECTURE.md)

This record covers one local development session on **2026-09-21**. It records implementation decisions, failed checks and verification checkpoints. It does not imply a longer development history or independent external review. A passing checkpoint applies to the code and tests at that moment; later audit findings remain visible.

## Committed milestones

These commits were verified from the local repository as this record was updated. They are listed oldest first; later integration commits are not predicted here.

| Commit | Actual subject | Purpose |
| --- | --- | --- |
| `184ff1a` | `docs: freeze exact-search scope and experiment targets` | Preserve the research question and target before implementation and timing. |
| `341b7df` | `feat: implement deterministic spatial block search` | Establish the exact-search engine and original-ID tie policy. |
| `2eff224` | `feat: add validated index persistence and bilingual CLI` | Add persistent indexes and the local file workflow. |
| `6b73052` | `build: lock dependencies and add reproducible CPU experiments` | Add the dependency lock, experiment infrastructure and malformed-file fixes. Its test-count statement is corrected below. |

Inspect the recorded changes with `git show <commit>` and the complete current sequence with `git log --oneline --reverse`. Commit boundaries describe delivered functionality, not separate research sessions.

## Implementation choices retained

The engine uses balanced spatial leaf blocks, axis-aligned bounds and compiled squared-distance evaluation. Search orders candidates by `(squared distance, original row ID)` and preserves boundary ties. The frozen target and contribution boundary were retained during debugging. No search-engine performance tuning or target changes were introduced in the fixes described below.

Persistence and CLI validation received particular attention because malformed files should fail through a clear supported error path. A checksum alone cannot validate the index's geometric invariants, and a filename extension alone cannot establish the kind of object returned by a loader. These findings strengthened the engineering boundary without changing the search algorithm.

## Checks that found problems

| Finding | Why it mattered | Change or status at this checkpoint |
| --- | --- | --- |
| The `sha256` metadata field could be null, a list or a string; malformed values reached dictionary operations and produced `AttributeError`. | A malformed manifest escaped the intended validation error path. | Added an explicit dictionary check and validation of the required key set, with regressions for malformed forms. |
| JSON `true` was accepted as `format_version: 1`. | In Python, `bool` is an `int` subclass and compares equal to 1; equality alone was too permissive for the schema. | Required the actual integer type for the format version. |
| An invalid numeric CLI argument produced an argparse message without Chinese text. | An ordinary user error bypassed the bilingual error contract. | Added `BilingualParser.error` handling and regression coverage. |
| Mypy could not infer the required array-shape and mmap-attribute types. | Runtime behavior and the static contract were not expressed consistently. | Added explicit shape and mmap-related annotations; these were typing fixes, not performance changes. |
| The initial dependency lock selected a newer setuptools/wheel pair than the declared build backend. | A resolved lock did not by itself establish that a clean install used the declared build toolchain. | Added exact development pins aligned with `setuptools==80.9.0` and `wheel==0.45.1`. Clean-install acceptance must be verified separately. |
| The second audit found an empty NPY input raising `EOFError`. | The normal file-error handling was incomplete. | Fix and regression verification were in progress when this checkpoint was written. |
| The second audit found an NPZ archive masquerading as a `.npy` file, leading to an unexpected loader object, handle-lifecycle concerns and `AttributeError`. | The CLI assumed that `np.load` had returned an ndarray. | Array-type validation, resource handling and regression verification were in progress at this checkpoint. |

The dependency-lock issue illustrates a distinction between resolving dependencies and reproducing an installation: both the runtime dependency set and the build backend must agree with the recorded configuration. Likewise, a successful search test does not exercise every malformed-input path.

## Verification checkpoints

| Checkpoint | Observed outcome | Scope and limit |
| --- | --- | --- |
| Initial automated suite | **87 tests passed** | An initial correctness and workflow checkpoint; later malformed-input cases were not all covered. |
| Expanded suite before the second audit | **104 tests passed in 4.28 seconds** | Passed after the first validation and bilingual-error fixes. This preceded the empty-NPY and disguised-NPZ findings. |
| Walkthrough execution | Four operational code blocks passed | Synthetic data preparation, CLI build/query, result inspection, and Python exhaustive comparison plus save/load were executed. The English and Chinese code blocks matched. This was not a clean-install or performance claim. |
| Second audit at the initial documentation checkpoint | Two additional malformed-file paths identified | The audit was still open when the initial checkpoint above was recorded. |
| Second audit completed in the original work environment | **111 tests passed in 6.91 seconds** | Empty-NPY and disguised-NPZ handling was fixed, with three additional regression cases. This observed full-suite count is independent of the fresh-environment run. |
| Full suite in the clean project `.venv` | **111 tests passed in 30.77 seconds** | A separate completed run after the clean installation; the earlier pending status is superseded by this observation. |
| Clean installation | Hash-locked dependency installation and `pip check` passed | Establishes installation and dependency consistency for the tested environment, not every supported platform. |
| Static checks | Ruff check and format checks passed for 15 code files; mypy passed for 5 source files | Independent of runtime tests and benchmark results. |
| Full benchmark launch | Started against source revision `6b73052` | Launch was observed; performance outcomes were still pending at this checkpoint. No heavy tests were started alongside timing for this documentation work. |

### Record correction

The body of commit `6b73052` prematurely recorded **112 tests**. The actual completed suite in the original work environment reported **111 passed in 6.91 seconds**. The commit is preserved without rewriting history; this entry corrects the count. No extra test or extra successful run is inferred to reconcile the discrepancy.

When this correction was first recorded, the separately started full-suite run in the clean `.venv` was still pending. It subsequently completed with **111 passed in 30.77 seconds**, as recorded separately above. Neither observed run supports the commit body's count of 112.

Each elapsed test time above describes a single local run; it is not a performance measurement of the search engine. Test counts can grow when regressions are added and must not be used as a substitute for the cases they cover.

## Documentation validation before release

Run `python scripts/check_docs.py` after the final documentation and generated results are present. The [checker](../../scripts/check_docs.py) validates paired document filenames, required root-language pairs, code-fence balance and relative file-link targets. It does not verify semantic translation parity, section anchors or remote pages; those require review. This paragraph records the validation procedure, not an unobserved successful check.

Review the paired [architecture contract](ARCHITECTURE.md), [walkthrough](WALKTHROUGH.md), [research rationale](RESEARCH.md) and [experimental results](RESULTS.md) against the implementation and raw evidence. The walkthrough's executable blocks were checked as recorded above. Compare the final results to the unchanged [protocol](PROTOCOL.md), preserve adverse cases, and identify skipped checks explicitly in release acceptance material.

## Evidence boundaries

The [frozen protocol](PROTOCOL.md) governs experimental targets and timing. Raw benchmark evidence and generated [Results](RESULTS.md) answer performance questions; this development record answers what was built and what failed along the way. Local tests, static checks, package installation, container execution and remote CI are separate acceptance activities. A pending or unavailable activity must not be reported as passed because another activity succeeded.

Later validation should be appended as an observed checkpoint, preserving the earlier findings and their status at the time. Any engine change after timing requires rerunning affected experiments and updating the recorded source fingerprint. Do not rewrite the initial target to fit the measurements.

## Public verification checkpoint

Commit `58c2e8e` preserves the complete benchmark and corrects the earlier test-count typo; `2f5a4cd` adds the reviewed bilingual documents and local acceptance. GitHub run `35600862457` completed successfully for both Ubuntu 24.04 and macOS 14 at `2f5a4cd`. This is ordinary CPU CI evidence, separate from the local M1 Pro full experiment. The current publication receipt records repository, CI and release state.

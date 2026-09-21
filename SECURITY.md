# Security and operating limits

[简体中文](SECURITY_zh.md)

BoundScout is a research prototype for offline search over trusted local vectors and operator-created or reviewed indexes. It has no production security claim and has not undergone an independent security audit. The core library and CLI provide neither a network service nor authentication, access control, or encryption.

## Files and integrity

NPY loading uses `allow_pickle=False`; object arrays are unsupported. Index loading checks metadata, numeric types, sizes, IDs and block geometry. Default `verify=True` checks SHA256 digests. A digest detects a change relative to its manifest; it does not authenticate the author. Someone who replaces both arrays and metadata can create a different valid index. `verify=False` skips hashes while retaining structural and geometry checks.

NumPy must parse file headers and open memory maps before every structural check is complete. The index loader checks file sizes before mapping and logical array sizes after loading, but these checks do not make NPY a hostile-input format. Malformed headers, unusually large inputs and validation work can consume CPU, address space and temporary memory. Do not expose this loader directly to arbitrary uploads. Use trusted files and appropriate operating-system resource limits when running unfamiliar workloads.

Keep a loaded index directory immutable. Another process modifying or truncating a mapped file can invalidate results or crash the process; a read-only mapping cannot prevent external writes. Use one writer and a new output directory for each index. Atomic publication is not a complete power-loss durability guarantee.

## Numerical and resource limits

- Accepted real numeric inputs are converted to finite float64 values, with 1–4,096 dimensions and absolute coordinates at most `1e100`. Integer conversion can lose precision above float64's exact integer range. Complex, Boolean, object and nonfinite inputs are rejected.
- Results follow computed squared-L2 distance and original row-ID ordering. The floating-point pruning guard is not a formal proof of real-number ranking for arbitrarily close inputs; see the [numerical contract](docs/en/ARCHITECTURE.md#numerical-contract).
- `max_bytes` limits logical stored index arrays, with a conservative build estimate. It is not a hard process-RSS limit. Input conversion, building, hash and geometry validation, sorting, and concurrent searches require additional memory and time. The output-byte limit is separate.
- Use a context manager or `close()` to release loaded mappings. Concurrent read-only searches are supported; concurrent `close()` and search are unsupported. Do not mutate internal arrays or use the internal array constructor.

## Reporting a concern

For a local checkout, record a minimal reproduction privately for the operator or maintainer. Include the version, platform, expected behavior and a small synthetic input; omit personal vectors, credentials and private paths. If a shared repository later offers private security advisories, verify that the facility is enabled before using it. This document does not promise an active private reporting channel. Avoid publishing sensitive data or exploit details in a public issue.

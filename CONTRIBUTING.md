<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Contributing to Logit Loom

Thank you for helping build dependable token-stream tools. Contributions are
welcome for backend-neutral contracts, transforms, observers, adapters,
examples, portability, and documentation.

## Before opening a change

For a defect, include the smallest reproducible contract, token sequence, or
candidate array possible. Do not attach model weights, private prompts, secrets,
or proprietary output corpora. For a public API proposal, describe its causal
boundary, failure behavior, resource bounds, and backend assumptions.

The project documents functionality, not model efficacy. A feature may be
useful without claiming it improves output quality. Research comparisons and
semantic evaluations should live in a separate project until their evidence
and publication scope are agreed.

## Development setup

The workspace pins Rust 1.97.1. Building the llama.cpp adapter also requires a
C/C++ toolchain, CMake, and libclang. Accelerator backends require the
corresponding platform SDK.

```sh
git clone https://github.com/paudley/logit-loom.git
cd logit-loom
make check-core
make check
make doc
```

The ordinary suite compiles native code but does not load or execute a model.
Use a backend feature explicitly when building an application, for example:

```sh
cargo check -p logit-loom-llamacpp --features vulkan
```

Backend features should generally be selected one deployment at a time. See
[`docs/compatibility.md`](docs/compatibility.md).

## Pull requests

A focused pull request should include:

- the mechanical behavior and boundary being changed;
- tests for success, invalid input, and failure behavior;
- documentation for ordering, state mutation, and compatibility;
- any serialized contract or digest-domain impact;
- the commands used for validation.

Run `scripts/release-check.sh --allow-dirty` before requesting review. Native
model-backed tests, if relevant, must be opt-in and use caller-supplied local
artifacts. Never add automatic model downloads.

The check compiles every example target and runs crate README doctests. New
examples should remain deterministic and model-free unless their filename and
documentation clearly identify an opt-in local-artifact requirement.

## Compatibility

The API is currently pre-1.0, but changes should still be deliberate. Do not
silently reinterpret an existing serialized shape, digest domain, receipt, or
checkpoint identity. Prefer an explicitly versioned replacement and note the
change in `CHANGELOG.md`.

## License of contributions

Unless explicitly stated otherwise, contributions intentionally submitted for
inclusion are offered under `MIT OR Apache-2.0`, as described in
[`LICENSING.md`](LICENSING.md). Substantial contributions may require the
project Contributor License Agreement so the copyright holder can continue to
offer separate proprietary licenses. See [`CLA.md`](CLA.md) for the process.

By submitting a contribution, you represent that you have the right to do so
and that it does not include incompatible code, model assets, private data, or
third-party material without appropriate notice.

## Conduct and security

Participation is governed by [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). Report
security issues privately as described in [`SECURITY.md`](SECURITY.md).

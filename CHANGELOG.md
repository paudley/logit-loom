<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Changelog

All notable changes to Logit Loom are documented here. The project follows
[Semantic Versioning](https://semver.org/). Before `1.0.0`, minor releases may
include breaking API changes.

## [Unreleased]

## [0.1.1] - 2026-07-23

### Added

- Tag-gated, OIDC-authenticated publishing for the coordinated crates.io
  workspace release.

### Changed

- Release validation now binds the release tag, shared workspace version, dated
  changelog entry, clean repository checks, and exact-lockfile RustSec audit
  before any crate is published.

## [0.1.0] - 2026-07-23

### Added

- `logit-loom-core` backend-neutral plans, identities, and mechanical receipts.
- `logit-loom` ordered transactional transforms, exact-byte observers,
  cooperative cancellation, and first-party bias transforms.
- Backend-selected candidate-view execution for custom native adapters, with
  sparse-bound checks and transactional write-back.
- Validated token-ID deserialization and explicit DRY/Mirostat sampler bounds.
- `logit-loom-llamacpp` prefill, bounded generation, native sampler
  composition, checkpoints, `LoRA`, and control-vector scopes.
- Conservative session poisoning after an unobserved steering-cleanup failure
  or partial native checkpoint restore.
- Public architecture, compatibility, contribution, security, and release
  documentation.
- Scheduled and dependency-change RustSec advisory auditing in hosted CI.
- Compile-tested crate-level guides plus runnable plan, transform, and
  exact-byte observer examples.
- Focused accounting, lifecycle, first-party transform, and native-option
  boundary tests.
- Validating `StateSnapshot` reconstruction and ownership APIs for
  application-defined checkpoint persistence.
- A shared-family project logo, editable social card, rendered sharing image,
  and public brand guidance.
- Up-front README rationale and mechanically scoped use cases.

### Changed

- Transform invocations now require consecutive zero-based steps, and
  backend-selected candidate views reject duplicate token identifiers.
- Observer delivery now requires a preceding poll, enforces the requested
  token bound, and treats a cooperative stop as terminal for the call.
- Transform token admission now requires an unmatched successful invocation;
  controlled-prefill polling and stopping enforce the same lifecycle cadence.
- In-progress prefill accounting is represented explicitly, with the prefill
  receipt digest domain advanced to `prefill-receipt-v2`; it cannot be supplied
  as a terminal monitor finish.
- Grammar strings, first-party token-bias inputs, and generation collections
  now have explicit public bounds.
- Prefill progress requires a nonzero request, and Mirostat v2 requires the
  unused v1-only window field to remain zero.
- Generation-plan bias token IDs must be unique, and the native adapter rejects
  bias IDs outside the loaded model vocabulary.
- Native tokenization inputs are bounded and reject NUL bytes before entering
  the binding.
- Caller-supplied token IDs are checked against the loaded vocabulary before
  detokenization, prefill, decode, or checkpoint restore.
- Receipt position checks reject arithmetic overflow instead of accepting
  saturated accounting.
- Pipeline and checkpoint receipts validate structural identities and
  accounting before producing a digest.
- `Session::clear` now returns `Result` so a poisoned session cannot mutate
  native causal state through an infallible escape hatch.
- Checkpoint restore verifies the recorded opaque state byte count.
- Checkpoint receipts use `checkpoint-receipt-v2`; their backend identity now
  binds exact session allocation options, and restore rejects positions beyond
  the destination context before calling native code.
- Partial native checkpoint restore now poisons the session; poisoned mutation
  reports `Error::Poisoned` and `Session::poison_reason` exposes the retained
  cause.
- Model and `LoRA` files now use distinct artifact digest domains.
- Crate manifests no longer declare unused direct or development dependencies.
- Public error and native placement enums are non-exhaustive so compatible
  releases can add variants without breaking downstream matches.
- Native repetition and DRY samplers are seeded from causal history without
  advancing the output grammar through prompt tokens.
- Model and `LoRA` loading reject files whose content identity changes across
  the native load operation.

[Unreleased]: https://github.com/paudley/logit-loom/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/paudley/logit-loom/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/paudley/logit-loom/releases/tag/v0.1.0

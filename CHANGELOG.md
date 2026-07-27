<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Changelog

All notable changes to Logit Loom are documented here. The project follows
[Semantic Versioning](https://semver.org/). Before `1.0.0`, minor releases may
include breaking API changes.

## [Unreleased]

### Added

- `logit-loom-runtime`, an explicit higher-level local llama.cpp façade for
  model loading, exact text replacement/append, bounded one-shot and stateful
  generation, checkpoints, and typed steering scopes.
- Bounded pipeline and observer builders with versioned automatic identities
  for first-party rank bias, token bias, and cooperative cancellation.
- Runtime examples for model-free mechanics, local generation, and
  compatibility-checked checkpoint branching.
- Eight end-to-end mechanical experiment runbooks with compiled text and image
  examples, structured JSON reports, success criteria, variations, and failure
  diagnosis for checkpoints, transforms, exact bytes, stopping, cancellation,
  scoped `LoRA`, direct-RGB state, and latent state.
- A bounded optional-model catalog and repository-local acquisition tooling for
  exact, caller-fetched Qwen3 0.6B Q8_0, MiniT2I-B/16, and Krea 2 Turbo
  artifacts, with pinned revisions, weight hashes, no-network dry runs, local
  verification, and explicit gated-license acknowledgement.
- A bounded path-free acquisition report recording exact verified artifacts
  for Qwen, MiniT2I, and Krea across repository fetches and caller-managed
  stores.
- Exact Qwen profile loading with pre/post native artifact verification and a
  deterministic checkpoint replay example.
- `logit-loom-diffusion`, with backend-neutral tensor, schedule, plan,
  checkpoint, transactional intervention, observer, and receipt contracts.
- `logit-loom-executor`, with transport-neutral worker-local lifecycle,
  exact borrowed-buffer, cancellation, cleanup-receipt, and classified-failure
  contracts.
- Serializable whole-image execution plans and receipts for exact
  text-to-image, image-to-image, inpaint, outpaint, VAE, LoRA, installed
  operator, observation, placement, and buffer mechanics.
- `logit-loom-diffusion-sdcpp`, a safe adapter over companion ABI version 1 for
  an exact pinned stable-diffusion.cpp revision, with explicit accelerator
  placement and focused unsafe-boundary tests.
- Stable-diffusion.cpp image ABI version 2 and safe adapter contract version 4,
  with direct caller-owned RGB output, bounded source/mask/reference inputs,
  negative conditioning, fixed request-local LoRA stacks with verified tensor
  participation and cleanup, direct Krea VAE encode/decode, lifecycle epochs,
  and reuse-aware error classification.
- Version-two whole-image plans and stable-diffusion.cpp lowering for
  authenticated checkpoint envelopes, installed scheduler-state operators,
  observations, exact post-step cancellation, bounded deterministic RGB8
  compositing, explicit output routes, and cleanup disposition.
- A reproducible native companion patch/build script, model-free ABI probe,
  complete MiniT2I and Krea checkpoint experiments, and an output-free
  model-acceptance report schema.
- Native per-step denoiser-plus-Euler timing returned as non-deterministic
  deployment measurements outside receipts and content identities.
- Retained, output-free Vulkan acceptance reports for exact Qwen checkpoint
  replay and MiniT2I and Krea unchanged/intervened checkpoint branches,
  including Krea deployment timing and qualified memory observations.
- First-class Qwen, MiniT2I, and Krea catalog status, with repository checks
  requiring passed profiles to retain a matching passed acceptance report.
- A phased first-class model integration plan with separate text, compact
  image, advanced image, and release gates.
- An unpublished partial `logit-loom-tokenizer` workspace crate containing
  versioned identities; a safe Gigatoken-derived ranked-BPE kernel with pinned
  packed-SIMD mechanics; reusable direct/count sinks; a bounded caller-sized
  pool; exact-oracle qualification receipts; cancellation; batch/scatter;
  token-aware chunking; and collision-checked cache primitives.
- A higher-level interface guide covering ownership, ordering, identities,
  exact bytes, concurrency boundaries, checkpoints, steering, and low-level
  escape hatches.

### Changed

- Prepared the publishable workspace crates and exact internal dependencies
  for the coordinated `0.2.0` release; the partial tokenizer crate remains
  explicitly unpublished.
- The stable-diffusion.cpp adapter now requires the image-v2 symbol set; a
  companion carrying only the earlier step-v1 surface is rejected at load.
- Stable-diffusion.cpp generation receipts now bind the exact session epoch
  used by the operation; serialized experiment receipt identities advance to
  `sdcpp-generation-receipt-v2`.
- Checkpoint restore now reconstructs the next-token logit boundary after
  loading llama.cpp causal-memory bytes; failed final-position removal or
  re-decode poisons the session instead of sampling stale logits.
- The stable-diffusion.cpp companion suppresses its fallback stdout progress
  renderer so image examples preserve a single valid structured-report stream.
- Diffusion component identifiers accept the catalog's bounded lowercase
  dotted slugs, including `wan-2.1-vae`.
- Model tooling and release checks invoke the unpublished workspace xtask
  package explicitly and no longer depend on a caller-local Cargo alias.
- Root, crate, capability, architecture, compatibility, getting-started, and
  release documentation now describe the expanded text, model-catalog, and
  diffusion workspace.

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

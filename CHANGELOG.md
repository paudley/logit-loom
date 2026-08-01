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
- Stable-diffusion.cpp image ABI version 2 and safe adapter contract version 5,
  with direct caller-owned RGB output, bounded source/mask/reference inputs,
  negative conditioning, fixed request-local LoRA stacks with verified tensor
  participation and cleanup, direct Krea VAE encode/decode, lifecycle epochs,
  and reuse-aware error classification.
- Version-two whole-image plans and stable-diffusion.cpp lowering for
  authenticated checkpoint envelopes, installed scheduler-state operators,
  observations, exact post-step cancellation, bounded deterministic RGB8
  compositing, explicit output routes, and cleanup disposition.
- A default-built `ImageProgramPlanV1` family for bounded resident staged image
  graphs, with typed single-assignment values, multi-native and VAE chaining,
  deterministic joins, checkpoint conversion, liveness-derived arena limits,
  completed-stage receipts, cleanup uncertainty, and deployment measurements
  outside deterministic identity.
- Model-block application ABI version 5 and safe adapter contract version 7,
  with native Krea graph-branch accounting, exact selected-transition bitmaps,
  loaded-topology evidence, request-local cleanup confirmation, and a separate
  digest-bound `ModelBlockApplicationReceiptV1`.
- Krea activation ABI version 6 and safe adapter contract version 8, with
  runtime-derived sites, exact token/CFG/boundary selection, resident sealed
  donor and vector inputs, same-run device-snapshot donors, ordered transplant
  and projection operations, exact callback/resource/placement evidence,
  same-session reuse, and idempotent cleanup poisoning.
- A deterministic create-new projected SafeTensors component transform with
  exact source, topology, orthonormal basis, formula, implementation, output,
  and per-tensor before/after lineage; source and unselected bytes are
  preserved.
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
- Backend-neutral, versioned text topology, activation capture, vector-bank,
  ordered activation-program, provisional telemetry, target-authoritative
  speculation, checkpoint-envelope, and aggregate V2 mechanics contracts.
- A deterministic content-free activation accumulator for bounded group means,
  paired differences, and optional unit normalization without prompt or
  semantic labeling.
- Exact llama.cpp topology/profile validation, bounded activation capture,
  transactional scaled-add and projection-removal operators, and one-sequence
  target-authoritative MTP and EAGLE-3 generation with explicit target/draft
  activation policies, pre-allocation native-pair compatibility checks, and
  admitted/rejected proposal receipts.
- A caller-supplied-model MTP example that writes arbitrary generated token
  bytes without assuming UTF-8.
- Reusable process-local MTP/EAGLE-3 checkpoints with exact target, draft,
  implementation, opaque sampler, activation, stop-prefix, causal, boundary,
  and parent lineage plus fail-before-allocation branch restore.
- Ordered aggregate text steering that applies multiple `LoRA`s and an
  optional control vector as one scope, rolls back partial application,
  removes resources in reverse order, and poisons cleanup uncertainty.
- `execute_text_mechanics`, a whole-operation llama.cpp lowerer for
  `TextMechanicsPlanV2` that preflights every topology, callback, activation,
  steering, and checkpoint identity; composes ordinary or target-authoritative
  speculative execution; supports controlled-prefill cancellation; and emits
  exact terminal and cleanup evidence.
- Aggregate ordinary checkpoints that retain an opaque native sampler and
  cross-operation stop prefix while binding causal state to the complete
  mechanics which built it, plus speculative continuation that reapplies and
  clears exact target steering before quiescent successor capture.

### Changed

- Newly loaded stable-diffusion.cpp sessions now begin at nonzero runtime epoch
  `1`, so first-operation Krea activation receipts have a valid stale-handle
  identity before any explicit session clear.
- Resident native stages now accept bounded tight RGB8/RGBA8 reference images
  at their own serialized dimensions while source images and masks remain
  canvas-bound. Lowering preserves the exact reference bytes and geometry; no
  serialized shape, native ABI, or digest domain changed.
- The text adapter now pins public `llama-cpp-4` 0.4.2 successor revision
  `d76356b9725a3736212b3bfd16c66fc80c995c29`, forward-ported to llama.cpp
  `f87067841bac583bc089a225382248d857791ca8`; source builds no longer require
  an adjacent checkout, while crates.io packaging remains blocked until the
  successor is registry-published.
- The binding successor is submitted upstream as
  [`eugenehp/llama-cpp-rs#301`](https://github.com/eugenehp/llama-cpp-rs/pull/301);
  the open pull request does not relax the registry publication gate.
- The successor binding validates speculative context shape and lifecycle,
  bounds copied prompt/state data, admits the `gpt-oss` EAGLE-3 v3 terminal
  extraction site, and converts missing EAGLE extraction output and C++
  exceptions into failures before they can cross the Rust FFI boundary. It
  also provides allocation-reusing tokenization/raw-piece sinks and a
  count-only query for bounded coordinator utility work.
- Native checkpoint compatibility now binds the exact safe-binding source and
  literal llama.cpp revisions through new binding and session digest domains.
- Release validation now rejects path and Git `llama-cpp-4` successors so the
  adapter cannot be mistaken for a crates.io-ready package.
- The RustSec gate records one narrow informational exception for the
  unmaintained `paste` dependency inherited through pinned `tokenizers`
  0.23.1; vulnerabilities and every other warning remain denied.
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

- `GenerationPlanV2` adds bounded eager/lazy grammar activation with exact
  ordered pattern/token triggers while preserving every v1 generation shape
  and digest unchanged.
- Structured projection now binds caller-owned compiler, validator, grammar,
  and exact-byte feedback identities; the llama.cpp controller restores one
  exact boundary after cancellation or rejection and exposes only
  caller-explicit bounded retries.
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

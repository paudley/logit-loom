# Logit Loom

<p align="center">
  <img src="docs/logit-loom-logo.svg" alt="Logit Loom black cat holding a token-transform loom" width="260">
</p>

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSING.md)
[![CI](https://github.com/paudley/logit-loom/actions/workflows/ci.yml/badge.svg)](https://github.com/paudley/logit-loom/actions/workflows/ci.yml)

Logit Loom is a Rust toolkit for observing, transforming, steering, stopping,
resuming, and accounting for token and diffusion generation. It provides
backend-neutral mechanical contracts plus llama.cpp and stable-diffusion.cpp
adapters without prescribing what a model should say, think, or depict.

The project is functionality-oriented. It makes no claim that a particular
transform, sampler, adapter, or steering method improves model quality.

## Why this exists

Native inference backends expose useful low-level hooks, but applications still
need their ordering, bounds, causal timing, ownership, and failure behavior to
be explicit. Logit Loom puts a small, typed Rust boundary around those
mechanics: backend-neutral contracts where possible, contained callback
execution, and native details isolated in an adapter.

## Use cases

- **Compose custom decoding mechanics.** Run ordered full-vocabulary or bounded
  sparse logit transforms before native sampling. Candidate changes commit only
  when every stage succeeds, so an error or panic cannot leave a partial
  write-back.
- **Probe and intervene at exact model sites.** Capture bounded residual or
  router rows, accumulate caller-labelled vectors, and apply ordered
  position-scoped tensor operations through an exact topology/backend profile.
  The receipts establish what ran, not what a direction means.
- **Compare target and draft mechanics directly.** Run MTP or EAGLE-3 with the
  target model as sole causal authority, retain proposed/accepted/rejected
  accounting, and resolve provisional tensor telemetry before observers see
  admitted tokens.
- **Instrument generation at the causal boundary.** Observe arbitrary token
  bytes only after native admission, then implement logging, counters,
  cooperative stops, or application-specific control flow without assuming
  that each token is valid UTF-8.
- **Recover structured projection attempts exactly.** Bind caller-owned
  compiler, validator, grammar, and byte-feedback identities; restore one
  causal checkpoint after cancellation or rejection; and let the caller
  explicitly choose each retry without an internal retry loop.
- **Record mechanical provenance.** Serialize bounded plans and retain
  content-bound receipts for configuration, lineage, and token-accounting
  checks. Receipts describe what mechanics ran; they do not judge the generated
  content.
- **Keep tools independent of one model backend.** Build planning, transform,
  and observation components against the foundational crates, then connect
  them to llama.cpp through the adapter or supply another backend integration.
- **Run resumable local inference workers.** Capture and restore opaque
  llama.cpp state with model, backend-build, allocation, and token-history
  identity checks rather than treating native state as a portable file format.
- **Manage steering resources explicitly.** Scope caller-supplied LoRA adapters
  or control vectors to one session and poison the session if cleanup fails,
  avoiding silent continuation with uncertain native state.
- **Build resident image workers without hiding their mechanics.** Bind exact
  prompt, source, mask, reference, LoRA, tensor, schedule, placement, and
  output-buffer identities to a serializable plan, then keep transport,
  admission, and retry policy in the downstream application.

## Crates

| Crate | Purpose |
| --- | --- |
| [`logit-loom-runtime`](crates/runtime) | Explicit higher-level model loading, exact text admission, generation, controls, checkpoints, and steering. |
| [`logit-loom-models`](models) | Pinned optional model profiles and path-free artifact verification receipts. |
| [`logit-loom-core`](crates/core) | Serializable token, sampling, steering, checkpoint, and receipt contracts. |
| [`logit-loom-executor`](crates/executor) | Transport-neutral worker-local lifecycle, borrowed-buffer, cancellation, cleanup, and failure contracts. |
| [`logit-loom`](crates/loom) | Safe transform pipelines, observer fan-out, cancellation, and first-party transforms. |
| [`logit-loom-llamacpp`](crates/llamacpp) | llama.cpp model/session integration through `llama-cpp-4`. |
| [`logit-loom-diffusion`](crates/diffusion) | Backend-neutral diffusion plans, checkpoints, transactional state interventions, observations, and versioned whole-image graphs. |
| [`logit-loom-diffusion-sdcpp`](crates/diffusion-sdcpp) | Safe single-owner adapter and whole-plan lowerer for the versioned stable-diffusion.cpp companion ABI. |
| [`logit-loom-tokenizer`](crates/tokenizer) | Unpublished safe ranked-BPE, direct sink/count, dedicated-pool, batching, chunking, cancellation, cache, and oracle-qualification mechanics. |

The backend-neutral crates contain no model runtime. Applications can use the
token contracts with another text backend or the diffusion contracts with
another tensor runtime at their documented boundaries.

## Getting started

Start with `logit-loom-runtime` for an explicit local llama.cpp workflow.
Start with `logit-loom-core` when an application only needs serializable plans
and receipts. Add `logit-loom` for backend-neutral executable transforms and
observers. Use `logit-loom-llamacpp` directly when the application must own the
native runtime, model, and session objects separately.

The [getting-started guide](docs/getting-started.md) walks through dependency
selection, the transform lifecycle, exact-byte observation, and opt-in native
execution. Each crate README is also its compiled crate-level rustdoc, so its
examples are checked as doctests.

For an in-process image backend or downstream resident worker, see the
[worker-local image execution guide](docs/image-execution.md).

## Optional model profiles

The [profile integration plan](NEXT_STEPS.md) covers three caller-fetched
profiles: Qwen3 0.6B Q8_0 for small text experiments, MiniT2I-B/16 for compact
direct-RGB image experiments, and Krea 2 Turbo for advanced latent
experiments.

The repository now has a machine-checked
[acquisition catalog](models/README.md) with exact upstream revisions, selected
files, byte counts, weight hashes, and license locations:

```sh
cargo run --quiet -p logit-loom-xtask -- models list
cargo run --quiet -p logit-loom-xtask -- models fetch qwen3-0.6b-q8-0 \
  --dir /path/to/model-store \
  --dry-run
```

Qwen has an exact
[profile loader and replay runbook](docs/runbooks/06-qwen-profile-replay.md).
MiniT2I and Krea share a reviewed, versioned stable-diffusion.cpp adapter and
have complete [image fork](docs/runbooks/07-minit2i-fork.md) and
[latent transplant](docs/runbooks/08-krea2-latent-transplant.md) runbooks.
All three profiles have
[retained, output-free accelerator reports](docs/acceptance/README.md) that
pass their mechanical acceptance gates and are checked in as
**first-class** with `passed` acceptance status. Krea's report binds the exact
license and runtime components, selected Vulkan device, checkpoint replay,
bounded latent intervention, step timings, and qualified deployment-memory
observations. No acquisition command runs during tests or CI, and no model
code, weights, prompts, or generated images are bundled.

## A local runtime

The higher-level crate keeps the consequential choices visible: the caller
supplies a local model, exact text, tokenization flags, session allocation,
backend feature, and generation bound.

```rust
use logit_loom_runtime::{
    GenerationRequest, Loom, LoomOptions, Tokenization,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let loom = Loom::load("model.gguf", LoomOptions::default())?;
    let request = GenerationRequest::new(64)?;
    let tokenization = Tokenization { add_bos: true };
    let output = loom.complete("The creature opened its eyes and", tokenization, request)?;
    let _exact_generated_bytes = output.bytes();
    Ok(())
}
```

Token pieces are arbitrary bytes. The runtime never downloads a model, selects
a chat template, or silently changes a rejected accelerator placement to
CPU-only inference. See the
[runtime interface guide](docs/runtime-interface.md) for one-shot and stateful
operation, controls, identities, and escape hatches.

## A small loom

```rust
use logit_loom::{
    CandidateMode, Digest, LogitTransform, Pipeline, Stage, TokenId,
    TransformContext, TransformError, TransformSpec,
};

struct LiftToken(TokenId);

impl LogitTransform for LiftToken {
    fn apply(&mut self, mut context: TransformContext<'_>) -> Result<(), TransformError> {
        for (token, logit) in context.candidates_mut() {
            if token == self.0 {
                *logit += 1.5;
            }
        }
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let spec = TransformSpec::new(
        Digest::of_bytes("example-transform", b"lift-token-42-v1"),
        CandidateMode::FullVocabulary,
        32,
    )?;
    let mut loom =
        Pipeline::new(vec![Stage::new(spec, LiftToken(TokenId::new(42)?))?])?;

    loom.begin(&[])?;
    let mut logits = vec![0.0; 128];
    loom.apply_to_vocabulary(0, &[], &mut logits)?;
    assert_eq!(logits[42], 1.5);
    Ok(())
}
```

Pipelines are ordered and call-scoped. Each stage is content-identified,
callback failures and panics are contained, and execution produces mechanical
receipts suitable for inspection or replay checks.

## Current functionality

- Full-vocabulary and bounded sparse candidate exposure.
- Transactional execution over either a raw vocabulary or a backend-selected
  candidate scratch view.
- Ordered, stateful Rust logit-transform pipelines.
- Generated-token observation and ordered observer fan-out.
- Cooperative cancellation at explicit safe boundaries.
- Exact byte token pieces rather than assumed UTF-8 fragments.
- Native sampling plans for greedy, temperature, top-k, top-p, min-p, typical,
  repetition, DRY, Mirostat, logit bias, and grammar composition.
- llama.cpp causal prefill, generation, checkpoint/restore, ordered aggregate
  LoRA/control-vector scopes, and fail-closed cleanup.
- Whole-operation `TextMechanicsPlanV2` execution across bounded prefill,
  transforms, observers, ordered target steering, activation, ordinary or
  target-authoritative speculative generation, exact branch restore,
  cooperative cancellation, checkpoint capture, and aggregate receipts.
- Topology-bound activation capture, deterministic content-free vector
  accumulation, and transactional scaled-add/projection-removal programs.
- One-sequence target-authoritative MTP and EAGLE-3 generation with exact
  proposal boundaries, rejected-suffix rollback, optional independent
  target/draft activation programs, reusable process-local checkpoint
  branches, and no ordinary-generation fallback.
- An explicit higher-level local runtime with separate replace/append
  admission, bounded one-shot and stateful generation, control builders, and
  compatibility identity access.
- Bounded diffusion tensor, schedule, plan, checkpoint, intervention,
  observer, and receipt contracts that remain distinct from token mechanics.
- Serializable whole-image plans plus a transport-neutral local-executor seam
  over exact borrowed inputs, caller-owned outputs, lifecycle state, cleanup
  receipts, and classified failures.
- A default-built resident image-program contract over typed single-assignment
  values, multiple native/VAE stages, deterministic joins, checkpoint
  conversion, liveness-derived arena bounds, stage-prefix receipts, and
  deployment measurements kept outside deterministic identity. The mandatory
  stable-diffusion.cpp program ABI implements native execution, scheduled
  adapters, independently sized bounded reference images, RGB8/RGBA8/PNG
  outputs, exact-step loaded-topology-validated Krea residual block controls,
  and exact arena cleanup; model-backed acceptance remains a separate opt-in
  boundary.
- An exact dynamic companion ABI for MiniT2I and Krea with transactional
  post-Euler state callbacks, deterministic-prefix checkpoints, explicit
  accelerator placement, per-step native timing outside deterministic
  receipts, and no CPU-only retry.
- Image ABI v2 text-to-image, image-to-image, inpaint, outpaint, reference
  images, fixed request-local LoRA stacks, direct Krea VAE encode/decode, and
  verified native LoRA participation.
- Content-bound plans and mechanical execution receipts.

See [architecture](docs/architecture.md) for boundaries and
[compatibility](docs/compatibility.md) for the native dependency policy. The
[capability status](docs/capabilities.md) distinguishes in-memory behavioral
tests, adapter compilation, and opt-in model execution.

Runnable examples that require no model:

```sh
cargo run -p logit-loom-core --example generation_plan
cargo run -p logit-loom --example token_bias
cargo run -p logit-loom --example observe_tokens
cargo run -p logit-loom-runtime --example mechanics
cargo run -p logit-loom-diffusion-sdcpp --example probe_companion -- \
  /path/to/libstable-diffusion.so
```

Model-backed runtime examples require a caller-supplied local GGUF and an
explicit backend feature:

```sh
cargo run -p logit-loom-runtime --example generate \
  --features vulkan -- /path/to/model.gguf "Prompt"
cargo run -p logit-loom-runtime --example branch \
  --features vulkan -- /path/to/model.gguf "Branch from here:"
cargo run -p logit-loom-llamacpp --example speculative_mtp \
  --features vulkan -- /path/to/mtp-model.gguf "Draft from here:"
```

The MTP example requires a compatible GGUF reporting native NextN heads. It
uses the same exact model bytes for target and draft contexts, keeps target
sampling causally authoritative, and reports proposal accounting on standard
error.

## End-to-end experiment runbooks

Eight [mechanical experiment runbooks](docs/runbooks/README.md) take a specific
token-stream or diffusion-state question from caller-supplied artifacts
through execution, inspection, success criteria, and failure diagnosis:

| Runbook | Mechanical question | Main surfaces |
| --- | --- | --- |
| [Fork and jolt](docs/runbooks/01-fork-and-jolt.md) | What happens when the same checkpoint is replayed with one ordered logit transform? | Checkpoints, full-vocabulary rank bias, pipeline receipts |
| [Token byte microscope](docs/runbooks/02-token-byte-microscope.md) | Do post-admission observer events reconstruct the exact generated bytes? | Arbitrary token bytes, causal positions, observer receipts |
| [Exact byte tripwire](docs/runbooks/03-exact-byte-tripwire.md) | At which admitted token does an exact byte suffix stop generation? | Cross-token byte stops, terminal selection, bounded generation |
| [Causal circuit breaker](docs/runbooks/04-causal-circuit-breaker.md) | Can one observer trigger cancellation that another sees at the same safe boundary? | Observer ordering, cooperative cancellation, retained causal work |
| [LoRA transplant](docs/runbooks/05-lora-transplant.md) | Can steering be applied and cleared between deterministic checkpoint replays? | Scoped LoRA, cleanup receipts, checkpoint isolation, session health |
| [Qwen profile replay](docs/runbooks/06-qwen-profile-replay.md) | Does the pinned small text model replay one exact checkpoint? | Artifact verification, accelerator placement, exact token replay |
| [MiniT2I image fork](docs/runbooks/07-minit2i-fork.md) | Does one direct-RGB checkpoint replay unchanged and accept one bounded state operation? | Companion ABI, diffusion checkpoint, channel bias, image identities |
| [Krea 2 latent transplant](docs/runbooks/08-krea2-latent-transplant.md) | Does one pinned latent checkpoint replay unchanged and accept one bounded channel operation? | Gated components, latent lineage, exact plan identity, image identities |

Each runbook is backed by a compiled Rust example. Successful runs write a
structured JSON report containing exact artifact/runtime identities, plans,
placement evidence, execution receipts, and—where available—non-deterministic
deployment measurements. General text reports may include exact generated
bytes and token IDs for local inspection; image reports retain only pixel
identities and write bytes to caller-selected PPM files. Cargo and native
diagnostics remain on standard error, so reports can be retained with `tee`
and inspected with `jq`.

The examples never download a model, choose a prompt template, or silently
retry rejected accelerator placement as CPU-only inference. Output differences
are observations, not evidence of model quality or efficacy; unchanged output
does not by itself mean that a transform or steering lifecycle failed.
Output-free retained acceptance evidence follows the
[model-run schema](docs/acceptance/model-run.schema.json).

## Status

The API is pre-1.0 and may change between minor releases. The backend-neutral
crates are intended to remain small. Native backend churn is isolated in
adapter crates.

No model weights, generated output corpus, adapters, or control vectors are
included. Native tests that execute a model are opt-in and must use an
explicitly supplied local model and backend feature.

## Development

The pinned toolchain is Rust 1.97.1.

```sh
make check
make doc
make package-list
```

`make check-core` validates the backend-neutral crates without compiling
llama.cpp. `make package` performs staged Cargo package verification and
requires a clean checkout plus already-indexed foundational dependencies. See
[CONTRIBUTING.md](CONTRIBUTING.md) and the
[release process](docs/releasing.md) for the complete workflows.

## Contributing and security

Focused bug reports and generally useful token-stream primitives are welcome.
See [CONTRIBUTING.md](CONTRIBUTING.md) for scope, validation, and inbound
licensing. Report vulnerabilities privately as described in
[SECURITY.md](SECURITY.md), not through a public issue.

## License

Logit Loom is available under your choice of the MIT License or Apache License
2.0. Separate proprietary/commercial licensing is also available from
Blackcat Informatics Inc. See [LICENSING.md](LICENSING.md).

Project logo and social-sharing assets are documented in the
[brand guide](docs/BRAND.md).

# Logit Loom

<p align="center">
  <img src="docs/logit-loom-logo.svg" alt="Logit Loom black cat holding a token-transform loom" width="260">
</p>

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSING.md)
[![CI](https://github.com/paudley/logit-loom/actions/workflows/ci.yml/badge.svg)](https://github.com/paudley/logit-loom/actions/workflows/ci.yml)

Logit Loom is a Rust toolkit for observing, transforming, steering, stopping,
resuming, and accounting for token generation. It provides backend-neutral
mechanical contracts and a llama.cpp adapter without prescribing what a model
should say or think.

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
- **Instrument generation at the causal boundary.** Observe arbitrary token
  bytes only after native admission, then implement logging, counters,
  cooperative stops, or application-specific control flow without assuming
  that each token is valid UTF-8.
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

## Crates

| Crate | Purpose |
| --- | --- |
| [`logit-loom-runtime`](crates/runtime) | Explicit higher-level model loading, exact text admission, generation, controls, checkpoints, and steering. |
| [`logit-loom-core`](crates/core) | Serializable token, sampling, steering, checkpoint, and receipt contracts. |
| [`logit-loom`](crates/loom) | Safe transform pipelines, observer fan-out, cancellation, and first-party transforms. |
| [`logit-loom-llamacpp`](crates/llamacpp) | llama.cpp model/session integration through `llama-cpp-4`. |

The two foundational crates contain no model runtime. Applications can use
them with a different inference backend by adapting candidate logits and token
events at the documented boundaries.

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
- llama.cpp causal prefill, generation, checkpoint/restore, scoped LoRA, and
  scoped control-vector integration.
- An explicit higher-level local runtime with separate replace/append
  admission, bounded one-shot and stateful generation, control builders, and
  compatibility identity access.
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
```

Model-backed runtime examples require a caller-supplied local GGUF and an
explicit backend feature:

```sh
cargo run -p logit-loom-runtime --example generate \
  --features vulkan -- /path/to/model.gguf "Prompt"
cargo run -p logit-loom-runtime --example branch \
  --features vulkan -- /path/to/model.gguf "Branch from here:"
```

## End-to-end experiment runbooks

Five [mechanical experiment runbooks](docs/runbooks/README.md) take a specific
token-stream question from caller-supplied artifacts through execution,
inspection, success criteria, variations, and failure diagnosis:

| Runbook | Mechanical question | Main surfaces |
| --- | --- | --- |
| [Fork and jolt](docs/runbooks/01-fork-and-jolt.md) | What happens when the same checkpoint is replayed with one ordered logit transform? | Checkpoints, full-vocabulary rank bias, pipeline receipts |
| [Token byte microscope](docs/runbooks/02-token-byte-microscope.md) | Do post-admission observer events reconstruct the exact generated bytes? | Arbitrary token bytes, causal positions, observer receipts |
| [Exact byte tripwire](docs/runbooks/03-exact-byte-tripwire.md) | At which admitted token does an exact byte suffix stop generation? | Cross-token byte stops, terminal selection, bounded generation |
| [Causal circuit breaker](docs/runbooks/04-causal-circuit-breaker.md) | Can one observer trigger cancellation that another sees at the same safe boundary? | Observer ordering, cooperative cancellation, retained causal work |
| [LoRA transplant](docs/runbooks/05-lora-transplant.md) | Can steering be applied and cleared between deterministic checkpoint replays? | Scoped LoRA, cleanup receipts, checkpoint isolation, session health |

Each runbook is backed by a compiled `logit-loom-runtime` example. Successful
runs write a structured JSON report containing the serialized generation plan,
model and backend identities, reported devices, placement and session
allocation, exact generated bytes and token IDs, and the relevant execution
receipts. Cargo and native diagnostics remain on standard error, so reports can
be retained with `tee` and inspected with `jq`.

The examples never download a model, choose a prompt template, or silently
retry rejected accelerator placement as CPU-only inference. Output differences
are observations, not evidence of model quality or efficacy; unchanged output
does not by itself mean that a transform or steering lifecycle failed.

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

<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Getting started

Logit Loom is split so applications can depend on the smallest useful layer.
None of the ordinary examples download or execute a model.

## Choose a crate

| Need | Crate |
| --- | --- |
| Serializable token, sampler, steering, checkpoint, and receipt contracts | `logit-loom-core` |
| Safe transform pipelines, observers, and cancellation | `logit-loom` |
| A causal llama.cpp session and native sampler adapter | `logit-loom-llamacpp` |

The `logit-loom` crate re-exports the core contracts. The llama.cpp adapter
depends on both foundational crates.

```toml
[dependencies]
# Choose the smallest layer that provides the APIs you need.
logit-loom-core = "=0.1.1"
# logit-loom = "=0.1.1"
```

When using the adapter, depend directly on `logit-loom` for plan and callback
types and keep the workspace versions aligned and exact:

```toml
[dependencies]
logit-loom = "=0.1.1"
logit-loom-llamacpp = { version = "=0.1.1", features = ["vulkan"] }
```

## Try backend-neutral mechanics

The in-memory examples exercise real public APIs without loading a model:

```sh
cargo run -p logit-loom-core --example generation_plan
cargo run -p logit-loom --example token_bias
cargo run -p logit-loom --example observe_tokens
```

A transform call follows this lifecycle:

1. Construct every `TransformSpec`, then construct the ordered `Pipeline`.
2. Call `Pipeline::begin` with the exact causal token prefix.
3. Call `apply_to_vocabulary` or `apply_to_candidates` with consecutive
   zero-based steps.
4. Commit the selected token to backend causal state.
5. Call `Pipeline::accept` once for that unmatched successful step, with the
   admitted token.
6. Read the mechanical receipt after the call finishes.

Candidate mutations remain in a scratch view until every stage succeeds. A
callback error, panic, step-bound violation, or `NaN` leaves the caller's
candidate logits unchanged for that invocation.

## Observe admitted tokens

Generated-token observers have two boundaries:

1. `ObserverSet::poll` runs before a sampling decision.
2. `ObserverSet::observe` runs after the selected token is decoded into causal
   state.

An `ObservedToken` carries the exact token piece as `&[u8]`. Buffer pieces as
bytes and decode only the complete application-level unit; one token piece
need not be valid UTF-8.

Both observer stop requests and cancellation are cooperative. The backend
retains causal work already admitted before the boundary where it notices the
request.

## Build a generation plan

`GenerationPlan::validate` checks sampler numbers, collection sizes, grammar
strings, and exact byte stops. `GenerationPlan::digest` validates first and
then hashes deterministic serialized bytes under a versioned domain.

Receipts use the same pattern. Their identities establish exact mechanics and
lineage; they are not signatures and do not establish output quality,
truthfulness, or fitness for a workload.

## Use llama.cpp explicitly

The adapter forwards backend features without enabling one by default. A live
example requires a caller-supplied local GGUF and an explicitly selected
backend:

```sh
cargo run -p logit-loom-llamacpp --example generate \
  --features vulkan -- /path/to/model.gguf "Prompt"
```

`ModelOptions::default` requires accelerator participation. Logit Loom rejects
a load with no reported accelerator device instead of silently retrying
CPU-only inference. Record the selected Cargo feature and `Model::devices`
alongside model-backed acceptance results.

Checkpoint state is opaque and bound to the model bytes, adapter build, and
exact session allocation options. Keep it within a controlled deployment; do
not treat it as a portable file format.

## Next reading

- [Architecture](architecture.md) defines ordering and failure boundaries.
- [Capabilities](capabilities.md) distinguishes in-memory tests, adapter
  compilation, and opt-in model execution.
- [Compatibility](compatibility.md) covers Rust, native features, checkpoints,
  and artifact assumptions.
- [Contributing](../CONTRIBUTING.md) lists the complete validation workflow.

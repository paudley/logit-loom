<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Getting started

Logit Loom is split so applications can depend on the smallest useful layer.
None of the ordinary examples download or execute a model.

## Choose a crate

| Need | Crate |
| --- | --- |
| Explicit higher-level local model loading and generation | `logit-loom-runtime` |
| Serializable token, sampler, steering, checkpoint, and receipt contracts | `logit-loom-core` |
| Safe transform pipelines, observers, and cancellation | `logit-loom` |
| A causal llama.cpp session and native sampler adapter | `logit-loom-llamacpp` |
| Transport-neutral worker-local lifecycle and exact buffers | `logit-loom-executor` |
| Serializable diffusion plans, whole-image programs, and state operations | `logit-loom-diffusion` |
| Pinned MiniT2I/Krea execution through stable-diffusion.cpp | `logit-loom-diffusion-sdcpp` |
| Exact optional model catalog and artifact receipts | `logit-loom-models` |

The `logit-loom` crate re-exports the core contracts. The llama.cpp adapter
depends on both foundational crates. The runtime crate composes all three and
re-exports the common caller-facing types.

```toml
[dependencies]
# Choose the smallest layer that provides the APIs you need.
logit-loom-core = "=0.2.0"
# logit-loom = "=0.2.0"
```

When using the adapter, depend directly on `logit-loom` for plan and callback
types and keep the workspace versions aligned and exact:

```toml
[dependencies]
logit-loom = "=0.2.0"
logit-loom-llamacpp = { version = "=0.2.0", features = ["vulkan"] }
```

For the higher-level local workflow, select the native backend through the
runtime crate:

```toml
[dependencies]
logit-loom-runtime = { version = "=0.2.0", features = ["vulkan"] }
```

## Try backend-neutral mechanics

The in-memory examples exercise real public APIs without loading a model:

```sh
cargo run -p logit-loom-core --example generation_plan
cargo run -p logit-loom --example token_bias
cargo run -p logit-loom --example observe_tokens
cargo run -p logit-loom-runtime --example mechanics
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

## Use the higher-level local runtime

`Loom::load` owns one process runtime and one caller-supplied local model.
`Loom::complete` creates a fresh session for one exact prompt replacement and
bounded generation call. `Loom::session` exposes separate replace, append,
generate, checkpoint, and steering operations when causal state must persist.

The façade does not select a chat template or retry rejected accelerator
placement as CPU-only inference. Generation output remains exact bytes, and
custom transform or observer callbacks require a caller-defined stable
implementation digest.

```sh
cargo run -p logit-loom-runtime --example generate \
  --features vulkan -- /path/to/model.gguf "Prompt"

cargo run -p logit-loom-runtime --example branch \
  --features vulkan -- /path/to/model.gguf "Branch from here:"
```

The [mechanical experiment runbooks](runbooks/README.md) turn eight specific
text and image questions into complete commands, JSON evidence reports,
success criteria, and failure diagnosis. Model execution remains opt-in and
uses only caller-supplied local artifacts.

See the [runtime interface guide](runtime-interface.md) for ownership,
ordering, automatic identities, checkpoints, steering, and the lower-level
escape hatch.

## Use the llama.cpp adapter directly

The adapter forwards backend features without enabling one by default. A live
example requires a caller-supplied local GGUF and an explicitly selected
backend:

```sh
cargo run -p logit-loom-llamacpp --example generate \
  --features vulkan -- /path/to/model.gguf "Prompt"

cargo run -p logit-loom-llamacpp --example speculative_mtp \
  --features vulkan -- /path/to/mtp-model.gguf "Draft from here:"
```

`ModelOptions::default` requires accelerator participation. Logit Loom rejects
a load with no reported accelerator device instead of silently retrying
CPU-only inference. Record the selected Cargo feature and `Model::devices`
alongside model-backed acceptance results.

The MTP example requires one GGUF that reports native NextN heads. MTP uses
two contexts over the same exact model; EAGLE-3 instead requires separately
loaded, vocabulary-compatible target and `eagle3` draft models. The draft must
name exactly three valid target extraction layers. Both mechanisms preserve
target sampling as causal authority, expose no ordinary-generation fallback,
and currently support one sequence through the high-level operation.

Checkpoint state is opaque and bound to the model bytes, adapter build, and
exact session allocation options. Keep it within a controlled deployment; do
not treat it as a portable file format.

## Use the diffusion contracts and adapter

`logit-loom-diffusion` keeps iterative tensor state distinct from token IDs and
logits. It provides model-free plans, exact schedules, typed tensor boundaries,
ordered transactional interventions, observers, checkpoints, whole-image
execution contracts, and receipts. `logit-loom-executor` supplies the
transport-neutral lifecycle, buffer, cancellation, cleanup, and failure seam
used by local workers.

`logit-loom-diffusion-sdcpp` binds those contracts to companion ABI version 2
plus image extension version 3 for the exact catalogued MiniT2I and Krea
component layouts. The caller builds the pinned native companion, supplies
every artifact path, selects an exact non-CPU backend, and chooses the prompt,
seed, shape, guidance, and custom Euler schedule. `ImagePlanExecutor` lowers
the supported `ImageExecutionPlanV3` subset through one resident owner; see
the support matrix before constructing operators, routes, or checkpoints.

```toml
[dependencies]
logit-loom-diffusion = "=0.2.0"
logit-loom-diffusion-sdcpp = "=0.2.0"
```

Probe the native contract without loading a model:

```sh
cargo run --quiet -p logit-loom-diffusion-sdcpp \
  --example probe_companion -- \
  /path/to/libstable-diffusion.so
```

Then follow the [MiniT2I image fork](runbooks/07-minit2i-fork.md) or
[Krea 2 latent transplant](runbooks/08-krea2-latent-transplant.md) runbook.
The examples never download weights and never retry an unavailable accelerator
on CPU.

See [worker-local image execution](image-execution.md) for the whole-image
support matrix, lifecycle, failure dispositions, and current ABI limitations.

## Next reading

- [Architecture](architecture.md) defines ordering and failure boundaries.
- [Capabilities](capabilities.md) distinguishes in-memory tests, adapter
  compilation, and opt-in model execution.
- [Compatibility](compatibility.md) covers Rust, native features, checkpoints,
  and artifact assumptions.
- [Runtime interface](runtime-interface.md) documents the higher-level local
  workflow and its explicit boundaries.
- [Mechanical experiment runbooks](runbooks/README.md) provide eight opt-in
  model-backed workflows with structured evidence reports.
- [Contributing](../CONTRIBUTING.md) lists the complete validation workflow.

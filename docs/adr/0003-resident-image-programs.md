<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# ADR 0003: resident staged image programs

- Status: accepted; backend-neutral contract and safe execution driver implemented, native execution pending
- Decision date: 2026-07-27
- Builds on: [ADR 0001](0001-stable-diffusion-runtime.md)
- Initial runtime family: pinned stable-diffusion.cpp companion
- Initial sampler boundary: Euler only

## Context

ADR 0001 selected a pinned stable-diffusion.cpp revision and a narrow companion
ABI for exact post-Euler scheduler-state mechanics. The current adapter has
since added text-to-image, image-to-image, inpaint, outpaint, reference
images, fixed request-local `LoRA` stacks, direct VAE encode/decode,
checkpoints, scheduler-state operators, observations, deterministic RGB8 mask
blending, explicit outputs, cancellation, and cleanup receipts.

The public `ImageExecutionPlanV3` still has one native diffusion primary.
Compositing follows that primary, and another native inference operation
requires another request. That boundary cannot express a resident workflow
which generates a base image, refines it, runs independent spatial repairs,
merges those repairs deterministically, and performs a final consistency
pass.

The backend-neutral image contract already describes scheduled `LoRA` scales,
conditioning and model-block selectors, direct VAE operations, tensor
observations, and multiple output representations. The current
stable-diffusion.cpp adapter rejects the portions its companion ABI cannot
execute. Adding those mechanics must use a new exact program and ABI identity
rather than broadening a previously advertised support matrix.

Passing every intermediate image or latent through a coordinator would add
avoidable copies, expose model-local representations outside their worker,
and split one mechanical transaction into loosely related requests.
Conversely, the reviewed stable-diffusion.cpp tensor boundary does not prove
that every intermediate is accelerator-resident. The next interface must keep
values native-owned while reporting actual placement honestly.

## Decision evidence

| Requirement | Evidence at the current adapter |
| --- | --- |
| Resident model | One `Runtime` retains the loaded profile across operations |
| Primary operations | Text-to-image, image-to-image, inpaint, and outpaint lower through the advanced companion call |
| Native tensors | Direct VAE encode/decode exchanges bounded finite native-layout `f32` tensors |
| Exact iterative state | The post-Euler callback exposes a mutable contiguous host `f32` scheduler state |
| Conditioning | Native token IDs and conditioning tensors are synchronously identified |
| Fixed adapters | Request-local `LoRA` entries are verified as applied and explicitly cleared |
| Program contracts | Public schedules, selectors, operators, observations, routing, cleanup, and receipts are already versioned |
| Deterministic join | RGB8 mask blending has exact integer rounding and ordered receipt semantics |

This establishes that a resident program can be built by extending the
companion and adapter. It does not establish device-only latent residency,
memory fit, throughput, visual quality, or equivalence to another runtime.

## Decision

Add a new `ImageProgramPlanV1` and `ImageProgramReceiptV1`. Preserve
`ImageExecutionPlan`, `ImageExecutionPlanV2`, `ImageExecutionPlanV3`, their
receipts, and every existing digest domain unchanged.

An image program is a bounded ordered graph over typed single-assignment value
slots. Serialization declares logical values and dependencies; native runtime
handles never appear in a plan or receipt.

The initial stage set contains:

- text-to-image;
- image-to-image;
- inpaint and outpaint;
- VAE encode and decode;
- deterministic RGB8 mask blend;
- checkpoint restore and capture;
- installed tensor intervention and observation; and
- explicit image, tensor, checkpoint, and receipt output routing.

Each diffusion stage binds its own conditioning identities, seed, RNG,
schedule, guidance, strength, ordered `LoRA` stack and scale schedules,
operators, observations, checkpoint mechanics, and output value contract.
Shared artifacts may be loaded once by the resident executor, but every
stage records the exact mechanics it used.

### Values and graph validation

`ImageValueSpecV1` describes the exact logical representation of each external
or stage-produced value:

- tightly described RGB8, RGBA8, or Gray8 bytes;
- a typed tensor with dtype, shape, layout, and representation identity;
- opaque authenticated checkpoint bytes; or
- exact backend-native conditioning identified by a compatibility domain.

Every value slot has one producer. A stage may reference only external inputs
or values produced by earlier stages. Dependencies are derived from those
references rather than repeated in a second caller-maintained edge list.
Stage inputs are immutable; a stage always produces new value slots.

Validation rejects before allocation or native calls:

- forward references, duplicate producers, cycles, and unused external
  inputs;
- incompatible image geometry, mask geometry, tensor shape, dtype, layout, or
  representation;
- output aliases and multiple mutable consumers;
- unsupported stage/operator/profile combinations;
- invalid or non-canonical schedules;
- excessive stages, values, inputs, outputs, tensor elements, or retained
  bytes; and
- a liveness-derived scratch bound above the public maximum.

The executor computes value liveness before execution. A native value is
released immediately after its final stage or output consumer unless an
explicit retained output requires materialization.

### Branching and deterministic joins

Independent repairs may consume the same earlier image. Their outputs are
joined with ordered `MaskBlend` stages before a later native operation.

For each tightly packed RGB8 channel:

`(base * (255 - mask) + overlay * mask + 127) / 255`

A zero mask byte selects the base and 255 selects the overlay. When multiple
repair masks overlap, later declared blend stages take precedence. The stage
order and every input, mask, intermediate, and result identity appear in the
receipt.

This supports controlled branch ablation while retaining an unambiguous final
input. The library does not choose regions, create masks, interpret a repair,
or assess the result.

## Native value arena

Add a version-three stable-diffusion.cpp companion ABI for one request-scoped
program over a resident context.

The companion owns a value arena. Each private handle contains an arena
generation and slot index and is validated against an exact value descriptor.
Handles:

- are never raw pointers;
- cannot be serialized or sent over IPC;
- cannot cross a runtime or request epoch;
- are rejected after release; and
- are all invalidated by program cleanup.

The arena may own host images, native tensors, conditioning, or
backend-specific values. The companion reports the actual placement of each
value and every host/device transfer. Native ownership promises that
intermediates do not round-trip through Rust or a coordinator; it does not
promise accelerator-only residency.

Deterministic plan and receipt identities exclude elapsed time and deployment
transfer measurements. A separate measurements record reports per-stage
wall time, native compute time where available, peak arena bytes, value
placement, and transfer counts/bytes.

## Scheduled LoRA mechanics

The existing canonical `ScaleSchedule` remains the public per-stage contract.
Each schedule begins at step zero and changes scale at strictly increasing
zero-based pre-denoiser boundaries.

The companion:

1. loads each exact adapter once into the request-local resident set;
2. verifies that it participates in at least one intended native model tensor;
3. applies the declared ordered scale vector before each denoiser evaluation;
4. records every applied scale boundary;
5. rejects a schedule the native runtime cannot realize exactly; and
6. clears all adapter state before a reusable return.

An adapter may remain loaded as an immutable resident artifact cache, but its
applied scale is request-local state. A cleanup failure poisons the runtime.
No fixed-scale request is reinterpreted as scheduled, and no schedule is
approximated by fewer native updates.

## Tensor intervention and observation

The companion adds exact profile-defined hook sites for scheduler state,
conditioning tensors, and selected model-block nodes. A selector binds the
profile, companion ABI, upstream runtime revision, component, block, site,
dtype, shape, layout, and callback timing.

At a mutable hook the adapter copies the selected tensor into bounded
Rust-owned storage, executes installed operators in order, validates the
complete result, and writes back exactly once. Errors and panics produce no
partial write-back and cannot unwind through C. Digest and statistics
observations run after installed interventions at the same boundary;
snapshots route only to an explicitly sized output value.

Another runtime revision, graph name, tensor layout, or hook timing requires a
new selector implementation identity. Unsupported selectors fail before the
program begins.

## Transaction, cancellation, and checkpoint semantics

A program has one terminal result:

- completed;
- cancelled before start;
- cancelled after an exact stage or post-Euler boundary;
- failed at an exact stage or callback; or
- cleanup uncertain.

The receipt contains the completed stage prefix, value identities,
observations, checkpoints, initialized output prefixes, terminal boundary,
and cleanup disposition. A failed stage does not publish its output slots.
Previously completed immutable values remain evidence but are not returned
unless the plan explicitly routes partial output for that terminal.

Cancellation is checked before native start, between stages, and at each
supported post-Euler boundary. It never fabricates completion for an
unreached stage.

Diffusion checkpoints remain opaque post-Euler state bound to the exact stage,
model artifacts, conditioning, seed, RNG, schedule, adapter schedule,
operators, backend, ABI, and program lineage. A native arena handle is not a
checkpoint. Persisting an intermediate requires an explicit typed output
route and content identity.

Cleanup releases live arena values, clears callbacks and applied adapters, and
advances the runtime epoch. Any uncertainty poisons the runtime; later
programs cannot reuse it.

## Coordinator and worker boundary

A downstream coordinator exposes the program only under a new exact,
default-enabled capability and schema. Once the resident-program
implementation is present, support is not optional: the coordinator
advertises that capability or fails startup. It must not broaden a previously
advertised single-primary RGB8 capability.

The coordinator owns admission, queueing, verified sealed descriptors,
resource reservations, cancellation delivery, and sealed outputs. The worker
owns the resident model, value arena, native handles, execution, cleanup, and
mechanical receipt. Large prompts, images, masks, adapters, tensors, and
checkpoints cross the boundary only through exact verified descriptors.
Intermediate arena values remain worker-local.

Private source graphs, prompts, artifacts, profiles, and visual evaluation
remain downstream. The public repository supplies only generic mechanics and
synthetic fixtures.

## Safety contract

Local unsafe code remains confined to the private dynamic-ABI module of the
stable-diffusion.cpp adapter. Its `SAFETY.md` is extended for:

- version-three struct and symbol validation;
- arena-handle generation, ownership, and release;
- tensor pointer length, dtype, shape, layout, and lifetime;
- callback synchronization and transaction ordering;
- panic and error containment;
- native output ownership and deallocation; and
- complete cleanup and poisoning.

The backend-neutral diffusion and executor crates remain free of unsafe Rust.
No public API exposes a native pointer or unchecked handle.

## Alternatives considered

### Chain single-operation requests through a coordinator

This reuses current APIs but materializes every intermediate, adds IPC and
verification work, loses one resident transaction, and makes cleanup and
branch lineage harder to state exactly.

### Require every latent to remain accelerator-resident

The reviewed runtime exposes host-resident state at several useful boundaries.
Requiring device-only residency would turn an unverified performance goal
into a false compatibility promise. The accepted contract reports actual
placement and transfers.

### Execute repairs only as a sequential chain

Sequential execution is expressible by the program, but making it the only
join semantics couples otherwise independent stages and prevents clean branch
ablation. Ordered mask blending provides an exact built-in join.

### Permit arbitrary dynamically supplied operators

Unbounded code or unchecked graph selectors would defeat validation,
reproducibility, and callback containment. Operators are installed, versioned,
bounded, and compatibility-bound.

### Extend `ImageExecutionPlanV3` in place

V3 promises one native primary followed by deterministic compositing. Multiple
native stages and a native value arena are a different execution model and
receive a new plan family and digest domain.

## Validation and acceptance

Model-free contract and fake-executor tests cover:

- empty, extreme, oversized, cyclic, forward, duplicate, and type-invalid
  graphs;
- deterministic topological execution and value liveness;
- exact scratch accounting and early release;
- stale, cross-epoch, mismatched, and double-released handle rejection;
- RGB8 blend endpoints, rounding, overlapping-mask precedence, and arbitrary
  byte values;
- scheduled `LoRA` changes at exact pre-denoiser boundaries;
- ordered tensor operators and no write-back on error or panic;
- checkpoint lineage and mismatch rejection;
- cancellation before start, between stages, and after exact Euler steps;
- completed-stage receipt prefixes and output initialization;
- cleanup failure and poisoned-runtime behavior; and
- placement and transfer measurements excluded from deterministic identities.

The native preparation check compiles the exact version-three companion patch
against its pinned stable-diffusion.cpp revision. The model-free probe checks
required symbols, ABI, upstream commit, library-byte stability, supported
sites, handle limits, and bounded device reports.

Live acceptance is opt-in, uses caller-supplied local artifacts, requires a
reported accelerator, records exact profile and device placement, and never
falls back to CPU. It exercises:

1. one resident generation/refinement/repair/merge/final-pass program;
2. direct VAE and native-owned intermediate reuse;
3. scheduled `LoRA` boundaries;
4. one installed tensor intervention and observation;
5. branch checkpoint and deterministic join;
6. cancellation and confirmed cleanup; and
7. actual placement, transfer, and peak-memory reporting.

Successful execution establishes only the declared mechanical capability.
Visual assessment, model comparison, prompt selection, and efficacy remain
separate downstream evidence.

## Consequences

- One resident request can express multiple native diffusion and VAE stages
  with exact branch lineage and cleanup.
- Native-owned intermediates avoid coordinator and Rust round trips while
  retaining placement honesty.
- The companion ABI and safety surface become larger and require a new pinned
  implementation identity.
- Resource admission must include all resident artifacts, arena liveness,
  scheduled adapters, callbacks, and output buffers.
- Existing single-primary capabilities and receipts remain valid and narrow.
- Downstream adoption requires a new default-enabled capability, exact
  lowering, model-free return, explicitly authorized live mechanical
  acceptance, and separate human evaluation.

<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# ADR 0002: transactional text mechanics over llama.cpp

- Status: accepted; source implementation complete except persistent
  speculative checkpoint restore; live acceptance pending
- Decision date: 2026-07-27
- Reviewed binding: local `llama-cpp-4` 0.4.2 successor at
  `d76356b9725a3736212b3bfd16c66fc80c995c29`
- Reviewed llama.cpp revision: `f87067841bac583bc089a225382248d857791ca8`
- Delivery order: before ADR 0003

## Context

Before this decision, Logit Loom could compose logit transforms,
admitted-token observers, bounded generation, checkpoints, scoped `LoRA`
adapters, and one static llama.cpp control vector. The backend-neutral
`TextMechanicsPlanV1` also had identity fields for a draft implementation and
a speculative-token bound, but those contracts did not execute activation
capture, per-position activation programs, MTP, or EAGLE-3.

The reviewed safe binding successor exposes useful native mechanics:

- model layer count, embedding width, expert count, and NextN/MTP head count;
- copied `f32` transformer-layer and exact named-tensor capture through
  llama.cpp's evaluation callback;
- native control-vector application;
- MTP sessions over a target and same-model draft context; and
- EAGLE-3 sessions over a target and separately trained draft model.

The successor binding owns an exact-selector tensor transaction surface.
llama.cpp invokes the evaluation callback after synchronizing a selected graph
node and before evaluating later dependent nodes. The binding copies that
tensor into bounded Rust-owned storage, contains callback errors and
unwinding, and commits one complete finite write-back. Logit Loom does not
expose the native tensor pointer or add a second unsafe graph boundary locally.

Speculative execution also has more state than the target context alone. It
can include a draft context, deferred hidden rows, prompt carry-over, draft
sampler state, and implementation-specific state. The upstream common
speculation layer has an optional state interface. The successor patch adds
versioned, size-checked MTP and EAGLE-3 implementation state, including MTP
pending hidden rows and EAGLE-3 deferred boundary state. Target sampler state
remains an opaque in-process clone; portable sampler serialization is not
claimed.

The next text layer must support these mechanics without treating a tensor
direction, model layer, router choice, or acceptance rate as evidence about
model behavior or quality.

## Decision evidence

| Requirement | Evidence at the reviewed binding |
| --- | --- |
| Architecture validation | `n_layer`, `n_embd`, `n_expert`, and `n_layer_nextn` are available from the loaded model |
| Residual capture | `TensorCapture::for_layers` copies `l_out-N` tensors as exact `f32` rows |
| Named-node capture | Exact-name and prefix filters can retain graph nodes such as router logits, probabilities, and selected-expert tensors when the pinned graph exposes them |
| Mutation boundary | The native scheduler synchronizes a requested graph node before its completion callback, and later graph nodes have not yet executed |
| Existing steering | llama.cpp accepts one dense per-layer control-vector matrix and an inclusive layer range |
| MTP | The binding exposes process, draft, accept, topology, and multi-head NextN mechanics |
| EAGLE-3 | The binding exposes the same lifecycle over a separate compatible draft model |
| Native state | Target and draft contexts expose opaque state bytes; the common speculation layer has an optional implementation-state interface |

This establishes mechanical feasibility. It does not establish that a
particular tensor name is stable across llama.cpp revisions, that a captured
direction encodes a particular concept, that an intervention improves an
outcome, or that speculation increases throughput on a particular model.

## Decision

Add new backend-neutral activation, topology, speculation, and aggregate text
contracts. Preserve `ControlVectorSpec`, `TextMechanicsPlanV1`, their receipt,
and all existing digest domains unchanged.

The new public contract family consists of:

1. `TextModelTopologyV1`, binding the model artifact, backend build,
   architecture implementation, layer count, embedding width, expert
   topology when reported, and supported speculative mechanisms;
2. `TextTensorSiteV1`, selecting an exact layer output, router-logit tensor,
   router-probability tensor, selected-expert tensor, or
   backend-profile-defined named site;
3. `ActivationCapturePlanV1`, selecting sites, last-prefill-token or explicit
   inclusive causal positions, and bounded digest, statistics, or snapshot
   retention;
4. `ActivationVectorBankV1`, binding canonical sparse per-layer rows, their
   exact `f32` byte identity, width, site family, topology, normalization, and
   mechanical provenance;
5. `ActivationProgramV1`, declaring ordered, position-scoped tensor
   operations and observation requests;
6. `SpeculationPlanV1`, selecting MTP or EAGLE-3 and binding target, draft,
   topology, implementation, draft bounds, probability floor, and activation
   policy; and
7. `TextMechanicsPlanV2` and `TextMechanicsReceiptV2`, composing generation,
   transforms, observers, steering, activation, speculation, checkpoint
   lineage, terminal state, and cleanup evidence.

Every new serialized shape and identity uses a new versioned digest domain.
No V1 value is reinterpreted.

### Tensor sites and operators

Layer-output sites use stable numeric layer indices validated against the
loaded topology. Architecture-specific sites use an exact selector
implementation identity bound to the model architecture and backend build.
Changing llama.cpp graph names, node timing, dtype, shape, or meaning requires
a new selector implementation identity and renewed acceptance.

The first built-in mutable sites are finite `f32` residual layer outputs and
MoE router logits. Router probabilities and integer selected-expert tensors
are observable, but are not built-in mutable sites. Another mutable site
requires a separately versioned operator schema and adapter support.

The first built-in operators are:

- scaled vector addition, `x' = x + alpha * v`; and
- scaled projection removal,
  `x' = x - alpha * dot(x, v) * v`, where `v` is unit-normalized.

An operator binds exact IEEE-754 scale bits, vector-bank identity, selected
layers, selected causal positions, and execution phase. Sparse rows are
expanded only inside an adapter when a native API requires a dense matrix.
Missing rows are not inferred, and a vector bank never silently crosses a
model or topology identity.

Operators execute in declared order on a bounded Rust-owned copy. The adapter
writes the complete tensor back exactly once after every operator succeeds
and the result passes shape, dtype, layout, and numeric validation. An error
or panic causes no write-back.

### Capture and vector construction

The llama.cpp adapter adds a lifetime-safe probe session. For
last-prefill-token capture it decodes the bounded prompt prefix normally,
clears prior capture state, and decodes the final prompt token as a one-token
batch. This avoids retaining a full prompt-length copy for each selected
layer while preserving the exact selected causal row.

A content-free accumulator accepts already captured, topology-compatible
rows in caller-declared order and can produce deterministic group means,
paired differences, and normalized vector banks. It records counts, ordering,
input identities, accumulator implementation, and normalization outcome. It
does not retain prompts, assign semantic labels, choose datasets, or claim
what a resulting direction represents.

All capture counts, selected sites, layers, positions, elements, and retained
bytes are bounded and validated before allocation or decode.

## Speculation and causal authority

The target model remains the sole causal authority. MTP and EAGLE-3 may
propose token IDs, but a proposal becomes an admitted token only after target
verification and native causal admission.

The adapter records proposed, accepted, and rejected counts at each
speculation boundary. Tensor telemetry produced while evaluating proposals is
tagged as provisional and then as admitted or rejected. Existing token
observers receive only target-admitted token IDs and arbitrary token-piece
bytes. Rejected proposals and end-of-generation selections are never reported
as causal admissions.

Activation and speculation compose explicitly:

- `TargetOnly` applies the target activation program during target prefill,
  ordinary generation, and verification. The draft remains unmodified.
- `SeparateDraftProgram` binds a second activation program whose topology and
  sites must match the draft model. The same program may be named twice only
  when both topology validations independently succeed.

There is no implicit mirroring. In particular, an EAGLE-3 draft model is not
assumed to share the target width, layers, or tensor sites.

The adapter never falls back from MTP or EAGLE-3 to ordinary generation. A
missing head, incompatible draft, unsupported site, or unavailable
implementation fails before generation starts.

## Checkpoint semantics

A speculative checkpoint is captured only at a quiescent boundary after the
current proposal set has been completely accepted or rejected and both
speculative contexts are synchronized.

Its opaque envelope binds and contains, as applicable:

- target and draft model identities and opaque context state;
- exact backend and safe-binding build identities;
- speculative implementation, configuration, and opaque implementation state;
- all deferred target/draft rows, prompt carry-over, and draft sampler state;
- target sampling state and admitted token history;
- causal position and completed speculation boundary;
- active activation programs and vector-bank identities; and
- transform, observer, grammar, stop, and checkpoint lineage.

The safe binding must extend MTP and EAGLE-3 state access until a
capture-restore-capture differential test establishes that all continuation
state is included. Partial speculative state is unsupported rather than
silently omitted.

Restore creates compatible target and draft contexts, consumes every
checkpoint component, restores the quiescent speculative boundary, and
re-establishes the declared activation state before another decode.
Any size, identity, topology, state-accounting, or replay mismatch fails
closed. Native bytes remain opaque and non-portable.

## Safe binding and failure contract

The required successor to `llama-cpp-4` owns the native evaluation callback
and speculative-state shim. Its safe surface:

1. borrows callback state for the complete context lifetime;
2. accepts only bounded selectors and expected tensor contracts;
3. maps decode-batch rows to exact sequence and causal positions;
4. copies selected native tensors into owned storage;
5. catches Rust errors and unwinding before returning to C;
6. writes a successful transaction back through the native backend API;
7. exposes callback failure as a typed decode error; and
8. provides complete, size-checked speculative state get/set operations.

If native execution may have advanced after a callback or restore failure,
Logit Loom poisons the session. A poisoned session permits only explicit
cleanup or drop. The `Session` and every activation/speculation wrapper remain
single-owner and neither `Send` nor `Sync`, regardless of traits exposed by
the lower binding.

Foundational Logit Loom crates remain free of unsafe Rust. There is no raw
pointer, native tensor, or unchecked graph-node handle in the public API.

## Alternatives considered

### Continue using only llama.cpp control vectors

The existing API is suitable for one static dense matrix but does not express
bounded capture, sparse per-layer artifacts, per-position programs, router
sites, provisional verification rows, or complete speculative checkpoints.

### Expose a raw evaluation callback

This would move graph lifetime, synchronization, dtype, and pointer safety to
every caller. It would also make panic containment and transactional
write-back optional. The safe binding owns this boundary instead.

### Keep steering, speculation, and checkpoints mutually exclusive

This would simplify the first executor but leave the aggregate contract unable
to represent the intended mechanics. The accepted design supports their
composition by making target/draft application explicit and checkpoints
complete.

### Edit model weights before inference

Weight transformation is a different artifact-production workflow. It is not
reversible session mechanics and does not preserve the stock model identity.
It remains outside Logit Loom.

### Treat graph names as stable public semantics

Graph node names and timing are backend implementation details. Exact named
sites are therefore compatibility-bound to a selector implementation, model
architecture, and backend build.

## Validation and acceptance

Implemented model-free tests cover:

- empty, duplicate, unordered, extreme, and topology-mismatched site sets;
- malformed vector widths, layer ranges, normalization, and non-finite data;
- deterministic accumulation and ordering;
- multi-row causal-position selection;
- ordered operator execution and one final write-back;
- operator error and panic containment with no partial mutation;
- provisional, admitted, and rejected speculation accounting;
- zero, partial, and complete draft acceptance;
- end-of-generation and cooperative-stop boundaries;
- target-only and separately bound draft activation;
- callback failure and panic containment;
- quiescent MTP and EAGLE-3 implementation-state bounds and lifecycle; and
- every model, backend, topology, vector, and checkpoint mismatch.

Adapter compilation is pinned to the reviewed successor binding. Live tests
are opt-in, use caller-supplied local artifacts, require a reported
accelerator, record exact target/draft artifacts and placement, and never
fall back to CPU. Separate live profiles exercise layer capture and mutation,
MoE router-logit mutation and observation, MTP, and EAGLE-3.

These tests establish contract behavior and state accounting. They do not
establish the meaning of a captured direction, generation quality, throughput
benefit, or any semantic outcome.

## Implementation status

The backend-neutral V2 contracts, deterministic vector accumulator,
llama.cpp topology/profile validation, activation capture and transactional
operators, provisional telemetry resolution, and one-sequence
target-authoritative MTP/EAGLE-3 operation are implemented. The operation
fails before allocation for unsupported topology, policy, capacity,
implementation identity, pinned vocabulary-compatibility rules, MTP row
width, or EAGLE-3 draft metadata and never falls back to ordinary generation.

The binding successor is forward-ported to the literal revision recorded
above. It owns bounded tensor transactions, native begin/end decode hooks,
exclusive-lifetime speculative wrappers, contained C++ lifecycle failures,
and versioned MTP/EAGLE-3 implementation-state capture and restore.

Two acceptance items remain deliberately open:

1. `generate_speculative` does not yet expose a persistent checkpoint/restore
   façade. The complete envelope is specified and the native target, draft,
   and implementation state is available, but llama.cpp exposes target sampler
   continuation only as an opaque in-process clone. No incomplete portable
   checkpoint is advertised.
2. Activation graph execution, MTP, and EAGLE-3 still require opt-in
   accelerator-backed fixtures using caller-supplied compatible artifacts.
   Model-free tests and adapter compilation are not live-model evidence.

The local adjacent-path binding is suitable for implementation review, not a
public crate release. It must land at an immutable public source for a
source-only revision and as a registry package before the crates.io adapter
can be published.

## Coordinator and availability boundary

The activation and speculative mechanics are unconditional public API and
default-built adapter functionality. A downstream coordinator which includes
this implementation exposes the new exact capability and schema by default.
It does not have a feature, configuration switch, dormant registration path,
or deployment mode which leaves the implementation present but unavailable.
If the coordinator cannot advertise and route the exact capability, it fails
startup.

This availability rule does not authorize model-backed execution. Live
acceptance still requires caller-supplied compatible artifacts, an explicitly
authorized accelerator-backed transaction, exact placement, and a receipt.
Unsupported model topology, graph sites, draft compatibility, or admission
fails the operation; it does not make the compiled capability optional.

## Consequences

- Logit Loom gains exact contracts for capture, runtime activation programs,
  MTP, EAGLE-3, and their checkpoint envelope, plus one-shot native execution
  for the first four mechanics.
- The safe binding must land and be pinned before the adapter releases the new
  capability.
- Exact graph-site profiles are renewed when llama.cpp or a model architecture
  changes.
- Full speculative checkpoints are larger and more conservative than target
  context state alone.
- Timing, transfer, and acceptance-rate measurements are deployment
  observations outside deterministic plan and receipt identities.
- Downstream coordinators expose the new exact capability by default or fail
  startup; they must not broaden an existing generation identity or move
  model-local state across IPC.

<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Architecture

The diffusion runtime decision and its post-Euler state boundary are recorded
in [ADR 0001](adr/0001-stable-diffusion-runtime.md).
Topology-bound text activation and target-authoritative speculation are
recorded in
[ADR 0002](adr/0002-transactional-text-mechanics.md).
Bounded resident staged image programs are recorded in
[ADR 0003](adr/0003-resident-image-programs.md).

Logit Loom separates stable mechanics contracts from callback execution and
from fast-moving model backends. Token candidates and diffusion tensors remain
different typed surfaces.

```text
application
  ├─ exact optional model profiles ─────────── logit-loom-models
  ├─ worker-local lifecycle and buffers ────── logit-loom-executor
  ├─ explicit local text workflow ──────────── logit-loom-runtime
  ├─ token plans, IDs, receipts ────────────── logit-loom-core
  ├─ logit transforms and token observers ──── logit-loom
  ├─ diffusion plans and state operations ──── logit-loom-diffusion
  ├─ llama.cpp model/session adapter ───────── logit-loom-llamacpp
  └─ stable-diffusion.cpp image adapter ────── logit-loom-diffusion-sdcpp

text lane
  logit-loom-runtime
  ├─ logit-loom
  └─ logit-loom-llamacpp
       ├─ ordinary causal session
       ├─ activation transaction controller
       └─ MTP/EAGLE-3 target verifier
            → reviewed llama-cpp-4 successor → pinned llama.cpp

image lane
  logit-loom-diffusion-sdcpp
  ├─ logit-loom-executor
  ├─ logit-loom-diffusion
  └─ companion ABI v1 + image ABI v2 + program ABI v3
       → pinned stable-diffusion.cpp
```

This split lets another text backend consume `logit-loom-core` and
`logit-loom`, or another image backend consume `logit-loom-diffusion`, without
importing either native runtime's types. Native ownership, unsafety, and
compatibility churn remain in adapter crates.

## Runtime façade

`logit-loom-runtime` composes one loaded model, the native runtime, borrowing
sessions, generation plans, transforms, observers, checkpoints, and steering
scopes. It adds no new sampling or native state semantics. The lower-layer
ordering, callback containment, causal accounting, poisoning, and
compatibility checks remain authoritative.

One-shot completion creates a fresh session and performs exact-text replacement
followed by one bounded generation call. Stateful sessions expose replacement
and append as different methods. Tokenization flags, allocation options,
device policy, controls, bytes, and mechanical identities stay visible.

Pipelines and observers are mutable borrowed controls for one synchronous
generation call. Their `GenerationPlan` remains owned and serializable; callback
objects and native handles are deliberately not serialized. First-party façade
helpers assign versioned configuration-bound implementation identities.
Arbitrary callbacks require a caller-defined `Digest`.

The façade does not manage chat messages, templates, asynchronous execution, or
workers. Those are downstream application concerns.

## Worker-local executor seam

`logit-loom-executor` is the transport-neutral boundary for a downstream
resident worker. It defines exact borrowed inputs, caller-owned output
allocations, cooperative cancellation, lifecycle states, cleanup receipts, and
the `Rejected`, `Cancelled`, and `Poisoned` reuse dispositions. It deliberately
does not define sockets, queues, resource admission, artifact stores, retries,
or process supervision.

The seam is synchronous so a backend cannot retain borrowed storage after a
call. Native adapters keep their stronger ownership rules; in particular,
`Sdcpp` remains neither `Send` nor `Sync`. A downstream application that needs
concurrency should give each resident owner to one worker and communicate with
that worker using its own authenticated transport.

## Diffusion step boundary

`logit-loom-diffusion` defines bounded tensor, schedule, plan, checkpoint,
intervention, observer, whole-image execution, and receipt contracts without
owning a tensor runtime. Whole-image plans bind exact input slots and layouts,
output format, placement, seed policy, schedule, ordered LoRAs, installed
operators, and batched observations. They never represent pixels or latent
elements as token IDs or logits.

The default-built `ImageProgramPlanV1` family is a separate multi-native-stage
contract. It numbers typed values canonically, requires one producer per
value, permits only earlier-stage references, validates operation-specific
types and geometry, derives release points and a conservative arena peak from
value liveness, and binds exact output and receipt allocations. Deterministic
receipts retain the completed-stage prefix and cleanup result; wall time,
native time, placement, and transfers remain in a separate measurements
record. No native pointer or arena handle enters a serialized value.
`SdcppResidentProgram` executes this family over a request-scoped native value
arena with generation-checked private handles. Ordinary repository tests and
the retained native build establish contract/lowering and compilation only;
model-backed behavior remains opt-in acceptance.

The stable-diffusion.cpp adapter supports only the exact catalogued MiniT2I and
Krea 2 component layouts. It dynamically loads companion ABI version 1 plus
the required image ABI version 2 and program ABI version 3 at the exact
upstream commit recorded in ADR 0001. Before context creation it verifies
component bytes, the library digest,
required symbols, ABI/revision, the bounded device report, and exact non-CPU
backend names. No failed placement is retried on CPU.

Native conditioning tensors are delivered first and hashed synchronously. The
adapter then constructs an exact `DiffusionPlan` containing component,
conditioning, RNG, seed, tensor, and custom schedule identities. After each
Euler update, the companion exposes one contiguous host `f32` state with exact
shape and sigma boundaries. Model evaluation remains on the caller-selected
accelerator; the host callback boundary is reported separately. The companion
also measures denoiser-plus-Euler elapsed time immediately before each
callback. The adapter returns those non-deterministic deployment measurements
separately from plans, receipts, checkpoints, and content identities.

The adapter copies that complete state before calling Rust. An optional
`PipelineProgram` applies ordered backend-neutral interventions to the copy,
optionally at one selected step. A program error, panic, wrong shape, exhausted
bound, or non-finite result returns a callback failure with no native
write-back. Observers receive the complete finite post-intervention copy.
Only after both phases succeed is the copy committed to native state.

The lower-copy image-v2 path validates tightly packed source, mask, and
reference bytes, applies a fixed request-local LoRA stack, verifies that each
requested LoRA participated in at least one model tensor, and writes one RGB8
image to caller-owned storage. It also exposes direct bounded Krea VAE
encode/decode.

Safe adapter contract v4 introduced `ImagePlanExecutor`, which combines those
advanced inputs with the transactional full-state callback in one native
generation. Its version-two plan restores/captures an authenticated
checkpoint before installed scheduler-state operators, observes and cancels
after those operators, executes a bounded deterministic RGB8 mask-blend graph,
preflights explicit output routes, accounts for retain/clear cleanup, and only
then initializes caller-owned outputs. That version-two path continues to
reject scheduled `LoRA` scales, arbitrary model-block/conditioning operators,
snapshots, multiple native inference operations, and direct VAE stages rather
than changing its established identity.

Safe adapter contract v5 adds the separate `SdcppResidentProgram` and mandatory
program ABI v3 for scheduled adapters, snapshots, multiple native/VAE stages,
typed RGB8/RGBA8/PNG/tensor/checkpoint outputs, and private arena liveness.
Only installed scheduler-state selectors are accepted; unresolved model-block
and conditioning selectors fail whole-program preflight. See
[worker-local image execution](image-execution.md) for the support matrix and
failure rules.

`DiffusionCheckpoint` stores exact little-endian state bytes plus conservative
lineage. Initial restore uses deterministic-prefix replay: rerun the exact
plan and seed, require the recomputed post-step identity to match, restore the
authenticated bytes, then branch. It does not skip earlier native work or
permit a different schedule, conditioning input, artifact set, or backend.

`Sdcpp` has one owner and is neither `Send` nor `Sync`. Raw pointers and dynamic
symbols remain private; the complete local unsafe contract is in the adapter's
[`SAFETY.md`](../crates/diffusion-sdcpp/SAFETY.md).

## Candidate and sampling sequence

For each generation step, the llama.cpp adapter performs these mechanics in
order:

1. Poll generated-token observers at the pre-sampling boundary.
2. Copy the current raw vocabulary logits from the causal context.
3. Select the full vocabulary or a deterministic sparse top-ranked view.
4. Run Rust transform stages in declared order on the copied view.
5. Commit transformed candidates only if every stage succeeds.
6. Apply native grammar, logit bias, repetition/DRY penalties, probability
   filters, temperature, and the terminal sampler.
7. Treat an end-of-generation token as terminal without decoding it.
8. Obtain the selected token's exact bytes, decode the token into causal state,
   then notify transforms and observers of the admission.
9. Check exact byte stop suffixes after admission.

Sparse selection orders finite raw logits from highest to lowest and breaks
ties by ascending token ID. It is an exposure optimization, not a semantic
selection policy. All stages in one pipeline use the same exposure mode.

Repetition and DRY sampler state is initialized with the exact causal token
history. `GenerationPlan` retains the original eager grammar and digest
contract. `GenerationPlanV2` preserves those v1 mechanics under a new identity
and selects either eager activation or bounded ordered native regex/token
triggers for a lazy grammar. Prompt tokens never consume either grammar; every
admitted generated token is accepted by the complete native sampler chain.

Structured projection adds a controller above that same call boundary. Its
compiled plan binds caller-owned compiler and validator identities, exact
constraints, eager/lazy grammar, incremental byte-feedback identity and
bounds, optional transforms/observers, and a maximum number of
caller-explicit attempts. The llama.cpp adapter content-identifies every
tokenizer ID, exact piece byte sequence, and EOG classification. Feedback sees
the exact piece for every eligible candidate and updates only after the
selected non-EOG token is causally decoded.

The controller captures one exact boundary before its first attempt. Observer
cancellation and authoritative validation rejection restore that checkpoint
before returning. A callback or native error also attempts exact restoration;
restoration uncertainty poisons the session. There is no automatic retry:
each subsequent attempt is a separate caller method invocation, and only one
complete conforming terminal attempt remains causal.

## Activation transaction boundary

`TextModelTopologyV1` binds exact model bytes, backend build, architecture
implementation, layer/embedding/expert dimensions, NextN heads, and supported
speculative mechanisms. `TextTensorSiteV1` is backend-neutral; the llama.cpp
adapter lowers it only through an exact selector profile tied to that topology
and pinned graph implementation.

The successor binding owns the native callback lifetime. Exact begin/end hooks
cover direct decodes and llama.cpp decodes performed inside MTP or EAGLE-3.
For each selected graph node it checks name, dtype, contiguous shape, row
width, row count, sequence IDs, causal positions, and aggregate byte bounds.
It then copies the complete tensor into Rust-owned storage.

Read-only selections commit retained captures only after native decode and
complete row coverage succeed. Mutable `f32` selections run ordered
scaled-add or scaled-projection-removal operations on the owned copy. One
complete finite tensor is written back after every operation succeeds; a
callback error or panic produces no partial write-back and poisons the
high-level session.

Capture plans select the last prefill token or explicit inclusive positions
and retain a digest, deterministic scalar statistics, or a bounded snapshot.
The content-free accumulator consumes already captured rows in caller order to
produce means or differences of means. It records mechanical provenance but
does not assign a semantic label or make an efficacy claim.

## Target-authoritative speculation

`generate_speculative` creates separate target and draft contexts and an MTP
or EAGLE-3 native session. It currently accepts exactly one sequence. Plans
requesting another sequence count, insufficient verification-batch capacity,
missing recurrent rollback headroom, a mismatched implementation identity, or
an incompatible model/topology relationship fail before generation. The
pre-allocation pairing check mirrors the pinned llama.cpp revision's
vocabulary rules and validates MTP row width or EAGLE-3 architecture and
extraction-layer metadata before either context is created.

At each boundary:

1. The draft proposes zero or more token IDs.
2. The target decodes the pending sampled token plus the proposal and exposes
   logits for every row.
3. The draft implementation processes the same verified target state.
4. The target sampler accepts only the longest exact proposal prefix.
5. Both native contexts remove the rejected suffix and the draft
   implementation records the accepted count.
6. Provisional target/draft activation records resolve against the final
   contiguous causal prefix.
7. Transforms, exact token bytes, and observers advance only for tokens now
   admitted by the target.

The sampled mismatch becomes pending input for the next boundary and is not
observed until its target decode succeeds. An end-of-generation selection is
never decoded. Observer or exact-byte stop requests may shorten the accepted
prefix; the unobserved suffix is rolled back before the boundary receipt is
committed. There is no fallback to ordinary generation.

At every completed boundary the successor binding can expose exact target,
draft, and versioned MTP/EAGLE-3 implementation state.
`generate_speculative_checkpointed` retains those bytes with the opaque target
sampler, activation configuration, stop-prefix state, causal history, boundary
count, and parent lineage. `resume_speculative_checkpointed` validates the
complete envelope before allocation and clones the sampler for an independent
branch.

`SpeculativeCheckpointReceiptV1` authenticates the envelope but cannot
reconstruct it. Target sampler continuation remains an opaque in-process
native clone, so the snapshot is thread-affine and process-local rather than a
falsely portable checkpoint.

## Transactional transforms

A `Pipeline` copies its candidate view before invoking user code. If a stage
returns an error, exceeds its step bound, produces a `NaN`, or panics, no
candidate changes from that invocation are written back to the backend view.
The pipeline enters a failed state until the caller begins a new call.

Successful invocations use consecutive zero-based step values. This makes the
declared invocation bound enforceable even for a custom adapter and prevents a
caller from reusing an earlier step number to bypass it. Backend-selected
candidate views must contain unique token identifiers. A nonterminal
successful invocation must be matched by one causally admitted token before
the next transform step begins.

`apply_to_vocabulary` performs Logit Loom's deterministic full/sparse exposure
selection. `apply_to_candidates` instead accepts a view already selected and
copied by a backend adapter. In that form, the adapter is responsible for
proving complete full-vocabulary exposure or its native sparse-selection rule;
Logit Loom still enforces array shape, declared sparse bounds, stage limits,
containment, accounting, and transactional write-back to the supplied scratch
slice.

Implementation-local state in stages that ran before a later failure cannot be
rolled back. This is why retries require `begin`, which resets every stage and
starts new accounting.

## Exact-byte observation

Tokenizer pieces are byte slices. Individual pieces are not required to be
valid UTF-8, and observers receive them without conversion. `GenerationOutput`
can return the complete byte buffer as text only when the complete output is
valid UTF-8.

Observers run synchronously and in declared order. Every observer is polled at
the same boundary even when an earlier observer requests a stop. Token
callbacks occur only after native decode succeeds, so each observed token is
already part of causal state.

An observer set requires at least one successful poll before each delivered
token and rejects delivery beyond the call's requested token count. A stop
request is terminal for that observer call. Exact byte stop sequences are
checked after transform admission callbacks and generated-token observers; an
observer stop therefore takes precedence when both conditions occur on the
same admitted token. Stop bytes remain in output and causal state, and the
lowest declared stop index wins when several suffixes match.

## Causal sessions

A llama.cpp `Session` owns one mutable context and exact admitted token history.
It is deliberately neither `Send` nor `Sync`. Applications that need
concurrency should place a session inside a single-owner worker and communicate
with it through their own bounded channel or actor interface.

Prefill submits complete bounded chunks. A cooperative stop or callback error
retains every chunk that native decode already accepted. With `clear_first`,
the prior context is cleared before the first observer poll; this mutation is
part of the requested replacement operation.

## Checkpoints

A `StateSnapshot` combines opaque llama.cpp state bytes with exact token
history and a receipt. Restore verifies:

- the model-file content identity;
- the adapter build compatibility identity;
- the context, batch, micro-batch, and thread allocation contract;
- the state-byte identity;
- the token-history identity and causal position.

Snapshots wrap backend-owned state and are not a portable interchange format.
Capture and restore are rejected while a steering scope is active so the
checkpoint cannot omit required active steering state.

Applications that persist a snapshot choose their own container format.
`StateSnapshot::from_parts` revalidates the opaque byte count and identity,
token-history identity, and causal position before a later restore checks the
model and backend identities.

The pinned llama.cpp state bytes contain causal memory but not output logits.
After restoring all bytes, the adapter removes the final sequence position and
re-decodes its exact recorded token to reconstruct the next-token boundary.
Failure to remove or re-decode that position poisons the session instead of
sampling from stale logits.

## Steering scopes

`LoRA` adapters and control vectors are applied through scopes that exclusively
borrow the session. Only one steering resource may be active. Explicit
`clear()` returns a lifecycle receipt; dropping a scope also attempts cleanup.

The safe upstream control-vector binding cannot pass a null slice to the native
clear operation. Logit Loom therefore validates a complete model-sized vector
and neutralizes it with an explicit all-zero vector on cleanup. This restores a
zero steering contribution while staying inside the safe binding.

If automatic `LoRA` removal, vector neutralization, or complete checkpoint
restore fails, the session is poisoned. Later mutation returns
`Error::Poisoned` rather than silently running with uncertain native state.
Callers can inspect `Session::is_healthy` and `Session::poison_reason`.

## Identities and receipts

Digest domains are explicit and versioned. A plan digest binds exact serialized
mechanics; a receipt binds exact accounting and lineage. Changing a serialized
shape or its interpretation requires a new versioned digest domain.

Receipts are not cryptographic signatures and make no statement about semantic
quality, truth, or efficacy. Applications may persist or sign them when they
need provenance beyond content identity.

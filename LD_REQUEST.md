<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Downstream requirements for production image inference

## Status

This document is a capability request from a production local-inference
consumer. It is not an implemented-capability claim, an acceptance report, or
a request to weaken the staged gates in [`NEXT_STEPS.md`](NEXT_STEPS.md).

Revised 2026-07-25 with operator sign-off: the production coordinator,
worker host, queueing, admission, and device-resource authority previously
requested here are now owned by a private downstream coordinator and are
withdrawn from this request. Logit Loom is asked for the public, in-process
executor surface a downstream worker links directly.

The consumer will keep its prompts, visual references, application ontology,
model selections, generated artifacts, and semantic evaluation private.
Logit Loom is asked to own the generally useful, mechanically testable
inference surfaces described below.

### Production-consumer return review — 2026-07-27

A production consumer accepts immutable Logit Loom commit
`984ef6832c0d8032e0325f31a1d1cf5634ecd58c` (tree
`b3c15493b48cdff65b5503c99dbf4ee1c41a9716`) as the reproducible public
substrate for its fixed-stack Krea Turbo baseline. The consumer vendors that
exact tree, verifies a complete file/mode/extra-path manifest, links the
stable-diffusion.cpp adapter inside its admitted worker, and does not start or
wrap an external daemon.

That immutable return does not yet close these requested public mechanics:

1. One public, single-owner executor must consume a complete
   `ImageExecutionPlan` or a versioned successor, execute its ordered graph in
   one session, and return one whole-plan receipt. It must cover transactional
   checkpoint restore, intervention, compositing, output routing,
   cancellation, and cleanup disposition instead of requiring the consumer to
   privately sequence public primitives.
2. The complete advertised option set needs hostile-input, cancellation,
   rollback, stale-handle, partial-cleanup, and receipt-lineage tests. Until
   then the consumer advertises only its narrower fixed-stack baseline; the
   broader image-program feature remains hidden.
3. The tokenizer crate still needs the pinned Gigatoken-derived SIMD BPE
   kernel, vector-free exact/threshold count paths, reusable output sinks, and
   a caller-sized dedicated pool. The consumer will not activate an alternate
   method from the current utility primitives alone.
4. Krea 2 Raw remains a separately identified future profile and must not be
   inferred from Turbo mechanics or receipts.

Please land these as immutable public revisions with no worker process,
listener, daemonization, CPU model execution, retry, or fallback. The consumer
will replace its vendored snapshot only after reviewing that exact return.

## Required outcome

Publish a Rust-first, policy-neutral image-inference surface that can:

1. load exact Krea 2 Raw and Turbo component sets in-process;
2. run deterministic and explicitly scheduled diffusion operations;
3. apply arbitrary caller-selected compatible LoRA adapters without attaching
   semantic policy to them;
4. expose checkpoint, intervention, lifecycle, placement, and resource
   mechanics with conservative receipts; and
5. link as an in-process library inside a consumer-owned worker process:
   load from verified read-only artifact descriptors, execute a validated
   plan to content-addressed receipt identities, cancel cooperatively at a
   declared boundary, and unload verifiably — with backend-native tensor
   handles and operators (no tensor serialization inside one worker), no
   helper processes, and tolerance of seccomp confinement that denies INET
   socket creation.

The surface must not bundle weights, execute remote repository code, silently
download artifacts, hide model or device placement, retry on CPU, or make a
claim about image quality or model behavior.

## Bulk-tokenizer substrate for downstream consumers

The downstream consumer also requires a publishable in-process tokenizer
substrate. This is non-model utility computation: it must never create a CPU
model allocation or provide a model fallback. The initial implementation
should import only the needed MIT-licensed mechanics from Gigatoken `0.9.0` at
`0d9765fa7312af7534535e6315a5c49d74807b2a` and retain an exact import
manifest and notice.

The return must provide:

- exact tokenizer, normalizer, pretokenizer, vocabulary, merge, Unicode,
  added-token, special-token, implementation, and policy identities;
- a pinned SIMD BPE implementation with token IDs and optional source offsets
  exactly matching a supplied engine oracle—not counts alone;
- count-only and inclusive early-stop counting that do not materialize the
  complete token-ID vector;
- token-aware chunk planning from the same verified pass where exactness
  permits;
- bounded normalized-pretoken duplicate elimination and a collision-checked
  frequency-aware span cache with independent entry, byte, and span ceilings;
- allocation-reusing scratch/output-sink APIs suitable for read-only mapped
  sealed inputs and directly streamed outputs;
- a dedicated bounded execution-pool abstraction whose task count is supplied
  by the admitting caller, with no ambient Rayon pool, process-global thread
  mutation, helper process, Python, socket, listener, retry, or fallback; and
- cancellation only at documented safe boundaries plus content-free
  accounting for work, cache behavior, and peak scratch.

Differential tests must cover cold/warm cache, every pool and batch candidate,
multilingual and pathological input, offsets and reconstruction, threshold
counting, cancellation, eviction, and partitioned versus whole execution.
The downstream application owns profiles, resource leases,
batching/coalescing policy, adaptive CPU/GPU selection, receipts, and worker
lifecycle. Logit Loom supplies mechanics only. Delivery requires an immutable
revision; an uncommitted sibling checkout is not an activatable dependency.

## Model profiles and acquisition

### Krea profiles

- Retain the pinned Krea 2 Turbo profile and add a separately pinned Krea 2
  Raw profile when its source and license review pass.
- Bind the transformer, conditioner/tokenizer, scheduler, autoencoder, and
  every auxiliary artifact independently as well as through one complete
  profile identity.
- Keep exact revisions immutable. A new revision or component set starts a new
  profile and acceptance lineage.
- Allow a caller to supply a locally defined compatible profile manifest for a
  fine-tune or private checkpoint. This is an extension point, not a promise
  that arbitrary sibling checkpoints work.
- Validate the loaded architecture from artifact metadata. Do not encode
  undocumented assumptions about transformer layer count, component names, or
  conditioning layout.

### Artifact sources

- Keep Hugging Face acquisition exact-file and exact-revision based.
- Add an artifact-source interface capable of supporting version-pinned
  Civitai downloads and offline verification.
- Authentication must come from a credential helper or environment variable.
  Tokens must never appear in process arguments, logs, receipts, dry-run
  output, or persisted manifests.
- Public profiles may reference gated artifacts and their licenses but must
  not redistribute them or infer acceptance on the operator's behalf.

## Diffusion contracts

The first implementation may remain adapter-local as planned. The following
contracts should become public only after the MiniT2I and Krea spikes establish
their actual stable boundaries.

### Plans and identities

A durable diffusion plan must bind at least:

- exact model-component identities;
- exact adapter-stack identity and declared order;
- input conditioning identity;
- seed and random-number-generator algorithm/state identity;
- width, height, batch size, dtype, and device placement;
- scheduler algorithm, parameters, timesteps, and guidance parameters;
- operation kind and its bounded inputs; and
- observation/intervention boundaries.

Supported operation mechanics should include text-to-image, image-to-image,
masked inpainting, outpainting through an expanded masked canvas, explicit VAE
encode/decode, and continuation from a validated latent checkpoint.

Plans and receipts describe mechanics and lineage only. They do not assert
that an image is useful, faithful, safe, unrestricted, or aesthetically good.

### Step boundary and interventions

- Expose one documented pre-step or post-step tensor boundary with shape,
  dtype, device, scheduler position, conditioning identity, and allocation
  identity.
- Execute ordered interventions transactionally. A validation error, callback
  error, or unwind must not partially commit the candidate state.
- Reject non-finite values, shape or device mismatches, out-of-range steps,
  stale allocation identities, and unsupported component selectors before
  state mutation.
- If model-specific block or conditioning intervention is supported, selectors
  must be resolved against the validated loaded architecture rather than a
  hard-coded layer vector.

### Checkpoint and replay

A checkpoint must conservatively bind:

- all component and adapter bytes;
- backend build and allocation compatibility;
- latent bytes and identity;
- scheduler state and exact step;
- RNG state;
- conditioning state;
- image/mask inputs where applicable; and
- every opaque backend byte required for replay.

Two unchanged branches restored from one checkpoint must remain identical.
Restoration with a mismatched artifact, backend, allocation, schedule,
conditioning input, or adapter stack must fail before mutation.

### LoRA mechanics

- Load adapters independently from applying them.
- Support an explicitly ordered stack with finite per-adapter scales.
- Represent transformer and text-conditioning targets without assuming that
  every adapter affects both.
- Where the backend supports it, permit a bounded deterministic scale schedule
  over denoise steps.
- Apply and clear the complete stack transactionally. Cleanup failure poisons
  the image session or worker instead of allowing uncertain continuation.
- Receipts bind adapter content, declared targets, order, scales, schedule, and
  observed lifecycle outcome.

The API is deliberately content-neutral. Compatibility and mechanics are
validated; semantic effects are downstream research.

## Runtime: in-process executor surface

The in-process adapter is the authoritative model implementation and the
only runtime surface requested. The consumer owns the worker process, its
protocol, its confinement, and all queueing/admission; Logit Loom provides
the executor the worker links:

- inspect exact build and capability identity (schema/implementation digest
  pairs suitable for a registration manifest);
- load and validate an exact model profile from verified read-only artifact
  descriptors (sealed `memfd`), never from arbitrary host paths;
- execute one validated, bounded, serializable plan to content-addressed
  receipt identities;
- cancel cooperatively at one documented boundary;
- unload with verifiable release, poisoning on cleanup uncertainty; and
- calibrate `ResourceProfile`-shaped measurements (resident memory, load
  peak, execution peak, host peak) for the exact model/backend/device/dtype
  combination that produced them, with uncalibrated output distinguishable.

No worker binary, Unix-socket protocol, HTTP server, network listener,
service-manager integration, implicit daemonization, retry loop, or
fallback route belongs in this surface. Measurement-provider libraries
(AMDGPU sysfs, DRM fdinfo accounting, `MemAvailable`, cgroup-v2 `dmem`)
remain welcome as reusable read-only components.

## Withdrawn: accelerator coordination

The previously requested `WorkerRegistration`/lease shapes, reference
coordinator, admission authority, and worker host are withdrawn. That
production boundary is owned by the private downstream coordinator, which
already implements registration-by-digest, residency/transition/execution
admission, strict-priority queueing, sealed-descriptor transport, epochs, and
verified release. Duplicate public mechanisms would create the split ownership
this revision removes.

## Acceptance gates

### Model-free

- Bounded serialization and hostile-frame rejection.
- Stable identities for plans, profiles, checkpoints, adapter stacks, and
  receipts.
- Ordered transactional intervention and LoRA-stack behavior.
- Callback errors and unwinding without partial write-back.
- Checkpoint mismatch rejection.
- Executor state-machine and poisoning tests against a fake backend.

### Opt-in accelerator

- Exact Krea profile loads on the explicitly selected accelerator with no
  unreported fallback.
- Baseline generation completes with all component identities recorded.
- Two unchanged checkpoint branches are byte-identical.
- One bounded intervention changes state identity and retains consistent
  lineage.
- An ordered compatible LoRA stack can be applied and cleared with observable
  cleanup.
- Cancellation occurs at the documented boundary and a captured continuation
  resumes conservatively.
- Load, operation, and unload reports include peak host/device memory,
  placement, dtype, and latency.
- Unload completion is not reported until resource release is observed.

## Publication boundary

Public source, examples, fixtures, and reports must contain only synthetic
mechanical inputs. They must not contain downstream prompts, identities,
ontologies, private paths, model outputs, credentials, or semantic evaluation.

If an otherwise required adapter cannot be published under compatible terms,
keep the backend-neutral contract public and document the unavailable
capability. The consuming application will bear only the license-constrained
shim, not a duplicate of the public mechanics.

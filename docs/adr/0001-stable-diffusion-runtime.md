<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# ADR 0001: stable-diffusion.cpp for pinned diffusion profiles

- Status: accepted for implementation
- Decision date: 2026-07-25
- Runtime revision:
  `leejet/stable-diffusion.cpp@ea4e566ccffa10f853ecc3f29e74b1820bc91beb`
- Initial sampler boundary: Euler only

## Context

Logit Loom needs one maintained image runtime for the pinned MiniT2I-B/16 and
Krea 2 Turbo profiles. The runtime must expose exact iterative state without
turning pixels or latents into token-shaped contracts. It must also preserve
the repository's existing rules for explicit placement, bounded inputs,
transactional callbacks, conservative checkpoints, and opt-in model
execution.

The reviewed stable-diffusion.cpp revision has maintained loaders for both
profiles:

- MiniT2I uses its diffusion transformer plus FLAN-T5-Large, produces a
  three-channel direct-RGB iterative state, and documents Euler sampling.
- Krea 2 uses its diffusion transformer, Qwen3-VL text encoder, and Wan 2.1
  VAE, and can use the same Euler scheduler boundary.

Its model evaluation supports several accelerator backends and its public C
API already provides explicit component paths, backend selection, seeds,
samplers, schedules, and image ownership. Its internal Euler state is an
owning, contiguous host `sd::Tensor<float>` whose first dimension has unit
stride. That makes a bounded post-step callback mechanically precise.

The reviewed public API does not expose that state. Progress and preview
callbacks are insufficient: progress has no state, while a decoded preview is
not the exact scheduler state and is not mutable.

## Decision evidence

The spike evaluated the runtime against the concrete requirements of both
pinned profiles:

| Criterion | Evidence at the pinned revision |
| --- | --- |
| Profile/operator coverage | Maintained MiniT2I transformer/T5 and Krea transformer/Qwen3-VL/Wan VAE loaders and execution graphs |
| Artifact formats | Direct safetensors and GGUF loading; no Python or remote repository code |
| Accelerator backends | Explicit Vulkan, HIP, CUDA, and Metal builds plus named model/parameter backend assignment |
| Callback tensor | Owning contiguous dimension-zero-fastest host `f32` Euler state with exact native shape |
| Model dtypes | Native runtime owns quantized/internal dtypes; the public intervention conversion is one explicit `f32` contract |
| Random state | Explicit native `CPU_RNG` selection and seed; exact replay still requires model-backed deterministic-prefix verification |
| Memory behavior | Optional memory mapping and explicit backend placement; allocation failure is reported, while peak host/device use remains a live deployment measurement |
| Failure containment | Synchronous callback status permits Rust error/panic containment and transactional copied state |
| Maintenance cost | One exact upstream revision and a narrow companion patch instead of two duplicated transformer/tokenizer/VAE stacks |

This evidence establishes mechanical feasibility. It does not establish
numerical equivalence to another framework, throughput, memory fit, output
quality, or deterministic replay on a particular device; those remain opt-in
acceptance facts.

## Decision

Add a small, versioned companion patch for the exact runtime revision and a
safe Rust adapter over its dynamically loaded shared library.

The companion ABI:

1. reports its own ABI version and exact upstream commit;
2. creates a context from explicit component paths and an explicit accelerator
   backend;
3. permits only the Euler sampler while a state callback is installed;
4. reports the exact retained token-ID and conditioning tensors before
   sampling, then invokes one state callback immediately after each Euler
   update;
5. exposes zero-based completed-step index, total steps, sigma before and
   after the update, exact shape, a mutable contiguous `f32` slice, and
   steady-clock denoiser-plus-Euler elapsed time measured immediately before
   the callback;
6. treats any nonzero callback result or non-finite returned element as a
   generation failure; and
7. returns image memory through the runtime's own deallocator.

The patch remains source text in this repository. A preparation script applies
it only to the pinned upstream checkout in caller-selected source and build
directories. Tests, package builds, documentation builds, and CI never clone
or compile the native runtime. No native source or model weight is vendored.

The Rust adapter loads the library named by the caller. It rejects:

- another companion ABI version or upstream commit;
- a missing or CPU-named backend;
- no reported accelerator device;
- a profile/artifact mismatch;
- an unsupported sampler, schedule, shape, or step count;
- a callback shape, dtype, layout, device, plan, or step mismatch;
- a non-finite or negative native step-time measurement;
- callback errors or unwinding;
- non-finite state; and
- a native generation failure or malformed output image.

The state exposed by this revision is a contiguous host `f32` tensor even when
model evaluation uses an accelerator. Receipts report that boundary as host
state and separately record the native accelerator backend/device report. No
claim is made that intervention arithmetic itself executes on the accelerator.
Per-step elapsed times describe one deployment execution and are returned
outside deterministic plans, receipts, checkpoint lineage, and identities.

## Transaction and checkpoint semantics

The adapter copies the callback state into bounded Rust-owned storage. Ordered
interventions run on that copy. The native pointer is updated only after every
stage succeeds and the result passes all shape and finite-value checks.
Errors and panics are caught in Rust and converted to a nonzero C callback
result; unwinding never crosses the ABI.

A checkpoint stores the exact post-step `f32` bytes plus:

- all catalogued model artifact identities;
- companion ABI and upstream runtime identities;
- backend and reported devices;
- prompt/conditioning identity;
- seed and random implementation;
- image shape and tensor contract;
- exact sigma schedule; and
- completed-step index.

Initial restore is deterministic-prefix replay: rerun the same exact plan and
seed, require the recomputed state at the selected post-step boundary to match
the checkpoint identity, replace it with the checkpoint bytes, then continue.
This is an exact branch boundary, but it does not skip the native computation
before that step. Receipts state that replay mode explicitly.

## Safety contract

Local unsafe code is limited to the adapter's private dynamic-ABI module.
Its complete contract is maintained in the adapter crate's `SAFETY.md`.
Focused model-free Rust tests cover malformed callback ranks/lengths,
invalid native timing, callback error and panic containment, no write-back on
failed intervention, complete successful write-back, image descriptor
validation/copying, checkpoint mismatch, and compile-fail
non-`Send`/non-`Sync` behavior.

The separate model-free `probe_companion` path loads a caller-built library and
checks required symbols, ABI, upstream commit, library-byte stability, and the
bounded device report. The native preparation check compiles the exact patch
against the pinned upstream revision. Neither operation is model acceptance,
and neither runs in ordinary tests or CI.

The safe layer never exposes a native pointer. A generation session owns its
callback state and cannot outlive the native call.

## Alternatives considered

### Implement both models directly in a new Rust tensor stack

This would maximize control but duplicate two transformer families,
tokenizers, GGUF/safetensors loading, VAE behavior, scheduler details, and
accelerator kernels before testing the experiment boundary. It has a much
larger maintenance and numerical cross-check surface.

### Use `diffusion-rs` / `diffusion-rs-sys` unchanged

The reviewed crate version wraps stable-diffusion.cpp and is permissively
licensed, but its public API has the same progress/preview limitation. Its
bundled static native source also makes the required exact patch and build
identity less explicit, and the higher-level crate adds acquisition choices
that conflict with Logit Loom's no-hidden-download boundary.

### Use the stable-diffusion.cpp process CLI

A process boundary is useful for ordinary image generation but cannot safely
offer a synchronous mutable scheduler-state slice. Serializing the complete
state after every step would be slower and would introduce another protocol
without avoiding the native patch.

### Generalize beyond Euler immediately

Sampler methods differ in their state history, stochastic inputs, and number
of model evaluations per displayed step. One callback label would be
misleading across all of them. The first public boundary is therefore the
single-state Euler update used by both pinned profiles. Another sampler
requires its own reviewed boundary and compatibility domain.

## Consequences

- The backend-neutral diffusion crate can remain free of native types and
  unsafe Rust.
- Both image profiles share one exact step contract and adapter family.
- Native preparation is explicit and reproducible, but remains an opt-in
  deployment step.
- Model-backed acceptance is blocked when no accelerator device is visible;
  there is no CPU fallback.
- Updating stable-diffusion.cpp, the companion ABI, tensor layout, or callback
  timing is a compatibility change requiring a new identity and renewed
  acceptance.

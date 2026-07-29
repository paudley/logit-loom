<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# logit-loom-diffusion-sdcpp

This crate is the safe adapter between
[`logit-loom-diffusion`](https://docs.rs/logit-loom-diffusion/0.2.0/logit_loom_diffusion/)
and the versioned Logit Loom companion ABI for stable-diffusion.cpp.

It supports only the exact catalogued MiniT2I-B/16 and Krea 2 Turbo component
sets. The caller supplies:

- a companion shared-library path;
- every model-component path;
- explicit accelerator and parameter backends;
- thread and flash-attention settings;
- exact prompt, seed, shape, guidance, and sigma schedule; and
- either an optional synchronous step program or an image-ABI-v2 whole-image
  request.

The adapter verifies artifact bytes, companion ABI and upstream commit,
reported accelerator devices, conditioning tensors, and each post-Euler state
boundary. It copies state transactionally before a step program runs. Callback
errors, panics, wrong shapes, and non-finite values return to native code as an
error without partial write-back or unwinding across the ABI.

Each successful generation also returns the native denoiser-plus-Euler-update
latency for every completed step. These non-deterministic deployment
measurements are separate from mechanical receipts and do not affect replay or
content identities.

The safe Rust adapter contract is version 5. Its native-facing paths are
intentionally explicit:

- The full-state path copies each post-Euler host `f32` tensor for
  transactional interventions, observations, and checkpoints.
- The control-only image-v2 path validates boundary metadata without copying
  or hashing scheduler-state elements. It supports text-to-image,
  image-to-image, inpaint, outpaint, ordered reference images, negative
  conditioning, and fixed request-local `LoRA` stacks, then writes one RGB8
  image directly to caller-owned storage or a synchronous sink.
- `generate_advanced_program_to` combines the advanced image inputs and fixed
  request-local `LoRA` stack with the full-state step program in the same
  native generation call.

`ImagePlanExecutor` implements the single-owner
`LocalExecutor<ImageExecutionPlanV3>` boundary. It validates resident
profile/load/RNG/placement identities and every borrowed buffer before native
entry; restores or captures an authenticated checkpoint at the declared
post-Euler boundary before installed scheduler-state operators; records
digest/statistics observations after those operators; and observes
cancellation after that same boundary. It then runs the ordered deterministic
RGB8 mask-blend graph, preflights every route, performs the requested
retain/clear cleanup, and only then initializes caller-owned outputs. A cleanup
failure therefore cannot leave a partially initialized routed output.

`ResidentImageProgramDriver` is the default-built execution state machine for
`ImageProgramPlanV1`. Its backend seam addresses only validated logical value
numbers, so native arena handles remain private. The driver owns exact stage
order, liveness-derived release calls, pre-start/between-stage/post-Euler
cancellation terminals, clean rejected-stage receipts, direct materialization
into caller-owned output allocations, fixed-point receipt serialization,
atomic initialized-prefix publication, cleanup disposition,
placement/transfer measurements, and poisoning on uncertainty. Model-free
fake-arena tests exercise those mechanics without loading a companion library
or model.

`SdcppResidentProgram` implements that backend over the mandatory
stable-diffusion.cpp program ABI v3. One arena can execute multiple ordered
diffusion and direct-VAE stages, scheduled request-local `LoRA` scales,
authenticated checkpoint restore/capture, scheduler-state interventions and
observations, exact snapshots, deterministic RGB8 joins, and typed
RGB8/RGBA8/PNG/tensor/checkpoint outputs. Native stages keep source images and
masks canvas-bound while forwarding bounded reference images with their own
exact RGB8/RGBA8 dimensions and bytes. PNG values bind decoded geometry, RGB
or RGBA color, a deterministic encoder identity, and a bounded encoded length.
Every intermediate remains behind a generation-checked native handle; cleanup
uncertainty poisons the owner.

Image-v2 success confirms that every requested `LoRA` participated in at least
one native model tensor. The native stack is cleared before reusable returns;
uncertain cleanup poisons the single-owner session. Direct bounded Krea VAE
encode/decode exchanges finite native-layout tensors without making that
layout portable to another profile or backend build.

The older `ImagePlanExecutor` whole-plan lowerer supports one native diffusion
operation followed by bounded deterministic `MaskBlend` stages and explicit
RGB8/checkpoint routes.
Per-step `LoRA` schedules, arbitrary model-block or conditioning operators,
snapshot observations, PNG/RGBA/tensor routes, multiple native inference
operations, and version-two direct VAE execution are rejected before native
entry instead of being approximated. The complete worker-local support matrix
is in
[`docs/image-execution.md`](https://github.com/paudley/logit-loom/blob/main/docs/image-execution.md).

All whole-plan tests in the ordinary repository gate are model-free. Live
execution of either the version-two graph or resident program remains an
opt-in acceptance lane over caller-supplied artifacts and an explicit non-CPU
device.

No model is downloaded or executed by tests, CI, package builds, or
documentation builds. Build the caller-selected native library with
[`scripts/prepare-sdcpp.sh`](https://github.com/paudley/logit-loom/blob/main/scripts/prepare-sdcpp.sh),
then follow the opt-in
[`MiniT2I`](https://github.com/paudley/logit-loom/blob/main/docs/runbooks/07-minit2i-fork.md)
or
[Krea 2](https://github.com/paudley/logit-loom/blob/main/docs/runbooks/08-krea2-latent-transplant.md)
runbook. The
`probe_companion` example verifies a library's symbols, ABI, revision, bytes,
and device report without loading a model.

The local unsafe boundary is documented in [`SAFETY.md`](SAFETY.md).

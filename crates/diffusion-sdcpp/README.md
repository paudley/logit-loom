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

Two execution paths are intentionally distinct:

- The full-state path copies each post-Euler host `f32` tensor for
  transactional interventions, observations, and checkpoints.
- The control-only image-v2 path validates boundary metadata without copying
  or hashing scheduler-state elements. It supports text-to-image,
  image-to-image, inpaint, outpaint, ordered reference images, negative
  conditioning, and fixed request-local `LoRA` stacks, then writes one RGB8
  image directly to caller-owned storage or a synchronous sink.

Image-v2 success confirms that every requested `LoRA` participated in at least
one native model tensor. The native stack is cleared before reusable returns;
uncertain cleanup poisons the single-owner session. Direct bounded Krea VAE
encode/decode exchanges finite native-layout tensors without making that
layout portable to another profile or backend build.

Per-step `LoRA` schedules, model-specific target selectors, PNG encoding,
arbitrary model-block operators, and multi-operation graphs are not
implemented by image ABI v2. A downstream lowerer must reject those mechanics
instead of approximating them. The complete worker-local support matrix is in
[`docs/image-execution.md`](https://github.com/paudley/logit-loom/blob/main/docs/image-execution.md).

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

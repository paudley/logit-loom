<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# stable-diffusion.cpp companion ABI

This directory contains a narrow source patch for
`leejet/stable-diffusion.cpp@2988060a4338a5dcd94721c964a8e87003c1ac8b`
(2026-10-11) and its `ggml` at
`leejet/ggml@d25b121ce1d6ae0c7e62ad080460d9fc1e46dd93`, both vendored as
plain files under [`upstream/`](upstream/README.md). Every patch was
ported from the `ea4e566ccffa10f853ecc3f29e74b1820bc91beb` stack (ggml
`eced84c86f8b012c752c016f7fe789adea168e1e`); none became redundant, and one
hunk was dropped (see "Port to 2988060a" below).
That upstream project is MIT licensed. The patch is maintained under Logit
Loom's `MIT OR Apache-2.0` terms and vendors upstream source only unmodified, under `upstream/`.

The companion ABI exposes:

- exact ABI and upstream revision queries;
- an optional process-wide Vulkan allocation ceiling enforced atomically
  before the embedded ggml allocator calls Vulkan;
- explicit context construction for MiniT2I-B/16, Krea 2 Turbo, Z-Image Turbo,
  Qwen-Image, FLUX.2-dev, and FLUX.2-klein-4B;
- deterministic CPU random-state selection with explicit accelerator model
  and parameter backends;
- exact conditioning tensors, including retained tokenizer ID tensors;
- one mutable boundary after each Euler state update, with native
  denoiser-plus-update elapsed time measured immediately before the callback;
- an image ABI v3 extension for bounded source/mask/reference views, negative
  conditioning, fixed request-local `LoRA` entries, direct VAE tensors, and explicit
  session cleanup, plus exact native continuation from a finite post-Euler
  checkpoint at its recorded next transition;
- a mandatory program ABI v3 value arena for multiple diffusion/VAE stages,
  scheduled request-local `LoRA` scales, checkpoint and snapshot state,
  deterministic joins, exact RGB8/RGBA8 conversion, bounded deterministic
  PNG encoding, generation-checked handles, and verified cleanup;
- a model-block ABI v4 extension for typed, exact-step Krea residual scaling
  with loaded-topology validation;
- a model-block application ABI v5 extension that returns actual-transition
  bitmaps, graph-branch counts, loaded topology, and confirmed request-local
  control cleanup;
- a Krea activation ABI v6 extension that publishes the loaded activation
  topology, retains generation-checked donor/vector inputs, installs exact
  request-local captures and operations, reports callback/resource/placement
  evidence, and clears inputs and hooks idempotently; and
- explicit continue, cooperative-stop, callback-error, unsupported-mechanic,
  invalid-argument, and native-error results; and
- complete native exception and failed-invariant diagnostics returned through
  the safe adapter without digest substitution or message truncation.

Krea 2 reference images implement the community edit-LoRA protocol
(`krea2_ostris_edit`): every reference is encoded both into the Qwen3-VL
multimodal context and as timestep-zero latent tokens appended to the DiT
sequence. The base Krea 2 model was never trained on appended reference
tokens, and a text-only Qwen3-VL artifact silently drops the multimodal half,
which deterministically corrupts every reference-bound generation into a
per-latent-cell mosaic. The companion therefore rejects Krea 2 reference
requests with a native error when the loaded text encoder has no vision
weights, instead of returning corrupted images. Reference conditioning
requires the vision-capable Qwen3-VL artifact together with a Krea 2 edit
LoRA.

Image ABI v3 checks that every requested `LoRA` participates in at least one
model tensor before reporting success. It clears the request-local adapter
stack on reusable return paths. A native exception still returns an error so
the safe Rust owner can poison and replace the session.

The dedicated context also suppresses the upstream fallback progress renderer,
which otherwise writes to standard output and corrupts the embedding
application's structured report stream. Native diagnostics remain available
on standard error. The patch does not download weights or accept model
licenses.

Elapsed step times are non-deterministic deployment measurements. They exclude
the Rust callback itself and are deliberately kept out of plans, receipts, and
content identities.

## Build

Use source and build directories outside this checkout:

```sh
scripts/prepare-sdcpp.sh \
  --source /path/to/stable-diffusion.cpp \
  --build /path/to/sdcpp-build \
  --backend vulkan
```

The script verifies the vendored trees' identities, copies them to the
source path (replacing only a source directory it created earlier), applies
[`logit-loom-step-v1.patch`](logit-loom-step-v1.patch) followed by
[`logit-loom-image-v2.patch`](logit-loom-image-v2.patch) and
[`logit-loom-program-v3.patch`](logit-loom-program-v3.patch), then the
[`logit-loom-model-block-v4.patch`](logit-loom-model-block-v4.patch) extension,
then
[`logit-loom-model-block-application-v5.patch`](logit-loom-model-block-application-v5.patch),
then
[`logit-loom-krea-activation-v6.patch`](logit-loom-krea-activation-v6.patch),
then
[`logit-loom-resume-v7.patch`](logit-loom-resume-v7.patch),
then
[`logit-loom-native-errors-v9.patch`](logit-loom-native-errors-v9.patch),
then
[`logit-loom-image-strength-window-v15.patch`](logit-loom-image-strength-window-v15.patch),
then
[`logit-loom-krea-reference-guard-v16.patch`](logit-loom-krea-reference-guard-v16.patch),
then
[`logit-loom-vae-tiling-v20.patch`](logit-loom-vae-tiling-v20.patch),
then
[`logit-loom-eager-load-v21.patch`](logit-loom-eager-load-v21.patch),
then
[`logit-loom-upstream-parity-v23.patch`](logit-loom-upstream-parity-v23.patch),
[`logit-loom-sealed-sources-v23.patch`](logit-loom-sealed-sources-v23.patch),
and
[`logit-loom-igpu-capacity-v23.patch`](logit-loom-igpu-capacity-v23.patch),
and finally the exact ggml
[`logit-loom-vulkan-budget-v8.patch`](logit-loom-vulkan-budget-v8.patch),
[`logit-loom-vulkan-errors-v10.patch`](logit-loom-vulkan-errors-v10.patch), and
[`logit-loom-vulkan-strix-halo-v11.patch`](logit-loom-vulkan-strix-halo-v11.patch),
followed by
[`logit-loom-vulkan-strix-halo-v12.patch`](logit-loom-vulkan-strix-halo-v12.patch),
then
[`logit-loom-vulkan-strix-halo-v13.patch`](logit-loom-vulkan-strix-halo-v13.patch),
then
[`logit-loom-vulkan-strix-halo-v14.patch`](logit-loom-vulkan-strix-halo-v14.patch),
then
[`logit-loom-vulkan-dispatch-bounds-v18.patch`](logit-loom-vulkan-dispatch-bounds-v18.patch),
then
[`logit-loom-vulkan-uma-prefer-host-v19.patch`](logit-loom-vulkan-uma-prefer-host-v19.patch),
the ggml patches to the copied `ggml/`, and builds a shared library. A tree
whose identity differs, or a patch that does not apply, fails the build. The script never runs from
tests, CI, documentation, package builds, or `make check`.

Select `vulkan`, `hip`, `cuda`, or `metal` according to the deployment. The
Rust adapter subsequently verifies the companion ABI, upstream commit,
library bytes, reported devices, and explicit backend before creating a model
context. It does not retry a failed accelerator configuration on CPU.

Probe the compiled ABI without loading a model:

```sh
cargo run --quiet -p logit-loom-diffusion-sdcpp \
  --example probe_companion -- \
  /path/to/sdcpp-build/bin/libstable-diffusion.so
```

The probe may honestly report only a CPU device when the selected accelerator
is unavailable to that process. That proves the library contract loaded; it
does not satisfy either image profile's accelerator acceptance gate.

When `LOGIT_LOOM_VULKAN_MAX_BYTES` is absent, the companion preserves upstream
allocation behavior. When it is present, it must be one positive decimal byte
count; invalid or zero values fail allocations closed. The counter covers each
embedded ggml Vulkan memory allocation and is decremented exactly once on
release. This is a hard process ceiling, not a quality or performance claim.
Every failed Vulkan memory attempt reports its complete requested size, memory
type, required flags, Vulkan error code/category/message, and C++ exception to
the installed native log callback; allocation failures are never collapsed to
an anonymous null buffer. On AMD Strix Halo device `1002:1586`, all Vulkan
cooperative-matrix implementations are disabled because workloads using them
can lose that device. Scalar Vulkan implementations remain available, and
graph submission is synchronous on that exact device so an in-flight async
submission cannot leave the native session or device state uncertain. Each
logical node or fusion is a separate synchronized submission there, bounding
individual compute-ring jobs and reporting the exact node, operation, type,
and shape if submission fails.

Every Vulkan matrix-multiply dispatch is bounded by the device's
`maxComputeWorkGroupCount` on all three axes. Upstream chunked only the batch
(z) axis and asserted on the row (x) and column (y) axes, so any matmul whose
tiled row or column count exceeded the limit (65535 per axis on RADV) aborted
the process; Krea2's text-fusion projector reaches that on prompts longer than
1024 tokens (`n = 2560 * tokens`, 32-wide tiles). The dispatch now iterates
the grid in device-sized chunks, the matmul shaders add the chunk's
work-group offset from push constants, and the `split_k` reduction is chunked
the same way. Chunk counts are computed in 64 bits: RADV reports the x-axis
limit as `0xFFFFFFFF`, where a 32-bit ceiling division wraps to zero and would
request no descriptor set for a dispatch that still happens (the v17 revision
of this patch did exactly that and aborted every matmul on Strix Halo).
Results are bit-identical to an unchunked dispatch; no shape is narrowed or
clamped.

On a unified-memory device every device buffer is placed host-visible first
(`logit-loom-vulkan-uma-prefer-host-v19.patch`), exactly what
`GGML_VK_PREFER_HOST_MEMORY` selects. Upstream's UMA order asks first for the
`DeviceLocal|HostVisible|HostCoherent` type, which RADV backs with the 1 GiB
VRAM carve-out on Strix Halo; buffers that fit fill it, amdgpu page tables can
then no longer be placed, and the kernel's `BO_VA -ENOMEM` path wedged the GPU
on 2026-09-22. The patch has to be code: Brainstem passes workers no ambient
environment. Buffers remain host-visible, so UMA mapping and borrowing paths
are unchanged, and discrete GPUs are untouched.

Every image decode through an `AutoEncoderKL` VAE (the FLUX-lineage VAEs of
Z-Image Turbo, FLUX.2-dev and FLUX.2-klein-4B) uses upstream's default spatial
tiling: 256 px (32 latent cells) per tile with 0.5 overlap
(`logit-loom-vae-tiling-v20.patch`). A full-frame 1024x1024 decode needs one
Vulkan buffer above the device's ~4 GiB `maxBufferSize`, so it can never
allocate, and upstream's tiling retry chooses its tiles from the failed
allocation. Tiling is unconditional for this VAE, so an image never depends
on whether an allocation failed. The Wan VAE of Krea 2 Turbo and
Qwen-Image is not `AutoEncoderKL` and decodes exactly as before.

A companion context loads every model parameter before it returns
(`logit-loom-eager-load-v21.patch`). The artifact paths it is given are
`/proc/self/fd` descriptors that the caller may close once the context exists,
so a lazily loaded module reopened a closed descriptor on first use and its
compute came back empty. With every parameter resident at creation, a caller
can also disable `enable_mmap` and hold no host mapping of the weights after
load.

Image-to-image, inpaint, and outpaint strength may select a suffix of the
declared Euler schedule. The companion reports those executed transitions at
their original global indices and preserves the original total, including for
checkpoint continuation, scheduled LoRA boundaries, model-block operators,
and Krea activation mechanics. Rust validates the exact initial index and
reports observed and expected indices and totals on any mismatch.

## Port to 2988060a

Upstream moved the generation pipeline out of `src/stable-diffusion.cpp`
(`14eddb3`), unified runner lifecycles and weight residency (`462d675`,
`6b47fec`), and split the ggml Vulkan backend into several files. The port
keeps the companion ABI header byte-identical, so every ABI and extension
version is unchanged; only the reported upstream commit differs.

- The C API stays in `src/stable-diffusion.cpp`; condition reporting and the
  Krea reference guard moved to `src/pipeline/image.cpp`, and the sampling
  hooks to `src/pipeline/diffusion_engine.cpp`. Direct VAE calls run under
  upstream's `ExecutionScope`, and request-local `LoRA` detachment under its
  `ContextOperation`; a detachment failure is now a native error.
- `LoRA` attestation reads upstream's per-module `RuntimeLora` entries.
- The model-block extensions use a virtual `model_block_count()` instead of a
  `Krea2Runner` cast. Krea activation events are read through upstream's
  `read_outputs` hook while the graph's buffers are still allocated.
- Upstream now skips a zero-length Euler step without evaluating the model;
  the step callback still observes it, so every schedule index is reported.
- The `native-errors-v9` logger hunk is dropped: upstream `b8248a8` (#2002)
  formats log messages without the 4 KiB truncation it fixed.
- `vae-tiling-v20` is kept although upstream `b68d586`, `97d932b` and
  `ce95c09` added a tiling retry without `auto_fit`: that retry's tiles depend
  on which allocation failed. The default tile is now specified in image
  pixels (256 px, 32 latent cells for an 8x VAE).
- `upstream-parity-v23` holds four new upstream defaults at the old
  companion's behavior: `auto_fit` off, monolithic graphs
  (`disable_segmented_compute`), no batched CFG forward, and no conditioning
  cache.
- `sealed-sources-v23` stops a companion context from re-stamping, refreshing
  or demoting to disk residency the `/proc/self/fd` model sources it was given:
  upstream checks every model file's size and mtime before each execution
  and can reload evicted weights from their paths, and those descriptors are
  closed once the context exists.
- `igpu-capacity-v23` backports upstream PR #2124 (open, head
  `cdc66779`, issue #2022): a wrapped Vulkan free-memory report on an
  integrated GPU falls back to tracked residency instead of reporting zero.
- The eight ggml patches are ported to the split backend: the allocation
  ceiling's counter and helpers live in `ggml-vulkan-types.h` as C++17 inline
  definitions shared by every backend translation unit, allocation and its
  diagnostics in `ggml-vulkan-buffers.cpp`, and the dispatch-bound push
  constants in `ggml-vulkan-push-constants.h`. Upstream has none of the Strix
  Halo, UMA, ceiling or dispatch-bound changes.

The reviewed decision and boundary are in
[`docs/adr/0001-stable-diffusion-runtime.md`](../../docs/adr/0001-stable-diffusion-runtime.md).

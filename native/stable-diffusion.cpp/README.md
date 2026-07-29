<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# stable-diffusion.cpp companion ABI

This directory contains a narrow source patch for
`leejet/stable-diffusion.cpp@ea4e566ccffa10f853ecc3f29e74b1820bc91beb`.
That upstream project is MIT licensed. The patch is maintained under Logit
Loom's `MIT OR Apache-2.0` terms and does not vendor upstream source.

The companion ABI exposes:

- exact ABI and upstream revision queries;
- explicit MiniT2I-B/16 and Krea 2 Turbo context construction;
- deterministic CPU random-state selection with explicit accelerator model
  and parameter backends;
- exact conditioning tensors, including retained tokenizer ID tensors;
- one mutable boundary after each Euler state update, with native
  denoiser-plus-update elapsed time measured immediately before the callback;
- an image ABI v2 extension for bounded source/mask/reference views, negative
  conditioning, fixed request-local `LoRA` entries, direct VAE tensors, and explicit
  session cleanup;
- a mandatory program ABI v3 value arena for multiple diffusion/VAE stages,
  scheduled request-local `LoRA` scales, checkpoint and snapshot state,
  deterministic joins, exact RGB8/RGBA8 conversion, bounded deterministic
  PNG encoding, generation-checked handles, and verified cleanup;
- a model-block ABI v4 extension for typed, exact-step Krea residual scaling
  with loaded-topology validation;
- a model-block application ABI v5 extension that returns actual-transition
  bitmaps, graph-branch counts, loaded topology, and confirmed request-local
  control cleanup; and
- explicit continue, cooperative-stop, callback-error, unsupported-mechanic,
  invalid-argument, and native-error results.

Image ABI v2 checks that every requested `LoRA` participates in at least one
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

If the source path does not exist, the script clones the upstream repository.
It checks out the exact revision, applies
[`logit-loom-step-v1.patch`](logit-loom-step-v1.patch) followed by
[`logit-loom-image-v2.patch`](logit-loom-image-v2.patch) and
[`logit-loom-program-v3.patch`](logit-loom-program-v3.patch), then the
[`logit-loom-model-block-v4.patch`](logit-loom-model-block-v4.patch) extension,
and finally
[`logit-loom-model-block-application-v5.patch`](logit-loom-model-block-application-v5.patch),
initializes only the required `ggml` submodule, and builds a shared library.
Existing incompatible source changes are rejected. The script never runs from
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

The reviewed decision and boundary are in
[`docs/adr/0001-stable-diffusion-runtime.md`](../../docs/adr/0001-stable-diffusion-runtime.md).

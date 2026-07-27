<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Compatibility policy

## Rust and crate versions

The current minimum supported Rust version is 1.97.1, pinned in
`rust-toolchain.toml`. All publishable crates use the same release version; the
unpublished repository `xtask` remains `0.0.0`. The API is pre-1.0 and may
change between minor releases.

`logit-loom-runtime` is a convenience layer over the text crates, not a
separate compatibility authority. Its `LoomSession` preserves the adapter's
single-owner lifetime and checkpoint rules. Applications needing independent
native runtime/model ownership should use `logit-loom-llamacpp` directly.

`logit-loom-diffusion` is the backend-neutral image mechanics layer.
`logit-loom-diffusion-sdcpp` is a separate native adapter; it is not hidden
behind the text runtime.

`logit-loom-executor` defines a transport-neutral synchronous ownership seam.
It is not a wire protocol. Its borrowed slices and lifecycle traits may be
used inside a downstream worker, but their Rust representation is not a
cross-process compatibility promise.

## Contract bounds

Serialized plans expose their collection and byte limits as public constants.
The current generation contract bounds logit biases, DRY sequence breakers,
grammar source/root bytes, and exact stop sequences. Transform and observer
runtimes also bound stage, candidate, and fan-out counts before copying caller
data or entering a native boundary.

Tightening a bound changes accepted input and is recorded as a compatibility
change. Changing a serialized field or digest interpretation requires a new
versioned digest domain rather than reusing an existing identity.

The first release uses `prefill-receipt-v2`; it distinguishes in-progress
monitor accounting from the terminal `complete` and `stopped` states.
Checkpoint metadata uses `checkpoint-receipt-v2`, whose backend identity binds
the exact session allocation contract as well as the adapter build.

The higher-level built-in helper domains are
`runtime-rank-bias-v1`, `runtime-token-bias-v1`, and
`runtime-cancellation-observer-v1`. Runnable experiment reports identify exact
token IDs plus output bytes under `runtime-generation-output-v1`. A change to
the corresponding built-in mechanics or normalized configuration shape
requires a new domain. Custom transforms and observers retain caller-defined
implementation identities.

Diffusion contracts use distinct domains including
`diffusion-tensor-spec-v1`, `diffusion-schedule-v1`,
`diffusion-plan-v1`, `diffusion-intervention-spec-v1`,
`diffusion-channel-bias-v1`, and
`sdcpp-deterministic-prefix-replay-v1`. Whole-image and worker-local contracts
add `executor-buffer-spec-v1`, `image-execution-plan-v1`, and
`image-execution-receipt-v1`. Image ABI v2 requests use
`sdcpp-image-request-v2`; source, mask, reference, LoRA, and VAE byte
identities have separate versioned domains rather than incorporating local
paths. Image experiment reports identify an exact serialized native generation
receipt under
`sdcpp-generation-receipt-v2`; version 2 adds the exact native session epoch
and does not reinterpret version 1 identities. Final pixel bytes retain the
`sdcpp-image-u8-v1` identity assigned by the adapter. Tensor layout, callback
timing, schedule interpretation, state-byte encoding, or serialized-shape
changes require new domains rather than reinterpretation.

`logit-loom-llamacpp` pins `llama-cpp-4` exactly to version 0.4.2. A binding
upgrade is a reviewed compatibility change: compile the complete workspace,
inspect changed native semantics, rerun opt-in model fixtures, and update this
document and `CHANGELOG.md`.

## Native build features

The adapter forwards these `llama-cpp-4` features without selecting one by
default. `logit-loom-runtime` forwards the same feature names to the adapter:

| Logit Loom feature | Native backend or build mode |
| --- | --- |
| `cuda` | NVIDIA CUDA |
| `hip` | AMD HIP |
| `vulkan` | Vulkan |
| `metal` | Apple Metal |
| `opencl` | OpenCL |
| `webgpu` | WebGPU |
| `blas` | BLAS support |
| `openmp` | OpenMP support |
| `rpc` | llama.cpp RPC support |
| `dynamic-link` | Dynamic native libraries |
| `prebuilt` | Binding-provided prebuilt artifacts |
| `native-cpu` | Host-native CPU tuning |

Features are additive at Cargo's resolver. Applications should select a
supported deployment combination explicitly instead of enabling every feature.
The default feature set is useful for API compilation and does not promise an
accelerated runtime.

The image adapter dynamically loads companion ABI version `1` and requires
whole-image extension version `2`, built from
`stable-diffusion.cpp@ea4e566ccffa10f853ecc3f29e74b1820bc91beb`. The safe
Rust adapter contract is version `4`. The exact ABI, extension, commit,
required symbol set, and shared-library bytes are checked before model context
creation. A library carrying only the earlier step-v1 symbols is incompatible
with this adapter version. Updating the upstream revision, companion layout,
Euler callback boundary, image operation layout, or native tensor
interpretation is a reviewed compatibility event and requires renewed
model-backed acceptance.

The version `1` step descriptor includes a finite, nonnegative elapsed-time
field measured across the native denoiser and Euler update immediately before
the Rust callback. The adapter retains those per-step values as
non-deterministic deployment measurements outside plans, receipts, checkpoint
lineage, and content identities. Any future descriptor-layout change requires
a new companion ABI version.

Image extension version `2` adds exact RGB/RGBA/Gray byte views, bounded
negative conditioning and references, fixed request-local LoRA bindings,
direct VAE tensors, and explicit native session cleanup. It supports only one
fixed scale for each LoRA during a complete request. Per-step scale schedules,
model-specific target selectors, PNG encoding, arbitrary model-block
operators, and multiple native inference operations require another reviewed
adapter contract; a lowerer must reject them instead of approximating them.

Safe contract version `4` does not change companion ABI v1 or image extension
v2. It adds a public `ImageExecutionPlanV3` lowering that combines advanced
image inputs and fixed `LoRA` bindings with the existing full-state callback,
then performs versioned checkpoint routing, bounded deterministic RGB8
compositing, explicit output routing, and cleanup accounting in Rust. The
version-one image plan and receipt domains remain unchanged.

The companion is prepared explicitly for `vulkan`, `hip`, `cuda`, or `metal`.
This is independent of the llama.cpp Cargo feature selection. A model-free
companion probe may report CPU-only devices; image generation additionally
requires an exact caller-selected non-CPU device name.

## Device placement

`ModelOptions::default` requests the maximum GPU layer offload and uses
`DevicePolicy::RequireAccelerator`. The policy rejects a loaded model when
llama.cpp reports no GPU, integrated-GPU, or accelerator device. It confirms
accelerator participation; it does not claim that every tensor, tokenizer
operation, or orchestration step runs on the accelerator.

Applications with stricter placement requirements should inspect
`Model::devices`, record deployment telemetry, and reject a configuration that
does not meet their own policy. Logit Loom does not silently retry a rejected
load with CPU-only inference.

## Checkpoint compatibility

Checkpoint receipts bind:

- exact GGUF file bytes;
- the Logit Loom and `llama-cpp-4` versions;
- target architecture, operating system, endianness, and enabled adapter
  features;
- exact context, batch, micro-batch, and thread options;
- exact native state and token-history bytes.

These checks are intentionally conservative but not a guarantee that arbitrary
native builds serialize identical state. Dynamic-link deployments can replace
native libraries without changing Cargo metadata; keep their checkpoints
within one controlled deployment and add an application-level native library
identity if long-lived portability matters.

`StateSnapshot::from_parts` makes application-defined persistence possible; it
does not relax these compatibility rules. Authenticate the stored bytes and
metadata before reconstructing untrusted state.

The pinned llama.cpp state API serializes causal memory but not the
next-token output logits. After consuming every checkpoint byte, the adapter
therefore removes only the final sequence position and re-decodes its exact
recorded token to restore the next-token boundary. A backend whose memory
cannot remove that position is incompatible with checkpoint replay. A partial
native restore or a failed post-restore logit refresh poisons the session
because its resulting backend state is unknown.

Diffusion checkpoints are exact post-Euler host `f32` bytes bound to the full
`DiffusionPlan`, companion/runtime identity, and completed step. Restore is
deterministic-prefix replay: the adapter recomputes and authenticates the
selected boundary before copying checkpoint bytes. It does not promise native
skip-ahead, cross-backend portability, or compatibility across conditioning
or schedule changes.

## Optional model profile compatibility

The optional acquisition catalog uses
`logit-loom-model-catalog-v1`. Each source is bound to an exact upstream
repository commit, an explicit file allow-list, byte counts, and SHA-256
digests for weight artifacts. The acquisition command does not use moving
branches, wildcard file selection, or remote model code.

A profile revision or file-set update is a reviewed compatibility event. It
does not replace the old artifact identity or inherit its acceptance result.
The new profile must pass local artifact verification and repeat the applicable
opt-in accelerator acceptance lane before its status can advance.

`catalogued` means only that acquisition metadata passes repository
validation. Qwen, MiniT2I, and Krea have maintained loader/adapter and runbook
code plus exact Vulkan acceptance reports, so all three profiles are
first-class with passed acceptance status. Repository checks reject a
first-class profile without passed acceptance status and reject a passed
status without a matching retained passed report. Model-backed acceptance
records exact artifact, adapter-build, feature, device-placement, and
allocation identities using the output-free acceptance schema.

Model licenses are upstream terms and are not changed by this repository's
license. A gated profile requires prior upstream acceptance and explicit local
acknowledgement before download. Authentication remains under the `hf` CLI;
the acquisition command does not accept credentials as arguments.

## Artifact compatibility

Text `LoRA` compatibility is ultimately validated by llama.cpp at
load/application. Image ABI v2 additionally requires every requested
request-local LoRA to match and participate in at least one native model tensor
before returning success; absence of a compatible target is a rejected
mechanic, not a successful no-op.
Model and `LoRA` files use distinct versioned content-identity domains even
when their bytes happen to match. The adapter hashes each file before and after
native loading and rejects an ordinary concurrent modification; callers should
still treat artifact paths as immutable for the complete load.
Control vectors are checked before application for finite values, model
embedding width, complete rows for layers `1..n_layer`, and an inclusive layer
range within the model. Layer zero is not steerable.

Model files, adapters, grammars, control vectors, and state bytes are untrusted
native inputs. Pin their provenance and authenticate them at the application
layer when they cross a trust boundary.

## Native build cache inputs

`llama-cpp-sys-4` uses a shared CMake cache keyed by source, target, and backend
features. Its current cache key does not include ambient `CFLAGS` or
`CXXFLAGS`. Changing those variables after a native build can therefore reuse
incompatible objects. In particular, GCC slim-LTO objects cannot be bundled
into a Rust archive and linked by LLVM's linker.

For reproducible builds, avoid ambient native `-flto` flags and use a clean or
isolated Cargo target when compiler flags change. This is a build-artifact
compatibility issue; it does not involve model execution.

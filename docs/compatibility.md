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

The llama.cpp adapter also exposes read-only native backend and model borrows
for coordinators that must compose non-Logit-Loom mechanics in the same
process without initializing or loading a second native owner. Those calls
remain outside Logit Loom's plan and receipt identities. The adapter retains
backend/model lifecycle ownership, and no raw pointer or mutable native handle
is exposed.

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

Transactional text mechanics use distinct domains including
`text-model-topology-v1`, `activation-capture-plan-v1`,
`activation-vector-bank-v1`, `activation-program-v1`,
`activation-invocation-receipt-v1`, `speculation-plan-v1`,
`speculation-boundary-receipt-v1`, `speculation-receipt-v1`,
`text-mechanics-plan-v2`, `text-mechanics-checkpoint-receipt-v2`, and
`text-mechanics-receipt-v2`. Provisional and resolved activation rows have
different receipt identities. No version-one aggregate text-mechanics value is
reinterpreted.

Diffusion contracts use distinct domains including
`diffusion-tensor-spec-v1`, `diffusion-schedule-v1`,
`diffusion-plan-v1`, `diffusion-intervention-spec-v1`,
`diffusion-channel-bias-v1`, and
`sdcpp-deterministic-prefix-replay-v1`. Whole-image and worker-local contracts
add `executor-buffer-spec-v1`, `image-execution-plan-v1`, and
`image-execution-receipt-v1`. Resident staged programs add
`image-program-stage-operation-v1`, `image-program-plan-v1`, and
`image-program-receipt-v1`; their deployment measurements are deliberately
excluded from those deterministic identities. Existing image-execution
domains are not reinterpreted. Image ABI v2 requests use
`sdcpp-image-request-v2`; source, mask, reference, LoRA, and VAE byte
identities have separate versioned domains rather than incorporating local
paths. Image experiment reports identify an exact serialized native generation
receipt under `sdcpp-generation-receipt-v3`; version 3 adds optional exact
direct-continuation checkpoint lineage, while version 2 added the exact native
session epoch. Earlier identities are not reinterpreted. Final pixel bytes
retain the `sdcpp-image-u8-v1` identity assigned by the adapter. Tensor layout, callback
timing, schedule interpretation, state-byte encoding, or serialized-shape
changes require new domains rather than reinterpretation.

Krea activation mechanics use distinct domains including
`krea-activation-topology-v1`, `krea-activation-input-content-v1`,
`krea-activation-plan-v1`, and `krea-activation-receipt-v1`. Topology,
deterministic evidence, and non-deterministic placement, transfer, and resource
measurements remain separate. Create-new component transforms use
`projected-component-plan-v1`, `projected-component-source-v1`,
`projected-component-basis-f32-le-v1`, `projected-component-output-v1`, and
`projected-component-manifest-v1`. Derived output bytes never inherit the
source artifact identity.

`logit-loom-llamacpp` pins the registry-published public `llama-cpp-4` 0.5.0
release. Its crates.io package records upstream source revision
`f1c5dd05906a11aee5c2eaf1265851bf29752d67` and carries literal llama.cpp
revision `221f0f6356efe2260023208365705ec5d5a7c8f5` (`b10235`). The binding adds
bounded tensor transactions, native decode-lifecycle hooks, lifetime-bound MTP
and EAGLE-3 sessions, versioned exact implementation state,
allocation-reusing tokenization and raw-piece sinks, and a count-only tokenizer
query. The workspace resolves the exact registry version and checksum rather
than a mutable branch or adjacent checkout.

At this revision an EAGLE-3 v3 draft for `gpt-oss` may name the terminal
target `NextN` extraction site as layer index `n_layer`. The adapter admits
that otherwise-out-of-range index only for the exact `gpt-oss` architecture
profile. Missing target or draft extraction output returns a native process
failure instead of aborting the host process.

The successor was merged upstream in
[`eugenehp/llama-cpp-rs#301`](https://github.com/eugenehp/llama-cpp-rs/pull/301)
and released to crates.io, satisfying the adapter's registry-publication gate.
The release check continues to reject path and Git sources. Changing the
binding source, llama.cpp revision, graph-selector profile, decode hooks, or
speculative-state envelope is a reviewed compatibility event: compile the
complete workspace, inspect changed native semantics, and update this document
and `CHANGELOG.md`. Renew the opt-in model fixtures before claiming live-model
compatibility for a changed native revision.

The readable runtime compatibility label includes the exact binding version,
binding source revision, literal llama.cpp revision, Rust target, and selected
native feature set.
`llamacpp-binding-identity-v2` binds that label, and
`llamacpp-session-compatibility-v3` additionally binds exact context, batch,
micro-batch, thread, context-type, and recurrent-slot allocation. These
domains supersede the pre-revision identities rather than reinterpreting them.

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

At the reviewed 0.5.0 release, the binding still cannot authenticate prebuilt
archives against its active native patch set. Selecting `prebuilt` therefore
uses the verified source build with a warning, and an explicit unverifiable
`LLAMA_PREBUILT_DIR` is rejected. This feature must not skip the source build
until successor assets carry an exact patch identity.

Features are additive at Cargo's resolver. Applications should select a
supported deployment combination explicitly instead of enabling every feature.
The default feature set is useful for API compilation and does not promise an
accelerated runtime.

The image adapter dynamically loads companion ABI version `2` and requires
whole-image extension version `3`, resident-program extension version `3`,
model-block extension version `4`, application-evidence extension version `5`,
and Krea activation extension version `6`, built from
`stable-diffusion.cpp@ea4e566ccffa10f853ecc3f29e74b1820bc91beb`. The safe
Rust adapter contract is version `9`. The exact ABIs, extensions, commit,
required symbol set, and shared-library bytes are checked before model context
creation. A library carrying only the earlier step-v1 symbols is incompatible
with this adapter version. Updating the upstream revision, companion layout,
Euler callback boundary, image operation layout, or native tensor
interpretation is a reviewed compatibility event and requires renewed
model-backed acceptance.

The version `2` step descriptor includes a finite, nonnegative elapsed-time
field measured across the native denoiser and Euler update immediately before
the Rust callback. The adapter retains those per-step values as
non-deterministic deployment measurements outside plans, receipts, checkpoint
lineage, and content identities. Any future descriptor-layout change requires
a new companion ABI version.

Image extension version `3` retains the version-two image mechanics and adds
an optional exact post-step scheduler-state continuation pointer, element
count, and next-transition index. The companion validates the finite state
against the native latent geometry and begins Euler sampling at that exact
index. Version `2` originally added exact RGB/RGBA/Gray byte views, bounded
negative conditioning and references, fixed request-local LoRA bindings,
direct VAE tensors, and explicit native session cleanup. It supports only one
fixed scale for each LoRA during a complete request. Per-step scale schedules,
model-specific target selectors, PNG encoding, arbitrary model-block
operators, and multiple native inference operations require another reviewed
adapter contract; a lowerer must reject them instead of approximating them.

Safe contract version `9` retains the version-four `ImageExecutionPlanV3`
lowering and requires resident-program extension v3, model-block extension v4,
application-evidence extension v5, and Krea activation extension v6 for
`ImageProgramPlanV1`. The resident contract supports multiple native/VAE
stages, typed values, scheduled adapters, checkpoint and snapshot mechanics,
Krea residual block scaling at exact denoising transitions, and explicit
value-arena cleanup. Source images and masks remain stage-canvas bound;
reference images preserve their own bounded RGB8/RGBA8 geometry and exact
bytes. The Krea block count is discovered from loaded weights and does not
assign semantic meaning to any block.

Activation ABI v6 adds runtime-derived site widths and supported
site/domain/branch/boundary masks; generation-checked resident donor/vector
handles; request-local capture and operation arrays; callback evidence;
observed host/device peaks; and explicit release/clear calls. Supplying an
activation plan installs it completely; there is no compatibility mode that
silently drops an operation. Sealed inputs are imported once and re-verified
before each same-session job. Same-run device-snapshot donors report no
host-to-device transfer. Any layout, selector, callback, handle, resource, or
cleanup change requires a new activation ABI and safe contract version.

The projected-component transform is not part of the native ABI. It creates a
new immutable SafeTensors artifact before ordinary import. Its exact source,
basis, topology, tensor selection, formula, reduction, implementation, output,
and manifest identities are compatibility inputs; changing any transform
mechanic requires a new versioned domain.

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
recorded token to restore the next-token boundary. Checkpoint-capable ordinary
aggregate execution allocates one recurrent-state rollback snapshot for a
recurrent or hybrid model and binds the native context's resulting slot count
into `llamacpp-session-compatibility-v3`. A backend architecture that still
cannot remove that position is incompatible with checkpoint replay. A partial
native restore or a failed post-restore logit refresh poisons the session
because its resulting backend state is unknown.

The successor binding can capture target sequence state, draft sequence state,
and versioned MTP or EAGLE-3 implementation state at a quiescent boundary.
`generate_speculative_checkpointed` adds the exact in-process target-sampler
clone, activation configuration, stop-prefix state, causal history, completed
boundaries, and parent lineage. `resume_speculative_checkpointed` authenticates
that complete envelope before allocation and clones the sampler for each
independent branch.

`SpeculativeCheckpointReceiptV1` is portable evidence, not a portable
checkpoint container. The opaque sampler has no encoding and the snapshot is
therefore process-local and thread-affine. Logit Loom does not substitute the
serializable receipt for unavailable native sampler state.

Diffusion checkpoints are exact post-Euler host `f32` bytes bound to the full
`DiffusionPlan`, companion/runtime identity, and completed step. Before native
entry, the adapter authenticates the state envelope, step range, and current
backend. Native conditioning reconstructs the plan; the first resumed boundary
must authenticate that plan before any result is accepted. The companion
copies the finite checkpoint state into the latent and resumes at the recorded
next Euler transition without replaying completed transitions. It does not
promise cross-backend portability or compatibility across conditioning or
schedule changes. A resident program that captures and restores its own
checkpoint is rejected during preflight when its statically bound profile,
load, dimensions, seed, RNG, placement, schedule, guidance, conditioning,
reference, or `LoRA` lineage already proves that the reconstructed plans will
differ.

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
load/application. Image ABI v3 additionally requires every requested
request-local LoRA to match and participate in at least one native model tensor
before returning success; absence of a compatible target is a rejected
mechanic, not a successful no-op.
Model and `LoRA` files use distinct versioned content-identity domains even
when their bytes happen to match. The adapter hashes each file before and after
native loading and rejects an ordinary concurrent modification; callers should
still treat artifact paths as immutable for the complete load.

`logit-loom-llamacpp` also exposes narrow provider-authorized model and `LoRA`
load contracts for local resource authorities that own authentication and an
immutable sealed-descriptor lifecycle. The caller supplies a stable artifact
identity and exact byte length; Logit Loom checks length around native loading
without rereading the entire payload. `Model::load_preverified` remains
available when the authority already has a raw BLAKE3 model digest. These paths
do not weaken or replace the ordinary path contract: mutable and otherwise
unverified paths retain before-and-after content verification.
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

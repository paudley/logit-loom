<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Compatibility policy

## Rust and crate versions

The current minimum supported Rust version is 1.97.1, pinned in
`rust-toolchain.toml`. All workspace crates use the same release version. The
API is alpha and may change before a stable release.

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

`logit-loom-llamacpp` pins `llama-cpp-4` exactly to version 0.4.2. A binding
upgrade is a reviewed compatibility change: compile the complete workspace,
inspect changed native semantics, rerun opt-in model fixtures, and update this
document and `CHANGELOG.md`.

## Native build features

The adapter forwards these `llama-cpp-4` features without selecting one by
default:

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

A native restore that consumes fewer than all checkpoint bytes poisons the
session because its resulting backend state is unknown.

## Artifact compatibility

`LoRA` compatibility is ultimately validated by llama.cpp at load/application.
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

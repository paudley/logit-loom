<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Capability status

This document records implemented functionality and its validation boundary.
It intentionally makes no model-quality or research-efficacy claims.

| Capability | Public surface | Repository validation |
| --- | --- | --- |
| Serializable plans and receipts | `logit-loom-core` | Unit tests, doctests, strict lint, rustdoc |
| Full-vocabulary transforms | `logit-loom::Pipeline` | In-memory behavioral tests and compiled examples |
| Sparse ranked transforms | `logit-loom::Pipeline` | Ordering and write-back tests |
| Backend-selected candidate transforms | `Pipeline::apply_to_candidates` | In-memory shape, sparse-bound, and transactional tests |
| Callback error/panic containment | `Pipeline`, `ObserverSet`, `PrefillMonitor` | In-memory failure tests |
| Exact byte token observation | `ObservedToken` | Non-UTF-8 fixture test and runnable example |
| Cooperative cancellation | `CancellationToken`, observer control flow | Cross-thread signal test |
| Native sampler translation | `GenerationPlan` through llama.cpp adapter | Type/API compilation against pinned binding |
| Causal prefill and generation | `Session` | Type/API compilation; model execution is opt-in |
| Checkpoint capture/restore | `StateSnapshot` | Reconstruction/accounting unit tests; native capture/restore is opt-in |
| Scoped `LoRA` | `LoraScope` | Type/API compilation; model/adapter execution is opt-in |
| Scoped control vectors | `ControlVectorScope` | Dimension/lifecycle code and type/API compilation; model execution is opt-in |
| Higher-level request and control construction | `GenerationRequest`, `PipelineBuilder`, `ObserversBuilder` | Model-free bounds, identity, transform, observer, and exact-byte tests |
| Higher-level one-shot and stateful workflows | `Loom`, `LoomSession` | Complete type/API compilation and doctests; model execution is opt-in |
| Typed higher-level steering scopes | `LoraSession`, `ControlVectorSession` | Type/API compilation; resource execution is opt-in |
| Worker-local executor contracts | `logit-loom-executor` | Exact buffer-length, initialized-prefix, metadata serialization, lifecycle, cleanup, cancellation, and failure-disposition unit tests plus strict lint/rustdoc |
| Backend-neutral diffusion contracts | `logit-loom-diffusion` | Bounds, schedule/shape identity, checkpoint mismatch, transactional order/write-back, panic/error, observer, and receipt tests |
| Serializable whole-image execution | `ImageExecutionPlan`/`ImageExecutionReceipt` and versioned `ImageExecutionPlanV3`/`ImageExecutionReceiptV3` | Version-one compatibility plus version-two graph/reference bounds, 512 MiB aggregate scratch ceiling, exact checkpoint consumption/routing, deterministic mask blending, output accounting, cleanup disposition, cancellation, and receipt-lineage unit tests |
| Versioned stable-diffusion.cpp boundary | Companion ABI v1, image ABI v2, and `probe_companion` | Exact patch compilation/probe is opt-in; Rust symbol/descriptor/callback safety and per-step timing validation paths have model-free tests |
| Advanced whole-image operations | `AdvancedImageRequest`, `Sdcpp::generate_advanced_program_to`, `ImagePlanExecutor` | Model-free tests for exact source/mask geometry, path-free request receipts, bounded references, fixed LoRA stacks, checkpoint-envelope authentication, stale-backend rejection, post-observation cancellation, output accounting, and lifecycle classification; complete version-two graph model execution is opt-in and not yet retained as an acceptance fixture |
| Direct Krea VAE boundary | `Sdcpp::vae_encode`, `Sdcpp::vae_decode`, `VaeTensor` | Rank, shape, finite-value, element-count, native-descriptor, and allocation-ownership paths have model-free tests; model execution is opt-in |
| Request-local image LoRA cleanup | Image ABI v2 fixed LoRA stack | Native build verifies each requested adapter reaches at least one model tensor, then clears the stack before reusable returns; compatible model execution remains opt-in |
| Pinned Qwen execution | `logit-loom-runtime` | First-class exact artifact/device checks plus retained Vulkan checkpoint replay and exact token-byte identity |
| Pinned MiniT2I execution | `logit-loom-diffusion-sdcpp` | First-class exact artifact/ABI/device checks plus retained Vulkan checkpoint replay, intervention, state dtype/placement, and native step-latency acceptance |
| Pinned Krea execution | `logit-loom-diffusion-sdcpp` | First-class exact artifact/ABI/device checks plus retained Vulkan checkpoint replay, bounded latent intervention, state dtype/placement, native step latency, and qualified deployment-memory observations |
| End-to-end mechanical experiment runbooks | Eight text and image examples with structured reports | Complete type/API compilation; caller-run model, image, and `LoRA` execution is opt-in |
| Pinned optional model acquisition | `models/profiles.json`, `logit-loom-xtask` | Bounded catalog and report validation, exact command tests, no-network dry run, and exact artifact verification for all three profiles |
| Output-free acceptance projection | `docs/acceptance/model-run.schema.json` | Bounded version/domain checks plus enforced report presence for passed profiles; retained Qwen, MiniT2I, and Krea reports separately pass JSON Schema validation |
| Bulk-tokenizer substrate (unpublished, partial) | `logit-loom-tokenizer` | Versioned identities; safe ranked BPE with packed `u32x8` short-span and deterministic heap paths; direct/reusable sinks; counting without a complete output token vector; bounded dedicated pool/backpressure; cache, batch, and chunk tests; and a caller-supplied exact oracle framework. Model-specific normalization/pretokenization/special-token adapters, retained engine-oracle parity, and whole/partitioned stress qualification remain open |

## Validation vocabulary

- **In-memory behavioral test** means repository tests executed the Rust
  behavior without a model runtime.
- **Type/API compilation** means the adapter compiled against the exact pinned
  native binding and exercised no model.
- **Opt-in model execution** requires a caller-supplied compatible artifact and
  explicit accelerator backend. It is not part of ordinary CI.
- **Catalogued model profile** means exact acquisition metadata is checked in
  and validated. It does not mean that an adapter or model-backed fixture has
  passed.
- **First-class model profile** means the maintained adapter and runbook are
  present and a catalog-bound, output-free accelerator acceptance report has
  passed the repository checks.
- **Companion probe** means the exact dynamic symbols, ABI, upstream revision,
  library bytes, and bounded device report loaded without a model. It is not
  accelerator inference.

Passing a mechanical test proves that the described contract path works for
that fixture. It does not establish output quality, usefulness for a workload,
or performance at a particular prompt length. Those questions require a
separate corpus, workload definition, baselines, and statistical analysis.

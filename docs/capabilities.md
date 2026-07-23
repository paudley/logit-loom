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

## Validation vocabulary

- **In-memory behavioral test** means repository tests executed the Rust
  behavior without a model runtime.
- **Type/API compilation** means the adapter compiled against the exact pinned
  native binding and exercised no model.
- **Opt-in model execution** requires a caller-supplied compatible artifact and
  explicit accelerator backend. It is not part of ordinary CI.

Passing a mechanical test proves that the described contract path works for
that fixture. It does not establish output quality, usefulness for a workload,
or performance at a particular prompt length. Those questions require a
separate corpus, workload definition, baselines, and statistical analysis.

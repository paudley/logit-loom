<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# ADR 0005: topology-bound Krea activation mechanics

- Status: accepted; public contract, safe adapter, and native ABI implemented;
  model-backed semantic efficacy is outside this decision
- Decision date: 2026-08-01
- Builds on: [ADR 0003](0003-resident-image-programs.md)
- Native runtime: pinned stable-diffusion.cpp companion

## Context

The resident image program can already operate on the post-Euler scheduler
state and on complete Krea transformer-block residuals. Those boundaries do
not expose conditioning-layer outputs, post-fusion or post-projection values,
text residuals, or token-scoped transformer residuals. They also cannot retain
a donor activation or vector bank once and reuse it across jobs in one
resident session.

Adding application-specific activation policy to the public crate would be the
wrong boundary. The reusable contract needs exact topology, selection,
operation, resource, and cleanup mechanics while leaving vector construction,
prompt choice, and interpretation to its consumer.

## Decision

Add the always-available `KreaActivationPlanV1` contract to
`logit-loom-diffusion` and implement it in
`logit-loom-diffusion-sdcpp` through native Krea activation ABI version 6 and
safe adapter contract version 8.

The loaded runtime publishes the exact available sites, widths, boundary
kinds, token domains, CFG branches, backend identity, and Krea block count as
`KreaActivationTopologyV1`. Plans bind that topology and the exact denoising
transition count. There is no feature flag, policy gate, classifier, or
optional enablement field: supplying a valid plan installs the complete
request.

One plan may:

- capture selected text, image, or reference-token rows as a digest,
  deterministic statistics, or a device-resident snapshot;
- import exact sealed donor tensors or vector banks once into the resident
  session;
- consume a same-run device snapshot as an SSA donor without a host-to-device
  transfer;
- transplant donor rows, add or subtract one scaled vector, or remove an
  orthogonal or one-sided projection;
- scope each operation to an exact site, token selection, logical
  pre-denoiser or transition boundary, and CFG branch; and
- reuse the imported inputs across jobs in the same resident session after
  verifying their native handle, shape, placement, and identity.

All arrays, handles, hooks, and callbacks remain request-local or
session-local inside one synchronous in-process native invocation. The ABI
does not introduce per-step process IPC.

The plan bounds retained host bytes, retained device bytes, and aggregate
native applications. Native execution reports the observed host and device
peaks, exact capture/application counts, before/after content identities,
unchanged writes, imported-input placement and copy accounting, terminal
boundary, and confirmed cleanup. Evidence inconsistent with the plan poisons
the resident owner. Clearing an activation plan releases every imported input;
release and final clear are idempotent.

The logical pre-denoiser boundary is evidenced once per generation. The
companion may reconstruct the corresponding operation inside its native graph
execution; that implementation detail does not create additional public
boundaries or receipts.

## Validation

Default model-free tests cover topology validation, canonical selections,
bounds, input consumption, orthonormal-vector validation, capture SSA,
operation compatibility, zero-strength behavior, receipt lineage, native
descriptor lowering, callback accounting, resource peaks, stale handles, and
cleanup poisoning. The complete patch stack compiles from the pinned upstream
revision and exports the six required ABI-v6 symbols without loading a model.

Model-backed execution requires caller-supplied artifacts and explicit
accelerator admission. A mechanical execution receipt proves only that the
selected native boundaries and operations ran. It does not prove that a
capture or operation represents a concept, improves an image, changes safety,
or has any other useful semantic effect.

## Consequences

Private consumers can install exact Krea activation programs without adding
model-loading or accelerator-coordination code and can keep sealed inputs
resident across jobs. Other model families and new tensor sites require a new
reviewed topology mapping; the adapter must reject them rather than infer a
compatible meaning.

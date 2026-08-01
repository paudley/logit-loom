<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# ADR 0006: create-new projected components

- Status: accepted and implemented
- Decision date: 2026-08-01
- Public owner: `logit-loom-diffusion`

## Context

Some consumers need an artifact whose selected weight matrices have a bounded
projection removed before ordinary model import. Treating this as a mutable
load option would hide a persistent byte transformation inside runtime setup,
make source and derived artifacts share an identity, and risk modifying a
verified source component in place.

The transformation is reusable model mechanics. Artifact storage, admission,
deployment policy, and choosing a basis remain downstream responsibilities.

## Decision

Add `ProjectedComponentPlanV1` and
`project_safetensors_component_v1` as a deterministic create-new transform for
floating-point SafeTensors components.

The plan binds the exact source bytes, runtime-compatible topology identity,
canonical little-endian orthonormal basis, selected matrix names, matrix side,
projection strength, deterministic reduction rule, and implementation
identity. It supports exactly:

```text
W' = (I - alpha U U^T) W
W' = W (I - alpha U U^T)
```

Selected tensors must be finite rank-two `F32` matrices with a feature axis
matching the declared basis. Basis rows must be finite and orthonormal within
the documented mechanical tolerance. Computation follows canonical
row/rank order, accumulates in `f64`, and casts once to `f32`.

The transform allocates new output bytes. It preserves the exact source
header, metadata, tensor offsets, file length, and every unselected byte. It
never writes to the source. `ProjectedComponentManifestV1` records complete
source, topology, basis, plan, implementation, output, and per-tensor
before/after lineage. Verification recomputes the complete output and manifest.

The derived component has its own content identity and is imported later by
an ordinary runtime artifact path. This function does not load a model,
publish an artifact, or make the derived bytes equivalent to the source.

## Validation

Model-free tests cover both matrix sides, deterministic output, zero-strength
identity, source preservation, exact manifest verification, malformed and
aliased SafeTensors ranges, unsupported dtype/rank, missing selectors,
non-finite values, incompatible axes, non-orthonormal bases, canonical
ordering, and identity substitution.

These checks prove a byte transform and its lineage. They do not prove that a
chosen basis has a semantic interpretation or that the derived component
improves model behavior.

## Consequences

Consumers receive a narrow, reproducible artifact-building primitive rather
than a hidden runtime mutation. They must store and admit the derived output
as a separate immutable artifact and keep the manifest with its provenance.

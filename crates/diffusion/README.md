<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# logit-loom-diffusion

Backend-neutral mechanics for iterative image-generation state. This crate
keeps diffusion tensors distinct from token IDs and logits.

It provides bounded, serializable tensor, schedule, plan, checkpoint, stage,
and receipt contracts; ordered transactional intervention pipelines; and
post-step observers. A pipeline copies one finite `f32` state, applies stages
in declared order, and commits to the caller's slice only after every stage
succeeds. Callback errors and panics are contained before commit.

The crate also defines compatible worker-local whole-image contract families:

- `ImageExecutionPlan` and `ImageExecutionReceipt` preserve the version-one
  operation, buffer, placement, seed, schedule, `LoRA`, operator,
  observation, and terminal-position domains.
- `ImageExecutionPlanV2` and `ImageExecutionReceiptV2` add transactional
  checkpoint restore/capture, an ordered deterministic compositing graph,
  explicit caller-owned output routes, and request-scope cleanup disposition
  under new identity domains.
- `ImageExecutionPlanV3` and `ImageExecutionReceiptV3` preserve the current
  single-native-primary lowering under their own identity domains.
- `ImageProgramPlanV1` and `ImageProgramReceiptV1` define a separate bounded
  resident program over typed single-assignment values and multiple ordered
  native, mask-blend, checkpoint-restore, and checkpoint-capture stages.

The single-primary contracts define one bounded RGB8 diffusion primary
followed by zero or more exact integer `MaskBlend` stages. The resident-program
contract adds canonical value producers, earlier-stage-only references,
operation-specific type and geometry checks, immutable branching, mutable
checkpoint-state single consumption, liveness-derived release points, a
2 GiB arena ceiling, explicit value and receipt routes, completed-stage
prefix receipts, cleanup uncertainty, and placement/transfer measurements
outside deterministic identity. Stage-produced values use the
`image-program-value-content-v1` content domain so an executor can verify
explicit materialization without reinterpreting the value.

Source images and masks are bound to the native stage canvas. Reference images
instead retain their own bounded, tightly packed RGB8 or RGBA8 dimensions in
the serialized value specification and plan identity; the contract performs no
implicit resize, crop, or byte transformation.

`mask_blend_rgb8` validates every length before its first write. These
contracts contain no paths, transport, queue, native handles, or feature-gated
availability. An adapter must reject plan mechanics it cannot implement
exactly.

The crate owns no tensor runtime and performs no model inference. An adapter
must prove that its native state has the reported shape, dtype, device,
schedule step, plan identity, and contiguous dimension-zero-fastest layout
before calling these mechanics.

```rust
use std::collections::BTreeMap;

use logit_loom_core::Digest;
use logit_loom_diffusion::{
    ChannelBias, DiffusionPlan, DiffusionSchedule, Pipeline, StepContext,
    TensorDType, TensorLayout, TensorSpec,
};

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let tensor = TensorSpec::new(
        vec![2, 2, 3, 1],
        TensorDType::F32,
        TensorLayout::DimensionZeroFastest,
        "vulkan0",
    )?;
    let schedule = DiffusionSchedule::new(
        Digest::of_bytes("scheduler", b"linear-flow-v1"),
        vec![1.0, 0.5, 0.0],
    )?;
    let mut components = BTreeMap::new();
    components.insert("transformer".to_owned(), Digest::of_bytes("artifact", b"model"));
    let plan = DiffusionPlan::new(
        components,
        Digest::of_bytes("conditioning", b"exact-tokenization-output"),
        Digest::of_bytes("rng", b"cpu-rng-v1"),
        7,
        tensor.clone(),
        schedule,
    )?;
    let context = StepContext::for_plan(&plan, 0)?;
    let stage = ChannelBias::new(&tensor, 2, 0, 0.05, 0.1)?;
    let mut pipeline = Pipeline::new(plan.digest()?, tensor, vec![Box::new(stage)])?;
    pipeline.begin()?;

    let mut state = vec![0.0; 12];
    pipeline.apply(&context, &mut state)?;
    assert_eq!(pipeline.receipt().invocations, 1);
    Ok(())
}
```

Receipts establish ordering, bounds, identities, and state lineage. They do
not establish that an intervention improves an image or model behavior.

// SPDX-License-Identifier: MIT OR Apache-2.0

#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod contract;
mod error;
mod execution;
mod execution_v2;
mod execution_v3;
mod observer;
mod pipeline;

pub use contract::{
    DiffusionCheckpointReceipt, DiffusionPlan, DiffusionSchedule, InterventionFailure,
    InterventionSpec, MAX_COMPONENTS, MAX_DEVICE_LABEL_BYTES, MAX_DIFFUSION_STEPS,
    MAX_INTERVENTION_STAGES, MAX_TENSOR_DIMENSIONS, MAX_TENSOR_ELEMENTS, PipelineReceipt,
    PipelineSpec, StageReceipt, StepContext, TensorDType, TensorLayout, TensorSpec,
};
pub use error::{Error, Result};
pub use execution::{
    ImageBufferBinding, ImageBufferLayout, ImageBufferRole, ImageExecutionPlan,
    ImageExecutionReceipt, ImageOperation, ImageOutputFormat, ImageTerminal, LoraStackEntry,
    MAX_IMAGE_BUFFERS, MAX_IMAGE_DIMENSION, MAX_IMAGE_LORAS, MAX_IMAGE_OBSERVATIONS,
    MAX_IMAGE_OPERATORS, MAX_OPERATOR_CONTROL_BYTES, MAX_SELECTOR_LABEL_BYTES, ObservationKind,
    ObservationRequest, OperatorInvocation, ScalePoint, ScaleSchedule, SeedSelection, StepSelector,
    TensorSelector,
};
pub use execution_v2::{
    ImageCheckpointPlan, ImageCleanupDisposition, ImageCleanupPolicy, ImageCompositeOperation,
    ImageCompositeReceipt, ImageCompositeStage, ImageExecutionPlanV2, ImageExecutionReceiptV2,
    ImageOutputReceiptV2, ImageOutputRoute, ImageOutputSource, ImageValueSource,
    MAX_IMAGE_COMPOSITE_STAGES, MAX_IMAGE_GRAPH_SCRATCH_BYTES, mask_blend_rgb8,
};
pub use execution_v3::{ImageExecutionPlanV3, ImageExecutionReceiptV3, ImageOutputReceiptV3};
pub use observer::{ObserverReceipt, ObserverSet, StepObserver};
pub use pipeline::{ChannelBias, Intervention, Pipeline};

pub use logit_loom_core::{ControlFlow, Digest};
pub use logit_loom_executor::BufferSpec;

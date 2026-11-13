// SPDX-License-Identifier: MIT OR Apache-2.0

#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod contract;
mod error;
mod execution;
mod execution_v2;
mod execution_v3;
mod flow_schedule;
mod image_program;
mod krea_activation;
mod krea_schedule;
mod observer;
mod pipeline;
mod projected_component;

pub use contract::{
    DiffusionCheckpointReceipt, DiffusionPlan, DiffusionSchedule, InterventionFailure,
    InterventionSpec, MAX_DIFFUSION_STEPS, MAX_TENSOR_DIMENSIONS, PipelineReceipt, PipelineSpec,
    StageReceipt, StepContext, TensorDType, TensorLayout, TensorSpec,
};
pub use error::{Error, Result};
pub use execution::{
    IMAGE_EXTENT_PRODUCT_REFUSAL, ImageBufferBinding, ImageBufferLayout, ImageBufferRole,
    ImageExecutionPlan, ImageExecutionReceipt, ImageOperation, ImageOutputFormat, ImageTerminal,
    LoraStackEntry, MAX_IMAGE_DIMENSION, ObservationKind, ObservationRequest, OperatorInvocation,
    ScalePoint, ScaleSchedule, SeedSelection, StepSelector, TensorSelector,
    image_extent_fits_native_int,
};
pub use execution_v2::{
    ImageCheckpointPlan, ImageCleanupDisposition, ImageCleanupPolicy, ImageCompositeOperation,
    ImageCompositeReceipt, ImageCompositeStage, ImageExecutionPlanV2, ImageExecutionReceiptV2,
    ImageOutputReceiptV2, ImageOutputRoute, ImageOutputSource, ImageValueSource, mask_blend_rgb8,
};
pub use execution_v3::{ImageExecutionPlanV3, ImageExecutionReceiptV3, ImageOutputReceiptV3};
pub use flow_schedule::{FlowMatchScheduleV1, FlowScheduleFamilyV1};
pub use image_program::{
    ImageOpaqueValueKindV1, ImagePngColorV1, ImageProgramCleanupDispositionV1,
    ImageProgramInputBindingV1, ImageProgramInputV1, ImageProgramLivenessV1, ImageProgramLoraV1,
    ImageProgramMeasurementsV1, ImageProgramNativeOutputRoleV1, ImageProgramNativeOutputV1,
    ImageProgramNativeStageV1, ImageProgramOutputReceiptV1, ImageProgramOutputRouteV1,
    ImageProgramOutputSourceV1, ImageProgramPlanV1, ImageProgramReceiptV1, ImageProgramReleaseV1,
    ImageProgramStageOperationV1, ImageProgramStageReceiptV1, ImageProgramStageV1,
    ImageProgramTerminalV1, ImageProgramValueMeasurementV1, ImageProgramValuePlacementV1,
    ImageProgramValueReceiptV1, ImageProgramValueSpecV1, ImageProgramValueV1,
    image_program_value_content,
};
pub use krea_activation::*;
pub use krea_schedule::{
    KREA_2_TURBO_DEFAULT_FLOW_SHIFT, KREA_2_TURBO_DEFAULT_STEPS, KreaDiscreteFlowScheduleV1,
};
pub use observer::{ObserverReceipt, ObserverSet, StepObserver};
pub use pipeline::{ChannelBias, Intervention, Pipeline};
pub use projected_component::*;

pub use logit_loom_core::{ActivationStatisticsV1, ControlFlow, Digest};
pub use logit_loom_executor::BufferSpec;

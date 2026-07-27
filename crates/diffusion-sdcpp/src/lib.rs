// SPDX-License-Identifier: MIT OR Apache-2.0

#![doc = include_str!("../README.md")]
#![allow(unsafe_code)]

mod contract;
mod error;
mod execution;
mod ffi;
mod image;
mod program;
mod runtime;

pub use contract::{
    BoundaryControl, BoundaryReceipt, CompanionReceipt, ContinueControl,
    ControlledGenerationOutput, ControlledGenerationReceipt, DiffusionCheckpoint,
    GenerationMeasurements, GenerationOutput, GenerationReceipt, ImageExecutionBindings,
    ImageOutputSink, ImageRequest, ImageRequestReceipt, MAX_BACKEND_LABEL_BYTES,
    MAX_CHECKPOINT_ENVELOPE_BYTES, MAX_CHECKPOINT_RECEIPT_BYTES, MAX_CHECKPOINT_STATE_BYTES,
    MAX_IMAGE_DIMENSION, MAX_PROMPT_BYTES, NativeRuntimeReceipt, NoopProgram, Profile,
    ProfileArtifacts, ProfileReceipt, SdcppOptions, StepProgram, StepReceipt,
};
pub use error::{Error, Result};
pub use execution::{
    ArtifactPathResolver, ChannelBiasControlV1, ImagePlanExecutor, RejectArtifactPaths,
    channel_bias_schema_v1, lora_target_v1,
};
pub use image::{
    AdvancedGenerationOutput, AdvancedGenerationReceipt, AdvancedImageRequest,
    AdvancedImageRequestReceipt, AdvancedProgramGenerationOutput, AdvancedProgramGenerationReceipt,
    ImagePixels, LoraBinding, LoraBindingReceipt, MAX_REFERENCE_IMAGES, MAX_REQUEST_LORAS,
    MAX_VAE_TENSOR_RANK, PixelReceipt, VaeImageOutput, VaeOperationReceipt, VaeTensor,
    VaeTensorOutput,
};
pub use program::{ForkProgram, PipelineProgram};
pub use runtime::{Sdcpp, probe_companion};

/// Companion ABI version implemented by this adapter.
pub const COMPANION_ABI_VERSION: u32 = 1;
/// Version of the whole-image extension layered over companion ABI v1.
pub const IMAGE_ABI_VERSION: u32 = 2;
/// Version of the safe Rust execution contract layered over the companion.
pub const ADAPTER_CONTRACT_VERSION: u32 = 4;
/// Exact stable-diffusion.cpp revision required by this adapter.
pub const UPSTREAM_COMMIT: &str = "ea4e566ccffa10f853ecc3f29e74b1820bc91beb";

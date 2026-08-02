// SPDX-License-Identifier: MIT OR Apache-2.0

#![doc = include_str!("../README.md")]
#![allow(unsafe_code)]

mod application;
mod contract;
mod error;
mod execution;
mod ffi;
mod image;
mod krea_activation;
mod native_program;
mod program;
mod resident;
mod runtime;

pub use application::{ModelBlockApplicationReceiptV1, ModelBlockApplicationV1};
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
    ArtifactPathResolver, ChannelBiasControlV1, ImagePlanExecutor,
    ModelBlockResidualScaleControlV1, RejectArtifactPaths, channel_bias_schema_v1, lora_target_v1,
    model_block_residual_scale_schema_v1,
};
pub use image::{
    AdvancedGenerationOutput, AdvancedGenerationReceipt, AdvancedImageRequest,
    AdvancedImageRequestReceipt, AdvancedProgramGenerationOutput, AdvancedProgramGenerationReceipt,
    ImagePixels, LoraBinding, LoraBindingReceipt, MAX_REFERENCE_IMAGES, MAX_REQUEST_LORAS,
    MAX_VAE_TENSOR_RANK, PixelReceipt, VaeImageOutput, VaeOperationReceipt, VaeTensor,
    VaeTensorOutput,
};
pub use krea_activation::{KreaActivationExecutionV1, KreaActivationInputBuffer};
pub use native_program::{
    RejectResidentArtifactPaths, ResidentArtifactPathResolver, SdcppResidentProgram,
    resident_checkpoint_compatibility_v1, resident_checkpoint_conversion_v1,
    resident_lora_target_v1, resident_png_encoding_v1, resident_png_maximum_bytes_v1,
};
pub use program::{ForkProgram, PipelineProgram};
pub use resident::{
    ResidentImageProgramBackend, ResidentImageProgramDriver, ResidentImageProgramExecution,
    ResidentProgramCompletedStage, ResidentProgramFinish, ResidentProgramStageTerminal,
};
pub use runtime::{Sdcpp, probe_companion};

/// Companion ABI version implemented by this adapter.
pub const COMPANION_ABI_VERSION: u32 = 2;
/// Version of the whole-image extension layered over companion ABI v2.
pub const IMAGE_ABI_VERSION: u32 = 3;
/// Version of the resident native value-arena extension.
pub const PROGRAM_ABI_VERSION: u32 = 3;
/// Version of the native model-block operator and application-attestation
/// extension.
pub const MODEL_BLOCK_ABI_VERSION: u32 = 5;
/// Version of the native Krea activation topology, capture, resident-input,
/// operator, and cleanup extension.
pub const KREA_ACTIVATION_ABI_VERSION: u32 = 6;
/// Version of the safe Rust execution contract layered over the companion.
pub const ADAPTER_CONTRACT_VERSION: u32 = 9;
/// Exact stable-diffusion.cpp revision required by this adapter.
pub const UPSTREAM_COMMIT: &str = "ea4e566ccffa10f853ecc3f29e74b1820bc91beb";

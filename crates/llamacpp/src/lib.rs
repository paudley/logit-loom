// SPDX-License-Identifier: MIT OR Apache-2.0

#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod activation;
mod error;
mod mechanics;
mod model;
mod sampler;
mod session;
mod speculation;
mod steering;
mod structured;

pub use activation::{
    ActivationCaptureOutput, ActivationConfiguration, ActivationProgramOutput,
    LlamaCppTensorProfile,
};
pub use error::Error;
pub use mechanics::{
    OrdinaryTextMechanicsSnapshot, TextMechanicsCheckpoint, TextMechanicsExecutionOutput,
    TextMechanicsOutput, TextMechanicsRequest, TextMechanicsResume, execute_text_mechanics,
};
pub use model::{
    AuthorizedModelArtifact, DevicePolicy, MAX_TOKENIZATION_BYTES, Model, ModelOptions,
    PreverifiedModelArtifact, Runtime, Tokenization,
};
pub use session::{GenerationOutput, PrefillOutput, Session, SessionOptions, StateSnapshot};
pub use speculation::{
    SpeculativeActivationOutput, SpeculativeActivations, SpeculativeCheckpointOutput,
    SpeculativeCheckpointRequest, SpeculativeContinuationRequest, SpeculativeGenerationOutput,
    SpeculativeRequest, SpeculativeSessionOptions, SpeculativeStateSnapshot, generate_speculative,
    generate_speculative_checkpointed, resume_speculative, resume_speculative_checkpointed,
    speculation_implementation_identity,
};
pub use steering::{
    AuthorizedLoraArtifact, ControlVector, ControlVectorScope, LoraAdapter, LoraApplication,
    LoraScope, SteeringScope,
};
pub use structured::{
    ByteFeedback, ByteFeedbackCandidate, ByteFeedbackError, InstalledByteFeedback,
    InstalledStructuredValidator, MAX_STRUCTURED_TOKENIZER_TABLE_BYTES,
    MAX_STRUCTURED_VALIDATION_EVIDENCE_BYTES, StructuredAttemptOutput,
    StructuredProjectionController, StructuredValidationContext, StructuredValidationOutcome,
    StructuredValidator, StructuredValidatorError,
};

/// Exact native Rust binding version used by this adapter release.
pub const LLAMA_CPP_BINDING_VERSION: &str = "llama-cpp-4:0.5.0+logit-loom-adr0002";
/// Exact source revision of the reviewed native Rust binding successor.
pub const LLAMA_CPP_BINDING_SOURCE_REVISION: &str = "f1c5dd05906a11aee5c2eaf1265851bf29752d67";
/// Exact llama.cpp revision carried by the ADR 0002 binding successor.
pub const LLAMA_CPP_REVISION: &str = "221f0f6356efe2260023208365705ec5d5a7c8f5";

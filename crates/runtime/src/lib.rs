// SPDX-License-Identifier: MIT OR Apache-2.0

#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod error;
mod observer;
mod pipeline;
mod request;
mod runtime;
mod session;
mod steering;

pub use error::{Error, Result};
pub use observer::ObserversBuilder;
pub use pipeline::PipelineBuilder;
pub use request::GenerationRequest;
pub use runtime::{Loom, LoomOptions, NativeLogPolicy};
pub use session::{AdmissionOutput, CompletionOutput, LoomSession};
pub use steering::{ControlVectorSession, LoraSession};

pub use logit_loom::*;
pub use logit_loom_llamacpp::{
    ActivationCaptureOutput, ActivationConfiguration, ActivationProgramOutput, ControlVector,
    DevicePolicy, GenerationOutput, LLAMA_CPP_BINDING_VERSION, LlamaCppTensorProfile, LoraAdapter,
    MAX_TOKENIZATION_BYTES, ModelOptions, PrefillOutput, SessionOptions,
    SpeculativeActivationOutput, SpeculativeActivations, SpeculativeGenerationOutput,
    SpeculativeRequest, SpeculativeSessionOptions, StateSnapshot, Tokenization,
    generate_speculative, speculation_implementation_identity,
};
pub use logit_loom_models::ArtifactReceipt as ModelArtifactReceipt;

/// Direct access to the lower-level crates when the façade is intentionally
/// insufficient.
pub mod low_level {
    pub use logit_loom as mechanics;
    pub use logit_loom_llamacpp as llamacpp;
}

// SPDX-License-Identifier: MIT OR Apache-2.0

#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod activation;
mod error;
mod model;
mod sampler;
mod session;
mod speculation;
mod steering;

pub use activation::{
    ActivationCaptureOutput, ActivationConfiguration, ActivationProgramOutput,
    LlamaCppTensorProfile,
};
pub use error::Error;
pub use model::{DevicePolicy, MAX_TOKENIZATION_BYTES, Model, ModelOptions, Runtime, Tokenization};
pub use session::{GenerationOutput, PrefillOutput, Session, SessionOptions, StateSnapshot};
pub use speculation::{
    SpeculativeActivationOutput, SpeculativeActivations, SpeculativeGenerationOutput,
    SpeculativeRequest, SpeculativeSessionOptions, generate_speculative,
    speculation_implementation_identity,
};
pub use steering::{ControlVector, ControlVectorScope, LoraAdapter, LoraScope};

/// Exact native Rust binding version used by this adapter release.
pub const LLAMA_CPP_BINDING_VERSION: &str = "llama-cpp-4:0.4.2+logit-loom-adr0002";
/// Exact source revision of the reviewed native Rust binding successor.
pub const LLAMA_CPP_BINDING_SOURCE_REVISION: &str = "d76356b9725a3736212b3bfd16c66fc80c995c29";
/// Exact llama.cpp revision carried by the ADR 0002 binding successor.
pub const LLAMA_CPP_REVISION: &str = "f87067841bac583bc089a225382248d857791ca8";

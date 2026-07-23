// SPDX-License-Identifier: MIT OR Apache-2.0

#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod error;
mod model;
mod sampler;
mod session;
mod steering;

pub use error::Error;
pub use model::{DevicePolicy, MAX_TOKENIZATION_BYTES, Model, ModelOptions, Runtime, Tokenization};
pub use session::{GenerationOutput, PrefillOutput, Session, SessionOptions, StateSnapshot};
pub use steering::{ControlVector, ControlVectorScope, LoraAdapter, LoraScope};

/// Exact native Rust binding version used by this adapter release.
pub const LLAMA_CPP_BINDING_VERSION: &str = "llama-cpp-4:0.4.2";

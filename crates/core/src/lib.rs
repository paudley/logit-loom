// SPDX-License-Identifier: MIT OR Apache-2.0

#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod digest;
mod error;
mod generation;
mod observe;
mod sampling;
mod state;
mod steering;
mod token;
mod transform;

pub use digest::Digest;
pub use error::CoreError;
pub use generation::{GenerationFinish, GenerationReceipt};
pub use observe::{ControlFlow, ObserverReceipt, PrefillFinish, PrefillProgress, PrefillReceipt};
pub use sampling::{
    DrySampler, GenerationPlan, Grammar, LogitBias, MAX_DRY_SEQUENCE_BREAKER_BYTES,
    MAX_DRY_SEQUENCE_BREAKERS, MAX_GRAMMAR_ROOT_BYTES, MAX_GRAMMAR_SOURCE_BYTES, MAX_LOGIT_BIASES,
    MAX_STOP_SEQUENCE_BYTES, MAX_STOP_SEQUENCES, MirostatSampler, MirostatVersion,
    RepetitionSampler, SamplingPlan,
};
pub use state::CheckpointReceipt;
pub use steering::{ControlVectorSpec, LoraSpec, SteeringAction, SteeringKind, SteeringReceipt};
pub use token::{CandidateMode, MAX_SPARSE_CANDIDATES, TokenId};
pub use transform::{
    CallbackFailure, CallbackPhase, MAX_PIPELINE_STAGES, MAX_RETAINED_FAILURE_BYTES,
    PipelineReceipt, PipelineSpec, StageReceipt, TransformSpec,
};

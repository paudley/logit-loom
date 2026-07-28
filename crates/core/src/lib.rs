// SPDX-License-Identifier: MIT OR Apache-2.0

#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod activation;
mod activation_accumulator;
mod audio;
mod digest;
mod error;
mod generation;
mod observe;
mod sampling;
mod speculation;
mod state;
mod steering;
mod text_mechanics;
mod token;
mod transform;

pub use activation::*;
pub use activation_accumulator::*;
pub use audio::{
    AudioPrefillPlanV1, AudioPrefillReceiptV1, MAX_AUDIO_FRAMES, MAX_AUDIO_SAMPLE_RATE_HZ,
};
pub use digest::Digest;
pub use error::CoreError;
pub use generation::{GenerationFinish, GenerationReceipt};
pub use observe::{ControlFlow, ObserverReceipt, PrefillFinish, PrefillProgress, PrefillReceipt};
pub use sampling::{
    DrySampler, GenerationPlan, GenerationPlanV2, Grammar, GrammarActivationV2, GrammarPlanV2,
    LogitBias, MAX_DRY_SEQUENCE_BREAKER_BYTES, MAX_DRY_SEQUENCE_BREAKERS, MAX_GRAMMAR_ROOT_BYTES,
    MAX_GRAMMAR_SOURCE_BYTES, MAX_LAZY_GRAMMAR_TRIGGER_BYTES,
    MAX_LAZY_GRAMMAR_TRIGGER_PATTERN_BYTES, MAX_LAZY_GRAMMAR_TRIGGER_PATTERNS,
    MAX_LAZY_GRAMMAR_TRIGGER_TOKENS, MAX_LOGIT_BIASES, MAX_STOP_SEQUENCE_BYTES, MAX_STOP_SEQUENCES,
    MirostatSampler, MirostatVersion, RepetitionSampler, SamplingPlan,
};
pub use speculation::*;
pub use state::CheckpointReceipt;
pub use steering::{ControlVectorSpec, LoraSpec, SteeringAction, SteeringKind, SteeringReceipt};
pub use text_mechanics::{
    MAX_SPECULATIVE_TOKENS, MAX_TEXT_MECHANICS_CAPTURES, MAX_TEXT_MECHANICS_LORAS,
    TextMechanicsCleanupReceiptV2, TextMechanicsPlanV1, TextMechanicsPlanV2,
    TextMechanicsReceiptV1, TextMechanicsReceiptV2,
};
pub use token::{CandidateMode, MAX_SPARSE_CANDIDATES, TokenId};
pub use transform::{
    CallbackFailure, CallbackPhase, MAX_PIPELINE_STAGES, MAX_RETAINED_FAILURE_BYTES,
    PipelineReceipt, PipelineSpec, StageReceipt, TransformSpec,
};

// SPDX-License-Identifier: MIT OR Apache-2.0

//! Runtime callback and pipeline failures.

use logit_loom_core::{CallbackFailure, CoreError};
use thiserror::Error;

/// Error returned by a user-defined logit transform.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[error("{message}")]
pub struct TransformError {
    message: String,
}

impl TransformError {
    /// Creates an implementation error.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Error returned by a generated-token observer.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[error("{message}")]
pub struct ObserverError {
    message: String,
}

impl ObserverError {
    /// Creates an observer error.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Error returned by a prefill observer.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[error("{message}")]
pub struct PrefillObserverError {
    message: String,
}

impl PrefillObserverError {
    /// Creates a prefill observer error.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Failure while validating or executing an ordered pipeline.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum PipelineError {
    /// A public contract was invalid.
    #[error(transparent)]
    Contract(#[from] CoreError),
    /// A pipeline was used before a successful [`Pipeline::begin`](crate::Pipeline::begin).
    #[error("pipeline has not begun")]
    NotBegun,
    /// A failed pipeline was reused without beginning a fresh call.
    #[error("pipeline is failed; call begin before reuse")]
    Failed,
    /// The caller supplied a sampling step out of sequence.
    #[error("invalid transform step {actual}; expected {expected}")]
    InvalidStep {
        /// Next zero-based step expected by the pipeline.
        expected: u32,
        /// Step supplied by the caller.
        actual: u32,
    },
    /// A token was reported without a successful unmatched transform step.
    #[error("token admission has no unmatched successful transform invocation")]
    UnexpectedAdmission,
    /// A new step began before the prior selected token was reported.
    #[error("previous transform invocation is still awaiting causal admission")]
    AdmissionPending,
    /// Candidate arrays disagreed or exceeded a supported bound.
    #[error("invalid candidate view: {0}")]
    InvalidCandidates(String),
    /// Exact mechanical accounting exceeded its integer representation.
    #[error("pipeline accounting overflowed: {0}")]
    AccountingOverflow(&'static str),
    /// One user callback returned an error or panicked.
    #[error("transform stage {stage} failed: {failure:?}")]
    Callback {
        /// Zero-based stage index.
        stage: usize,
        /// Contained failure.
        failure: CallbackFailure,
    },
}

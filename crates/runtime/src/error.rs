// SPDX-License-Identifier: MIT OR Apache-2.0

//! Unified failures from façade construction and delegated execution.

use logit_loom::{CoreError, ObserverError, PipelineError, PrefillObserverError, TransformError};
use logit_loom_llamacpp::Error as AdapterError;
use thiserror::Error;

/// Result type returned by `logit-loom-runtime`.
pub type Result<T> = std::result::Result<T, Error>;

/// Failures from convenience-layer validation or delegated mechanics.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum Error {
    /// The llama.cpp adapter rejected an operation.
    #[error(transparent)]
    Adapter(#[from] AdapterError),
    /// A backend-neutral contract was invalid.
    #[error(transparent)]
    Contract(#[from] CoreError),
    /// A transform constructor rejected its configuration.
    #[error(transparent)]
    Transform(#[from] TransformError),
    /// A transform pipeline was invalid.
    #[error(transparent)]
    Pipeline(#[from] PipelineError),
    /// A generated-token observer set was invalid.
    #[error(transparent)]
    Observer(#[from] ObserverError),
    /// A controlled-prefill observer was invalid.
    #[error(transparent)]
    PrefillObserver(#[from] PrefillObserverError),
}

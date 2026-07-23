// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed adapter failures.

use logit_loom::{CoreError, ObserverError, PipelineError, PrefillObserverError};
use thiserror::Error;

/// Failures from native execution, callback boundaries, or identity checks.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum Error {
    /// llama.cpp or its safe binding returned an error.
    #[error("llama.cpp error: {0}")]
    Native(String),
    /// A backend-neutral public contract was invalid.
    #[error(transparent)]
    Contract(#[from] CoreError),
    /// A Logit Loom transform pipeline failed.
    #[error(transparent)]
    Pipeline(#[from] PipelineError),
    /// A generated-token observer failed.
    #[error(transparent)]
    Observer(#[from] ObserverError),
    /// A controlled-prefill observer failed.
    #[error(transparent)]
    PrefillObserver(#[from] PrefillObserverError),
    /// A caller supplied an invalid bounded argument.
    #[error("invalid argument: {0}")]
    Invalid(String),
    /// An artifact or checkpoint belongs to another execution identity.
    #[error("incompatible artifact: {0}")]
    Incompatible(String),
    /// Native state is uncertain after a failed reversible operation.
    #[error("session is poisoned: {0}")]
    Poisoned(String),
    /// A local artifact could not be read.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

pub(crate) fn native(error: impl std::fmt::Display) -> Error {
    Error::Native(error.to_string())
}

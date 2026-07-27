// SPDX-License-Identifier: MIT OR Apache-2.0

//! Diffusion contract and callback failures.

use logit_loom_core::CoreError;
use thiserror::Error;

/// Result returned by backend-neutral diffusion mechanics.
pub type Result<T> = std::result::Result<T, Error>;

/// Failure before native state can be committed.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// A serializable contract was invalid.
    #[error(transparent)]
    Contract(#[from] CoreError),
    /// The supplied state does not match the declared tensor contract.
    #[error("diffusion state is incompatible: {0}")]
    Incompatible(String),
    /// An ordered intervention stage returned an error.
    #[error("diffusion intervention stage {stage} failed: {message}")]
    Intervention {
        /// Zero-based stage index.
        stage: usize,
        /// Bounded callback detail.
        message: String,
    },
    /// A post-step observer returned an error.
    #[error("diffusion observer {observer} failed: {message}")]
    Observer {
        /// Zero-based observer index.
        observer: usize,
        /// Bounded callback detail.
        message: String,
    },
}

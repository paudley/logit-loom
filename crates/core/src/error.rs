// SPDX-License-Identifier: MIT OR Apache-2.0

//! Validation and deterministic-encoding failures.

use thiserror::Error;

/// Failures raised while constructing or validating public contracts.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum CoreError {
    /// A required field was empty.
    #[error("{field} must not be empty")]
    Empty {
        /// Stable field name.
        field: &'static str,
    },
    /// A floating-point field was NaN or infinite.
    #[error("{field} must be finite")]
    NonFinite {
        /// Stable field name.
        field: &'static str,
    },
    /// A field violated a bounded structural contract.
    #[error("{field} is invalid: {reason}")]
    Invalid {
        /// Stable field name.
        field: &'static str,
        /// Bounded explanation.
        reason: String,
    },
    /// Deterministic JSON encoding failed.
    #[error("unable to encode contract: {0}")]
    Serialization(#[from] serde_json::Error),
}

impl CoreError {
    /// Creates a structural validation error.
    pub fn invalid(field: &'static str, reason: impl Into<String>) -> Self {
        Self::Invalid {
            field,
            reason: reason.into(),
        }
    }
}

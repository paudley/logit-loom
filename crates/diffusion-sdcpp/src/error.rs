// SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::PathBuf;

use thiserror::Error;

/// Result returned by the stable-diffusion.cpp adapter.
pub type Result<T> = std::result::Result<T, Error>;

/// Safe adapter failure.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// A public input violated a documented bound or invariant.
    #[error("invalid stable-diffusion.cpp input: {0}")]
    Invalid(String),
    /// The dynamic library could not be opened or resolved.
    #[error("failed to load stable-diffusion.cpp companion {path}: {message}")]
    Library {
        /// Caller-selected library path.
        path: PathBuf,
        /// Loader detail.
        message: String,
    },
    /// The companion ABI or upstream revision differs.
    #[error("incompatible stable-diffusion.cpp companion: {0}")]
    Incompatible(String),
    /// A model catalog or artifact check failed.
    #[error(transparent)]
    Catalog(#[from] logit_loom_models::CatalogError),
    /// A local model artifact check failed.
    #[error(transparent)]
    Artifact(#[from] logit_loom_models::ArtifactError),
    /// A backend-neutral diffusion contract failed.
    #[error(transparent)]
    Diffusion(#[from] logit_loom_diffusion::Error),
    /// File inspection or hashing failed.
    #[error("failed to inspect {path}: {source}")]
    Io {
        /// Caller-selected path.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// Native context construction or generation failed.
    #[error("stable-diffusion.cpp native failure: {0}")]
    Native(String),
    /// Cooperative cancellation reached the documented post-step boundary.
    #[error("stable-diffusion.cpp execution cancelled")]
    Cancelled,
    /// Native state or cleanup is uncertain and this session cannot be reused.
    #[error("stable-diffusion.cpp session is poisoned: {0}")]
    Poisoned(String),
    /// A contained step-program callback failed.
    #[error("diffusion step program failed: {0}")]
    Callback(String),
    /// A caller-owned image sink rejected or failed to retain output bytes.
    #[error("diffusion image output failed: {0}")]
    Output(String),
}

impl logit_loom_executor::ClassifiedExecutionError for Error {
    fn disposition(&self) -> logit_loom_executor::FailureDisposition {
        use logit_loom_executor::FailureDisposition;

        match self {
            Self::Cancelled => FailureDisposition::Cancelled,
            Self::Native(_) | Self::Poisoned(_) | Self::Callback(_) => FailureDisposition::Poisoned,
            Self::Invalid(_)
            | Self::Library { .. }
            | Self::Incompatible(_)
            | Self::Catalog(_)
            | Self::Artifact(_)
            | Self::Diffusion(_)
            | Self::Io { .. }
            | Self::Output(_) => FailureDisposition::Rejected,
        }
    }
}

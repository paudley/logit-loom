// SPDX-License-Identifier: MIT OR Apache-2.0

//! Content-bound checkpoint accounting.

use serde::{Deserialize, Serialize};

use crate::{CoreError, Digest};

/// Serializable compatibility metadata for one backend-owned causal checkpoint.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointReceipt {
    /// Exact model artifact identity.
    pub model: Digest,
    /// Backend and compatibility identity.
    pub backend: Digest,
    /// Digest of the opaque state bytes.
    pub state: Digest,
    /// Number of opaque state bytes.
    pub state_bytes: u64,
    /// Digest of the exact causal token history.
    pub token_history: Digest,
    /// Causal token count and next native position.
    pub position: u64,
}

impl CheckpointReceipt {
    /// Validates backend-independent checkpoint accounting.
    ///
    /// # Errors
    ///
    /// Returns an error when no opaque state bytes were recorded.
    pub fn validate(&self) -> Result<(), CoreError> {
        if self.state_bytes == 0 {
            return Err(CoreError::invalid(
                "checkpoint receipt",
                "opaque state byte count must be greater than zero",
            ));
        }
        Ok(())
    }

    /// Returns a content identity for this checkpoint metadata.
    ///
    /// # Errors
    ///
    /// Returns a validation or serialization error.
    pub fn digest(&self) -> Result<Digest, CoreError> {
        self.validate()?;
        Digest::of_serializable("checkpoint-receipt-v2", self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_receipts_require_native_state_bytes() {
        let identity = Digest::of_bytes("test-checkpoint", b"one");
        let receipt = CheckpointReceipt {
            model: identity.clone(),
            backend: identity.clone(),
            state: identity.clone(),
            state_bytes: 0,
            token_history: identity,
            position: 0,
        };
        assert!(receipt.digest().is_err());
    }

    #[test]
    fn checkpoint_receipts_use_the_v2_digest_domain() {
        let identity = Digest::of_bytes("test-checkpoint", b"two");
        let receipt = CheckpointReceipt {
            model: identity.clone(),
            backend: identity.clone(),
            state: identity.clone(),
            state_bytes: 1,
            token_history: identity,
            position: 0,
        };
        assert_eq!(
            receipt.digest().unwrap(),
            Digest::of_serializable("checkpoint-receipt-v2", &receipt).unwrap()
        );
    }
}

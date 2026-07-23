// SPDX-License-Identifier: MIT OR Apache-2.0

//! Generation result accounting.

use serde::{Deserialize, Serialize};

use crate::{CoreError, Digest, TokenId};

/// Why one bounded generation call stopped.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum GenerationFinish {
    /// The model selected an end-of-generation token.
    EndOfGeneration {
        /// Selected terminal token, which was not admitted to causal state.
        token: TokenId,
    },
    /// The caller's token limit was reached.
    TokenLimit,
    /// An exact byte stop suffix was causally generated.
    StopSequence {
        /// Zero-based stop index in the generation plan.
        index: u32,
    },
    /// A Rust observer requested a cooperative stop.
    ObserverStop,
}

/// Mechanical receipt for one bounded generation call.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationReceipt {
    /// Exact generation-plan identity.
    pub plan: Digest,
    /// Position before generation.
    pub initial_position: u64,
    /// Causally admitted generated tokens.
    pub admitted_tokens: u32,
    /// Exact generated token-byte count.
    pub admitted_bytes: u64,
    /// Final causal position.
    pub final_position: u64,
    /// Terminal disposition.
    pub finish: GenerationFinish,
    /// Optional transform-pipeline receipt identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transform_receipt: Option<Digest>,
    /// Ordered observer receipt identities.
    #[serde(default)]
    pub observer_receipts: Vec<Digest>,
}

impl GenerationReceipt {
    /// Validates position progression and returns a content identity.
    ///
    /// # Errors
    ///
    /// Returns an error for inconsistent accounting or failed serialization.
    pub fn digest(&self) -> Result<Digest, CoreError> {
        let expected_position = self
            .initial_position
            .checked_add(u64::from(self.admitted_tokens))
            .ok_or_else(|| {
                CoreError::invalid("generation receipt", "causal position overflowed")
            })?;
        if self.final_position != expected_position {
            return Err(CoreError::invalid(
                "generation receipt",
                "position does not match admitted tokens",
            ));
        }
        Digest::of_serializable("generation-receipt-v1", self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_receipts_reject_position_overflow() {
        let receipt = GenerationReceipt {
            plan: Digest::of_bytes("test-plan", b"one"),
            initial_position: u64::MAX,
            admitted_tokens: 1,
            admitted_bytes: 1,
            final_position: u64::MAX,
            finish: GenerationFinish::TokenLimit,
            transform_receipt: None,
            observer_receipts: Vec::new(),
        };
        assert!(receipt.digest().is_err());
    }
}

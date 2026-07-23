// SPDX-License-Identifier: MIT OR Apache-2.0

//! Token and candidate-exposure contracts.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize};

use crate::CoreError;

/// Maximum sparse candidates exposed to one transform call.
pub const MAX_SPARSE_CANDIDATES: u32 = 65_536;

/// A non-negative model-tokenizer identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct TokenId(i32);

impl TokenId {
    /// Creates a token identifier.
    ///
    /// # Errors
    ///
    /// Returns an error for a negative identifier.
    pub fn new(raw: i32) -> Result<Self, CoreError> {
        if raw < 0 {
            return Err(CoreError::invalid(
                "token id",
                "token identifiers must be non-negative",
            ));
        }
        Ok(Self(raw))
    }

    /// Returns the backend-native integer representation.
    pub const fn get(self) -> i32 {
        self.0
    }
}

impl TryFrom<i32> for TokenId {
    type Error = CoreError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<TokenId> for i32 {
    fn from(value: TokenId) -> Self {
        value.get()
    }
}

impl<'de> Deserialize<'de> for TokenId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = i32::deserialize(deserializer)?;
        Self::new(raw).map_err(serde::de::Error::custom)
    }
}

impl fmt::Display for TokenId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

/// Candidate-logit exposure presented to every stage in a pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum CandidateMode {
    /// Expose every vocabulary token in tokenizer order.
    FullVocabulary,
    /// Expose at most the highest-ranked `limit` finite raw logits.
    Sparse {
        /// Maximum candidates copied into the transform boundary.
        limit: u32,
    },
}

impl CandidateMode {
    /// Validates sparse bounds.
    ///
    /// # Errors
    ///
    /// Returns an error for a zero or excessive sparse limit.
    pub fn validate(self) -> Result<(), CoreError> {
        if let Self::Sparse { limit } = self
            && !(1..=MAX_SPARSE_CANDIDATES).contains(&limit)
        {
            return Err(CoreError::invalid(
                "candidate mode",
                format!("sparse limit must be in 1..={MAX_SPARSE_CANDIDATES}"),
            ));
        }
        Ok(())
    }

    /// Returns the sparse bound, or `None` for full-vocabulary exposure.
    pub const fn sparse_limit(self) -> Option<u32> {
        match self {
            Self::FullVocabulary => None,
            Self::Sparse { limit } => Some(limit),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_identifiers_reject_negative_native_values() {
        assert!(TokenId::new(-1).is_err());
        assert_eq!(TokenId::new(0).unwrap().get(), 0);
        assert!(serde_json::from_str::<TokenId>("-1").is_err());
        assert_eq!(serde_json::from_str::<TokenId>("4").unwrap().get(), 4);
    }

    #[test]
    fn sparse_candidate_bounds_are_enforced() {
        assert!(CandidateMode::Sparse { limit: 0 }.validate().is_err());
        assert!(
            CandidateMode::Sparse {
                limit: MAX_SPARSE_CANDIDATES + 1,
            }
            .validate()
            .is_err()
        );
        assert!(CandidateMode::Sparse { limit: 1 }.validate().is_ok());
    }
}

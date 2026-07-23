// SPDX-License-Identifier: MIT OR Apache-2.0

//! Observer, cancellation, and prefill accounting contracts.

use serde::{Deserialize, Serialize};

use crate::{CallbackFailure, CoreError, Digest};

/// Cooperative action returned at a documented safe execution boundary.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlFlow {
    /// Continue the bounded operation.
    #[default]
    Continue,
    /// Stop after retaining all causal work admitted before this boundary.
    Stop,
}

/// Mechanical accounting for one generated-token observer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObserverReceipt {
    /// Caller-defined observer implementation identity.
    pub implementation: Digest,
    /// Native position before generation began.
    pub initial_position: u64,
    /// Maximum generated-token budget.
    pub requested_tokens: u32,
    /// Pre-sampling polls.
    pub polls: u32,
    /// Causally admitted tokens delivered to the observer.
    pub observed_tokens: u32,
    /// Exact bytes delivered across all token pieces.
    pub observed_bytes: u64,
    /// Final causal position observed by Rust.
    pub final_position: u64,
    /// Whether this observer requested a stop.
    pub stop_requested: bool,
    /// First contained callback failure, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<CallbackFailure>,
}

impl ObserverReceipt {
    /// Returns a content identity for this accounting.
    ///
    /// # Errors
    ///
    /// Returns a validation or serialization error.
    pub fn digest(&self) -> Result<Digest, CoreError> {
        let expected_position = self
            .initial_position
            .checked_add(u64::from(self.observed_tokens))
            .ok_or_else(|| CoreError::invalid("observer receipt", "causal position overflowed"))?;
        if self.requested_tokens == 0
            || self.observed_tokens > self.requested_tokens
            || self.polls < self.observed_tokens
            || self.polls > self.requested_tokens
            || self.final_position != expected_position
        {
            return Err(CoreError::invalid(
                "observer receipt",
                "token and position accounting is inconsistent",
            ));
        }
        Digest::of_serializable("token-observer-receipt-v1", self)
    }
}

/// Progress at one complete text-prefill boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefillProgress {
    /// Position before this prefill operation.
    pub initial_position: u64,
    /// Total tokens requested by the operation.
    pub requested_tokens: u64,
    /// Complete tokens admitted so far.
    pub admitted_tokens: u64,
    /// Complete chunks admitted so far.
    pub admitted_chunks: u32,
    /// Current causal position.
    pub position: u64,
}

impl PrefillProgress {
    /// Validates monotonic token and position accounting.
    ///
    /// # Errors
    ///
    /// Returns an error when progress exceeds the request or disagrees with
    /// the initial position.
    pub fn validate(self) -> Result<(), CoreError> {
        let expected_position = self
            .initial_position
            .checked_add(self.admitted_tokens)
            .ok_or_else(|| CoreError::invalid("prefill progress", "causal position overflowed"))?;
        if self.requested_tokens == 0
            || self.admitted_tokens > self.requested_tokens
            || u64::from(self.admitted_chunks) > self.admitted_tokens
            || self.position != expected_position
        {
            return Err(CoreError::invalid(
                "prefill progress",
                "token and position accounting is inconsistent",
            ));
        }
        Ok(())
    }
}

/// Current or terminal disposition of one controlled prefill operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefillFinish {
    /// The monitor has begun and no terminal disposition is recorded yet.
    InProgress,
    /// Every requested token was admitted.
    Complete,
    /// An observer stopped at a complete chunk boundary.
    Stopped,
}

/// Mechanical accounting for controlled prefill.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefillReceipt {
    /// Observer implementation identity.
    pub implementation: Digest,
    /// Current or final progress.
    pub progress: PrefillProgress,
    /// Current or terminal disposition.
    pub finish: PrefillFinish,
    /// Observer polls.
    pub polls: u32,
    /// Complete chunk callbacks.
    pub chunks: u32,
    /// First contained callback failure, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<CallbackFailure>,
}

impl PrefillReceipt {
    /// Returns a content identity for exact prefill accounting.
    ///
    /// # Errors
    ///
    /// Returns a validation or serialization error.
    pub fn digest(&self) -> Result<Digest, CoreError> {
        self.progress.validate()?;
        let maximum_polls = self
            .chunks
            .checked_add(1)
            .ok_or_else(|| CoreError::invalid("prefill receipt", "poll count overflowed"))?;
        if self.chunks != self.progress.admitted_chunks
            || self.polls < self.chunks
            || self.polls > maximum_polls
            || (self.finish == PrefillFinish::Complete && self.polls != self.chunks)
        {
            return Err(CoreError::invalid(
                "prefill receipt",
                "callback and chunk accounting is inconsistent",
            ));
        }
        match self.finish {
            PrefillFinish::Complete
                if self.progress.admitted_tokens != self.progress.requested_tokens =>
            {
                return Err(CoreError::invalid(
                    "prefill receipt",
                    "complete prefill did not admit every token",
                ));
            }
            PrefillFinish::Stopped
                if self.progress.admitted_tokens == self.progress.requested_tokens =>
            {
                return Err(CoreError::invalid(
                    "prefill receipt",
                    "stopped prefill admitted every requested token",
                ));
            }
            PrefillFinish::InProgress | PrefillFinish::Complete | PrefillFinish::Stopped => {}
        }
        Digest::of_serializable("prefill-receipt-v2", self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn implementation() -> Digest {
        Digest::of_bytes("test-observer", b"accounting")
    }

    #[test]
    fn observer_receipts_reject_position_overflow() {
        let receipt = ObserverReceipt {
            implementation: implementation(),
            initial_position: u64::MAX,
            requested_tokens: 1,
            polls: 1,
            observed_tokens: 1,
            observed_bytes: 1,
            final_position: u64::MAX,
            stop_requested: false,
            failure: None,
        };
        assert!(receipt.digest().is_err());
    }

    #[test]
    fn observer_receipts_reject_polls_beyond_the_token_budget() {
        let receipt = ObserverReceipt {
            implementation: implementation(),
            initial_position: 0,
            requested_tokens: 1,
            polls: 2,
            observed_tokens: 0,
            observed_bytes: 0,
            final_position: 0,
            stop_requested: false,
            failure: None,
        };
        assert!(receipt.digest().is_err());
    }

    #[test]
    fn prefill_progress_rejects_position_overflow() {
        let progress = PrefillProgress {
            initial_position: u64::MAX,
            requested_tokens: 1,
            admitted_tokens: 1,
            admitted_chunks: 1,
            position: u64::MAX,
        };
        assert!(progress.validate().is_err());
    }

    #[test]
    fn prefill_progress_rejects_more_chunks_than_tokens() {
        let progress = PrefillProgress {
            initial_position: 0,
            requested_tokens: 1,
            admitted_tokens: 1,
            admitted_chunks: 2,
            position: 1,
        };
        assert!(progress.validate().is_err());
    }

    #[test]
    fn prefill_progress_requires_requested_tokens() {
        let progress = PrefillProgress {
            initial_position: 0,
            requested_tokens: 0,
            admitted_tokens: 0,
            admitted_chunks: 0,
            position: 0,
        };
        assert!(progress.validate().is_err());
    }

    #[test]
    fn prefill_finish_must_match_admitted_work() {
        let progress = PrefillProgress {
            initial_position: 3,
            requested_tokens: 2,
            admitted_tokens: 1,
            admitted_chunks: 1,
            position: 4,
        };
        let receipt = PrefillReceipt {
            implementation: implementation(),
            progress,
            finish: PrefillFinish::Complete,
            polls: 1,
            chunks: 1,
            failure: None,
        };
        assert!(receipt.digest().is_err());
    }

    #[test]
    fn complete_prefill_requires_one_poll_per_chunk() {
        let progress = PrefillProgress {
            initial_position: 0,
            requested_tokens: 1,
            admitted_tokens: 1,
            admitted_chunks: 1,
            position: 1,
        };
        let receipt = PrefillReceipt {
            implementation: implementation(),
            progress,
            finish: PrefillFinish::Complete,
            polls: 2,
            chunks: 1,
            failure: None,
        };
        assert!(receipt.digest().is_err());
    }

    #[test]
    fn in_progress_prefill_uses_the_v2_digest_domain() {
        let receipt = PrefillReceipt {
            implementation: implementation(),
            progress: PrefillProgress {
                initial_position: 0,
                requested_tokens: 1,
                admitted_tokens: 0,
                admitted_chunks: 0,
                position: 0,
            },
            finish: PrefillFinish::InProgress,
            polls: 0,
            chunks: 0,
            failure: None,
        };
        assert_eq!(
            receipt.digest().unwrap(),
            Digest::of_serializable("prefill-receipt-v2", &receipt).unwrap()
        );
    }
}

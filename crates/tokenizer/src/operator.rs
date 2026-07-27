// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact operator and cancellation contracts.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use logit_loom_core::{Digest, TokenId};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{CountingSink, MAX_ROW_BYTES, MAX_TOKENS_PER_ROW, SinkFlow, TokenOutputSink};

/// Complete immutable identity of one text-to-token implementation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizationIdentity {
    /// Exact model artifact identity whose token IDs are projected.
    pub model: Digest,
    /// Exact tokenizer configuration artifact.
    pub tokenizer: Digest,
    /// Exact vocabulary table.
    pub vocabulary: Digest,
    /// Exact merge table or an explicit empty-table identity.
    pub merges: Digest,
    /// Exact normalizer and pretokenizer contract.
    pub normalizer: Digest,
    /// Exact Unicode-data revision.
    pub unicode: Digest,
    /// Exact added-token table.
    pub added_tokens: Digest,
    /// Exact linked implementation and kernel revision.
    pub implementation: Digest,
}

impl TokenizationIdentity {
    /// Derives the complete identity without depending on a model-family name.
    ///
    /// # Errors
    ///
    /// Returns an error only if deterministic contract encoding fails.
    pub fn digest(&self) -> Result<Digest, TokenizationError> {
        Digest::of_serializable("tokenization-identity-v1", self)
            .map_err(|error| TokenizationError::Identity(error.to_string()))
    }
}

/// Version-two immutable identity with separate normalizer, pretokenizer, and
/// special-token contracts.
///
/// This successor does not reinterpret [`TokenizationIdentity`] or its
/// `tokenization-identity-v1` domain.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizationIdentityV2 {
    /// Exact model artifact whose token IDs are projected.
    pub model: Digest,
    /// Exact tokenizer configuration artifact.
    pub tokenizer: Digest,
    /// Exact vocabulary table.
    pub vocabulary: Digest,
    /// Exact merge table.
    pub merges: Digest,
    /// Exact normalization contract.
    pub normalizer: Digest,
    /// Exact pretokenization contract.
    pub pretokenizer: Digest,
    /// Exact Unicode-data revision.
    pub unicode: Digest,
    /// Exact added-token table.
    pub added_tokens: Digest,
    /// Exact source-special-token recognition table.
    pub special_tokens: Digest,
    /// Exact linked implementation and kernel revision.
    pub implementation: Digest,
    /// Exact supported policy-schema contract.
    pub policy_schema: Digest,
}

impl TokenizationIdentityV2 {
    /// Derives the complete version-two identity.
    ///
    /// # Errors
    ///
    /// Returns an error only if deterministic contract encoding fails.
    pub fn digest(&self) -> Result<Digest, TokenizationError> {
        Digest::of_serializable("tokenization-identity-v2", self)
            .map_err(|error| TokenizationError::Identity(error.to_string()))
    }

    /// Derives a conservative legacy identity by folding every version-two
    /// contract absent from the version-one shape into its normalizer field.
    ///
    /// # Errors
    ///
    /// Returns an error only if deterministic contract encoding fails.
    pub fn legacy_v1(&self) -> Result<TokenizationIdentity, TokenizationError> {
        let combined = Digest::of_serializable(
            "tokenization-v2-legacy-contract-v1",
            &(
                &self.normalizer,
                &self.pretokenizer,
                &self.special_tokens,
                &self.policy_schema,
            ),
        )
        .map_err(|error| TokenizationError::Identity(error.to_string()))?;
        Ok(TokenizationIdentity {
            model: self.model.clone(),
            tokenizer: self.tokenizer.clone(),
            vocabulary: self.vocabulary.clone(),
            merges: self.merges.clone(),
            normalizer: combined,
            unicode: self.unicode.clone(),
            added_tokens: self.added_tokens.clone(),
            implementation: self.implementation.clone(),
        })
    }
}

/// Exact special-token and output policy applied by an operator.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizationPolicy {
    /// Beginning/end token insertion.
    pub boundary_tokens: BoundaryTokenPolicy,
    /// Recognition of configured special-token spellings in source text.
    pub source_special_tokens: SourceSpecialTokenPolicy,
    /// Token offset materialization.
    pub offsets: OffsetPolicy,
    /// Optional inclusive count threshold for early-stop counting.
    pub count_through: Option<u64>,
}

/// Exact beginning/end token insertion behavior.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BoundaryTokenPolicy {
    /// Add neither boundary token.
    None,
    /// Add only the configured beginning token.
    Beginning,
    /// Add only the configured ending token.
    Ending,
    /// Add both configured boundary tokens.
    Both,
}

/// Whether configured special-token spellings are recognized in source text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SourceSpecialTokenPolicy {
    /// Treat every source spelling as ordinary text.
    OrdinaryText,
    /// Recognize only special tokens in the exact tokenizer configuration.
    RecognizeConfigured,
}

/// Whether token source offsets are materialized.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum OffsetPolicy {
    /// Do not materialize offsets.
    Omit,
    /// Materialize exact half-open source byte ranges.
    Include,
}

impl TokenizationPolicy {
    /// Derives the policy identity.
    ///
    /// # Errors
    ///
    /// Returns an error only if deterministic contract encoding fails.
    pub fn digest(&self) -> Result<Digest, TokenizationError> {
        Digest::of_serializable("tokenization-policy-v1", self)
            .map_err(|error| TokenizationError::Identity(error.to_string()))
    }
}

/// One exact token and its half-open source-byte range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenSpan {
    /// Non-negative model tokenizer identifier.
    pub token: TokenId,
    /// Inclusive UTF-8 byte offset.
    pub start: u32,
    /// Exclusive UTF-8 byte offset.
    pub end: u32,
}

/// Count-only result, including whether an early-stop threshold was crossed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CountResult {
    /// Exact count when complete, or the first count greater than the bound.
    pub count: u64,
    /// Whether the entire source was consumed.
    pub complete: bool,
}

impl CountResult {
    /// Creates an exact complete result.
    #[must_use]
    pub const fn complete(count: u64) -> Self {
        Self {
            count,
            complete: true,
        }
    }
}

/// Thread-safe cooperative cancellation checked only at operator boundaries.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken {
    requested: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Requests cancellation.
    pub fn request(&self) {
        self.requested.store(true, Ordering::Release);
    }

    /// Reports whether cancellation was requested.
    #[must_use]
    pub fn is_requested(&self) -> bool {
        self.requested.load(Ordering::Acquire)
    }

    /// Returns an error at a declared safe boundary.
    ///
    /// # Errors
    ///
    /// Returns [`TokenizationError::Cancelled`] after cancellation.
    pub fn check(&self) -> Result<(), TokenizationError> {
        if self.is_requested() {
            Err(TokenizationError::Cancelled)
        } else {
            Ok(())
        }
    }
}

/// Mechanical tokenizer failure.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum TokenizationError {
    /// Cooperative cancellation reached a declared boundary.
    #[error("tokenization cancelled")]
    Cancelled,
    /// Input or output violated a public bound.
    #[error("{field} exceeds its bound {limit}")]
    Bound {
        /// Stable field name.
        field: &'static str,
        /// Inclusive public limit.
        limit: usize,
    },
    /// Input bytes or operator output were malformed.
    #[error("invalid tokenization data: {0}")]
    Invalid(String),
    /// Deterministic identity encoding failed.
    #[error("unable to derive tokenization identity: {0}")]
    Identity(String),
    /// Backend-specific exact tokenizer failure.
    #[error("tokenizer operator failed: {0}")]
    Operator(String),
}

/// Exact backend operator invoked by bounded bulk mechanics.
///
/// The default count implementation is a correctness fallback and
/// materializes spans. A backend that can count without token IDs should
/// override it.
pub trait ExactTokenizer: Send + Sync {
    /// Returns the exact immutable execution identity.
    fn identity(&self) -> &TokenizationIdentity;

    /// Returns a version-two split identity when the backend can provide one.
    fn identity_v2(&self) -> Option<&TokenizationIdentityV2> {
        None
    }

    /// Appends exact token IDs and source offsets into caller-owned scratch.
    ///
    /// Implementations must clear `output` before writing and must not retain
    /// source bytes after returning.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed input, cancellation, bounds, or backend
    /// failure.
    fn tokenize_into(
        &self,
        source: &[u8],
        policy: &TokenizationPolicy,
        output: &mut Vec<TokenSpan>,
        cancellation: &CancellationToken,
    ) -> Result<(), TokenizationError>;

    /// Streams exact tokens into a caller-owned destination.
    ///
    /// The compatibility implementation materializes a temporary span vector.
    /// A backend with reusable or vector-free mechanics should override this
    /// method and stop as soon as the sink returns [`SinkFlow::Stop`].
    ///
    /// # Errors
    ///
    /// Returns an error for malformed input, cancellation, bounds, backend
    /// failure, or sink rejection.
    fn tokenize_to_sink(
        &self,
        source: &[u8],
        policy: &TokenizationPolicy,
        sink: &mut dyn TokenOutputSink,
        cancellation: &CancellationToken,
    ) -> Result<bool, TokenizationError> {
        validate_source(source)?;
        cancellation.check()?;
        let mut spans = Vec::new();
        self.tokenize_into(source, policy, &mut spans, cancellation)?;
        validate_spans(source.len(), &spans)?;
        sink.begin()?;
        for token in spans {
            if sink.push(token)? == SinkFlow::Stop {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Counts tokens, optionally stopping after the inclusive threshold.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed input, cancellation, bounds, or backend
    /// failure.
    fn count(
        &self,
        source: &[u8],
        policy: &TokenizationPolicy,
        cancellation: &CancellationToken,
    ) -> Result<CountResult, TokenizationError> {
        validate_source(source)?;
        cancellation.check()?;
        let mut sink = CountingSink::new(policy.count_through);
        let consumed = self.tokenize_to_sink(source, policy, &mut sink, cancellation)?;
        if consumed {
            sink.finish();
        }
        Ok(sink.result())
    }
}

/// Validates source UTF-8 and the public byte bound.
///
/// # Errors
///
/// Returns an error for oversized, invalid UTF-8, or NUL-containing input.
pub fn validate_source(source: &[u8]) -> Result<&str, TokenizationError> {
    if source.len() > MAX_ROW_BYTES {
        return Err(TokenizationError::Bound {
            field: "source bytes",
            limit: MAX_ROW_BYTES,
        });
    }
    if source.contains(&0) {
        return Err(TokenizationError::Invalid(
            "source must not contain NUL bytes".to_owned(),
        ));
    }
    std::str::from_utf8(source)
        .map_err(|_| TokenizationError::Invalid("source must be valid UTF-8".to_owned()))
}

/// Validates stable, in-range token offsets.
///
/// # Errors
///
/// Returns an error for too many tokens or non-monotonic/out-of-range spans.
pub fn validate_spans(source_len: usize, spans: &[TokenSpan]) -> Result<(), TokenizationError> {
    if spans.len() > MAX_TOKENS_PER_ROW {
        return Err(TokenizationError::Bound {
            field: "token spans",
            limit: MAX_TOKENS_PER_ROW,
        });
    }
    let source_len = u32::try_from(source_len).map_err(|_| TokenizationError::Bound {
        field: "source bytes",
        limit: MAX_ROW_BYTES,
    })?;
    let mut prior_start = 0_u32;
    for span in spans {
        if span.start > span.end || span.end > source_len || span.start < prior_start {
            return Err(TokenizationError::Invalid(
                "token offsets are not stable half-open source ranges".to_owned(),
            ));
        }
        prior_start = span.start;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_and_span_bounds_are_fail_closed() {
        assert!(validate_source(b"hello").is_ok());
        assert!(validate_source(b"a\0b").is_err());
        assert!(validate_source(&[0xff]).is_err());
        assert!(
            validate_spans(
                2,
                &[TokenSpan {
                    token: TokenId::new(1).unwrap(),
                    start: 1,
                    end: 3,
                }]
            )
            .is_err()
        );
    }

    #[test]
    fn cancellation_is_shared_and_explicit() {
        let cancellation = CancellationToken::default();
        assert!(cancellation.check().is_ok());
        cancellation.request();
        assert_eq!(cancellation.check(), Err(TokenizationError::Cancelled));
    }
}

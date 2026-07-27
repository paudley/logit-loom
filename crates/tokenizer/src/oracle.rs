// SPDX-License-Identifier: MIT OR Apache-2.0

//! Caller-supplied exact token-ID and offset oracle qualification.

use logit_loom_core::Digest;
use serde::{Deserialize, Serialize};

use crate::{
    CancellationToken, ExactTokenizer, MAX_BULK_ROWS, MAX_ROW_BYTES, TokenSpan, TokenizationError,
    TokenizationPolicy,
};

/// One caller-supplied exact engine-oracle case.
#[derive(Clone, Copy, Debug)]
pub struct TokenizationOracleCase<'a> {
    /// Exact source bytes.
    pub source: &'a [u8],
    /// Exact expected token IDs and half-open source offsets.
    pub expected: &'a [TokenSpan],
}

/// Content-free differential qualification receipt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizationOracleReceipt {
    /// Exact tokenizer implementation identity.
    pub tokenizer: Digest,
    /// Exact execution policy identity.
    pub policy: Digest,
    /// Number of exact cases compared.
    pub cases: u32,
    /// Aggregate source bytes.
    pub source_bytes: u64,
    /// Aggregate expected tokens.
    pub tokens: u64,
    /// Ordered source/expected/actual transcript identity.
    pub transcript: Digest,
}

/// Compares exact token IDs and offsets against caller-supplied engine
/// oracles.
///
/// Source bytes and token vectors remain caller-owned and are not retained in
/// the receipt.
///
/// # Errors
///
/// Returns a bound, cancellation, tokenizer, malformed-oracle, or content-free
/// mismatch error.
pub fn qualify_tokenizer_oracle(
    tokenizer: &dyn ExactTokenizer,
    policy: &TokenizationPolicy,
    cases: &[TokenizationOracleCase<'_>],
    cancellation: &CancellationToken,
) -> Result<TokenizationOracleReceipt, TokenizationError> {
    if cases.is_empty() || cases.len() > MAX_BULK_ROWS {
        return Err(TokenizationError::Bound {
            field: "oracle cases",
            limit: MAX_BULK_ROWS,
        });
    }
    let tokenizer_identity = tokenizer.identity_v2().map_or_else(
        || tokenizer.identity().digest(),
        crate::TokenizationIdentityV2::digest,
    )?;
    let policy_identity = policy.digest()?;
    let mut transcript = Vec::with_capacity(cases.len());
    let mut source_bytes = 0_u64;
    let mut tokens = 0_u64;
    let mut actual = Vec::new();
    for (index, case) in cases.iter().enumerate() {
        cancellation.check()?;
        if case.source.len() > MAX_ROW_BYTES {
            return Err(TokenizationError::Bound {
                field: "oracle source bytes",
                limit: MAX_ROW_BYTES,
            });
        }
        crate::operator::validate_spans(case.source.len(), case.expected)?;
        tokenizer.tokenize_into(case.source, policy, &mut actual, cancellation)?;
        crate::operator::validate_spans(case.source.len(), &actual)?;
        if actual != case.expected {
            return Err(TokenizationError::Operator(format!(
                "oracle case {index} differs"
            )));
        }
        source_bytes = source_bytes
            .checked_add(u64::try_from(case.source.len()).map_err(|_| {
                TokenizationError::Bound {
                    field: "oracle source bytes",
                    limit: MAX_ROW_BYTES,
                }
            })?)
            .ok_or_else(|| {
                TokenizationError::Invalid("oracle source-byte accounting overflowed".to_owned())
            })?;
        tokens = tokens
            .checked_add(
                u64::try_from(actual.len()).map_err(|_| TokenizationError::Bound {
                    field: "oracle tokens",
                    limit: crate::MAX_TOKENS_PER_ROW,
                })?,
            )
            .ok_or_else(|| {
                TokenizationError::Invalid("oracle token accounting overflowed".to_owned())
            })?;
        transcript.push((
            Digest::of_bytes("tokenizer-oracle-source-v1", case.source),
            Digest::of_serializable("tokenizer-oracle-expected-tokens-v1", case.expected)
                .map_err(|error| TokenizationError::Identity(error.to_string()))?,
            Digest::of_serializable("tokenizer-oracle-actual-tokens-v1", &actual)
                .map_err(|error| TokenizationError::Identity(error.to_string()))?,
        ));
    }
    Ok(TokenizationOracleReceipt {
        tokenizer: tokenizer_identity,
        policy: policy_identity,
        cases: u32::try_from(cases.len()).map_err(|_| TokenizationError::Bound {
            field: "oracle cases",
            limit: MAX_BULK_ROWS,
        })?,
        source_bytes,
        tokens,
        transcript: Digest::of_serializable("tokenizer-oracle-transcript-v1", &transcript)
            .map_err(|error| TokenizationError::Identity(error.to_string()))?,
    })
}

#[cfg(test)]
mod tests {
    use logit_loom_core::TokenId;

    use crate::{
        BoundaryTokenPolicy, BpeBoundaryTokens, BpeMerge, OffsetPolicy, RankedBpe, RankedByteBpe,
        SourceSpecialTokenPolicy,
    };

    use super::*;

    #[test]
    fn exact_oracle_receipt_retains_no_source_or_tokens() {
        let byte_tokens =
            std::array::from_fn(|byte| TokenId::new(i32::try_from(byte).unwrap()).unwrap());
        let tokenizer = RankedByteBpe::new(
            Digest::of_bytes("model", b"one"),
            byte_tokens,
            BpeBoundaryTokens::default(),
            RankedBpe::new(vec![BpeMerge {
                left: TokenId::new(i32::from(b'a')).unwrap(),
                right: TokenId::new(i32::from(b'b')).unwrap(),
                merged: TokenId::new(256).unwrap(),
                rank: 0,
            }])
            .unwrap(),
        )
        .unwrap();
        let policy = TokenizationPolicy {
            boundary_tokens: BoundaryTokenPolicy::None,
            source_special_tokens: SourceSpecialTokenPolicy::OrdinaryText,
            offsets: OffsetPolicy::Include,
            count_through: None,
        };
        let expected = [TokenSpan {
            token: TokenId::new(256).unwrap(),
            start: 0,
            end: 2,
        }];
        let receipt = qualify_tokenizer_oracle(
            &tokenizer,
            &policy,
            &[TokenizationOracleCase {
                source: b"ab",
                expected: &expected,
            }],
            &CancellationToken::default(),
        )
        .unwrap();
        let json = serde_json::to_string(&receipt).unwrap();
        assert!(!json.contains("\"ab\""));
        assert_eq!(receipt.cases, 1);
        assert_eq!(receipt.tokens, 1);
    }

    #[test]
    fn mismatched_oracle_fails_closed() {
        let byte_tokens =
            std::array::from_fn(|byte| TokenId::new(i32::try_from(byte).unwrap()).unwrap());
        let tokenizer = RankedByteBpe::new(
            Digest::of_bytes("model", b"one"),
            byte_tokens,
            BpeBoundaryTokens::default(),
            RankedBpe::new(vec![BpeMerge {
                left: TokenId::new(i32::from(b'a')).unwrap(),
                right: TokenId::new(i32::from(b'b')).unwrap(),
                merged: TokenId::new(256).unwrap(),
                rank: 0,
            }])
            .unwrap(),
        )
        .unwrap();
        let policy = TokenizationPolicy {
            boundary_tokens: BoundaryTokenPolicy::None,
            source_special_tokens: SourceSpecialTokenPolicy::OrdinaryText,
            offsets: OffsetPolicy::Include,
            count_through: None,
        };
        let wrong = [TokenSpan {
            token: TokenId::new(1).unwrap(),
            start: 0,
            end: 2,
        }];
        assert!(
            qualify_tokenizer_oracle(
                &tokenizer,
                &policy,
                &[TokenizationOracleCase {
                    source: b"ab",
                    expected: &wrong,
                }],
                &CancellationToken::default(),
            )
            .is_err()
        );
    }
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! Token-aware source chunk planning.

use serde::{Deserialize, Serialize};

use crate::{MAX_TOKENS_PER_ROW, TokenSpan, TokenizationError};

/// Bounded token chunking contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChunkPolicy {
    /// Maximum tokens in one chunk.
    pub maximum_tokens: u32,
    /// Tokens repeated from the preceding chunk.
    pub overlap_tokens: u32,
    /// Maximum emitted chunks.
    pub maximum_chunks: u32,
}

impl ChunkPolicy {
    fn validate(self) -> Result<(), TokenizationError> {
        if self.maximum_tokens == 0
            || !usize::try_from(self.maximum_tokens).is_ok_and(|value| value <= MAX_TOKENS_PER_ROW)
            || self.overlap_tokens >= self.maximum_tokens
            || self.maximum_chunks == 0
            || !usize::try_from(self.maximum_chunks).is_ok_and(|value| value <= MAX_TOKENS_PER_ROW)
        {
            return Err(TokenizationError::Invalid(
                "chunk policy is outside public bounds".to_owned(),
            ));
        }
        Ok(())
    }
}

/// One stable token and source-byte range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceChunk {
    /// Inclusive token index.
    pub token_start: u64,
    /// Exclusive token index.
    pub token_end: u64,
    /// Inclusive source byte offset.
    pub source_start: u64,
    /// Exclusive source byte offset.
    pub source_end: u64,
}

/// Complete or threshold-bounded chunk plan.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChunkPlan {
    /// Stable ordered chunks.
    pub chunks: Vec<SourceChunk>,
    /// Whether all token spans were represented.
    pub complete: bool,
}

/// Produces stable source chunks from one exact tokenization pass.
///
/// Empty input produces an empty complete plan. Beginning/end tokens whose
/// offsets are empty remain attached to the nearest source-bearing chunk.
///
/// # Errors
///
/// Returns an error for invalid policy, invalid spans, or arithmetic overflow.
pub fn plan_chunks(
    source_len: usize,
    spans: &[TokenSpan],
    policy: ChunkPolicy,
) -> Result<ChunkPlan, TokenizationError> {
    policy.validate()?;
    super::operator::validate_spans(source_len, spans)?;
    if spans.is_empty() {
        return Ok(ChunkPlan {
            chunks: Vec::new(),
            complete: true,
        });
    }
    let width = usize::try_from(policy.maximum_tokens)
        .map_err(|_| TokenizationError::Invalid("chunk width overflowed".to_owned()))?;
    let overlap = usize::try_from(policy.overlap_tokens)
        .map_err(|_| TokenizationError::Invalid("chunk overlap overflowed".to_owned()))?;
    let maximum_chunks = usize::try_from(policy.maximum_chunks)
        .map_err(|_| TokenizationError::Invalid("chunk count overflowed".to_owned()))?;
    let step = width - overlap;
    let mut chunks = Vec::with_capacity(spans.len().div_ceil(step).min(maximum_chunks));
    let mut start = 0_usize;
    while start < spans.len() && chunks.len() < maximum_chunks {
        let end = start.saturating_add(width).min(spans.len());
        let source_start = spans[start..end]
            .iter()
            .map(|span| span.start)
            .min()
            .unwrap_or(0);
        let source_end = spans[start..end]
            .iter()
            .map(|span| span.end)
            .max()
            .unwrap_or(source_start);
        chunks.push(SourceChunk {
            token_start: u64::try_from(start)
                .map_err(|_| TokenizationError::Invalid("token offset overflowed".to_owned()))?,
            token_end: u64::try_from(end)
                .map_err(|_| TokenizationError::Invalid("token offset overflowed".to_owned()))?,
            source_start: u64::from(source_start),
            source_end: u64::from(source_end),
        });
        if end == spans.len() {
            break;
        }
        start = start
            .checked_add(step)
            .ok_or_else(|| TokenizationError::Invalid("chunk step overflowed".to_owned()))?;
    }
    Ok(ChunkPlan {
        complete: chunks
            .last()
            .is_some_and(|chunk| chunk.token_end == u64::try_from(spans.len()).unwrap_or(u64::MAX)),
        chunks,
    })
}

#[cfg(test)]
mod tests {
    use logit_loom_core::TokenId;

    use super::*;

    fn spans(count: u32) -> Vec<TokenSpan> {
        (0..count)
            .map(|index| TokenSpan {
                token: TokenId::new(i32::try_from(index).unwrap()).unwrap(),
                start: index,
                end: index + 1,
            })
            .collect()
    }

    #[test]
    fn chunks_overlap_and_preserve_source_ranges() {
        let plan = plan_chunks(
            10,
            &spans(10),
            ChunkPolicy {
                maximum_tokens: 4,
                overlap_tokens: 1,
                maximum_chunks: 10,
            },
        )
        .unwrap();
        assert!(plan.complete);
        assert_eq!(
            plan.chunks
                .iter()
                .map(|chunk| (chunk.token_start, chunk.token_end))
                .collect::<Vec<_>>(),
            vec![(0, 4), (3, 7), (6, 10)]
        );
    }

    #[test]
    fn chunk_limit_reports_incomplete() {
        let plan = plan_chunks(
            10,
            &spans(10),
            ChunkPolicy {
                maximum_tokens: 4,
                overlap_tokens: 0,
                maximum_chunks: 1,
            },
        )
        .unwrap();
        assert!(!plan.complete);
    }
}

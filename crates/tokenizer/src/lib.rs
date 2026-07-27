// SPDX-License-Identifier: MIT OR Apache-2.0

#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod batch;
mod cache;
mod chunk;
mod operator;

pub use batch::{
    BatchCandidate, BatchLimits, BatchPlan, BatchRow, LengthBucket, StableScatter,
    plan_length_buckets,
};
pub use cache::{CacheConfig, CacheDisposition, CacheStats, CachedProjection, SpanCache};
pub use chunk::{ChunkPlan, ChunkPolicy, SourceChunk, plan_chunks};
pub use operator::{
    BoundaryTokenPolicy, CancellationToken, CountResult, ExactTokenizer, OffsetPolicy,
    SourceSpecialTokenPolicy, TokenSpan, TokenizationError, TokenizationIdentity,
    TokenizationPolicy,
};

/// Maximum rows in one public bulk operation.
pub const MAX_BULK_ROWS: usize = 65_536;
/// Maximum token spans materialized for one row.
pub const MAX_TOKENS_PER_ROW: usize = 1_048_576;
/// Maximum source bytes accepted by one row.
pub const MAX_ROW_BYTES: usize = 16 * 1024 * 1024;

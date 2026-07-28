// SPDX-License-Identifier: MIT OR Apache-2.0

#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod batch;
mod bpe;
mod cache;
mod chunk;
mod operator;
mod oracle;
mod pool;
mod qwen;
mod sink;

pub use batch::{
    BatchCandidate, BatchLimits, BatchPlan, BatchRow, LengthBucket, StableScatter,
    plan_length_buckets,
};
pub use bpe::{
    BpeBoundaryTokens, BpeMerge, BpeScratch, GIGATOKEN_REVISION, MAX_BPE_MERGE_RANK,
    MAX_BPE_SYMBOLS, RankedBpe, RankedByteBpe,
};
pub use cache::{CacheConfig, CacheDisposition, CacheStats, CachedProjection, SpanCache};
pub use chunk::{ChunkPlan, ChunkPolicy, SourceChunk, plan_chunks};
pub use operator::{
    BoundaryTokenPolicy, CancellationToken, CountResult, ExactTokenizer, OffsetPolicy,
    SourceSpecialTokenPolicy, TokenSpan, TokenizationError, TokenizationIdentity,
    TokenizationIdentityV2, TokenizationPolicy,
};
pub use oracle::{TokenizationOracleCase, TokenizationOracleReceipt, qualify_tokenizer_oracle};
pub use pool::{
    DedicatedPool, MAX_POOL_QUEUE, MAX_POOL_WORKERS, PoolConfig, PoolError, PoolJob, PoolReceipt,
};
pub use qwen::{QwenPretokenizer, QwenRankedBpe, QwenTokenizerConfig};
pub use sink::{
    CountingSink, SinkFlow, TokenIdSliceSink, TokenOutputSink, VecTokenSink, tokenize_via_scratch,
    validate_sink_policy,
};

/// Maximum rows in one public bulk operation.
pub const MAX_BULK_ROWS: usize = 65_536;
/// Maximum token spans materialized for one row.
pub const MAX_TOKENS_PER_ROW: usize = 1_048_576;
/// Maximum source bytes accepted by one row.
pub const MAX_ROW_BYTES: usize = 16 * 1024 * 1024;

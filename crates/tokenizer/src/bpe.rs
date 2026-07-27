// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pinned, safe ranked-BPE mechanics derived from Gigatoken's merge core.

use std::{
    cmp::Reverse,
    collections::{BTreeMap, BinaryHeap, HashMap, HashSet},
};

use logit_loom_core::{Digest, TokenId};
use serde::{Deserialize, Serialize};
use wide::u32x8;

use crate::operator::{validate_source, validate_spans};
use crate::{
    BoundaryTokenPolicy, CancellationToken, CountResult, CountingSink, ExactTokenizer,
    MAX_TOKENS_PER_ROW, OffsetPolicy, SinkFlow, TokenOutputSink, TokenSpan, TokenizationError,
    TokenizationIdentity, TokenizationIdentityV2, TokenizationPolicy, VecTokenSink,
};

/// Exact Gigatoken source revision from which the ranked merge mechanics were
/// derived.
pub const GIGATOKEN_REVISION: &str = "0d9765fa7312af7534535e6315a5c49d74807b2a";
/// Maximum explicit BPE merge rank.
pub const MAX_BPE_MERGE_RANK: u32 = (1 << 27) - 1;
/// Maximum initial symbols accepted by one kernel call.
pub const MAX_BPE_SYMBOLS: usize = MAX_TOKENS_PER_ROW;

const SHORT_MERGE_MAX: usize = 16;
const NONE: usize = usize::MAX;

/// One exact ranked merge rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BpeMerge {
    /// Left token.
    pub left: TokenId,
    /// Right token.
    pub right: TokenId,
    /// Result token.
    pub merged: TokenId,
    /// Unique merge priority; lower ranks run first.
    pub rank: u32,
}

#[derive(Clone, Copy, Debug)]
struct MergeResult {
    token: TokenId,
    rank: u32,
}

/// Allocation-reusing ranked-BPE work storage.
#[derive(Debug, Default)]
pub struct BpeScratch {
    symbols: Vec<TokenSpan>,
    next: Vec<usize>,
    previous: Vec<usize>,
    heap: BinaryHeap<Reverse<(u32, usize)>>,
    peak_symbols: usize,
}

impl BpeScratch {
    /// Creates empty reusable work storage.
    pub const fn new() -> Self {
        Self {
            symbols: Vec::new(),
            next: Vec::new(),
            previous: Vec::new(),
            heap: BinaryHeap::new(),
            peak_symbols: 0,
        }
    }

    /// Returns the largest initial-symbol count observed by this scratch.
    pub const fn peak_symbols(&self) -> usize {
        self.peak_symbols
    }

    /// Returns retained symbol capacity for deployment accounting.
    pub fn symbol_capacity(&self) -> usize {
        self.symbols.capacity()
    }
}

/// Exact ranked-BPE merge kernel.
#[derive(Clone, Debug)]
pub struct RankedBpe {
    merges: HashMap<u64, MergeResult>,
    canonical: Vec<BpeMerge>,
    merges_identity: Digest,
    implementation: Digest,
}

impl RankedBpe {
    /// Builds a canonical ranked merge table.
    ///
    /// Rules must be strictly rank-ordered, use unique ranks and pairs, and
    /// remain inside the packed short-SIMD rank range.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty, oversized, non-canonical, or duplicate
    /// table.
    pub fn new(canonical: Vec<BpeMerge>) -> Result<Self, TokenizationError> {
        if canonical.is_empty()
            || canonical.len() > MAX_TOKENS_PER_ROW
            || canonical.iter().any(|rule| rule.rank > MAX_BPE_MERGE_RANK)
            || canonical
                .windows(2)
                .any(|pair| pair[0].rank >= pair[1].rank)
        {
            return Err(TokenizationError::Invalid(
                "ranked BPE rules are empty, excessive, or non-canonical".to_owned(),
            ));
        }
        let mut merges = HashMap::with_capacity(canonical.len());
        let mut ranks = HashSet::with_capacity(canonical.len());
        for rule in &canonical {
            if !ranks.insert(rule.rank)
                || merges
                    .insert(
                        pair_key(rule.left, rule.right),
                        MergeResult {
                            token: rule.merged,
                            rank: rule.rank,
                        },
                    )
                    .is_some()
            {
                return Err(TokenizationError::Invalid(
                    "ranked BPE rules repeat a rank or pair".to_owned(),
                ));
            }
        }
        let merges_identity = Digest::of_serializable("ranked-bpe-merges-v1", &canonical)
            .map_err(|error| TokenizationError::Identity(error.to_string()))?;
        let implementation = Digest::of_serializable(
            "ranked-bpe-implementation-v1",
            &(
                GIGATOKEN_REVISION,
                "gigatoken-ranked-short-merge",
                "wide-0.7.33",
                "safe-packed-u32x8-min-v1",
                &merges_identity,
            ),
        )
        .map_err(|error| TokenizationError::Identity(error.to_string()))?;
        Ok(Self {
            merges,
            canonical,
            merges_identity,
            implementation,
        })
    }

    /// Returns the exact canonical merge-table identity.
    pub const fn merges_identity(&self) -> &Digest {
        &self.merges_identity
    }

    /// Returns the exact source/kernel/table implementation identity.
    pub const fn implementation(&self) -> &Digest {
        &self.implementation
    }

    /// Returns canonical merge rules.
    pub fn rules(&self) -> &[BpeMerge] {
        &self.canonical
    }

    /// Merges exact initial symbols and streams survivors to a caller-owned
    /// sink.
    ///
    /// Cancellation is checked before work, after every committed merge, and
    /// before output. Output offsets span the complete merged source range.
    ///
    /// # Errors
    ///
    /// Returns a symbol-bound, offset, cancellation, merge, or sink error.
    pub fn merge_to_sink(
        &self,
        initial: &[TokenSpan],
        offset_policy: OffsetPolicy,
        scratch: &mut BpeScratch,
        sink: &mut dyn TokenOutputSink,
        cancellation: &CancellationToken,
    ) -> Result<bool, TokenizationError> {
        validate_initial_symbols(initial)?;
        cancellation.check()?;
        self.merge(initial, scratch, cancellation)?;
        cancellation.check()?;
        sink.begin()?;
        for symbol in scratch.symbols.iter().copied() {
            let symbol = if offset_policy == OffsetPolicy::Omit {
                TokenSpan {
                    token: symbol.token,
                    start: 0,
                    end: 0,
                }
            } else {
                symbol
            };
            if sink.push(symbol)? == SinkFlow::Stop {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn merge(
        &self,
        initial: &[TokenSpan],
        scratch: &mut BpeScratch,
        cancellation: &CancellationToken,
    ) -> Result<(), TokenizationError> {
        scratch.symbols.clear();
        scratch.symbols.extend_from_slice(initial);
        scratch.peak_symbols = scratch.peak_symbols.max(initial.len());
        self.merge_prepared(scratch, cancellation)
    }

    fn merge_prepared(
        &self,
        scratch: &mut BpeScratch,
        cancellation: &CancellationToken,
    ) -> Result<(), TokenizationError> {
        if scratch.symbols.len() < 2 {
            return Ok(());
        }
        if scratch.symbols.len() <= SHORT_MERGE_MAX {
            self.merge_short(scratch, cancellation)
        } else {
            self.merge_heap(scratch, cancellation)
        }
    }

    fn merge_short(
        &self,
        scratch: &mut BpeScratch,
        cancellation: &CancellationToken,
    ) -> Result<(), TokenizationError> {
        let count = scratch.symbols.len();
        let mut next = [0_u8; SHORT_MERGE_MAX];
        let mut previous = [0_u8; SHORT_MERGE_MAX];
        let mut packed = [u32::MAX; SHORT_MERGE_MAX];
        let mut merged = [None; SHORT_MERGE_MAX];
        for index in 0..count {
            next[index] = u8::try_from(index + 1)
                .map_err(|_| TokenizationError::Invalid("short BPE index exceeds u8".to_owned()))?;
            previous[index] = u8::try_from(index)
                .map_err(|_| TokenizationError::Invalid("short BPE index exceeds u8".to_owned()))?
                .wrapping_sub(1);
        }
        for index in 0..count - 1 {
            (packed[index], merged[index]) = self.packed_merge(
                scratch.symbols[index].token,
                scratch.symbols[index + 1].token,
                index,
            );
        }
        loop {
            let best = simd_min_packed(packed);
            if best == u32::MAX {
                break;
            }
            let index = usize::try_from(best & 0x1f).map_err(|_| {
                TokenizationError::Invalid("short BPE position exceeds usize".to_owned())
            })?;
            let token = merged[index].ok_or_else(|| {
                TokenizationError::Invalid("short BPE rank has no merged token".to_owned())
            })?;
            let dead = usize::from(next[index]);
            let new_right = usize::from(next[dead]);
            scratch.symbols[index].token = token;
            scratch.symbols[index].end = scratch.symbols[dead].end;
            next[index] = u8::try_from(new_right).map_err(|_| {
                TokenizationError::Invalid("short BPE neighbor exceeds u8".to_owned())
            })?;
            packed[dead] = u32::MAX;
            merged[dead] = None;
            if new_right < count {
                previous[new_right] = u8::try_from(index).map_err(|_| {
                    TokenizationError::Invalid("short BPE index exceeds u8".to_owned())
                })?;
                (packed[index], merged[index]) = self.packed_merge(
                    scratch.symbols[index].token,
                    scratch.symbols[new_right].token,
                    index,
                );
            } else {
                packed[index] = u32::MAX;
                merged[index] = None;
            }
            let left = usize::from(previous[index]);
            if left < count {
                (packed[left], merged[left]) = self.packed_merge(
                    scratch.symbols[left].token,
                    scratch.symbols[index].token,
                    left,
                );
            }
            cancellation.check()?;
        }
        compact_short(&mut scratch.symbols, &next, count);
        Ok(())
    }

    fn merge_heap(
        &self,
        scratch: &mut BpeScratch,
        cancellation: &CancellationToken,
    ) -> Result<(), TokenizationError> {
        let count = scratch.symbols.len();
        scratch.next.clear();
        scratch.next.extend(1..count);
        scratch.next.push(NONE);
        scratch.previous.clear();
        scratch.previous.push(NONE);
        scratch.previous.extend(0..count - 1);
        scratch.heap.clear();
        for index in 0..count - 1 {
            if let Some(result) = self.lookup(
                scratch.symbols[index].token,
                scratch.symbols[index + 1].token,
            ) {
                scratch.heap.push(Reverse((result.rank, index)));
            }
        }
        while let Some(Reverse((expected_rank, index))) = scratch.heap.pop() {
            let right = scratch.next[index];
            if right == NONE {
                continue;
            }
            let Some(result) =
                self.lookup(scratch.symbols[index].token, scratch.symbols[right].token)
            else {
                continue;
            };
            if result.rank != expected_rank {
                continue;
            }
            scratch.symbols[index].token = result.token;
            scratch.symbols[index].end = scratch.symbols[right].end;
            let new_right = scratch.next[right];
            scratch.next[index] = new_right;
            scratch.next[right] = NONE;
            if new_right != NONE {
                scratch.previous[new_right] = index;
            }
            let left = scratch.previous[index];
            if left != NONE
                && let Some(result) =
                    self.lookup(scratch.symbols[left].token, scratch.symbols[index].token)
            {
                scratch.heap.push(Reverse((result.rank, left)));
            }
            if new_right != NONE
                && let Some(result) = self.lookup(
                    scratch.symbols[index].token,
                    scratch.symbols[new_right].token,
                )
            {
                scratch.heap.push(Reverse((result.rank, index)));
            }
            cancellation.check()?;
        }
        compact_heap(&mut scratch.symbols, &scratch.next);
        Ok(())
    }

    fn packed_merge(
        &self,
        left: TokenId,
        right: TokenId,
        position: usize,
    ) -> (u32, Option<TokenId>) {
        self.lookup(left, right).map_or((u32::MAX, None), |result| {
            (
                (result.rank << 5) | u32::try_from(position).unwrap_or(31),
                Some(result.token),
            )
        })
    }

    fn lookup(&self, left: TokenId, right: TokenId) -> Option<MergeResult> {
        self.merges.get(&pair_key(left, right)).copied()
    }
}

/// Optional beginning/end token IDs for a byte-BPE tokenizer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BpeBoundaryTokens {
    /// Beginning token.
    pub beginning: Option<TokenId>,
    /// Ending token.
    pub ending: Option<TokenId>,
}

/// Exact identity-normalized, whole-span UTF-8 byte-BPE tokenizer.
///
/// This tokenizer intentionally performs no Unicode normalization and applies
/// merges across the complete source byte span. Tokenizers with regex,
/// metaspace, or model-specific pretokenization require a separate exact
/// adapter and must not be represented by this type.
#[derive(Clone, Debug)]
pub struct RankedByteBpe {
    identity: TokenizationIdentity,
    identity_v2: TokenizationIdentityV2,
    byte_tokens: Box<[TokenId; 256]>,
    boundaries: BpeBoundaryTokens,
    kernel: RankedBpe,
}

impl RankedByteBpe {
    /// Constructs an exact whole-span byte-BPE tokenizer and derives every
    /// implementation-owned identity from its tables.
    ///
    /// # Errors
    ///
    /// Returns an error when byte roots conflict, a merge references unknown
    /// symbols, or identity encoding fails.
    pub fn new(
        model: Digest,
        byte_tokens: [TokenId; 256],
        boundaries: BpeBoundaryTokens,
        kernel: RankedBpe,
    ) -> Result<Self, TokenizationError> {
        if byte_tokens
            .iter()
            .map(|token| token.get())
            .collect::<HashSet<_>>()
            .len()
            != byte_tokens.len()
        {
            return Err(TokenizationError::Invalid(
                "byte roots repeat a token ID".to_owned(),
            ));
        }
        let vocabulary = derive_vocabulary(&byte_tokens, kernel.rules())?;
        let vocabulary_identity = Digest::of_serializable(
            "ranked-byte-bpe-vocabulary-v1",
            &(&byte_tokens[..], &vocabulary),
        )
        .map_err(|error| TokenizationError::Identity(error.to_string()))?;
        let normalizer = Digest::of_bytes("tokenizer-normalizer-v1", b"utf8-identity");
        let pretokenizer = Digest::of_bytes("tokenizer-pretokenizer-v1", b"whole-source-byte-span");
        let unicode = Digest::of_bytes("tokenizer-unicode-contract-v1", b"utf8-scalar-values");
        let added_tokens =
            Digest::of_serializable("ranked-byte-bpe-boundary-tokens-v1", &boundaries)
                .map_err(|error| TokenizationError::Identity(error.to_string()))?;
        let special_tokens = Digest::of_bytes("tokenizer-special-token-table-v1", b"empty");
        let policy_schema = Digest::of_bytes(
            "tokenization-policy-schema-v1",
            b"boundary-source-special-offset-count-through",
        );
        let tokenizer = Digest::of_serializable(
            "ranked-byte-bpe-tokenizer-v1",
            &(
                &vocabulary_identity,
                kernel.merges_identity(),
                &normalizer,
                &pretokenizer,
                &unicode,
                &added_tokens,
                &special_tokens,
                &policy_schema,
            ),
        )
        .map_err(|error| TokenizationError::Identity(error.to_string()))?;
        let identity_v2 = TokenizationIdentityV2 {
            model,
            tokenizer,
            vocabulary: vocabulary_identity,
            merges: kernel.merges_identity().clone(),
            normalizer,
            pretokenizer,
            unicode,
            added_tokens,
            special_tokens,
            implementation: kernel.implementation().clone(),
            policy_schema,
        };
        let identity = identity_v2.legacy_v1()?;
        Ok(Self {
            identity,
            identity_v2,
            byte_tokens: Box::new(byte_tokens),
            boundaries,
            kernel,
        })
    }

    /// Tokenizes through caller-owned reusable BPE scratch and output storage.
    ///
    /// # Errors
    ///
    /// Returns an input, cancellation, merge, boundary-policy, or sink error.
    pub fn tokenize_with_scratch(
        &self,
        source: &[u8],
        policy: &TokenizationPolicy,
        scratch: &mut BpeScratch,
        sink: &mut dyn TokenOutputSink,
        cancellation: &CancellationToken,
    ) -> Result<bool, TokenizationError> {
        validate_source(source)?;
        if source.len() > MAX_BPE_SYMBOLS {
            return Err(TokenizationError::Bound {
                field: "BPE initial symbols",
                limit: MAX_BPE_SYMBOLS,
            });
        }
        let beginning = match policy.boundary_tokens {
            BoundaryTokenPolicy::Beginning | BoundaryTokenPolicy::Both => {
                Some(self.boundaries.beginning.ok_or_else(|| {
                    TokenizationError::Invalid("beginning token is not configured".to_owned())
                })?)
            }
            BoundaryTokenPolicy::None | BoundaryTokenPolicy::Ending => None,
        };
        let ending = match policy.boundary_tokens {
            BoundaryTokenPolicy::Ending | BoundaryTokenPolicy::Both => {
                Some(self.boundaries.ending.ok_or_else(|| {
                    TokenizationError::Invalid("ending token is not configured".to_owned())
                })?)
            }
            BoundaryTokenPolicy::None | BoundaryTokenPolicy::Beginning => None,
        };
        let source_end = u32::try_from(source.len()).map_err(|_| TokenizationError::Bound {
            field: "source bytes",
            limit: crate::MAX_ROW_BYTES,
        })?;
        cancellation.check()?;
        scratch.symbols.clear();
        scratch.symbols.reserve(source.len());
        for (offset, byte) in source.iter().copied().enumerate() {
            let start = u32::try_from(offset).map_err(|_| TokenizationError::Bound {
                field: "source bytes",
                limit: crate::MAX_ROW_BYTES,
            })?;
            scratch.symbols.push(TokenSpan {
                token: self.byte_tokens[usize::from(byte)],
                start,
                end: start + 1,
            });
        }
        scratch.peak_symbols = scratch.peak_symbols.max(source.len());
        self.kernel.merge_prepared(scratch, cancellation)?;
        cancellation.check()?;
        sink.begin()?;
        if let Some(token) = beginning
            && sink.push(TokenSpan {
                token,
                start: 0,
                end: 0,
            })? == SinkFlow::Stop
        {
            return Ok(false);
        }
        for symbol in scratch.symbols.iter().copied() {
            let symbol = if policy.offsets == OffsetPolicy::Omit {
                TokenSpan {
                    token: symbol.token,
                    start: 0,
                    end: 0,
                }
            } else {
                symbol
            };
            if sink.push(symbol)? == SinkFlow::Stop {
                return Ok(false);
            }
        }
        if let Some(token) = ending
            && sink.push(TokenSpan {
                token,
                start: source_end,
                end: source_end,
            })? == SinkFlow::Stop
        {
            return Ok(false);
        }
        Ok(true)
    }

    /// Counts through caller-owned reusable BPE scratch without retaining a
    /// complete output token vector.
    ///
    /// `policy.count_through` is inclusive: the incomplete result reports the
    /// first count greater than that threshold.
    ///
    /// # Errors
    ///
    /// Returns an input, cancellation, merge, boundary-policy, or counting
    /// error.
    pub fn count_with_scratch(
        &self,
        source: &[u8],
        policy: &TokenizationPolicy,
        scratch: &mut BpeScratch,
        cancellation: &CancellationToken,
    ) -> Result<CountResult, TokenizationError> {
        let mut sink = CountingSink::new(policy.count_through);
        let consumed =
            self.tokenize_with_scratch(source, policy, scratch, &mut sink, cancellation)?;
        if consumed {
            sink.finish();
        }
        Ok(sink.result())
    }
}

impl ExactTokenizer for RankedByteBpe {
    fn identity(&self) -> &TokenizationIdentity {
        &self.identity
    }

    fn identity_v2(&self) -> Option<&TokenizationIdentityV2> {
        Some(&self.identity_v2)
    }

    fn tokenize_into(
        &self,
        source: &[u8],
        policy: &TokenizationPolicy,
        output: &mut Vec<TokenSpan>,
        cancellation: &CancellationToken,
    ) -> Result<(), TokenizationError> {
        let mut scratch = BpeScratch::new();
        let maximum = MAX_TOKENS_PER_ROW
            .min(source.len().saturating_add(2))
            .max(1);
        let mut sink = VecTokenSink::new(output, maximum)?;
        let complete =
            self.tokenize_with_scratch(source, policy, &mut scratch, &mut sink, cancellation)?;
        if !complete {
            return Err(TokenizationError::Invalid(
                "vector output unexpectedly requested early stop".to_owned(),
            ));
        }
        validate_spans(source.len(), output)
    }

    fn tokenize_to_sink(
        &self,
        source: &[u8],
        policy: &TokenizationPolicy,
        sink: &mut dyn TokenOutputSink,
        cancellation: &CancellationToken,
    ) -> Result<bool, TokenizationError> {
        let mut scratch = BpeScratch::new();
        self.tokenize_with_scratch(source, policy, &mut scratch, sink, cancellation)
    }

    fn count(
        &self,
        source: &[u8],
        policy: &TokenizationPolicy,
        cancellation: &CancellationToken,
    ) -> Result<CountResult, TokenizationError> {
        let mut scratch = BpeScratch::new();
        self.count_with_scratch(source, policy, &mut scratch, cancellation)
    }
}

fn pair_key(left: TokenId, right: TokenId) -> u64 {
    let left = u32::try_from(left.get()).unwrap_or_default();
    let right = u32::try_from(right.get()).unwrap_or_default();
    (u64::from(left) << 32) | u64::from(right)
}

fn simd_min_packed(values: [u32; SHORT_MERGE_MAX]) -> u32 {
    let left = u32x8::from([
        values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7],
    ]);
    let right = u32x8::from([
        values[8], values[9], values[10], values[11], values[12], values[13], values[14],
        values[15],
    ]);
    left.min(right)
        .to_array()
        .into_iter()
        .min()
        .unwrap_or(u32::MAX)
}

fn compact_short(symbols: &mut Vec<TokenSpan>, next: &[u8; SHORT_MERGE_MAX], count: usize) {
    let mut write = 0;
    let mut index = 0;
    while index < count {
        symbols[write] = symbols[index];
        write += 1;
        index = usize::from(next[index]);
    }
    symbols.truncate(write);
}

fn compact_heap(symbols: &mut Vec<TokenSpan>, next: &[usize]) {
    let mut write = 0;
    let mut index = 0;
    loop {
        symbols[write] = symbols[index];
        write += 1;
        if next[index] == NONE {
            break;
        }
        index = next[index];
    }
    symbols.truncate(write);
}

fn validate_initial_symbols(initial: &[TokenSpan]) -> Result<(), TokenizationError> {
    if initial.is_empty() || initial.len() > MAX_BPE_SYMBOLS {
        return Err(TokenizationError::Bound {
            field: "BPE initial symbols",
            limit: MAX_BPE_SYMBOLS,
        });
    }
    if initial.iter().any(|symbol| symbol.start >= symbol.end)
        || initial.windows(2).any(|pair| pair[0].end != pair[1].start)
    {
        return Err(TokenizationError::Invalid(
            "BPE initial symbol offsets must form one contiguous span".to_owned(),
        ));
    }
    Ok(())
}

fn derive_vocabulary(
    byte_tokens: &[TokenId; 256],
    merges: &[BpeMerge],
) -> Result<Vec<(i32, Vec<u8>)>, TokenizationError> {
    let byte_root_ids = byte_tokens
        .iter()
        .map(|token| token.get())
        .collect::<HashSet<_>>();
    let mut vocabulary = BTreeMap::<i32, Vec<u8>>::new();
    for (byte, token) in byte_tokens.iter().copied().enumerate() {
        let byte = u8::try_from(byte)
            .map_err(|_| TokenizationError::Invalid("byte root exceeds u8".to_owned()))?;
        if !valid_utf8_stream_byte(byte) {
            continue;
        }
        if vocabulary.insert(token.get(), vec![byte]).is_some() {
            return Err(TokenizationError::Invalid(
                "valid UTF-8 byte roots repeat a token ID".to_owned(),
            ));
        }
    }
    for rule in merges {
        if byte_root_ids.contains(&rule.merged.get()) {
            return Err(TokenizationError::Invalid(
                "BPE merged token conflicts with a byte root".to_owned(),
            ));
        }
        let left = vocabulary.get(&rule.left.get()).cloned().ok_or_else(|| {
            TokenizationError::Invalid("BPE merge left token has no reconstructed bytes".to_owned())
        })?;
        let right = vocabulary.get(&rule.right.get()).ok_or_else(|| {
            TokenizationError::Invalid(
                "BPE merge right token has no reconstructed bytes".to_owned(),
            )
        })?;
        let mut bytes = left;
        bytes.extend_from_slice(right);
        if let Some(existing) = vocabulary.insert(rule.merged.get(), bytes.clone())
            && existing != bytes
        {
            return Err(TokenizationError::Invalid(
                "BPE merged token has conflicting reconstructed bytes".to_owned(),
            ));
        }
    }
    Ok(vocabulary.into_iter().collect())
}

const fn valid_utf8_stream_byte(byte: u8) -> bool {
    !matches!(byte, 0xc0 | 0xc1 | 0xf5..=0xff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SourceSpecialTokenPolicy;

    fn token(value: i32) -> TokenId {
        TokenId::new(value).unwrap()
    }

    fn byte_tokens() -> [TokenId; 256] {
        std::array::from_fn(|byte| token(i32::try_from(byte).unwrap()))
    }

    fn kernel() -> RankedBpe {
        RankedBpe::new(vec![
            BpeMerge {
                left: token(b'a'.into()),
                right: token(b'b'.into()),
                merged: token(256),
                rank: 0,
            },
            BpeMerge {
                left: token(256),
                right: token(b'c'.into()),
                merged: token(257),
                rank: 1,
            },
        ])
        .unwrap()
    }

    fn differential_kernel() -> RankedBpe {
        RankedBpe::new(vec![
            BpeMerge {
                left: token(b'a'.into()),
                right: token(b'b'.into()),
                merged: token(256),
                rank: 0,
            },
            BpeMerge {
                left: token(b'b'.into()),
                right: token(b'c'.into()),
                merged: token(257),
                rank: 1,
            },
            BpeMerge {
                left: token(256),
                right: token(b'c'.into()),
                merged: token(258),
                rank: 2,
            },
            BpeMerge {
                left: token(b'a'.into()),
                right: token(257),
                merged: token(259),
                rank: 3,
            },
        ])
        .unwrap()
    }

    fn policy(offsets: OffsetPolicy, count_through: Option<u64>) -> TokenizationPolicy {
        TokenizationPolicy {
            boundary_tokens: BoundaryTokenPolicy::None,
            source_special_tokens: SourceSpecialTokenPolicy::OrdinaryText,
            offsets,
            count_through,
        }
    }

    #[test]
    fn ranked_merges_follow_priority_and_preserve_offsets() {
        let tokenizer = RankedByteBpe::new(
            Digest::of_bytes("model", b"one"),
            byte_tokens(),
            BpeBoundaryTokens::default(),
            kernel(),
        )
        .unwrap();
        let mut output = Vec::new();
        tokenizer
            .tokenize_into(
                b"zabc!",
                &policy(OffsetPolicy::Include, None),
                &mut output,
                &CancellationToken::default(),
            )
            .unwrap();
        assert_eq!(
            output,
            vec![
                TokenSpan {
                    token: token(b'z'.into()),
                    start: 0,
                    end: 1,
                },
                TokenSpan {
                    token: token(257),
                    start: 1,
                    end: 4,
                },
                TokenSpan {
                    token: token(b'!'.into()),
                    start: 4,
                    end: 5,
                },
            ]
        );
    }

    #[test]
    fn direct_threshold_count_does_not_require_token_output() {
        let tokenizer = RankedByteBpe::new(
            Digest::of_bytes("model", b"one"),
            byte_tokens(),
            BpeBoundaryTokens::default(),
            kernel(),
        )
        .unwrap();
        let mut scratch = BpeScratch::new();
        let result = tokenizer
            .count_with_scratch(
                b"zabc!",
                &policy(OffsetPolicy::Omit, Some(1)),
                &mut scratch,
                &CancellationToken::default(),
            )
            .unwrap();
        assert_eq!(
            result,
            CountResult {
                count: 2,
                complete: false,
            }
        );
    }

    #[test]
    fn reusable_scratch_retains_capacity_and_bounds_before_sink_mutation() {
        let tokenizer = RankedByteBpe::new(
            Digest::of_bytes("model", b"one"),
            byte_tokens(),
            BpeBoundaryTokens::default(),
            kernel(),
        )
        .unwrap();
        let mut scratch = BpeScratch::new();
        let mut count = CountingSink::new(None);
        tokenizer
            .tokenize_with_scratch(
                b"abcabcabcabcabcabc",
                &policy(OffsetPolicy::Omit, None),
                &mut scratch,
                &mut count,
                &CancellationToken::default(),
            )
            .unwrap();
        let capacity = scratch.symbol_capacity();
        tokenizer
            .tokenize_with_scratch(
                b"a",
                &policy(OffsetPolicy::Omit, None),
                &mut scratch,
                &mut count,
                &CancellationToken::default(),
            )
            .unwrap();
        assert_eq!(scratch.symbol_capacity(), capacity);
        assert_eq!(scratch.peak_symbols(), 18);

        let sentinel = TokenSpan {
            token: token(999),
            start: 0,
            end: 0,
        };
        let mut output = vec![sentinel];
        let mut sink = VecTokenSink::new(&mut output, 4).unwrap();
        let mut invalid_policy = policy(OffsetPolicy::Include, None);
        invalid_policy.boundary_tokens = BoundaryTokenPolicy::Ending;
        assert!(
            tokenizer
                .tokenize_with_scratch(
                    b"abc",
                    &invalid_policy,
                    &mut scratch,
                    &mut sink,
                    &CancellationToken::default(),
                )
                .is_err()
        );
        assert_eq!(output, vec![sentinel]);
    }

    #[test]
    fn short_and_heap_paths_match_on_repeated_fixture() {
        let kernel = kernel();
        let mut short_initial = Vec::new();
        for (index, byte) in b"abcabcabc".iter().copied().enumerate() {
            short_initial.push(TokenSpan {
                token: token(byte.into()),
                start: u32::try_from(index).unwrap(),
                end: u32::try_from(index + 1).unwrap(),
            });
        }
        let mut long_initial = short_initial.clone();
        long_initial.extend(short_initial.iter().copied().map(|mut span| {
            span.start += 9;
            span.end += 9;
            span
        }));
        let mut scratch = BpeScratch::new();
        let mut short = Vec::new();
        let mut sink = VecTokenSink::new(&mut short, 32).unwrap();
        kernel
            .merge_to_sink(
                &short_initial,
                OffsetPolicy::Include,
                &mut scratch,
                &mut sink,
                &CancellationToken::default(),
            )
            .unwrap();
        let mut long = Vec::new();
        let mut sink = VecTokenSink::new(&mut long, 32).unwrap();
        kernel
            .merge_to_sink(
                &long_initial,
                OffsetPolicy::Include,
                &mut scratch,
                &mut sink,
                &CancellationToken::default(),
            )
            .unwrap();
        assert_eq!(short.len(), 3);
        assert_eq!(long.len(), 6);
        assert_eq!(
            long.iter().map(|span| span.token).collect::<Vec<_>>(),
            short
                .iter()
                .chain(&short)
                .map(|span| span.token)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn cancellation_is_checked_at_merge_boundaries() {
        let initial = (0_u32..20)
            .map(|index| TokenSpan {
                token: token(b'a'.into()),
                start: index,
                end: index + 1,
            })
            .collect::<Vec<_>>();
        let cancellation = CancellationToken::default();
        cancellation.request();
        let mut scratch = BpeScratch::new();
        let mut output = Vec::new();
        let mut sink = VecTokenSink::new(&mut output, 32).unwrap();
        assert_eq!(
            kernel().merge_to_sink(
                &initial,
                OffsetPolicy::Include,
                &mut scratch,
                &mut sink,
                &cancellation,
            ),
            Err(TokenizationError::Cancelled)
        );
        assert!(output.is_empty());
    }

    #[test]
    fn packed_short_and_heap_kernels_match_naive_ranked_bpe() {
        let kernel = differential_kernel();
        for length in 2..=80 {
            let source = (0..length)
                .map(|index| b"abc"[(index * 7 + length * 3) % 3])
                .collect::<Vec<_>>();
            let initial = source
                .iter()
                .copied()
                .enumerate()
                .map(|(index, byte)| TokenSpan {
                    token: token(byte.into()),
                    start: u32::try_from(index).unwrap(),
                    end: u32::try_from(index + 1).unwrap(),
                })
                .collect::<Vec<_>>();
            let expected = naive_merge(&initial, kernel.rules());
            let mut actual = Vec::new();
            let mut scratch = BpeScratch::new();
            let mut sink = VecTokenSink::new(&mut actual, 128).unwrap();
            kernel
                .merge_to_sink(
                    &initial,
                    OffsetPolicy::Include,
                    &mut scratch,
                    &mut sink,
                    &CancellationToken::default(),
                )
                .unwrap();
            assert_eq!(actual, expected, "source length {length}");
        }
    }

    #[test]
    fn split_identity_and_boundary_policy_are_explicit() {
        let tokenizer = RankedByteBpe::new(
            Digest::of_bytes("model", b"one"),
            byte_tokens(),
            BpeBoundaryTokens {
                beginning: Some(token(300)),
                ending: Some(token(301)),
            },
            kernel(),
        )
        .unwrap();
        let identity_v2 = tokenizer.identity_v2().unwrap();
        assert_ne!(identity_v2.normalizer, identity_v2.pretokenizer);
        assert_eq!(identity_v2.legacy_v1().unwrap(), *tokenizer.identity());
        let encoded = serde_json::to_vec(identity_v2).unwrap();
        assert_eq!(
            serde_json::from_slice::<TokenizationIdentityV2>(&encoded).unwrap(),
            *identity_v2
        );
        let mut changed_specials = identity_v2.clone();
        changed_specials.special_tokens =
            Digest::of_bytes("tokenizer-special-token-table-v1", b"changed");
        assert_ne!(
            changed_specials.legacy_v1().unwrap(),
            identity_v2.legacy_v1().unwrap()
        );

        let mut output = Vec::new();
        let mut boundary_policy = policy(OffsetPolicy::Include, None);
        boundary_policy.boundary_tokens = BoundaryTokenPolicy::Both;
        tokenizer
            .tokenize_into(
                b"ab",
                &boundary_policy,
                &mut output,
                &CancellationToken::default(),
            )
            .unwrap();
        assert_eq!(output.first().unwrap().token, token(300));
        assert_eq!(output.last().unwrap().token, token(301));
        assert_eq!((output[1].start, output[1].end), (0, 2));
    }

    #[test]
    fn byte_root_identity_rejects_token_aliases() {
        let mut roots = byte_tokens();
        roots[1] = roots[0];
        assert!(
            RankedByteBpe::new(
                Digest::of_bytes("model", b"one"),
                roots,
                BpeBoundaryTokens::default(),
                kernel(),
            )
            .is_err()
        );
    }

    #[test]
    fn multilingual_utf8_offsets_reconstruct_exact_source_bytes() {
        let tokenizer = RankedByteBpe::new(
            Digest::of_bytes("model", b"one"),
            byte_tokens(),
            BpeBoundaryTokens::default(),
            kernel(),
        )
        .unwrap();
        let source = "naïve 猫 🧪".as_bytes();
        let mut output = Vec::new();
        tokenizer
            .tokenize_into(
                source,
                &policy(OffsetPolicy::Include, None),
                &mut output,
                &CancellationToken::default(),
            )
            .unwrap();
        let reconstructed = output
            .iter()
            .flat_map(|span| {
                source[usize::try_from(span.start).unwrap()..usize::try_from(span.end).unwrap()]
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        assert_eq!(reconstructed, source);
    }

    fn naive_merge(initial: &[TokenSpan], rules: &[BpeMerge]) -> Vec<TokenSpan> {
        let by_pair = rules
            .iter()
            .map(|rule| ((rule.left, rule.right), (rule.rank, rule.merged)))
            .collect::<HashMap<_, _>>();
        let mut symbols = initial.to_vec();
        loop {
            let next = symbols
                .windows(2)
                .enumerate()
                .filter_map(|(index, pair)| {
                    by_pair
                        .get(&(pair[0].token, pair[1].token))
                        .map(|(rank, merged)| (*rank, index, *merged))
                })
                .min_by_key(|(rank, index, _)| (*rank, *index));
            let Some((_rank, index, merged)) = next else {
                break;
            };
            symbols[index].token = merged;
            symbols[index].end = symbols[index + 1].end;
            symbols.remove(index + 1);
        }
        symbols
    }
}

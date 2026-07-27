<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Logit Loom Tokenizer

`logit-loom-tokenizer` provides safe, bounded mechanics shared by bulk
tokenizer implementations:

- versioned tokenizer, normalizer, pretokenizer, vocabulary, merge,
  special-token, implementation, and policy identities;
- a ranked BPE merge kernel with a packed `u32x8` short-span scan and a
  deterministic heap path for longer spans;
- an identity-normalized, whole-span UTF-8 byte-BPE adapter;
- reusable vector, caller-sized token-ID/offset, and count-only sinks;
- exact and inclusive-threshold counting without a complete output token
  vector;
- cooperative cancellation;
- stable batch scatter and length buckets;
- token-aware source chunk planning;
- a caller-sized persistent pool with a bounded queue and explicit
  backpressure;
- content-free qualification receipts for caller-supplied exact token-ID and
  offset oracles; and
- a byte-capped, frequency-admitted span cache that verifies exact bytes on
  every hit.

The crate does not select a model, device, thread count, batch size, cache
capacity, or fallback route. Those are deployment-policy decisions. It also
does not download tokenizers, open network connections, start services, or
run a model.

The direct count path reuses caller-owned merge storage and retains no complete
output token vector:

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use logit_loom_core::{Digest, TokenId};
use logit_loom_tokenizer::{
    BoundaryTokenPolicy, BpeBoundaryTokens, BpeMerge, BpeScratch,
    CancellationToken, OffsetPolicy, RankedBpe, RankedByteBpe,
    SourceSpecialTokenPolicy, TokenizationPolicy,
};

let byte_tokens = std::array::from_fn(|byte| {
    TokenId::new(i32::try_from(byte).expect("byte fits i32"))
        .expect("byte token is non-negative")
});
let merges = RankedBpe::new(vec![BpeMerge {
    left: TokenId::new(i32::from(b'a'))?,
    right: TokenId::new(i32::from(b'b'))?,
    merged: TokenId::new(256)?,
    rank: 0,
}])?;
let tokenizer = RankedByteBpe::new(
    Digest::of_bytes("example-model", b"exact-model-bytes"),
    byte_tokens,
    BpeBoundaryTokens::default(),
    merges,
)?;
let policy = TokenizationPolicy {
    boundary_tokens: BoundaryTokenPolicy::None,
    source_special_tokens: SourceSpecialTokenPolicy::OrdinaryText,
    offsets: OffsetPolicy::Omit,
    count_through: None,
};
let mut scratch = BpeScratch::new();
let count = tokenizer.count_with_scratch(
    b"ab!",
    &policy,
    &mut scratch,
    &CancellationToken::default(),
)?;
assert_eq!(count.count, 2);
# Ok(())
# }
```

`RankedByteBpe` deliberately performs no normalization or model-specific
pretokenization and has no configured source-special-token spellings. A
tokenizer that uses regex splitting, metaspace, added-token recognition, or
another normalization contract needs its own exact adapter and must pass
caller-supplied engine-oracle qualification. The generic oracle helper does
not make that parity claim by itself.

This workspace crate is not currently published. The safe kernel, sinks, pool,
cache, batching, and qualification framework are implemented and tested
model-free. Exact adapters and retained differential qualification for the
intended production tokenizers, plus whole/partitioned stress coverage, remain
open in `NEXT_STEPS.md`.

The ranked merge mechanics and flat-batch direction are a safe Rust
reimplementation derived from Gigatoken 0.9.0 at commit
`0d9765fa7312af7534535e6315a5c49d74807b2a`. The implementation uses the
safe, exactly pinned `wide` 0.7.33 abstraction; it does not import upstream
unsafe intrinsics, unchecked indexing, prefetching, huge-page hints, `PyO3`,
`NumPy`, `Arrow`, `Rayon`, Hub, or runtime networking. See
`upstream/gigatoken.toml` and `upstream/GIGATOKEN-MIT.txt`.

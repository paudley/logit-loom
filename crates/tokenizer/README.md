<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Logit Loom Tokenizer

`logit-loom-tokenizer` provides safe, bounded mechanics shared by bulk
tokenizer implementations:

- exact tokenizer and policy identities;
- cooperative cancellation;
- stable batch scatter and length buckets;
- token-aware source chunk planning;
- count-only and early-stop operator contracts; and
- a byte-capped, frequency-admitted span cache that verifies exact bytes on
  every hit.

The crate does not select a model, device, thread count, batch size, cache
capacity, or fallback route. Those are deployment-policy decisions. It also
does not download tokenizers, open network connections, start services, or
run a model.

This workspace crate is not currently published. Its identity, cancellation,
batch, chunk, and cache primitives are available for review, while the pinned
SIMD BPE implementation and dedicated caller-sized pool remain explicit
follow-up work in `NEXT_STEPS.md`.

The design and flat-batch direction were informed by Gigatoken 0.9.0 at commit
`0d9765fa7312af7534535e6315a5c49d74807b2a`. No `PyO3`, `NumPy`, `Arrow`,
`Rayon`, Hub, or runtime-network surface is imported. See `upstream/gigatoken.toml`
and `upstream/GIGATOKEN-MIT.txt`.

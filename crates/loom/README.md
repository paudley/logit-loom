<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# logit-loom

Safe Rust machinery for composing token-stream transforms and observers.

The crate provides ordered transform pipelines, full or sparse candidate
exposure, transactional write-back, panic/error containment, observer fan-out,
cooperative cancellation, and mechanical receipts. It does not load a model.

```toml
[dependencies]
logit-loom = "=0.1.0-alpha.1"
```

## Example

Apply a first-party token bias to an in-memory vocabulary:

```rust
use logit_loom::{
    CandidateMode, Digest, Pipeline, Stage, TokenBias, TokenId, TransformSpec,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let target = TokenId::new(42)?;
    let transform = TokenBias::new([(target, 1.5)])?;
    let specification = TransformSpec::new(
        Digest::of_bytes("my-transform", b"token-42-bias-v1"),
        CandidateMode::FullVocabulary,
        16,
    )?;
    let mut pipeline = Pipeline::new(vec![Stage::new(specification, transform)?])?;

    pipeline.begin(&[])?;
    let mut logits = vec![0.0; 128];
    pipeline.apply_to_vocabulary(0, &[], &mut logits)?;

    assert_eq!(logits[42].to_bits(), 1.5_f32.to_bits());
    assert_eq!(pipeline.receipt().invocations, 1);
    Ok(())
}
```

Each call starts with `Pipeline::begin`. Successful transform invocations use
consecutive zero-based steps. A backend calls `Pipeline::accept` only after the
selected token has entered causal state, once for an unmatched successful
transform invocation.

## Failure and observation

- Candidate changes commit only after every stage succeeds.
- Callback errors and unwinding are contained and retained in receipts.
- Backend-selected candidate views must be non-empty, bounded, and contain
  unique token IDs.
- Observers are polled before sampling and see exact token bytes only after
  causal admission.

See the [workspace overview](https://github.com/paudley/logit-loom),
[getting-started guide](https://github.com/paudley/logit-loom/blob/main/docs/getting-started.md),
and [architecture](https://github.com/paudley/logit-loom/blob/main/docs/architecture.md).

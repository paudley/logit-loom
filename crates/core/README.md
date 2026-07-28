<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# logit-loom-core

Serializable, backend-neutral contracts for bounded token-stream mechanics.

This crate defines token identifiers, candidate exposure, sampling plans,
steering descriptors, checkpoints, bounded audio-prefill contracts,
topology-bound activation programs, target-authoritative speculation, and
structured-projection compiler/validator/feedback lineage and mechanical
receipts. It contains no model runtime or native code.

```toml
[dependencies]
logit-loom-core = "=0.2.0"
```

## Example

Build and validate a complete generation contract before handing it to an
adapter:

```rust
use logit_loom_core::{GenerationPlan, SamplingPlan};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let plan = GenerationPlan {
        sampling: SamplingPlan {
            seed: 7,
            temperature: 0.8,
            ..SamplingPlan::default()
        },
        max_tokens: 32,
        biases: Vec::new(),
        grammar: None,
        stops: vec![b"\n\n".to_vec()],
    };

    plan.validate()?;
    let identity = plan.digest()?;
    assert_eq!(identity.as_str().len(), 64);
    Ok(())
}
```

Plans and receipts implement `serde::Serialize` and `serde::Deserialize`.
Digest domains are explicit and versioned; changing a serialized shape or its
interpretation is a compatibility change.

## Boundaries

- Token pieces are arbitrary bytes, not necessarily standalone UTF-8.
- `AudioPrefillPlanV1` binds one projector identity, its declared sample rate,
  and a bounded caller-owned normalized mono PCM slice. Its receipt binds only
  a caller-provided audio identity and causal accounting; it never serializes
  PCM bytes.
- Public collections and native strings have documented bounds.
- `TextModelTopologyV1`, activation capture/vector/program contracts,
  `SpeculationPlanV1`, and their receipts describe exact mechanics and
  target/draft lineage without naming a native tensor pointer or model path.
- `TextMechanicsPlanV2` composes those mechanics, while
  `TextMechanicsCheckpointReceiptV2` binds an ordinary opaque state receipt to
  the exact aggregate mechanics and parent lineage which produced it.
- Provisional speculative telemetry resolves to admitted or rejected target
  positions before entering aggregate capture receipts.
- `StructuredProjectionPlanV1` binds semantic component identities without
  implementing their policy. Attempt receipts distinguish cancellation,
  authoritative rejection, and selected causal state.
- Receipts record mechanics and causal lineage, not model quality or semantic
  correctness.
- Checkpoint metadata binds an opaque backend state; it does not make that
  state portable across native builds.

See the [workspace overview](https://github.com/paudley/logit-loom),
[architecture](https://github.com/paudley/logit-loom/blob/main/docs/architecture.md),
and [compatibility policy](https://github.com/paudley/logit-loom/blob/main/docs/compatibility.md).

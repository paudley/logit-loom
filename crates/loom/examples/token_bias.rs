// SPDX-License-Identifier: MIT OR Apache-2.0

//! Applies one backend-neutral token-bias pipeline to an in-memory vocabulary.

use logit_loom::{CandidateMode, Digest, Pipeline, Stage, TokenBias, TokenId, TransformSpec};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let target = TokenId::new(42)?;
    let transform = TokenBias::new([(target, 1.5)])?;
    let specification = TransformSpec::new(
        Digest::of_bytes("example-transform", b"token-42-bias-v1"),
        CandidateMode::FullVocabulary,
        16,
    )?;
    let mut pipeline = Pipeline::new(vec![Stage::new(specification, transform)?])?;

    pipeline.begin(&[])?;
    let mut logits = vec![0.0; 128];
    pipeline.apply_to_vocabulary(0, &[], &mut logits)?;

    let target_index = usize::try_from(target.get())?;
    assert_eq!(logits[target_index].to_bits(), 1.5_f32.to_bits());
    println!("pipeline receipt: {}", pipeline.receipt().digest()?);
    Ok(())
}

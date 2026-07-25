// SPDX-License-Identifier: MIT OR Apache-2.0

//! Builds and exercises façade controls without loading a model.

use std::cell::RefCell;
use std::rc::Rc;

use logit_loom_runtime::{
    CandidateMode, ControlFlow, Digest, ObservedToken, ObserversBuilder, PipelineBuilder, TokenId,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut pipeline = PipelineBuilder::new(CandidateMode::FullVocabulary, 1)?
        .rank_bias(1, 2.0)?
        .build()?;
    pipeline.begin(&[])?;
    let mut logits = [3.0, 2.0, 1.0];
    pipeline.apply_to_vocabulary(0, &[], &mut logits)?;
    assert_eq!(
        logits.map(f32::to_bits),
        [3.0_f32.to_bits(), 4.0_f32.to_bits(), 1.0_f32.to_bits()]
    );

    let pieces = Rc::new(RefCell::new(Vec::new()));
    let sink = Rc::clone(&pieces);
    let mut observers = ObserversBuilder::new()
        .on_token(
            Digest::of_bytes("example-observer", b"collect-exact-bytes-v1"),
            move |token| {
                sink.borrow_mut().push(token.piece.to_vec());
                Ok(ControlFlow::Continue)
            },
        )?
        .build()?;
    observers.begin(0, 1)?;
    observers.poll()?;
    observers.observe(ObservedToken {
        token: TokenId::new(7)?,
        piece: &[0xff],
        position: 1,
    })?;
    assert_eq!(*pieces.borrow(), vec![vec![0xff]]);
    Ok(())
}

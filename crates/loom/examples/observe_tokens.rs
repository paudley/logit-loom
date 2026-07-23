// SPDX-License-Identifier: MIT OR Apache-2.0

//! Observes exact admitted token bytes without assuming UTF-8.

use logit_loom::{
    ControlFlow, Digest, ObservedToken, Observer, ObserverError, ObserverSet, TokenId,
};

#[derive(Default)]
struct ByteCounter;

impl Observer for ByteCounter {
    fn on_token(&mut self, _token: ObservedToken<'_>) -> Result<ControlFlow, ObserverError> {
        Ok(ControlFlow::Continue)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let implementation = Digest::of_bytes("example-observer", b"byte-counter-v1");
    let mut observers = ObserverSet::single(implementation, ByteCounter);
    observers.begin(10, 2)?;

    observers.poll()?;
    observers.observe(ObservedToken {
        token: TokenId::new(7)?,
        piece: &[0xff, 0x00],
        position: 11,
    })?;

    let receipt = observers
        .receipt(0)
        .ok_or("single observer receipt is missing")?;
    assert_eq!(receipt.observed_tokens, 1);
    assert_eq!(receipt.observed_bytes, 2);
    println!("observer receipt: {}", receipt.digest()?);
    Ok(())
}

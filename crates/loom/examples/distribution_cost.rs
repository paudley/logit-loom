// SPDX-License-Identifier: MIT OR Apache-2.0

//! Times `TokenDistribution::measure` on a Qwen3.6-sized vocabulary row.

use std::time::Instant;

use logit_loom::{TokenDistribution, TokenId};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vocabulary = 248_320_u32;
    let logits: Vec<f32> = (0..vocabulary)
        .map(|index| u16::try_from(index.wrapping_mul(2_654_435_761) >> 20).map(f32::from))
        .map(|scaled| scaled.map(|value| value * -0.004))
        .collect::<Result<_, _>>()?;
    let sampled = TokenId::new(12_345)?;
    let rounds = 2_000_u32;
    let mut sink = 0.0_f32;
    let started = Instant::now();
    for _ in 0..rounds {
        sink += TokenDistribution::measure(std::hint::black_box(&logits), sampled)?.entropy;
    }
    let micros = started.elapsed().as_secs_f64() * 1e6 / f64::from(rounds);
    println!("{micros:.1} us per {vocabulary}-entry row (checksum {sink:.1})");
    Ok(())
}

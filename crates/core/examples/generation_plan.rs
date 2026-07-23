// SPDX-License-Identifier: MIT OR Apache-2.0

//! Builds, validates, serializes, and identifies a generation plan.

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
    println!("{}", serde_json::to_string_pretty(&plan)?);
    println!("plan digest: {}", plan.digest()?);
    Ok(())
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! Parameterized MTP probe for staged hardware-risk testing.
//!
//! Extends `speculative_mtp` with explicit control over device placement and
//! context/draft geometry so the MTP graph can be validated CPU-only first,
//! then escalated on an accelerator one bounded step at a time.
//!
//! ```text
//! mtp_probe MODEL.gguf PROMPT [--cpu] [--ctx N] [--batch N] [--draft N]
//!           [--max-tokens N] [--threads N]
//! ```

use std::io::{self, Write as _};
use std::num::NonZeroU32;

use logit_loom::{
    GenerationPlan, SamplingPlan, SpeculationActivationPolicyV1, SpeculationPlanV1,
    TextSpeculativeMechanismV1,
};
use logit_loom_llamacpp::{
    DevicePolicy, Model, ModelOptions, Runtime, SessionOptions, SpeculativeRequest,
    SpeculativeSessionOptions, Tokenization, generate_speculative,
    speculation_implementation_identity,
};

const USAGE: &str = "usage: mtp_probe MODEL.gguf PROMPT [--cpu] [--ctx N] [--batch N] \
                     [--draft N] [--max-tokens N] [--threads N]";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let model_path = arguments.next().ok_or(USAGE)?;
    let prompt_text = arguments.next().ok_or(USAGE)?;

    let mut cpu = false;
    let mut verbose = false;
    let mut context_size = 4_096_u32;
    let mut batch_size = 512_u32;
    let mut micro_batch_size = 0_u32;
    let mut draft_tokens = 4_u32;
    let mut max_tokens = 64_u32;
    let mut threads = 4_i32;
    while let Some(flag) = arguments.next() {
        let mut value = |name: &str| -> Result<String, Box<dyn std::error::Error>> {
            arguments
                .next()
                .ok_or_else(|| format!("{name} needs a value; {USAGE}").into())
        };
        match flag.as_str() {
            "--cpu" => cpu = true,
            "--verbose" => verbose = true,
            "--ctx" => context_size = value("--ctx")?.parse()?,
            "--batch" => batch_size = value("--batch")?.parse()?,
            "--micro" => micro_batch_size = value("--micro")?.parse()?,
            "--draft" => draft_tokens = value("--draft")?.parse()?,
            "--max-tokens" => max_tokens = value("--max-tokens")?.parse()?,
            "--threads" => threads = value("--threads")?.parse()?,
            other => return Err(format!("unknown flag {other}; {USAGE}").into()),
        }
    }
    let batch_size = batch_size.min(context_size);
    let micro_batch_size = if micro_batch_size == 0 {
        batch_size
    } else {
        micro_batch_size.min(batch_size)
    };
    let session = SessionOptions {
        context_size: NonZeroU32::new(context_size).ok_or("--ctx must be nonzero")?,
        batch_size,
        micro_batch_size,
        threads,
        ..SessionOptions::default()
    };
    let model_options = if cpu {
        ModelOptions {
            gpu_layers: 0,
            main_gpu: 0,
            device_policy: DevicePolicy::Any,
        }
    } else {
        ModelOptions::default()
    };
    eprintln!(
        "mtp_probe: placement={} ctx={context_size} batch={batch_size} draft={draft_tokens} \
         max_tokens={max_tokens} threads={threads}",
        if cpu { "cpu" } else { "accelerator" }
    );

    let mut runtime = Runtime::initialize()?;
    if !verbose {
        runtime.silence_native_logs();
    }
    let model = Model::load(&runtime, model_path, model_options)?;
    let prompt = model.tokenize(&prompt_text, Tokenization { add_bos: true })?;
    let generation = GenerationPlan {
        sampling: SamplingPlan::default(),
        max_tokens,
        biases: Vec::new(),
        grammar: None,
        stops: Vec::new(),
    };
    let topology = model.topology().digest()?;
    let speculation = SpeculationPlanV1 {
        target_model: model.artifact_digest().clone(),
        target_topology: topology.clone(),
        draft_model: model.artifact_digest().clone(),
        draft_topology: topology,
        implementation: speculation_implementation_identity(),
        mechanism: TextSpeculativeMechanismV1::Mtp,
        sequences: 1,
        maximum_draft_tokens: draft_tokens,
        minimum_draft_tokens: 0,
        probability_floor_bits: 0.0_f32.to_bits(),
        activation: SpeculationActivationPolicyV1::None,
    };
    let request = SpeculativeRequest::new(&prompt, &generation, &speculation).with_options(
        SpeculativeSessionOptions {
            target: session,
            draft: session,
        },
    );
    let output = generate_speculative(&runtime, &model, &model, request)?;

    io::stdout().lock().write_all(&output.generation.bytes)?;
    eprintln!(
        "\n{} boundaries: {} proposed, {} accepted, {} rejected",
        output.speculation.boundaries.len(),
        output.speculation.proposed,
        output.speculation.accepted,
        output.speculation.rejected
    );
    Ok(())
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! Runs one target-authoritative MTP experiment with a caller-supplied model.

use std::io::{self, Write as _};

use logit_loom::{
    GenerationPlan, SamplingPlan, SpeculationActivationPolicyV1, SpeculationPlanV1,
    TextSpeculativeMechanismV1,
};
use logit_loom_llamacpp::{
    Model, ModelOptions, Runtime, SpeculativeRequest, Tokenization, generate_speculative,
    speculation_implementation_identity,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let model_path = arguments
        .next()
        .ok_or("usage: speculative_mtp MTP_MODEL.gguf PROMPT")?;
    let prompt = arguments
        .next()
        .ok_or("usage: speculative_mtp MTP_MODEL.gguf PROMPT")?
        .into_string()
        .map_err(|_| "prompt must be valid UTF-8")?;
    if arguments.next().is_some() {
        return Err("usage: speculative_mtp MTP_MODEL.gguf PROMPT".into());
    }

    let mut runtime = Runtime::initialize()?;
    runtime.silence_native_logs();
    let model = Model::load(&runtime, model_path, ModelOptions::default())?;
    let prompt = model.tokenize(&prompt, Tokenization { add_bos: true })?;
    let generation = GenerationPlan {
        sampling: SamplingPlan::default(),
        max_tokens: 64,
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
        maximum_draft_tokens: 4,
        minimum_draft_tokens: 0,
        probability_floor_bits: 0.0_f32.to_bits(),
        activation: SpeculationActivationPolicyV1::None,
    };
    let output = generate_speculative(
        &runtime,
        &model,
        &model,
        SpeculativeRequest::new(&prompt, &generation, &speculation),
    )?;

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

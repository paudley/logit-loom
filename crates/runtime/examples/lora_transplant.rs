// SPDX-License-Identifier: MIT OR Apache-2.0

//! Applies and clears one `LoRA` between checkpointed deterministic replays.

mod support;

use logit_loom_runtime::{
    CheckpointReceipt, GenerationPlan, GenerationRequest, Loom, LoomOptions, LoraSpec,
    NativeLogPolicy, SteeringReceipt, Tokenization,
};
use serde::Serialize;
use support::{AdmissionRecord, GenerationRecord, RunMetadata};

const GENERATED_TOKENS: u32 = 32;
const LORA_SCALE: f32 = 1.0;

#[derive(Debug, Serialize)]
struct Report {
    scenario: &'static str,
    metadata: RunMetadata,
    admission: AdmissionRecord,
    checkpoint: CheckpointReceipt,
    generation_plan: GenerationPlan,
    steering: LoraSpec,
    baseline: GenerationRecord,
    steered: GenerationRecord,
    replay: GenerationRecord,
    steering_applied: SteeringReceipt,
    steering_cleared: SteeringReceipt,
    baseline_replay_matches: bool,
    session_healthy_after_clear: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let model_path = arguments
        .next()
        .ok_or("usage: lora_transplant MODEL.gguf ADAPTER.gguf PROMPT")?;
    let adapter_path = arguments
        .next()
        .ok_or("usage: lora_transplant MODEL.gguf ADAPTER.gguf PROMPT")?;
    let prompt = arguments
        .next()
        .ok_or("usage: lora_transplant MODEL.gguf ADAPTER.gguf PROMPT")?
        .into_string()
        .map_err(|_| "prompt must be valid UTF-8")?;
    if arguments.next().is_some() {
        return Err("usage: lora_transplant MODEL.gguf ADAPTER.gguf PROMPT".into());
    }

    let loom = Loom::load(
        model_path,
        LoomOptions {
            native_logs: NativeLogPolicy::Silence,
            ..LoomOptions::default()
        },
    )?;
    let metadata = RunMetadata::capture(&loom);
    let mut adapter = loom.load_lora(adapter_path)?;
    let steering = LoraSpec {
        artifact: adapter.artifact_digest().clone(),
        scale: LORA_SCALE,
    };
    let mut session = loom.session()?;
    let admission = session.replace_text(&prompt, Tokenization { add_bos: true })?;
    let checkpoint = session.capture_state()?;

    let generation_plan = GenerationRequest::new(GENERATED_TOKENS)?.plan().clone();
    let baseline = session.generate(GenerationRequest::from_plan(generation_plan.clone())?)?;
    session.restore_state(&checkpoint)?;

    let (steering_applied, steered, steering_cleared) = {
        let mut steered_session = session.lora(&mut adapter, LORA_SCALE)?;
        let steering_applied = steered_session.applied_receipt().clone();
        let steered =
            steered_session.generate(GenerationRequest::from_plan(generation_plan.clone())?)?;
        let steering_cleared = steered_session.clear()?;
        (steering_applied, steered, steering_cleared)
    };
    let session_healthy_after_clear = session.is_healthy();

    session.restore_state(&checkpoint)?;
    let replay = session.generate(GenerationRequest::from_plan(generation_plan.clone())?)?;
    let baseline_replay_matches = baseline == replay;

    support::write_json(&Report {
        scenario: "lora_transplant",
        metadata,
        admission: admission.into(),
        checkpoint: checkpoint.receipt().clone(),
        generation_plan,
        steering,
        baseline: baseline.try_into()?,
        steered: steered.try_into()?,
        replay: replay.try_into()?,
        steering_applied,
        steering_cleared,
        baseline_replay_matches,
        session_healthy_after_clear,
    })
}

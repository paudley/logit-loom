// SPDX-License-Identifier: MIT OR Apache-2.0

//! Verifies the pinned small Qwen profile and replays one exact checkpoint.

mod support;

use logit_loom_runtime::{
    CheckpointReceipt, GenerationPlan, GenerationRequest, Loom, LoomOptions, NativeLogPolicy,
    Tokenization,
};
use serde::Serialize;
use support::{AdmissionRecord, GenerationRecord, RunMetadata};

const GENERATED_TOKENS: u32 = 16;

#[derive(Debug, Serialize)]
struct Report {
    scenario: &'static str,
    metadata: RunMetadata,
    input_bytes_hex: String,
    admission: AdmissionRecord,
    checkpoint: CheckpointReceipt,
    generation_plan: GenerationPlan,
    first: GenerationRecord,
    replay: GenerationRecord,
    exact_replay: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let model_path = arguments
        .next()
        .ok_or("usage: qwen_profile_replay MODEL.gguf PROMPT")?;
    let prompt = arguments
        .next()
        .ok_or("usage: qwen_profile_replay MODEL.gguf PROMPT")?
        .into_string()
        .map_err(|_| "prompt must be valid UTF-8")?;
    if arguments.next().is_some() {
        return Err("usage: qwen_profile_replay MODEL.gguf PROMPT".into());
    }

    let loom = Loom::load_qwen3_small(
        model_path,
        LoomOptions {
            native_logs: NativeLogPolicy::Silence,
            ..LoomOptions::default()
        },
    )?;
    let metadata = RunMetadata::capture(&loom);
    let input_bytes_hex = support::encode_hex(prompt.as_bytes());
    let mut session = loom.session()?;
    let admission = session.replace_text(&prompt, Tokenization { add_bos: true })?;
    let checkpoint = session.capture_state()?;
    let generation_plan = GenerationRequest::new(GENERATED_TOKENS)?.plan().clone();
    let first = session.generate(GenerationRequest::from_plan(generation_plan.clone())?)?;
    session.restore_state(&checkpoint)?;
    let replay = session.generate(GenerationRequest::from_plan(generation_plan.clone())?)?;
    let exact_replay = first.bytes == replay.bytes
        && first.tokens == replay.tokens
        && first.receipt == replay.receipt;

    support::write_json(&Report {
        scenario: "qwen_profile_replay",
        metadata,
        input_bytes_hex,
        admission: admission.into(),
        checkpoint: checkpoint.receipt().clone(),
        generation_plan,
        first: first.try_into()?,
        replay: replay.try_into()?,
        exact_replay,
    })
}

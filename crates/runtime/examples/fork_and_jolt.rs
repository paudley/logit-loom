// SPDX-License-Identifier: MIT OR Apache-2.0

//! Replays one checkpoint with and without a runner-up logit jolt.

mod support;

use logit_loom_runtime::{
    CandidateMode, CheckpointReceipt, GenerationPlan, GenerationRequest, Loom, LoomOptions,
    NativeLogPolicy, PipelineBuilder, PipelineReceipt, Tokenization,
};
use serde::Serialize;
use support::{AdmissionRecord, GenerationRecord, RunMetadata};

const GENERATED_TOKENS: u32 = 32;
const JOLTED_RANK: usize = 1;
const JOLT: f32 = 4.0;

#[derive(Debug, Serialize)]
struct Report {
    scenario: &'static str,
    metadata: RunMetadata,
    admission: AdmissionRecord,
    checkpoint: CheckpointReceipt,
    generation_plan: GenerationPlan,
    jolt: JoltConfiguration,
    baseline: GenerationRecord,
    jolted: GenerationRecord,
    outputs_differ: bool,
    transform: PipelineReceipt,
}

#[derive(Debug, Serialize)]
struct JoltConfiguration {
    zero_based_rank: u64,
    additive_bias: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let model_path = arguments
        .next()
        .ok_or("usage: fork_and_jolt MODEL.gguf PROMPT")?;
    let prompt = arguments
        .next()
        .ok_or("usage: fork_and_jolt MODEL.gguf PROMPT")?
        .into_string()
        .map_err(|_| "prompt must be valid UTF-8")?;
    if arguments.next().is_some() {
        return Err("usage: fork_and_jolt MODEL.gguf PROMPT".into());
    }

    let loom = Loom::load(
        model_path,
        LoomOptions {
            native_logs: NativeLogPolicy::Silence,
            ..LoomOptions::default()
        },
    )?;
    let metadata = RunMetadata::capture(&loom);
    let mut session = loom.session()?;
    let admission = session.replace_text(&prompt, Tokenization { add_bos: true })?;
    let checkpoint = session.capture_state()?;

    let generation_plan = GenerationRequest::new(GENERATED_TOKENS)?.plan().clone();
    let baseline = session.generate(GenerationRequest::from_plan(generation_plan.clone())?)?;
    session.restore_state(&checkpoint)?;

    let mut pipeline = PipelineBuilder::new(CandidateMode::FullVocabulary, GENERATED_TOKENS)?
        .rank_bias(JOLTED_RANK, JOLT)?
        .build()?;
    let jolted = session.generate(
        GenerationRequest::from_plan(generation_plan.clone())?.pipeline(&mut pipeline)?,
    )?;
    let outputs_differ = baseline.bytes != jolted.bytes || baseline.tokens != jolted.tokens;

    support::write_json(&Report {
        scenario: "fork_and_jolt",
        metadata,
        admission: admission.into(),
        checkpoint: checkpoint.receipt().clone(),
        generation_plan,
        jolt: JoltConfiguration {
            zero_based_rank: u64::try_from(JOLTED_RANK)?,
            additive_bias: JOLT,
        },
        baseline: baseline.into(),
        jolted: jolted.into(),
        outputs_differ,
        transform: pipeline.receipt(),
    })
}

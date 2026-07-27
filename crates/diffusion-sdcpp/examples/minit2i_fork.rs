// SPDX-License-Identifier: MIT OR Apache-2.0

//! Forks one exact `MiniT2I` trajectory into unchanged and intervened replays.

mod support;

use logit_loom_diffusion::{ChannelBias, Digest, Pipeline};
use logit_loom_diffusion_sdcpp::{
    ForkProgram, ImageRequest, NoopProgram, PipelineProgram, ProfileArtifacts, Sdcpp, SdcppOptions,
};
use support::{ForkChecks, ForkReport};

const WIDTH: u32 = 512;
const HEIGHT: u32 = 512;
const SEED: u64 = 7;
const CFG_SCALE: f32 = 1.0;
const STEPS: u32 = 12;
const FORK_STEP: u32 = 5;
const CHANNEL: u64 = 0;
const DELTA: f32 = 0.10;
const MAXIMUM_DELTA: f32 = 0.25;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let library = next(&mut arguments)?;
    let diffusion_model = next(&mut arguments)?;
    let text_encoder = next(&mut arguments)?;
    let backend = next_utf8(&mut arguments, "backend")?;
    let threads = support::parse_threads(next(&mut arguments)?)?;
    let output_directory = support::output_directory(next(&mut arguments)?)?;
    let prompt = next_utf8(&mut arguments, "prompt")?;
    if arguments.next().is_some() {
        return Err(usage().into());
    }

    let artifacts = ProfileArtifacts::minit2i(diffusion_model, text_encoder);
    let options = SdcppOptions::new(backend.clone(), backend, threads)?;
    let mut runtime = Sdcpp::load(library, &artifacts, options)?;
    let request = ImageRequest::linear_euler(prompt, WIDTH, HEIGHT, SEED, CFG_SCALE, STEPS)?;
    let backend_identity = runtime.native_receipt().identity.clone();

    let mut capture =
        ForkProgram::capture(FORK_STEP, backend_identity.clone(), NoopProgram::default())?;
    let baseline = runtime.generate(&request, &mut capture)?;
    let checkpoint_captured = capture.applied();
    let checkpoint = capture
        .take_checkpoint()
        .ok_or("generation did not reach the checkpoint step")?;

    let mut replay = ForkProgram::replay(
        checkpoint.clone(),
        backend_identity.clone(),
        NoopProgram::default(),
    )?;
    let replay_output = runtime.generate(&request, &mut replay)?;

    let intervention_identity =
        Digest::of_bytes("minit2i-fork-intervention-v1", b"channel-0-plus-0.10");
    let pipeline_program = PipelineProgram::at_step(
        &intervention_identity,
        FORK_STEP,
        Box::new(|plan| {
            let bias = ChannelBias::new(&plan.tensor, 2, CHANNEL, DELTA, MAXIMUM_DELTA)
                .map_err(|error| error.to_string())?;
            Pipeline::new(
                plan.digest().map_err(|error| error.to_string())?,
                plan.tensor.clone(),
                vec![Box::new(bias)],
            )
            .map_err(|error| error.to_string())
        }),
        None,
    )?;
    let mut branch = ForkProgram::replay(checkpoint.clone(), backend_identity, pipeline_program)?;
    let branch_output = runtime.generate(&request, &mut branch)?;
    let intervention = branch
        .delegate()
        .pipeline_receipt()
        .ok_or("intervention pipeline was not initialized")?
        .clone();

    support::write_ppm(&output_directory, "baseline.ppm", &baseline)?;
    support::write_ppm(&output_directory, "replay.ppm", &replay_output)?;
    support::write_ppm(&output_directory, "branch.ppm", &branch_output)?;

    let checks = ForkChecks {
        checkpoint_captured: checkpoint_captured.into(),
        replay_applied: replay.applied().into(),
        branch_applied: branch.applied().into(),
        replay_steps_identical: (baseline.receipt.steps == replay_output.receipt.steps).into(),
        replay_image_identical: (baseline.bytes == replay_output.bytes).into(),
        branch_state_changed: branch_output
            .receipt
            .steps
            .iter()
            .any(|step| step.step_index == FORK_STEP && step.elements_changed > 0)
            .into(),
        branch_image_different: (baseline.bytes != branch_output.bytes).into(),
        pipeline_committed_once: (intervention.invocations == 1
            && intervention.failed_stage.is_none())
        .into(),
    };
    let report = ForkReport::new(
        support::ForkScenario::from_catalog("minit2i_fork", "minit2i-b16")?,
        checkpoint.receipt().clone(),
        baseline,
        replay_output,
        branch_output,
        intervention,
        checks,
    )?;
    support::write_json(&report)?;
    if !report.passed {
        return Err("one or more mechanical acceptance checks failed".into());
    }
    Ok(())
}

fn next(
    arguments: &mut impl Iterator<Item = std::ffi::OsString>,
) -> Result<std::ffi::OsString, Box<dyn std::error::Error>> {
    arguments.next().ok_or_else(|| usage().into())
}

fn next_utf8(
    arguments: &mut impl Iterator<Item = std::ffi::OsString>,
    label: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    next(arguments)?
        .into_string()
        .map_err(|_| format!("{label} must be valid UTF-8").into())
}

fn usage() -> &'static str {
    "usage: minit2i_fork COMPANION_LIBRARY DIFFUSION_MODEL TEXT_ENCODER \
     BACKEND THREADS OUTPUT_DIRECTORY PROMPT"
}

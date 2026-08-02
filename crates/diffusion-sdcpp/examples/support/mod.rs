// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared path-safe output and report helpers for image experiments.

use std::{
    fs::{self, OpenOptions},
    io::{self, BufWriter, Write as _},
    path::{Path, PathBuf},
};

use logit_loom_diffusion::{DiffusionCheckpointReceipt, Digest, PipelineReceipt};
use logit_loom_diffusion_sdcpp::{GenerationMeasurements, GenerationOutput, GenerationReceipt};
use logit_loom_models::Catalog;
use serde::Serialize;

/// Result of one explicit mechanical assertion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckStatus {
    /// The assertion held.
    Passed,
    /// The assertion did not hold.
    Failed,
}

impl From<bool> for CheckStatus {
    fn from(value: bool) -> Self {
        if value { Self::Passed } else { Self::Failed }
    }
}

/// Mechanical checks shared by the two checkpoint experiments.
#[derive(Debug, Serialize)]
pub struct ForkChecks {
    /// Capture reached the declared post-step boundary.
    pub checkpoint_captured: CheckStatus,
    /// Unchanged replay reached and authenticated the checkpoint.
    pub replay_applied: CheckStatus,
    /// Intervened replay reached and authenticated the checkpoint.
    pub branch_applied: CheckStatus,
    /// Every post-step state receipt matched during unchanged replay.
    pub replay_steps_identical: CheckStatus,
    /// Final pixel bytes matched during unchanged replay.
    pub replay_image_identical: CheckStatus,
    /// The selected branch step changed at least one state element.
    pub branch_state_changed: CheckStatus,
    /// Final pixel bytes differed after the declared intervention.
    pub branch_image_different: CheckStatus,
    /// The intervention pipeline ran exactly once without failure.
    pub pipeline_committed_once: CheckStatus,
}

impl ForkChecks {
    /// Returns whether every mechanical acceptance condition passed.
    pub const fn all_passed(&self) -> bool {
        matches!(self.checkpoint_captured, CheckStatus::Passed)
            && matches!(self.replay_applied, CheckStatus::Passed)
            && matches!(self.branch_applied, CheckStatus::Passed)
            && matches!(self.replay_steps_identical, CheckStatus::Passed)
            && matches!(self.replay_image_identical, CheckStatus::Passed)
            && matches!(self.branch_state_changed, CheckStatus::Passed)
            && matches!(self.branch_image_different, CheckStatus::Passed)
            && matches!(self.pipeline_committed_once, CheckStatus::Passed)
    }
}

/// Stable catalog metadata for one three-way image experiment.
#[derive(Clone, Copy, Debug)]
pub struct ForkScenario {
    scenario: &'static str,
    profile_id: &'static str,
    integration_status: &'static str,
}

impl ForkScenario {
    /// Resolves an experiment's current status from the packaged model catalog.
    pub fn from_catalog(
        scenario: &'static str,
        profile_id: &'static str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let catalog = Catalog::embedded()?;
        let profile = catalog
            .find_profile(profile_id)
            .ok_or_else(|| format!("catalog does not contain profile {profile_id:?}"))?;
        Ok(Self {
            scenario,
            profile_id,
            integration_status: profile.integration_status().as_str(),
        })
    }
}

/// Path-free report for one three-way image checkpoint experiment.
#[derive(Debug, Serialize)]
pub struct ForkReport {
    /// Report schema version.
    pub schema_version: u32,
    /// Stable experiment identifier.
    pub scenario: &'static str,
    /// Catalog profile identifier.
    pub profile_id: &'static str,
    /// Current checked-in catalog integration status.
    pub integration_status: &'static str,
    /// Exact checkpoint lineage without native state bytes.
    pub checkpoint: DiffusionCheckpointReceipt,
    /// Content identities and completed-step count for the baseline.
    pub baseline_run: RunIdentities,
    /// Non-deterministic deployment measurements for the baseline.
    pub baseline_measurements: GenerationMeasurements,
    /// Baseline mechanics and final image identity.
    pub baseline: GenerationReceipt,
    /// Content identities and completed-step count for the unchanged replay.
    pub replay_run: RunIdentities,
    /// Non-deterministic deployment measurements for the unchanged replay.
    pub replay_measurements: GenerationMeasurements,
    /// Unchanged replay mechanics and final image identity.
    pub replay: GenerationReceipt,
    /// Content identities and completed-step count for the intervened branch.
    pub branch_run: RunIdentities,
    /// Non-deterministic deployment measurements for the intervened branch.
    pub branch_measurements: GenerationMeasurements,
    /// Intervened replay mechanics and final image identity.
    pub branch: GenerationReceipt,
    /// Backend-neutral transactional intervention accounting.
    pub intervention: PipelineReceipt,
    /// Explicit mechanical assertions.
    pub checks: ForkChecks,
    /// Whether all mechanical assertions passed.
    pub passed: bool,
}

impl ForkReport {
    /// Constructs a report while computing each path-free run identity.
    pub fn new(
        scenario: ForkScenario,
        checkpoint: DiffusionCheckpointReceipt,
        baseline: GenerationOutput,
        replay: GenerationOutput,
        branch: GenerationOutput,
        intervention: PipelineReceipt,
        checks: ForkChecks,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let baseline_run = run_identities(&baseline.receipt)?;
        let replay_run = run_identities(&replay.receipt)?;
        let branch_run = run_identities(&branch.receipt)?;
        let passed = checks.all_passed();
        Ok(Self {
            schema_version: 1,
            scenario: scenario.scenario,
            profile_id: scenario.profile_id,
            integration_status: scenario.integration_status,
            checkpoint,
            baseline_run,
            baseline_measurements: baseline.measurements,
            baseline: baseline.receipt,
            replay_run,
            replay_measurements: replay.measurements,
            replay: replay.receipt,
            branch_run,
            branch_measurements: branch.measurements,
            branch: branch.receipt,
            intervention,
            checks,
            passed,
        })
    }
}

/// Path-free identities projected into a retained acceptance report.
#[derive(Debug, Serialize)]
pub struct RunIdentities {
    /// Exact diffusion plan identity.
    pub plan_identity: Digest,
    /// Exact serialized generation-receipt identity.
    pub receipt_identity: Digest,
    /// Exact output pixel-byte identity.
    pub output_identity: Digest,
    /// Number of completed post-Euler transitions.
    pub completed_steps: u32,
}

/// Computes the stable identities for one generated image and its mechanics.
pub fn run_identities(
    receipt: &GenerationReceipt,
) -> Result<RunIdentities, Box<dyn std::error::Error>> {
    Ok(RunIdentities {
        plan_identity: receipt.plan.digest()?,
        receipt_identity: Digest::of_serializable("sdcpp-generation-receipt-v3", receipt)?,
        output_identity: receipt.image.clone(),
        completed_steps: u32::try_from(receipt.steps.len())?,
    })
}

/// Parses a positive native thread count.
pub fn parse_threads(value: std::ffi::OsString) -> Result<u32, Box<dyn std::error::Error>> {
    let value = value
        .into_string()
        .map_err(|_| "thread count must be valid UTF-8")?;
    let threads = value.parse::<u32>()?;
    if threads == 0 {
        return Err("thread count must be positive".into());
    }
    Ok(threads)
}

/// Returns one explicit output directory, creating it when absent.
pub fn output_directory(value: std::ffi::OsString) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let path = PathBuf::from(value);
    if path.exists() {
        if !path.is_dir() {
            return Err("output path exists and is not a directory".into());
        }
    } else {
        fs::create_dir_all(&path)?;
    }
    Ok(path)
}

/// Writes one image as a new binary PPM file without overwriting.
pub fn write_ppm(
    directory: &Path,
    name: &str,
    image: &GenerationOutput,
) -> Result<(), Box<dyn std::error::Error>> {
    if image.receipt.channels < 3 {
        return Err("PPM output requires at least three channels".into());
    }
    let channels = usize::try_from(image.receipt.channels)?;
    let expected = usize::try_from(image.receipt.width)?
        .checked_mul(usize::try_from(image.receipt.height)?)
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or("image byte count overflowed")?;
    if image.bytes.len() != expected {
        return Err("image bytes disagree with their receipt".into());
    }

    let path = directory.join(name);
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut output = BufWriter::new(file);
    writeln!(
        output,
        "P6\n{} {}\n255",
        image.receipt.width, image.receipt.height
    )?;
    for pixel in image.bytes.chunks_exact(channels) {
        output.write_all(&pixel[..3])?;
    }
    output.flush()?;
    Ok(())
}

/// Writes one pretty JSON value followed by a newline.
pub fn write_json(value: &impl Serialize) -> Result<(), Box<dyn std::error::Error>> {
    let stdout = io::stdout();
    let mut output = stdout.lock();
    serde_json::to_writer_pretty(&mut output, value)?;
    writeln!(output)?;
    Ok(())
}

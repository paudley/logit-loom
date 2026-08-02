// SPDX-License-Identifier: MIT OR Apache-2.0

//! Whole-plan lowering for the pinned stable-diffusion.cpp adapter.

use std::path::PathBuf;

use logit_loom_diffusion::{
    ControlFlow, DiffusionPlan, Digest, ImageBufferBinding, ImageBufferLayout, ImageBufferRole,
    ImageCheckpointPlan, ImageCleanupDisposition, ImageCleanupPolicy, ImageCompositeOperation,
    ImageExecutionPlan, ImageExecutionPlanV3, ImageExecutionReceiptV3, ImageOperation,
    ImageOutputReceiptV3, ImageOutputSource, ImageTerminal, ImageValueSource, Intervention,
    InterventionSpec, ObservationKind, ObservationRequest, OperatorInvocation, Pipeline,
    SeedSelection, StepContext, StepSelector, TensorSelector, mask_blend_rgb8,
};
use logit_loom_executor::{
    CancellationProbe, CleanupReceipt, ExecutorState, InputBuffer, LocalExecutor, OutputBuffer,
};
use serde::{Deserialize, Serialize};

use crate::{
    AdvancedImageRequest, AdvancedProgramGenerationReceipt, DiffusionCheckpoint, Error,
    ImageOutputSink, ImagePixels, ImageRequest, LoraBinding, MAX_CHECKPOINT_ENVELOPE_BYTES, Result,
    Sdcpp, StepProgram,
};

/// Worker-local signal that distinguishes checkpoint suspension from terminal
/// cancellation.
///
/// This is deliberately separate from [`CancellationProbe`]: existing
/// executors that implement only terminal cancellation cannot accidentally
/// claim that they retained resumable state.
pub trait CheckpointSuspensionProbe {
    /// Reports whether execution should stop at the next exact resumable
    /// post-Euler boundary.
    fn is_suspension_requested(&self) -> bool;
}

/// Opaque same-runtime continuation for one interrupted whole-image plan.
///
/// The scheduler state is an authenticated [`DiffusionCheckpoint`]. Remaining
/// fields retain bounded receipt, observation, and caller-requested checkpoint
/// lineage so a resumed execution can publish one exact terminal receipt
/// without replaying completed transitions.
#[derive(Clone, Debug)]
pub struct ImagePlanContinuation {
    plan: Digest,
    backend: Digest,
    session_epoch: u64,
    scheduler: DiffusionCheckpoint,
    lineage: PlanCheckpointLineage,
    generation_segments: Vec<AdvancedProgramGenerationReceipt>,
    observation_segments: Vec<Vec<Digest>>,
}

impl ImagePlanContinuation {
    /// Returns the exact plan identity that owns this continuation.
    pub const fn plan(&self) -> &Digest {
        &self.plan
    }

    /// Returns the authenticated native scheduler checkpoint used for direct
    /// continuation.
    pub const fn scheduler_checkpoint(&self) -> &DiffusionCheckpoint {
        &self.scheduler
    }
}

/// Result of one checkpoint-aware whole-image execution attempt.
#[derive(Clone, Debug)]
pub enum CheckpointedImageExecution {
    /// The plan reached one terminal receipt and initialized its declared
    /// outputs as normal.
    Terminal(Box<ImageExecutionReceiptV3>),
    /// The plan stopped without publishing outputs and retained exact state for
    /// same-runtime continuation.
    Suspended(Box<ImagePlanContinuation>),
}

/// Resolves an already verified opaque artifact to a caller-managed path that
/// the pinned native ABI can reopen synchronously.
///
/// A confined worker can return a descriptor path such as `/proc/self/fd/N`.
/// The resolver remains responsible for keeping that descriptor live and for
/// ensuring the path names the exact bytes bound by `input`. Paths are never
/// serialized or retained in receipts.
pub trait ArtifactPathResolver {
    /// Returns a synchronous reopenable path for one exact input.
    ///
    /// # Errors
    ///
    /// Returns an error when the artifact has no safe reopenable descriptor.
    fn resolve_path(
        &mut self,
        binding: &ImageBufferBinding,
        input: &InputBuffer<'_>,
    ) -> std::result::Result<PathBuf, String>;
}

/// Resolver that rejects every path-backed mechanic.
#[derive(Clone, Copy, Debug, Default)]
pub struct RejectArtifactPaths;

impl ArtifactPathResolver for RejectArtifactPaths {
    fn resolve_path(
        &mut self,
        _binding: &ImageBufferBinding,
        _input: &InputBuffer<'_>,
    ) -> std::result::Result<PathBuf, String> {
        Err("no artifact-path resolver was installed".to_owned())
    }
}

/// Fixed binary controls for the installed scheduler-state channel-bias
/// operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChannelBiasControlV1 {
    /// Dimension-zero-fastest tensor axis.
    pub axis: u32,
    /// Channel index on `axis`.
    pub channel: u64,
    /// Exact additive delta bits.
    pub delta_bits: u32,
    /// Exact positive maximum-absolute-delta bits.
    pub maximum_delta_bits: u32,
}

impl ChannelBiasControlV1 {
    /// Creates bounded finite channel-bias controls.
    ///
    /// # Errors
    ///
    /// Returns an error unless the delta is finite and within a positive
    /// declared maximum no greater than 16.
    pub fn new(axis: u32, channel: u64, delta: f32, maximum_absolute_delta: f32) -> Result<Self> {
        if !delta.is_finite()
            || !maximum_absolute_delta.is_finite()
            || maximum_absolute_delta <= 0.0
            || maximum_absolute_delta > 16.0
            || delta.abs() > maximum_absolute_delta
        {
            return Err(Error::Invalid(
                "channel-bias delta is outside its finite declared bound".to_owned(),
            ));
        }
        Ok(Self {
            axis,
            channel,
            delta_bits: delta.to_bits(),
            maximum_delta_bits: maximum_absolute_delta.to_bits(),
        })
    }

    /// Returns the exact additive delta.
    pub fn delta(self) -> f32 {
        f32::from_bits(self.delta_bits)
    }

    /// Returns the exact declared maximum.
    pub fn maximum_absolute_delta(self) -> f32 {
        f32::from_bits(self.maximum_delta_bits)
    }

    /// Encodes the fixed 20-byte little-endian control body.
    pub fn to_control_bytes(self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(20);
        bytes.extend_from_slice(&self.axis.to_le_bytes());
        bytes.extend_from_slice(&self.channel.to_le_bytes());
        bytes.extend_from_slice(&self.delta_bits.to_le_bytes());
        bytes.extend_from_slice(&self.maximum_delta_bits.to_le_bytes());
        bytes
    }

    /// Decodes and validates the fixed 20-byte control body.
    ///
    /// # Errors
    ///
    /// Returns an error for another length or invalid scalar bounds.
    pub fn from_control_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != 20 {
            return Err(Error::Invalid(
                "channel-bias controls must contain exactly 20 bytes".to_owned(),
            ));
        }
        Self::new(
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            u64::from_le_bytes([
                bytes[4], bytes[5], bytes[6], bytes[7], bytes[8], bytes[9], bytes[10], bytes[11],
            ]),
            f32::from_bits(u32::from_le_bytes([
                bytes[12], bytes[13], bytes[14], bytes[15],
            ])),
            f32::from_bits(u32::from_le_bytes([
                bytes[16], bytes[17], bytes[18], bytes[19],
            ])),
        )
    }

    /// Returns the exact installed implementation identity for a tensor and
    /// step selection.
    ///
    /// # Errors
    ///
    /// Returns a tensor validation or deterministic serialization error.
    pub fn implementation_for(
        self,
        tensor: &logit_loom_diffusion::TensorSpec,
        steps: &StepSelector,
    ) -> Result<Digest> {
        let tensor = tensor.digest().map_err(logit_loom_diffusion::Error::from)?;
        Digest::of_serializable(
            "sdcpp-channel-bias-operator-v1",
            &(channel_bias_schema_v1(), tensor, steps, self),
        )
        .map_err(logit_loom_diffusion::Error::from)
        .map_err(Into::into)
    }
}

/// Returns the public schema identity for [`ChannelBiasControlV1`].
pub fn channel_bias_schema_v1() -> Digest {
    Digest::of_bytes(
        "sdcpp-installed-operator-schema-v1",
        b"scheduler-channel-bias-le20-v1",
    )
}

/// Fixed binary controls for the installed model-block residual-scale
/// operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelBlockResidualScaleControlV1 {
    /// Exact residual multiplier bits.
    pub scale_bits: u32,
    /// Exact positive maximum-absolute-scale bits.
    pub maximum_scale_bits: u32,
}

impl ModelBlockResidualScaleControlV1 {
    /// Creates one bounded finite residual multiplier.
    ///
    /// A scale of zero bypasses the selected block. A scale of one preserves
    /// its unmodified output. Other values interpolate or extrapolate the
    /// complete block residual against its input.
    ///
    /// # Errors
    ///
    /// Returns an error unless the scale is finite and within a positive
    /// declared maximum no greater than 16.
    pub fn new(scale: f32, maximum_absolute_scale: f32) -> Result<Self> {
        if !scale.is_finite()
            || !maximum_absolute_scale.is_finite()
            || maximum_absolute_scale <= 0.0
            || maximum_absolute_scale > 16.0
            || scale.abs() > maximum_absolute_scale
        {
            return Err(Error::Invalid(
                "model-block residual scale is outside its finite declared bound".to_owned(),
            ));
        }
        Ok(Self {
            scale_bits: scale.to_bits(),
            maximum_scale_bits: maximum_absolute_scale.to_bits(),
        })
    }

    /// Returns the exact residual multiplier.
    pub fn scale(self) -> f32 {
        f32::from_bits(self.scale_bits)
    }

    /// Returns the exact declared maximum.
    pub fn maximum_absolute_scale(self) -> f32 {
        f32::from_bits(self.maximum_scale_bits)
    }

    /// Encodes the fixed eight-byte little-endian control body.
    pub fn to_control_bytes(self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(8);
        bytes.extend_from_slice(&self.scale_bits.to_le_bytes());
        bytes.extend_from_slice(&self.maximum_scale_bits.to_le_bytes());
        bytes
    }

    /// Decodes and validates the fixed eight-byte control body.
    ///
    /// # Errors
    ///
    /// Returns an error for another length or invalid scalar bounds.
    pub fn from_control_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != 8 {
            return Err(Error::Invalid(
                "model-block residual controls must contain exactly 8 bytes".to_owned(),
            ));
        }
        Self::new(
            f32::from_bits(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])),
            f32::from_bits(u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]])),
        )
    }

    /// Returns the exact installed implementation identity.
    ///
    /// # Errors
    ///
    /// Returns a selector validation or deterministic serialization error.
    pub fn implementation_for(
        self,
        selector: &TensorSelector,
        steps: &StepSelector,
    ) -> Result<Digest> {
        selector
            .validate()
            .map_err(logit_loom_diffusion::Error::from)?;
        Digest::of_serializable(
            "sdcpp-model-block-residual-scale-operator-v1",
            &(
                model_block_residual_scale_schema_v1(),
                selector,
                steps,
                self,
            ),
        )
        .map_err(logit_loom_diffusion::Error::from)
        .map_err(Into::into)
    }
}

/// Returns the public schema identity for
/// [`ModelBlockResidualScaleControlV1`].
pub fn model_block_residual_scale_schema_v1() -> Digest {
    Digest::of_bytes(
        "sdcpp-installed-operator-schema-v1",
        b"model-block-residual-scale-le8-v1",
    )
}

/// Returns the exact fixed-scale whole-request `LoRA` target identity.
pub fn lora_target_v1(high_noise: bool) -> Digest {
    Digest::of_bytes(
        "sdcpp-lora-target-v1",
        if high_noise {
            b"whole-model-high-noise"
        } else {
            b"whole-model"
        },
    )
}

/// Single-owner whole-plan executor over one resident [`Sdcpp`] runtime.
pub struct ImagePlanExecutor<R = RejectArtifactPaths> {
    runtime: Sdcpp,
    artifact_paths: R,
}

impl std::fmt::Debug for ImagePlanExecutor<RejectArtifactPaths> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ImagePlanExecutor")
            .field("runtime", &self.runtime)
            .finish_non_exhaustive()
    }
}

impl ImagePlanExecutor<RejectArtifactPaths> {
    /// Wraps a resident runtime while rejecting path-backed `LoRA` mechanics.
    pub const fn new(runtime: Sdcpp) -> Self {
        Self {
            runtime,
            artifact_paths: RejectArtifactPaths,
        }
    }
}

impl<R> ImagePlanExecutor<R> {
    /// Wraps a resident runtime and a caller-owned opaque-artifact resolver.
    pub const fn with_artifact_paths(runtime: Sdcpp, artifact_paths: R) -> Self {
        Self {
            runtime,
            artifact_paths,
        }
    }

    /// Returns the resident adapter owner.
    pub const fn runtime(&self) -> &Sdcpp {
        &self.runtime
    }

    /// Returns the mutable resident adapter owner.
    pub fn runtime_mut(&mut self) -> &mut Sdcpp {
        &mut self.runtime
    }

    /// Removes the resident adapter owner.
    pub fn into_runtime(self) -> Sdcpp {
        self.runtime
    }
}

impl<R: ArtifactPathResolver> ImagePlanExecutor<R> {
    /// Executes or resumes a whole-image plan with exact scheduler checkpoint
    /// suspension.
    ///
    /// A suspended attempt initializes no caller output. Its returned
    /// continuation is valid only for the same plan and backend runtime. A
    /// resumed attempt starts directly at `next_step`; completed Euler
    /// transitions are never replayed.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid bindings, foreign continuation state,
    /// uncertain cleanup, native failure, or inconsistent segment lineage.
    #[allow(clippy::too_many_lines)]
    pub fn execute_checkpointed(
        &mut self,
        plan: &ImageExecutionPlanV3,
        inputs: &[InputBuffer<'_>],
        outputs: &mut [OutputBuffer<'_>],
        cancellation: &dyn CancellationProbe,
        suspension: &dyn CheckpointSuspensionProbe,
        continuation: Option<ImagePlanContinuation>,
    ) -> Result<CheckpointedImageExecution> {
        plan.validate().map_err(logit_loom_diffusion::Error::from)?;
        self.validate_bindings(plan)?;
        validate_bound_inputs(&plan.primary, inputs)?;
        validate_bound_outputs(plan, outputs)?;
        validate_supported_mechanics(&plan.primary)?;

        let bindings = self.runtime.execution_bindings()?;
        let plan_digest = plan.digest().map_err(logit_loom_diffusion::Error::from)?;
        if continuation.is_none() && cancellation.is_cancelled() {
            return self
                .execute_inner(plan, inputs, outputs, cancellation)
                .map(Box::new)
                .map(CheckpointedImageExecution::Terminal);
        }

        let (
            session_epoch,
            lineage,
            mut generation_segments,
            mut observation_segments,
            scheduler_restore,
        ) = if let Some(continuation) = continuation {
            if continuation.plan != plan_digest || continuation.backend != bindings.backend {
                return Err(Error::Incompatible(
                    "whole-image continuation owner differs from the plan or backend".to_owned(),
                ));
            }
            (
                continuation.session_epoch,
                continuation.lineage,
                continuation.generation_segments,
                continuation.observation_segments,
                Some(continuation.scheduler),
            )
        } else {
            let restore = plan
                .checkpoint
                .restore_from
                .map(|slot| {
                    let input = input_for_slot(&plan.primary, inputs, slot)?;
                    DiffusionCheckpoint::from_envelope_bytes(input.bytes())
                })
                .transpose()?;
            (
                self.runtime.session_epoch(),
                PlanCheckpointLineage {
                    restore,
                    ..PlanCheckpointLineage::default()
                },
                Vec::new(),
                Vec::new(),
                None,
            )
        };

        let request = lower_request(&plan.primary, inputs, &mut self.artifact_paths)?;
        let mut program = PlanProgram::checkpointed(
            plan,
            cancellation,
            Some(suspension),
            lineage,
            scheduler_restore.clone(),
            bindings.backend.clone(),
        )?;
        let image_len = rgb_len(plan.primary.width, plan.primary.height)?;
        let mut primary_bytes = vec![0_u8; image_len];
        let mut sink = PlanImageSink(&mut primary_bytes);
        let generated = match scheduler_restore.as_ref() {
            Some(checkpoint) => self.runtime.generate_advanced_program_from_checkpoint_to(
                &request,
                checkpoint,
                &mut program,
                &mut sink,
            ),
            None => self
                .runtime
                .generate_advanced_program_to(&request, &mut program, &mut sink),
        }?;
        validate_checkpointed_generation_segment(
            generation_segments.last(),
            &generated.receipt,
            scheduler_restore.as_ref(),
        )?;
        let completed_steps = generated
            .receipt
            .generation
            .steps
            .last()
            .and_then(|step| step.step_index.checked_add(1))
            .ok_or_else(|| Error::Incompatible("native completed no Euler boundary".to_owned()))?;
        let terminal = if generated.receipt.generation.stopped {
            ImageTerminal::CancelledAfterStep {
                step: completed_steps - 1,
            }
        } else {
            ImageTerminal::Completed
        };
        program.validate_terminal(&terminal)?;
        let scheduler_checkpoint = program.take_suspension_checkpoint();
        let observations = program.finish_observations();
        let (checkpoint_identities, checkpoint_envelope) = program.finish_checkpoints()?;
        let lineage = program.into_lineage();
        generation_segments.push(generated.receipt);
        observation_segments.push(observations);

        if matches!(terminal, ImageTerminal::CancelledAfterStep { .. })
            && suspension.is_suspension_requested()
            && !cancellation.is_cancelled()
        {
            let scheduler = scheduler_checkpoint.ok_or_else(|| {
                Error::Poisoned(
                    "whole-image suspension stopped without an exact scheduler checkpoint"
                        .to_owned(),
                )
            })?;
            self.runtime.clear_session()?;
            return Ok(CheckpointedImageExecution::Suspended(Box::new(
                ImagePlanContinuation {
                    plan: plan_digest,
                    backend: bindings.backend,
                    session_epoch,
                    scheduler,
                    lineage,
                    generation_segments,
                    observation_segments,
                },
            )));
        }
        if scheduler_checkpoint.is_some() {
            return Err(Error::Poisoned(
                "whole-image terminal execution retained scheduler state".to_owned(),
            ));
        }

        let primary_receipt = if generation_segments.len() == 1 {
            Digest::of_serializable(
                "sdcpp-advanced-program-generation-receipt-v1",
                &generation_segments[0],
            )
        } else {
            Digest::of_serializable(
                "sdcpp-checkpointed-image-plan-generation-v1",
                &generation_segments,
            )
        }
        .map_err(logit_loom_diffusion::Error::from)?;
        let observation_identities = finish_checkpointed_observations(&observation_segments)?;
        let (composite_bytes, composite_receipts) =
            execute_composites(plan, inputs, &primary_bytes)?;
        let output_receipts = preflight_route_payloads(
            plan,
            inputs,
            outputs,
            &primary_bytes,
            &composite_bytes,
            checkpoint_envelope.as_deref(),
        )?;
        let receipt = self.finish_execution(
            plan,
            inputs,
            outputs,
            CompletedExecution {
                primary_bytes,
                composite_bytes,
                composite_receipts,
                checkpoint_envelope,
                checkpoint_identities,
                observation_identities,
                output_receipts,
                primary_receipt,
                plan_digest,
                backend: bindings.backend,
                profile: bindings.profile,
                session_epoch,
                completed_steps,
                terminal,
            },
        )?;
        Ok(CheckpointedImageExecution::Terminal(Box::new(receipt)))
    }

    fn execute_inner(
        &mut self,
        plan: &ImageExecutionPlanV3,
        inputs: &[InputBuffer<'_>],
        outputs: &mut [OutputBuffer<'_>],
        cancellation: &dyn CancellationProbe,
    ) -> Result<ImageExecutionReceiptV3> {
        plan.validate().map_err(logit_loom_diffusion::Error::from)?;
        self.validate_bindings(plan)?;
        validate_bound_inputs(&plan.primary, inputs)?;
        validate_bound_outputs(plan, outputs)?;
        validate_supported_mechanics(&plan.primary)?;

        let bindings = self.runtime.execution_bindings()?;
        let plan_digest = plan.digest().map_err(logit_loom_diffusion::Error::from)?;
        let session_epoch = self.runtime.session_epoch();
        if cancellation.is_cancelled() {
            let receipt = ImageExecutionReceiptV3 {
                plan: plan_digest,
                backend: bindings.backend,
                profile: bindings.profile,
                session_epoch,
                completed_steps: 0,
                terminal: ImageTerminal::CancelledBeforeStart,
                primary: None,
                checkpoints: Vec::new(),
                composites: Vec::new(),
                outputs: Vec::new(),
                observations: Vec::new(),
                cleanup: ImageCleanupDisposition::NotRequired,
            };
            receipt
                .validate_for(plan)
                .map_err(logit_loom_diffusion::Error::from)?;
            return Ok(receipt);
        }

        let restore = plan
            .checkpoint
            .restore_from
            .map(|slot| {
                let input = input_for_slot(&plan.primary, inputs, slot)?;
                DiffusionCheckpoint::from_envelope_bytes(input.bytes())
            })
            .transpose()?;
        let request = lower_request(&plan.primary, inputs, &mut self.artifact_paths)?;
        let mut program = PlanProgram::new(plan, cancellation, restore, bindings.backend.clone())?;
        let image_len = rgb_len(plan.primary.width, plan.primary.height)?;
        let mut primary_bytes = vec![0_u8; image_len];
        let mut sink = PlanImageSink(&mut primary_bytes);
        let generated =
            self.runtime
                .generate_advanced_program_to(&request, &mut program, &mut sink)?;
        let completed_steps = u32::try_from(generated.receipt.generation.steps.len())
            .map_err(|_| Error::Incompatible("completed steps exceed u32".to_owned()))?;
        let terminal = if generated.receipt.generation.stopped {
            let step = completed_steps
                .checked_sub(1)
                .ok_or_else(|| Error::Incompatible("native stopped before a step".to_owned()))?;
            ImageTerminal::CancelledAfterStep { step }
        } else {
            ImageTerminal::Completed
        };
        program.validate_terminal(&terminal)?;

        let primary_receipt = Digest::of_serializable(
            "sdcpp-advanced-program-generation-receipt-v1",
            &generated.receipt,
        )
        .map_err(logit_loom_diffusion::Error::from)?;
        let (checkpoint_identities, checkpoint_envelope) = program.finish_checkpoints()?;
        let observation_identities = program.finish_observations();
        let (composite_bytes, composite_receipts) =
            execute_composites(plan, inputs, &primary_bytes)?;
        let output_receipts = preflight_route_payloads(
            plan,
            inputs,
            outputs,
            &primary_bytes,
            &composite_bytes,
            checkpoint_envelope.as_deref(),
        )?;
        self.finish_execution(
            plan,
            inputs,
            outputs,
            CompletedExecution {
                primary_bytes,
                composite_bytes,
                composite_receipts,
                checkpoint_envelope,
                checkpoint_identities,
                observation_identities,
                output_receipts,
                primary_receipt,
                plan_digest,
                backend: bindings.backend,
                profile: bindings.profile,
                session_epoch,
                completed_steps,
                terminal,
            },
        )
    }

    fn finish_execution(
        &mut self,
        plan: &ImageExecutionPlanV3,
        inputs: &[InputBuffer<'_>],
        outputs: &mut [OutputBuffer<'_>],
        mut execution: CompletedExecution,
    ) -> Result<ImageExecutionReceiptV3> {
        let cleanup = match plan.cleanup {
            ImageCleanupPolicy::RetainSession => ImageCleanupDisposition::Retained,
            ImageCleanupPolicy::ClearSession => {
                let receipt = self.runtime.clear_session()?;
                receipt.validate().map_err(|error| {
                    Error::Poisoned(format!("cleanup receipt was invalid: {error}"))
                })?;
                ImageCleanupDisposition::Confirmed {
                    cleared_epoch: receipt.cleared_epoch,
                }
            }
        };
        if let Some(checkpoint) = execution.checkpoint_envelope.as_deref() {
            execution
                .checkpoint_identities
                .push(Digest::of_bytes("sdcpp-checkpoint-envelope-v1", checkpoint));
        }
        let receipt = ImageExecutionReceiptV3 {
            plan: execution.plan_digest,
            backend: execution.backend,
            profile: execution.profile,
            session_epoch: execution.session_epoch,
            completed_steps: execution.completed_steps,
            terminal: execution.terminal,
            primary: Some(execution.primary_receipt),
            checkpoints: execution.checkpoint_identities,
            composites: execution.composite_receipts,
            outputs: execution.output_receipts,
            observations: execution.observation_identities,
            cleanup,
        };
        receipt
            .validate_for(plan)
            .map_err(logit_loom_diffusion::Error::from)?;
        write_routes(
            plan,
            inputs,
            outputs,
            &execution.primary_bytes,
            &execution.composite_bytes,
            execution.checkpoint_envelope.as_deref(),
        )?;
        Ok(receipt)
    }

    fn validate_bindings(&self, plan: &ImageExecutionPlanV3) -> Result<()> {
        let bindings = self.runtime.execution_bindings()?;
        if plan.primary.profile != bindings.profile
            || plan.primary.load != bindings.load
            || plan.primary.rng != bindings.rng
            || plan.primary.placement != bindings.placement
        {
            return Err(Error::Incompatible(
                "whole-image plan profile, load, RNG, or placement differs from the resident owner"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

struct CompletedExecution {
    primary_bytes: Vec<u8>,
    composite_bytes: Vec<Vec<u8>>,
    composite_receipts: Vec<logit_loom_diffusion::ImageCompositeReceipt>,
    checkpoint_envelope: Option<Vec<u8>>,
    checkpoint_identities: Vec<Digest>,
    observation_identities: Vec<Digest>,
    output_receipts: Vec<ImageOutputReceiptV3>,
    primary_receipt: Digest,
    plan_digest: Digest,
    backend: Digest,
    profile: Digest,
    session_epoch: u64,
    completed_steps: u32,
    terminal: ImageTerminal,
}

impl<R: ArtifactPathResolver> LocalExecutor for ImagePlanExecutor<R> {
    type Plan = ImageExecutionPlanV3;
    type Receipt = ImageExecutionReceiptV3;
    type Error = Error;

    fn state(&self) -> ExecutorState {
        self.runtime.state()
    }

    fn warm(
        &mut self,
        _plan: &Self::Plan,
        _cancellation: &dyn CancellationProbe,
    ) -> Result<Self::Receipt> {
        Err(Error::Invalid(
            "whole-image warm-up requires the exact bound inputs; call execute".to_owned(),
        ))
    }

    fn execute(
        &mut self,
        plan: &Self::Plan,
        inputs: &[InputBuffer<'_>],
        outputs: &mut [OutputBuffer<'_>],
        cancellation: &dyn CancellationProbe,
    ) -> Result<Self::Receipt> {
        self.execute_inner(plan, inputs, outputs, cancellation)
    }

    fn clear_session(&mut self) -> Result<CleanupReceipt> {
        self.runtime.clear_session()
    }

    fn close(self) -> Result<CleanupReceipt> {
        self.runtime.close()
    }
}

struct PlanImageSink<'a>(&'a mut [u8]);

impl ImageOutputSink for PlanImageSink<'_> {
    fn expected_len(&self) -> usize {
        self.0.len()
    }

    fn write_image(&mut self, bytes: &[u8]) -> std::result::Result<(), String> {
        if bytes.len() != self.0.len() {
            return Err("whole-plan image length differs".to_owned());
        }
        self.0.copy_from_slice(bytes);
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
struct PlanCheckpointLineage {
    restore: Option<DiffusionCheckpoint>,
    restored: Option<logit_loom_diffusion::DiffusionCheckpointReceipt>,
    captured: Option<DiffusionCheckpoint>,
}

struct PlanProgram<'a> {
    implementation: Digest,
    expected_schedule: logit_loom_diffusion::DiffusionSchedule,
    expected_rng: Digest,
    expected_seed: u64,
    operators: Vec<OperatorInvocation>,
    observations: Vec<ObservationAccumulator>,
    checkpoint: ImageCheckpointPlan,
    lineage: PlanCheckpointLineage,
    scheduler_restore: Option<DiffusionCheckpoint>,
    suspension_checkpoint: Option<DiffusionCheckpoint>,
    pipeline: Option<Pipeline>,
    actual_plan: Option<DiffusionPlan>,
    backend: Digest,
    cancellation: &'a dyn CancellationProbe,
    suspension: Option<&'a dyn CheckpointSuspensionProbe>,
}

impl<'a> PlanProgram<'a> {
    fn new(
        plan: &ImageExecutionPlanV3,
        cancellation: &'a dyn CancellationProbe,
        restore: Option<DiffusionCheckpoint>,
        backend: Digest,
    ) -> Result<Self> {
        Self::checkpointed(
            plan,
            cancellation,
            None,
            PlanCheckpointLineage {
                restore,
                ..PlanCheckpointLineage::default()
            },
            None,
            backend,
        )
    }

    fn checkpointed(
        plan: &ImageExecutionPlanV3,
        cancellation: &'a dyn CancellationProbe,
        suspension: Option<&'a dyn CheckpointSuspensionProbe>,
        lineage: PlanCheckpointLineage,
        scheduler_restore: Option<DiffusionCheckpoint>,
        backend: Digest,
    ) -> Result<Self> {
        let schedule = plan.primary.schedule.clone().ok_or_else(|| {
            Error::Invalid("whole-image execution requires a diffusion schedule".to_owned())
        })?;
        let seed = match plan.primary.seed {
            SeedSelection::Fixed { seed } => seed,
            SeedSelection::WorkerSelected { .. } => {
                return Err(Error::Invalid(
                    "stable-diffusion.cpp requires a caller-supplied fixed seed".to_owned(),
                ));
            }
        };
        let implementation = Digest::of_serializable(
            "sdcpp-image-plan-program-v1",
            &(
                plan.digest().map_err(logit_loom_diffusion::Error::from)?,
                &plan.primary.operators,
                &plan.primary.observations,
                &plan.checkpoint,
            ),
        )
        .map_err(logit_loom_diffusion::Error::from)?;
        let observations = plan
            .primary
            .observations
            .iter()
            .cloned()
            .map(ObservationAccumulator::new)
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            implementation,
            expected_schedule: schedule,
            expected_rng: plan.primary.rng.clone(),
            expected_seed: seed,
            operators: plan.primary.operators.clone(),
            observations,
            checkpoint: plan.checkpoint.clone(),
            lineage,
            scheduler_restore,
            suspension_checkpoint: None,
            pipeline: None,
            actual_plan: None,
            backend,
            cancellation,
            suspension,
        })
    }

    fn validate_terminal(&self, terminal: &ImageTerminal) -> Result<()> {
        if *terminal == ImageTerminal::Completed
            && (self.checkpoint.restore_from.is_some() != self.lineage.restored.is_some()
                || self.checkpoint.capture_after_step.is_some() != self.lineage.captured.is_some())
        {
            return Err(Error::Incompatible(
                "completed execution did not reach every checkpoint boundary".to_owned(),
            ));
        }
        if let ImageTerminal::CancelledAfterStep { step } = terminal {
            let restore_next_step = self
                .lineage
                .restore
                .as_ref()
                .map(|checkpoint| checkpoint.receipt().next_step)
                .or_else(|| {
                    self.lineage
                        .restored
                        .as_ref()
                        .map(|receipt| receipt.next_step)
                });
            let restore_reached =
                restore_next_step.is_some_and(|next| next <= step.saturating_add(1));
            let capture_reached = self
                .checkpoint
                .capture_after_step
                .is_some_and(|capture| capture <= *step);
            if restore_reached != self.lineage.restored.is_some()
                || capture_reached != self.lineage.captured.is_some()
            {
                return Err(Error::Incompatible(
                    "cancelled execution checkpoint lineage differs from the reached boundary"
                        .to_owned(),
                ));
            }
        }
        Ok(())
    }

    fn finish_checkpoints(&self) -> Result<(Vec<Digest>, Option<Vec<u8>>)> {
        let plan = self
            .actual_plan
            .as_ref()
            .ok_or_else(|| Error::Incompatible("step program was not initialized".to_owned()))?;
        let mut identities = Vec::new();
        if let Some(restored) = &self.lineage.restored {
            identities.push(
                restored
                    .digest_for(plan)
                    .map_err(logit_loom_diffusion::Error::from)?,
            );
        }
        if let Some(checkpoint) = &self.lineage.captured {
            identities.push(
                checkpoint
                    .receipt()
                    .digest_for(plan)
                    .map_err(logit_loom_diffusion::Error::from)?,
            );
        }
        let envelope = self
            .lineage
            .captured
            .as_ref()
            .map(DiffusionCheckpoint::to_envelope_bytes)
            .transpose()?;
        Ok((identities, envelope))
    }

    fn finish_observations(&self) -> Vec<Digest> {
        self.observations
            .iter()
            .map(ObservationAccumulator::finish)
            .collect()
    }

    fn take_suspension_checkpoint(&mut self) -> Option<DiffusionCheckpoint> {
        self.suspension_checkpoint.take()
    }

    fn into_lineage(self) -> PlanCheckpointLineage {
        self.lineage
    }
}

impl StepProgram for PlanProgram<'_> {
    fn implementation(&self) -> &Digest {
        &self.implementation
    }

    fn begin(&mut self, plan: &DiffusionPlan) -> std::result::Result<(), String> {
        if plan.schedule != self.expected_schedule
            || plan.rng != self.expected_rng
            || plan.seed != self.expected_seed
        {
            return Err(
                "native schedule, RNG, or seed differs from the whole-image plan".to_owned(),
            );
        }
        if let Some(checkpoint) = &self.lineage.restore {
            checkpoint
                .receipt()
                .validate_for(plan)
                .map_err(|error| error.to_string())?;
            if checkpoint.receipt().backend != self.backend {
                return Err("checkpoint backend identity differs".to_owned());
            }
        }
        if let Some(checkpoint) = &self.scheduler_restore {
            checkpoint
                .receipt()
                .validate_for(plan)
                .map_err(|error| error.to_string())?;
            if checkpoint.receipt().backend != self.backend {
                return Err("scheduler checkpoint backend identity differs".to_owned());
            }
        }
        let mut stages: Vec<Box<dyn Intervention>> = Vec::with_capacity(self.operators.len());
        for operator in &self.operators {
            stages.push(Box::new(
                InstalledChannelBias::from_invocation(operator, plan)
                    .map_err(|error| error.to_string())?,
            ));
        }
        if !stages.is_empty() {
            let mut pipeline = Pipeline::new(
                plan.digest().map_err(|error| error.to_string())?,
                plan.tensor.clone(),
                stages,
            )
            .map_err(|error| error.to_string())?;
            pipeline.begin().map_err(|error| error.to_string())?;
            self.pipeline = Some(pipeline);
        }
        self.actual_plan = Some(plan.clone());
        Ok(())
    }

    fn intervene(
        &mut self,
        context: &StepContext,
        state: &mut [f32],
    ) -> std::result::Result<(), String> {
        let plan = self
            .actual_plan
            .as_ref()
            .ok_or_else(|| "step program was not initialized".to_owned())?;
        if let Some(checkpoint) = &self.lineage.restore
            && checkpoint.receipt().next_step == context.step_index.saturating_add(1)
        {
            if self.lineage.restored.is_some() {
                return Err("checkpoint restore boundary repeated".to_owned());
            }
            checkpoint
                .restore(plan, &self.backend, context, state)
                .map_err(|error| error.to_string())?;
            self.lineage.restored = Some(checkpoint.receipt().clone());
            self.lineage.restore = None;
        }
        if self.checkpoint.capture_after_step == Some(context.step_index) {
            if self.lineage.captured.is_some() {
                return Err("checkpoint capture boundary repeated".to_owned());
            }
            self.lineage.captured = Some(
                DiffusionCheckpoint::capture(plan, &self.backend, context, state)
                    .map_err(|error| error.to_string())?,
            );
        }
        if let Some(pipeline) = &mut self.pipeline {
            pipeline
                .apply(context, state)
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    fn observe(
        &mut self,
        context: &StepContext,
        state: &[f32],
    ) -> std::result::Result<ControlFlow, String> {
        for observation in &mut self.observations {
            observation.record(context, state)?;
        }
        if self.cancellation.is_cancelled() {
            return Ok(ControlFlow::Stop);
        }
        if self
            .suspension
            .is_some_and(CheckpointSuspensionProbe::is_suspension_requested)
        {
            let plan = self
                .actual_plan
                .as_ref()
                .ok_or_else(|| "step program was not initialized".to_owned())?;
            let next_step = usize::try_from(context.step_index)
                .ok()
                .and_then(|step| step.checked_add(1))
                .ok_or_else(|| "scheduler checkpoint step overflowed".to_owned())?;
            if next_step < plan.schedule.steps() {
                if self.suspension_checkpoint.is_some() {
                    return Err("scheduler checkpoint boundary repeated".to_owned());
                }
                self.suspension_checkpoint = Some(
                    DiffusionCheckpoint::capture(plan, &self.backend, context, state)
                        .map_err(|error| error.to_string())?,
                );
                return Ok(ControlFlow::Stop);
            }
        }
        Ok(ControlFlow::Continue)
    }
}

pub(crate) struct InstalledChannelBias {
    specification: InterventionSpec,
    tensor: logit_loom_diffusion::TensorSpec,
    steps: StepSelector,
    axis: usize,
    channel: u64,
    delta: f32,
}

pub(crate) struct InstalledModelBlockResidualScale {
    pub(crate) block: u32,
    pub(crate) scale: f32,
    pub(crate) steps: StepSelector,
}

impl InstalledModelBlockResidualScale {
    pub(crate) fn from_invocation(operator: &OperatorInvocation) -> Result<Self> {
        let TensorSelector::ModelBlock {
            component,
            block,
            site,
        } = &operator.selector
        else {
            return Err(Error::Invalid(
                "model-block residual scale requires a model-block selector".to_owned(),
            ));
        };
        if component != "krea2"
            || site != "residual"
            || operator.schema != model_block_residual_scale_schema_v1()
        {
            return Err(Error::Invalid(
                "stable-diffusion.cpp does not install this model-block component, site, or schema"
                    .to_owned(),
            ));
        }
        let controls = ModelBlockResidualScaleControlV1::from_control_bytes(&operator.controls)?;
        let expected = controls.implementation_for(&operator.selector, &operator.steps)?;
        if operator.implementation != expected {
            return Err(Error::Incompatible(
                "installed model-block implementation identity differs".to_owned(),
            ));
        }
        Ok(Self {
            block: *block,
            scale: controls.scale(),
            steps: operator.steps.clone(),
        })
    }
}

impl InstalledChannelBias {
    pub(crate) fn from_invocation(
        operator: &OperatorInvocation,
        plan: &DiffusionPlan,
    ) -> Result<Self> {
        if operator.schema != channel_bias_schema_v1()
            || operator.selector != TensorSelector::SchedulerState
        {
            return Err(Error::Invalid(
                "stable-diffusion.cpp supports only the installed scheduler channel-bias schema"
                    .to_owned(),
            ));
        }
        let controls = ChannelBiasControlV1::from_control_bytes(&operator.controls)?;
        let axis = usize::try_from(controls.axis)
            .map_err(|_| Error::Invalid("channel-bias axis exceeds usize".to_owned()))?;
        if axis >= plan.tensor.shape.len() || controls.channel >= plan.tensor.shape[axis] {
            return Err(Error::Invalid(
                "channel-bias axis or channel is outside the native tensor".to_owned(),
            ));
        }
        let expected = controls.implementation_for(&plan.tensor, &operator.steps)?;
        if operator.implementation != expected {
            return Err(Error::Incompatible(
                "installed channel-bias implementation identity differs".to_owned(),
            ));
        }
        let tensor = plan
            .tensor
            .digest()
            .map_err(logit_loom_diffusion::Error::from)?;
        Ok(Self {
            specification: InterventionSpec {
                implementation: operator.implementation.clone(),
                tensor,
                max_invocations: u32::try_from(plan.schedule.steps())
                    .map_err(|_| Error::Invalid("diffusion steps exceed u32".to_owned()))?,
            },
            tensor: plan.tensor.clone(),
            steps: operator.steps.clone(),
            axis,
            channel: controls.channel,
            delta: controls.delta(),
        })
    }
}

impl Intervention for InstalledChannelBias {
    fn specification(&self) -> &InterventionSpec {
        &self.specification
    }

    fn apply(
        &mut self,
        context: &StepContext,
        state: &mut [f32],
    ) -> std::result::Result<(), String> {
        if !step_selected(&self.steps, context.step_index) {
            return Ok(());
        }
        let stride = self.tensor.shape[..self.axis]
            .iter()
            .try_fold(1_u64, |product, dimension| product.checked_mul(*dimension))
            .ok_or_else(|| "channel-bias stride overflowed".to_owned())?;
        let dimension = self.tensor.shape[self.axis];
        for (flat, value) in state.iter_mut().enumerate() {
            let flat = u64::try_from(flat).map_err(|_| "flat index exceeds u64".to_owned())?;
            if (flat / stride) % dimension == self.channel {
                *value += self.delta;
            }
        }
        Ok(())
    }
}

pub(crate) struct ObservationAccumulator {
    request: ObservationRequest,
    hasher: blake3::Hasher,
    observations: u32,
}

impl ObservationAccumulator {
    pub(crate) fn new(request: ObservationRequest) -> Result<Self> {
        if request.selector != TensorSelector::SchedulerState
            || request.kind == ObservationKind::Snapshot
        {
            return Err(Error::Invalid(
                "stable-diffusion.cpp supports scheduler digest/statistics observations only"
                    .to_owned(),
            ));
        }
        let identity = Digest::of_serializable("sdcpp-observation-request-v1", &request)
            .map_err(logit_loom_diffusion::Error::from)?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"logit-loom\0sdcpp-plan-observation-v1\0");
        hasher.update(identity.as_str().as_bytes());
        Ok(Self {
            request,
            hasher,
            observations: 0,
        })
    }

    pub(crate) fn record(
        &mut self,
        context: &StepContext,
        state: &[f32],
    ) -> std::result::Result<(), String> {
        if !step_selected(&self.request.steps, context.step_index) {
            return Ok(());
        }
        self.hasher.update(&context.step_index.to_le_bytes());
        match self.request.kind {
            ObservationKind::Digest => {
                for value in state {
                    self.hasher.update(&value.to_bits().to_le_bytes());
                }
            }
            ObservationKind::Statistics => {
                let mut minimum = f32::INFINITY;
                let mut maximum = f32::NEG_INFINITY;
                let mut sum = 0.0_f64;
                for value in state {
                    minimum = minimum.min(*value);
                    maximum = maximum.max(*value);
                    sum += f64::from(*value);
                }
                self.hasher.update(
                    &u64::try_from(state.len())
                        .map_err(|_| "observation state length exceeds u64".to_owned())?
                        .to_le_bytes(),
                );
                self.hasher.update(&minimum.to_bits().to_le_bytes());
                self.hasher.update(&maximum.to_bits().to_le_bytes());
                self.hasher.update(&sum.to_bits().to_le_bytes());
            }
            ObservationKind::Snapshot => {
                return Err("snapshot observation passed preflight".to_owned());
            }
        }
        self.observations = self
            .observations
            .checked_add(1)
            .ok_or_else(|| "observation count overflowed".to_owned())?;
        Ok(())
    }

    pub(crate) fn finish(&self) -> Digest {
        let mut hasher = self.hasher.clone();
        hasher.update(&self.observations.to_le_bytes());
        Digest::of_bytes("sdcpp-plan-observation-v1", hasher.finalize().as_bytes())
    }
}

fn validate_supported_mechanics(plan: &ImageExecutionPlan) -> Result<()> {
    if !matches!(
        plan.operation,
        ImageOperation::TextToImage
            | ImageOperation::ImageToImage
            | ImageOperation::Inpaint
            | ImageOperation::Outpaint
    ) {
        return Err(Error::Invalid(
            "whole-image executor currently requires a diffusion image operation".to_owned(),
        ));
    }
    if plan.operators.iter().any(|operator| {
        operator.schema != channel_bias_schema_v1()
            || operator.selector != TensorSelector::SchedulerState
    }) || plan.observations.iter().any(|observation| {
        observation.selector != TensorSelector::SchedulerState
            || observation.kind == ObservationKind::Snapshot
    }) {
        return Err(Error::Invalid(
            "whole-image plan requests an operator or observation absent from this adapter"
                .to_owned(),
        ));
    }
    for lora in &plan.loras {
        if lora.scales.points.len() != 1 || lora.scales.points[0].step != 0 {
            return Err(Error::Invalid(
                "image ABI v2 supports one fixed LoRA scale for the complete request".to_owned(),
            ));
        }
        if lora.target != lora_target_v1(false) && lora.target != lora_target_v1(true) {
            return Err(Error::Invalid(
                "image ABI v2 LoRA target identity is unsupported".to_owned(),
            ));
        }
    }
    Ok(())
}

fn validate_bound_inputs(plan: &ImageExecutionPlan, inputs: &[InputBuffer<'_>]) -> Result<()> {
    if inputs.len() != plan.inputs.len() {
        return Err(Error::Invalid(
            "whole-image inputs must match declared plan order and count".to_owned(),
        ));
    }
    for (binding, input) in plan.inputs.iter().zip(inputs) {
        if binding.role == ImageBufferRole::Checkpoint
            && input.bytes().len() > MAX_CHECKPOINT_ENVELOPE_BYTES
        {
            return Err(Error::Invalid(
                "checkpoint input exceeds the adapter envelope bound".to_owned(),
            ));
        }
        if input.specification() != &binding.buffer {
            return Err(Error::Invalid(
                "whole-image input specification differs from its declared binding".to_owned(),
            ));
        }
        binding
            .validate_bytes(input.bytes())
            .map_err(logit_loom_diffusion::Error::from)?;
    }
    Ok(())
}

fn validate_bound_outputs(plan: &ImageExecutionPlanV3, outputs: &[OutputBuffer<'_>]) -> Result<()> {
    if outputs.len() != plan.outputs.len() {
        return Err(Error::Invalid(
            "whole-image outputs must match declared route order and count".to_owned(),
        ));
    }
    for (route, output) in plan.outputs.iter().zip(outputs) {
        if matches!(route.source, ImageOutputSource::Checkpoint)
            && output.specification().byte_length
                > u64::try_from(MAX_CHECKPOINT_ENVELOPE_BYTES)
                    .map_err(|_| Error::Invalid("checkpoint bound exceeds u64".to_owned()))?
        {
            return Err(Error::Invalid(
                "checkpoint output exceeds the adapter envelope bound".to_owned(),
            ));
        }
        if output.specification() != &route.buffer || output.written() != 0 {
            return Err(Error::Invalid(
                "whole-image output allocation differs or is already initialized".to_owned(),
            ));
        }
    }
    Ok(())
}

fn lower_request<'a, R: ArtifactPathResolver>(
    plan: &ImageExecutionPlan,
    inputs: &'a [InputBuffer<'a>],
    paths: &mut R,
) -> Result<AdvancedImageRequest<'a>> {
    let prompt = input_for_role(plan, inputs, ImageBufferRole::PositiveConditioning)?
        .ok_or_else(|| Error::Invalid("positive conditioning input is missing".to_owned()))?;
    let prompt = std::str::from_utf8(prompt.1.bytes())
        .map_err(|_| Error::Invalid("positive conditioning must be UTF-8".to_owned()))?;
    let seed = match plan.seed {
        SeedSelection::Fixed { seed } => seed,
        SeedSelection::WorkerSelected { .. } => {
            return Err(Error::Invalid(
                "stable-diffusion.cpp requires a fixed seed".to_owned(),
            ));
        }
    };
    let base = ImageRequest::new(
        prompt,
        plan.width,
        plan.height,
        seed,
        plan.guidance_scale(),
        plan.schedule.clone().ok_or_else(|| {
            Error::Invalid("diffusion operation is missing its schedule".to_owned())
        })?,
    )?;
    let source = input_for_role(plan, inputs, ImageBufferRole::SourceImage)?
        .map(|input| pixels_for_binding(plan, inputs, input.0))
        .transpose()?;
    let mask = input_for_role(plan, inputs, ImageBufferRole::Mask)?
        .map(|input| pixels_for_binding(plan, inputs, input.0))
        .transpose()?;
    let mut request = AdvancedImageRequest::text_to_image(base)?;
    request = match plan.operation {
        ImageOperation::TextToImage => request,
        ImageOperation::ImageToImage => request.image_to_image(
            source.ok_or_else(|| Error::Invalid("source image is missing".to_owned()))?,
            plan.strength(),
        )?,
        ImageOperation::Inpaint => request.inpaint(
            source.ok_or_else(|| Error::Invalid("source image is missing".to_owned()))?,
            mask.ok_or_else(|| Error::Invalid("mask image is missing".to_owned()))?,
            plan.strength(),
        )?,
        ImageOperation::Outpaint => request.outpaint(
            source.ok_or_else(|| Error::Invalid("expanded source image is missing".to_owned()))?,
            mask.ok_or_else(|| Error::Invalid("mask image is missing".to_owned()))?,
            plan.strength(),
        )?,
        ImageOperation::VaeEncode | ImageOperation::VaeDecode => {
            return Err(Error::Invalid(
                "direct VAE operation passed diffusion preflight".to_owned(),
            ));
        }
    };
    if let Some((_, negative)) =
        input_for_role(plan, inputs, ImageBufferRole::NegativeConditioning)?
    {
        request = request.with_negative_prompt(
            std::str::from_utf8(negative.bytes())
                .map_err(|_| Error::Invalid("negative conditioning must be UTF-8".to_owned()))?,
        )?;
    }
    for (index, binding) in plan.inputs.iter().enumerate() {
        if binding.role == ImageBufferRole::ReferenceImage {
            request =
                request.with_reference(pixels_from(&binding.layout, inputs[index].bytes())?)?;
        }
    }
    let mut loras = Vec::with_capacity(plan.loras.len());
    for lora in &plan.loras {
        let (binding, input) = input_for_slot_with_binding(plan, inputs, lora.input_slot)?;
        let path = paths
            .resolve_path(binding, input)
            .map_err(|message| Error::Invalid(format!("LoRA path resolution failed: {message}")))?;
        let high_noise = lora.target == lora_target_v1(true);
        let scale = lora.scales.points[0].scale();
        loras.push(
            LoraBinding::new(path, binding.buffer.identity.clone(), scale)?
                .with_high_noise(high_noise),
        );
    }
    request.with_loras(loras)
}

fn pixels_for_binding<'a>(
    plan: &ImageExecutionPlan,
    inputs: &'a [InputBuffer<'a>],
    binding: &ImageBufferBinding,
) -> Result<ImagePixels<'a>> {
    let input = input_for_slot(plan, inputs, binding.slot)?;
    pixels_from(&binding.layout, input.bytes())
}

fn pixels_from<'a>(layout: &ImageBufferLayout, bytes: &'a [u8]) -> Result<ImagePixels<'a>> {
    match layout {
        ImageBufferLayout::Rgb8 {
            width,
            height,
            row_stride,
        } if *row_stride == u64::from(*width) * 3 => ImagePixels::rgb8(bytes, *width, *height),
        ImageBufferLayout::Rgba8 {
            width,
            height,
            row_stride,
        } if *row_stride == u64::from(*width) * 4 => ImagePixels::rgba8(bytes, *width, *height),
        ImageBufferLayout::Gray8 {
            width,
            height,
            row_stride,
        } if *row_stride == u64::from(*width) => ImagePixels::gray8(bytes, *width, *height),
        _ => Err(Error::Invalid(
            "native image inputs must be tightly packed RGB8, RGBA8, or Gray8".to_owned(),
        )),
    }
}

fn input_for_role<'p, 'a>(
    plan: &'p ImageExecutionPlan,
    inputs: &'a [InputBuffer<'a>],
    role: ImageBufferRole,
) -> Result<Option<(&'p ImageBufferBinding, &'a InputBuffer<'a>)>> {
    plan.inputs
        .iter()
        .enumerate()
        .find(|(_, binding)| binding.role == role)
        .map(|(index, binding)| {
            inputs
                .get(index)
                .map(|input| (binding, input))
                .ok_or_else(|| Error::Invalid("input binding index is absent".to_owned()))
        })
        .transpose()
}

fn input_for_slot<'a>(
    plan: &ImageExecutionPlan,
    inputs: &'a [InputBuffer<'a>],
    slot: u16,
) -> Result<&'a InputBuffer<'a>> {
    input_for_slot_with_binding(plan, inputs, slot).map(|(_, input)| input)
}

fn input_for_slot_with_binding<'p, 'a>(
    plan: &'p ImageExecutionPlan,
    inputs: &'a [InputBuffer<'a>],
    slot: u16,
) -> Result<(&'p ImageBufferBinding, &'a InputBuffer<'a>)> {
    let index = plan
        .inputs
        .iter()
        .position(|binding| binding.slot == slot)
        .ok_or_else(|| Error::Invalid("input slot is not bound".to_owned()))?;
    Ok((&plan.inputs[index], &inputs[index]))
}

fn execute_composites(
    plan: &ImageExecutionPlanV3,
    inputs: &[InputBuffer<'_>],
    primary: &[u8],
) -> Result<(
    Vec<Vec<u8>>,
    Vec<logit_loom_diffusion::ImageCompositeReceipt>,
)> {
    let mut values: Vec<Vec<u8>> = Vec::with_capacity(plan.composites.len());
    let mut receipts = Vec::with_capacity(plan.composites.len());
    for stage in &plan.composites {
        match stage.operation {
            ImageCompositeOperation::MaskBlend {
                base,
                overlay,
                mask_slot,
            } => {
                let base = resolve_image(plan, inputs, primary, &values, base)?;
                let overlay = resolve_image(plan, inputs, primary, &values, overlay)?;
                let mask = input_for_slot(&plan.primary, inputs, mask_slot)?.bytes();
                let mut output = vec![0_u8; primary.len()];
                let mut receipt = mask_blend_rgb8(base, overlay, mask, &mut output)
                    .map_err(logit_loom_diffusion::Error::from)?;
                receipt.stage = stage.stage;
                values.push(output);
                receipts.push(receipt);
            }
        }
    }
    Ok((values, receipts))
}

fn resolve_image<'a>(
    plan: &ImageExecutionPlanV3,
    inputs: &'a [InputBuffer<'a>],
    primary: &'a [u8],
    composites: &'a [Vec<u8>],
    source: ImageValueSource,
) -> Result<&'a [u8]> {
    match source {
        ImageValueSource::Primary => Ok(primary),
        ImageValueSource::Input { slot } => {
            Ok(input_for_slot(&plan.primary, inputs, slot)?.bytes())
        }
        ImageValueSource::Composite { stage } => composites
            .get(usize::from(stage))
            .map(Vec::as_slice)
            .ok_or_else(|| Error::Invalid("composite source was not produced".to_owned())),
    }
}

fn preflight_route_payloads(
    plan: &ImageExecutionPlanV3,
    inputs: &[InputBuffer<'_>],
    outputs: &[OutputBuffer<'_>],
    primary: &[u8],
    composites: &[Vec<u8>],
    checkpoint: Option<&[u8]>,
) -> Result<Vec<ImageOutputReceiptV3>> {
    let available = available_route_count(plan, checkpoint)?;
    let mut receipts = Vec::with_capacity(available);
    for (index, (route, output)) in plan.outputs.iter().zip(outputs).take(available).enumerate() {
        let payload = route_payload(plan, inputs, primary, composites, checkpoint, route.source)?;
        let allocation_len = usize::try_from(output.specification().byte_length).map_err(|_| {
            Error::Invalid(format!("output route {index} allocation exceeds usize"))
        })?;
        if payload.len() > allocation_len {
            return Err(Error::Invalid(format!(
                "output route {index} payload exceeds its allocation"
            )));
        }
        if matches!(route.source, ImageOutputSource::Image { .. })
            && payload.len() != allocation_len
        {
            return Err(Error::Invalid(format!(
                "image output route {index} must fill its exact allocation"
            )));
        }
        receipts.push(ImageOutputReceiptV3 {
            route: u16::try_from(index)
                .map_err(|_| Error::Invalid("output route exceeds u16".to_owned()))?,
            allocation: route.buffer.identity.clone(),
            content: Digest::of_bytes("image-execution-output-bytes-v2", payload),
            bytes_written: u64::try_from(payload.len())
                .map_err(|_| Error::Invalid("output byte count exceeds u64".to_owned()))?,
        });
    }
    Ok(receipts)
}

fn write_routes(
    plan: &ImageExecutionPlanV3,
    inputs: &[InputBuffer<'_>],
    outputs: &mut [OutputBuffer<'_>],
    primary: &[u8],
    composites: &[Vec<u8>],
    checkpoint: Option<&[u8]>,
) -> Result<()> {
    let available = available_route_count(plan, checkpoint)?;
    for (route, output) in plan.outputs.iter().zip(outputs).take(available) {
        let payload = route_payload(plan, inputs, primary, composites, checkpoint, route.source)?;
        output.bytes_mut()[..payload.len()].copy_from_slice(payload);
        output
            .set_written(payload.len())
            .map_err(|error| Error::Output(error.to_string()))?;
    }
    Ok(())
}

fn available_route_count(plan: &ImageExecutionPlanV3, checkpoint: Option<&[u8]>) -> Result<usize> {
    if checkpoint.is_some() || plan.checkpoint.capture_after_step.is_none() {
        return Ok(plan.outputs.len());
    }
    let count = plan
        .outputs
        .len()
        .checked_sub(1)
        .ok_or_else(|| Error::Incompatible("checkpoint route is absent".to_owned()))?;
    if !matches!(plan.outputs[count].source, ImageOutputSource::Checkpoint) {
        return Err(Error::Incompatible(
            "unavailable checkpoint is not the final route".to_owned(),
        ));
    }
    Ok(count)
}

fn route_payload<'a>(
    plan: &ImageExecutionPlanV3,
    inputs: &'a [InputBuffer<'a>],
    primary: &'a [u8],
    composites: &'a [Vec<u8>],
    checkpoint: Option<&'a [u8]>,
    source: ImageOutputSource,
) -> Result<&'a [u8]> {
    match source {
        ImageOutputSource::Image { source } => {
            resolve_image(plan, inputs, primary, composites, source)
        }
        ImageOutputSource::Checkpoint => checkpoint
            .ok_or_else(|| Error::Incompatible("checkpoint output was not captured".to_owned())),
    }
}

fn step_selected(selector: &StepSelector, step: u32) -> bool {
    match selector {
        StepSelector::All => true,
        StepSelector::Exact { steps } => steps.binary_search(&step).is_ok(),
    }
}

fn rgb_len(width: u32, height: u32) -> Result<usize> {
    usize::try_from(width)
        .ok()
        .and_then(|width| {
            usize::try_from(height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| Error::Invalid("RGB8 canvas length overflowed".to_owned()))
}

fn validate_checkpointed_generation_segment(
    prefix: Option<&AdvancedProgramGenerationReceipt>,
    continuation: &AdvancedProgramGenerationReceipt,
    restored: Option<&DiffusionCheckpoint>,
) -> Result<()> {
    match (prefix, restored) {
        (None, None) if continuation.generation.resumed_from.is_none() => Ok(()),
        (Some(prefix), Some(restored))
            if continuation.generation.resumed_from.as_ref() == Some(restored.receipt())
                && prefix.request == continuation.request
                && prefix.generation.profile == continuation.generation.profile
                && prefix.generation.native == continuation.generation.native
                && prefix.generation.request == continuation.generation.request
                && prefix.generation.plan == continuation.generation.plan
                && prefix.generation.program == continuation.generation.program
                && prefix
                    .generation
                    .steps
                    .last()
                    .and_then(|step| step.step_index.checked_add(1))
                    == Some(restored.receipt().next_step)
                && continuation
                    .generation
                    .steps
                    .first()
                    .is_some_and(|step| step.step_index == restored.receipt().next_step) =>
        {
            Ok(())
        }
        _ => Err(Error::Incompatible(
            "whole-image continuation segment lineage differs".to_owned(),
        )),
    }
}

fn finish_checkpointed_observations(segments: &[Vec<Digest>]) -> Result<Vec<Digest>> {
    let Some(first) = segments.first() else {
        return Err(Error::Incompatible(
            "whole-image execution has no observation segment".to_owned(),
        ));
    };
    if segments.iter().any(|segment| segment.len() != first.len()) {
        return Err(Error::Incompatible(
            "whole-image observation segment arity differs".to_owned(),
        ));
    }
    if segments.len() == 1 {
        return Ok(first.clone());
    }
    (0..first.len())
        .map(|observation| {
            let lineage = segments
                .iter()
                .map(|segment| segment[observation].clone())
                .collect::<Vec<_>>();
            Digest::of_serializable("sdcpp-checkpointed-plan-observation-v1", &lineage)
                .map_err(logit_loom_diffusion::Error::from)
                .map_err(Into::into)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        sync::atomic::{AtomicBool, Ordering},
    };

    use logit_loom_diffusion::{
        DiffusionSchedule, ImageBufferBinding, ImageCompositeStage, ImageOutputFormat,
        ImageOutputRoute, LoraStackEntry, ScalePoint, ScaleSchedule, TensorDType, TensorLayout,
        TensorSpec,
    };
    use logit_loom_executor::{BufferSpec, NeverCancel};

    use super::*;

    fn diffusion_plan() -> DiffusionPlan {
        let tensor = TensorSpec::new(
            vec![2, 2],
            TensorDType::F32,
            TensorLayout::DimensionZeroFastest,
            "host-f32:test",
        )
        .unwrap();
        let schedule =
            DiffusionSchedule::new(Digest::of_bytes("schedule", b"one"), vec![1.0, 0.0]).unwrap();
        let mut components = BTreeMap::new();
        components.insert("model".to_owned(), Digest::of_bytes("model", b"one"));
        DiffusionPlan::new(
            components,
            Digest::of_bytes("conditioning", b"one"),
            Digest::of_bytes("rng", b"one"),
            7,
            tensor,
            schedule,
        )
        .unwrap()
    }

    fn image_plan(
        plan: &DiffusionPlan,
        checkpoint: ImageCheckpointPlan,
        observations: Vec<ObservationRequest>,
    ) -> ImageExecutionPlanV3 {
        let mut inputs = vec![ImageBufferBinding {
            slot: 0,
            role: ImageBufferRole::PositiveConditioning,
            buffer: BufferSpec::new(
                Digest::of_bytes("prompt", b"one"),
                1,
                "text/plain; charset=utf-8",
            )
            .unwrap(),
            layout: ImageBufferLayout::Utf8,
        }];
        if checkpoint.restore_from.is_some() {
            inputs.push(ImageBufferBinding {
                slot: 1,
                role: ImageBufferRole::Checkpoint,
                buffer: BufferSpec::new(
                    Digest::of_bytes("checkpoint", b"one"),
                    128,
                    "application/octet-stream",
                )
                .unwrap(),
                layout: ImageBufferLayout::Opaque,
            });
        }
        let mut outputs = vec![ImageOutputRoute {
            source: ImageOutputSource::Image {
                source: ImageValueSource::Primary,
            },
            buffer: BufferSpec::new(Digest::of_bytes("output", b"one"), 6, "image/rgb").unwrap(),
            layout: ImageBufferLayout::Rgb8 {
                width: 2,
                height: 1,
                row_stride: 6,
            },
        }];
        if checkpoint.capture_after_step.is_some() {
            outputs.push(ImageOutputRoute {
                source: ImageOutputSource::Checkpoint,
                buffer: BufferSpec::new(
                    Digest::of_bytes("checkpoint-output", b"one"),
                    1_024,
                    "application/octet-stream",
                )
                .unwrap(),
                layout: ImageBufferLayout::Opaque,
            });
        }
        ImageExecutionPlanV3 {
            primary: ImageExecutionPlan {
                profile: Digest::of_bytes("profile", b"one"),
                load: Digest::of_bytes("load", b"one"),
                operation: ImageOperation::TextToImage,
                width: 2,
                height: 1,
                output_format: ImageOutputFormat::Rgb8,
                seed: SeedSelection::Fixed { seed: plan.seed },
                rng: plan.rng.clone(),
                placement: Digest::of_bytes("placement", b"one"),
                schedule: Some(plan.schedule.clone()),
                guidance_scale_bits: 1.0_f32.to_bits(),
                strength_bits: 1.0_f32.to_bits(),
                inputs,
                loras: Vec::new(),
                operators: Vec::new(),
                observations,
            },
            checkpoint,
            composites: Vec::<ImageCompositeStage>::new(),
            outputs,
            cleanup: ImageCleanupPolicy::RetainSession,
        }
    }

    #[test]
    fn channel_bias_control_encoding_and_identity_are_exact() {
        let plan = diffusion_plan();
        let control = ChannelBiasControlV1::new(1, 0, 0.25, 0.5).unwrap();
        assert_eq!(
            ChannelBiasControlV1::from_control_bytes(&control.to_control_bytes()).unwrap(),
            control
        );
        let steps = StepSelector::Exact { steps: vec![0] };
        let implementation = control.implementation_for(&plan.tensor, &steps).unwrap();
        let invocation = OperatorInvocation {
            schema: channel_bias_schema_v1(),
            implementation,
            selector: TensorSelector::SchedulerState,
            steps,
            controls: control.to_control_bytes(),
        };
        assert!(InstalledChannelBias::from_invocation(&invocation, &plan).is_ok());
    }

    #[test]
    fn installed_bias_rejects_identity_or_tensor_mismatch() {
        let plan = diffusion_plan();
        let control = ChannelBiasControlV1::new(1, 0, 0.25, 0.5).unwrap();
        let mut invocation = OperatorInvocation {
            schema: channel_bias_schema_v1(),
            implementation: Digest::of_bytes("wrong", b"identity"),
            selector: TensorSelector::SchedulerState,
            steps: StepSelector::All,
            controls: control.to_control_bytes(),
        };
        assert!(InstalledChannelBias::from_invocation(&invocation, &plan).is_err());
        invocation.implementation = control
            .implementation_for(&plan.tensor, &invocation.steps)
            .unwrap();
        invocation.controls = ChannelBiasControlV1::new(9, 0, 0.25, 0.5)
            .unwrap()
            .to_control_bytes();
        assert!(InstalledChannelBias::from_invocation(&invocation, &plan).is_err());
    }

    #[test]
    fn model_block_residual_control_encoding_and_identity_are_exact() {
        let control = ModelBlockResidualScaleControlV1::new(0.0, 1.0).unwrap();
        assert_eq!(
            ModelBlockResidualScaleControlV1::from_control_bytes(&control.to_control_bytes())
                .unwrap(),
            control
        );
        let selector = TensorSelector::ModelBlock {
            component: "krea2".to_owned(),
            block: 9,
            site: "residual".to_owned(),
        };
        let steps = StepSelector::Exact { steps: vec![0] };
        let invocation = OperatorInvocation {
            schema: model_block_residual_scale_schema_v1(),
            implementation: control.implementation_for(&selector, &steps).unwrap(),
            selector,
            steps,
            controls: control.to_control_bytes(),
        };
        let installed = InstalledModelBlockResidualScale::from_invocation(&invocation).unwrap();
        assert_eq!(installed.block, 9);
        assert_eq!(installed.scale.to_bits(), 0.0_f32.to_bits());
        assert_eq!(installed.steps, StepSelector::Exact { steps: vec![0] });
    }

    #[test]
    fn model_block_residual_control_rejects_uninstalled_or_unbounded_inputs() {
        assert!(ModelBlockResidualScaleControlV1::new(17.0, 17.0).is_err());
        assert!(ModelBlockResidualScaleControlV1::from_control_bytes(&[0; 7]).is_err());

        let control = ModelBlockResidualScaleControlV1::new(0.5, 1.0).unwrap();
        let selector = TensorSelector::ModelBlock {
            component: "krea2".to_owned(),
            block: 10,
            site: "attention".to_owned(),
        };
        let steps = StepSelector::All;
        let invocation = OperatorInvocation {
            schema: model_block_residual_scale_schema_v1(),
            implementation: control.implementation_for(&selector, &steps).unwrap(),
            selector,
            steps,
            controls: control.to_control_bytes(),
        };
        assert!(InstalledModelBlockResidualScale::from_invocation(&invocation).is_err());
    }

    #[test]
    fn unsupported_lora_schedule_fails_before_native_lowering() {
        let input = ImageBufferBinding {
            slot: 0,
            role: ImageBufferRole::Lora,
            buffer: BufferSpec::new(
                Digest::of_bytes("lora", b"one"),
                1,
                "application/octet-stream",
            )
            .unwrap(),
            layout: ImageBufferLayout::Opaque,
        };
        let plan = ImageExecutionPlan {
            profile: Digest::of_bytes("profile", b"one"),
            load: Digest::of_bytes("load", b"one"),
            operation: ImageOperation::TextToImage,
            width: 64,
            height: 64,
            output_format: ImageOutputFormat::Rgb8,
            seed: SeedSelection::Fixed { seed: 7 },
            rng: Digest::of_bytes("rng", b"one"),
            placement: Digest::of_bytes("placement", b"one"),
            schedule: Some(
                DiffusionSchedule::new(Digest::of_bytes("schedule", b"one"), vec![1.0, 0.5, 0.0])
                    .unwrap(),
            ),
            guidance_scale_bits: 1.0_f32.to_bits(),
            strength_bits: 1.0_f32.to_bits(),
            inputs: vec![input],
            loras: vec![LoraStackEntry {
                input_slot: 0,
                target: lora_target_v1(false),
                scales: ScaleSchedule {
                    points: vec![
                        ScalePoint::new(0, 1.0).unwrap(),
                        ScalePoint::new(1, 0.5).unwrap(),
                    ],
                },
            }],
            operators: Vec::new(),
            observations: Vec::new(),
        };
        assert!(validate_supported_mechanics(&plan).is_err());
    }

    #[test]
    fn lowering_preserves_independent_reference_geometry_and_bytes() {
        let native = diffusion_plan();
        let mut plan = image_plan(&native, ImageCheckpointPlan::default(), Vec::new());
        plan.primary.inputs.push(ImageBufferBinding {
            slot: 1,
            role: ImageBufferRole::ReferenceImage,
            buffer: BufferSpec::new(
                Digest::of_bytes("reference", b"independent-geometry"),
                6,
                "image/rgb",
            )
            .unwrap(),
            layout: ImageBufferLayout::Rgb8 {
                width: 1,
                height: 2,
                row_stride: 3,
            },
        });
        plan.primary.validate().unwrap();

        let prompt = *b"x";
        let reference = [1_u8, 2, 3, 4, 5, 6];
        let inputs = [
            InputBuffer::new(&plan.primary.inputs[0].buffer, &prompt).unwrap(),
            InputBuffer::new(&plan.primary.inputs[1].buffer, &reference).unwrap(),
        ];
        let request = lower_request(&plan.primary, &inputs, &mut RejectArtifactPaths).unwrap();
        let lowered = request.references()[0];
        assert_eq!(lowered.width(), 1);
        assert_eq!(lowered.height(), 2);
        assert_eq!(lowered.channels(), 3);
        assert_eq!(lowered.bytes(), reference);
    }

    #[test]
    fn stale_checkpoint_backend_fails_at_program_begin() {
        let native = diffusion_plan();
        let context = StepContext::for_plan(&native, 0).unwrap();
        let checkpoint = DiffusionCheckpoint::capture(
            &native,
            &Digest::of_bytes("backend", b"old"),
            &context,
            &[0.0; 4],
        )
        .unwrap();
        let plan = image_plan(
            &native,
            ImageCheckpointPlan {
                restore_from: Some(1),
                capture_after_step: None,
            },
            Vec::new(),
        );
        let mut program = PlanProgram::new(
            &plan,
            &NeverCancel,
            Some(checkpoint),
            Digest::of_bytes("backend", b"new"),
        )
        .unwrap();
        assert!(program.begin(&native).is_err());
        assert!(program.actual_plan.is_none());
    }

    #[test]
    fn cancellation_is_reported_after_the_same_observation_boundary() {
        struct ToggleCancel(AtomicBool);

        impl CancellationProbe for ToggleCancel {
            fn is_cancelled(&self) -> bool {
                self.0.load(Ordering::Acquire)
            }
        }

        let native = diffusion_plan();
        let plan = image_plan(
            &native,
            ImageCheckpointPlan::default(),
            vec![ObservationRequest {
                selector: TensorSelector::SchedulerState,
                steps: StepSelector::All,
                kind: ObservationKind::Digest,
            }],
        );
        let cancellation = ToggleCancel(AtomicBool::new(false));
        let mut stopped = PlanProgram::new(
            &plan,
            &cancellation,
            None,
            Digest::of_bytes("backend", b"one"),
        )
        .unwrap();
        stopped.begin(&native).unwrap();
        let context = StepContext::for_plan(&native, 0).unwrap();
        let mut state = [0.0, 1.0, 2.0, 3.0];
        stopped.intervene(&context, &mut state).unwrap();
        cancellation.0.store(true, Ordering::Release);
        assert_eq!(
            stopped.observe(&context, &state).unwrap(),
            ControlFlow::Stop
        );

        let mut continued = PlanProgram::new(
            &plan,
            &NeverCancel,
            None,
            Digest::of_bytes("backend", b"one"),
        )
        .unwrap();
        continued.begin(&native).unwrap();
        assert_eq!(
            continued.observe(&context, &state).unwrap(),
            ControlFlow::Continue
        );
        assert_eq!(
            stopped.finish_observations(),
            continued.finish_observations()
        );
    }

    #[test]
    fn checkpointed_plan_suspends_with_exact_post_intervention_state() {
        struct Suspend(AtomicBool);

        impl CheckpointSuspensionProbe for Suspend {
            fn is_suspension_requested(&self) -> bool {
                self.0.load(Ordering::Acquire)
            }
        }

        let mut native = diffusion_plan();
        native.schedule = DiffusionSchedule::new(
            Digest::of_bytes("schedule", b"checkpointed-plan"),
            vec![2.0, 1.0, 0.0],
        )
        .unwrap();
        let plan = image_plan(&native, ImageCheckpointPlan::default(), Vec::new());
        let backend = Digest::of_bytes("backend", b"checkpointed-plan");
        let suspension = Suspend(AtomicBool::new(true));
        let mut program = PlanProgram::checkpointed(
            &plan,
            &NeverCancel,
            Some(&suspension),
            PlanCheckpointLineage::default(),
            None,
            backend.clone(),
        )
        .unwrap();
        program.begin(&native).unwrap();
        let context = StepContext::for_plan(&native, 0).unwrap();
        let mut state = [0.0, 1.0, 2.0, 3.0];
        program.intervene(&context, &mut state).unwrap();
        assert_eq!(
            program.observe(&context, &state).unwrap(),
            ControlFlow::Stop
        );
        let checkpoint = program.take_suspension_checkpoint().unwrap();
        assert_eq!(checkpoint.receipt().next_step, 1);
        assert_eq!(checkpoint.receipt().backend, backend);
        let expected = state
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(checkpoint.state_bytes(), expected);
    }

    #[test]
    fn checkpointed_observations_bind_segments_without_changing_one_segment() {
        let first = vec![Digest::of_bytes("observation", b"first")];
        let second = vec![Digest::of_bytes("observation", b"second")];
        assert_eq!(
            finish_checkpointed_observations(std::slice::from_ref(&first)).unwrap(),
            first
        );
        let combined = finish_checkpointed_observations(&[first.clone(), second]).unwrap();
        assert_eq!(combined.len(), 1);
        assert_ne!(combined, first);
        assert!(finish_checkpointed_observations(&[first, Vec::new()]).is_err());
    }

    #[test]
    fn checkpoint_route_remains_unavailable_before_capture_boundary() {
        let native = diffusion_plan();
        let plan = image_plan(
            &native,
            ImageCheckpointPlan {
                restore_from: None,
                capture_after_step: Some(0),
            },
            Vec::new(),
        );
        assert_eq!(available_route_count(&plan, None).unwrap(), 1);
        assert_eq!(
            available_route_count(&plan, Some(b"checkpoint")).unwrap(),
            2
        );
        let mut program = PlanProgram::new(
            &plan,
            &NeverCancel,
            None,
            Digest::of_bytes("backend", b"one"),
        )
        .unwrap();
        program.begin(&native).unwrap();
        let terminal = ImageTerminal::CancelledAfterStep { step: 0 };
        assert!(program.validate_terminal(&terminal).is_err());
        let context = StepContext::for_plan(&native, 0).unwrap();
        program
            .intervene(&context, &mut [0.0, 1.0, 2.0, 3.0])
            .unwrap();
        assert!(program.validate_terminal(&terminal).is_ok());
    }
}

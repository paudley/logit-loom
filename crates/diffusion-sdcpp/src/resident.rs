// SPDX-License-Identifier: MIT OR Apache-2.0

//! Safe execution driver for resident staged image programs.

use std::collections::HashMap;

use logit_loom_diffusion::{
    Digest, ImageCleanupPolicy, ImageProgramCleanupDispositionV1, ImageProgramMeasurementsV1,
    ImageProgramOutputReceiptV1, ImageProgramOutputSourceV1, ImageProgramPlanV1,
    ImageProgramReceiptV1, ImageProgramStageReceiptV1, ImageProgramStageV1, ImageProgramTerminalV1,
    ImageProgramValueMeasurementV1, ImageProgramValuePlacementV1, image_program_value_content,
};
use logit_loom_executor::{
    CancellationProbe, ClassifiedExecutionError, FailureDisposition, InputBuffer, OutputBuffer,
};

use crate::{
    CheckpointSuspensionProbe, Error, ModelBlockApplicationReceiptV1, ModelBlockApplicationV1,
    Result,
};

/// One completed native or deterministic program stage.
#[derive(Clone, Debug)]
pub struct ResidentProgramCompletedStage {
    /// Exact deterministic stage receipt.
    pub receipt: ImageProgramStageReceiptV1,
    /// Exact native model-block application records in operator order.
    pub model_block_applications: Vec<ModelBlockApplicationV1>,
    /// Observed wall time in nanoseconds.
    pub wall_time_ns: u64,
    /// Native compute time in nanoseconds when the backend exposes it.
    pub native_time_ns: Option<u64>,
    /// Placement and transfer accounting for values published by this stage.
    pub values: Vec<ImageProgramValueMeasurementV1>,
}

/// Exact result of attempting one resident program stage.
#[derive(Debug)]
pub enum ResidentProgramStageTerminal<C> {
    /// The stage published every declared output.
    Completed(ResidentProgramCompletedStage),
    /// Cooperative cancellation stopped a native stage after one exact Euler
    /// boundary and published no stage outputs.
    CancelledAfterStep {
        /// Zero-based completed Euler transition.
        step: u32,
    },
    /// Checkpoint suspension stopped a native stage after one exact safe
    /// boundary and published no stage outputs.
    SuspendedAfterStep {
        /// Zero-based completed backend transition.
        step: u32,
        /// Backend-private exact continuation for this stage.
        checkpoint: C,
    },
}

/// Confirmed backend finalization for one request-scoped resident program.
#[derive(Clone, Debug)]
pub struct ResidentProgramFinish {
    /// Cleanup disposition for the exact runtime epoch.
    pub cleanup: ImageProgramCleanupDispositionV1,
    /// Observed peak value-arena bytes.
    pub peak_arena_bytes: u64,
}

/// Backend boundary used by [`ResidentImageProgramDriver`].
///
/// Implementations keep native handles private. Logical values are addressed
/// only by the bounded identifiers already validated in
/// [`ImageProgramPlanV1`].
pub trait ResidentImageProgramBackend {
    /// Backend-private continuation captured at a safe in-stage boundary.
    type StageCheckpoint: std::fmt::Debug;
    /// Backend-private bounded host representation of one live logical value.
    type ValueCheckpoint: std::fmt::Debug;

    /// Returns the exact backend build/runtime identity.
    ///
    /// # Errors
    ///
    /// Returns an error when the identity cannot be encoded exactly.
    fn backend_identity(&self) -> Result<Digest>;

    /// Returns the current runtime epoch to which private handles are bound.
    fn runtime_epoch(&self) -> u64;

    /// Rejects unsupported mechanics and incompatible load identities without
    /// allocating request-local state or calling a model.
    ///
    /// # Errors
    ///
    /// Returns a rejected error for unsupported or incompatible mechanics.
    fn validate_program(&self, plan: &ImageProgramPlanV1) -> Result<()>;

    /// Creates one request-local arena and imports every external input.
    ///
    /// Returned measurements must describe the imported logical values in
    /// input order.
    ///
    /// # Errors
    ///
    /// Returns a classified import, allocation, or native-arena error.
    fn begin_program(
        &mut self,
        plan: &ImageProgramPlanV1,
        inputs: &[InputBuffer<'_>],
    ) -> Result<Vec<ImageProgramValueMeasurementV1>>;

    /// Executes one stage against private resident values.
    ///
    /// # Errors
    ///
    /// Returns a classified stage or native-state error.
    fn execute_stage(
        &mut self,
        plan: &ImageProgramPlanV1,
        stage: &ImageProgramStageV1,
        cancellation: &dyn CancellationProbe,
        suspension: &dyn CheckpointSuspensionProbe,
        checkpoint: Option<Self::StageCheckpoint>,
    ) -> Result<ResidentProgramStageTerminal<Self::StageCheckpoint>>;

    /// Reconstructs one bounded intermediate value captured from an earlier
    /// arena of the same exact program.
    ///
    /// # Errors
    ///
    /// Returns an error unless the value bytes match its declared logical
    /// representation and can be imported into the active arena.
    fn checkpoint_value(
        &mut self,
        plan: &ImageProgramPlanV1,
        value: u16,
    ) -> Result<ResidentProgramValueCheckpoint<Self::ValueCheckpoint>>;

    /// Reconstructs one value captured by [`Self::checkpoint_value`].
    ///
    /// # Errors
    ///
    /// Returns an error unless the checkpoint is compatible with the active
    /// arena and exact program value.
    fn restore_value(
        &mut self,
        plan: &ImageProgramPlanV1,
        checkpoint: ResidentProgramValueCheckpoint<Self::ValueCheckpoint>,
    ) -> Result<ImageProgramValueMeasurementV1>;

    /// Copies one explicitly routed serializable value directly into the
    /// caller-owned output allocation and returns its initialized length.
    ///
    /// The driver does not publish that initialized prefix until every value
    /// is verified and program cleanup succeeds.
    ///
    /// # Errors
    ///
    /// Returns an error for an absent, stale, released, incompatible, or
    /// under-sized output allocation.
    fn materialize_value(
        &mut self,
        plan: &ImageProgramPlanV1,
        value: u16,
        output: &mut [u8],
    ) -> Result<usize>;

    /// Releases one logical value at its canonical final-consumer boundary.
    ///
    /// # Errors
    ///
    /// Returns an error for an absent, stale, or already released value.
    fn release_value(&mut self, value: u16) -> Result<()>;

    /// Invalidates the request arena, clears callbacks and adapter state, and
    /// applies the plan's model-session cleanup policy.
    ///
    /// # Errors
    ///
    /// Returns a poisoning error unless every requested cleanup action is
    /// confirmed.
    fn finish_program(&mut self, cleanup: ImageCleanupPolicy) -> Result<ResidentProgramFinish>;

    /// Marks resident state unusable after an uncertain native boundary.
    fn poison(&mut self);
}

/// Complete deterministic and deployment evidence for one execution attempt.
#[derive(Clone, Debug)]
pub struct ResidentImageProgramExecution {
    /// Deterministic mechanical receipt.
    pub receipt: ImageProgramReceiptV1,
    /// Native model-block application evidence bound to `receipt`.
    pub model_block_applications: ModelBlockApplicationReceiptV1,
    /// Non-deterministic placement, transfer, and timing measurements.
    pub measurements: ImageProgramMeasurementsV1,
}

/// One bounded logical value retained across request-arena reconstruction.
#[derive(Debug)]
pub struct ResidentProgramValueCheckpoint<V> {
    value: u16,
    content: Digest,
    resident_bytes: u64,
    checkpoint: V,
}

impl<V> ResidentProgramValueCheckpoint<V> {
    /// Binds one backend-private bounded host value to its exact logical value
    /// and canonical content identity.
    ///
    /// # Errors
    ///
    /// Returns an error when the retained host representation is empty.
    pub fn new(value: u16, content: Digest, resident_bytes: u64, checkpoint: V) -> Result<Self> {
        if resident_bytes == 0 {
            return Err(Error::Invalid(
                "resident value checkpoint is empty".to_owned(),
            ));
        }
        Ok(Self {
            value,
            content,
            resident_bytes,
            checkpoint,
        })
    }

    /// Returns the logical program value.
    #[must_use]
    pub const fn value(&self) -> u16 {
        self.value
    }

    /// Returns the canonical content identity of the captured native bytes.
    #[must_use]
    pub const fn content(&self) -> &Digest {
        &self.content
    }

    /// Returns the bounded host bytes retained by the backend representation.
    #[must_use]
    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    /// Removes the backend-private representation.
    pub fn into_checkpoint(self) -> V {
        self.checkpoint
    }
}

/// Exact in-memory continuation for one resident image program.
///
/// Native handles never enter this value. Intermediate values are copied into
/// bounded host bytes and the old request arena is confirmed closed before the
/// continuation is returned.
#[derive(Debug)]
pub struct ResidentImageProgramContinuation<C, V> {
    state: DriverState,
    next_stage: u16,
    stage_checkpoint: Option<C>,
    values: Vec<ResidentProgramValueCheckpoint<V>>,
}

impl<C, V> ResidentImageProgramContinuation<C, V> {
    /// Returns the next stage cursor. A checkpoint within a stage names that
    /// same stage; a checkpoint between stages names the following stage.
    #[must_use]
    pub const fn next_stage(&self) -> u16 {
        self.next_stage
    }
}

/// Result of checkpoint-aware resident program execution.
#[derive(Debug)]
pub enum CheckpointedResidentImageProgramExecution<C, V> {
    /// The request reached one terminal result.
    Terminal(Box<ResidentImageProgramExecution>),
    /// The request yielded at an exact safe boundary with no visible outputs.
    Suspended(Box<ResidentImageProgramContinuation<C, V>>),
}

struct NeverSuspend;

impl CheckpointSuspensionProbe for NeverSuspend {
    fn is_suspension_requested(&self) -> bool {
        false
    }
}

/// Single-owner execution driver over one resident image-program backend.
#[derive(Debug)]
pub struct ResidentImageProgramDriver<B> {
    backend: B,
}

impl<B> ResidentImageProgramDriver<B> {
    /// Wraps one exclusively owned resident backend.
    pub const fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Returns the resident backend.
    pub const fn backend(&self) -> &B {
        &self.backend
    }

    /// Returns the mutable resident backend.
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    /// Removes the resident backend.
    pub fn into_backend(self) -> B {
        self.backend
    }
}

impl<B: ResidentImageProgramBackend> ResidentImageProgramDriver<B> {
    /// Executes one complete validated program over borrowed inputs and
    /// caller-owned outputs.
    ///
    /// Operational stage failures are returned as exact terminal receipts
    /// after confirmed cleanup. Validation, binding, output, or uncertain
    /// native-state failures return a classified adapter error.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid bindings, unsupported mechanics, uncertain
    /// cleanup, invalid backend evidence, or an output that cannot be
    /// atomically published.
    pub fn execute(
        &mut self,
        plan: &ImageProgramPlanV1,
        inputs: &[InputBuffer<'_>],
        outputs: &mut [OutputBuffer<'_>],
        cancellation: &dyn CancellationProbe,
    ) -> Result<ResidentImageProgramExecution> {
        match self.execute_checkpointed(plan, inputs, outputs, cancellation, &NeverSuspend, None)? {
            CheckpointedResidentImageProgramExecution::Terminal(execution) => Ok(*execution),
            CheckpointedResidentImageProgramExecution::Suspended(_) => Err(Error::Poisoned(
                "resident execution suspended without a suspension request".to_owned(),
            )),
        }
    }

    /// Executes or resumes one program with bounded checkpoint suspension.
    ///
    /// A suspended result has no externally initialized output. The backend's
    /// old request arena has been confirmed closed, and the returned
    /// continuation owns only bounded host values plus one backend-private
    /// in-stage checkpoint.
    ///
    /// # Errors
    ///
    /// Returns an error for a foreign continuation, uncertain cleanup,
    /// reconstruction failure, invalid backend evidence, or an output that
    /// cannot be atomically published.
    #[allow(clippy::too_many_lines)]
    pub fn execute_checkpointed(
        &mut self,
        plan: &ImageProgramPlanV1,
        inputs: &[InputBuffer<'_>],
        outputs: &mut [OutputBuffer<'_>],
        cancellation: &dyn CancellationProbe,
        suspension: &dyn CheckpointSuspensionProbe,
        continuation: Option<
            ResidentImageProgramContinuation<B::StageCheckpoint, B::ValueCheckpoint>,
        >,
    ) -> Result<CheckpointedResidentImageProgramExecution<B::StageCheckpoint, B::ValueCheckpoint>>
    {
        plan.validate().map_err(logit_loom_diffusion::Error::from)?;
        validate_buffers(plan, inputs, outputs)?;
        self.backend.validate_program(plan)?;
        let backend = self.backend.backend_identity()?;
        let runtime_epoch = self.backend.runtime_epoch();
        if cancellation.is_cancelled() {
            return cancelled_before_start(plan, backend, runtime_epoch)
                .map(Box::new)
                .map(CheckpointedResidentImageProgramExecution::Terminal);
        }

        let imported = match self.backend.begin_program(plan, inputs) {
            Ok(imported) => imported,
            Err(error) if error.disposition() == FailureDisposition::Rejected => {
                return Err(error);
            }
            Err(error) => {
                self.backend.poison();
                return Err(Error::Poisoned(format!(
                    "resident program arena initialization was uncertain: {}",
                    failure_identity(&error)
                )));
            }
        };
        let liveness = plan.liveness().map_err(logit_loom_diffusion::Error::from)?;
        let (mut state, next_stage, mut stage_checkpoint) = match continuation {
            Some(continuation) => {
                let expected_plan = plan.digest().map_err(logit_loom_diffusion::Error::from)?;
                if continuation.state.plan != expected_plan
                    || continuation.state.backend != backend
                    || usize::from(continuation.next_stage) > plan.stages.len()
                    || continuation.state.stages.len() != usize::from(continuation.next_stage)
                    || (continuation.stage_checkpoint.is_some()
                        && usize::from(continuation.next_stage) == plan.stages.len())
                {
                    self.backend.poison();
                    return Err(Error::Poisoned(
                        "resident program continuation identity or cursor differs".to_owned(),
                    ));
                }
                let mut state = continuation.state;
                state.runtime_epoch = runtime_epoch;
                if let Err(error) = self.restore_continuation(
                    plan,
                    &liveness.releases,
                    continuation.next_stage,
                    &state,
                    continuation.values,
                    imported,
                ) {
                    self.backend.poison();
                    return Err(Error::Poisoned(format!(
                        "resident program continuation reconstruction was uncertain: {}",
                        failure_identity(&error)
                    )));
                }
                (
                    state,
                    continuation.next_stage,
                    continuation.stage_checkpoint,
                )
            }
            None => (
                DriverState::new(plan, backend, runtime_epoch, imported)?,
                0,
                None,
            ),
        };

        if suspension.is_suspension_requested() {
            return self.suspend_program(
                plan,
                &liveness.releases,
                state,
                next_stage,
                stage_checkpoint,
            );
        }

        for stage in plan.stages.iter().skip(usize::from(next_stage)) {
            let stage_index = stage.stage;
            let resume = if stage_index == next_stage {
                stage_checkpoint.take()
            } else {
                None
            };
            match self
                .backend
                .execute_stage(plan, stage, cancellation, suspension, resume)
            {
                Ok(ResidentProgramStageTerminal::Completed(completed)) => {
                    state.push_completed(plan, completed)?;
                    if let Err(error) =
                        release_after(&mut self.backend, &liveness.releases, Some(stage_index))
                    {
                        self.backend.poison();
                        return Err(error);
                    }
                    if cancellation.is_cancelled() {
                        return self
                            .finish_terminal(
                                plan,
                                state,
                                ImageProgramTerminalV1::CancelledAfterStage { stage: stage_index },
                            )
                            .map(Box::new)
                            .map(CheckpointedResidentImageProgramExecution::Terminal);
                    }
                    let following = stage_index.checked_add(1).ok_or_else(|| {
                        Error::Poisoned("resident program stage cursor overflowed".to_owned())
                    })?;
                    if suspension.is_suspension_requested()
                        && usize::from(following) < plan.stages.len()
                    {
                        return self.suspend_program(
                            plan,
                            &liveness.releases,
                            state,
                            following,
                            None,
                        );
                    }
                }
                Ok(ResidentProgramStageTerminal::CancelledAfterStep { step }) => {
                    return self
                        .finish_terminal(
                            plan,
                            state,
                            ImageProgramTerminalV1::CancelledAfterStep {
                                stage: stage_index,
                                step,
                            },
                        )
                        .map(Box::new)
                        .map(CheckpointedResidentImageProgramExecution::Terminal);
                }
                Ok(ResidentProgramStageTerminal::SuspendedAfterStep { checkpoint, .. }) => {
                    return self.suspend_program(
                        plan,
                        &liveness.releases,
                        state,
                        stage_index,
                        Some(checkpoint),
                    );
                }
                Err(error) if error.disposition() == FailureDisposition::Rejected => {
                    let terminal = ImageProgramTerminalV1::FailedAtStage {
                        stage: stage_index,
                        failure: failure_identity(&error),
                    };
                    return self
                        .finish_terminal(plan, state, terminal)
                        .map(Box::new)
                        .map(CheckpointedResidentImageProgramExecution::Terminal);
                }
                Err(error) => {
                    self.backend.poison();
                    return Err(Error::Poisoned(format!(
                        "resident program stage {stage_index} left native state uncertain: {}",
                        failure_identity(&error)
                    )));
                }
            }
        }

        let initialized = match self.materialize_outputs(plan, inputs, outputs, &state) {
            Ok(initialized) => initialized,
            Err(error) => {
                if let Err(cleanup) = self.backend.finish_program(plan.cleanup) {
                    self.backend.poison();
                    return Err(Error::Poisoned(format!(
                        "resident program output failure was followed by uncertain cleanup: {}",
                        failure_identity(&cleanup)
                    )));
                }
                return Err(error);
            }
        };
        if let Err(error) = release_after(&mut self.backend, &liveness.releases, None) {
            self.backend.poison();
            return Err(error);
        }
        let finish = match self.backend.finish_program(plan.cleanup) {
            Ok(finish) => finish,
            Err(error) => {
                self.backend.poison();
                return Err(Error::Poisoned(format!(
                    "resident program final cleanup was uncertain: {}",
                    failure_identity(&error)
                )));
            }
        };
        state.finish(finish);
        complete_outputs(plan, outputs, initialized, state)
            .map(Box::new)
            .map(CheckpointedResidentImageProgramExecution::Terminal)
    }

    fn restore_continuation(
        &mut self,
        plan: &ImageProgramPlanV1,
        releases: &[logit_loom_diffusion::ImageProgramReleaseV1],
        next_stage: u16,
        state: &DriverState,
        values: Vec<ResidentProgramValueCheckpoint<B::ValueCheckpoint>>,
        imported: Vec<ImageProgramValueMeasurementV1>,
    ) -> Result<()> {
        let input_values = plan
            .inputs
            .iter()
            .map(|input| input.value)
            .collect::<Vec<_>>();
        validate_measurement_prefix(plan, &input_values, imported)?;

        let expected_values = live_intermediate_values(plan, releases, next_stage, state)?;
        if values.len() != expected_values.len() {
            return Err(Error::Incompatible(
                "resident continuation intermediate-value count differs".to_owned(),
            ));
        }
        for (checkpoint, expected) in values.into_iter().zip(expected_values) {
            if checkpoint.value() != expected
                || state.contents.get(&expected) != Some(checkpoint.content())
            {
                return Err(Error::Incompatible(format!(
                    "resident continuation value {expected} identity differs"
                )));
            }
            let measurement = self.backend.restore_value(plan, checkpoint)?;
            validate_measurement_prefix(plan, &[expected], vec![measurement])?;
        }

        for release in releases {
            if plan.inputs.iter().any(|input| input.value == release.value)
                && release.after_stage.is_some_and(|stage| stage < next_stage)
            {
                self.backend.release_value(release.value)?;
            }
        }
        Ok(())
    }

    fn suspend_program(
        &mut self,
        plan: &ImageProgramPlanV1,
        releases: &[logit_loom_diffusion::ImageProgramReleaseV1],
        mut state: DriverState,
        next_stage: u16,
        stage_checkpoint: Option<B::StageCheckpoint>,
    ) -> Result<CheckpointedResidentImageProgramExecution<B::StageCheckpoint, B::ValueCheckpoint>>
    {
        let values = match self.capture_live_values(plan, releases, next_stage, &state) {
            Ok(values) => values,
            Err(error) => {
                let _ = self
                    .backend
                    .finish_program(ImageCleanupPolicy::RetainSession);
                self.backend.poison();
                return Err(Error::Poisoned(format!(
                    "resident checkpoint capture was uncertain: {}",
                    failure_identity(&error)
                )));
            }
        };
        let finish = match self
            .backend
            .finish_program(ImageCleanupPolicy::RetainSession)
        {
            Ok(finish) => finish,
            Err(error) => {
                self.backend.poison();
                return Err(Error::Poisoned(format!(
                    "resident checkpoint arena cleanup was uncertain: {}",
                    failure_identity(&error)
                )));
            }
        };
        if finish.cleanup != ImageProgramCleanupDispositionV1::Retained {
            self.backend.poison();
            return Err(Error::Poisoned(
                "resident checkpoint cleanup advanced the runtime epoch".to_owned(),
            ));
        }
        state.observe_peak(finish.peak_arena_bytes);
        Ok(CheckpointedResidentImageProgramExecution::Suspended(
            Box::new(ResidentImageProgramContinuation {
                state,
                next_stage,
                stage_checkpoint,
                values,
            }),
        ))
    }

    fn capture_live_values(
        &mut self,
        plan: &ImageProgramPlanV1,
        releases: &[logit_loom_diffusion::ImageProgramReleaseV1],
        next_stage: u16,
        state: &DriverState,
    ) -> Result<Vec<ResidentProgramValueCheckpoint<B::ValueCheckpoint>>> {
        let values = live_intermediate_values(plan, releases, next_stage, state)?;
        let captured = values
            .into_iter()
            .map(|value| {
                let maximum = plan.values[usize::from(value)]
                    .spec
                    .maximum_bytes()
                    .map_err(logit_loom_diffusion::Error::from)?;
                let checkpoint = self.backend.checkpoint_value(plan, value)?;
                if checkpoint.value() != value
                    || checkpoint.resident_bytes() > maximum
                    || state.contents.get(&value) != Some(checkpoint.content())
                {
                    return Err(Error::Output(format!(
                        "resident checkpoint value {value} metadata differs"
                    )));
                }
                Ok(checkpoint)
            })
            .collect::<Result<Vec<_>>>()?;
        let total = captured.iter().try_fold(0_u64, |total, checkpoint| {
            total
                .checked_add(checkpoint.resident_bytes())
                .ok_or_else(|| {
                    Error::Output("resident checkpoint byte bound overflowed".to_owned())
                })
        })?;
        let arena_bound = plan
            .liveness()
            .map_err(logit_loom_diffusion::Error::from)?
            .peak_bytes;
        if total > arena_bound {
            return Err(Error::Output(
                "resident checkpoint exceeds the program arena bound".to_owned(),
            ));
        }
        Ok(captured)
    }

    fn materialize_outputs(
        &mut self,
        plan: &ImageProgramPlanV1,
        inputs: &[InputBuffer<'_>],
        outputs: &mut [OutputBuffer<'_>],
        state: &DriverState,
    ) -> Result<Vec<Option<usize>>> {
        let mut initialized = Vec::with_capacity(plan.outputs.len());
        for (route, output) in plan.outputs.iter().zip(outputs) {
            match route.source {
                ImageProgramOutputSourceV1::Value { value } => {
                    let written =
                        self.backend
                            .materialize_value(plan, value, output.bytes_mut())?;
                    if written == 0 || written > output.bytes_mut().len() {
                        return Err(Error::Output(format!(
                            "program output route {} has an invalid initialized length",
                            route.route
                        )));
                    }
                    let bytes = &output.bytes_mut()[..written];
                    let expected = state.contents.get(&value).ok_or_else(|| {
                        Error::Incompatible(format!(
                            "program output value {value} has no published content"
                        ))
                    })?;
                    let content_matches = plan
                        .inputs
                        .iter()
                        .position(|input| input.value == value)
                        .map_or_else(
                            || image_program_value_content(bytes) == *expected,
                            |input| inputs[input].bytes() == bytes,
                        );
                    if !content_matches {
                        return Err(Error::Incompatible(format!(
                            "program output value {value} content differs from its stage receipt"
                        )));
                    }
                    let length = u64::try_from(bytes.len())
                        .map_err(|_| Error::Output("program output exceeds u64".to_owned()))?;
                    if length > route.buffer.byte_length {
                        return Err(Error::Output(format!(
                            "program output route {} exceeds its allocation",
                            route.route
                        )));
                    }
                    initialized.push(Some(written));
                }
                ImageProgramOutputSourceV1::ProgramReceipt => initialized.push(None),
            }
        }
        Ok(initialized)
    }

    fn finish_terminal(
        &mut self,
        plan: &ImageProgramPlanV1,
        mut state: DriverState,
        terminal: ImageProgramTerminalV1,
    ) -> Result<ResidentImageProgramExecution> {
        match self.backend.finish_program(plan.cleanup) {
            Ok(finish) => state.finish(finish),
            Err(error) => {
                self.backend.poison();
                return Err(Error::Poisoned(format!(
                    "resident program cleanup was uncertain: {}",
                    failure_identity(&error)
                )));
            }
        }
        state.outcome(plan, terminal, Vec::new())
    }
}

#[derive(Debug)]
struct DriverState {
    plan: Digest,
    backend: Digest,
    runtime_epoch: u64,
    stages: Vec<ImageProgramStageReceiptV1>,
    model_block_applications: Vec<ModelBlockApplicationV1>,
    outputs: Vec<ImageProgramOutputReceiptV1>,
    contents: HashMap<u16, Digest>,
    wall_times: Vec<u64>,
    native_times: Vec<Option<u64>>,
    value_measurements: Vec<ImageProgramValueMeasurementV1>,
    cleanup: Option<ImageProgramCleanupDispositionV1>,
    peak_arena_bytes: Option<u64>,
}

impl DriverState {
    fn new(
        plan: &ImageProgramPlanV1,
        backend: Digest,
        runtime_epoch: u64,
        value_measurements: Vec<ImageProgramValueMeasurementV1>,
    ) -> Result<Self> {
        let contents = plan
            .inputs
            .iter()
            .map(|input| (input.value, input.buffer.identity.clone()))
            .collect();
        let input_values = plan
            .inputs
            .iter()
            .map(|input| input.value)
            .collect::<Vec<_>>();
        let value_measurements =
            validate_measurement_prefix(plan, &input_values, value_measurements)?;
        Ok(Self {
            plan: plan.digest().map_err(logit_loom_diffusion::Error::from)?,
            backend,
            runtime_epoch,
            stages: Vec::with_capacity(plan.stages.len()),
            model_block_applications: Vec::new(),
            outputs: Vec::with_capacity(plan.outputs.len()),
            contents,
            wall_times: Vec::with_capacity(plan.stages.len()),
            native_times: Vec::with_capacity(plan.stages.len()),
            value_measurements,
            cleanup: None,
            peak_arena_bytes: None,
        })
    }

    fn push_completed(
        &mut self,
        plan: &ImageProgramPlanV1,
        completed: ResidentProgramCompletedStage,
    ) -> Result<()> {
        let stage_index = self.stages.len();
        let expected = plan.stages.get(stage_index).ok_or_else(|| {
            Error::Incompatible("backend published more stages than the program".to_owned())
        })?;
        let expected_operation = expected
            .operation
            .digest()
            .map_err(logit_loom_diffusion::Error::from)?;
        if usize::from(completed.receipt.stage) != stage_index
            || completed.receipt.operation != expected_operation
        {
            return Err(Error::Incompatible(format!(
                "backend stage receipt {stage_index} differs from the exact operation"
            )));
        }
        for output in &completed.receipt.outputs {
            if self
                .contents
                .insert(output.value, output.content.clone())
                .is_some()
            {
                return Err(Error::Incompatible(format!(
                    "backend published program value {} more than once",
                    output.value
                )));
            }
        }
        let expected_values = completed
            .receipt
            .outputs
            .iter()
            .map(|output| output.value)
            .collect::<Vec<_>>();
        let values = validate_measurement_prefix(plan, &expected_values, completed.values)?;
        self.model_block_applications
            .extend(completed.model_block_applications);
        self.stages.push(completed.receipt);
        self.wall_times.push(completed.wall_time_ns);
        self.native_times.push(completed.native_time_ns);
        self.value_measurements.extend(values);
        Ok(())
    }

    fn finish(&mut self, finish: ResidentProgramFinish) {
        self.cleanup = Some(finish.cleanup);
        self.observe_peak(finish.peak_arena_bytes);
    }

    fn observe_peak(&mut self, peak_arena_bytes: u64) {
        self.peak_arena_bytes = Some(
            self.peak_arena_bytes
                .map_or(peak_arena_bytes, |prior| prior.max(peak_arena_bytes)),
        );
    }

    fn outcome(
        self,
        plan: &ImageProgramPlanV1,
        terminal: ImageProgramTerminalV1,
        outputs: Vec<ImageProgramOutputReceiptV1>,
    ) -> Result<ResidentImageProgramExecution> {
        let completed_stages = u16::try_from(self.stages.len())
            .map_err(|_| Error::Incompatible("completed stage count exceeds u16".to_owned()))?;
        let receipt = ImageProgramReceiptV1 {
            plan: self.plan.clone(),
            backend: self.backend.clone(),
            runtime_epoch: self.runtime_epoch,
            completed_stages,
            stages: self.stages,
            outputs,
            terminal,
            cleanup: self.cleanup.ok_or_else(|| {
                Error::Incompatible("program outcome is missing cleanup evidence".to_owned())
            })?,
        };
        receipt
            .validate_for(plan)
            .map_err(logit_loom_diffusion::Error::from)?;
        let program_receipt = receipt
            .digest_for(plan)
            .map_err(logit_loom_diffusion::Error::from)?;
        let model_block_applications = ModelBlockApplicationReceiptV1 {
            plan: self.plan.clone(),
            program_receipt,
            backend: self.backend.clone(),
            runtime_epoch: self.runtime_epoch,
            completed_stages,
            applications: self.model_block_applications,
        };
        model_block_applications.validate_for(plan, &receipt)?;
        let measurements = ImageProgramMeasurementsV1 {
            plan: self.plan,
            backend: self.backend,
            runtime_epoch: self.runtime_epoch,
            stage_wall_time_ns: self.wall_times,
            stage_native_time_ns: self.native_times,
            peak_arena_bytes: self.peak_arena_bytes.ok_or_else(|| {
                Error::Incompatible("program outcome is missing peak arena bytes".to_owned())
            })?,
            values: self.value_measurements,
        };
        measurements
            .validate_for(plan, &receipt)
            .map_err(logit_loom_diffusion::Error::from)?;
        Ok(ResidentImageProgramExecution {
            receipt,
            model_block_applications,
            measurements,
        })
    }
}

fn validate_buffers(
    plan: &ImageProgramPlanV1,
    inputs: &[InputBuffer<'_>],
    outputs: &[OutputBuffer<'_>],
) -> Result<()> {
    if inputs.len() != plan.inputs.len() || outputs.len() != plan.outputs.len() {
        return Err(Error::Invalid(
            "resident program input or output count differs from the exact plan".to_owned(),
        ));
    }
    for (declared, actual) in plan.inputs.iter().zip(inputs) {
        if &declared.buffer != actual.specification() {
            return Err(Error::Invalid(format!(
                "resident program input value {} metadata differs",
                declared.value
            )));
        }
    }
    for (declared, actual) in plan.outputs.iter().zip(outputs) {
        if &declared.buffer != actual.specification() {
            return Err(Error::Invalid(format!(
                "resident program output route {} metadata differs",
                declared.route
            )));
        }
    }
    Ok(())
}

fn cancelled_before_start(
    plan: &ImageProgramPlanV1,
    backend: Digest,
    runtime_epoch: u64,
) -> Result<ResidentImageProgramExecution> {
    let plan_digest = plan.digest().map_err(logit_loom_diffusion::Error::from)?;
    let receipt = ImageProgramReceiptV1 {
        plan: plan_digest.clone(),
        backend: backend.clone(),
        runtime_epoch,
        completed_stages: 0,
        stages: Vec::new(),
        outputs: Vec::new(),
        terminal: ImageProgramTerminalV1::CancelledBeforeStart,
        cleanup: ImageProgramCleanupDispositionV1::NotRequired,
    };
    receipt
        .validate_for(plan)
        .map_err(logit_loom_diffusion::Error::from)?;
    let program_receipt = receipt
        .digest_for(plan)
        .map_err(logit_loom_diffusion::Error::from)?;
    let model_block_applications = ModelBlockApplicationReceiptV1 {
        plan: plan_digest.clone(),
        program_receipt,
        backend: backend.clone(),
        runtime_epoch,
        completed_stages: 0,
        applications: Vec::new(),
    };
    model_block_applications.validate_for(plan, &receipt)?;
    let measurements = ImageProgramMeasurementsV1 {
        plan: plan_digest,
        backend,
        runtime_epoch,
        stage_wall_time_ns: Vec::new(),
        stage_native_time_ns: Vec::new(),
        peak_arena_bytes: 0,
        values: plan
            .inputs
            .iter()
            .map(|input| ImageProgramValueMeasurementV1 {
                value: input.value,
                placement: ImageProgramValuePlacementV1::Host,
                host_to_device_transfers: 0,
                host_to_device_bytes: 0,
                device_to_host_transfers: 0,
                device_to_host_bytes: 0,
            })
            .collect(),
    };
    measurements
        .validate_for(plan, &receipt)
        .map_err(logit_loom_diffusion::Error::from)?;
    Ok(ResidentImageProgramExecution {
        receipt,
        model_block_applications,
        measurements,
    })
}

fn validate_measurement_prefix(
    plan: &ImageProgramPlanV1,
    expected: &[u16],
    measurements: Vec<ImageProgramValueMeasurementV1>,
) -> Result<Vec<ImageProgramValueMeasurementV1>> {
    if measurements.len() != expected.len()
        || measurements
            .iter()
            .zip(expected)
            .any(|(measurement, expected)| measurement.value != *expected)
    {
        return Err(Error::Incompatible(
            "backend value measurements differ from the published value order".to_owned(),
        ));
    }
    for measurement in &measurements {
        if usize::from(measurement.value) >= plan.values.len() {
            return Err(Error::Incompatible(
                "backend measured a value outside the program".to_owned(),
            ));
        }
    }
    Ok(measurements)
}

fn live_intermediate_values(
    plan: &ImageProgramPlanV1,
    releases: &[logit_loom_diffusion::ImageProgramReleaseV1],
    next_stage: u16,
    state: &DriverState,
) -> Result<Vec<u16>> {
    let mut values = releases
        .iter()
        .filter(|release| {
            !plan.inputs.iter().any(|input| input.value == release.value)
                && state.contents.contains_key(&release.value)
                && release.after_stage.is_none_or(|stage| stage >= next_stage)
        })
        .map(|release| release.value)
        .collect::<Vec<_>>();
    values.sort_unstable();
    values.dedup();
    if values
        .iter()
        .any(|value| usize::from(*value) >= plan.values.len())
    {
        return Err(Error::Incompatible(
            "resident continuation names a value outside the program".to_owned(),
        ));
    }
    Ok(values)
}

fn release_after<B: ResidentImageProgramBackend>(
    backend: &mut B,
    releases: &[logit_loom_diffusion::ImageProgramReleaseV1],
    after_stage: Option<u16>,
) -> Result<()> {
    for release in releases
        .iter()
        .filter(|release| release.after_stage == after_stage)
    {
        backend.release_value(release.value)?;
    }
    Ok(())
}

fn complete_outputs(
    plan: &ImageProgramPlanV1,
    outputs: &mut [OutputBuffer<'_>],
    mut initialized: Vec<Option<usize>>,
    mut state: DriverState,
) -> Result<ResidentImageProgramExecution> {
    for (route, written) in plan.outputs.iter().zip(&initialized) {
        if let Some(written) = written {
            let ImageProgramOutputSourceV1::Value { value } = route.source else {
                return Err(Error::Incompatible(
                    "program receipt route unexpectedly carried an initialized prefix".to_owned(),
                ));
            };
            let content = state.contents.get(&value).cloned().ok_or_else(|| {
                Error::Incompatible(format!(
                    "program output value {value} has no published identity"
                ))
            })?;
            state.outputs.push(ImageProgramOutputReceiptV1 {
                route: route.route,
                allocation: route.buffer.identity.clone(),
                content: Some(content),
                bytes_written: u64::try_from(*written)
                    .map_err(|_| Error::Output("program output exceeds u64".to_owned()))?,
            });
        } else {
            state.outputs.push(ImageProgramOutputReceiptV1 {
                route: route.route,
                allocation: route.buffer.identity.clone(),
                content: None,
                bytes_written: 1,
            });
        }
    }
    let receipt_route = plan
        .outputs
        .iter()
        .position(|route| route.source == ImageProgramOutputSourceV1::ProgramReceipt)
        .ok_or_else(|| Error::Incompatible("program receipt route is absent".to_owned()))?;
    let receipt_bytes = converge_receipt_bytes(plan, &mut state, receipt_route)?;
    if receipt_bytes.len() > outputs[receipt_route].bytes_mut().len() {
        return Err(Error::Output(
            "serialized program receipt exceeds its output allocation".to_owned(),
        ));
    }
    outputs[receipt_route].bytes_mut()[..receipt_bytes.len()].copy_from_slice(&receipt_bytes);
    initialized[receipt_route] = Some(receipt_bytes.len());
    let output_receipts = state.outputs.clone();
    let execution = state.outcome(plan, ImageProgramTerminalV1::Completed, output_receipts)?;
    for (output, written) in outputs.iter_mut().zip(initialized) {
        let written = written.ok_or_else(|| {
            Error::Incompatible("completed program output prefix is absent".to_owned())
        })?;
        if written > output.bytes_mut().len() {
            return Err(Error::Output(
                "program output exceeds its caller-owned allocation".to_owned(),
            ));
        }
        output
            .set_written(written)
            .map_err(|error| Error::Output(error.to_string()))?;
    }
    Ok(execution)
}

fn converge_receipt_bytes(
    plan: &ImageProgramPlanV1,
    state: &mut DriverState,
    receipt_route: usize,
) -> Result<Vec<u8>> {
    for _ in 0..8 {
        let completed_stages = u16::try_from(state.stages.len())
            .map_err(|_| Error::Incompatible("completed stage count exceeds u16".to_owned()))?;
        let receipt = ImageProgramReceiptV1 {
            plan: state.plan.clone(),
            backend: state.backend.clone(),
            runtime_epoch: state.runtime_epoch,
            completed_stages,
            stages: state.stages.clone(),
            outputs: state.outputs.clone(),
            terminal: ImageProgramTerminalV1::Completed,
            cleanup: state.cleanup.clone().ok_or_else(|| {
                Error::Incompatible("program receipt is missing cleanup evidence".to_owned())
            })?,
        };
        let bytes = serde_json::to_vec(&receipt).map_err(|error| {
            Error::Output(format!("program receipt serialization failed: {error}"))
        })?;
        let length = u64::try_from(bytes.len())
            .map_err(|_| Error::Output("program receipt length exceeds u64".to_owned()))?;
        if state.outputs[receipt_route].bytes_written == length {
            receipt
                .validate_for(plan)
                .map_err(logit_loom_diffusion::Error::from)?;
            return Ok(bytes);
        }
        state.outputs[receipt_route].bytes_written = length;
    }
    Err(Error::Output(
        "program receipt length did not reach a fixed point".to_owned(),
    ))
}

fn failure_identity(error: &Error) -> Digest {
    Digest::of_bytes(
        "sdcpp-resident-program-failure-v1",
        error.to_string().as_bytes(),
    )
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
    };

    use logit_loom_diffusion::{
        DiffusionSchedule, ImageBufferRole, ImageOperation, ImageOutputFormat,
        ImageProgramInputBindingV1, ImageProgramInputV1, ImageProgramNativeOutputRoleV1,
        ImageProgramNativeOutputV1, ImageProgramNativeStageV1, ImageProgramOutputRouteV1,
        ImageProgramStageOperationV1, ImageProgramValueReceiptV1, ImageProgramValueSpecV1,
        ImageProgramValueV1, SeedSelection, mask_blend_rgb8,
    };
    use logit_loom_executor::{BufferSpec, NeverCancel};

    use super::*;

    const RECEIPT_BYTES: usize = 16 * 1024;

    type FakeCheckpointExecution = CheckpointedResidentImageProgramExecution<Vec<u8>, Vec<u8>>;
    type FakeCheckpointRun = (
        Result<FakeCheckpointExecution>,
        Vec<u8>,
        Vec<u8>,
        usize,
        usize,
    );

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    enum FakeLifecycle {
        #[default]
        Idle,
        Active,
        Poisoned,
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    enum FakeFault {
        #[default]
        None,
        Finish,
        Materialization,
    }

    #[derive(Debug, Default)]
    struct FakeBackend {
        values: HashMap<u16, Vec<u8>>,
        live_bytes: u64,
        peak_bytes: u64,
        epoch: u64,
        lifecycle: FakeLifecycle,
        begin_count: u32,
        finish_count: u32,
        released: Vec<u16>,
        cancel_during_stage: Option<u16>,
        cancel_after_stage: Option<(u16, Arc<AtomicBool>)>,
        suspend_during_stage: Option<(u16, Arc<AtomicBool>)>,
        suspend_after_stage: Option<(u16, Arc<AtomicBool>)>,
        resumed_stages: Vec<u16>,
        reject_stage: Option<u16>,
        fault: FakeFault,
    }

    impl FakeBackend {
        fn measurement(value: u16) -> ImageProgramValueMeasurementV1 {
            ImageProgramValueMeasurementV1 {
                value,
                placement: ImageProgramValuePlacementV1::Host,
                host_to_device_transfers: 0,
                host_to_device_bytes: 0,
                device_to_host_transfers: 0,
                device_to_host_bytes: 0,
            }
        }

        fn insert_value(&mut self, value: u16, bytes: Vec<u8>) -> Result<()> {
            let byte_length = u64::try_from(bytes.len())
                .map_err(|_| Error::Native("fake value exceeds u64".to_owned()))?;
            if self.values.insert(value, bytes).is_some() {
                return Err(Error::Incompatible(format!(
                    "fake arena value {value} was produced twice"
                )));
            }
            self.live_bytes = self
                .live_bytes
                .checked_add(byte_length)
                .ok_or_else(|| Error::Native("fake arena bytes overflowed".to_owned()))?;
            self.peak_bytes = self.peak_bytes.max(self.live_bytes);
            Ok(())
        }

        fn output_bytes(
            &self,
            plan: &ImageProgramPlanV1,
            stage: &ImageProgramStageV1,
            value: u16,
        ) -> Result<Vec<u8>> {
            match &stage.operation {
                ImageProgramStageOperationV1::MaskBlend {
                    base,
                    overlay,
                    mask,
                    ..
                } => {
                    let base = self.value(*base)?;
                    let overlay = self.value(*overlay)?;
                    let mask = self.value(*mask)?;
                    let mut output = vec![0_u8; base.len()];
                    mask_blend_rgb8(base, overlay, mask, &mut output)
                        .map_err(logit_loom_diffusion::Error::from)?;
                    Ok(output)
                }
                ImageProgramStageOperationV1::RestoreCheckpoint { checkpoint, .. } => {
                    Ok(self.value(*checkpoint)?.to_vec())
                }
                ImageProgramStageOperationV1::CaptureCheckpoint { state, .. } => {
                    Ok(self.value(*state)?.to_vec())
                }
                ImageProgramStageOperationV1::Native { .. } => {
                    fake_value_bytes(&plan.values[usize::from(value)].spec, stage.stage, value)
                }
            }
        }

        fn value(&self, value: u16) -> Result<&[u8]> {
            self.values
                .get(&value)
                .map(Vec::as_slice)
                .ok_or_else(|| Error::Invalid(format!("fake arena value {value} is not live")))
        }
    }

    impl ResidentImageProgramBackend for FakeBackend {
        type StageCheckpoint = Vec<u8>;
        type ValueCheckpoint = Vec<u8>;

        fn backend_identity(&self) -> Result<Digest> {
            Ok(Digest::of_bytes("fake-resident-backend", b"v1"))
        }

        fn runtime_epoch(&self) -> u64 {
            self.epoch
        }

        fn validate_program(&self, _plan: &ImageProgramPlanV1) -> Result<()> {
            match self.lifecycle {
                FakeLifecycle::Idle => Ok(()),
                FakeLifecycle::Active => {
                    Err(Error::Invalid("fake arena is already active".to_owned()))
                }
                FakeLifecycle::Poisoned => {
                    Err(Error::Poisoned("fake backend is poisoned".to_owned()))
                }
            }
        }

        fn begin_program(
            &mut self,
            plan: &ImageProgramPlanV1,
            inputs: &[InputBuffer<'_>],
        ) -> Result<Vec<ImageProgramValueMeasurementV1>> {
            self.begin_count += 1;
            self.lifecycle = FakeLifecycle::Active;
            self.values.clear();
            self.live_bytes = 0;
            self.peak_bytes = 0;
            for (binding, input) in plan.inputs.iter().zip(inputs) {
                self.insert_value(binding.value, input.bytes().to_vec())?;
            }
            Ok(plan
                .inputs
                .iter()
                .map(|input| Self::measurement(input.value))
                .collect())
        }

        fn execute_stage(
            &mut self,
            plan: &ImageProgramPlanV1,
            stage: &ImageProgramStageV1,
            _cancellation: &dyn CancellationProbe,
            _suspension: &dyn CheckpointSuspensionProbe,
            checkpoint: Option<Self::StageCheckpoint>,
        ) -> Result<ResidentProgramStageTerminal<Self::StageCheckpoint>> {
            if self.lifecycle != FakeLifecycle::Active {
                return Err(Error::Native("fake arena is not active".to_owned()));
            }
            if self.cancel_during_stage == Some(stage.stage) {
                return Ok(ResidentProgramStageTerminal::CancelledAfterStep { step: 0 });
            }
            if let Some((suspend_stage, suspended)) = &self.suspend_during_stage
                && *suspend_stage == stage.stage
                && checkpoint.is_none()
            {
                suspended.store(true, Ordering::Release);
                return Ok(ResidentProgramStageTerminal::SuspendedAfterStep {
                    step: 0,
                    checkpoint: stage.stage.to_le_bytes().to_vec(),
                });
            }
            if let Some(checkpoint) = checkpoint {
                if checkpoint != stage.stage.to_le_bytes() {
                    return Err(Error::Incompatible(
                        "fake stage checkpoint identity differs".to_owned(),
                    ));
                }
                self.resumed_stages.push(stage.stage);
            }
            if self.reject_stage == Some(stage.stage) {
                return Err(Error::Invalid("synthetic rejected stage".to_owned()));
            }
            let mut receipts = Vec::new();
            let mut measurements = Vec::new();
            for value in stage.operation.produced_values() {
                let bytes = self.output_bytes(plan, stage, value)?;
                receipts.push(ImageProgramValueReceiptV1 {
                    value,
                    content: image_program_value_content(&bytes),
                    bytes: u64::try_from(bytes.len())
                        .map_err(|_| Error::Native("fake output exceeds u64".to_owned()))?,
                });
                measurements.push(Self::measurement(value));
                self.insert_value(value, bytes)?;
            }
            if let Some((cancel_stage, cancelled)) = &self.cancel_after_stage
                && *cancel_stage == stage.stage
            {
                cancelled.store(true, Ordering::Release);
            }
            if let Some((suspend_stage, suspended)) = &self.suspend_after_stage
                && *suspend_stage == stage.stage
            {
                suspended.store(true, Ordering::Release);
            }
            let observations = match &stage.operation {
                ImageProgramStageOperationV1::Native { plan } => plan
                    .observations
                    .iter()
                    .enumerate()
                    .map(|(index, _)| {
                        Digest::of_bytes(
                            "fake-resident-observation",
                            &[
                                u8::try_from(stage.stage).unwrap_or(u8::MAX),
                                u8::try_from(index).unwrap_or(u8::MAX),
                            ],
                        )
                    })
                    .collect(),
                _ => Vec::new(),
            };
            Ok(ResidentProgramStageTerminal::Completed(
                ResidentProgramCompletedStage {
                    receipt: ImageProgramStageReceiptV1 {
                        stage: stage.stage,
                        operation: stage
                            .operation
                            .digest()
                            .map_err(logit_loom_diffusion::Error::from)?,
                        outputs: receipts,
                        observations,
                    },
                    model_block_applications: Vec::new(),
                    wall_time_ns: u64::from(stage.stage) + 10,
                    native_time_ns: Some(u64::from(stage.stage) + 5),
                    values: measurements,
                },
            ))
        }

        fn checkpoint_value(
            &mut self,
            plan: &ImageProgramPlanV1,
            value: u16,
        ) -> Result<ResidentProgramValueCheckpoint<Self::ValueCheckpoint>> {
            let bytes = self.value(value)?.to_vec();
            let resident_bytes = u64::try_from(bytes.len())
                .map_err(|_| Error::Output("fake checkpoint exceeds u64".to_owned()))?;
            if resident_bytes
                > plan.values[usize::from(value)]
                    .spec
                    .maximum_bytes()
                    .map_err(logit_loom_diffusion::Error::from)?
            {
                return Err(Error::Output(
                    "fake checkpoint exceeds its value bound".to_owned(),
                ));
            }
            ResidentProgramValueCheckpoint::new(
                value,
                image_program_value_content(&bytes),
                resident_bytes,
                bytes,
            )
        }

        fn restore_value(
            &mut self,
            _plan: &ImageProgramPlanV1,
            checkpoint: ResidentProgramValueCheckpoint<Self::ValueCheckpoint>,
        ) -> Result<ImageProgramValueMeasurementV1> {
            let value = checkpoint.value();
            let bytes = checkpoint.into_checkpoint();
            self.insert_value(value, bytes)?;
            Ok(Self::measurement(value))
        }

        fn materialize_value(
            &mut self,
            _plan: &ImageProgramPlanV1,
            value: u16,
            output: &mut [u8],
        ) -> Result<usize> {
            let bytes = self.value(value)?;
            if bytes.len() > output.len() {
                return Err(Error::Output(
                    "fake materialization output is undersized".to_owned(),
                ));
            }
            output[..bytes.len()].copy_from_slice(bytes);
            if self.fault == FakeFault::Materialization {
                output[0] ^= 1;
            }
            Ok(bytes.len())
        }

        fn release_value(&mut self, value: u16) -> Result<()> {
            let bytes = self.values.remove(&value).ok_or_else(|| {
                Error::Invalid(format!("fake value {value} was already released"))
            })?;
            self.live_bytes = self
                .live_bytes
                .checked_sub(
                    u64::try_from(bytes.len())
                        .map_err(|_| Error::Native("fake value exceeds u64".to_owned()))?,
                )
                .ok_or_else(|| Error::Native("fake live bytes underflowed".to_owned()))?;
            self.released.push(value);
            Ok(())
        }

        fn finish_program(&mut self, cleanup: ImageCleanupPolicy) -> Result<ResidentProgramFinish> {
            self.finish_count += 1;
            if self.fault == FakeFault::Finish {
                return Err(Error::Poisoned("synthetic cleanup uncertainty".to_owned()));
            }
            if self.lifecycle != FakeLifecycle::Active {
                return Err(Error::Invalid("fake arena is not active".to_owned()));
            }
            self.lifecycle = FakeLifecycle::Idle;
            self.values.clear();
            self.live_bytes = 0;
            let cleanup = match cleanup {
                ImageCleanupPolicy::RetainSession => ImageProgramCleanupDispositionV1::Retained,
                ImageCleanupPolicy::ClearSession => {
                    let cleared_epoch = self.epoch;
                    self.epoch = self
                        .epoch
                        .checked_add(1)
                        .ok_or_else(|| Error::Poisoned("fake epoch overflowed".to_owned()))?;
                    ImageProgramCleanupDispositionV1::Confirmed { cleared_epoch }
                }
            };
            Ok(ResidentProgramFinish {
                cleanup,
                peak_arena_bytes: self.peak_bytes,
            })
        }

        fn poison(&mut self) {
            self.lifecycle = FakeLifecycle::Poisoned;
            self.values.clear();
        }
    }

    struct ToggleCancellation(Arc<AtomicBool>);

    impl CancellationProbe for ToggleCancellation {
        fn is_cancelled(&self) -> bool {
            self.0.load(Ordering::Acquire)
        }
    }

    struct AlwaysCancel;

    impl CancellationProbe for AlwaysCancel {
        fn is_cancelled(&self) -> bool {
            true
        }
    }

    struct ToggleSuspension(Arc<AtomicBool>);

    impl CheckpointSuspensionProbe for ToggleSuspension {
        fn is_suspension_requested(&self) -> bool {
            self.0.load(Ordering::Acquire)
        }
    }

    fn buffer(label: &str, bytes: u64, media_type: &str) -> BufferSpec {
        BufferSpec::new(
            Digest::of_bytes("fake-program-buffer", label.as_bytes()),
            bytes,
            media_type,
        )
        .unwrap()
    }

    fn blend_plan(two_stages: bool) -> ImageProgramPlanV1 {
        let mut values = vec![
            ImageProgramValueV1 {
                value: 0,
                spec: ImageProgramValueSpecV1::Rgb8 {
                    width: 1,
                    height: 1,
                },
            },
            ImageProgramValueV1 {
                value: 1,
                spec: ImageProgramValueSpecV1::Rgb8 {
                    width: 1,
                    height: 1,
                },
            },
            ImageProgramValueV1 {
                value: 2,
                spec: ImageProgramValueSpecV1::Gray8 {
                    width: 1,
                    height: 1,
                },
            },
            ImageProgramValueV1 {
                value: 3,
                spec: ImageProgramValueSpecV1::Rgb8 {
                    width: 1,
                    height: 1,
                },
            },
        ];
        let mut stages = vec![ImageProgramStageV1 {
            stage: 0,
            operation: ImageProgramStageOperationV1::MaskBlend {
                base: 0,
                overlay: 1,
                mask: 2,
                output: 3,
            },
        }];
        let output = if two_stages {
            values.push(ImageProgramValueV1 {
                value: 4,
                spec: ImageProgramValueSpecV1::Rgb8 {
                    width: 1,
                    height: 1,
                },
            });
            stages.push(ImageProgramStageV1 {
                stage: 1,
                operation: ImageProgramStageOperationV1::MaskBlend {
                    base: 3,
                    overlay: 1,
                    mask: 2,
                    output: 4,
                },
            });
            4
        } else {
            3
        };
        ImageProgramPlanV1 {
            values,
            inputs: vec![
                ImageProgramInputV1 {
                    value: 0,
                    buffer: buffer("base", 3, "image/rgb"),
                },
                ImageProgramInputV1 {
                    value: 1,
                    buffer: buffer("overlay", 3, "image/rgb"),
                },
                ImageProgramInputV1 {
                    value: 2,
                    buffer: buffer("mask", 1, "image/gray"),
                },
            ],
            stages,
            outputs: vec![
                ImageProgramOutputRouteV1 {
                    route: 0,
                    source: ImageProgramOutputSourceV1::Value { value: output },
                    buffer: buffer("image-output", 3, "image/rgb"),
                },
                ImageProgramOutputRouteV1 {
                    route: 1,
                    source: ImageProgramOutputSourceV1::ProgramReceipt,
                    buffer: buffer(
                        "receipt-output",
                        u64::try_from(RECEIPT_BYTES).unwrap(),
                        "application/json",
                    ),
                },
            ],
            cleanup: ImageCleanupPolicy::ClearSession,
        }
    }

    fn native_plan() -> ImageProgramPlanV1 {
        let prompt = buffer("prompt", 1, "text/plain; charset=utf-8");
        ImageProgramPlanV1 {
            values: vec![
                ImageProgramValueV1 {
                    value: 0,
                    spec: ImageProgramValueSpecV1::Utf8 { maximum_bytes: 1 },
                },
                ImageProgramValueV1 {
                    value: 1,
                    spec: ImageProgramValueSpecV1::Rgb8 {
                        width: 1,
                        height: 1,
                    },
                },
            ],
            inputs: vec![ImageProgramInputV1 {
                value: 0,
                buffer: prompt,
            }],
            stages: vec![ImageProgramStageV1 {
                stage: 0,
                operation: ImageProgramStageOperationV1::Native {
                    plan: Box::new(ImageProgramNativeStageV1 {
                        profile: Digest::of_bytes("fake-profile", b"one"),
                        load: Digest::of_bytes("fake-load", b"one"),
                        operation: ImageOperation::TextToImage,
                        width: 1,
                        height: 1,
                        output_format: ImageOutputFormat::Rgb8,
                        seed: SeedSelection::Fixed { seed: 7 },
                        rng: Digest::of_bytes("fake-rng", b"one"),
                        placement: Digest::of_bytes("fake-placement", b"host"),
                        schedule: Some(
                            DiffusionSchedule::new(
                                Digest::of_bytes("fake-schedule", b"one"),
                                vec![1.0, 0.0],
                            )
                            .unwrap(),
                        ),
                        guidance_scale_bits: 1.0_f32.to_bits(),
                        strength_bits: 1.0_f32.to_bits(),
                        inputs: vec![ImageProgramInputBindingV1 {
                            role: ImageBufferRole::PositiveConditioning,
                            value: 0,
                        }],
                        loras: Vec::new(),
                        operators: Vec::new(),
                        observations: Vec::new(),
                        checkpoint_restore_at_step: None,
                        checkpoint_after_step: None,
                        outputs: vec![ImageProgramNativeOutputV1 {
                            role: ImageProgramNativeOutputRoleV1::Primary,
                            value: 1,
                        }],
                    }),
                },
            }],
            outputs: vec![
                ImageProgramOutputRouteV1 {
                    route: 0,
                    source: ImageProgramOutputSourceV1::Value { value: 1 },
                    buffer: buffer("native-image-output", 3, "image/rgb"),
                },
                ImageProgramOutputRouteV1 {
                    route: 1,
                    source: ImageProgramOutputSourceV1::ProgramReceipt,
                    buffer: buffer(
                        "native-receipt-output",
                        u64::try_from(RECEIPT_BYTES).unwrap(),
                        "application/json",
                    ),
                },
            ],
            cleanup: ImageCleanupPolicy::ClearSession,
        }
    }

    fn fake_value_bytes(spec: &ImageProgramValueSpecV1, stage: u16, value: u16) -> Result<Vec<u8>> {
        let exact = usize::try_from(
            spec.maximum_bytes()
                .map_err(logit_loom_diffusion::Error::from)?,
        )
        .map_err(|_| Error::Native("fake value size exceeds usize".to_owned()))?;
        let length = match spec {
            ImageProgramValueSpecV1::Utf8 { .. }
            | ImageProgramValueSpecV1::Checkpoint { .. }
            | ImageProgramValueSpecV1::Opaque { .. }
            | ImageProgramValueSpecV1::Png { .. } => exact.min(4),
            _ => exact,
        };
        let mut bytes = vec![0_u8; length];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = u8::try_from(
                usize::from(stage)
                    .wrapping_add(usize::from(value))
                    .wrapping_add(index),
            )
            .unwrap_or(u8::MAX);
        }
        Ok(bytes)
    }

    fn input_bytes(plan: &ImageProgramPlanV1) -> Vec<Vec<u8>> {
        plan.inputs
            .iter()
            .map(|input| match input.value {
                0 => vec![10, 20, 30],
                1 => vec![110, 120, 130],
                2 => vec![128],
                _ => vec![b'x'],
            })
            .collect()
    }

    fn execute_with(
        driver: &mut ResidentImageProgramDriver<FakeBackend>,
        plan: &ImageProgramPlanV1,
        input_bytes: &[Vec<u8>],
        cancellation: &dyn CancellationProbe,
    ) -> (
        Result<ResidentImageProgramExecution>,
        Vec<u8>,
        Vec<u8>,
        usize,
        usize,
    ) {
        let inputs = plan
            .inputs
            .iter()
            .zip(input_bytes)
            .map(|(input, bytes)| InputBuffer::new(&input.buffer, bytes).unwrap())
            .collect::<Vec<_>>();
        let mut image = vec![0_u8; 3];
        let mut receipt = vec![0_u8; RECEIPT_BYTES];
        let mut outputs = vec![
            OutputBuffer::new(&plan.outputs[0].buffer, &mut image).unwrap(),
            OutputBuffer::new(&plan.outputs[1].buffer, &mut receipt).unwrap(),
        ];
        let result = driver.execute(plan, &inputs, &mut outputs, cancellation);
        let image_written = outputs[0].written();
        let receipt_written = outputs[1].written();
        drop(outputs);
        (result, image, receipt, image_written, receipt_written)
    }

    fn execute_checkpointed_with(
        driver: &mut ResidentImageProgramDriver<FakeBackend>,
        plan: &ImageProgramPlanV1,
        input_bytes: &[Vec<u8>],
        suspension: &dyn CheckpointSuspensionProbe,
        continuation: Option<ResidentImageProgramContinuation<Vec<u8>, Vec<u8>>>,
    ) -> FakeCheckpointRun {
        let inputs = plan
            .inputs
            .iter()
            .zip(input_bytes)
            .map(|(input, bytes)| InputBuffer::new(&input.buffer, bytes).unwrap())
            .collect::<Vec<_>>();
        let mut image = vec![0_u8; 3];
        let mut receipt = vec![0_u8; RECEIPT_BYTES];
        let mut outputs = vec![
            OutputBuffer::new(&plan.outputs[0].buffer, &mut image).unwrap(),
            OutputBuffer::new(&plan.outputs[1].buffer, &mut receipt).unwrap(),
        ];
        let result = driver.execute_checkpointed(
            plan,
            &inputs,
            &mut outputs,
            &NeverCancel,
            suspension,
            continuation,
        );
        let image_written = outputs[0].written();
        let receipt_written = outputs[1].written();
        drop(outputs);
        (result, image, receipt, image_written, receipt_written)
    }

    #[test]
    fn completed_program_publishes_outputs_atomically_and_releases_liveness() {
        let plan = blend_plan(false);
        let bytes = input_bytes(&plan);
        let mut driver = ResidentImageProgramDriver::new(FakeBackend::default());
        let (result, image, receipt_bytes, image_written, receipt_written) =
            execute_with(&mut driver, &plan, &bytes, &NeverCancel);
        let execution = result.unwrap();
        assert_eq!(image_written, 3);
        let mut expected = [0_u8; 3];
        mask_blend_rgb8(&bytes[0], &bytes[1], &bytes[2], &mut expected).unwrap();
        assert_eq!(image, expected);
        let serialized: ImageProgramReceiptV1 =
            serde_json::from_slice(&receipt_bytes[..receipt_written]).unwrap();
        assert_eq!(serialized, execution.receipt);
        assert_eq!(driver.backend().released, [0, 1, 2, 3]);
        assert_eq!(driver.backend().begin_count, 1);
        assert_eq!(driver.backend().finish_count, 1);
        assert_eq!(driver.backend().epoch, 1);
    }

    #[test]
    fn cancellation_before_start_creates_no_arena_or_output() {
        let plan = blend_plan(false);
        let bytes = input_bytes(&plan);
        let mut driver = ResidentImageProgramDriver::new(FakeBackend::default());
        let (result, _, _, image_written, receipt_written) =
            execute_with(&mut driver, &plan, &bytes, &AlwaysCancel);
        let execution = result.unwrap();
        assert_eq!(
            execution.receipt.terminal,
            ImageProgramTerminalV1::CancelledBeforeStart
        );
        assert_eq!(image_written, 0);
        assert_eq!(receipt_written, 0);
        assert_eq!(driver.backend().begin_count, 0);
        assert_eq!(driver.backend().finish_count, 0);
    }

    #[test]
    fn cancellation_between_stages_returns_the_exact_completed_prefix() {
        let plan = blend_plan(true);
        let bytes = input_bytes(&plan);
        let cancelled = Arc::new(AtomicBool::new(false));
        let backend = FakeBackend {
            cancel_after_stage: Some((0, Arc::clone(&cancelled))),
            ..FakeBackend::default()
        };
        let mut driver = ResidentImageProgramDriver::new(backend);
        let (result, _, _, image_written, receipt_written) =
            execute_with(&mut driver, &plan, &bytes, &ToggleCancellation(cancelled));
        let execution = result.unwrap();
        assert_eq!(
            execution.receipt.terminal,
            ImageProgramTerminalV1::CancelledAfterStage { stage: 0 }
        );
        assert_eq!(execution.receipt.completed_stages, 1);
        assert_eq!(image_written, 0);
        assert_eq!(receipt_written, 0);
        assert_eq!(driver.backend().finish_count, 1);
    }

    #[test]
    fn native_cancellation_names_the_exact_post_euler_boundary() {
        let plan = native_plan();
        let bytes = vec![vec![b'x']];
        let backend = FakeBackend {
            cancel_during_stage: Some(0),
            ..FakeBackend::default()
        };
        let mut driver = ResidentImageProgramDriver::new(backend);
        let (result, _, _, _, _) = execute_with(&mut driver, &plan, &bytes, &NeverCancel);
        let execution = result.unwrap();
        assert_eq!(
            execution.receipt.terminal,
            ImageProgramTerminalV1::CancelledAfterStep { stage: 0, step: 0 }
        );
        assert_eq!(execution.receipt.completed_stages, 0);
    }

    #[test]
    fn between_stage_suspension_reconstructs_live_values_and_publishes_once() {
        let plan = blend_plan(true);
        let bytes = input_bytes(&plan);
        let suspended = Arc::new(AtomicBool::new(false));
        let backend = FakeBackend {
            suspend_after_stage: Some((0, Arc::clone(&suspended))),
            ..FakeBackend::default()
        };
        let mut driver = ResidentImageProgramDriver::new(backend);
        let (first, _, _, image_written, receipt_written) = execute_checkpointed_with(
            &mut driver,
            &plan,
            &bytes,
            &ToggleSuspension(Arc::clone(&suspended)),
            None,
        );
        let CheckpointedResidentImageProgramExecution::Suspended(continuation) = first.unwrap()
        else {
            panic!("expected a suspended continuation");
        };
        assert_eq!(continuation.next_stage(), 1);
        assert_eq!(image_written, 0);
        assert_eq!(receipt_written, 0);
        assert_eq!(driver.backend().finish_count, 1);
        assert_eq!(driver.backend().lifecycle, FakeLifecycle::Idle);

        suspended.store(false, Ordering::Release);
        let (second, image, _, image_written, receipt_written) = execute_checkpointed_with(
            &mut driver,
            &plan,
            &bytes,
            &ToggleSuspension(suspended),
            Some(*continuation),
        );
        let CheckpointedResidentImageProgramExecution::Terminal(execution) = second.unwrap() else {
            panic!("expected a completed execution");
        };
        assert_eq!(
            execution.receipt.terminal,
            ImageProgramTerminalV1::Completed
        );
        assert_eq!(execution.receipt.completed_stages, 2);
        assert_eq!(image_written, 3);
        assert!(receipt_written > 0);
        let mut expected = [0_u8; 3];
        let mut stage_zero = [0_u8; 3];
        mask_blend_rgb8(&bytes[0], &bytes[1], &bytes[2], &mut stage_zero).unwrap();
        mask_blend_rgb8(&stage_zero, &bytes[1], &bytes[2], &mut expected).unwrap();
        assert_eq!(image, expected);
        assert_eq!(driver.backend().begin_count, 2);
        assert_eq!(driver.backend().finish_count, 2);
        assert_eq!(driver.backend().epoch, 1);
    }

    #[test]
    fn in_stage_suspension_restores_the_backend_checkpoint() {
        let plan = native_plan();
        let bytes = vec![vec![b'x']];
        let suspended = Arc::new(AtomicBool::new(false));
        let backend = FakeBackend {
            suspend_during_stage: Some((0, Arc::clone(&suspended))),
            ..FakeBackend::default()
        };
        let mut driver = ResidentImageProgramDriver::new(backend);
        let (first, _, _, image_written, receipt_written) = execute_checkpointed_with(
            &mut driver,
            &plan,
            &bytes,
            &ToggleSuspension(Arc::clone(&suspended)),
            None,
        );
        let CheckpointedResidentImageProgramExecution::Suspended(continuation) = first.unwrap()
        else {
            panic!("expected a suspended continuation");
        };
        assert_eq!(continuation.next_stage(), 0);
        assert_eq!(image_written, 0);
        assert_eq!(receipt_written, 0);

        suspended.store(false, Ordering::Release);
        let (second, _, _, image_written, receipt_written) = execute_checkpointed_with(
            &mut driver,
            &plan,
            &bytes,
            &ToggleSuspension(suspended),
            Some(*continuation),
        );
        let CheckpointedResidentImageProgramExecution::Terminal(execution) = second.unwrap() else {
            panic!("expected a completed execution");
        };
        assert_eq!(
            execution.receipt.terminal,
            ImageProgramTerminalV1::Completed
        );
        assert_eq!(image_written, 3);
        assert!(receipt_written > 0);
        assert_eq!(driver.backend().resumed_stages, [0]);
        assert_eq!(driver.backend().begin_count, 2);
        assert_eq!(driver.backend().finish_count, 2);
    }

    #[test]
    fn rejected_stage_is_a_cleaned_terminal_not_a_poisoning_claim() {
        let plan = blend_plan(false);
        let bytes = input_bytes(&plan);
        let backend = FakeBackend {
            reject_stage: Some(0),
            ..FakeBackend::default()
        };
        let mut driver = ResidentImageProgramDriver::new(backend);
        let (result, _, _, image_written, receipt_written) =
            execute_with(&mut driver, &plan, &bytes, &NeverCancel);
        let execution = result.unwrap();
        assert!(matches!(
            execution.receipt.terminal,
            ImageProgramTerminalV1::FailedAtStage { stage: 0, .. }
        ));
        assert_eq!(image_written, 0);
        assert_eq!(receipt_written, 0);
        assert_ne!(driver.backend().lifecycle, FakeLifecycle::Poisoned);
        assert_eq!(driver.backend().finish_count, 1);
    }

    #[test]
    fn corrupt_materialization_writes_nothing_and_still_cleans_up() {
        let plan = blend_plan(false);
        let bytes = input_bytes(&plan);
        let backend = FakeBackend {
            fault: FakeFault::Materialization,
            ..FakeBackend::default()
        };
        let mut driver = ResidentImageProgramDriver::new(backend);
        let (result, _, _, image_written, receipt_written) =
            execute_with(&mut driver, &plan, &bytes, &NeverCancel);
        assert!(matches!(result, Err(Error::Incompatible(_))));
        assert_eq!(image_written, 0);
        assert_eq!(receipt_written, 0);
        assert_eq!(driver.backend().finish_count, 1);
        assert_ne!(driver.backend().lifecycle, FakeLifecycle::Poisoned);
    }

    #[test]
    fn cleanup_uncertainty_poisons_the_backend_and_suppresses_outputs() {
        let plan = blend_plan(false);
        let bytes = input_bytes(&plan);
        let backend = FakeBackend {
            fault: FakeFault::Finish,
            ..FakeBackend::default()
        };
        let mut driver = ResidentImageProgramDriver::new(backend);
        let (result, _, _, image_written, receipt_written) =
            execute_with(&mut driver, &plan, &bytes, &NeverCancel);
        assert!(matches!(result, Err(Error::Poisoned(_))));
        assert_eq!(image_written, 0);
        assert_eq!(receipt_written, 0);
        assert_eq!(driver.backend().lifecycle, FakeLifecycle::Poisoned);
    }

    #[test]
    fn fake_arena_rejects_double_release_and_post_cleanup_access() {
        let plan = blend_plan(false);
        let bytes = input_bytes(&plan);
        let inputs = plan
            .inputs
            .iter()
            .zip(&bytes)
            .map(|(input, bytes)| InputBuffer::new(&input.buffer, bytes).unwrap())
            .collect::<Vec<_>>();
        let mut backend = FakeBackend::default();
        backend.begin_program(&plan, &inputs).unwrap();
        backend.release_value(0).unwrap();
        assert!(backend.release_value(0).is_err());
        backend
            .finish_program(ImageCleanupPolicy::ClearSession)
            .unwrap();
        assert!(backend.materialize_value(&plan, 1, &mut [0_u8; 3]).is_err());
    }
}

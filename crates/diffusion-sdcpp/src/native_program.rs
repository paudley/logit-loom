// SPDX-License-Identifier: MIT OR Apache-2.0

//! Concrete resident image-program backend for the pinned companion ABI v3.

use std::{
    collections::{HashMap, HashSet},
    ffi::{CString, c_void},
    path::PathBuf,
    ptr::NonNull,
    str::FromStr as _,
    time::Instant,
};

use logit_loom_diffusion::{
    ControlFlow, DiffusionPlan, Digest, ImageBufferRole, ImageCleanupPolicy,
    ImageOpaqueValueKindV1, ImageOperation, ImageOutputFormat, ImagePngColorV1,
    ImageProgramCleanupDispositionV1, ImageProgramLoraV1, ImageProgramNativeOutputRoleV1,
    ImageProgramNativeStageV1, ImageProgramPlanV1, ImageProgramStageOperationV1,
    ImageProgramStageReceiptV1, ImageProgramStageV1, ImageProgramValueMeasurementV1,
    ImageProgramValuePlacementV1, ImageProgramValueReceiptV1, ImageProgramValueSpecV1,
    Intervention, KreaActivationPlanV1, KreaActivationTerminalV1, KreaActivationTopologyV1,
    MAX_IMAGE_DIMENSION, MAX_IMAGE_PROGRAM_VALUE_BYTES, ObservationKind, Pipeline, SeedSelection,
    StepContext, StepSelector, TensorDType, TensorLayout, TensorSelector,
};
use logit_loom_executor::{CancellationProbe, ExecutorState, InputBuffer};

use crate::{
    ADAPTER_CONTRACT_VERSION, DiffusionCheckpoint, Error, ImageRequest, MODEL_BLOCK_ABI_VERSION,
    ModelBlockApplicationV1, PROGRAM_ABI_VERSION, ResidentImageProgramBackend,
    ResidentProgramCompletedStage, ResidentProgramFinish, ResidentProgramStageTerminal, Result,
    Sdcpp, StepProgram, UPSTREAM_COMMIT,
    contract::component_map,
    execution::{
        InstalledChannelBias, InstalledModelBlockResidualScale, ObservationAccumulator,
        lora_target_v1,
    },
    ffi::{
        self, ImageViewV2, KreaApplicationResultV6, KreaCaptureResultV6, LoraScalePointV3,
        LoraScheduleV3, ModelBlockOperatorV5, NativeModelBlockApplicationV5, ProgramImageParamsV3,
        ProgramImageParamsV5, ProgramImageParamsV6, ProgramImageResultV3, ProgramImageResultV5,
        ProgramImageResultV6, ProgramOutputV3, TensorViewV2, ValueDescriptorV3, ValueHandleV3,
    },
    krea_activation::{
        InstalledKreaActivation, KreaCallbackState, LoweredKreaActivation, krea_event_callback,
    },
    runtime::{
        CallbackState, condition_callback, native_status_error, path_c_string, step_callback,
    },
};

/// Resolves one verified program-local `LoRA` artifact to a synchronous,
/// caller-retained descriptor path.
pub trait ResidentArtifactPathResolver {
    /// Returns a path such as `/proc/self/fd/N` for the exact input.
    ///
    /// The caller remains responsible for retaining the descriptor through
    /// program cleanup and for ensuring it names the bytes bound by `input`.
    ///
    /// # Errors
    ///
    /// Returns an error if no exact retained descriptor path can be supplied.
    fn resolve_lora_path(
        &mut self,
        value: u16,
        input: &InputBuffer<'_>,
    ) -> std::result::Result<PathBuf, String>;
}

/// Resolver that rejects path-backed program values before native allocation.
#[derive(Clone, Copy, Debug, Default)]
pub struct RejectResidentArtifactPaths;

impl ResidentArtifactPathResolver for RejectResidentArtifactPaths {
    fn resolve_lora_path(
        &mut self,
        _value: u16,
        _input: &InputBuffer<'_>,
    ) -> std::result::Result<PathBuf, String> {
        Err("no resident LoRA descriptor-path resolver was installed".to_owned())
    }
}

/// Exact scheduled request-local `LoRA` target installed by companion ABI v3.
pub fn resident_lora_target_v1(high_noise: bool) -> Digest {
    Digest::of_bytes(
        "sdcpp-resident-lora-target-v1",
        if high_noise {
            b"whole-model-high-noise-scheduled"
        } else {
            b"whole-model-scheduled"
        },
    )
}

/// Exact deterministic resident PNG encoder implementation.
///
/// # Errors
///
/// Returns a deterministic identity serialization error.
pub fn resident_png_encoding_v1(color: ImagePngColorV1) -> Result<Digest> {
    Digest::of_serializable(
        "sdcpp-resident-png-encoding-v1",
        &(
            PROGRAM_ABI_VERSION,
            ADAPTER_CONTRACT_VERSION,
            UPSTREAM_COMMIT,
            "stb-image-write-default-compression-v1",
            color,
        ),
    )
    .map_err(logit_loom_diffusion::Error::from)
    .map_err(Into::into)
}

/// Conservative encoded length required by the resident PNG implementation.
///
/// # Errors
///
/// Returns an error for invalid geometry or arithmetic beyond the public
/// resident-value bound.
pub fn resident_png_maximum_bytes_v1(
    width: u32,
    height: u32,
    color: ImagePngColorV1,
) -> Result<u64> {
    if width == 0 || height == 0 || width > MAX_IMAGE_DIMENSION || height > MAX_IMAGE_DIMENSION {
        return Err(Error::Invalid(
            "resident PNG dimensions exceed the public image bound".to_owned(),
        ));
    }
    let row_bytes = u64::from(width)
        .checked_mul(u64::from(color.channels()))
        .ok_or_else(|| Error::Invalid("resident PNG row length overflowed".to_owned()))?;
    let filtered_bytes = row_bytes
        .checked_add(1)
        .and_then(|row| row.checked_mul(u64::from(height)))
        .ok_or_else(|| Error::Invalid("resident PNG filtered length overflowed".to_owned()))?;
    let blocks = filtered_bytes
        .checked_add(32_766)
        .map(|bytes| bytes / 32_767)
        .ok_or_else(|| Error::Invalid("resident PNG block count overflowed".to_owned()))?;
    let maximum = blocks
        .checked_mul(5)
        .and_then(|overhead| overhead.checked_add(63))
        .and_then(|overhead| filtered_bytes.checked_add(overhead))
        .ok_or_else(|| Error::Invalid("resident PNG encoded bound overflowed".to_owned()))?;
    if maximum > MAX_IMAGE_PROGRAM_VALUE_BYTES {
        return Err(Error::Invalid(
            "resident PNG encoded bound exceeds the public value limit".to_owned(),
        ));
    }
    Ok(maximum)
}

/// Exact checkpoint-state compatibility for one resident runtime binding.
///
/// # Errors
///
/// Returns a deterministic identity serialization error.
pub fn resident_checkpoint_compatibility_v1(
    bindings: &crate::ImageExecutionBindings,
) -> Result<Digest> {
    Digest::of_serializable(
        "sdcpp-resident-checkpoint-compatibility-v1",
        &(
            PROGRAM_ABI_VERSION,
            ADAPTER_CONTRACT_VERSION,
            UPSTREAM_COMMIT,
            &bindings.backend,
            &bindings.profile,
            &bindings.load,
            &bindings.rng,
            &bindings.placement,
        ),
    )
    .map_err(logit_loom_diffusion::Error::from)
    .map_err(Into::into)
}

/// Exact restore/capture conversion implementation for resident checkpoints.
///
/// # Errors
///
/// Returns a deterministic identity serialization error.
pub fn resident_checkpoint_conversion_v1(
    compatibility: &Digest,
    restoring: bool,
) -> Result<Digest> {
    Digest::of_serializable(
        "sdcpp-resident-checkpoint-conversion-v1",
        &(
            PROGRAM_ABI_VERSION,
            ADAPTER_CONTRACT_VERSION,
            if restoring { "restore" } else { "capture" },
            compatibility,
        ),
    )
    .map_err(logit_loom_diffusion::Error::from)
    .map_err(Into::into)
}

/// Borrowed resident-program backend over one exclusively owned runtime.
pub struct SdcppResidentProgram<'a, R = RejectResidentArtifactPaths> {
    runtime: &'a mut Sdcpp,
    artifact_paths: R,
    arena: Option<NonNull<c_void>>,
    handles: HashMap<u16, ValueHandleV3>,
    input_checkpoints: HashMap<u16, DiffusionCheckpoint>,
    checkpoint_states: HashMap<u16, DiffusionCheckpoint>,
    captured_states: HashMap<u16, CapturedState>,
    text_values: HashMap<u16, String>,
    krea_activation_executions: Vec<crate::KreaActivationExecutionV1>,
}

impl<R> std::fmt::Debug for SdcppResidentProgram<'_, R> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SdcppResidentProgram")
            .field("runtime", &self.runtime)
            .field("arena_active", &self.arena.is_some())
            .field("live_handles", &self.handles.len())
            .field("krea_activation", &self.runtime.krea_activation.is_some())
            .finish_non_exhaustive()
    }
}

impl Sdcpp {
    /// Borrows this runtime for one or more resident-program executions while
    /// rejecting path-backed `LoRA` values.
    pub fn resident_program(&mut self) -> SdcppResidentProgram<'_, RejectResidentArtifactPaths> {
        SdcppResidentProgram::new(self, RejectResidentArtifactPaths)
    }

    /// Borrows this runtime with an exact caller-owned `LoRA` path resolver.
    pub fn resident_program_with_paths<R>(
        &mut self,
        artifact_paths: R,
    ) -> SdcppResidentProgram<'_, R> {
        SdcppResidentProgram::new(self, artifact_paths)
    }
}

impl<'a, R> SdcppResidentProgram<'a, R> {
    fn new(runtime: &'a mut Sdcpp, artifact_paths: R) -> Self {
        Self {
            runtime,
            artifact_paths,
            arena: None,
            handles: HashMap::new(),
            input_checkpoints: HashMap::new(),
            checkpoint_states: HashMap::new(),
            captured_states: HashMap::new(),
            text_values: HashMap::new(),
            krea_activation_executions: Vec::new(),
        }
    }

    /// Returns the borrowed runtime.
    pub const fn runtime(&self) -> &Sdcpp {
        self.runtime
    }

    /// Returns the mutable borrowed runtime when no arena operation is active.
    ///
    /// # Errors
    ///
    /// Returns an error while one request-scoped arena is live.
    pub fn runtime_mut(&mut self) -> Result<&mut Sdcpp> {
        if self.arena.is_some() {
            return Err(Error::Invalid(
                "cannot access the resident runtime while a program arena is active".to_owned(),
            ));
        }
        Ok(self.runtime)
    }

    /// Returns the installed path resolver.
    pub const fn artifact_paths(&self) -> &R {
        &self.artifact_paths
    }

    /// Returns the mutable installed path resolver.
    pub fn artifact_paths_mut(&mut self) -> &mut R {
        &mut self.artifact_paths
    }

    /// Imports and installs one complete Krea activation plan for subsequent
    /// resident image jobs on this same native session.
    ///
    /// Sealed values are copied to device storage exactly once here. Capture
    /// sources remain device-local within each native call. The installation
    /// remains active across programs that retain the model session until the
    /// caller explicitly clears it or a program requests session clearing.
    ///
    /// # Errors
    ///
    /// Returns an error for an active arena, a second installation, invalid
    /// topology/plan/input bytes, or uncertain native placement.
    pub fn install_krea_activation(
        &mut self,
        topology: KreaActivationTopologyV1,
        plan: KreaActivationPlanV1,
        inputs: &[crate::KreaActivationInputBuffer<'_>],
    ) -> Result<()> {
        if self.arena.is_some() {
            return Err(Error::Invalid(
                "cannot install Krea activation while a program arena is active".to_owned(),
            ));
        }
        if self.runtime.krea_activation.is_some() {
            return Err(Error::Invalid(
                "a Krea activation plan is already installed".to_owned(),
            ));
        }
        let activation = InstalledKreaActivation::install(self.runtime, topology, plan, inputs)?;
        self.runtime.krea_activation = Some(activation);
        Ok(())
    }

    /// Releases every installed resident Krea input and advances the native
    /// activation-handle epoch.
    ///
    /// # Errors
    ///
    /// Returns an error while an arena is active or if native release cannot
    /// be confirmed. Release uncertainty poisons the runtime.
    pub fn clear_krea_activation(&mut self) -> Result<()> {
        if self.arena.is_some() {
            return Err(Error::Invalid(
                "cannot clear Krea activation while a program arena is active".to_owned(),
            ));
        }
        if let Some(activation) = self.runtime.krea_activation.take() {
            activation.release(self.runtime)?;
        }
        Ok(())
    }

    /// Removes all completed Krea activation executions accumulated since the
    /// previous call without changing the installed resident plan.
    pub fn take_krea_activation_executions(&mut self) -> Vec<crate::KreaActivationExecutionV1> {
        std::mem::take(&mut self.krea_activation_executions)
    }

    fn arena(&self) -> Result<NonNull<c_void>> {
        self.arena
            .ok_or_else(|| Error::Native("resident program arena is not active".to_owned()))
    }

    fn handle(&self, value: u16) -> Result<ValueHandleV3> {
        self.handles
            .get(&value)
            .copied()
            .ok_or_else(|| Error::Invalid(format!("resident value {value} is not live")))
    }

    fn insert_handle(&mut self, value: u16, handle: ValueHandleV3) -> Result<()> {
        if handle.is_empty() || self.handles.insert(value, handle).is_some() {
            return Err(Error::Incompatible(format!(
                "resident value {value} was absent or published twice"
            )));
        }
        Ok(())
    }
}

impl<R> SdcppResidentProgram<'_, R> {
    fn validate_runtime_state(&self) -> Result<()> {
        if !cfg!(target_endian = "little") {
            return Err(Error::Incompatible(
                "resident ABI v3 requires little-endian f32 checkpoint bytes".to_owned(),
            ));
        }
        if self.arena.is_some() {
            return Err(Error::Invalid(
                "a resident program arena is already active".to_owned(),
            ));
        }
        match self.runtime.state {
            ExecutorState::Resident => Ok(()),
            ExecutorState::Poisoned => Err(Error::Poisoned(
                "resident runtime is already poisoned".to_owned(),
            )),
            state => Err(Error::Invalid(format!(
                "cannot begin a resident program while runtime is {state:?}"
            ))),
        }
    }

    fn validate_value_specs(
        plan: &ImageProgramPlanV1,
        checkpoint_compatibility: &Digest,
    ) -> Result<()> {
        for value in &plan.values {
            match &value.spec {
                ImageProgramValueSpecV1::Tensor { tensor, .. }
                    if tensor.dtype != TensorDType::F32
                        || tensor.layout != TensorLayout::DimensionZeroFastest =>
                {
                    return Err(Error::Invalid(format!(
                        "resident tensor value {} requires finite dimension-zero-fastest f32",
                        value.value
                    )));
                }
                ImageProgramValueSpecV1::Opaque {
                    opaque_kind: ImageOpaqueValueKindV1::Conditioning,
                    ..
                } => {
                    return Err(Error::Invalid(
                        "native conditioning values have no installed ABI v3 selector".to_owned(),
                    ));
                }
                ImageProgramValueSpecV1::Checkpoint { compatibility, .. }
                | ImageProgramValueSpecV1::Opaque {
                    opaque_kind: ImageOpaqueValueKindV1::CheckpointState,
                    compatibility,
                    ..
                } if compatibility != checkpoint_compatibility => {
                    return Err(Error::Incompatible(
                        "resident checkpoint compatibility differs from the loaded runtime"
                            .to_owned(),
                    ));
                }
                ImageProgramValueSpecV1::Png {
                    width,
                    height,
                    color,
                    encoding,
                    maximum_bytes,
                } => {
                    if encoding != &resident_png_encoding_v1(*color)? {
                        return Err(Error::Incompatible(
                            "resident PNG encoder identity differs".to_owned(),
                        ));
                    }
                    if *maximum_bytes < resident_png_maximum_bytes_v1(*width, *height, *color)? {
                        return Err(Error::Invalid(
                            "resident PNG allocation is below the encoder upper bound".to_owned(),
                        ));
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn validate_external_routes(plan: &ImageProgramPlanV1) -> Result<()> {
        let input_values = plan
            .inputs
            .iter()
            .map(|input| input.value)
            .collect::<HashSet<_>>();
        if plan.values.iter().any(|value| {
            input_values.contains(&value.value)
                && matches!(
                    value.spec,
                    ImageProgramValueSpecV1::Opaque {
                        opaque_kind: ImageOpaqueValueKindV1::CheckpointState,
                        ..
                    }
                )
        }) {
            return Err(Error::Invalid(
                "native checkpoint-state handles cannot be external program inputs".to_owned(),
            ));
        }
        if plan.outputs.iter().any(|output| {
            let logit_loom_diffusion::ImageProgramOutputSourceV1::Value { value } = output.source
            else {
                return false;
            };
            matches!(
                plan.values[usize::from(value)].spec,
                ImageProgramValueSpecV1::Opaque {
                    opaque_kind: ImageOpaqueValueKindV1::LoraArtifact,
                    ..
                }
            )
        }) {
            return Err(Error::Invalid(
                "resident LoRA descriptor paths cannot be materialized as outputs".to_owned(),
            ));
        }
        for output in &plan.outputs {
            let logit_loom_diffusion::ImageProgramOutputSourceV1::Value { value } = output.source
            else {
                continue;
            };
            let ImageProgramValueSpecV1::Png {
                width,
                height,
                color,
                ..
            } = &plan.values[usize::from(value)].spec
            else {
                continue;
            };
            let required = resident_png_maximum_bytes_v1(*width, *height, *color)?;
            if output.buffer.byte_length < required {
                return Err(Error::Invalid(
                    "resident PNG output allocation is below the encoder upper bound".to_owned(),
                ));
            }
        }
        Ok(())
    }

    fn validate_stage_bindings(
        &self,
        plan: &ImageProgramPlanV1,
        bindings: &crate::ImageExecutionBindings,
        restore_implementation: &Digest,
        capture_implementation: &Digest,
    ) -> Result<()> {
        for stage in &plan.stages {
            match &stage.operation {
                ImageProgramStageOperationV1::Native { plan: native } => {
                    validate_native_stage(self.runtime, plan, native, bindings)?;
                }
                ImageProgramStageOperationV1::RestoreCheckpoint { implementation, .. }
                    if implementation != restore_implementation =>
                {
                    return Err(Error::Incompatible(
                        "resident checkpoint restore implementation differs".to_owned(),
                    ));
                }
                ImageProgramStageOperationV1::CaptureCheckpoint { implementation, .. }
                    if implementation != capture_implementation =>
                {
                    return Err(Error::Incompatible(
                        "resident checkpoint capture implementation differs".to_owned(),
                    ));
                }
                ImageProgramStageOperationV1::MaskBlend { .. }
                | ImageProgramStageOperationV1::RestoreCheckpoint { .. }
                | ImageProgramStageOperationV1::CaptureCheckpoint { .. } => {}
            }
        }
        Ok(())
    }

    fn validate_krea_activation_program(&self, plan: &ImageProgramPlanV1) -> Result<()> {
        let Some(activation) = &self.runtime.krea_activation else {
            return Ok(());
        };
        let mut diffusion_stages = plan.stages.iter().filter_map(|stage| {
            let ImageProgramStageOperationV1::Native { plan } = &stage.operation else {
                return None;
            };
            uses_diffusion(plan.operation).then_some(plan)
        });
        let stage = diffusion_stages.next().ok_or_else(|| {
            Error::Invalid(
                "an installed Krea activation plan requires one diffusion stage".to_owned(),
            )
        })?;
        if diffusion_stages.next().is_some() {
            return Err(Error::Invalid(
                "one Krea activation plan cannot span multiple diffusion stages".to_owned(),
            ));
        }
        let schedule = stage.schedule.as_ref().ok_or_else(|| {
            Error::Invalid("the Krea activation diffusion stage has no schedule".to_owned())
        })?;
        let transitions = schedule.sigmas.len().checked_sub(1).ok_or_else(|| {
            Error::Invalid("the Krea activation schedule has no transition".to_owned())
        })?;
        if u32::try_from(transitions).ok() != Some(activation.plan.step_count) {
            return Err(Error::Invalid(
                "Krea activation transition count differs from the diffusion stage".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone)]
struct CapturedState {
    plan: DiffusionPlan,
    step: u32,
}

enum PreparedInput {
    Text {
        bytes: Vec<u8>,
        text: String,
    },
    Image {
        bytes: Vec<u8>,
        width: u32,
        height: u32,
        channels: u32,
    },
    Tensor {
        values: Vec<f32>,
        shape: Vec<i64>,
    },
    Checkpoint {
        envelope: Vec<u8>,
        checkpoint: DiffusionCheckpoint,
    },
    Lora {
        path: CString,
        high_noise: bool,
    },
}

enum ResidentObservation {
    Accumulator(Box<ObservationAccumulator>),
    Snapshot,
}

struct ResidentStageProgram<'a> {
    implementation: Digest,
    expected_schedule: logit_loom_diffusion::DiffusionSchedule,
    expected_rng: Digest,
    expected_seed: u64,
    operators: Vec<logit_loom_diffusion::OperatorInvocation>,
    observations: Vec<ResidentObservation>,
    restore: Option<DiffusionCheckpoint>,
    restored: bool,
    pipeline: Option<Pipeline>,
    actual_plan: Option<DiffusionPlan>,
    checkpoint_backend: Digest,
    cancellation: &'a dyn CancellationProbe,
}

impl<'a> ResidentStageProgram<'a> {
    fn new(
        stage: &ImageProgramNativeStageV1,
        operation: Digest,
        restore: Option<DiffusionCheckpoint>,
        checkpoint_backend: Digest,
        cancellation: &'a dyn CancellationProbe,
    ) -> Result<Self> {
        let expected_schedule = stage.schedule.clone().ok_or_else(|| {
            Error::Invalid("resident diffusion stage is missing its schedule".to_owned())
        })?;
        let expected_seed = fixed_seed(&stage.seed)?;
        let observations = stage
            .observations
            .iter()
            .cloned()
            .map(|observation| {
                if observation.kind == ObservationKind::Snapshot {
                    Ok(ResidentObservation::Snapshot)
                } else {
                    ObservationAccumulator::new(observation)
                        .map(Box::new)
                        .map(ResidentObservation::Accumulator)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            implementation: operation,
            expected_schedule,
            expected_rng: stage.rng.clone(),
            expected_seed,
            operators: stage.operators.clone(),
            observations,
            restore,
            restored: false,
            pipeline: None,
            actual_plan: None,
            checkpoint_backend,
            cancellation,
        })
    }

    fn observation_receipts(&self, snapshot_contents: &[Digest]) -> Result<Vec<Digest>> {
        let mut snapshots = snapshot_contents.iter();
        let receipts = self
            .observations
            .iter()
            .map(|observation| match observation {
                ResidentObservation::Accumulator(accumulator) => Ok(accumulator.finish()),
                ResidentObservation::Snapshot => snapshots.next().cloned().ok_or_else(|| {
                    Error::Incompatible("native snapshot receipt is missing".to_owned())
                }),
            })
            .collect::<Result<Vec<_>>>()?;
        if snapshots.next().is_some() {
            return Err(Error::Incompatible(
                "native snapshot receipt count exceeds the plan".to_owned(),
            ));
        }
        Ok(receipts)
    }
}

impl StepProgram for ResidentStageProgram<'_> {
    fn implementation(&self) -> &Digest {
        &self.implementation
    }

    fn begin(&mut self, plan: &DiffusionPlan) -> std::result::Result<(), String> {
        if plan.schedule != self.expected_schedule
            || plan.rng != self.expected_rng
            || plan.seed != self.expected_seed
        {
            return Err("resident native schedule, RNG, or seed differs".to_owned());
        }
        if let Some(checkpoint) = &self.restore {
            checkpoint
                .receipt()
                .validate_for(plan)
                .map_err(|error| error.to_string())?;
            if checkpoint.receipt().backend != self.checkpoint_backend {
                return Err("resident checkpoint backend identity differs".to_owned());
            }
        }
        let mut interventions: Vec<Box<dyn Intervention>> =
            Vec::with_capacity(self.operators.len());
        for operator in self
            .operators
            .iter()
            .filter(|operator| operator.selector == TensorSelector::SchedulerState)
        {
            interventions.push(Box::new(
                InstalledChannelBias::from_invocation(operator, plan)
                    .map_err(|error| error.to_string())?,
            ));
        }
        if !interventions.is_empty() {
            let mut pipeline = Pipeline::new(
                plan.digest().map_err(|error| error.to_string())?,
                plan.tensor.clone(),
                interventions,
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
            .ok_or_else(|| "resident step program was not initialized".to_owned())?;
        if let Some(checkpoint) = &self.restore
            && checkpoint.receipt().next_step == context.step_index.saturating_add(1)
        {
            if self.restored {
                return Err("resident checkpoint restore boundary repeated".to_owned());
            }
            checkpoint
                .restore(plan, &self.checkpoint_backend, context, state)
                .map_err(|error| error.to_string())?;
            self.restored = true;
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
            if let ResidentObservation::Accumulator(accumulator) = observation {
                accumulator.record(context, state)?;
            }
        }
        Ok(if self.cancellation.is_cancelled() {
            ControlFlow::Stop
        } else {
            ControlFlow::Continue
        })
    }
}

impl<R: ResidentArtifactPathResolver> ResidentImageProgramBackend for SdcppResidentProgram<'_, R> {
    fn backend_identity(&self) -> Result<Digest> {
        resident_backend_identity(self.runtime)
    }

    fn runtime_epoch(&self) -> u64 {
        self.runtime.session_epoch
    }

    fn validate_program(&self, plan: &ImageProgramPlanV1) -> Result<()> {
        plan.validate().map_err(logit_loom_diffusion::Error::from)?;
        self.validate_runtime_state()?;
        let bindings = self.runtime.execution_bindings()?;
        let checkpoint_compatibility = resident_checkpoint_compatibility_v1(&bindings)?;
        let restore_implementation =
            resident_checkpoint_conversion_v1(&checkpoint_compatibility, true)?;
        let capture_implementation =
            resident_checkpoint_conversion_v1(&checkpoint_compatibility, false)?;
        Self::validate_value_specs(plan, &checkpoint_compatibility)?;
        Self::validate_external_routes(plan)?;
        self.validate_stage_bindings(
            plan,
            &bindings,
            &restore_implementation,
            &capture_implementation,
        )?;
        self.validate_krea_activation_program(plan)
    }

    fn begin_program(
        &mut self,
        plan: &ImageProgramPlanV1,
        inputs: &[InputBuffer<'_>],
    ) -> Result<Vec<ImageProgramValueMeasurementV1>> {
        self.validate_program(plan)?;
        if let Some(activation) = self.runtime.krea_activation.take() {
            let result = activation.verify_resident(self.runtime);
            self.runtime.krea_activation = Some(activation);
            result?;
        }
        let prepared = self.prepare_inputs(plan, inputs)?;
        let liveness = plan.liveness().map_err(logit_loom_diffusion::Error::from)?;
        let mut arena = std::ptr::null_mut();
        // SAFETY: The runtime is exclusively borrowed, resident, and has no
        // active callbacks. The out-pointer remains live for this call.
        let status = unsafe {
            self.runtime.api.program_begin_v3(
                self.runtime.context.as_ptr(),
                plan.values.len(),
                liveness.peak_bytes,
                &mut arena,
            )
        };
        if status != ffi::STATUS_OK {
            return Err(native_status_error(status));
        }
        let arena = NonNull::new(arena).ok_or_else(|| {
            Error::Native("companion returned an empty resident arena".to_owned())
        })?;
        self.arena = Some(arena);
        self.runtime.state = ExecutorState::Busy;
        self.handles.clear();
        self.input_checkpoints.clear();
        self.checkpoint_states.clear();
        self.captured_states.clear();
        self.text_values.clear();

        let mut measurements = Vec::with_capacity(prepared.len());
        for (value, input) in prepared {
            let handle = self.import_prepared(value, input)?;
            self.insert_handle(value, handle)?;
            measurements.push(self.measurement(plan, value, handle)?);
        }
        Ok(measurements)
    }

    fn execute_stage(
        &mut self,
        plan: &ImageProgramPlanV1,
        stage: &ImageProgramStageV1,
        cancellation: &dyn CancellationProbe,
    ) -> Result<ResidentProgramStageTerminal> {
        let started = Instant::now();
        let execution = match self.execute_stage_operation(plan, stage, cancellation)? {
            StageOperationOutcome::Completed(execution) => execution,
            StageOperationOutcome::Cancelled { step } => {
                return Ok(ResidentProgramStageTerminal::CancelledAfterStep { step });
            }
        };
        let (receipts, measurements) = self.receipts_for_values(plan, &execution.outputs)?;
        let wall_time_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        Ok(ResidentProgramStageTerminal::Completed(
            ResidentProgramCompletedStage {
                receipt: ImageProgramStageReceiptV1 {
                    stage: stage.stage,
                    operation: stage
                        .operation
                        .digest()
                        .map_err(logit_loom_diffusion::Error::from)?,
                    outputs: receipts,
                    observations: execution.observations,
                },
                model_block_applications: execution.model_block_applications,
                wall_time_ns,
                native_time_ns: execution.native_time_ns,
                values: measurements,
            },
        ))
    }

    fn materialize_value(
        &mut self,
        _plan: &ImageProgramPlanV1,
        value: u16,
        output: &mut [u8],
    ) -> Result<usize> {
        self.copy_handle_into(self.handle(value)?, output)
    }

    fn release_value(&mut self, value: u16) -> Result<()> {
        let handle = self
            .handles
            .remove(&value)
            .ok_or_else(|| Error::Invalid(format!("resident value {value} is not live")))?;
        // SAFETY: This handle was removed from the live map and is released
        // exactly once against its originating arena.
        let status = unsafe {
            self.runtime
                .api
                .program_release_v3(self.arena()?.as_ptr(), handle)
        };
        if status != ffi::STATUS_OK {
            return Err(Error::Native(format!(
                "native release of value {value} failed: {}",
                native_status_error(status)
            )));
        }
        self.input_checkpoints.remove(&value);
        self.checkpoint_states.remove(&value);
        self.captured_states.remove(&value);
        self.text_values.remove(&value);
        Ok(())
    }

    fn finish_program(&mut self, cleanup: ImageCleanupPolicy) -> Result<ResidentProgramFinish> {
        let arena = self
            .arena
            .take()
            .ok_or_else(|| Error::Native("resident program arena is not active".to_owned()))?;
        let clear = cleanup == ImageCleanupPolicy::ClearSession;
        let mut peak_arena_bytes = 0;
        // SAFETY: `arena` is removed from this owner and consumed exactly once.
        let status = unsafe {
            self.runtime
                .api
                .program_finish_v3(arena.as_ptr(), clear, &mut peak_arena_bytes)
        };
        self.handles.clear();
        self.input_checkpoints.clear();
        self.checkpoint_states.clear();
        self.captured_states.clear();
        self.text_values.clear();
        if status != ffi::STATUS_OK {
            self.runtime.state = ExecutorState::Poisoned;
            return Err(Error::Poisoned(format!(
                "resident native cleanup failed: {}",
                native_status_error(status)
            )));
        }
        self.runtime.state = ExecutorState::Resident;
        let disposition = if clear {
            // Successful native session clearing includes the v6 resident
            // activation store and advances its handle generation.
            self.runtime.krea_activation.take();
            let cleared_epoch = self.runtime.session_epoch;
            self.runtime.session_epoch =
                self.runtime.session_epoch.checked_add(1).ok_or_else(|| {
                    self.runtime.state = ExecutorState::Poisoned;
                    Error::Poisoned("resident session epoch overflowed".to_owned())
                })?;
            ImageProgramCleanupDispositionV1::Confirmed { cleared_epoch }
        } else {
            ImageProgramCleanupDispositionV1::Retained
        };
        Ok(ResidentProgramFinish {
            cleanup: disposition,
            peak_arena_bytes,
        })
    }

    fn poison(&mut self) {
        if let Some(arena) = self.arena.take() {
            let mut ignored_peak = 0;
            // SAFETY: Best-effort terminal cleanup consumes the only arena
            // pointer. The runtime remains poisoned regardless of the result.
            let _ = unsafe {
                self.runtime
                    .api
                    .program_finish_v3(arena.as_ptr(), true, &mut ignored_peak)
            };
        }
        self.handles.clear();
        self.input_checkpoints.clear();
        self.checkpoint_states.clear();
        self.captured_states.clear();
        self.text_values.clear();
        self.runtime.krea_activation.take();
        self.runtime.state = ExecutorState::Poisoned;
    }
}

impl<R: ResidentArtifactPathResolver> SdcppResidentProgram<'_, R> {
    fn execute_stage_operation(
        &mut self,
        plan: &ImageProgramPlanV1,
        stage: &ImageProgramStageV1,
        cancellation: &dyn CancellationProbe,
    ) -> Result<StageOperationOutcome> {
        let execution = match &stage.operation {
            ImageProgramStageOperationV1::Native { plan: native }
                if uses_diffusion(native.operation) =>
            {
                return match self.execute_diffusion_stage(plan, stage, native, cancellation)? {
                    NativeStageOutcome::Completed {
                        outputs,
                        observations,
                        model_block_applications,
                        native_time_ns,
                    } => Ok(StageOperationOutcome::Completed(StageExecution {
                        outputs,
                        observations,
                        model_block_applications,
                        native_time_ns,
                    })),
                    NativeStageOutcome::Cancelled { step } => {
                        Ok(StageOperationOutcome::Cancelled { step })
                    }
                };
            }
            ImageProgramStageOperationV1::Native { plan: native } => {
                StageExecution::without_observations(self.execute_vae_stage(plan, native)?)
            }
            ImageProgramStageOperationV1::MaskBlend {
                base,
                overlay,
                mask,
                output,
            } => StageExecution::without_observations(
                self.execute_mask_blend(*base, *overlay, *mask, *output)?,
            ),
            ImageProgramStageOperationV1::RestoreCheckpoint {
                checkpoint, state, ..
            } => StageExecution::without_observations(
                self.execute_checkpoint_restore(*checkpoint, *state)?,
            ),
            ImageProgramStageOperationV1::CaptureCheckpoint {
                state, checkpoint, ..
            } => StageExecution::without_observations(
                self.execute_checkpoint_capture(*state, *checkpoint)?,
            ),
        };
        Ok(StageOperationOutcome::Completed(execution))
    }

    fn execute_mask_blend(
        &mut self,
        base: u16,
        overlay: u16,
        mask: u16,
        output: u16,
    ) -> Result<Vec<u16>> {
        let mut handle = ValueHandleV3::EMPTY;
        // SAFETY: Every input handle belongs to this exclusively owned live
        // arena and the out-handle is writable.
        let status = unsafe {
            self.runtime.api.program_mask_blend_v3(
                self.arena()?.as_ptr(),
                self.handle(base)?,
                self.handle(overlay)?,
                self.handle(mask)?,
                &mut handle,
            )
        };
        Self::require_stage_status(status, "resident mask blend")?;
        self.insert_handle(output, handle)?;
        Ok(vec![output])
    }

    fn execute_checkpoint_restore(&mut self, checkpoint: u16, state: u16) -> Result<Vec<u16>> {
        let restored = self
            .input_checkpoints
            .get(&checkpoint)
            .cloned()
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "checkpoint value {checkpoint} is not an authenticated external input"
                ))
            })?;
        let values = f32_from_le_bytes(restored.state_bytes())?;
        let shape = vec![
            i64::try_from(values.len())
                .map_err(|_| Error::Invalid("checkpoint state length exceeds i64".to_owned()))?,
        ];
        let handle = self.import_tensor(&values, &shape, true)?;
        self.insert_handle(state, handle)?;
        self.checkpoint_states.insert(state, restored);
        Ok(vec![state])
    }

    fn execute_checkpoint_capture(&mut self, state: u16, checkpoint: u16) -> Result<Vec<u16>> {
        let metadata = self.captured_states.get(&state).cloned().ok_or_else(|| {
            Error::Invalid(format!(
                "checkpoint state value {state} was not captured by a native stage"
            ))
        })?;
        let state_bytes = self.copy_handle(self.handle(state)?)?;
        let values = f32_from_native_bytes(&state_bytes)?;
        let context = StepContext::for_plan(
            &metadata.plan,
            usize::try_from(metadata.step)
                .map_err(|_| Error::Invalid("checkpoint step exceeds usize".to_owned()))?,
        )
        .map_err(logit_loom_diffusion::Error::from)?;
        let checkpoint_value = DiffusionCheckpoint::capture(
            &metadata.plan,
            &self.runtime.execution_bindings()?.backend,
            &context,
            &values,
        )?;
        let envelope = checkpoint_value.to_envelope_bytes()?;
        let handle = self.import_bytes(&envelope)?;
        self.insert_handle(checkpoint, handle)?;
        Ok(vec![checkpoint])
    }

    fn prepare_inputs(
        &mut self,
        plan: &ImageProgramPlanV1,
        inputs: &[InputBuffer<'_>],
    ) -> Result<Vec<(u16, PreparedInput)>> {
        if inputs.len() != plan.inputs.len() {
            return Err(Error::Invalid(
                "resident inputs differ from the declared program count".to_owned(),
            ));
        }
        let mut prepared = Vec::with_capacity(inputs.len());
        for (binding, input) in plan.inputs.iter().zip(inputs) {
            if input.specification() != &binding.buffer {
                return Err(Error::Invalid(format!(
                    "resident input value {} metadata differs",
                    binding.value
                )));
            }
            let bytes = input.bytes();
            let value = match &plan.values[usize::from(binding.value)].spec {
                ImageProgramValueSpecV1::Utf8 { .. } => {
                    let text = std::str::from_utf8(bytes).map_err(|_| {
                        Error::Invalid(format!(
                            "resident text value {} is not UTF-8",
                            binding.value
                        ))
                    })?;
                    if text.is_empty() || text.contains('\0') {
                        return Err(Error::Invalid(format!(
                            "resident text value {} is empty or contains NUL",
                            binding.value
                        )));
                    }
                    PreparedInput::Text {
                        bytes: bytes.to_vec(),
                        text: text.to_owned(),
                    }
                }
                ImageProgramValueSpecV1::Rgb8 { width, height } => PreparedInput::Image {
                    bytes: bytes.to_vec(),
                    width: *width,
                    height: *height,
                    channels: 3,
                },
                ImageProgramValueSpecV1::Rgba8 { width, height } => PreparedInput::Image {
                    bytes: bytes.to_vec(),
                    width: *width,
                    height: *height,
                    channels: 4,
                },
                ImageProgramValueSpecV1::Gray8 { width, height } => PreparedInput::Image {
                    bytes: bytes.to_vec(),
                    width: *width,
                    height: *height,
                    channels: 1,
                },
                ImageProgramValueSpecV1::Png { .. } => {
                    return Err(Error::Invalid(format!(
                        "resident PNG value {} cannot be an external native input",
                        binding.value
                    )));
                }
                ImageProgramValueSpecV1::Tensor { tensor, .. } => PreparedInput::Tensor {
                    values: f32_from_le_bytes(bytes)?,
                    shape: i64_shape(&tensor.shape)?,
                },
                ImageProgramValueSpecV1::Checkpoint { .. } => {
                    let checkpoint = DiffusionCheckpoint::from_envelope_bytes(bytes)?;
                    PreparedInput::Checkpoint {
                        envelope: bytes.to_vec(),
                        checkpoint,
                    }
                }
                ImageProgramValueSpecV1::Opaque {
                    opaque_kind: ImageOpaqueValueKindV1::LoraArtifact,
                    ..
                } => {
                    let path = self
                        .artifact_paths
                        .resolve_lora_path(binding.value, input)
                        .map_err(|message| {
                            Error::Invalid(format!(
                                "resident LoRA value {} path resolution failed: {message}",
                                binding.value
                            ))
                        })?;
                    PreparedInput::Lora {
                        path: path_c_string(&path)?,
                        high_noise: lora_high_noise(plan, binding.value)?,
                    }
                }
                ImageProgramValueSpecV1::Opaque { .. } => {
                    return Err(Error::Invalid(format!(
                        "resident opaque value {} cannot be an external input",
                        binding.value
                    )));
                }
            };
            prepared.push((binding.value, value));
        }
        Ok(prepared)
    }

    fn import_prepared(&mut self, value: u16, prepared: PreparedInput) -> Result<ValueHandleV3> {
        match prepared {
            PreparedInput::Text { bytes, text } => {
                let handle = self.import_bytes(&bytes)?;
                self.text_values.insert(value, text);
                Ok(handle)
            }
            PreparedInput::Image {
                bytes,
                width,
                height,
                channels,
            } => {
                let view = ImageViewV2 {
                    data: bytes.as_ptr(),
                    bytes: bytes.len(),
                    width,
                    height,
                    channels,
                };
                let mut handle = ValueHandleV3::EMPTY;
                // SAFETY: `bytes` and `view` remain live for the synchronous
                // copy into this exclusively owned arena.
                let status = unsafe {
                    self.runtime.api.program_import_image_v3(
                        self.arena()?.as_ptr(),
                        &view,
                        &mut handle,
                    )
                };
                Self::require_import_status(status, value)?;
                Ok(handle)
            }
            PreparedInput::Tensor { values, shape } => self.import_tensor(&values, &shape, false),
            PreparedInput::Checkpoint {
                envelope,
                checkpoint,
            } => {
                let handle = self.import_bytes(&envelope)?;
                self.input_checkpoints.insert(value, checkpoint);
                Ok(handle)
            }
            PreparedInput::Lora { path, high_noise } => {
                let mut handle = ValueHandleV3::EMPTY;
                // SAFETY: The bounded C string remains live for this
                // synchronous call and the companion copies the path.
                let status = unsafe {
                    self.runtime.api.program_import_lora_v3(
                        self.arena()?.as_ptr(),
                        path.as_ptr(),
                        high_noise,
                        &mut handle,
                    )
                };
                Self::require_import_status(status, value)?;
                Ok(handle)
            }
        }
    }

    fn import_bytes(&self, bytes: &[u8]) -> Result<ValueHandleV3> {
        let mut handle = ValueHandleV3::EMPTY;
        // SAFETY: The byte slice remains live for the synchronous arena copy.
        let status = unsafe {
            self.runtime
                .api
                .program_import_bytes_v3(self.arena()?.as_ptr(), bytes, &mut handle)
        };
        if status != ffi::STATUS_OK {
            return Err(Error::Native(format!(
                "resident byte import failed: {}",
                native_status_error(status)
            )));
        }
        Ok(handle)
    }

    fn import_tensor(
        &self,
        values: &[f32],
        shape: &[i64],
        checkpoint: bool,
    ) -> Result<ValueHandleV3> {
        let view = TensorViewV2 {
            abi_version: crate::IMAGE_ABI_VERSION,
            data: values.as_ptr(),
            element_count: values.len(),
            shape: shape.as_ptr(),
            rank: shape.len(),
        };
        let mut handle = ValueHandleV3::EMPTY;
        // SAFETY: Both slices remain live and exact for this synchronous copy.
        let status = unsafe {
            self.runtime.api.program_import_tensor_v3(
                self.arena()?.as_ptr(),
                &view,
                checkpoint,
                &mut handle,
            )
        };
        if status != ffi::STATUS_OK {
            return Err(Error::Native(format!(
                "resident tensor import failed: {}",
                native_status_error(status)
            )));
        }
        Ok(handle)
    }

    fn execute_vae_stage(
        &mut self,
        plan: &ImageProgramPlanV1,
        native: &ImageProgramNativeStageV1,
    ) -> Result<Vec<u16>> {
        let input_role = match native.operation {
            ImageOperation::VaeEncode => ImageBufferRole::SourceImage,
            ImageOperation::VaeDecode => ImageBufferRole::TensorSnapshot,
            _ => {
                return Err(Error::Invalid(
                    "diffusion operation passed direct-VAE lowering".to_owned(),
                ));
            }
        };
        let input = native
            .inputs
            .iter()
            .find(|input| input.role == input_role)
            .ok_or_else(|| Error::Invalid("direct-VAE input is missing".to_owned()))?;
        let output = native
            .outputs
            .first()
            .ok_or_else(|| Error::Invalid("direct-VAE primary output is missing".to_owned()))?;
        let decoded_output = if native.operation == ImageOperation::VaeDecode {
            Some(native_output_contract(plan, native)?)
        } else {
            None
        };
        let mut handle = ValueHandleV3::EMPTY;
        // SAFETY: The exact live input handle and writable output belong to
        // this exclusively owned arena.
        let status = unsafe {
            match native.operation {
                ImageOperation::VaeEncode => self.runtime.api.program_vae_encode_v3(
                    self.arena()?.as_ptr(),
                    self.handle(input.value)?,
                    &mut handle,
                ),
                ImageOperation::VaeDecode => {
                    let output = decoded_output
                        .ok_or_else(|| Error::Invalid("VAE decode output is missing".to_owned()))?;
                    self.runtime.api.program_vae_decode_v3(
                        self.arena()?.as_ptr(),
                        self.handle(input.value)?,
                        &output,
                        &mut handle,
                    )
                }
                _ => unreachable!("operation matched above"),
            }
        };
        Self::require_stage_status(status, "resident direct VAE")?;
        self.insert_handle(output.value, handle)?;
        Ok(vec![output.value])
    }

    fn execute_diffusion_stage(
        &mut self,
        program: &ImageProgramPlanV1,
        stage: &ImageProgramStageV1,
        native: &ImageProgramNativeStageV1,
        cancellation: &dyn CancellationProbe,
    ) -> Result<NativeStageOutcome> {
        let prepared = self.prepare_diffusion_call(program, native)?;
        match self.run_diffusion_call(stage, native, cancellation, &prepared)? {
            DiffusionCallOutcome::Cancelled { step } => Ok(NativeStageOutcome::Cancelled { step }),
            DiffusionCallOutcome::Completed(completed) => {
                self.publish_diffusion_outputs(native, *completed)
            }
        }
    }

    fn prepare_diffusion_call(
        &self,
        program: &ImageProgramPlanV1,
        native: &ImageProgramNativeStageV1,
    ) -> Result<PreparedDiffusionCall> {
        let positive_value = value_for_role(native, ImageBufferRole::PositiveConditioning)?
            .ok_or_else(|| Error::Invalid("positive conditioning is missing".to_owned()))?;
        let prompt = self
            .text_values
            .get(&positive_value)
            .cloned()
            .ok_or_else(|| {
                Error::Invalid("positive conditioning is not an imported UTF-8 value".to_owned())
            })?;
        let negative_value = value_for_role(native, ImageBufferRole::NegativeConditioning)?;
        let positive = self.handle(positive_value)?;
        let negative = negative_value
            .map(|value| self.handle(value))
            .transpose()?
            .unwrap_or(ValueHandleV3::EMPTY);
        let init_image = value_for_role(native, ImageBufferRole::SourceImage)?
            .map(|value| self.handle(value))
            .transpose()?
            .unwrap_or(ValueHandleV3::EMPTY);
        let mask_image = value_for_role(native, ImageBufferRole::Mask)?
            .map(|value| self.handle(value))
            .transpose()?
            .unwrap_or(ValueHandleV3::EMPTY);
        let reference_images = native
            .inputs
            .iter()
            .filter(|input| input.role == ImageBufferRole::ReferenceImage)
            .map(|input| self.handle(input.value))
            .collect::<Result<Vec<_>>>()?;
        let scale_points = native
            .loras
            .iter()
            .map(|lora| {
                lora.scales
                    .points
                    .iter()
                    .map(|point| LoraScalePointV3 {
                        step: point.step,
                        scale: point.scale(),
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let lora_schedules = native
            .loras
            .iter()
            .zip(&scale_points)
            .map(|(lora, points)| {
                Ok(LoraScheduleV3 {
                    lora: self.handle(lora.value)?,
                    points: pointer_or_null(points),
                    point_count: points.len(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let model_blocks = lower_model_block_operators(&native.operators)?;
        let snapshot_steps = native
            .observations
            .iter()
            .filter_map(|observation| {
                if observation.kind != ObservationKind::Snapshot {
                    return None;
                }
                match &observation.steps {
                    StepSelector::Exact { steps } => steps.first().copied(),
                    StepSelector::All => None,
                }
            })
            .collect::<Vec<_>>();
        let seed = fixed_seed(&native.seed)?;
        let schedule = native.schedule.clone().ok_or_else(|| {
            Error::Invalid("resident diffusion stage is missing a schedule".to_owned())
        })?;
        let restore = value_for_role(native, ImageBufferRole::Checkpoint)?
            .map(|value| {
                self.checkpoint_states.get(&value).cloned().ok_or_else(|| {
                    Error::Invalid(
                        "native checkpoint input was not produced by a restore stage".to_owned(),
                    )
                })
            })
            .transpose()?;
        let output = native_output_contract(program, native)?;
        Ok(PreparedDiffusionCall {
            prompt,
            positive,
            negative,
            init_image,
            mask_image,
            reference_images,
            _scale_points: scale_points,
            lora_schedules,
            _model_block_steps: model_blocks.steps,
            model_block_operators: model_blocks.operators,
            snapshot_steps,
            seed,
            schedule,
            restore,
            output,
        })
    }

    #[allow(clippy::too_many_lines)]
    fn run_diffusion_call<'c>(
        &mut self,
        stage: &ImageProgramStageV1,
        native: &ImageProgramNativeStageV1,
        cancellation: &'c dyn CancellationProbe,
        prepared: &PreparedDiffusionCall,
    ) -> Result<DiffusionCallOutcome<'c>> {
        let request = ImageRequest::new(
            prepared.prompt.clone(),
            native.width,
            native.height,
            prepared.seed,
            native.guidance_scale(),
            prepared.schedule.clone(),
        )?;
        let bindings = self.runtime.execution_bindings()?;
        let checkpoint_backend = bindings.backend.clone();
        let mut stage_program = ResidentStageProgram::new(
            native,
            stage
                .operation
                .digest()
                .map_err(logit_loom_diffusion::Error::from)?,
            prepared.restore.clone(),
            checkpoint_backend.clone(),
            cancellation,
        )?;
        let profile = self.runtime.profile;
        let profile_receipt = self.runtime.profile_receipt.clone();
        let native_receipt = self.runtime.native_receipt.clone();
        let components = component_map(&profile_receipt, &native_receipt)?;
        let mut callbacks = CallbackState::new_full(
            profile,
            &profile_receipt,
            &native_receipt,
            &request,
            components,
            &mut stage_program,
            None,
        )?;
        let callback_pointer = (&raw mut callbacks).cast::<c_void>();
        let params = Self::diffusion_params(native, prepared)?;
        let mut snapshot_handles = vec![ValueHandleV3::EMPTY; prepared.snapshot_steps.len()];
        let mut krea_call = match self.runtime.krea_activation.take() {
            Some(activation) => {
                let prepared = activation
                    .verify_resident(self.runtime)
                    .and_then(|()| Ok((activation.lower()?, activation.callback_state())));
                self.runtime.krea_activation = Some(activation);
                Some(prepared?)
            }
            None => None,
        };
        let krea_callback_pointer = krea_call
            .as_mut()
            .map_or(std::ptr::null_mut(), |(_, callback)| {
                (&raw mut *callback).cast::<c_void>()
            });
        let krea_native = krea_call
            .as_ref()
            .map(|(lowered, _)| (lowered, krea_callback_pointer));
        // SAFETY: Every handle and borrowed array remains live for this
        // synchronous call. Both callbacks contain panics and validate all
        // native descriptors before forming Rust slices.
        let mut invocation = unsafe {
            self.invoke_diffusion_native(
                params,
                prepared,
                callback_pointer,
                &mut snapshot_handles,
                krea_native,
            )
        }?;
        let callback_error = callbacks.take_error();
        let last_step = callbacks.last_completed_step();
        let native_time_ns = callbacks.native_time_ns();
        let actual_plan = callbacks.plan().cloned();
        drop(callbacks);
        if let Some(error) = callback_error {
            return Err(Error::Callback(error));
        }
        if let Some(error) = krea_call
            .as_mut()
            .and_then(|(_, callback)| callback.take_error())
        {
            return Err(Error::Callback(error));
        }
        if invocation.status == ffi::STATUS_STOPPED {
            let step = last_step.ok_or_else(|| {
                Error::Incompatible(
                    "native cancellation returned without a completed boundary".to_owned(),
                )
            })?;
            self.finish_krea_invocation(
                invocation.krea_activation.take(),
                krea_call,
                KreaActivationTerminalV1::Cancelled {
                    after_transition: Some(step),
                },
            )?;
            return Ok(DiffusionCallOutcome::Cancelled { step });
        }
        Self::require_stage_status(invocation.status, "resident diffusion")?;
        let model_block_applications =
            verified_model_block_applications(stage.stage, prepared, &invocation)?;
        let result = invocation.result;
        if result.abi_version != PROGRAM_ABI_VERSION
            || result.primary.is_empty()
            || result.snapshot_count != snapshot_handles.len()
            || result.checkpoint_state.is_empty() != native.checkpoint_after_step.is_none()
            || stage_program.restored != native.checkpoint_restore_at_step.is_some()
        {
            return Err(Error::Incompatible(
                "resident diffusion result or checkpoint accounting differs".to_owned(),
            ));
        }
        let actual_plan = actual_plan.ok_or_else(|| {
            Error::Incompatible("resident diffusion produced no exact step plan".to_owned())
        })?;
        self.finish_krea_invocation(
            invocation.krea_activation.take(),
            krea_call,
            KreaActivationTerminalV1::Completed,
        )?;
        Ok(DiffusionCallOutcome::Completed(Box::new(
            CompletedDiffusionCall {
                result,
                model_block_applications,
                snapshot_handles,
                stage_program,
                actual_plan,
                native_time_ns,
            },
        )))
    }

    fn finish_krea_invocation(
        &mut self,
        native: Option<NativeKreaInvocation>,
        callback: Option<(LoweredKreaActivation, KreaCallbackState)>,
        terminal: KreaActivationTerminalV1,
    ) -> Result<()> {
        let runtime_epoch = self.runtime.session_epoch;
        match (native, callback, self.runtime.krea_activation.as_mut()) {
            (None, None, None) => Ok(()),
            (Some(native), Some((_lowered, callback)), Some(activation)) => {
                let execution = activation.finish_job(
                    runtime_epoch,
                    terminal,
                    &native.captures,
                    &native.applications,
                    native.peak_host_bytes,
                    native.peak_device_bytes,
                    callback,
                )?;
                self.krea_activation_executions.push(execution);
                Ok(())
            }
            _ => {
                self.runtime.state = ExecutorState::Poisoned;
                Err(Error::Poisoned(
                    "native Krea activation invocation coverage differs".to_owned(),
                ))
            }
        }
    }

    /// Invokes the exact resident ABI selected by the prepared operators.
    ///
    /// # Safety
    ///
    /// The callback state, every array nested beneath `params` and `prepared`,
    /// snapshot storage, and result storage must remain live for the complete
    /// synchronous native call.
    #[allow(clippy::too_many_lines)]
    unsafe fn invoke_diffusion_native(
        &mut self,
        params: ProgramImageParamsV3,
        prepared: &PreparedDiffusionCall,
        callback_pointer: *mut c_void,
        snapshot_handles: &mut [ValueHandleV3],
        krea_activation: Option<(&LoweredKreaActivation, *mut c_void)>,
    ) -> Result<NativeDiffusionInvocation> {
        let arena = self.arena()?.as_ptr();
        if prepared.model_block_operators.is_empty() && krea_activation.is_none() {
            let mut result = ProgramImageResultV3::default();
            // SAFETY: Forwarded from this method's caller contract.
            let status = unsafe {
                self.runtime.api.program_generate_image_v3(
                    arena,
                    &params,
                    condition_callback,
                    callback_pointer,
                    step_callback,
                    callback_pointer,
                    snapshot_handles,
                    &mut result,
                )
            };
            return Ok(NativeDiffusionInvocation {
                status,
                result,
                applications: Vec::new(),
                transition_masks: Vec::new(),
                transition_words_per_operator: 0,
                krea_activation: None,
            });
        }
        let step_count = params.sigma_count.checked_sub(1).ok_or_else(|| {
            Error::Invalid("resident diffusion schedule has no transition".to_owned())
        })?;
        let transition_words_per_operator = if prepared.model_block_operators.is_empty() {
            0
        } else {
            step_count.checked_add(63).ok_or_else(|| {
                Error::Invalid("resident model-block transition count overflowed".to_owned())
            })? / 64
        };
        let transition_mask_words = prepared
            .model_block_operators
            .len()
            .checked_mul(transition_words_per_operator)
            .ok_or_else(|| {
                Error::Invalid("resident model-block transition mask overflowed".to_owned())
            })?;
        let mut applications =
            vec![NativeModelBlockApplicationV5::default(); prepared.model_block_operators.len()];
        let mut transition_masks = vec![0_u64; transition_mask_words];
        let params = ProgramImageParamsV5 {
            abi_version: MODEL_BLOCK_ABI_VERSION,
            image: params,
            model_block_operators: pointer_or_null(&prepared.model_block_operators),
            model_block_operator_count: prepared.model_block_operators.len(),
        };
        if let Some((activation, activation_callback_pointer)) = krea_activation {
            let mut captures = vec![KreaCaptureResultV6::default(); activation.captures.len()];
            let mut activation_applications =
                vec![KreaApplicationResultV6::default(); activation.operations.len()];
            let params = ProgramImageParamsV6 {
                abi_version: crate::KREA_ACTIVATION_ABI_VERSION,
                image: params,
                captures: pointer_or_null(&activation.captures),
                capture_count: activation.captures.len(),
                operations: pointer_or_null(&activation.operations),
                operation_count: activation.operations.len(),
                maximum_host_bytes: self
                    .runtime
                    .krea_activation
                    .as_ref()
                    .map_or(0, |installed| installed.plan.maximum_host_bytes),
                maximum_device_bytes: self
                    .runtime
                    .krea_activation
                    .as_ref()
                    .map_or(0, |installed| installed.plan.maximum_device_bytes),
                maximum_applications: self
                    .runtime
                    .krea_activation
                    .as_ref()
                    .map_or(0, |installed| installed.plan.maximum_applications),
            };
            let mut result = ProgramImageResultV6::default();
            // SAFETY: Forwarded from this method's caller contract.
            let status = unsafe {
                self.runtime.api.program_generate_image_v6(
                    arena,
                    &params,
                    condition_callback,
                    callback_pointer,
                    step_callback,
                    callback_pointer,
                    krea_event_callback,
                    activation_callback_pointer,
                    snapshot_handles,
                    &mut applications,
                    &mut transition_masks,
                    &mut captures,
                    &mut activation_applications,
                    &mut result,
                )
            };
            if result.abi_version != crate::KREA_ACTIVATION_ABI_VERSION
                || result.image.abi_version != MODEL_BLOCK_ABI_VERSION
                || result.image.image.abi_version != PROGRAM_ABI_VERSION
                || result.image.model_block_application_count != applications.len()
                || result.image.transition_words_per_operator != transition_words_per_operator
                || result.image.controls_cleared != 1
                || result.capture_count != captures.len()
                || result.operation_count != activation_applications.len()
                || result.activation_controls_cleared != 1
            {
                return Err(Error::Poisoned(
                    "native Krea activation evidence or cleanup attestation differs".to_owned(),
                ));
            }
            return Ok(NativeDiffusionInvocation {
                status,
                result: result.image.image,
                applications,
                transition_masks,
                transition_words_per_operator,
                krea_activation: Some(NativeKreaInvocation {
                    captures,
                    applications: activation_applications,
                    peak_host_bytes: result.peak_host_bytes,
                    peak_device_bytes: result.peak_device_bytes,
                }),
            });
        }
        let mut result = ProgramImageResultV5::default();
        // SAFETY: Forwarded from this method's caller contract.
        let status = unsafe {
            self.runtime.api.program_generate_image_v5(
                arena,
                &params,
                condition_callback,
                callback_pointer,
                step_callback,
                callback_pointer,
                snapshot_handles,
                &mut applications,
                &mut transition_masks,
                &mut result,
            )
        };
        if result.abi_version != MODEL_BLOCK_ABI_VERSION
            || result.image.abi_version != PROGRAM_ABI_VERSION
            || result.model_block_application_count != applications.len()
            || result.transition_words_per_operator != transition_words_per_operator
            || result.controls_cleared != 1
        {
            return Err(Error::Poisoned(
                "native model-block application or cleanup attestation differs".to_owned(),
            ));
        }
        Ok(NativeDiffusionInvocation {
            status,
            result: result.image,
            applications,
            transition_masks,
            transition_words_per_operator,
            krea_activation: None,
        })
    }

    fn diffusion_params(
        native: &ImageProgramNativeStageV1,
        prepared: &PreparedDiffusionCall,
    ) -> Result<ProgramImageParamsV3> {
        Ok(ProgramImageParamsV3 {
            abi_version: PROGRAM_ABI_VERSION,
            operation: native_operation(native.operation)?,
            positive_conditioning: prepared.positive,
            negative_conditioning: prepared.negative,
            width: i32::try_from(native.width)
                .map_err(|_| Error::Invalid("resident width exceeds i32".to_owned()))?,
            height: i32::try_from(native.height)
                .map_err(|_| Error::Invalid("resident height exceeds i32".to_owned()))?,
            output_format: prepared.output.format,
            maximum_output_bytes: prepared.output.maximum_bytes,
            seed: i64::try_from(prepared.seed)
                .map_err(|_| Error::Invalid("resident seed exceeds i64".to_owned()))?,
            cfg_scale: native.guidance_scale(),
            strength: native.strength(),
            sigmas: prepared.schedule.sigmas.as_ptr(),
            sigma_count: prepared.schedule.sigmas.len(),
            init_image: prepared.init_image,
            mask_image: prepared.mask_image,
            reference_images: pointer_or_null(&prepared.reference_images),
            reference_image_count: prepared.reference_images.len(),
            loras: pointer_or_null(&prepared.lora_schedules),
            lora_count: prepared.lora_schedules.len(),
            checkpoint_after_step: native.checkpoint_after_step.unwrap_or(u32::MAX),
            snapshot_after_steps: pointer_or_null(&prepared.snapshot_steps),
            snapshot_count: prepared.snapshot_steps.len(),
        })
    }

    fn publish_diffusion_outputs(
        &mut self,
        native: &ImageProgramNativeStageV1,
        completed: CompletedDiffusionCall<'_>,
    ) -> Result<NativeStageOutcome> {
        let CompletedDiffusionCall {
            result,
            model_block_applications,
            snapshot_handles,
            stage_program,
            actual_plan,
            native_time_ns,
        } = completed;
        let mut outputs = Vec::with_capacity(native.outputs.len());
        self.publish_primary_and_checkpoint(native, &result, &actual_plan, &mut outputs)?;
        let snapshot_contents = self.publish_snapshots(native, snapshot_handles, &mut outputs)?;
        if outputs.len() != native.outputs.len() {
            return Err(Error::Incompatible(
                "resident diffusion returned fewer outputs than declared".to_owned(),
            ));
        }
        let observations = stage_program.observation_receipts(&snapshot_contents)?;
        Ok(NativeStageOutcome::Completed {
            outputs,
            observations,
            model_block_applications,
            native_time_ns,
        })
    }

    fn publish_primary_and_checkpoint(
        &mut self,
        native: &ImageProgramNativeStageV1,
        result: &ProgramImageResultV3,
        actual_plan: &DiffusionPlan,
        outputs: &mut Vec<u16>,
    ) -> Result<()> {
        let primary = native.outputs.first().ok_or_else(|| {
            Error::Invalid("resident diffusion primary output is missing".to_owned())
        })?;
        self.insert_handle(primary.value, result.primary)?;
        outputs.push(primary.value);
        let Some(step) = native.checkpoint_after_step else {
            return Ok(());
        };
        let output = native.outputs.get(1).ok_or_else(|| {
            Error::Invalid("resident checkpoint-state output is missing".to_owned())
        })?;
        if output.role != ImageProgramNativeOutputRoleV1::CheckpointState {
            return Err(Error::Incompatible(
                "resident checkpoint-state output order differs".to_owned(),
            ));
        }
        self.insert_handle(output.value, result.checkpoint_state)?;
        self.captured_states.insert(
            output.value,
            CapturedState {
                plan: actual_plan.clone(),
                step,
            },
        );
        outputs.push(output.value);
        Ok(())
    }

    fn publish_snapshots(
        &mut self,
        native: &ImageProgramNativeStageV1,
        snapshot_handles: Vec<ValueHandleV3>,
        outputs: &mut Vec<u16>,
    ) -> Result<Vec<Digest>> {
        let observation_indices = native
            .observations
            .iter()
            .enumerate()
            .filter(|(_, observation)| observation.kind == ObservationKind::Snapshot)
            .map(|(index, _)| {
                u16::try_from(index)
                    .map_err(|_| Error::Invalid("observation index exceeds u16".to_owned()))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut contents = Vec::with_capacity(snapshot_handles.len());
        for (handle, observation) in snapshot_handles.into_iter().zip(observation_indices) {
            if handle.is_empty() {
                return Err(Error::Incompatible(
                    "resident snapshot handle is empty".to_owned(),
                ));
            }
            let output = native
                .outputs
                .get(outputs.len())
                .ok_or_else(|| Error::Invalid("resident snapshot output is missing".to_owned()))?;
            if output.role != (ImageProgramNativeOutputRoleV1::Observation { observation }) {
                return Err(Error::Incompatible(
                    "resident snapshot output order differs".to_owned(),
                ));
            }
            self.insert_handle(output.value, handle)?;
            contents.push(self.content_digest(handle)?);
            outputs.push(output.value);
        }
        Ok(contents)
    }

    fn receipts_for_values(
        &self,
        plan: &ImageProgramPlanV1,
        values: &[u16],
    ) -> Result<(
        Vec<ImageProgramValueReceiptV1>,
        Vec<ImageProgramValueMeasurementV1>,
    )> {
        let mut receipts = Vec::with_capacity(values.len());
        let mut measurements = Vec::with_capacity(values.len());
        for value in values {
            let handle = self.handle(*value)?;
            let descriptor = self.descriptor(handle)?;
            validate_descriptor(&plan.values[usize::from(*value)].spec, &descriptor)?;
            receipts.push(ImageProgramValueReceiptV1 {
                value: *value,
                content: self.content_digest(handle)?,
                bytes: descriptor.bytes,
            });
            measurements.push(measurement_from_descriptor(*value, &descriptor)?);
        }
        Ok((receipts, measurements))
    }

    fn measurement(
        &self,
        plan: &ImageProgramPlanV1,
        value: u16,
        handle: ValueHandleV3,
    ) -> Result<ImageProgramValueMeasurementV1> {
        let descriptor = self.descriptor(handle)?;
        validate_descriptor(&plan.values[usize::from(value)].spec, &descriptor)?;
        measurement_from_descriptor(value, &descriptor)
    }

    fn descriptor(&self, handle: ValueHandleV3) -> Result<ValueDescriptorV3> {
        let mut descriptor = ValueDescriptorV3::default();
        // SAFETY: The handle and arena are live for this synchronous query.
        let status = unsafe {
            self.runtime
                .api
                .program_describe_v3(self.arena()?.as_ptr(), handle, &mut descriptor)
        };
        if status != ffi::STATUS_OK {
            return Err(Error::Native(format!(
                "resident value description failed: {}",
                native_status_error(status)
            )));
        }
        if descriptor.abi_version != PROGRAM_ABI_VERSION {
            return Err(Error::Incompatible(
                "resident value descriptor ABI differs".to_owned(),
            ));
        }
        Ok(descriptor)
    }

    fn content_digest(&self, handle: ValueHandleV3) -> Result<Digest> {
        let descriptor = self.descriptor(handle)?;
        let mut state = NativeHashState::new(descriptor.bytes)?;
        // SAFETY: The native callback is synchronous and never retains
        // `state`; it validates the single exposed byte span before hashing.
        let status = unsafe {
            self.runtime.api.program_read_v3(
                self.arena()?.as_ptr(),
                handle,
                hash_native_value,
                (&raw mut state).cast::<c_void>(),
            )
        };
        if status != ffi::STATUS_OK || state.failed || state.seen != descriptor.bytes {
            return Err(Error::Incompatible(
                "resident value hashing did not observe its exact descriptor length".to_owned(),
            ));
        }
        Digest::from_str(state.hasher.finalize().to_hex().as_ref())
            .map_err(logit_loom_diffusion::Error::from)
            .map_err(Into::into)
    }

    fn copy_handle(&self, handle: ValueHandleV3) -> Result<Vec<u8>> {
        let descriptor = self.descriptor(handle)?;
        let length = usize::try_from(descriptor.bytes)
            .map_err(|_| Error::Invalid("resident value exceeds usize".to_owned()))?;
        let mut bytes = vec![0_u8; length];
        let written = self.copy_handle_into(handle, &mut bytes)?;
        if written != length {
            return Err(Error::Incompatible(
                "resident value materialization length differs".to_owned(),
            ));
        }
        Ok(bytes)
    }

    fn copy_handle_into(&self, handle: ValueHandleV3, output: &mut [u8]) -> Result<usize> {
        let descriptor = self.descriptor(handle)?;
        let length = usize::try_from(descriptor.bytes)
            .map_err(|_| Error::Invalid("resident value exceeds usize".to_owned()))?;
        if output.len() < length {
            return Err(Error::Output(
                "resident caller-owned output allocation is undersized".to_owned(),
            ));
        }
        let mut written = 0;
        // SAFETY: The exact-sized writable allocation and live handle remain
        // valid for this synchronous copy.
        let status = unsafe {
            self.runtime.api.program_copy_v3(
                self.arena()?.as_ptr(),
                handle,
                &mut output[..length],
                &mut written,
            )
        };
        if status != ffi::STATUS_OK || written != length {
            return Err(Error::Incompatible(
                "resident value materialization length differs".to_owned(),
            ));
        }
        Ok(written)
    }

    fn require_import_status(status: i32, value: u16) -> Result<()> {
        if status == ffi::STATUS_OK {
            Ok(())
        } else {
            Err(Error::Native(format!(
                "native import of resident value {value} failed: {}",
                native_status_error(status)
            )))
        }
    }

    fn require_stage_status(status: i32, operation: &str) -> Result<()> {
        match status {
            ffi::STATUS_OK => Ok(()),
            ffi::STATUS_INVALID_ARGUMENT => Err(Error::Invalid(format!(
                "{operation} arguments differ from the preflighted plan"
            ))),
            ffi::STATUS_UNSUPPORTED => Err(Error::Incompatible(format!(
                "{operation} could not realize the declared mechanic exactly"
            ))),
            other => Err(native_status_error(other)),
        }
    }
}

impl<R> Drop for SdcppResidentProgram<'_, R> {
    fn drop(&mut self) {
        if let Some(arena) = self.arena.take() {
            let mut ignored_peak = 0;
            // SAFETY: Dropping an unfinished backend consumes its sole arena
            // allocation with best-effort clearing before poisoning reuse.
            let _ = unsafe {
                self.runtime
                    .api
                    .program_finish_v3(arena.as_ptr(), true, &mut ignored_peak)
            };
            self.runtime.krea_activation.take();
            self.runtime.state = ExecutorState::Poisoned;
        }
    }
}

struct StageExecution {
    outputs: Vec<u16>,
    observations: Vec<Digest>,
    model_block_applications: Vec<ModelBlockApplicationV1>,
    native_time_ns: Option<u64>,
}

impl StageExecution {
    fn without_observations(outputs: Vec<u16>) -> Self {
        Self {
            outputs,
            observations: Vec::new(),
            model_block_applications: Vec::new(),
            native_time_ns: None,
        }
    }
}

enum StageOperationOutcome {
    Completed(StageExecution),
    Cancelled { step: u32 },
}

struct PreparedDiffusionCall {
    prompt: String,
    positive: ValueHandleV3,
    negative: ValueHandleV3,
    init_image: ValueHandleV3,
    mask_image: ValueHandleV3,
    reference_images: Vec<ValueHandleV3>,
    _scale_points: Vec<Vec<LoraScalePointV3>>,
    lora_schedules: Vec<LoraScheduleV3>,
    _model_block_steps: Vec<Vec<u32>>,
    model_block_operators: Vec<ModelBlockOperatorV5>,
    snapshot_steps: Vec<u32>,
    seed: u64,
    schedule: logit_loom_diffusion::DiffusionSchedule,
    restore: Option<DiffusionCheckpoint>,
    output: ProgramOutputV3,
}

struct LoweredModelBlockOperators {
    steps: Vec<Vec<u32>>,
    operators: Vec<ModelBlockOperatorV5>,
}

fn lower_model_block_operators(
    operators: &[logit_loom_diffusion::OperatorInvocation],
) -> Result<LoweredModelBlockOperators> {
    let steps = operators
        .iter()
        .filter_map(|operator| match &operator.selector {
            TensorSelector::ModelBlock { .. } => Some(match &operator.steps {
                StepSelector::All => Vec::new(),
                StepSelector::Exact { steps } => steps.clone(),
            }),
            _ => None,
        })
        .collect::<Vec<_>>();
    let native = operators
        .iter()
        .enumerate()
        .filter(|(_, operator)| matches!(operator.selector, TensorSelector::ModelBlock { .. }))
        .zip(&steps)
        .map(|((operator_index, operator), selected_steps)| {
            let installed = InstalledModelBlockResidualScale::from_invocation(operator)?;
            Ok(ModelBlockOperatorV5 {
                operator_index: u32::try_from(operator_index).map_err(|_| {
                    Error::Invalid("resident model-block operator index exceeds u32".to_owned())
                })?,
                component: ffi::MODEL_COMPONENT_KREA2_V5,
                block: installed.block,
                site: ffi::MODEL_BLOCK_RESIDUAL_V5,
                residual_scale: installed.scale,
                step_selection: if matches!(installed.steps, StepSelector::All) {
                    ffi::STEP_ALL_V5
                } else {
                    ffi::STEP_EXACT_V5
                },
                steps: pointer_or_null(selected_steps),
                step_count: selected_steps.len(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(LoweredModelBlockOperators {
        steps,
        operators: native,
    })
}

struct NativeDiffusionInvocation {
    status: i32,
    result: ProgramImageResultV3,
    applications: Vec<NativeModelBlockApplicationV5>,
    transition_masks: Vec<u64>,
    transition_words_per_operator: usize,
    krea_activation: Option<NativeKreaInvocation>,
}

struct NativeKreaInvocation {
    captures: Vec<KreaCaptureResultV6>,
    applications: Vec<KreaApplicationResultV6>,
    peak_host_bytes: u64,
    peak_device_bytes: u64,
}

fn verified_model_block_applications(
    stage: u16,
    prepared: &PreparedDiffusionCall,
    invocation: &NativeDiffusionInvocation,
) -> Result<Vec<ModelBlockApplicationV1>> {
    if prepared.model_block_operators.is_empty() {
        if !invocation.applications.is_empty()
            || !invocation.transition_masks.is_empty()
            || invocation.transition_words_per_operator != 0
        {
            return Err(Error::Poisoned(
                "native model-block evidence appeared without an operator".to_owned(),
            ));
        }
        return Ok(Vec::new());
    }
    if invocation.applications.len() != prepared.model_block_operators.len()
        || invocation.transition_masks.len()
            != invocation
                .applications
                .len()
                .checked_mul(invocation.transition_words_per_operator)
                .ok_or_else(|| {
                    Error::Poisoned("native model-block evidence length overflowed".to_owned())
                })?
    {
        return Err(Error::Poisoned(
            "native model-block evidence buffer accounting differs".to_owned(),
        ));
    }

    invocation
        .applications
        .iter()
        .zip(&prepared.model_block_operators)
        .enumerate()
        .map(|(application_index, (application, operator))| {
            if application.operator_index != operator.operator_index
                || application.block != operator.block
                || application.residual_scale.to_bits() != operator.residual_scale.to_bits()
                || application.loaded_model_blocks == 0
                || application.block >= application.loaded_model_blocks
            {
                return Err(Error::Poisoned(
                    "native model-block evidence does not echo the exact operator".to_owned(),
                ));
            }
            let start = application_index
                .checked_mul(invocation.transition_words_per_operator)
                .ok_or_else(|| {
                    Error::Poisoned("native model-block transition offset overflowed".to_owned())
                })?;
            let end = start
                .checked_add(invocation.transition_words_per_operator)
                .ok_or_else(|| {
                    Error::Poisoned("native model-block transition extent overflowed".to_owned())
                })?;
            Ok(ModelBlockApplicationV1 {
                stage,
                operator: u16::try_from(operator.operator_index).map_err(|_| {
                    Error::Poisoned("native model-block operator exceeds u16".to_owned())
                })?,
                loaded_model_blocks: application.loaded_model_blocks,
                block: application.block,
                residual_scale_bits: application.residual_scale.to_bits(),
                selected_transitions: invocation.transition_masks[start..end].to_vec(),
                graph_applications: application.graph_applications,
                ordinary_graphs: application.ordinary_graphs,
                bypassed_graphs: application.bypassed_graphs,
                scaled_residual_graphs: application.scaled_residual_graphs,
            })
        })
        .collect()
}

struct CompletedDiffusionCall<'a> {
    result: ProgramImageResultV3,
    model_block_applications: Vec<ModelBlockApplicationV1>,
    snapshot_handles: Vec<ValueHandleV3>,
    stage_program: ResidentStageProgram<'a>,
    actual_plan: DiffusionPlan,
    native_time_ns: Option<u64>,
}

enum DiffusionCallOutcome<'a> {
    Completed(Box<CompletedDiffusionCall<'a>>),
    Cancelled { step: u32 },
}

enum NativeStageOutcome {
    Completed {
        outputs: Vec<u16>,
        observations: Vec<Digest>,
        model_block_applications: Vec<ModelBlockApplicationV1>,
        native_time_ns: Option<u64>,
    },
    Cancelled {
        step: u32,
    },
}

fn resident_backend_identity(runtime: &Sdcpp) -> Result<Digest> {
    let bindings = runtime.execution_bindings()?;
    Digest::of_serializable(
        "sdcpp-resident-program-backend-v1",
        &(
            PROGRAM_ABI_VERSION,
            ADAPTER_CONTRACT_VERSION,
            UPSTREAM_COMMIT,
            bindings,
            runtime.native_receipt(),
        ),
    )
    .map_err(logit_loom_diffusion::Error::from)
    .map_err(Into::into)
}

fn validate_native_stage(
    runtime: &Sdcpp,
    program: &ImageProgramPlanV1,
    stage: &ImageProgramNativeStageV1,
    bindings: &crate::ImageExecutionBindings,
) -> Result<()> {
    if stage.profile != bindings.profile
        || stage.load != bindings.load
        || stage.rng != bindings.rng
        || stage.placement != bindings.placement
    {
        return Err(Error::Incompatible(
            "resident native stage bindings differ from the loaded runtime".to_owned(),
        ));
    }
    runtime
        .profile
        .validate_dimensions(stage.width, stage.height)?;
    fixed_seed(&stage.seed)?;
    if stage.operation != ImageOperation::VaeEncode {
        native_output_contract(program, stage)?;
    }
    if matches!(
        stage.operation,
        ImageOperation::VaeEncode | ImageOperation::VaeDecode
    ) && runtime.profile != crate::Profile::Krea2Turbo
    {
        return Err(Error::Invalid(
            "direct resident VAE operations require the Krea profile".to_owned(),
        ));
    }
    for input in &stage.inputs {
        if matches!(
            input.role,
            ImageBufferRole::PositiveConditioning | ImageBufferRole::NegativeConditioning
        ) && !matches!(
            program.values[usize::from(input.value)].spec,
            ImageProgramValueSpecV1::Utf8 { .. }
        ) {
            return Err(Error::Invalid(
                "resident conditioning currently requires exact UTF-8 values".to_owned(),
            ));
        }
    }
    for operator in &stage.operators {
        match &operator.selector {
            TensorSelector::SchedulerState => {}
            TensorSelector::ModelBlock { .. } if runtime.profile == crate::Profile::Krea2Turbo => {
                InstalledModelBlockResidualScale::from_invocation(operator)?;
            }
            TensorSelector::ModelBlock { .. } => {
                return Err(Error::Invalid(
                    "resident model-block operators require the Krea profile".to_owned(),
                ));
            }
            TensorSelector::Conditioning { .. } => {
                return Err(Error::Invalid(
                    "resident conditioning operators are not installed".to_owned(),
                ));
            }
        }
    }
    for observation in &stage.observations {
        if observation.selector != TensorSelector::SchedulerState {
            return Err(Error::Invalid(
                "resident model-block and conditioning observations are not installed".to_owned(),
            ));
        }
    }
    for lora in &stage.loras {
        validate_lora_target(lora)?;
    }
    Ok(())
}

fn validate_lora_target(lora: &ImageProgramLoraV1) -> Result<()> {
    let scheduled = lora.scales.points.len() > 1;
    let fixed_target = lora.target == lora_target_v1(false) || lora.target == lora_target_v1(true);
    let resident_target = lora.target == resident_lora_target_v1(false)
        || lora.target == resident_lora_target_v1(true);
    if !(resident_target || fixed_target && !scheduled) {
        return Err(Error::Invalid(
            "resident LoRA target or schedule identity is unsupported".to_owned(),
        ));
    }
    if lora
        .scales
        .points
        .iter()
        .any(|point| point.scale().abs() > 64.0)
    {
        return Err(Error::Invalid(
            "resident LoRA scale exceeds the native finite bound".to_owned(),
        ));
    }
    Ok(())
}

fn lora_high_noise(plan: &ImageProgramPlanV1, value: u16) -> Result<bool> {
    let mut selected = None;
    for stage in &plan.stages {
        let ImageProgramStageOperationV1::Native { plan: native } = &stage.operation else {
            continue;
        };
        for lora in native.loras.iter().filter(|lora| lora.value == value) {
            validate_lora_target(lora)?;
            let high_noise =
                lora.target == lora_target_v1(true) || lora.target == resident_lora_target_v1(true);
            if selected
                .replace(high_noise)
                .is_some_and(|prior| prior != high_noise)
            {
                return Err(Error::Invalid(format!(
                    "resident LoRA value {value} has conflicting target identities"
                )));
            }
        }
    }
    selected.ok_or_else(|| {
        Error::Invalid(format!(
            "resident LoRA value {value} has no native-stage consumer"
        ))
    })
}

fn fixed_seed(seed: &SeedSelection) -> Result<u64> {
    match seed {
        SeedSelection::Fixed { seed } if i64::try_from(*seed).is_ok() => Ok(*seed),
        SeedSelection::Fixed { .. } => Err(Error::Invalid(
            "resident fixed seed exceeds the native i64 range".to_owned(),
        )),
        SeedSelection::WorkerSelected { .. } => Err(Error::Invalid(
            "resident adapter requires the coordinator to resolve a fixed seed".to_owned(),
        )),
    }
}

fn native_operation(operation: ImageOperation) -> Result<i32> {
    match operation {
        ImageOperation::TextToImage => Ok(ffi::OPERATION_TEXT_TO_IMAGE),
        ImageOperation::ImageToImage => Ok(ffi::OPERATION_IMAGE_TO_IMAGE),
        ImageOperation::Inpaint => Ok(ffi::OPERATION_INPAINT),
        ImageOperation::Outpaint => Ok(ffi::OPERATION_OUTPAINT),
        ImageOperation::VaeEncode | ImageOperation::VaeDecode => Err(Error::Invalid(
            "direct VAE operation passed resident diffusion lowering".to_owned(),
        )),
    }
}

fn native_output_contract(
    program: &ImageProgramPlanV1,
    stage: &ImageProgramNativeStageV1,
) -> Result<ProgramOutputV3> {
    let output = stage
        .outputs
        .first()
        .ok_or_else(|| Error::Invalid("resident primary output is missing".to_owned()))?;
    let specification = program
        .values
        .get(usize::from(output.value))
        .ok_or_else(|| Error::Invalid("resident primary output value is absent".to_owned()))?;
    let maximum_bytes = specification
        .spec
        .maximum_bytes()
        .map_err(logit_loom_diffusion::Error::from)?;
    let output_format = match (&stage.output_format, &specification.spec) {
        (ImageOutputFormat::Rgb8, ImageProgramValueSpecV1::Rgb8 { .. }) => ffi::PROGRAM_RGB8_V3,
        (ImageOutputFormat::Rgba8, ImageProgramValueSpecV1::Rgba8 { .. }) => ffi::PROGRAM_RGBA8_V3,
        (
            ImageOutputFormat::Png,
            ImageProgramValueSpecV1::Png {
                color, encoding, ..
            },
        ) => {
            if encoding != &resident_png_encoding_v1(*color)? {
                return Err(Error::Incompatible(
                    "resident PNG encoder identity differs".to_owned(),
                ));
            }
            if maximum_bytes < resident_png_maximum_bytes_v1(stage.width, stage.height, *color)? {
                return Err(Error::Invalid(
                    "resident PNG allocation is below the encoder upper bound".to_owned(),
                ));
            }
            match color {
                ImagePngColorV1::Rgb8 => ffi::PROGRAM_PNG_RGB8_V3,
                ImagePngColorV1::Rgba8 => ffi::PROGRAM_PNG_RGBA8_V3,
            }
        }
        _ => {
            return Err(Error::Invalid(
                "resident native output format differs from its logical value".to_owned(),
            ));
        }
    };
    Ok(ProgramOutputV3 {
        width: stage.width,
        height: stage.height,
        format: output_format,
        maximum_bytes,
    })
}

fn uses_diffusion(operation: ImageOperation) -> bool {
    matches!(
        operation,
        ImageOperation::TextToImage
            | ImageOperation::ImageToImage
            | ImageOperation::Inpaint
            | ImageOperation::Outpaint
    )
}

fn value_for_role(stage: &ImageProgramNativeStageV1, role: ImageBufferRole) -> Result<Option<u16>> {
    let mut values = stage
        .inputs
        .iter()
        .filter(|input| input.role == role)
        .map(|input| input.value);
    let value = values.next();
    if values.next().is_some() {
        return Err(Error::Invalid(format!(
            "resident native stage repeats {role:?}"
        )));
    }
    Ok(value)
}

fn f32_from_le_bytes(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return Err(Error::Invalid(
            "resident f32 bytes are empty or incomplete".to_owned(),
        ));
    }
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if value.is_finite() {
                Ok(value)
            } else {
                Err(Error::Invalid(
                    "resident tensor contains a non-finite f32".to_owned(),
                ))
            }
        })
        .collect()
}

fn f32_from_native_bytes(bytes: &[u8]) -> Result<Vec<f32>> {
    if !cfg!(target_endian = "little") {
        return Err(Error::Incompatible(
            "native checkpoint bytes require a little-endian host".to_owned(),
        ));
    }
    f32_from_le_bytes(bytes)
}

fn i64_shape(shape: &[u64]) -> Result<Vec<i64>> {
    shape
        .iter()
        .map(|dimension| {
            i64::try_from(*dimension)
                .map_err(|_| Error::Invalid("resident tensor dimension exceeds i64".to_owned()))
        })
        .collect()
}

fn pointer_or_null<T>(values: &[T]) -> *const T {
    if values.is_empty() {
        std::ptr::null()
    } else {
        values.as_ptr()
    }
}

fn validate_descriptor(
    specification: &ImageProgramValueSpecV1,
    descriptor: &ValueDescriptorV3,
) -> Result<()> {
    if descriptor.bytes == 0
        || descriptor.bytes
            > specification
                .maximum_bytes()
                .map_err(logit_loom_diffusion::Error::from)?
    {
        return Err(Error::Incompatible(
            "native value byte length exceeds its logical specification".to_owned(),
        ));
    }
    let valid = match specification {
        ImageProgramValueSpecV1::Utf8 { .. } | ImageProgramValueSpecV1::Checkpoint { .. } => {
            descriptor.kind == ffi::VALUE_BYTES_V3
        }
        ImageProgramValueSpecV1::Rgb8 { width, height } => {
            descriptor.kind == ffi::VALUE_IMAGE_V3
                && descriptor.width == *width
                && descriptor.height == *height
                && descriptor.channels == 3
                && descriptor.bytes == u64::from(*width) * u64::from(*height) * 3
        }
        ImageProgramValueSpecV1::Rgba8 { width, height } => {
            descriptor.kind == ffi::VALUE_IMAGE_V3
                && descriptor.width == *width
                && descriptor.height == *height
                && descriptor.channels == 4
                && descriptor.bytes == u64::from(*width) * u64::from(*height) * 4
        }
        ImageProgramValueSpecV1::Png {
            width,
            height,
            color,
            ..
        } => {
            descriptor.kind == ffi::VALUE_PNG_V3
                && descriptor.width == *width
                && descriptor.height == *height
                && descriptor.channels == color.channels()
        }
        ImageProgramValueSpecV1::Gray8 { width, height } => {
            descriptor.kind == ffi::VALUE_IMAGE_V3
                && descriptor.width == *width
                && descriptor.height == *height
                && descriptor.channels == 1
                && descriptor.bytes == u64::from(*width) * u64::from(*height)
        }
        ImageProgramValueSpecV1::Tensor { tensor, .. } => {
            let rank = usize::try_from(descriptor.rank).ok();
            descriptor.kind == ffi::VALUE_TENSOR_V3
                && descriptor.dtype == ffi::TENSOR_F32
                && descriptor.element_count
                    == tensor
                        .elements()
                        .map_err(logit_loom_diffusion::Error::from)?
                && rank == Some(tensor.shape.len())
                && rank.is_some_and(|rank| {
                    descriptor.shape[..rank]
                        .iter()
                        .zip(&tensor.shape)
                        .all(|(native, expected)| u64::try_from(*native) == Ok(*expected))
                })
        }
        ImageProgramValueSpecV1::Opaque {
            opaque_kind: ImageOpaqueValueKindV1::LoraArtifact,
            ..
        } => descriptor.kind == ffi::VALUE_LORA_V3,
        ImageProgramValueSpecV1::Opaque {
            opaque_kind: ImageOpaqueValueKindV1::CheckpointState,
            ..
        } => {
            descriptor.kind == ffi::VALUE_CHECKPOINT_STATE_V3
                && descriptor.dtype == ffi::TENSOR_F32
                && descriptor.rank > 0
                && descriptor.rank <= 8
                && descriptor.element_count > 0
        }
        ImageProgramValueSpecV1::Opaque {
            opaque_kind: ImageOpaqueValueKindV1::Conditioning,
            ..
        } => false,
    };
    if valid {
        Ok(())
    } else {
        Err(Error::Incompatible(
            "native value descriptor differs from its logical type".to_owned(),
        ))
    }
}

fn measurement_from_descriptor(
    value: u16,
    descriptor: &ValueDescriptorV3,
) -> Result<ImageProgramValueMeasurementV1> {
    let placement = match descriptor.placement {
        ffi::VALUE_HOST_V3 => ImageProgramValuePlacementV1::Host,
        ffi::VALUE_MIXED_V3 => {
            return Err(Error::Incompatible(
                "native mixed placement omitted exact device identities".to_owned(),
            ));
        }
        other => {
            return Err(Error::Incompatible(format!(
                "native value reports unknown placement {other}"
            )));
        }
    };
    Ok(ImageProgramValueMeasurementV1 {
        value,
        placement,
        host_to_device_transfers: descriptor.host_to_device_transfers,
        host_to_device_bytes: descriptor.host_to_device_bytes,
        device_to_host_transfers: descriptor.device_to_host_transfers,
        device_to_host_bytes: descriptor.device_to_host_bytes,
    })
}

struct NativeHashState {
    hasher: blake3::Hasher,
    expected: u64,
    seen: u64,
    failed: bool,
}

impl NativeHashState {
    fn new(expected: u64) -> Result<Self> {
        let domain = b"image-program-value-content-v1";
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"logit-loom\0");
        hasher.update(
            &u64::try_from(domain.len())
                .map_err(|_| Error::Invalid("digest domain exceeds u64".to_owned()))?
                .to_le_bytes(),
        );
        hasher.update(domain);
        hasher.update(&expected.to_le_bytes());
        Ok(Self {
            hasher,
            expected,
            seen: 0,
            failed: false,
        })
    }
}

unsafe extern "C" fn hash_native_value(
    bytes: *const u8,
    byte_count: usize,
    data: *mut c_void,
) -> i32 {
    if bytes.is_null() || data.is_null() || byte_count == 0 {
        return ffi::CALLBACK_ERROR;
    }
    // SAFETY: The companion calls synchronously with one live state and exact
    // readable byte span. The callback never retains either pointer.
    let state = unsafe { &mut *data.cast::<NativeHashState>() };
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let count = u64::try_from(byte_count).map_err(|_| ())?;
        let next = state.seen.checked_add(count).ok_or(())?;
        if next > state.expected {
            return Err(());
        }
        // SAFETY: The companion's read ABI promises `byte_count` readable
        // bytes for the duration of this callback.
        let value = unsafe { std::slice::from_raw_parts(bytes, byte_count) };
        state.hasher.update(value);
        state.seen = next;
        Ok(())
    }));
    if let Ok(Ok(())) = outcome {
        ffi::CALLBACK_CONTINUE
    } else {
        state.failed = true;
        ffi::CALLBACK_ERROR
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ModelBlockResidualScaleControlV1, model_block_residual_scale_schema_v1};
    use logit_loom_diffusion::OperatorInvocation;

    #[test]
    fn incremental_value_hash_matches_public_content_domain() {
        let bytes = b"resident-value";
        let mut state = NativeHashState::new(u64::try_from(bytes.len()).unwrap()).unwrap();
        // SAFETY: Test pointers are live and exact for the synchronous call.
        let result = unsafe {
            hash_native_value(
                bytes.as_ptr(),
                bytes.len(),
                (&raw mut state).cast::<c_void>(),
            )
        };
        assert_eq!(result, ffi::CALLBACK_CONTINUE);
        let actual = Digest::from_str(state.hasher.finalize().to_hex().as_ref()).unwrap();
        assert_eq!(
            actual,
            logit_loom_diffusion::image_program_value_content(bytes)
        );
    }

    #[test]
    fn scheduled_and_fixed_lora_targets_remain_distinct() {
        assert_ne!(resident_lora_target_v1(false), lora_target_v1(false));
        assert_ne!(
            resident_lora_target_v1(false),
            resident_lora_target_v1(true)
        );
    }

    #[test]
    fn model_block_lowering_retains_exact_step_storage() {
        let all_control = ModelBlockResidualScaleControlV1::new(0.0, 1.0).unwrap();
        let all_selector = TensorSelector::ModelBlock {
            component: "krea2".to_owned(),
            block: 9,
            site: "residual".to_owned(),
        };
        let all_steps = StepSelector::All;
        let exact_control = ModelBlockResidualScaleControlV1::new(0.5, 1.0).unwrap();
        let exact_selector = TensorSelector::ModelBlock {
            component: "krea2".to_owned(),
            block: 10,
            site: "residual".to_owned(),
        };
        let exact_steps = StepSelector::Exact { steps: vec![1, 3] };
        let invocations = vec![
            OperatorInvocation {
                schema: model_block_residual_scale_schema_v1(),
                implementation: all_control
                    .implementation_for(&all_selector, &all_steps)
                    .unwrap(),
                selector: all_selector,
                steps: all_steps,
                controls: all_control.to_control_bytes(),
            },
            OperatorInvocation {
                schema: model_block_residual_scale_schema_v1(),
                implementation: exact_control
                    .implementation_for(&exact_selector, &exact_steps)
                    .unwrap(),
                selector: exact_selector,
                steps: exact_steps,
                controls: exact_control.to_control_bytes(),
            },
        ];

        let lowered = lower_model_block_operators(&invocations).unwrap();
        assert_eq!(lowered.steps, [Vec::new(), vec![1, 3]]);
        assert_eq!(lowered.operators.len(), 2);
        assert_eq!(lowered.operators[0].block, 9);
        assert_eq!(lowered.operators[0].step_selection, ffi::STEP_ALL_V5);
        assert!(lowered.operators[0].steps.is_null());
        assert_eq!(lowered.operators[1].block, 10);
        assert_eq!(
            lowered.operators[1].residual_scale.to_bits(),
            0.5_f32.to_bits()
        );
        assert_eq!(lowered.operators[1].step_selection, ffi::STEP_EXACT_V5);
        assert_eq!(lowered.operators[1].step_count, 2);
        assert_eq!(lowered.operators[1].steps, lowered.steps[1].as_ptr());
    }

    #[test]
    fn png_output_lowering_binds_color_encoder_and_maximum() {
        assert_eq!(
            resident_png_maximum_bytes_v1(2, 1, ImagePngColorV1::Rgba8).unwrap(),
            77
        );
        for (color, expected_format) in [
            (ImagePngColorV1::Rgb8, ffi::PROGRAM_PNG_RGB8_V3),
            (ImagePngColorV1::Rgba8, ffi::PROGRAM_PNG_RGBA8_V3),
        ] {
            let encoding = resident_png_encoding_v1(color).unwrap();
            let program = ImageProgramPlanV1 {
                values: vec![logit_loom_diffusion::ImageProgramValueV1 {
                    value: 0,
                    spec: ImageProgramValueSpecV1::Png {
                        width: 2,
                        height: 1,
                        color,
                        encoding,
                        maximum_bytes: 1_024,
                    },
                }],
                inputs: Vec::new(),
                stages: Vec::new(),
                outputs: Vec::new(),
                cleanup: ImageCleanupPolicy::ClearSession,
            };
            let stage = ImageProgramNativeStageV1 {
                profile: Digest::of_bytes("profile", b"profile"),
                load: Digest::of_bytes("load", b"load"),
                operation: ImageOperation::TextToImage,
                width: 2,
                height: 1,
                output_format: ImageOutputFormat::Png,
                seed: SeedSelection::Fixed { seed: 7 },
                rng: Digest::of_bytes("rng", b"rng"),
                placement: Digest::of_bytes("placement", b"placement"),
                schedule: None,
                guidance_scale_bits: 1.0_f32.to_bits(),
                strength_bits: 1.0_f32.to_bits(),
                inputs: Vec::new(),
                loras: Vec::new(),
                operators: Vec::new(),
                observations: Vec::new(),
                checkpoint_restore_at_step: None,
                checkpoint_after_step: None,
                outputs: vec![logit_loom_diffusion::ImageProgramNativeOutputV1 {
                    role: ImageProgramNativeOutputRoleV1::Primary,
                    value: 0,
                }],
            };
            assert_eq!(
                native_output_contract(&program, &stage).unwrap(),
                ProgramOutputV3 {
                    width: 2,
                    height: 1,
                    format: expected_format,
                    maximum_bytes: 1_024,
                }
            );
        }
    }
}

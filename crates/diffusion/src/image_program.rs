// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded resident image-program contracts, receipts, and measurements.

use std::collections::{HashMap, HashSet};

use logit_loom_core::{CoreError, Digest};
use logit_loom_executor::BufferSpec;
use serde::{Deserialize, Serialize};

use crate::{
    DiffusionSchedule, ImageBufferRole, ImageCleanupPolicy, ImageOperation, ImageOutputFormat,
    MAX_DEVICE_LABEL_BYTES, MAX_IMAGE_DIMENSION, MAX_IMAGE_LORAS, MAX_IMAGE_OBSERVATIONS,
    MAX_IMAGE_OPERATORS, ObservationKind, ObservationRequest, OperatorInvocation, ScaleSchedule,
    SeedSelection, StepSelector, TensorDType, TensorSpec,
};

/// Maximum ordered stages in one resident image program.
pub const MAX_IMAGE_PROGRAM_STAGES: usize = 64;
/// Maximum typed values in one resident image program.
pub const MAX_IMAGE_PROGRAM_VALUES: usize = 128;
/// Maximum external inputs in one resident image program.
pub const MAX_IMAGE_PROGRAM_INPUTS: usize = 32;
/// Maximum caller-owned output routes in one resident image program.
pub const MAX_IMAGE_PROGRAM_OUTPUTS: usize = 32;
/// Maximum declared size of one logical program value.
pub const MAX_IMAGE_PROGRAM_VALUE_BYTES: u64 = 1024 * 1024 * 1024;
/// Maximum liveness-derived resident value-arena size.
pub const MAX_IMAGE_PROGRAM_ARENA_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Returns the canonical content identity for one produced program value.
///
/// External input identities remain caller-supplied [`BufferSpec`] identities.
/// Every stage-produced serializable value and the native representation of an
/// opaque stage-produced value use this domain so materialization can be
/// checked without interpreting the value.
pub fn image_program_value_content(bytes: &[u8]) -> Digest {
    Digest::of_bytes("image-program-value-content-v1", bytes)
}

/// Exact semantics of one backend-native opaque value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ImageOpaqueValueKindV1 {
    /// Backend-native prompt conditioning.
    Conditioning,
    /// Verified adapter artifact retained for native application.
    LoraArtifact,
    /// Request-local post-Euler state that has not been serialized.
    CheckpointState,
}

/// Exact decoded pixel representation carried by one PNG value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ImagePngColorV1 {
    /// Eight-bit red, green, and blue samples.
    Rgb8,
    /// Eight-bit red, green, blue, and alpha samples.
    Rgba8,
}

impl ImagePngColorV1 {
    /// Returns the exact decoded channel count.
    pub const fn channels(self) -> u32 {
        match self {
            Self::Rgb8 => 3,
            Self::Rgba8 => 4,
        }
    }
}

/// Exact logical representation of one resident program value.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "kind")]
pub enum ImageProgramValueSpecV1 {
    /// Valid UTF-8 bytes with a public maximum length.
    Utf8 {
        /// Maximum accepted byte length.
        maximum_bytes: u64,
    },
    /// Tightly packed interleaved RGB bytes.
    Rgb8 {
        /// Pixel width.
        width: u32,
        /// Pixel height.
        height: u32,
    },
    /// Tightly packed interleaved RGBA bytes.
    Rgba8 {
        /// Pixel width.
        width: u32,
        /// Pixel height.
        height: u32,
    },
    /// Lossless PNG bytes with exact decoded geometry and encoder identity.
    Png {
        /// Decoded pixel width.
        width: u32,
        /// Decoded pixel height.
        height: u32,
        /// Decoded sample representation.
        color: ImagePngColorV1,
        /// Exact deterministic encoder implementation.
        encoding: Digest,
        /// Maximum accepted encoded byte length.
        maximum_bytes: u64,
    },
    /// Tightly packed single-channel bytes.
    Gray8 {
        /// Pixel width.
        width: u32,
        /// Pixel height.
        height: u32,
    },
    /// Exact typed tensor bytes under one representation identity.
    Tensor {
        /// Tensor shape, scalar type, layout, and placement contract.
        tensor: TensorSpec,
        /// Exact native representation and conversion identity.
        representation: Digest,
    },
    /// Authenticated serialized checkpoint envelope.
    Checkpoint {
        /// Exact checkpoint compatibility and lineage domain.
        compatibility: Digest,
        /// Maximum serialized byte length.
        maximum_bytes: u64,
    },
    /// Backend-native bytes or state unavailable to public interpretation.
    Opaque {
        /// Mechanical role of the opaque value.
        opaque_kind: ImageOpaqueValueKindV1,
        /// Exact producer/consumer compatibility domain.
        compatibility: Digest,
        /// Maximum resident byte length.
        maximum_bytes: u64,
    },
}

impl ImageProgramValueSpecV1 {
    /// Validates dimensions, tensor metadata, and public byte bounds.
    ///
    /// # Errors
    ///
    /// Returns the first invalid dimension, tensor, or byte bound.
    pub fn validate(&self) -> Result<(), CoreError> {
        let bytes = match self {
            Self::Utf8 { maximum_bytes }
            | Self::Checkpoint { maximum_bytes, .. }
            | Self::Opaque { maximum_bytes, .. } => {
                validate_variable_bytes(*maximum_bytes)?;
                *maximum_bytes
            }
            Self::Png {
                width,
                height,
                maximum_bytes,
                ..
            } => {
                validate_dimensions(*width, *height)?;
                validate_variable_bytes(*maximum_bytes)?;
                *maximum_bytes
            }
            Self::Rgb8 { width, height } => image_bytes(*width, *height, 3)?,
            Self::Rgba8 { width, height } => image_bytes(*width, *height, 4)?,
            Self::Gray8 { width, height } => image_bytes(*width, *height, 1)?,
            Self::Tensor { tensor, .. } => {
                tensor.validate()?;
                tensor_bytes(tensor)?
            }
        };
        if bytes > MAX_IMAGE_PROGRAM_VALUE_BYTES {
            return Err(CoreError::invalid(
                "image program value",
                format!("exceeds {MAX_IMAGE_PROGRAM_VALUE_BYTES} bytes"),
            ));
        }
        Ok(())
    }

    /// Returns the maximum resident bytes represented by this value.
    ///
    /// # Errors
    ///
    /// Returns an error when the value specification is invalid.
    pub fn maximum_bytes(&self) -> Result<u64, CoreError> {
        self.validate()?;
        match self {
            Self::Utf8 { maximum_bytes }
            | Self::Checkpoint { maximum_bytes, .. }
            | Self::Opaque { maximum_bytes, .. }
            | Self::Png { maximum_bytes, .. } => Ok(*maximum_bytes),
            Self::Rgb8 { width, height } => image_bytes(*width, *height, 3),
            Self::Rgba8 { width, height } => image_bytes(*width, *height, 4),
            Self::Gray8 { width, height } => image_bytes(*width, *height, 1),
            Self::Tensor { tensor, .. } => tensor_bytes(tensor),
        }
    }

    fn validate_buffer_length(&self, bytes: u64) -> Result<(), CoreError> {
        self.validate()?;
        let valid = match self {
            Self::Utf8 { maximum_bytes }
            | Self::Checkpoint { maximum_bytes, .. }
            | Self::Opaque { maximum_bytes, .. }
            | Self::Png { maximum_bytes, .. } => bytes > 0 && bytes <= *maximum_bytes,
            Self::Rgb8 { .. } | Self::Rgba8 { .. } | Self::Gray8 { .. } | Self::Tensor { .. } => {
                bytes == self.maximum_bytes()?
            }
        };
        if !valid {
            return Err(CoreError::invalid(
                "image program buffer",
                "length is incompatible with its value specification",
            ));
        }
        Ok(())
    }
}

/// One numbered logical value in a resident image program.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageProgramValueV1 {
    /// Zero-based value identifier.
    pub value: u16,
    /// Exact logical representation.
    pub spec: ImageProgramValueSpecV1,
}

/// One external value and its exact caller-owned input metadata.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageProgramInputV1 {
    /// Logical value initialized by this input.
    pub value: u16,
    /// Exact readable input metadata.
    pub buffer: BufferSpec,
}

/// One native-stage input role bound to an earlier logical value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageProgramInputBindingV1 {
    /// Mechanical input role.
    pub role: ImageBufferRole,
    /// Earlier logical value supplying the role.
    pub value: u16,
}

/// One scheduled adapter applied by a native stage.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageProgramLoraV1 {
    /// Earlier opaque `LoRA` artifact value.
    pub value: u16,
    /// Exact native target-resolution implementation.
    pub target: Digest,
    /// Exact pre-denoiser scale schedule.
    pub scales: ScaleSchedule,
}

/// Mechanical role of one value produced by a native stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "kind")]
pub enum ImageProgramNativeOutputRoleV1 {
    /// Primary image or tensor result.
    Primary,
    /// Post-Euler checkpoint state captured at the declared boundary.
    CheckpointState,
    /// Tensor snapshot for one zero-based observation request.
    Observation {
        /// Observation request index.
        observation: u16,
    },
}

/// One typed value produced by a native image stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageProgramNativeOutputV1 {
    /// Mechanical output role.
    pub role: ImageProgramNativeOutputRoleV1,
    /// New logical value receiving the result.
    pub value: u16,
}

/// One exact diffusion or direct-VAE operation inside a resident program.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageProgramNativeStageV1 {
    /// Exact compatible profile descriptor.
    pub profile: Digest,
    /// Exact resident load identity.
    pub load: Digest,
    /// Native image operation.
    pub operation: ImageOperation,
    /// Output width.
    pub width: u32,
    /// Output height.
    pub height: u32,
    /// Requested primary representation.
    pub output_format: ImageOutputFormat,
    /// Exact seed policy.
    pub seed: SeedSelection,
    /// Exact RNG implementation.
    pub rng: Digest,
    /// Exact placement selected at load.
    pub placement: Digest,
    /// Diffusion schedule, present exactly for diffusion operations.
    pub schedule: Option<DiffusionSchedule>,
    /// Exact IEEE-754 guidance scale bits.
    pub guidance_scale_bits: u32,
    /// Exact IEEE-754 source-image strength bits.
    pub strength_bits: u32,
    /// Earlier values bound to native input roles.
    pub inputs: Vec<ImageProgramInputBindingV1>,
    /// Ordered scheduled adapter stack.
    pub loras: Vec<ImageProgramLoraV1>,
    /// Ordered installed tensor operators.
    pub operators: Vec<OperatorInvocation>,
    /// Ordered tensor observations.
    pub observations: Vec<ObservationRequest>,
    /// Optional zero-based post-Euler boundary at which an input checkpoint
    /// state replaces the recomputed prefix.
    pub checkpoint_restore_at_step: Option<u32>,
    /// Optional zero-based post-Euler boundary captured as checkpoint state.
    pub checkpoint_after_step: Option<u32>,
    /// Canonically ordered primary, checkpoint, and snapshot outputs.
    pub outputs: Vec<ImageProgramNativeOutputV1>,
}

impl ImageProgramNativeStageV1 {
    /// Returns the exact guidance scale.
    pub fn guidance_scale(&self) -> f32 {
        f32::from_bits(self.guidance_scale_bits)
    }

    /// Returns the exact source-image strength.
    pub fn strength(&self) -> f32 {
        f32::from_bits(self.strength_bits)
    }

    fn references(&self) -> impl Iterator<Item = u16> + '_ {
        self.inputs
            .iter()
            .map(|input| input.value)
            .chain(self.loras.iter().map(|lora| lora.value))
    }

    fn output_values(&self) -> impl Iterator<Item = u16> + '_ {
        self.outputs.iter().map(|output| output.value)
    }

    fn validate_for(&self, values: &[ImageProgramValueV1]) -> Result<(), CoreError> {
        self.validate_scalars()?;
        if self.inputs.len() > crate::MAX_IMAGE_BUFFERS
            || self.loras.len() > MAX_IMAGE_LORAS
            || self.operators.len() > MAX_IMAGE_OPERATORS
            || self.observations.len() > MAX_IMAGE_OBSERVATIONS
        {
            return Err(CoreError::invalid(
                "image program native stage",
                "a public collection bound was exceeded",
            ));
        }
        let mut roles = Vec::new();
        for input in &self.inputs {
            if roles.contains(&input.role) && is_singleton_role(input.role) {
                return Err(CoreError::invalid(
                    "image program native inputs",
                    "a singleton role was repeated",
                ));
            }
            roles.push(input.role);
            let spec = value_spec(values, input.value)?;
            validate_role_spec(input.role, spec, self.width, self.height)?;
        }
        self.validate_required_inputs()?;
        let step_count = self.schedule.as_ref().map_or(0, DiffusionSchedule::steps);
        if self.operation.uses_diffusion() != self.schedule.is_some() {
            return Err(CoreError::invalid(
                "image program native schedule",
                "must be present exactly for diffusion operations",
            ));
        }
        if let Some(schedule) = &self.schedule {
            schedule.validate()?;
        }
        for lora in &self.loras {
            if !matches!(
                value_spec(values, lora.value)?,
                ImageProgramValueSpecV1::Opaque {
                    opaque_kind: ImageOpaqueValueKindV1::LoraArtifact,
                    ..
                }
            ) {
                return Err(CoreError::invalid(
                    "image program LoRA",
                    "must reference an opaque LoRA artifact value",
                ));
            }
            lora.scales.validate_for(step_count)?;
        }
        self.validate_checkpoint_restore_binding()?;
        if !self.operation.uses_diffusion()
            && (!self.loras.is_empty()
                || !self.operators.is_empty()
                || !self.observations.is_empty()
                || self.checkpoint_restore_at_step.is_some()
                || self.checkpoint_after_step.is_some()
                || self
                    .inputs
                    .iter()
                    .any(|input| input.role == ImageBufferRole::Checkpoint))
        {
            return Err(CoreError::invalid(
                "image program VAE stage",
                "cannot carry diffusion-scoped mechanics",
            ));
        }
        for operator in &self.operators {
            operator.validate_for(step_count)?;
        }
        for observation in &self.observations {
            observation.validate_for(step_count)?;
        }
        self.validate_snapshot_order()?;
        if [self.checkpoint_restore_at_step, self.checkpoint_after_step]
            .into_iter()
            .flatten()
            .any(|step| usize::try_from(step).map_or(true, |step| step >= step_count))
        {
            return Err(CoreError::invalid(
                "image program checkpoint boundary",
                "must name an in-range post-Euler step",
            ));
        }
        self.validate_outputs(values)?;
        self.validate_checkpoint_compatibility(values)
    }

    fn validate_snapshot_order(&self) -> Result<(), CoreError> {
        let snapshot_steps = self.observations.iter().filter_map(|observation| {
            if observation.kind != ObservationKind::Snapshot {
                return None;
            }
            match &observation.steps {
                StepSelector::Exact { steps } => steps.first().copied(),
                StepSelector::All => None,
            }
        });
        if snapshot_steps
            .scan(None, |previous, step| {
                let ordered = previous.is_none_or(|previous| previous < step);
                *previous = Some(step);
                Some(ordered)
            })
            .all(|ordered| ordered)
        {
            return Ok(());
        }
        Err(CoreError::invalid(
            "image snapshot observations",
            "must name strictly increasing distinct boundaries",
        ))
    }

    fn validate_required_inputs(&self) -> Result<(), CoreError> {
        let has = |role| self.inputs.iter().any(|input| input.role == role);
        let needs_source = matches!(
            self.operation,
            ImageOperation::ImageToImage
                | ImageOperation::Inpaint
                | ImageOperation::Outpaint
                | ImageOperation::VaeEncode
        );
        let needs_mask = matches!(
            self.operation,
            ImageOperation::Inpaint | ImageOperation::Outpaint
        );
        if (self.operation.uses_diffusion() && !has(ImageBufferRole::PositiveConditioning))
            || (needs_source && !has(ImageBufferRole::SourceImage))
            || (needs_mask && !has(ImageBufferRole::Mask))
            || (self.operation == ImageOperation::VaeDecode
                && !has(ImageBufferRole::TensorSnapshot))
        {
            return Err(CoreError::invalid(
                "image program native inputs",
                "a required operation input is missing",
            ));
        }
        Ok(())
    }

    fn validate_checkpoint_restore_binding(&self) -> Result<(), CoreError> {
        let checkpoint_inputs = self
            .inputs
            .iter()
            .filter(|input| input.role == ImageBufferRole::Checkpoint)
            .count();
        if checkpoint_inputs != usize::from(self.checkpoint_restore_at_step.is_some()) {
            return Err(CoreError::invalid(
                "image program checkpoint restore",
                "must bind exactly one state value when a restore boundary is declared",
            ));
        }
        Ok(())
    }

    fn validate_scalars(&self) -> Result<(), CoreError> {
        validate_dimensions(self.width, self.height)?;
        if !self.guidance_scale().is_finite()
            || !self.strength().is_finite()
            || !(0.0..=1.0).contains(&self.strength())
        {
            return Err(CoreError::invalid(
                "image program native scalar",
                "guidance must be finite and strength must be finite within 0..=1",
            ));
        }
        let output_compatible = match self.operation {
            ImageOperation::VaeEncode => self.output_format == ImageOutputFormat::Tensor,
            _ => matches!(
                self.output_format,
                ImageOutputFormat::Rgb8 | ImageOutputFormat::Rgba8 | ImageOutputFormat::Png
            ),
        };
        if !output_compatible {
            return Err(CoreError::invalid(
                "image program native output",
                "format is unsupported or incompatible with the operation",
            ));
        }
        let strength_compatible = match self.operation {
            ImageOperation::TextToImage | ImageOperation::VaeEncode | ImageOperation::VaeDecode => {
                self.strength_bits == 1.0_f32.to_bits()
            }
            ImageOperation::ImageToImage | ImageOperation::Inpaint | ImageOperation::Outpaint => {
                self.strength() > 0.0
            }
        };
        if !strength_compatible {
            return Err(CoreError::invalid(
                "image program native strength",
                "is not canonical for the selected operation",
            ));
        }
        Ok(())
    }

    fn validate_outputs(&self, values: &[ImageProgramValueV1]) -> Result<(), CoreError> {
        let expected_count = 1
            + usize::from(self.checkpoint_after_step.is_some())
            + self
                .observations
                .iter()
                .filter(|request| request.kind == ObservationKind::Snapshot)
                .count();
        if self.outputs.len() != expected_count
            || self.outputs.first().map(|output| output.role)
                != Some(ImageProgramNativeOutputRoleV1::Primary)
        {
            return Err(CoreError::invalid(
                "image program native outputs",
                "must contain one primary followed by every declared state or snapshot",
            ));
        }
        let primary = value_spec(values, self.outputs[0].value)?;
        validate_primary_output(
            self.operation,
            self.output_format,
            self.width,
            self.height,
            primary,
        )?;
        let mut cursor = 1;
        if self.checkpoint_after_step.is_some() {
            let Some(output) = self.outputs.get(cursor) else {
                return Err(CoreError::invalid(
                    "image program checkpoint output",
                    "is missing",
                ));
            };
            if output.role != ImageProgramNativeOutputRoleV1::CheckpointState
                || !matches!(
                    value_spec(values, output.value)?,
                    ImageProgramValueSpecV1::Opaque {
                        opaque_kind: ImageOpaqueValueKindV1::CheckpointState,
                        ..
                    }
                )
            {
                return Err(CoreError::invalid(
                    "image program checkpoint output",
                    "must be a checkpoint-state value in canonical position",
                ));
            }
            cursor += 1;
        }
        for (index, observation) in self.observations.iter().enumerate() {
            if observation.kind != ObservationKind::Snapshot {
                continue;
            }
            let observation = u16::try_from(index).map_err(|_| {
                CoreError::invalid("image program observation output", "index exceeds u16")
            })?;
            let Some(output) = self.outputs.get(cursor) else {
                return Err(CoreError::invalid(
                    "image program observation output",
                    "is missing",
                ));
            };
            if output.role != (ImageProgramNativeOutputRoleV1::Observation { observation })
                || !matches!(
                    value_spec(values, output.value)?,
                    ImageProgramValueSpecV1::Tensor { .. }
                )
            {
                return Err(CoreError::invalid(
                    "image program observation output",
                    "must be a tensor in request order",
                ));
            }
            cursor += 1;
        }
        Ok(())
    }

    fn validate_checkpoint_compatibility(
        &self,
        values: &[ImageProgramValueV1],
    ) -> Result<(), CoreError> {
        let restored = self
            .inputs
            .iter()
            .find(|input| input.role == ImageBufferRole::Checkpoint)
            .map(|input| checkpoint_state_compatibility(values, input.value))
            .transpose()?;
        let captured = self
            .outputs
            .iter()
            .find(|output| output.role == ImageProgramNativeOutputRoleV1::CheckpointState)
            .map(|output| checkpoint_state_compatibility(values, output.value))
            .transpose()?;
        if let (Some(restored), Some(captured)) = (restored, captured)
            && restored != captured
        {
            return Err(CoreError::invalid(
                "image program checkpoint state",
                "restored and captured compatibility domains differ",
            ));
        }
        Ok(())
    }
}

/// One ordered resident program operation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "kind")]
pub enum ImageProgramStageOperationV1 {
    /// Diffusion or direct-VAE work performed by the resident native runtime.
    Native {
        /// Exact native-stage mechanics.
        plan: Box<ImageProgramNativeStageV1>,
    },
    /// Exact deterministic RGB8 mask blend.
    MaskBlend {
        /// Earlier RGB8 base value.
        base: u16,
        /// Earlier RGB8 overlay value.
        overlay: u16,
        /// Earlier Gray8 mask value.
        mask: u16,
        /// New RGB8 output value.
        output: u16,
    },
    /// Deserialize an authenticated checkpoint into request-local native state.
    RestoreCheckpoint {
        /// Earlier serialized checkpoint value.
        checkpoint: u16,
        /// New opaque checkpoint-state value.
        state: u16,
        /// Exact restore implementation identity.
        implementation: Digest,
    },
    /// Serialize request-local native state into an authenticated checkpoint.
    CaptureCheckpoint {
        /// Earlier opaque checkpoint-state value.
        state: u16,
        /// New serialized checkpoint value.
        checkpoint: u16,
        /// Exact capture implementation identity.
        implementation: Digest,
    },
}

impl ImageProgramStageOperationV1 {
    /// Returns logical values consumed by this operation in canonical order.
    pub fn referenced_values(&self) -> Vec<u16> {
        self.references()
    }

    /// Returns logical values produced by this operation in canonical order.
    pub fn produced_values(&self) -> Vec<u16> {
        self.outputs()
    }

    fn references(&self) -> Vec<u16> {
        match self {
            Self::Native { plan } => plan.references().collect(),
            Self::MaskBlend {
                base,
                overlay,
                mask,
                ..
            } => vec![*base, *overlay, *mask],
            Self::RestoreCheckpoint { checkpoint, .. } => vec![*checkpoint],
            Self::CaptureCheckpoint { state, .. } => vec![*state],
        }
    }

    fn outputs(&self) -> Vec<u16> {
        match self {
            Self::Native { plan } => plan.output_values().collect(),
            Self::MaskBlend { output, .. } => vec![*output],
            Self::RestoreCheckpoint { state, .. } => vec![*state],
            Self::CaptureCheckpoint { checkpoint, .. } => vec![*checkpoint],
        }
    }

    fn validate_types(&self, values: &[ImageProgramValueV1]) -> Result<(), CoreError> {
        match self {
            Self::Native { plan } => plan.validate_for(values),
            Self::MaskBlend {
                base,
                overlay,
                mask,
                output,
            } => validate_mask_blend(values, *base, *overlay, *mask, *output),
            Self::RestoreCheckpoint {
                checkpoint, state, ..
            } => validate_checkpoint_conversion(values, *checkpoint, *state, true),
            Self::CaptureCheckpoint {
                state, checkpoint, ..
            } => validate_checkpoint_conversion(values, *checkpoint, *state, false),
        }
    }

    /// Returns the identity of this exact stage operation.
    ///
    /// # Errors
    ///
    /// Returns an error when deterministic serialization fails.
    pub fn digest(&self) -> Result<Digest, CoreError> {
        Digest::of_serializable("image-program-stage-operation-v1", self)
    }
}

/// One ordered stage in a resident image program.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageProgramStageV1 {
    /// Zero-based stage index.
    pub stage: u16,
    /// Exact stage mechanics.
    pub operation: ImageProgramStageOperationV1,
}

/// Value selected for one caller-owned output allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "kind")]
pub enum ImageProgramOutputSourceV1 {
    /// One typed logical value.
    Value {
        /// Value to materialize.
        value: u16,
    },
    /// Deterministic program receipt serialized after execution.
    ProgramReceipt,
}

/// One ordered caller-owned output route.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageProgramOutputRouteV1 {
    /// Zero-based route index.
    pub route: u16,
    /// Exact source written to the route.
    pub source: ImageProgramOutputSourceV1,
    /// Exact writable output allocation.
    pub buffer: BufferSpec,
}

/// Release point derived for one logical program value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageProgramReleaseV1 {
    /// Value released after its final consumer.
    pub value: u16,
    /// Stage after which release occurs, or `None` after output materialization.
    pub after_stage: Option<u16>,
}

/// Conservative preflight value-arena accounting.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageProgramLivenessV1 {
    /// Maximum sum of declared live-value byte bounds.
    pub peak_bytes: u64,
    /// Canonical value-release order.
    pub releases: Vec<ImageProgramReleaseV1>,
}

/// Complete bounded resident image program.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageProgramPlanV1 {
    /// Canonically numbered logical values.
    pub values: Vec<ImageProgramValueV1>,
    /// External value producers.
    pub inputs: Vec<ImageProgramInputV1>,
    /// Ordered native and deterministic stages.
    pub stages: Vec<ImageProgramStageV1>,
    /// Ordered caller-owned output routes.
    pub outputs: Vec<ImageProgramOutputRouteV1>,
    /// Request-scope resident-session cleanup policy.
    pub cleanup: ImageCleanupPolicy,
}

impl ImageProgramPlanV1 {
    /// Validates the complete typed graph and its conservative arena bound.
    ///
    /// # Errors
    ///
    /// Returns the first collection, type, producer, ordering, routing, or
    /// liveness defect.
    pub fn validate(&self) -> Result<(), CoreError> {
        self.validate_collections_and_values()?;
        let producers = self.validate_producers_and_stages()?;
        let consumers = self.validate_outputs_and_consumers(&producers)?;
        let liveness = self.compute_liveness(&consumers)?;
        if liveness.peak_bytes > MAX_IMAGE_PROGRAM_ARENA_BYTES {
            return Err(CoreError::invalid(
                "image program arena",
                format!("exceeds {MAX_IMAGE_PROGRAM_ARENA_BYTES} live bytes"),
            ));
        }
        Ok(())
    }

    /// Computes canonical value release points and peak declared live bytes.
    ///
    /// # Errors
    ///
    /// Returns an error when the complete program is invalid.
    pub fn liveness(&self) -> Result<ImageProgramLivenessV1, CoreError> {
        self.validate_collections_and_values()?;
        let producers = self.validate_producers_and_stages()?;
        let consumers = self.validate_outputs_and_consumers(&producers)?;
        let liveness = self.compute_liveness(&consumers)?;
        if liveness.peak_bytes > MAX_IMAGE_PROGRAM_ARENA_BYTES {
            return Err(CoreError::invalid(
                "image program arena",
                format!("exceeds {MAX_IMAGE_PROGRAM_ARENA_BYTES} live bytes"),
            ));
        }
        Ok(liveness)
    }

    /// Returns the identity of this exact resident program.
    ///
    /// # Errors
    ///
    /// Returns a validation or deterministic serialization error.
    pub fn digest(&self) -> Result<Digest, CoreError> {
        self.validate()?;
        Digest::of_serializable("image-program-plan-v1", self)
    }

    fn validate_collections_and_values(&self) -> Result<(), CoreError> {
        if self.values.is_empty()
            || self.values.len() > MAX_IMAGE_PROGRAM_VALUES
            || self.inputs.is_empty()
            || self.inputs.len() > MAX_IMAGE_PROGRAM_INPUTS
            || self.stages.is_empty()
            || self.stages.len() > MAX_IMAGE_PROGRAM_STAGES
            || self.outputs.is_empty()
            || self.outputs.len() > MAX_IMAGE_PROGRAM_OUTPUTS
        {
            return Err(CoreError::invalid(
                "image program collections",
                "a required collection is empty or exceeds its public bound",
            ));
        }
        for (index, value) in self.values.iter().enumerate() {
            if usize::from(value.value) != index {
                return Err(CoreError::invalid(
                    "image program values",
                    "identifiers must be contiguous and declared in order",
                ));
            }
            value.spec.validate()?;
        }
        Ok(())
    }

    fn validate_producers_and_stages(&self) -> Result<Vec<Option<Producer>>, CoreError> {
        let mut producers = vec![None; self.values.len()];
        let mut input_allocations = HashSet::new();
        for input in &self.inputs {
            input.buffer.validate()?;
            value_spec(&self.values, input.value)?
                .validate_buffer_length(input.buffer.byte_length)?;
            if !input_allocations.insert(input.buffer.identity.clone()) {
                return Err(CoreError::invalid(
                    "image program inputs",
                    "allocation identities must be unique",
                ));
            }
            assign_producer(&mut producers, input.value, Producer::Input)?;
        }
        for (index, stage) in self.stages.iter().enumerate() {
            if usize::from(stage.stage) != index {
                return Err(CoreError::invalid(
                    "image program stages",
                    "indices must be contiguous and declared in execution order",
                ));
            }
            for reference in stage.operation.references() {
                let Some(producer) = producer_at(&producers, reference)? else {
                    return Err(CoreError::invalid(
                        "image program stage reference",
                        "must name an external input or an earlier stage output",
                    ));
                };
                if matches!(producer, Producer::Stage(producer) if producer >= index) {
                    return Err(CoreError::invalid(
                        "image program stage reference",
                        "must name a value produced by an earlier stage",
                    ));
                }
            }
            stage.operation.validate_types(&self.values)?;
            let mut stage_outputs = HashSet::new();
            for output in stage.operation.outputs() {
                if !stage_outputs.insert(output) {
                    return Err(CoreError::invalid(
                        "image program stage outputs",
                        "a stage cannot produce one value more than once",
                    ));
                }
                assign_producer(&mut producers, output, Producer::Stage(index))?;
            }
        }
        if producers.iter().any(Option::is_none) {
            return Err(CoreError::invalid(
                "image program values",
                "every declared value must have exactly one producer",
            ));
        }
        Ok(producers)
    }

    fn validate_outputs_and_consumers(
        &self,
        producers: &[Option<Producer>],
    ) -> Result<Vec<Vec<usize>>, CoreError> {
        let mut consumers = vec![Vec::new(); self.values.len()];
        for (stage_index, stage) in self.stages.iter().enumerate() {
            for reference in stage.operation.references() {
                consumers
                    .get_mut(usize::from(reference))
                    .ok_or_else(|| {
                        CoreError::invalid("image program consumer", "value is outside the plan")
                    })?
                    .push(stage_index);
            }
        }
        self.validate_output_routes(&mut consumers)?;
        for (index, value_consumers) in consumers.iter().enumerate() {
            if value_consumers.is_empty() {
                return Err(CoreError::invalid(
                    "image program value",
                    "every input and stage output must have a consumer",
                ));
            }
            if matches!(
                self.values[index].spec,
                ImageProgramValueSpecV1::Opaque {
                    opaque_kind: ImageOpaqueValueKindV1::CheckpointState,
                    ..
                }
            ) && value_consumers.len() != 1
            {
                return Err(CoreError::invalid(
                    "image program checkpoint state",
                    "must have exactly one mutable consumer",
                ));
            }
            let producer = producers[index]
                .ok_or_else(|| CoreError::invalid("image program value", "producer is missing"))?;
            if let Producer::Stage(stage) = producer
                && value_consumers.iter().any(|consumer| *consumer <= stage)
            {
                return Err(CoreError::invalid(
                    "image program value",
                    "cannot be consumed before it is produced",
                ));
            }
        }
        Ok(consumers)
    }

    fn validate_output_routes(&self, consumers: &mut [Vec<usize>]) -> Result<(), CoreError> {
        let output_boundary = self.stages.len();
        let mut allocations = self
            .inputs
            .iter()
            .map(|input| input.buffer.identity.clone())
            .collect::<HashSet<_>>();
        let mut routed_values = HashSet::new();
        let mut receipt_routes = 0;
        for (index, route) in self.outputs.iter().enumerate() {
            if usize::from(route.route) != index {
                return Err(CoreError::invalid(
                    "image program outputs",
                    "route indices must be contiguous and declared in order",
                ));
            }
            route.buffer.validate()?;
            if !allocations.insert(route.buffer.identity.clone()) {
                return Err(CoreError::invalid(
                    "image program outputs",
                    "output allocations must not alias an input or another output",
                ));
            }
            match route.source {
                ImageProgramOutputSourceV1::Value { value } => {
                    let spec = value_spec(&self.values, value)?;
                    if matches!(
                        spec,
                        ImageProgramValueSpecV1::Opaque {
                            opaque_kind: ImageOpaqueValueKindV1::CheckpointState,
                            ..
                        }
                    ) || !routed_values.insert(value)
                    {
                        return Err(CoreError::invalid(
                            "image program output value",
                            "must be a uniquely routed serializable value",
                        ));
                    }
                    spec.validate_buffer_length(route.buffer.byte_length)?;
                    consumers[usize::from(value)].push(output_boundary);
                }
                ImageProgramOutputSourceV1::ProgramReceipt => {
                    receipt_routes += 1;
                    if index + 1 != self.outputs.len()
                        || route.buffer.byte_length > MAX_IMAGE_PROGRAM_VALUE_BYTES
                    {
                        return Err(CoreError::invalid(
                            "image program receipt output",
                            "must be the single final bounded route",
                        ));
                    }
                }
            }
        }
        if receipt_routes != 1 {
            return Err(CoreError::invalid(
                "image program receipt output",
                "must be routed exactly once",
            ));
        }
        Ok(())
    }

    fn compute_liveness(
        &self,
        consumers: &[Vec<usize>],
    ) -> Result<ImageProgramLivenessV1, CoreError> {
        let mut live = vec![false; self.values.len()];
        let mut bytes = 0_u64;
        for input in &self.inputs {
            add_live_value(&self.values, &mut live, &mut bytes, input.value)?;
        }
        let mut peak_bytes = bytes;
        let mut releases = Vec::with_capacity(self.values.len());
        for (stage_index, stage) in self.stages.iter().enumerate() {
            for output in stage.operation.outputs() {
                add_live_value(&self.values, &mut live, &mut bytes, output)?;
            }
            peak_bytes = peak_bytes.max(bytes);
            for (value, value_consumers) in consumers.iter().enumerate() {
                if live[value] && value_consumers.iter().max() == Some(&stage_index) {
                    release_live_value(
                        &self.values,
                        &mut live,
                        &mut bytes,
                        value,
                        Some(stage_index),
                        &mut releases,
                    )?;
                }
            }
        }
        let output_boundary = self.stages.len();
        for (value, value_consumers) in consumers.iter().enumerate() {
            if live[value] && value_consumers.iter().max() == Some(&output_boundary) {
                release_live_value(
                    &self.values,
                    &mut live,
                    &mut bytes,
                    value,
                    None,
                    &mut releases,
                )?;
            }
        }
        if bytes != 0 || live.iter().any(|is_live| *is_live) {
            return Err(CoreError::invalid(
                "image program liveness",
                "not every produced value reached its final consumer",
            ));
        }
        Ok(ImageProgramLivenessV1 {
            peak_bytes,
            releases,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Producer {
    Input,
    Stage(usize),
}

/// Exact content identity and initialized length of one produced value.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageProgramValueReceiptV1 {
    /// Produced logical value.
    pub value: u16,
    /// Exact initialized content identity.
    pub content: Digest,
    /// Exact initialized byte length.
    pub bytes: u64,
}

/// Deterministic receipt for one completed program stage.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageProgramStageReceiptV1 {
    /// Zero-based completed stage index.
    pub stage: u16,
    /// Exact stage-operation identity.
    pub operation: Digest,
    /// Produced values in the operation's canonical output order.
    pub outputs: Vec<ImageProgramValueReceiptV1>,
    /// Observation result identities in request order.
    pub observations: Vec<Digest>,
}

/// Exact initialized output accounting.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageProgramOutputReceiptV1 {
    /// Zero-based output route.
    pub route: u16,
    /// Caller-owned allocation identity.
    pub allocation: Digest,
    /// Exact content identity, absent only for the self-containing receipt route.
    pub content: Option<Digest>,
    /// Exact initialized prefix length.
    pub bytes_written: u64,
}

/// Terminal boundary reached by one resident image program.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "kind")]
pub enum ImageProgramTerminalV1 {
    /// Every stage and output route completed.
    Completed,
    /// Cancellation was observed before request-local state existed.
    CancelledBeforeStart,
    /// Cancellation was observed after one completed stage.
    CancelledAfterStage {
        /// Last completed stage.
        stage: u16,
    },
    /// Cancellation was observed inside a native stage at a post-Euler boundary.
    CancelledAfterStep {
        /// Native stage that did not complete.
        stage: u16,
        /// Zero-based completed Euler transition.
        step: u32,
    },
    /// One stage failed before publishing its outputs.
    FailedAtStage {
        /// Stage that did not complete.
        stage: u16,
        /// Complete stage failure detail.
        failure: String,
    },
    /// Cleanup or resident-state certainty was lost.
    CleanupUncertain {
        /// Last completed stage, or `None` when no stage completed.
        after_stage: Option<u16>,
        /// Complete cleanup failure detail.
        failure: String,
    },
}

/// Observed cleanup outcome for one resident image program.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "kind")]
pub enum ImageProgramCleanupDispositionV1 {
    /// The plan retained a known resident session.
    Retained,
    /// Cleanup was confirmed for one prior runtime epoch.
    Confirmed {
        /// Runtime epoch invalidated by cleanup.
        cleared_epoch: u64,
    },
    /// Cancellation preceded request-local state.
    NotRequired,
    /// Cleanup or state release could not be verified.
    Uncertain {
        /// Complete cleanup failure detail.
        failure: String,
    },
}

/// Deterministic mechanical receipt for one resident image program.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageProgramReceiptV1 {
    /// Exact program-plan identity.
    pub plan: Digest,
    /// Exact backend build/runtime identity.
    pub backend: Digest,
    /// Runtime epoch used by native handles.
    pub runtime_epoch: u64,
    /// Number of completely published stages.
    pub completed_stages: u16,
    /// Completed stage receipts as an ordered prefix.
    pub stages: Vec<ImageProgramStageReceiptV1>,
    /// Completed output writes in route order.
    pub outputs: Vec<ImageProgramOutputReceiptV1>,
    /// Terminal program boundary.
    pub terminal: ImageProgramTerminalV1,
    /// Observed cleanup outcome.
    pub cleanup: ImageProgramCleanupDispositionV1,
}

impl ImageProgramReceiptV1 {
    /// Validates plan lineage, stage prefix, output writes, terminal position,
    /// and cleanup disposition.
    ///
    /// # Errors
    ///
    /// Returns the first inconsistency with the exact program.
    pub fn validate_for(&self, plan: &ImageProgramPlanV1) -> Result<(), CoreError> {
        plan.validate()?;
        if self.plan != plan.digest()? || usize::from(self.completed_stages) != self.stages.len() {
            return Err(CoreError::invalid(
                "image program receipt",
                "plan identity or completed-stage count differs",
            ));
        }
        let mut contents = plan
            .inputs
            .iter()
            .map(|input| {
                (
                    input.value,
                    (input.buffer.identity.clone(), input.buffer.byte_length),
                )
            })
            .collect::<HashMap<_, _>>();
        for (index, receipt) in self.stages.iter().enumerate() {
            let stage = plan.stages.get(index).ok_or_else(|| {
                CoreError::invalid("image program stage receipt", "exceeds the plan")
            })?;
            let expected_outputs = stage.operation.outputs();
            let expected_observations = match &stage.operation {
                ImageProgramStageOperationV1::Native { plan } => plan.observations.len(),
                _ => 0,
            };
            if usize::from(receipt.stage) != index
                || receipt.operation != stage.operation.digest()?
                || receipt.outputs.len() != expected_outputs.len()
                || receipt.observations.len() != expected_observations
            {
                return Err(CoreError::invalid(
                    "image program stage receipt",
                    "is not the exact completed operation prefix",
                ));
            }
            for (output, expected) in receipt.outputs.iter().zip(expected_outputs) {
                if output.value != expected {
                    return Err(CoreError::invalid(
                        "image program value receipt",
                        "value order differs from the stage operation",
                    ));
                }
                value_spec(&plan.values, output.value)?.validate_buffer_length(output.bytes)?;
                if contents
                    .insert(output.value, (output.content.clone(), output.bytes))
                    .is_some()
                {
                    return Err(CoreError::invalid(
                        "image program value receipt",
                        "published a value more than once",
                    ));
                }
            }
        }
        self.validate_terminal(plan)?;
        self.validate_outputs(plan, &contents)?;
        self.validate_cleanup(plan)
    }

    /// Returns the identity of this exact deterministic receipt.
    ///
    /// # Errors
    ///
    /// Returns a validation or deterministic serialization error.
    pub fn digest_for(&self, plan: &ImageProgramPlanV1) -> Result<Digest, CoreError> {
        self.validate_for(plan)?;
        Digest::of_serializable("image-program-receipt-v1", self)
    }

    fn validate_terminal(&self, plan: &ImageProgramPlanV1) -> Result<(), CoreError> {
        let completed = usize::from(self.completed_stages);
        let valid = match self.terminal {
            ImageProgramTerminalV1::Completed => completed == plan.stages.len(),
            ImageProgramTerminalV1::CancelledBeforeStart => completed == 0,
            ImageProgramTerminalV1::CancelledAfterStage { stage } => {
                usize::from(stage).checked_add(1) == Some(completed)
            }
            ImageProgramTerminalV1::CancelledAfterStep { stage, step } => {
                usize::from(stage) == completed
                    && plan.stages.get(usize::from(stage)).is_some_and(|stage| {
                        match &stage.operation {
                            ImageProgramStageOperationV1::Native { plan } => {
                                plan.schedule.as_ref().is_some_and(|schedule| {
                                    usize::try_from(step).is_ok_and(|step| step < schedule.steps())
                                })
                            }
                            _ => false,
                        }
                    })
            }
            ImageProgramTerminalV1::FailedAtStage { stage, .. } => {
                usize::from(stage) == completed && completed < plan.stages.len()
            }
            ImageProgramTerminalV1::CleanupUncertain { after_stage, .. } => after_stage
                .map_or(completed == 0, |stage| {
                    usize::from(stage).checked_add(1) == Some(completed)
                }),
        };
        if !valid {
            return Err(CoreError::invalid(
                "image program terminal",
                "does not match the completed-stage prefix",
            ));
        }
        match &self.terminal {
            ImageProgramTerminalV1::FailedAtStage { failure, .. }
            | ImageProgramTerminalV1::CleanupUncertain { failure, .. }
                if failure.is_empty() =>
            {
                return Err(CoreError::invalid(
                    "image program terminal",
                    "failure detail is empty",
                ));
            }
            _ => {}
        }
        if self.terminal == ImageProgramTerminalV1::Completed {
            if self.outputs.len() != plan.outputs.len() {
                return Err(CoreError::invalid(
                    "image program receipt",
                    "completed terminal requires every output route",
                ));
            }
        } else if !self.outputs.is_empty() {
            return Err(CoreError::invalid(
                "image program receipt",
                "non-completed terminals cannot publish output routes",
            ));
        }
        Ok(())
    }

    fn validate_outputs(
        &self,
        plan: &ImageProgramPlanV1,
        contents: &HashMap<u16, (Digest, u64)>,
    ) -> Result<(), CoreError> {
        for (index, output) in self.outputs.iter().enumerate() {
            let route = plan.outputs.get(index).ok_or_else(|| {
                CoreError::invalid("image program output receipt", "route exceeds the plan")
            })?;
            if usize::from(output.route) != index
                || output.allocation != route.buffer.identity
                || output.bytes_written == 0
                || output.bytes_written > route.buffer.byte_length
            {
                return Err(CoreError::invalid(
                    "image program output receipt",
                    "route, allocation, or initialized length differs",
                ));
            }
            match route.source {
                ImageProgramOutputSourceV1::Value { value } => {
                    let Some((content, bytes)) = contents.get(&value) else {
                        return Err(CoreError::invalid(
                            "image program output receipt",
                            "source value was not produced",
                        ));
                    };
                    if output.content.as_ref() != Some(content) || output.bytes_written != *bytes {
                        return Err(CoreError::invalid(
                            "image program output receipt",
                            "content or initialized length differs from its value",
                        ));
                    }
                }
                ImageProgramOutputSourceV1::ProgramReceipt if output.content.is_none() => {}
                ImageProgramOutputSourceV1::ProgramReceipt => {
                    return Err(CoreError::invalid(
                        "image program output receipt",
                        "self-containing receipt route cannot claim its own content digest",
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_cleanup(&self, plan: &ImageProgramPlanV1) -> Result<(), CoreError> {
        let valid = match &self.terminal {
            ImageProgramTerminalV1::CancelledBeforeStart => {
                self.cleanup == ImageProgramCleanupDispositionV1::NotRequired
            }
            ImageProgramTerminalV1::CleanupUncertain { failure, .. } => matches!(
                &self.cleanup,
                ImageProgramCleanupDispositionV1::Uncertain {
                    failure: cleanup_failure
                } if cleanup_failure == failure
            ),
            ImageProgramTerminalV1::Completed
            | ImageProgramTerminalV1::CancelledAfterStage { .. }
            | ImageProgramTerminalV1::CancelledAfterStep { .. }
            | ImageProgramTerminalV1::FailedAtStage { .. } => matches!(
                (plan.cleanup, &self.cleanup),
                (
                    ImageCleanupPolicy::RetainSession,
                    ImageProgramCleanupDispositionV1::Retained
                ) | (
                    ImageCleanupPolicy::ClearSession,
                    ImageProgramCleanupDispositionV1::Confirmed { .. }
                )
            ),
        };
        if valid {
            Ok(())
        } else {
            Err(CoreError::invalid(
                "image program cleanup",
                "disposition differs from the plan or terminal state",
            ))
        }
    }
}

/// Observed native placement of one logical value.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "kind")]
pub enum ImageProgramValuePlacementV1 {
    /// Value remained in host memory.
    Host,
    /// Value remained on one exact device.
    Device {
        /// Adapter-reported device identity.
        device: String,
    },
    /// Value occupied host memory and/or more than one device.
    Mixed {
        /// Canonical sorted unique device identities.
        devices: Vec<String>,
    },
}

impl ImageProgramValuePlacementV1 {
    fn validate(&self) -> Result<(), CoreError> {
        let valid_device = |device: &str| {
            !device.is_empty() && device.len() <= MAX_DEVICE_LABEL_BYTES && !device.contains('\0')
        };
        match self {
            Self::Host => Ok(()),
            Self::Device { device } if valid_device(device) => Ok(()),
            Self::Mixed { devices }
                if !devices.is_empty()
                    && devices.iter().all(|device| valid_device(device))
                    && devices.windows(2).all(|pair| pair[0] < pair[1]) =>
            {
                Ok(())
            }
            _ => Err(CoreError::invalid(
                "image program value placement",
                "device labels must be bounded, NUL-free, sorted, and unique",
            )),
        }
    }
}

/// Non-deterministic placement and transfer measurements for one value.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageProgramValueMeasurementV1 {
    /// Logical value.
    pub value: u16,
    /// Observed native placement.
    pub placement: ImageProgramValuePlacementV1,
    /// Count of host-to-device transfers.
    pub host_to_device_transfers: u64,
    /// Host-to-device bytes.
    pub host_to_device_bytes: u64,
    /// Count of device-to-host transfers.
    pub device_to_host_transfers: u64,
    /// Device-to-host bytes.
    pub device_to_host_bytes: u64,
}

/// Deployment measurements excluded from deterministic plan and receipt identities.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageProgramMeasurementsV1 {
    /// Exact program-plan identity.
    pub plan: Digest,
    /// Exact backend build/runtime identity.
    pub backend: Digest,
    /// Runtime epoch used by native handles.
    pub runtime_epoch: u64,
    /// Wall time for each completed stage, in nanoseconds.
    pub stage_wall_time_ns: Vec<u64>,
    /// Native compute time for each completed stage, when available.
    pub stage_native_time_ns: Vec<Option<u64>>,
    /// Observed peak native value-arena bytes.
    pub peak_arena_bytes: u64,
    /// Placement and transfer accounting for every value that existed.
    pub values: Vec<ImageProgramValueMeasurementV1>,
}

impl ImageProgramMeasurementsV1 {
    /// Validates measurements against one exact receipt without incorporating
    /// them into deterministic identity.
    ///
    /// # Errors
    ///
    /// Returns the first lineage, count, placement, or byte-bound defect.
    pub fn validate_for(
        &self,
        plan: &ImageProgramPlanV1,
        receipt: &ImageProgramReceiptV1,
    ) -> Result<(), CoreError> {
        receipt.validate_for(plan)?;
        if self.plan != plan.digest()?
            || self.backend != receipt.backend
            || self.runtime_epoch != receipt.runtime_epoch
            || self.stage_wall_time_ns.len() != receipt.stages.len()
            || self.stage_native_time_ns.len() != receipt.stages.len()
            || self.peak_arena_bytes > plan.liveness()?.peak_bytes
        {
            return Err(CoreError::invalid(
                "image program measurements",
                "lineage, stage count, or peak bytes differ from execution",
            ));
        }
        let mut expected = plan
            .inputs
            .iter()
            .map(|input| input.value)
            .collect::<HashSet<_>>();
        for stage in plan.stages.iter().take(receipt.stages.len()) {
            expected.extend(stage.operation.outputs());
        }
        let mut measured = HashSet::new();
        for value in &self.values {
            value.placement.validate()?;
            value_spec(&plan.values, value.value)?;
            if !measured.insert(value.value) {
                return Err(CoreError::invalid(
                    "image program measurements",
                    "a value was measured more than once",
                ));
            }
        }
        if measured != expected {
            return Err(CoreError::invalid(
                "image program measurements",
                "must describe exactly the values that existed",
            ));
        }
        Ok(())
    }
}

fn value_spec(
    values: &[ImageProgramValueV1],
    value: u16,
) -> Result<&ImageProgramValueSpecV1, CoreError> {
    values
        .get(usize::from(value))
        .map(|value| &value.spec)
        .ok_or_else(|| CoreError::invalid("image program value", "identifier is outside the plan"))
}

fn assign_producer(
    producers: &mut [Option<Producer>],
    value: u16,
    producer: Producer,
) -> Result<(), CoreError> {
    let entry = producers
        .get_mut(usize::from(value))
        .ok_or_else(|| CoreError::invalid("image program producer", "value is outside the plan"))?;
    if entry.replace(producer).is_some() {
        return Err(CoreError::invalid(
            "image program producer",
            "a value has more than one producer",
        ));
    }
    Ok(())
}

fn producer_at(producers: &[Option<Producer>], value: u16) -> Result<Option<Producer>, CoreError> {
    producers
        .get(usize::from(value))
        .copied()
        .ok_or_else(|| CoreError::invalid("image program reference", "value is outside the plan"))
}

fn add_live_value(
    values: &[ImageProgramValueV1],
    live: &mut [bool],
    bytes: &mut u64,
    value: u16,
) -> Result<(), CoreError> {
    let index = usize::from(value);
    let is_live = live
        .get_mut(index)
        .ok_or_else(|| CoreError::invalid("image program liveness", "value is outside the plan"))?;
    if *is_live {
        return Err(CoreError::invalid(
            "image program liveness",
            "value became live more than once",
        ));
    }
    *bytes = bytes
        .checked_add(value_spec(values, value)?.maximum_bytes()?)
        .ok_or_else(|| CoreError::invalid("image program liveness", "byte count overflowed"))?;
    *is_live = true;
    Ok(())
}

fn release_live_value(
    values: &[ImageProgramValueV1],
    live: &mut [bool],
    bytes: &mut u64,
    value: usize,
    after_stage: Option<usize>,
    releases: &mut Vec<ImageProgramReleaseV1>,
) -> Result<(), CoreError> {
    *bytes = bytes
        .checked_sub(values[value].spec.maximum_bytes()?)
        .ok_or_else(|| CoreError::invalid("image program liveness", "byte count underflowed"))?;
    live[value] = false;
    releases.push(ImageProgramReleaseV1 {
        value: u16::try_from(value)
            .map_err(|_| CoreError::invalid("image program liveness", "value exceeds u16"))?,
        after_stage: after_stage
            .map(u16::try_from)
            .transpose()
            .map_err(|_| CoreError::invalid("image program liveness", "stage exceeds u16"))?,
    });
    Ok(())
}

fn validate_variable_bytes(bytes: u64) -> Result<(), CoreError> {
    if bytes == 0 || bytes > MAX_IMAGE_PROGRAM_VALUE_BYTES {
        return Err(CoreError::invalid(
            "image program value bytes",
            format!("must be within 1..={MAX_IMAGE_PROGRAM_VALUE_BYTES}"),
        ));
    }
    Ok(())
}

fn validate_dimensions(width: u32, height: u32) -> Result<(), CoreError> {
    if width == 0 || height == 0 || width > MAX_IMAGE_DIMENSION || height > MAX_IMAGE_DIMENSION {
        return Err(CoreError::invalid(
            "image program dimensions",
            format!("must be within 1..={MAX_IMAGE_DIMENSION}"),
        ));
    }
    Ok(())
}

fn image_bytes(width: u32, height: u32, channels: u64) -> Result<u64, CoreError> {
    validate_dimensions(width, height)?;
    u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or_else(|| CoreError::invalid("image program image bytes", "length overflowed"))
}

fn tensor_bytes(tensor: &TensorSpec) -> Result<u64, CoreError> {
    let scalar_bytes = match tensor.dtype {
        TensorDType::F32 => 4,
        TensorDType::F16 | TensorDType::Bf16 => 2,
    };
    tensor
        .elements()?
        .checked_mul(scalar_bytes)
        .ok_or_else(|| CoreError::invalid("image program tensor bytes", "length overflowed"))
}

const fn is_singleton_role(role: ImageBufferRole) -> bool {
    !matches!(
        role,
        ImageBufferRole::ReferenceImage | ImageBufferRole::Lora
    )
}

fn validate_role_spec(
    role: ImageBufferRole,
    spec: &ImageProgramValueSpecV1,
    width: u32,
    height: u32,
) -> Result<(), CoreError> {
    let compatible = match role {
        ImageBufferRole::PositiveConditioning | ImageBufferRole::NegativeConditioning => matches!(
            spec,
            ImageProgramValueSpecV1::Utf8 { .. }
                | ImageProgramValueSpecV1::Tensor { .. }
                | ImageProgramValueSpecV1::Opaque {
                    opaque_kind: ImageOpaqueValueKindV1::Conditioning,
                    ..
                }
        ),
        ImageBufferRole::SourceImage => matches!(
            spec,
            ImageProgramValueSpecV1::Rgb8 {
                width: value_width,
                height: value_height
            } | ImageProgramValueSpecV1::Rgba8 {
                width: value_width,
                height: value_height
            } if (*value_width, *value_height) == (width, height)
        ),
        ImageBufferRole::ReferenceImage => matches!(
            spec,
            ImageProgramValueSpecV1::Rgb8 { .. } | ImageProgramValueSpecV1::Rgba8 { .. }
        ),
        ImageBufferRole::Mask => matches!(
            spec,
            ImageProgramValueSpecV1::Gray8 {
                width: value_width,
                height: value_height
            } if (*value_width, *value_height) == (width, height)
        ),
        ImageBufferRole::Lora => false,
        ImageBufferRole::Checkpoint => matches!(
            spec,
            ImageProgramValueSpecV1::Opaque {
                opaque_kind: ImageOpaqueValueKindV1::CheckpointState,
                ..
            }
        ),
        ImageBufferRole::TensorSnapshot => {
            matches!(spec, ImageProgramValueSpecV1::Tensor { .. })
        }
    };
    if compatible {
        Ok(())
    } else {
        Err(CoreError::invalid(
            "image program native input",
            "role is incompatible with the typed value",
        ))
    }
}

fn validate_primary_output(
    operation: ImageOperation,
    format: ImageOutputFormat,
    width: u32,
    height: u32,
    spec: &ImageProgramValueSpecV1,
) -> Result<(), CoreError> {
    let compatible = match (operation, format, spec) {
        (
            ImageOperation::VaeEncode,
            ImageOutputFormat::Tensor,
            ImageProgramValueSpecV1::Tensor { .. },
        ) => true,
        (
            _,
            ImageOutputFormat::Rgb8,
            ImageProgramValueSpecV1::Rgb8 {
                width: value_width,
                height: value_height,
            },
        )
        | (
            _,
            ImageOutputFormat::Rgba8,
            ImageProgramValueSpecV1::Rgba8 {
                width: value_width,
                height: value_height,
            },
        )
        | (
            _,
            ImageOutputFormat::Png,
            ImageProgramValueSpecV1::Png {
                width: value_width,
                height: value_height,
                ..
            },
        ) => (*value_width, *value_height) == (width, height),
        _ => false,
    };
    if compatible {
        Ok(())
    } else {
        Err(CoreError::invalid(
            "image program primary output",
            "typed value differs from operation, format, or canvas",
        ))
    }
}

fn validate_mask_blend(
    values: &[ImageProgramValueV1],
    base: u16,
    overlay: u16,
    mask: u16,
    output: u16,
) -> Result<(), CoreError> {
    let ImageProgramValueSpecV1::Rgb8 { width, height } = value_spec(values, base)? else {
        return Err(CoreError::invalid(
            "image program mask blend",
            "base must be RGB8",
        ));
    };
    if value_spec(values, overlay)?
        != &(ImageProgramValueSpecV1::Rgb8 {
            width: *width,
            height: *height,
        })
        || value_spec(values, mask)?
            != &(ImageProgramValueSpecV1::Gray8 {
                width: *width,
                height: *height,
            })
        || value_spec(values, output)?
            != &(ImageProgramValueSpecV1::Rgb8 {
                width: *width,
                height: *height,
            })
    {
        return Err(CoreError::invalid(
            "image program mask blend",
            "overlay, mask, and output must match the RGB8 base canvas",
        ));
    }
    Ok(())
}

fn validate_checkpoint_conversion(
    values: &[ImageProgramValueV1],
    checkpoint: u16,
    state: u16,
    restoring: bool,
) -> Result<(), CoreError> {
    let ImageProgramValueSpecV1::Checkpoint {
        compatibility: checkpoint_compatibility,
        ..
    } = value_spec(values, checkpoint)?
    else {
        return Err(CoreError::invalid(
            "image program checkpoint",
            "serialized value has the wrong type",
        ));
    };
    let ImageProgramValueSpecV1::Opaque {
        opaque_kind: ImageOpaqueValueKindV1::CheckpointState,
        compatibility: state_compatibility,
        ..
    } = value_spec(values, state)?
    else {
        return Err(CoreError::invalid(
            "image program checkpoint",
            "native state value has the wrong type",
        ));
    };
    if checkpoint_compatibility != state_compatibility {
        return Err(CoreError::invalid(
            "image program checkpoint",
            if restoring {
                "restore compatibility differs"
            } else {
                "capture compatibility differs"
            },
        ));
    }
    Ok(())
}

fn checkpoint_state_compatibility(
    values: &[ImageProgramValueV1],
    state: u16,
) -> Result<&Digest, CoreError> {
    let ImageProgramValueSpecV1::Opaque {
        opaque_kind: ImageOpaqueValueKindV1::CheckpointState,
        compatibility,
        ..
    } = value_spec(values, state)?
    else {
        return Err(CoreError::invalid(
            "image program checkpoint state",
            "value has the wrong type",
        ));
    };
    Ok(compatibility)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ImageCleanupPolicy, TensorLayout};

    fn digest(domain: &str) -> Digest {
        Digest::of_bytes(domain, domain.as_bytes())
    }

    fn buffer(domain: &str, bytes: u64) -> BufferSpec {
        BufferSpec::new(digest(domain), bytes, "application/octet-stream").unwrap()
    }

    fn value(value: u16, spec: ImageProgramValueSpecV1) -> ImageProgramValueV1 {
        ImageProgramValueV1 { value, spec }
    }

    fn schedule() -> DiffusionSchedule {
        DiffusionSchedule::new(digest("schedule"), vec![1.0, 0.5, 0.0]).unwrap()
    }

    fn native(
        operation: ImageOperation,
        inputs: Vec<ImageProgramInputBindingV1>,
        output: u16,
        output_format: ImageOutputFormat,
    ) -> ImageProgramStageOperationV1 {
        let diffusion = operation.uses_diffusion();
        ImageProgramStageOperationV1::Native {
            plan: Box::new(ImageProgramNativeStageV1 {
                profile: digest("profile"),
                load: digest("load"),
                operation,
                width: 2,
                height: 1,
                output_format,
                seed: SeedSelection::Fixed { seed: 7 },
                rng: digest("rng"),
                placement: digest("placement"),
                schedule: diffusion.then(schedule),
                guidance_scale_bits: 1.0_f32.to_bits(),
                strength_bits: if matches!(
                    operation,
                    ImageOperation::ImageToImage
                        | ImageOperation::Inpaint
                        | ImageOperation::Outpaint
                ) {
                    0.75_f32.to_bits()
                } else {
                    1.0_f32.to_bits()
                },
                inputs,
                loras: Vec::new(),
                operators: Vec::new(),
                observations: Vec::new(),
                checkpoint_restore_at_step: None,
                checkpoint_after_step: None,
                outputs: vec![ImageProgramNativeOutputV1 {
                    role: ImageProgramNativeOutputRoleV1::Primary,
                    value: output,
                }],
            }),
        }
    }

    fn binding(role: ImageBufferRole, value: u16) -> ImageProgramInputBindingV1 {
        ImageProgramInputBindingV1 { role, value }
    }

    fn graph_values() -> Vec<ImageProgramValueV1> {
        let tensor = TensorSpec::new(
            vec![1, 2],
            TensorDType::F32,
            TensorLayout::DimensionZeroFastest,
            "host",
        )
        .unwrap();
        vec![
            value(0, ImageProgramValueSpecV1::Utf8 { maximum_bytes: 16 }),
            value(
                1,
                ImageProgramValueSpecV1::Gray8 {
                    width: 2,
                    height: 1,
                },
            ),
            value(
                2,
                ImageProgramValueSpecV1::Rgb8 {
                    width: 2,
                    height: 1,
                },
            ),
            value(
                3,
                ImageProgramValueSpecV1::Rgb8 {
                    width: 2,
                    height: 1,
                },
            ),
            value(
                4,
                ImageProgramValueSpecV1::Rgb8 {
                    width: 2,
                    height: 1,
                },
            ),
            value(
                5,
                ImageProgramValueSpecV1::Rgb8 {
                    width: 2,
                    height: 1,
                },
            ),
            value(
                6,
                ImageProgramValueSpecV1::Tensor {
                    tensor,
                    representation: digest("tensor-representation"),
                },
            ),
            value(
                7,
                ImageProgramValueSpecV1::Rgb8 {
                    width: 2,
                    height: 1,
                },
            ),
        ]
    }

    fn graph_stages() -> Vec<ImageProgramStageV1> {
        vec![
            ImageProgramStageV1 {
                stage: 0,
                operation: native(
                    ImageOperation::TextToImage,
                    vec![binding(ImageBufferRole::PositiveConditioning, 0)],
                    2,
                    ImageOutputFormat::Rgb8,
                ),
            },
            ImageProgramStageV1 {
                stage: 1,
                operation: native(
                    ImageOperation::Inpaint,
                    vec![
                        binding(ImageBufferRole::PositiveConditioning, 0),
                        binding(ImageBufferRole::SourceImage, 2),
                        binding(ImageBufferRole::Mask, 1),
                    ],
                    3,
                    ImageOutputFormat::Rgb8,
                ),
            },
            ImageProgramStageV1 {
                stage: 2,
                operation: native(
                    ImageOperation::Inpaint,
                    vec![
                        binding(ImageBufferRole::PositiveConditioning, 0),
                        binding(ImageBufferRole::SourceImage, 2),
                        binding(ImageBufferRole::Mask, 1),
                    ],
                    4,
                    ImageOutputFormat::Rgb8,
                ),
            },
            ImageProgramStageV1 {
                stage: 3,
                operation: ImageProgramStageOperationV1::MaskBlend {
                    base: 3,
                    overlay: 4,
                    mask: 1,
                    output: 5,
                },
            },
            ImageProgramStageV1 {
                stage: 4,
                operation: native(
                    ImageOperation::VaeEncode,
                    vec![binding(ImageBufferRole::SourceImage, 5)],
                    6,
                    ImageOutputFormat::Tensor,
                ),
            },
            ImageProgramStageV1 {
                stage: 5,
                operation: native(
                    ImageOperation::VaeDecode,
                    vec![binding(ImageBufferRole::TensorSnapshot, 6)],
                    7,
                    ImageOutputFormat::Rgb8,
                ),
            },
        ]
    }

    fn graph() -> ImageProgramPlanV1 {
        ImageProgramPlanV1 {
            values: graph_values(),
            inputs: vec![
                ImageProgramInputV1 {
                    value: 0,
                    buffer: buffer("prompt-input", 6),
                },
                ImageProgramInputV1 {
                    value: 1,
                    buffer: buffer("mask-input", 2),
                },
            ],
            stages: graph_stages(),
            outputs: vec![
                ImageProgramOutputRouteV1 {
                    route: 0,
                    source: ImageProgramOutputSourceV1::Value { value: 7 },
                    buffer: buffer("image-output", 6),
                },
                ImageProgramOutputRouteV1 {
                    route: 1,
                    source: ImageProgramOutputSourceV1::ProgramReceipt,
                    buffer: buffer("receipt-output", 16_384),
                },
            ],
            cleanup: ImageCleanupPolicy::ClearSession,
        }
    }

    fn completed_receipt(plan: &ImageProgramPlanV1) -> ImageProgramReceiptV1 {
        let mut contents = plan
            .inputs
            .iter()
            .map(|input| (input.value, input.buffer.identity.clone()))
            .collect::<HashMap<_, _>>();
        let stages = plan
            .stages
            .iter()
            .map(|stage| {
                let outputs = stage
                    .operation
                    .outputs()
                    .into_iter()
                    .map(|value| {
                        let content =
                            Digest::of_bytes("synthetic-program-value", &value.to_le_bytes());
                        contents.insert(value, content.clone());
                        ImageProgramValueReceiptV1 {
                            value,
                            content,
                            bytes: plan.values[usize::from(value)]
                                .spec
                                .maximum_bytes()
                                .unwrap(),
                        }
                    })
                    .collect();
                ImageProgramStageReceiptV1 {
                    stage: stage.stage,
                    operation: stage.operation.digest().unwrap(),
                    outputs,
                    observations: Vec::new(),
                }
            })
            .collect::<Vec<_>>();
        let final_content = contents[&7].clone();
        ImageProgramReceiptV1 {
            plan: plan.digest().unwrap(),
            backend: digest("backend"),
            runtime_epoch: 9,
            completed_stages: u16::try_from(plan.stages.len()).unwrap(),
            stages,
            outputs: vec![
                ImageProgramOutputReceiptV1 {
                    route: 0,
                    allocation: plan.outputs[0].buffer.identity.clone(),
                    content: Some(final_content),
                    bytes_written: 6,
                },
                ImageProgramOutputReceiptV1 {
                    route: 1,
                    allocation: plan.outputs[1].buffer.identity.clone(),
                    content: None,
                    bytes_written: 1_024,
                },
            ],
            terminal: ImageProgramTerminalV1::Completed,
            cleanup: ImageProgramCleanupDispositionV1::Confirmed { cleared_epoch: 9 },
        }
    }

    #[test]
    fn branching_native_graph_round_trips_and_has_a_new_identity() {
        let plan = graph();
        plan.validate().unwrap();
        let encoded = serde_json::to_vec(&plan).unwrap();
        assert_eq!(
            serde_json::from_slice::<ImageProgramPlanV1>(&encoded).unwrap(),
            plan
        );
        assert_eq!(plan.digest().unwrap(), plan.digest().unwrap());
        assert_ne!(
            plan.digest().unwrap(),
            Digest::of_serializable("image-execution-plan-v3", &plan).unwrap()
        );
    }

    #[test]
    fn png_primary_binds_decoded_type_encoder_and_encoded_bound() {
        let mut plan = graph();
        plan.values[7].spec = ImageProgramValueSpecV1::Png {
            width: 2,
            height: 1,
            color: ImagePngColorV1::Rgba8,
            encoding: digest("png-encoding"),
            maximum_bytes: 1_024,
        };
        let ImageProgramStageOperationV1::Native { plan: native } = &mut plan.stages[5].operation
        else {
            panic!("fixture stage must be native");
        };
        native.output_format = ImageOutputFormat::Png;
        plan.outputs[0].buffer = buffer("png-output", 1_024);
        plan.validate().unwrap();

        plan.outputs[0].buffer = buffer("oversized-png-output", 1_025);
        assert!(plan.validate().is_err());
    }

    #[test]
    fn graph_rejects_forward_references_and_duplicate_producers() {
        let mut forward = graph();
        if let ImageProgramStageOperationV1::Native { plan } = &mut forward.stages[0].operation {
            plan.inputs[0].value = 2;
        }
        assert!(forward.validate().is_err());

        let mut duplicate = graph();
        duplicate.inputs.push(ImageProgramInputV1 {
            value: 2,
            buffer: buffer("duplicate-producer", 6),
        });
        assert!(duplicate.validate().is_err());
    }

    #[test]
    fn graph_rejects_type_invalid_masks_and_unused_inputs() {
        let mut wrong_mask = graph();
        wrong_mask.values[1].spec = ImageProgramValueSpecV1::Rgb8 {
            width: 2,
            height: 1,
        };
        assert!(wrong_mask.validate().is_err());

        let mut unused = graph();
        unused
            .values
            .push(value(8, ImageProgramValueSpecV1::Utf8 { maximum_bytes: 8 }));
        unused.inputs.push(ImageProgramInputV1 {
            value: 8,
            buffer: buffer("unused", 4),
        });
        assert!(unused.validate().is_err());
    }

    #[test]
    fn reference_images_preserve_independent_geometry_and_identity() {
        let mut plan = graph();
        plan.values.push(value(
            8,
            ImageProgramValueSpecV1::Rgb8 {
                width: 1,
                height: 2,
            },
        ));
        plan.inputs.push(ImageProgramInputV1 {
            value: 8,
            buffer: buffer("reference-input", 6),
        });
        let ImageProgramStageOperationV1::Native { plan: native } = &mut plan.stages[0].operation
        else {
            panic!("fixture stage must be native");
        };
        native
            .inputs
            .push(binding(ImageBufferRole::ReferenceImage, 8));

        plan.validate().unwrap();
        let independent_digest = plan.digest().unwrap();
        let encoded = serde_json::to_vec(&plan).unwrap();
        let decoded: ImageProgramPlanV1 = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, plan);
        assert_eq!(
            decoded.values[8].spec,
            ImageProgramValueSpecV1::Rgb8 {
                width: 1,
                height: 2,
            }
        );

        plan.values[8].spec = ImageProgramValueSpecV1::Rgb8 {
            width: 2,
            height: 1,
        };
        plan.validate().unwrap();
        assert_ne!(plan.digest().unwrap(), independent_digest);

        plan.values[8].spec = ImageProgramValueSpecV1::Rgba8 {
            width: 1,
            height: 2,
        };
        plan.inputs[2].buffer = buffer("rgba-reference-input", 8);
        plan.validate().unwrap();
    }

    #[test]
    fn reference_images_reject_invalid_specs_and_lengths() {
        let independent_rgb = ImageProgramValueSpecV1::Rgb8 {
            width: 1,
            height: 2,
        };
        assert!(
            validate_role_spec(ImageBufferRole::ReferenceImage, &independent_rgb, 2, 1).is_ok()
        );
        assert!(validate_role_spec(ImageBufferRole::SourceImage, &independent_rgb, 2, 1).is_err());
        assert!(
            validate_role_spec(
                ImageBufferRole::Mask,
                &ImageProgramValueSpecV1::Gray8 {
                    width: 1,
                    height: 2,
                },
                2,
                1
            )
            .is_err()
        );
        assert!(
            validate_role_spec(
                ImageBufferRole::ReferenceImage,
                &ImageProgramValueSpecV1::Gray8 {
                    width: 1,
                    height: 2,
                },
                2,
                1
            )
            .is_err()
        );
        assert!(
            ImageProgramValueSpecV1::Rgb8 {
                width: 0,
                height: 1,
            }
            .validate()
            .is_err()
        );
        assert!(
            ImageProgramValueSpecV1::Rgba8 {
                width: MAX_IMAGE_DIMENSION + 1,
                height: 1,
            }
            .validate()
            .is_err()
        );
        assert!(independent_rgb.validate_buffer_length(5).is_err());
    }

    #[test]
    fn liveness_releases_branch_values_after_their_last_consumers() {
        let plan = graph();
        let liveness = plan.liveness().unwrap();
        assert!(liveness.peak_bytes > 0);
        assert_eq!(liveness.releases.len(), plan.values.len());
        assert_eq!(
            liveness
                .releases
                .iter()
                .find(|release| release.value == 2)
                .unwrap()
                .after_stage,
            Some(2)
        );
        assert_eq!(
            liveness.releases.last().unwrap(),
            &ImageProgramReleaseV1 {
                value: 7,
                after_stage: None
            }
        );
    }

    #[test]
    fn arena_overflow_is_rejected_before_execution() {
        let mut plan = graph();
        plan.values[0].spec = ImageProgramValueSpecV1::Utf8 {
            maximum_bytes: MAX_IMAGE_PROGRAM_VALUE_BYTES,
        };
        plan.inputs[0].buffer.byte_length = 6;
        plan.values[1].spec = ImageProgramValueSpecV1::Opaque {
            opaque_kind: ImageOpaqueValueKindV1::Conditioning,
            compatibility: digest("large-one"),
            maximum_bytes: MAX_IMAGE_PROGRAM_VALUE_BYTES,
        };
        plan.inputs[1].buffer.byte_length = 2;
        plan.values.push(value(
            8,
            ImageProgramValueSpecV1::Opaque {
                opaque_kind: ImageOpaqueValueKindV1::Conditioning,
                compatibility: digest("large-two"),
                maximum_bytes: MAX_IMAGE_PROGRAM_VALUE_BYTES,
            },
        ));
        plan.inputs.push(ImageProgramInputV1 {
            value: 8,
            buffer: buffer("large-input", 1),
        });
        if let ImageProgramStageOperationV1::Native { plan: native } = &mut plan.stages[0].operation
        {
            native
                .inputs
                .push(binding(ImageBufferRole::NegativeConditioning, 8));
        }
        assert!(plan.validate().is_err());
    }

    #[test]
    fn checkpoint_compatibility_and_mutable_fanout_are_rejected() {
        let mut plan = graph();
        plan.values.push(value(
            8,
            ImageProgramValueSpecV1::Checkpoint {
                compatibility: digest("checkpoint-a"),
                maximum_bytes: 128,
            },
        ));
        plan.values.push(value(
            9,
            ImageProgramValueSpecV1::Opaque {
                opaque_kind: ImageOpaqueValueKindV1::CheckpointState,
                compatibility: digest("checkpoint-b"),
                maximum_bytes: 128,
            },
        ));
        plan.inputs.push(ImageProgramInputV1 {
            value: 8,
            buffer: buffer("checkpoint-input", 64),
        });
        plan.stages.insert(
            0,
            ImageProgramStageV1 {
                stage: 0,
                operation: ImageProgramStageOperationV1::RestoreCheckpoint {
                    checkpoint: 8,
                    state: 9,
                    implementation: digest("restore"),
                },
            },
        );
        for (index, stage) in plan.stages.iter_mut().enumerate() {
            stage.stage = u16::try_from(index).unwrap();
        }
        assert!(plan.validate().is_err());

        plan.values[9].spec = ImageProgramValueSpecV1::Opaque {
            opaque_kind: ImageOpaqueValueKindV1::CheckpointState,
            compatibility: digest("checkpoint-a"),
            maximum_bytes: 128,
        };
        if let ImageProgramStageOperationV1::Native { plan: native } = &mut plan.stages[1].operation
        {
            native.inputs.push(binding(ImageBufferRole::Checkpoint, 9));
            native.checkpoint_restore_at_step = Some(0);
        }
        if let ImageProgramStageOperationV1::Native { plan: native } = &mut plan.stages[2].operation
        {
            native.inputs.push(binding(ImageBufferRole::Checkpoint, 9));
            native.checkpoint_restore_at_step = Some(0);
        }
        assert!(plan.validate().is_err());
    }

    #[test]
    fn completed_receipt_and_measurements_validate_separately() {
        let plan = graph();
        let receipt = completed_receipt(&plan);
        assert!(receipt.digest_for(&plan).is_ok());
        let measurements = ImageProgramMeasurementsV1 {
            plan: plan.digest().unwrap(),
            backend: receipt.backend.clone(),
            runtime_epoch: receipt.runtime_epoch,
            stage_wall_time_ns: vec![10; plan.stages.len()],
            stage_native_time_ns: vec![Some(8); plan.stages.len()],
            peak_arena_bytes: plan.liveness().unwrap().peak_bytes,
            values: plan
                .values
                .iter()
                .map(|value| ImageProgramValueMeasurementV1 {
                    value: value.value,
                    placement: ImageProgramValuePlacementV1::Host,
                    host_to_device_transfers: 0,
                    host_to_device_bytes: 0,
                    device_to_host_transfers: 0,
                    device_to_host_bytes: 0,
                })
                .collect(),
        };
        assert!(measurements.validate_for(&plan, &receipt).is_ok());

        let mut tampered = receipt;
        tampered.stages[2].stage = 7;
        assert!(tampered.digest_for(&plan).is_err());
    }

    #[test]
    fn cleanup_uncertainty_must_poison_the_terminal_contract() {
        let plan = graph();
        let failure = "cleanup failed because the native arena remained live".to_owned();
        let receipt = ImageProgramReceiptV1 {
            plan: plan.digest().unwrap(),
            backend: digest("backend"),
            runtime_epoch: 9,
            completed_stages: 0,
            stages: Vec::new(),
            outputs: Vec::new(),
            terminal: ImageProgramTerminalV1::CleanupUncertain {
                after_stage: None,
                failure: failure.clone(),
            },
            cleanup: ImageProgramCleanupDispositionV1::Uncertain { failure },
        };
        assert!(receipt.digest_for(&plan).is_ok());
    }
}

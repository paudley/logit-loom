// SPDX-License-Identifier: MIT OR Apache-2.0

//! Serializable whole-image execution plans over exact buffer identities.

use std::collections::HashSet;

use logit_loom_core::{CoreError, Digest};
use logit_loom_executor::BufferSpec;
use serde::{Deserialize, Serialize};

use crate::{DiffusionSchedule, TensorDType, TensorSpec};

/// Maximum image dimension accepted by the generic execution contract.
pub const MAX_IMAGE_DIMENSION: u32 = 4_096;
/// Maximum input or output buffer bindings in one image execution.
pub const MAX_IMAGE_BUFFERS: usize = 64;
/// Maximum ordered `LoRA` entries in one image execution.
pub const MAX_IMAGE_LORAS: usize = 32;
/// Maximum installed operator invocations in one image execution.
pub const MAX_IMAGE_OPERATORS: usize = 64;
/// Maximum observation requests in one image execution.
pub const MAX_IMAGE_OBSERVATIONS: usize = 64;
/// Maximum bytes in one selector label.
pub const MAX_SELECTOR_LABEL_BYTES: usize = 128;
/// Maximum schema-specific bytes in one installed operator invocation.
pub const MAX_OPERATOR_CONTROL_BYTES: usize = 4_096;

/// Whole-image operation selected by an exact execution plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ImageOperation {
    /// Text conditioning to a new image.
    TextToImage,
    /// Source image plus conditioning to a new image.
    ImageToImage,
    /// Masked replacement inside a source image.
    Inpaint,
    /// Expanded canvas plus mask completion.
    Outpaint,
    /// Image bytes to an exact backend latent representation.
    VaeEncode,
    /// Exact backend latent representation to image bytes.
    VaeDecode,
}

impl ImageOperation {
    pub(crate) const fn uses_diffusion(self) -> bool {
        matches!(
            self,
            Self::TextToImage | Self::ImageToImage | Self::Inpaint | Self::Outpaint
        )
    }
}

/// Requested final image representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ImageOutputFormat {
    /// Interleaved red, green, and blue bytes.
    Rgb8,
    /// Interleaved red, green, blue, and alpha bytes.
    Rgba8,
    /// Lossless PNG bytes.
    Png,
    /// Backend-native tensor bytes for VAE encode.
    Tensor,
}

/// Logical role of an exact execution buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ImageBufferRole {
    /// Positive prompt bytes or encoded conditioning.
    PositiveConditioning,
    /// Negative prompt bytes or encoded conditioning.
    NegativeConditioning,
    /// Image-to-image, inpaint, outpaint, or VAE source image.
    SourceImage,
    /// Additional reference image.
    ReferenceImage,
    /// Spatial mask.
    Mask,
    /// Compatible `LoRA` artifact bytes.
    Lora,
    /// Opaque authenticated checkpoint bytes.
    Checkpoint,
    /// Explicit typed tensor bytes.
    TensorSnapshot,
}

/// Exact interpretation of bytes at the public execution boundary.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "kind")]
pub enum ImageBufferLayout {
    /// Opaque backend bytes whose structure is identified by the media type.
    Opaque,
    /// Exact UTF-8 bytes validated when storage is bound.
    Utf8,
    /// Interleaved RGB bytes with an explicit row stride.
    Rgb8 {
        /// Pixel width.
        width: u32,
        /// Pixel height.
        height: u32,
        /// Bytes between adjacent rows.
        row_stride: u64,
    },
    /// Interleaved RGBA bytes with an explicit row stride.
    Rgba8 {
        /// Pixel width.
        width: u32,
        /// Pixel height.
        height: u32,
        /// Bytes between adjacent rows.
        row_stride: u64,
    },
    /// Single-channel mask bytes with an explicit row stride.
    Gray8 {
        /// Pixel width.
        width: u32,
        /// Pixel height.
        height: u32,
        /// Bytes between adjacent rows.
        row_stride: u64,
    },
    /// Exact typed tensor bytes.
    Tensor(TensorSpec),
}

impl ImageBufferLayout {
    fn validate(&self, byte_length: u64) -> Result<(), CoreError> {
        match self {
            Self::Opaque | Self::Utf8 => Ok(()),
            Self::Rgb8 {
                width,
                height,
                row_stride,
            } => validate_image_bytes(*width, *height, *row_stride, 3, byte_length),
            Self::Rgba8 {
                width,
                height,
                row_stride,
            } => validate_image_bytes(*width, *height, *row_stride, 4, byte_length),
            Self::Gray8 {
                width,
                height,
                row_stride,
            } => validate_image_bytes(*width, *height, *row_stride, 1, byte_length),
            Self::Tensor(tensor) => {
                tensor.validate()?;
                let scalar_bytes = match tensor.dtype {
                    TensorDType::F32 => 4,
                    TensorDType::F16 | TensorDType::Bf16 => 2,
                };
                let expected = tensor
                    .elements()?
                    .checked_mul(scalar_bytes)
                    .ok_or_else(|| {
                        CoreError::invalid("image tensor buffer", "byte length overflowed")
                    })?;
                if byte_length != expected {
                    return Err(CoreError::invalid(
                        "image tensor buffer",
                        "byte length does not match the tensor contract",
                    ));
                }
                Ok(())
            }
        }
    }
}

/// One exact input-slot binding in an image execution plan.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageBufferBinding {
    /// Caller-defined slot used by `LoRA` and operator plans.
    pub slot: u16,
    /// Logical mechanics role.
    pub role: ImageBufferRole,
    /// Exact content/allocation metadata.
    pub buffer: BufferSpec,
    /// Exact byte interpretation.
    pub layout: ImageBufferLayout,
}

impl ImageBufferBinding {
    /// Validates metadata, layout, and role compatibility.
    ///
    /// # Errors
    ///
    /// Returns the first invalid field.
    pub fn validate(&self) -> Result<(), CoreError> {
        self.buffer.validate()?;
        self.layout.validate(self.buffer.byte_length)?;
        let valid_layout = match self.role {
            ImageBufferRole::PositiveConditioning | ImageBufferRole::NegativeConditioning => {
                matches!(
                    self.layout,
                    ImageBufferLayout::Utf8 | ImageBufferLayout::Tensor(_)
                )
            }
            ImageBufferRole::SourceImage | ImageBufferRole::ReferenceImage => matches!(
                self.layout,
                ImageBufferLayout::Rgb8 { .. } | ImageBufferLayout::Rgba8 { .. }
            ),
            ImageBufferRole::Mask => matches!(self.layout, ImageBufferLayout::Gray8 { .. }),
            ImageBufferRole::Lora | ImageBufferRole::Checkpoint => {
                matches!(self.layout, ImageBufferLayout::Opaque)
            }
            ImageBufferRole::TensorSnapshot => {
                matches!(self.layout, ImageBufferLayout::Tensor(_))
            }
        };
        if !valid_layout {
            return Err(CoreError::invalid(
                "image buffer layout",
                "is incompatible with the selected role",
            ));
        }
        Ok(())
    }

    /// Validates bytes that cannot be checked from serialized metadata alone.
    ///
    /// # Errors
    ///
    /// Returns an error when the exact length differs or UTF-8 is malformed.
    pub fn validate_bytes(&self, bytes: &[u8]) -> Result<(), CoreError> {
        self.validate()?;
        let actual = u64::try_from(bytes.len())
            .map_err(|_| CoreError::invalid("image buffer bytes", "length exceeds u64"))?;
        if actual != self.buffer.byte_length {
            return Err(CoreError::invalid(
                "image buffer bytes",
                "length does not match the binding",
            ));
        }
        if matches!(self.layout, ImageBufferLayout::Utf8) && std::str::from_utf8(bytes).is_err() {
            return Err(CoreError::invalid(
                "image conditioning bytes",
                "must be valid UTF-8",
            ));
        }
        Ok(())
    }
}

/// Resolution policy for the exact execution seed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "kind")]
pub enum SeedSelection {
    /// Caller supplied the exact native seed.
    Fixed {
        /// Exact native seed.
        seed: u64,
    },
    /// The installed worker policy selects and receipts a seed.
    WorkerSelected {
        /// Exact seed-policy implementation.
        policy: Digest,
    },
}

/// Exact schedule boundaries selected for one mechanic.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "kind")]
pub enum StepSelector {
    /// Every completed state transition.
    All,
    /// A canonical sorted set of zero-based completed transitions.
    Exact {
        /// Strictly increasing step indices.
        steps: Vec<u32>,
    },
}

impl StepSelector {
    /// Validates selection against an exact step count.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty, non-canonical, or out-of-range set.
    pub fn validate_for(&self, step_count: usize) -> Result<(), CoreError> {
        match self {
            Self::All if step_count == 0 => Err(CoreError::invalid(
                "image step selector",
                "cannot select every step of a non-diffusion operation",
            )),
            Self::All => Ok(()),
            Self::Exact { steps } => {
                if steps.is_empty() || steps.len() > step_count {
                    return Err(CoreError::invalid(
                        "image step selector",
                        "must contain a bounded non-empty set",
                    ));
                }
                if steps.windows(2).any(|pair| pair[0] >= pair[1])
                    || steps
                        .iter()
                        .any(|step| usize::try_from(*step).map_or(true, |step| step >= step_count))
                {
                    return Err(CoreError::invalid(
                        "image step selector",
                        "steps must be unique, increasing, and in range",
                    ));
                }
                Ok(())
            }
        }
    }
}

/// One exact `LoRA` scale beginning at a completed-step boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalePoint {
    /// Zero-based step at which this scale becomes active.
    pub step: u32,
    /// Exact IEEE-754 scale bits.
    pub scale_bits: u32,
}

impl ScalePoint {
    /// Constructs a finite scale point.
    ///
    /// # Errors
    ///
    /// Returns an error when the scale is not finite.
    pub fn new(step: u32, scale: f32) -> Result<Self, CoreError> {
        if !scale.is_finite() {
            return Err(CoreError::invalid("LoRA scale", "must be finite"));
        }
        Ok(Self {
            step,
            scale_bits: scale.to_bits(),
        })
    }

    /// Returns the exact scale.
    pub fn scale(self) -> f32 {
        f32::from_bits(self.scale_bits)
    }
}

/// Canonical piecewise-constant `LoRA` scale schedule.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScaleSchedule {
    /// Strictly increasing scale-change points, beginning at step zero.
    pub points: Vec<ScalePoint>,
}

impl ScaleSchedule {
    /// Validates finite scale points against an execution schedule.
    ///
    /// # Errors
    ///
    /// Returns an error for a missing zero point, duplicate, or out-of-range
    /// step.
    pub fn validate_for(&self, step_count: usize) -> Result<(), CoreError> {
        if self.points.is_empty()
            || self.points.len() > step_count
            || self.points[0].step != 0
            || self.points.iter().any(|point| {
                !point.scale().is_finite()
                    || usize::try_from(point.step).map_or(true, |step| step >= step_count)
            })
            || self
                .points
                .windows(2)
                .any(|pair| pair[0].step >= pair[1].step)
        {
            return Err(CoreError::invalid(
                "LoRA scale schedule",
                "must begin at zero with finite, increasing, in-range points",
            ));
        }
        Ok(())
    }
}

/// One entry in the exact ordered `LoRA` stack.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoraStackEntry {
    /// Slot of the bound `LoRA` artifact.
    pub input_slot: u16,
    /// Exact native target-resolution implementation.
    pub target: Digest,
    /// Per-step multiplier schedule.
    pub scales: ScaleSchedule,
}

/// Exact tensor site selected by an installed mechanic.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "kind")]
pub enum TensorSelector {
    /// Scheduler state immediately after one state transition.
    SchedulerState,
    /// Named conditioning tensor.
    Conditioning {
        /// Exact adapter label.
        label: String,
    },
    /// Exact named site inside a model block.
    ModelBlock {
        /// Exact component identifier.
        component: String,
        /// Zero-based block index.
        block: u32,
        /// Exact site label inside the block.
        site: String,
    },
}

impl TensorSelector {
    /// Validates bounded labels.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty, oversized, or NUL-containing label.
    pub fn validate(&self) -> Result<(), CoreError> {
        let valid = |label: &str| {
            !label.is_empty() && label.len() <= MAX_SELECTOR_LABEL_BYTES && !label.contains('\0')
        };
        match self {
            Self::SchedulerState => Ok(()),
            Self::Conditioning { label } if valid(label) => Ok(()),
            Self::ModelBlock {
                component, site, ..
            } if valid(component) && valid(site) => Ok(()),
            _ => Err(CoreError::invalid(
                "image tensor selector",
                format!(
                    "labels must be non-empty, NUL-free, and at most {MAX_SELECTOR_LABEL_BYTES} bytes"
                ),
            )),
        }
    }
}

/// One exact installed operator invocation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperatorInvocation {
    /// Public control schema identity.
    pub schema: Digest,
    /// Exact installed implementation identity.
    pub implementation: Digest,
    /// Exact native tensor site.
    pub selector: TensorSelector,
    /// Boundaries at which the operator runs.
    pub steps: StepSelector,
    /// Bounded schema-specific control bytes.
    pub controls: Vec<u8>,
}

impl OperatorInvocation {
    pub(crate) fn validate_for(&self, step_count: usize) -> Result<(), CoreError> {
        self.selector.validate()?;
        self.steps.validate_for(step_count)?;
        if self.controls.len() > MAX_OPERATOR_CONTROL_BYTES {
            return Err(CoreError::invalid(
                "image operator controls",
                format!("exceed {MAX_OPERATOR_CONTROL_BYTES} bytes"),
            ));
        }
        Ok(())
    }
}

/// Data retained at selected execution boundaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ObservationKind {
    /// Content identity only.
    Digest,
    /// Bounded numerical summary.
    Statistics,
    /// Exact tensor snapshot in an output buffer.
    Snapshot,
}

/// One exact batched observation request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationRequest {
    /// Exact native tensor site.
    pub selector: TensorSelector,
    /// Boundaries to retain.
    pub steps: StepSelector,
    /// Retained representation.
    pub kind: ObservationKind,
}

impl ObservationRequest {
    pub(crate) fn validate_for(&self, step_count: usize) -> Result<(), CoreError> {
        self.selector.validate()?;
        self.steps.validate_for(step_count)?;
        if self.kind == ObservationKind::Snapshot
            && !matches!(&self.steps, StepSelector::Exact { steps } if steps.len() == 1)
        {
            return Err(CoreError::invalid(
                "image snapshot observation",
                "must select exactly one post-transition boundary",
            ));
        }
        Ok(())
    }
}

/// Complete exact image execution mechanics and input identities.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageExecutionPlan {
    /// Exact compatible profile descriptor.
    pub profile: Digest,
    /// Exact resident load identity.
    pub load: Digest,
    /// Whole-image operation.
    pub operation: ImageOperation,
    /// Output width.
    pub width: u32,
    /// Output height.
    pub height: u32,
    /// Requested output representation.
    pub output_format: ImageOutputFormat,
    /// Exact seed policy.
    pub seed: SeedSelection,
    /// Exact RNG implementation.
    pub rng: Digest,
    /// Exact placement selected at load.
    pub placement: Digest,
    /// Diffusion schedule, absent for direct VAE operations.
    pub schedule: Option<DiffusionSchedule>,
    /// Exact IEEE-754 guidance scale bits.
    pub guidance_scale_bits: u32,
    /// Exact IEEE-754 source-image strength bits.
    pub strength_bits: u32,
    /// Exact input identities and layouts.
    pub inputs: Vec<ImageBufferBinding>,
    /// Ordered `LoRA` stack.
    pub loras: Vec<LoraStackEntry>,
    /// Ordered installed operators.
    pub operators: Vec<OperatorInvocation>,
    /// Batched observation requests.
    pub observations: Vec<ObservationRequest>,
}

impl ImageExecutionPlan {
    /// Returns the exact guidance scale.
    pub fn guidance_scale(&self) -> f32 {
        f32::from_bits(self.guidance_scale_bits)
    }

    /// Returns the exact source-image strength.
    pub fn strength(&self) -> f32 {
        f32::from_bits(self.strength_bits)
    }

    /// Validates bounds, operation requirements, references, and schedules.
    ///
    /// # Errors
    ///
    /// Returns the first mechanically invalid field.
    pub fn validate(&self) -> Result<(), CoreError> {
        self.validate_scalars()?;
        self.validate_collection_bounds()?;
        self.validate_inputs()?;
        let step_count = self.validate_operation()?;
        self.validate_step_mechanics(step_count)
    }

    fn validate_scalars(&self) -> Result<(), CoreError> {
        if self.width == 0
            || self.height == 0
            || self.width > MAX_IMAGE_DIMENSION
            || self.height > MAX_IMAGE_DIMENSION
        {
            return Err(CoreError::invalid(
                "image execution dimensions",
                format!("must be within 1..={MAX_IMAGE_DIMENSION}"),
            ));
        }
        if !self.guidance_scale().is_finite()
            || !self.strength().is_finite()
            || !(0.0..=1.0).contains(&self.strength())
        {
            return Err(CoreError::invalid(
                "image execution scalar",
                "guidance must be finite and strength must be finite within 0..=1",
            ));
        }
        let output_compatible = match self.operation {
            ImageOperation::VaeEncode => self.output_format == ImageOutputFormat::Tensor,
            _ => self.output_format != ImageOutputFormat::Tensor,
        };
        if !output_compatible {
            return Err(CoreError::invalid(
                "image execution output format",
                "is incompatible with the selected operation",
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
                "image execution strength",
                "is not canonical for the selected operation",
            ));
        }
        Ok(())
    }

    fn validate_collection_bounds(&self) -> Result<(), CoreError> {
        if self.inputs.len() > MAX_IMAGE_BUFFERS
            || self.loras.len() > MAX_IMAGE_LORAS
            || self.operators.len() > MAX_IMAGE_OPERATORS
            || self.observations.len() > MAX_IMAGE_OBSERVATIONS
        {
            return Err(CoreError::invalid(
                "image execution collection",
                "a public collection bound was exceeded",
            ));
        }
        Ok(())
    }

    fn validate_inputs(&self) -> Result<(), CoreError> {
        let mut slots = HashSet::new();
        for input in &self.inputs {
            input.validate()?;
            if !slots.insert(input.slot) {
                return Err(CoreError::invalid("image input slots", "must be unique"));
            }
        }
        require_single_roles(&self.inputs)?;
        Ok(())
    }

    fn validate_operation(&self) -> Result<usize, CoreError> {
        let has = |role| self.inputs.iter().any(|input| input.role == role);
        let needs_positive = self.operation.uses_diffusion();
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
        let needs_tensor = self.operation == ImageOperation::VaeDecode;
        if (needs_positive && !has(ImageBufferRole::PositiveConditioning))
            || (needs_source && !has(ImageBufferRole::SourceImage))
            || (needs_mask && !has(ImageBufferRole::Mask))
            || (needs_tensor && !has(ImageBufferRole::TensorSnapshot))
        {
            return Err(CoreError::invalid(
                "image execution inputs",
                "required operation input is missing",
            ));
        }
        for input in self.inputs.iter().filter(|input| {
            matches!(
                input.role,
                ImageBufferRole::SourceImage | ImageBufferRole::Mask
            )
        }) {
            let geometry = match input.layout {
                ImageBufferLayout::Rgb8 { width, height, .. }
                | ImageBufferLayout::Rgba8 { width, height, .. }
                | ImageBufferLayout::Gray8 { width, height, .. } => Some((width, height)),
                _ => None,
            };
            if geometry != Some((self.width, self.height)) {
                return Err(CoreError::invalid(
                    "image execution canvas",
                    "source and mask geometry must match the requested output",
                ));
            }
        }
        if self.operation.uses_diffusion() != self.schedule.is_some() {
            return Err(CoreError::invalid(
                "image execution schedule",
                "must be present exactly for diffusion operations",
            ));
        }
        let step_count = self.schedule.as_ref().map_or(0, DiffusionSchedule::steps);
        if let Some(schedule) = &self.schedule {
            schedule.validate()?;
        }
        Ok(step_count)
    }

    fn validate_step_mechanics(&self, step_count: usize) -> Result<(), CoreError> {
        for lora in &self.loras {
            let Some(input) = self
                .inputs
                .iter()
                .find(|input| input.slot == lora.input_slot)
            else {
                return Err(CoreError::invalid(
                    "image LoRA slot",
                    "does not name an input binding",
                ));
            };
            if input.role != ImageBufferRole::Lora {
                return Err(CoreError::invalid(
                    "image LoRA slot",
                    "does not name a LoRA input",
                ));
            }
            lora.scales.validate_for(step_count)?;
        }
        if !self.operation.uses_diffusion()
            && (!self.loras.is_empty()
                || !self.operators.is_empty()
                || !self.observations.is_empty())
        {
            return Err(CoreError::invalid(
                "image VAE operation",
                "cannot carry step-scoped LoRAs, operators, or observations",
            ));
        }
        for operator in &self.operators {
            operator.validate_for(step_count)?;
        }
        for observation in &self.observations {
            observation.validate_for(step_count)?;
        }
        Ok(())
    }

    /// Returns the identity of this exact execution plan.
    ///
    /// # Errors
    ///
    /// Returns a validation or serialization error.
    pub fn digest(&self) -> Result<Digest, CoreError> {
        self.validate()?;
        Digest::of_serializable("image-execution-plan-v1", self)
    }
}

/// Terminal boundary reached by an image execution.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "kind")]
pub enum ImageTerminal {
    /// Every requested operation completed.
    Completed,
    /// Cancellation was observed before native execution.
    CancelledBeforeStart,
    /// Cancellation was observed after an exact completed diffusion step.
    CancelledAfterStep {
        /// Zero-based completed transition.
        step: u32,
    },
}

/// Deterministic mechanical receipt for one image execution.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageExecutionReceipt {
    /// Exact execution-plan identity.
    pub plan: Digest,
    /// Exact backend build/runtime identity.
    pub backend: Digest,
    /// Exact profile descriptor identity.
    pub profile: Digest,
    /// Session epoch used by native handles.
    pub session_epoch: u64,
    /// Completed diffusion transitions.
    pub completed_steps: u32,
    /// Terminal execution boundary.
    pub terminal: ImageTerminal,
    /// Exact produced output metadata.
    pub outputs: Vec<BufferSpec>,
    /// Produced checkpoint identities.
    pub checkpoints: Vec<Digest>,
    /// Batched observation identities.
    pub observations: Vec<Digest>,
}

impl ImageExecutionReceipt {
    /// Validates position, terminal state, output bounds, and plan identity.
    ///
    /// # Errors
    ///
    /// Returns the first inconsistent field.
    pub fn validate_for(&self, plan: &ImageExecutionPlan) -> Result<(), CoreError> {
        plan.validate()?;
        if self.plan != plan.digest()? || self.profile != plan.profile {
            return Err(CoreError::invalid(
                "image execution receipt",
                "plan or profile identity differs",
            ));
        }
        if self.outputs.len() > MAX_IMAGE_BUFFERS
            || self.checkpoints.len() > MAX_IMAGE_BUFFERS
            || self.observations.len() > MAX_IMAGE_OBSERVATIONS
        {
            return Err(CoreError::invalid(
                "image execution receipt",
                "a receipt collection bound was exceeded",
            ));
        }
        for output in &self.outputs {
            output.validate()?;
        }
        let step_count = plan.schedule.as_ref().map_or(0, DiffusionSchedule::steps);
        let completed = usize::try_from(self.completed_steps)
            .map_err(|_| CoreError::invalid("image execution receipt", "steps exceed usize"))?;
        if completed > step_count {
            return Err(CoreError::invalid(
                "image execution receipt",
                "completed steps exceed the plan",
            ));
        }
        match self.terminal {
            ImageTerminal::Completed if completed != step_count => Err(CoreError::invalid(
                "image execution receipt",
                "completed terminal requires every planned step",
            )),
            ImageTerminal::CancelledBeforeStart if completed != 0 => Err(CoreError::invalid(
                "image execution receipt",
                "pre-start cancellation cannot complete steps",
            )),
            ImageTerminal::CancelledAfterStep { step }
                if usize::try_from(step)
                    .ok()
                    .and_then(|step| step.checked_add(1))
                    != Some(completed) =>
            {
                Err(CoreError::invalid(
                    "image execution receipt",
                    "cancelled boundary does not match completed steps",
                ))
            }
            _ => Ok(()),
        }
    }

    /// Returns the identity of this exact mechanical receipt.
    ///
    /// # Errors
    ///
    /// Returns a validation or serialization error.
    pub fn digest_for(&self, plan: &ImageExecutionPlan) -> Result<Digest, CoreError> {
        self.validate_for(plan)?;
        Digest::of_serializable("image-execution-receipt-v1", self)
    }
}

fn validate_image_bytes(
    width: u32,
    height: u32,
    row_stride: u64,
    channels: u64,
    byte_length: u64,
) -> Result<(), CoreError> {
    if width == 0 || height == 0 || width > MAX_IMAGE_DIMENSION || height > MAX_IMAGE_DIMENSION {
        return Err(CoreError::invalid(
            "image buffer dimensions",
            format!("must be within 1..={MAX_IMAGE_DIMENSION}"),
        ));
    }
    let minimum_stride = u64::from(width)
        .checked_mul(channels)
        .ok_or_else(|| CoreError::invalid("image buffer stride", "overflowed"))?;
    let expected = row_stride
        .checked_mul(u64::from(height))
        .ok_or_else(|| CoreError::invalid("image buffer length", "overflowed"))?;
    if row_stride < minimum_stride || byte_length != expected {
        return Err(CoreError::invalid(
            "image buffer layout",
            "stride or byte length is inconsistent",
        ));
    }
    Ok(())
}

fn require_single_roles(inputs: &[ImageBufferBinding]) -> Result<(), CoreError> {
    for role in [
        ImageBufferRole::PositiveConditioning,
        ImageBufferRole::NegativeConditioning,
        ImageBufferRole::SourceImage,
        ImageBufferRole::Mask,
        ImageBufferRole::Checkpoint,
        ImageBufferRole::TensorSnapshot,
    ] {
        if inputs.iter().filter(|input| input.role == role).count() > 1 {
            return Err(CoreError::invalid(
                "image input roles",
                "a singleton role was repeated",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TensorLayout, TensorSpec};

    fn text(slot: u16, role: ImageBufferRole) -> ImageBufferBinding {
        ImageBufferBinding {
            slot,
            role,
            buffer: BufferSpec::new(
                Digest::of_bytes("test-input", &slot.to_le_bytes()),
                5,
                "text/plain; charset=utf-8",
            )
            .unwrap(),
            layout: ImageBufferLayout::Utf8,
        }
    }

    fn schedule() -> DiffusionSchedule {
        DiffusionSchedule::new(
            Digest::of_bytes("test-scheduler", b"v1"),
            vec![1.0, 0.5, 0.0],
        )
        .unwrap()
    }

    fn plan() -> ImageExecutionPlan {
        ImageExecutionPlan {
            profile: Digest::of_bytes("profile", b"krea"),
            load: Digest::of_bytes("load", b"resident"),
            operation: ImageOperation::TextToImage,
            width: 512,
            height: 512,
            output_format: ImageOutputFormat::Rgb8,
            seed: SeedSelection::Fixed { seed: 7 },
            rng: Digest::of_bytes("rng", b"cpu"),
            placement: Digest::of_bytes("placement", b"vulkan0"),
            schedule: Some(schedule()),
            guidance_scale_bits: 1.0_f32.to_bits(),
            strength_bits: 1.0_f32.to_bits(),
            inputs: vec![text(0, ImageBufferRole::PositiveConditioning)],
            loras: Vec::new(),
            operators: Vec::new(),
            observations: Vec::new(),
        }
    }

    #[test]
    fn text_to_image_plan_round_trips_with_exact_float_bits() {
        let plan = plan();
        plan.validate().unwrap();
        let encoded = serde_json::to_vec(&plan).unwrap();
        let decoded: ImageExecutionPlan = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, plan);
        assert_eq!(decoded.digest().unwrap(), plan.digest().unwrap());
    }

    #[test]
    fn operation_requirements_and_duplicate_slots_fail_closed() {
        let mut value = plan();
        value.operation = ImageOperation::Inpaint;
        assert!(value.validate().is_err());

        let mut value = plan();
        value
            .inputs
            .push(text(0, ImageBufferRole::NegativeConditioning));
        assert!(value.validate().is_err());
    }

    #[test]
    fn operation_output_strength_and_canvas_are_canonical() {
        let mut value = plan();
        value.output_format = ImageOutputFormat::Tensor;
        assert!(value.validate().is_err());

        let mut value = plan();
        value.strength_bits = 0.5_f32.to_bits();
        assert!(value.validate().is_err());

        let mut value = plan();
        value.operation = ImageOperation::ImageToImage;
        value.strength_bits = 0.5_f32.to_bits();
        value.inputs.push(ImageBufferBinding {
            slot: 1,
            role: ImageBufferRole::SourceImage,
            buffer: BufferSpec::new(
                Digest::of_bytes("test-source", b"wrong-canvas"),
                u64::from(256_u32 * 256 * 3),
                "image/rgb8",
            )
            .unwrap(),
            layout: ImageBufferLayout::Rgb8 {
                width: 256,
                height: 256,
                row_stride: 256 * 3,
            },
        });
        assert!(value.validate().is_err());
    }

    #[test]
    fn lora_schedule_is_ordered_and_bound_to_a_lora_slot() {
        let mut value = plan();
        value.inputs.push(ImageBufferBinding {
            slot: 3,
            role: ImageBufferRole::Lora,
            buffer: BufferSpec::new(
                Digest::of_bytes("test-lora", b"adapter"),
                7,
                "application/x-safetensors",
            )
            .unwrap(),
            layout: ImageBufferLayout::Opaque,
        });
        value.loras.push(LoraStackEntry {
            input_slot: 3,
            target: Digest::of_bytes("lora-target", b"all"),
            scales: ScaleSchedule {
                points: vec![
                    ScalePoint::new(0, 0.5).unwrap(),
                    ScalePoint::new(1, -0.25).unwrap(),
                ],
            },
        });
        value.validate().unwrap();
        value.loras[0].scales.points.swap(0, 1);
        assert!(value.validate().is_err());
    }

    #[test]
    fn snapshot_observation_names_one_exact_boundary() {
        let mut value = plan();
        value.observations.push(ObservationRequest {
            selector: TensorSelector::SchedulerState,
            steps: StepSelector::Exact { steps: vec![1] },
            kind: ObservationKind::Snapshot,
        });
        value.validate().unwrap();

        value.observations[0].steps = StepSelector::All;
        assert!(value.validate().is_err());
        value.observations[0].steps = StepSelector::Exact { steps: vec![0, 1] };
        assert!(value.validate().is_err());
    }

    #[test]
    fn buffer_layout_checks_exact_bytes_and_utf8() {
        let binding = text(0, ImageBufferRole::PositiveConditioning);
        assert!(binding.validate_bytes(b"hello").is_ok());
        assert!(
            binding
                .validate_bytes(&[0xff, 0xff, 0xff, 0xff, 0xff])
                .is_err()
        );

        let tensor = TensorSpec::new(
            vec![2, 2],
            TensorDType::F32,
            TensorLayout::DimensionZeroFastest,
            "host",
        )
        .unwrap();
        let binding = ImageBufferBinding {
            slot: 1,
            role: ImageBufferRole::TensorSnapshot,
            buffer: BufferSpec::new(
                Digest::of_bytes("test-tensor", b"bytes"),
                15,
                "application/octet-stream",
            )
            .unwrap(),
            layout: ImageBufferLayout::Tensor(tensor),
        };
        assert!(binding.validate().is_err());
    }

    #[test]
    fn receipt_binds_exact_terminal_boundary() {
        let plan = plan();
        let receipt = ImageExecutionReceipt {
            plan: plan.digest().unwrap(),
            backend: Digest::of_bytes("backend", b"test"),
            profile: plan.profile.clone(),
            session_epoch: 4,
            completed_steps: 2,
            terminal: ImageTerminal::Completed,
            outputs: vec![
                BufferSpec::new(
                    Digest::of_bytes("output", b"rgb"),
                    u64::from(512_u32 * 512 * 3),
                    "image/rgb8",
                )
                .unwrap(),
            ],
            checkpoints: Vec::new(),
            observations: Vec::new(),
        };
        receipt.validate_for(&plan).unwrap();
        let mut cancelled = receipt.clone();
        cancelled.terminal = ImageTerminal::CancelledAfterStep { step: 0 };
        assert!(cancelled.validate_for(&plan).is_err());
    }
}

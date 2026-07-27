// SPDX-License-Identifier: MIT OR Apache-2.0

//! Versioned whole-plan image graph, routing, and cleanup contracts.

use std::collections::HashSet;

use logit_loom_core::{CoreError, Digest};
use logit_loom_executor::BufferSpec;
use serde::{Deserialize, Serialize};

use crate::{
    ImageBufferLayout, ImageBufferRole, ImageCheckpointPlan, ImageCleanupDisposition,
    ImageCleanupPolicy, ImageCompositeOperation, ImageCompositeReceipt, ImageCompositeStage,
    ImageExecutionPlan, ImageOutputFormat, ImageOutputRoute, ImageOutputSource, ImageTerminal,
    ImageValueSource, MAX_IMAGE_COMPOSITE_STAGES, MAX_IMAGE_GRAPH_SCRATCH_BYTES,
};

/// Version-three whole-image execution contract.
///
/// The version-one primary plan and its identity are preserved unchanged.
/// This successor adds checkpoint routing, deterministic compositing, explicit
/// output routes, and request-scope cleanup without reinterpreting the
/// `image-execution-plan-v1` domain.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageExecutionPlanV3 {
    /// Exact primary diffusion or VAE operation.
    pub primary: ImageExecutionPlan,
    /// Optional transactional checkpoint mechanics.
    pub checkpoint: ImageCheckpointPlan,
    /// Ordered deterministic compositing graph.
    pub composites: Vec<ImageCompositeStage>,
    /// Ordered caller-owned output routes.
    pub outputs: Vec<ImageOutputRoute>,
    /// Request-scope cleanup behavior.
    pub cleanup: ImageCleanupPolicy,
}

impl ImageExecutionPlanV3 {
    /// Validates the primary plan, checkpoint references, ordered graph,
    /// output routes, and cleanup contract.
    ///
    /// # Errors
    ///
    /// Returns the first invalid or unsupported graph relationship.
    pub fn validate(&self) -> Result<(), CoreError> {
        self.primary.validate()?;
        if self.primary.output_format != ImageOutputFormat::Rgb8 || self.primary.schedule.is_none()
        {
            return Err(CoreError::invalid(
                "image execution v2 primary",
                "must be a diffusion operation producing RGB8 bytes",
            ));
        }
        if self.composites.len() > MAX_IMAGE_COMPOSITE_STAGES
            || self.outputs.is_empty()
            || self.outputs.len() > crate::MAX_IMAGE_BUFFERS
        {
            return Err(CoreError::invalid(
                "image execution v2 collections",
                "composite or output route count is outside its public bound",
            ));
        }
        let canvas_bytes = u64::from(self.primary.width)
            .checked_mul(u64::from(self.primary.height))
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or_else(|| CoreError::invalid("image graph scratch", "canvas overflowed"))?;
        let retained_images = u64::try_from(self.composites.len())
            .ok()
            .and_then(|stages| stages.checked_add(1))
            .ok_or_else(|| CoreError::invalid("image graph scratch", "stage count overflowed"))?;
        if canvas_bytes
            .checked_mul(retained_images)
            .is_none_or(|bytes| bytes > MAX_IMAGE_GRAPH_SCRATCH_BYTES)
        {
            return Err(CoreError::invalid(
                "image graph scratch",
                format!("exceeds {MAX_IMAGE_GRAPH_SCRATCH_BYTES} bytes"),
            ));
        }
        self.validate_checkpoint()?;
        for (index, stage) in self.composites.iter().enumerate() {
            if usize::from(stage.stage) != index {
                return Err(CoreError::invalid(
                    "image composite stages",
                    "stage indices must be contiguous and declared in order",
                ));
            }
            match stage.operation {
                ImageCompositeOperation::MaskBlend {
                    base,
                    overlay,
                    mask_slot,
                } => {
                    self.validate_rgb_source(base, index)?;
                    self.validate_rgb_source(overlay, index)?;
                    self.require_input_layout(mask_slot, ImageLayoutKind::Gray8)?;
                }
            }
        }
        self.validate_outputs()
    }

    fn validate_checkpoint(&self) -> Result<(), CoreError> {
        let Some(schedule) = self.primary.schedule.as_ref() else {
            if self.checkpoint != ImageCheckpointPlan::default() {
                return Err(CoreError::invalid(
                    "image checkpoint plan",
                    "direct VAE operations cannot restore or capture a diffusion checkpoint",
                ));
            }
            return Ok(());
        };
        if let Some(slot) = self.checkpoint.restore_from {
            let input = self
                .primary
                .inputs
                .iter()
                .find(|input| input.slot == slot)
                .ok_or_else(|| {
                    CoreError::invalid("image checkpoint restore", "does not name an input binding")
                })?;
            if input.role != ImageBufferRole::Checkpoint {
                return Err(CoreError::invalid(
                    "image checkpoint restore",
                    "does not name a checkpoint input",
                ));
            }
        }
        let checkpoint_inputs = self
            .primary
            .inputs
            .iter()
            .filter(|input| input.role == ImageBufferRole::Checkpoint)
            .count();
        if checkpoint_inputs != usize::from(self.checkpoint.restore_from.is_some()) {
            return Err(CoreError::invalid(
                "image checkpoint restore",
                "checkpoint inputs must be consumed exactly once",
            ));
        }
        if self
            .checkpoint
            .capture_after_step
            .is_some_and(|step| usize::try_from(step).map_or(true, |step| step >= schedule.steps()))
        {
            return Err(CoreError::invalid(
                "image checkpoint capture",
                "step is outside the primary schedule",
            ));
        }
        Ok(())
    }

    fn validate_outputs(&self) -> Result<(), CoreError> {
        let mut allocations = HashSet::new();
        let mut checkpoint_routes = 0_usize;
        for (index, route) in self.outputs.iter().enumerate() {
            route.buffer.validate()?;
            if !allocations.insert(route.buffer.identity.clone()) {
                return Err(CoreError::invalid(
                    "image output routes",
                    "allocation identities must be unique",
                ));
            }
            match route.source {
                ImageOutputSource::Image { source } => {
                    self.validate_rgb_source(source, self.composites.len())?;
                    validate_tight_rgb_output(
                        &route.layout,
                        &route.buffer,
                        self.primary.width,
                        self.primary.height,
                    )?;
                }
                ImageOutputSource::Checkpoint => {
                    checkpoint_routes += 1;
                    if !matches!(route.layout, ImageBufferLayout::Opaque)
                        || index + 1 != self.outputs.len()
                    {
                        return Err(CoreError::invalid(
                            "image checkpoint output",
                            "must be the final route and use an opaque buffer layout",
                        ));
                    }
                }
            }
        }
        let expected_checkpoint_routes = usize::from(self.checkpoint.capture_after_step.is_some());
        if checkpoint_routes != expected_checkpoint_routes {
            return Err(CoreError::invalid(
                "image checkpoint output",
                "must be routed exactly once when checkpoint capture is requested",
            ));
        }
        Ok(())
    }

    fn validate_rgb_source(
        &self,
        source: ImageValueSource,
        current_stage: usize,
    ) -> Result<(), CoreError> {
        match source {
            ImageValueSource::Primary if self.primary.output_format == ImageOutputFormat::Rgb8 => {
                Ok(())
            }
            ImageValueSource::Primary => Err(CoreError::invalid(
                "image graph primary source",
                "must produce RGB8 bytes",
            )),
            ImageValueSource::Input { slot } => {
                self.require_input_layout(slot, ImageLayoutKind::Rgb8)
            }
            ImageValueSource::Composite { stage }
                if usize::from(stage) < current_stage
                    && usize::from(stage) < self.composites.len() =>
            {
                Ok(())
            }
            ImageValueSource::Composite { .. } => Err(CoreError::invalid(
                "image graph composite source",
                "must name an earlier stage",
            )),
        }
    }

    fn require_input_layout(&self, slot: u16, kind: ImageLayoutKind) -> Result<(), CoreError> {
        let input = self
            .primary
            .inputs
            .iter()
            .find(|input| input.slot == slot)
            .ok_or_else(|| CoreError::invalid("image graph input", "slot is not bound"))?;
        let (width, height, row_stride, actual) = match input.layout {
            ImageBufferLayout::Rgb8 {
                width,
                height,
                row_stride,
            } => (width, height, row_stride, ImageLayoutKind::Rgb8),
            ImageBufferLayout::Gray8 {
                width,
                height,
                row_stride,
            } => (width, height, row_stride, ImageLayoutKind::Gray8),
            _ => {
                return Err(CoreError::invalid(
                    "image graph input",
                    "must use the required byte layout",
                ));
            }
        };
        let channels = match kind {
            ImageLayoutKind::Rgb8 => 3_u64,
            ImageLayoutKind::Gray8 => 1_u64,
        };
        let expected_stride = u64::from(self.primary.width)
            .checked_mul(channels)
            .ok_or_else(|| CoreError::invalid("image graph stride", "overflowed"))?;
        if actual != kind
            || width != self.primary.width
            || height != self.primary.height
            || row_stride != expected_stride
        {
            return Err(CoreError::invalid(
                "image graph input",
                "must be tightly packed at the primary canvas geometry",
            ));
        }
        Ok(())
    }

    /// Returns the identity of the complete version-three plan.
    ///
    /// # Errors
    ///
    /// Returns a validation or deterministic serialization error.
    pub fn digest(&self) -> Result<Digest, CoreError> {
        self.validate()?;
        Digest::of_serializable("image-execution-plan-v3", self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ImageLayoutKind {
    Rgb8,
    Gray8,
}

/// Exact output write accounting.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageOutputReceiptV3 {
    /// Zero-based route index.
    pub route: u16,
    /// Caller-owned allocation identity.
    pub allocation: Digest,
    /// Exact initialized output bytes.
    pub content: Digest,
    /// Initialized prefix length.
    pub bytes_written: u64,
}

/// Deterministic whole-plan receipt for [`ImageExecutionPlanV3`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageExecutionReceiptV3 {
    /// Exact version-three plan identity.
    pub plan: Digest,
    /// Exact backend build/runtime identity.
    pub backend: Digest,
    /// Exact primary profile identity.
    pub profile: Digest,
    /// Session epoch used by the primary operation.
    pub session_epoch: u64,
    /// Completed diffusion transitions.
    pub completed_steps: u32,
    /// Terminal execution boundary.
    pub terminal: ImageTerminal,
    /// Adapter-specific primary receipt identity, absent for pre-start
    /// cancellation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primary: Option<Digest>,
    /// Checkpoint lineage in deterministic order: restored receipt, captured
    /// receipt, then captured envelope bytes, omitting absent mechanics.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub checkpoints: Vec<Digest>,
    /// Completed deterministic composites in order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub composites: Vec<ImageCompositeReceipt>,
    /// Completed output writes in route order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outputs: Vec<ImageOutputReceiptV3>,
    /// Observation result identities in request order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub observations: Vec<Digest>,
    /// Request-scope cleanup outcome.
    pub cleanup: ImageCleanupDisposition,
}

impl ImageExecutionReceiptV3 {
    /// Validates plan lineage, terminal position, graph prefix, output writes,
    /// and cleanup disposition.
    ///
    /// # Errors
    ///
    /// Returns the first inconsistent receipt field.
    pub fn validate_for(&self, plan: &ImageExecutionPlanV3) -> Result<(), CoreError> {
        plan.validate()?;
        if self.plan != plan.digest()? || self.profile != plan.primary.profile {
            return Err(CoreError::invalid(
                "image execution receipt v2",
                "plan or profile identity differs",
            ));
        }
        let step_count = plan
            .primary
            .schedule
            .as_ref()
            .map_or(0, crate::DiffusionSchedule::steps);
        let completed = usize::try_from(self.completed_steps).map_err(|_| {
            CoreError::invalid("image execution receipt v2", "step count exceeds usize")
        })?;
        self.validate_collections(plan, completed, step_count)?;
        self.validate_terminal(plan, completed, step_count)?;
        self.validate_cleanup(plan)
    }

    fn validate_collections(
        &self,
        plan: &ImageExecutionPlanV3,
        completed: usize,
        step_count: usize,
    ) -> Result<(), CoreError> {
        if completed > step_count
            || self.checkpoints.len() > maximum_checkpoint_lineage(plan)
            || self.composites.len() > plan.composites.len()
            || self.outputs.len() > plan.outputs.len()
            || self.observations.len() > plan.primary.observations.len()
        {
            return Err(CoreError::invalid(
                "image execution receipt v2",
                "a completed collection or step count exceeds the plan",
            ));
        }
        for (index, composite) in self.composites.iter().enumerate() {
            if usize::from(composite.stage) != index {
                return Err(CoreError::invalid(
                    "image composite receipt",
                    "stages must form an ordered completed prefix",
                ));
            }
        }
        for (index, output) in self.outputs.iter().enumerate() {
            let route = plan.outputs.get(index).ok_or_else(|| {
                CoreError::invalid("image output receipt", "route is outside the plan")
            })?;
            if usize::from(output.route) != index
                || output.allocation != route.buffer.identity
                || output.bytes_written == 0
                || output.bytes_written > route.buffer.byte_length
                || (matches!(route.source, ImageOutputSource::Image { .. })
                    && output.bytes_written != route.buffer.byte_length)
            {
                return Err(CoreError::invalid(
                    "image output receipt",
                    "route, allocation, or initialized length is inconsistent",
                ));
            }
        }
        Ok(())
    }

    fn validate_terminal(
        &self,
        plan: &ImageExecutionPlanV3,
        completed: usize,
        step_count: usize,
    ) -> Result<(), CoreError> {
        match self.terminal {
            ImageTerminal::Completed
                if completed != step_count
                    || self.primary.is_none()
                    || self.checkpoints.len() != maximum_checkpoint_lineage(plan)
                    || self.composites.len() != plan.composites.len()
                    || self.outputs.len() != plan.outputs.len()
                    || self.observations.len() != plan.primary.observations.len() =>
            {
                return Err(CoreError::invalid(
                    "image execution receipt v2",
                    "completed terminal requires the whole graph and every route",
                ));
            }
            ImageTerminal::CancelledBeforeStart
                if completed != 0
                    || self.primary.is_some()
                    || !self.checkpoints.is_empty()
                    || !self.composites.is_empty()
                    || !self.outputs.is_empty()
                    || !self.observations.is_empty()
                    || self.cleanup != ImageCleanupDisposition::NotRequired =>
            {
                return Err(CoreError::invalid(
                    "image execution receipt v2",
                    "pre-start cancellation contains execution side effects",
                ));
            }
            ImageTerminal::CancelledAfterStep { step }
                if usize::try_from(step)
                    .ok()
                    .and_then(|step| step.checked_add(1))
                    != Some(completed)
                    || self.primary.is_none()
                    || self.observations.len() != plan.primary.observations.len() =>
            {
                return Err(CoreError::invalid(
                    "image execution receipt v2",
                    "cancelled boundary does not match completed steps",
                ));
            }
            _ => {}
        }
        if let ImageTerminal::CancelledAfterStep { step } = self.terminal {
            let capture_reached = plan
                .checkpoint
                .capture_after_step
                .is_some_and(|capture| capture <= step);
            let minimum = usize::from(capture_reached) * 2;
            let maximum = minimum + usize::from(plan.checkpoint.restore_from.is_some());
            if self.checkpoints.len() < minimum || self.checkpoints.len() > maximum {
                return Err(CoreError::invalid(
                    "image checkpoint receipt",
                    "lineage does not match the reached cancellation boundary",
                ));
            }
        }
        Ok(())
    }

    fn validate_cleanup(&self, plan: &ImageExecutionPlanV3) -> Result<(), CoreError> {
        if self.terminal != ImageTerminal::CancelledBeforeStart {
            match (plan.cleanup, &self.cleanup) {
                (ImageCleanupPolicy::RetainSession, ImageCleanupDisposition::Retained)
                | (ImageCleanupPolicy::ClearSession, ImageCleanupDisposition::Confirmed { .. }) => {
                }
                _ => {
                    return Err(CoreError::invalid(
                        "image execution receipt v2",
                        "cleanup disposition differs from the plan",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Returns the identity of this exact whole-plan receipt.
    ///
    /// # Errors
    ///
    /// Returns a validation or deterministic serialization error.
    pub fn digest_for(&self, plan: &ImageExecutionPlanV3) -> Result<Digest, CoreError> {
        self.validate_for(plan)?;
        Digest::of_serializable("image-execution-receipt-v3", self)
    }
}

fn maximum_checkpoint_lineage(plan: &ImageExecutionPlanV3) -> usize {
    usize::from(plan.checkpoint.restore_from.is_some())
        + usize::from(plan.checkpoint.capture_after_step.is_some()) * 2
}

fn validate_tight_rgb_output(
    layout: &ImageBufferLayout,
    buffer: &BufferSpec,
    width: u32,
    height: u32,
) -> Result<(), CoreError> {
    let expected_stride = u64::from(width)
        .checked_mul(3)
        .ok_or_else(|| CoreError::invalid("image output stride", "overflowed"))?;
    match layout {
        ImageBufferLayout::Rgb8 {
            width: output_width,
            height: output_height,
            row_stride,
        } if *output_width == width
            && *output_height == height
            && *row_stride == expected_stride =>
        {
            let expected = expected_stride
                .checked_mul(u64::from(height))
                .ok_or_else(|| CoreError::invalid("image output length", "overflowed"))?;
            if buffer.byte_length != expected {
                return Err(CoreError::invalid(
                    "image output allocation",
                    "length differs from the tightly packed RGB8 canvas",
                ));
            }
            Ok(())
        }
        _ => Err(CoreError::invalid(
            "image output route",
            "must be tightly packed RGB8 at the primary canvas geometry",
        )),
    }
}

#[cfg(test)]
mod tests {
    use logit_loom_executor::BufferSpec;

    use super::*;
    use crate::{
        DiffusionSchedule, ImageBufferBinding, ImageOperation, ObservationRequest, SeedSelection,
    };

    fn buffer(domain: &str, bytes: u64, media_type: &str) -> BufferSpec {
        BufferSpec::new(
            Digest::of_bytes(domain, domain.as_bytes()),
            bytes,
            media_type,
        )
        .unwrap()
    }

    fn primary() -> ImageExecutionPlan {
        let width = 2;
        let height = 1;
        let prompt = ImageBufferBinding {
            slot: 0,
            role: ImageBufferRole::PositiveConditioning,
            buffer: buffer("prompt", 4, "text/plain"),
            layout: ImageBufferLayout::Utf8,
        };
        let overlay = ImageBufferBinding {
            slot: 1,
            role: ImageBufferRole::ReferenceImage,
            buffer: buffer("overlay", 6, "image/rgb"),
            layout: ImageBufferLayout::Rgb8 {
                width,
                height,
                row_stride: 6,
            },
        };
        let mask = ImageBufferBinding {
            slot: 2,
            role: ImageBufferRole::Mask,
            buffer: buffer("mask", 2, "image/gray"),
            layout: ImageBufferLayout::Gray8 {
                width,
                height,
                row_stride: 2,
            },
        };
        ImageExecutionPlan {
            profile: Digest::of_bytes("profile", b"one"),
            load: Digest::of_bytes("load", b"one"),
            operation: ImageOperation::TextToImage,
            width,
            height,
            output_format: ImageOutputFormat::Rgb8,
            seed: SeedSelection::Fixed { seed: 7 },
            rng: Digest::of_bytes("rng", b"one"),
            placement: Digest::of_bytes("placement", b"one"),
            schedule: Some(
                DiffusionSchedule::new(Digest::of_bytes("schedule", b"one"), vec![1.0, 0.0])
                    .unwrap(),
            ),
            guidance_scale_bits: 1.0_f32.to_bits(),
            strength_bits: 1.0_f32.to_bits(),
            inputs: vec![prompt, overlay, mask],
            loras: Vec::new(),
            operators: Vec::new(),
            observations: Vec::<ObservationRequest>::new(),
        }
    }

    fn graph() -> ImageExecutionPlanV3 {
        let primary = primary();
        ImageExecutionPlanV3 {
            primary,
            checkpoint: ImageCheckpointPlan::default(),
            composites: vec![ImageCompositeStage {
                stage: 0,
                operation: ImageCompositeOperation::MaskBlend {
                    base: ImageValueSource::Primary,
                    overlay: ImageValueSource::Input { slot: 1 },
                    mask_slot: 2,
                },
            }],
            outputs: vec![ImageOutputRoute {
                source: ImageOutputSource::Image {
                    source: ImageValueSource::Composite { stage: 0 },
                },
                buffer: buffer("output", 6, "image/rgb"),
                layout: ImageBufferLayout::Rgb8 {
                    width: 2,
                    height: 1,
                    row_stride: 6,
                },
            }],
            cleanup: ImageCleanupPolicy::RetainSession,
        }
    }

    #[test]
    fn graph_requires_earlier_composites_and_exact_routes() {
        assert!(graph().validate().is_ok());
        let mut invalid = graph();
        invalid.composites[0].operation = ImageCompositeOperation::MaskBlend {
            base: ImageValueSource::Composite { stage: 0 },
            overlay: ImageValueSource::Input { slot: 1 },
            mask_slot: 2,
        };
        assert!(invalid.validate().is_err());

        let mut missing_checkpoint = graph();
        missing_checkpoint.checkpoint.capture_after_step = Some(0);
        assert!(missing_checkpoint.validate().is_err());
    }

    #[test]
    fn version_two_round_trip_preserves_new_identity_domain() {
        let plan = graph();
        let encoded = serde_json::to_vec(&plan).unwrap();
        let decoded: ImageExecutionPlanV3 = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, plan);
        assert_ne!(plan.digest().unwrap(), plan.primary.digest().unwrap());

        let mut value = serde_json::to_value(&plan).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("future-field".to_owned(), serde_json::Value::Bool(true));
        assert!(serde_json::from_value::<ImageExecutionPlanV3>(value).is_err());
    }

    #[test]
    fn checkpoint_capture_uses_one_final_output_route() {
        let mut plan = graph();
        plan.checkpoint.capture_after_step = Some(0);
        plan.outputs.push(ImageOutputRoute {
            source: ImageOutputSource::Checkpoint,
            buffer: buffer("checkpoint-output", 1_024, "application/octet-stream"),
            layout: ImageBufferLayout::Opaque,
        });
        assert!(plan.validate().is_ok());
        plan.outputs.swap(0, 1);
        assert!(plan.validate().is_err());
    }

    #[test]
    fn mask_blend_is_exact_at_endpoints_and_midpoint() {
        let base = [0, 10, 20, 100, 110, 120, 20, 40, 60];
        let overlay = [200, 210, 220, 50, 60, 70, 120, 140, 160];
        let mask = [0, 255, 128];
        let mut output = [9_u8; 9];
        let receipt = crate::mask_blend_rgb8(&base, &overlay, &mask, &mut output).unwrap();
        assert_eq!(&output[..3], &base[..3]);
        assert_eq!(&output[3..6], &overlay[3..6]);
        assert_eq!(&output[6..], &[70, 90, 110]);
        assert_eq!(
            receipt.output,
            Digest::of_bytes("image-composite-rgb8-output-v1", &output)
        );
    }

    #[test]
    fn mask_blend_rejects_before_write() {
        let mut output = [9_u8; 3];
        assert!(crate::mask_blend_rgb8(&[0, 0, 0], &[1, 1], &[255], &mut output).is_err());
        assert_eq!(output, [9; 3]);
    }

    #[test]
    fn prestart_cancellation_receipt_has_no_side_effects() {
        let plan = graph();
        let mut receipt = ImageExecutionReceiptV3 {
            plan: plan.digest().unwrap(),
            backend: Digest::of_bytes("backend", b"one"),
            profile: plan.primary.profile.clone(),
            session_epoch: 4,
            completed_steps: 0,
            terminal: ImageTerminal::CancelledBeforeStart,
            primary: None,
            checkpoints: Vec::new(),
            composites: Vec::new(),
            outputs: Vec::new(),
            observations: Vec::new(),
            cleanup: ImageCleanupDisposition::NotRequired,
        };
        assert!(receipt.digest_for(&plan).is_ok());
        receipt
            .checkpoints
            .push(Digest::of_bytes("checkpoint", b"unexpected"));
        assert!(receipt.digest_for(&plan).is_err());
    }

    #[test]
    fn reached_checkpoint_capture_requires_receipt_and_envelope_lineage() {
        let mut plan = graph();
        plan.checkpoint.capture_after_step = Some(0);
        plan.outputs.push(ImageOutputRoute {
            source: ImageOutputSource::Checkpoint,
            buffer: buffer("checkpoint-output", 1_024, "application/octet-stream"),
            layout: ImageBufferLayout::Opaque,
        });
        plan.validate().unwrap();
        let receipt = ImageExecutionReceiptV3 {
            plan: plan.digest().unwrap(),
            backend: Digest::of_bytes("backend", b"one"),
            profile: plan.primary.profile.clone(),
            session_epoch: 4,
            completed_steps: 1,
            terminal: ImageTerminal::CancelledAfterStep { step: 0 },
            primary: Some(Digest::of_bytes("primary", b"one")),
            checkpoints: vec![Digest::of_bytes("checkpoint", b"receipt-only")],
            composites: Vec::new(),
            outputs: Vec::new(),
            observations: Vec::new(),
            cleanup: ImageCleanupDisposition::Retained,
        };
        assert!(receipt.validate_for(&plan).is_err());
    }

    #[test]
    fn cancellation_before_capture_can_leave_final_route_uninitialized() {
        let mut plan = graph();
        plan.primary.schedule = Some(
            DiffusionSchedule::new(
                Digest::of_bytes("schedule", b"two-steps"),
                vec![1.0, 0.5, 0.0],
            )
            .unwrap(),
        );
        plan.checkpoint.capture_after_step = Some(1);
        plan.outputs.push(ImageOutputRoute {
            source: ImageOutputSource::Checkpoint,
            buffer: buffer("checkpoint-output", 1_024, "application/octet-stream"),
            layout: ImageBufferLayout::Opaque,
        });
        let receipt = ImageExecutionReceiptV3 {
            plan: plan.digest().unwrap(),
            backend: Digest::of_bytes("backend", b"one"),
            profile: plan.primary.profile.clone(),
            session_epoch: 4,
            completed_steps: 1,
            terminal: ImageTerminal::CancelledAfterStep { step: 0 },
            primary: Some(Digest::of_bytes("primary", b"one")),
            checkpoints: Vec::new(),
            composites: vec![ImageCompositeReceipt {
                stage: 0,
                base: Digest::of_bytes("base", b"one"),
                overlay: Digest::of_bytes("overlay", b"one"),
                mask: Digest::of_bytes("mask", b"one"),
                output: Digest::of_bytes("composite", b"one"),
            }],
            outputs: vec![ImageOutputReceiptV3 {
                route: 0,
                allocation: plan.outputs[0].buffer.identity.clone(),
                content: Digest::of_bytes("content", b"image"),
                bytes_written: 6,
            }],
            observations: Vec::new(),
            cleanup: ImageCleanupDisposition::Retained,
        };
        assert!(receipt.validate_for(&plan).is_ok());
    }

    #[test]
    fn image_routes_require_the_exact_initialized_length() {
        let plan = graph();
        let receipt = ImageExecutionReceiptV3 {
            plan: plan.digest().unwrap(),
            backend: Digest::of_bytes("backend", b"one"),
            profile: plan.primary.profile.clone(),
            session_epoch: 4,
            completed_steps: 1,
            terminal: ImageTerminal::Completed,
            primary: Some(Digest::of_bytes("primary", b"one")),
            checkpoints: Vec::new(),
            composites: vec![ImageCompositeReceipt {
                stage: 0,
                base: Digest::of_bytes("base", b"one"),
                overlay: Digest::of_bytes("overlay", b"one"),
                mask: Digest::of_bytes("mask", b"one"),
                output: Digest::of_bytes("composite", b"one"),
            }],
            outputs: vec![ImageOutputReceiptV3 {
                route: 0,
                allocation: plan.outputs[0].buffer.identity.clone(),
                content: Digest::of_bytes("content", b"short"),
                bytes_written: 5,
            }],
            observations: Vec::new(),
            cleanup: ImageCleanupDisposition::Retained,
        };
        assert!(receipt.validate_for(&plan).is_err());
    }
}

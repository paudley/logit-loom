// SPDX-License-Identifier: MIT OR Apache-2.0

//! Serializable diffusion-state contracts and receipts.

use std::collections::{BTreeMap, HashSet};

use logit_loom_core::{CoreError, Digest, MAX_RETAINED_FAILURE_BYTES};
use serde::{Deserialize, Serialize};

/// Maximum tensor rank accepted at the public boundary.
pub const MAX_TENSOR_DIMENSIONS: usize = 8;
/// Maximum contiguous `f32` elements copied transactionally.
pub const MAX_TENSOR_ELEMENTS: u64 = 16 * 1024 * 1024;
/// Maximum scheduler steps in one plan.
pub const MAX_DIFFUSION_STEPS: usize = 4_096;
/// Maximum identified model components in one plan.
pub const MAX_COMPONENTS: usize = 32;
/// Maximum UTF-8 bytes in one device label.
pub const MAX_DEVICE_LABEL_BYTES: usize = 256;
/// Maximum stages in one intervention pipeline.
pub const MAX_INTERVENTION_STAGES: usize = 32;

/// Tensor scalar representation at the intervention boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorDType {
    /// IEEE 754 binary32 values exposed directly to Rust.
    F32,
    /// IEEE 754 binary16 native state, converted exactly at the adapter boundary.
    F16,
    /// Brain floating-point native state, converted exactly at the adapter boundary.
    Bf16,
}

/// Contiguous element ordering at the intervention boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorLayout {
    /// Dimension zero is contiguous and changes fastest.
    DimensionZeroFastest,
}

/// Exact shape, scalar, layout, and device of one diffusion state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorSpec {
    /// Positive dimensions in native order.
    pub shape: Vec<u64>,
    /// Native scalar representation.
    pub dtype: TensorDType,
    /// Contiguous layout exposed to callbacks.
    pub layout: TensorLayout,
    /// Adapter-reported device identity.
    pub device: String,
}

impl TensorSpec {
    /// Constructs and validates a tensor contract.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid rank, dimensions, element count, or device.
    pub fn new(
        shape: Vec<u64>,
        dtype: TensorDType,
        layout: TensorLayout,
        device: impl Into<String>,
    ) -> Result<Self, CoreError> {
        let specification = Self {
            shape,
            dtype,
            layout,
            device: device.into(),
        };
        specification.validate()?;
        Ok(specification)
    }

    /// Validates rank, dimensions, element count, and device identity.
    ///
    /// # Errors
    ///
    /// Returns the first invalid tensor invariant.
    pub fn validate(&self) -> Result<(), CoreError> {
        if !(1..=MAX_TENSOR_DIMENSIONS).contains(&self.shape.len()) {
            return Err(CoreError::invalid(
                "diffusion tensor shape",
                format!("requires 1..={MAX_TENSOR_DIMENSIONS} dimensions"),
            ));
        }
        if self.shape.contains(&0) {
            return Err(CoreError::invalid(
                "diffusion tensor shape",
                "dimensions must be positive",
            ));
        }
        let elements = self.elements()?;
        if elements > MAX_TENSOR_ELEMENTS {
            return Err(CoreError::invalid(
                "diffusion tensor shape",
                format!("exceeds the {MAX_TENSOR_ELEMENTS}-element bound"),
            ));
        }
        if self.device.is_empty()
            || self.device.len() > MAX_DEVICE_LABEL_BYTES
            || self.device.contains('\0')
        {
            return Err(CoreError::invalid(
                "diffusion tensor device",
                "must be a nonempty bounded label without NUL",
            ));
        }
        Ok(())
    }

    /// Returns the checked product of all dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error when the shape is empty or its product overflows.
    pub fn elements(&self) -> Result<u64, CoreError> {
        if self.shape.is_empty() {
            return Err(CoreError::Empty {
                field: "diffusion tensor shape",
            });
        }
        self.shape.iter().try_fold(1_u64, |product, dimension| {
            product.checked_mul(*dimension).ok_or_else(|| {
                CoreError::invalid("diffusion tensor shape", "element count overflowed")
            })
        })
    }

    /// Returns the content identity of this exact tensor contract.
    ///
    /// # Errors
    ///
    /// Returns a validation or serialization error.
    pub fn digest(&self) -> Result<Digest, CoreError> {
        self.validate()?;
        Digest::of_serializable("diffusion-tensor-spec-v1", self)
    }
}

/// Exact scheduler implementation and descending sigma sequence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiffusionSchedule {
    /// Caller-defined scheduler implementation identity.
    pub implementation: Digest,
    /// Sigma at each state boundary; `len() - 1` is the step count.
    pub sigmas: Vec<f32>,
}

impl DiffusionSchedule {
    /// Constructs and validates a scheduler contract.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty, oversized, non-finite, negative, or
    /// increasing sigma sequence.
    pub fn new(implementation: Digest, sigmas: Vec<f32>) -> Result<Self, CoreError> {
        let schedule = Self {
            implementation,
            sigmas,
        };
        schedule.validate()?;
        Ok(schedule)
    }

    /// Validates bounds and descending finite sigma ordering.
    ///
    /// # Errors
    ///
    /// Returns the first invalid schedule invariant.
    pub fn validate(&self) -> Result<(), CoreError> {
        if !(2..=MAX_DIFFUSION_STEPS + 1).contains(&self.sigmas.len()) {
            return Err(CoreError::invalid(
                "diffusion schedule",
                format!("requires 1..={MAX_DIFFUSION_STEPS} steps"),
            ));
        }
        if self
            .sigmas
            .iter()
            .any(|sigma| !sigma.is_finite() || *sigma < 0.0)
        {
            return Err(CoreError::invalid(
                "diffusion schedule",
                "sigmas must be finite and non-negative",
            ));
        }
        if self.sigmas.windows(2).any(|pair| pair[0] < pair[1]) {
            return Err(CoreError::invalid(
                "diffusion schedule",
                "sigmas must be non-increasing",
            ));
        }
        Ok(())
    }

    /// Returns the number of state transitions.
    pub fn steps(&self) -> usize {
        self.sigmas.len().saturating_sub(1)
    }

    /// Returns the content identity of this exact schedule.
    ///
    /// # Errors
    ///
    /// Returns a validation or serialization error.
    pub fn digest(&self) -> Result<Digest, CoreError> {
        self.validate()?;
        Digest::of_serializable("diffusion-schedule-v1", self)
    }
}

/// Exact model, conditioning, random-state, tensor, and schedule contract.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiffusionPlan {
    /// Named component artifact identities in deterministic key order.
    pub components: BTreeMap<String, Digest>,
    /// Identity of exact tokenization and conditioning output.
    pub conditioning: Digest,
    /// Random-number-generator implementation identity.
    pub rng: Digest,
    /// Exact random seed interpreted by the identified RNG.
    pub seed: u64,
    /// Iterative state exposed at step boundaries.
    pub tensor: TensorSpec,
    /// Exact schedule.
    pub schedule: DiffusionSchedule,
}

impl DiffusionPlan {
    /// Constructs and validates an exact diffusion execution plan.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid components, tensor, or schedule.
    pub fn new(
        components: BTreeMap<String, Digest>,
        conditioning: Digest,
        rng: Digest,
        seed: u64,
        tensor: TensorSpec,
        schedule: DiffusionSchedule,
    ) -> Result<Self, CoreError> {
        let plan = Self {
            components,
            conditioning,
            rng,
            seed,
            tensor,
            schedule,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// Validates component names and nested contracts.
    ///
    /// # Errors
    ///
    /// Returns the first invalid plan invariant.
    pub fn validate(&self) -> Result<(), CoreError> {
        if self.components.is_empty() || self.components.len() > MAX_COMPONENTS {
            return Err(CoreError::invalid(
                "diffusion components",
                format!("requires 1..={MAX_COMPONENTS} entries"),
            ));
        }
        for name in self.components.keys() {
            if name.is_empty()
                || name.len() > 64
                || !name.bytes().all(|byte| {
                    byte.is_ascii_lowercase()
                        || byte.is_ascii_digit()
                        || matches!(byte, b'-' | b'_' | b'.')
                })
            {
                return Err(CoreError::invalid(
                    "diffusion component name",
                    "must be a bounded lowercase identifier",
                ));
            }
        }
        self.tensor.validate()?;
        self.schedule.validate()
    }

    /// Returns the identity of this exact execution plan.
    ///
    /// # Errors
    ///
    /// Returns a validation or serialization error.
    pub fn digest(&self) -> Result<Digest, CoreError> {
        self.validate()?;
        Digest::of_serializable("diffusion-plan-v1", self)
    }
}

/// Exact post-step state boundary presented to Rust.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StepContext {
    /// Exact execution-plan identity.
    pub plan: Digest,
    /// Zero-based transition that just completed.
    pub step_index: u32,
    /// Sigma before the transition.
    pub sigma_from: f32,
    /// Sigma after the transition.
    pub sigma_to: f32,
    /// Tensor contract at this boundary.
    pub tensor: TensorSpec,
}

impl StepContext {
    /// Creates the exact post-step context for one plan transition.
    ///
    /// # Errors
    ///
    /// Returns an error when the plan or index is invalid.
    pub fn for_plan(plan: &DiffusionPlan, step_index: usize) -> Result<Self, CoreError> {
        plan.validate()?;
        if step_index >= plan.schedule.steps() {
            return Err(CoreError::invalid(
                "diffusion step",
                "index is outside the schedule",
            ));
        }
        let step_index_u32 = u32::try_from(step_index)
            .map_err(|_| CoreError::invalid("diffusion step", "index exceeds u32"))?;
        Ok(Self {
            plan: plan.digest()?,
            step_index: step_index_u32,
            sigma_from: plan.schedule.sigmas[step_index],
            sigma_to: plan.schedule.sigmas[step_index + 1],
            tensor: plan.tensor.clone(),
        })
    }

    /// Validates this boundary against its complete execution plan.
    ///
    /// # Errors
    ///
    /// Returns an error for an identity, index, sigma, or tensor mismatch.
    pub fn validate_for(&self, plan: &DiffusionPlan) -> Result<(), CoreError> {
        let expected = Self::for_plan(
            plan,
            usize::try_from(self.step_index)
                .map_err(|_| CoreError::invalid("diffusion step", "index exceeds usize"))?,
        )?;
        if *self != expected {
            return Err(CoreError::invalid(
                "diffusion step",
                "context does not match the execution plan",
            ));
        }
        Ok(())
    }
}

/// Exact contract for one ordered state intervention.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InterventionSpec {
    /// Caller-defined implementation and configuration identity.
    pub implementation: Digest,
    /// Exact tensor contract accepted by the stage.
    pub tensor: Digest,
    /// Maximum invocations in one pipeline run.
    pub max_invocations: u32,
}

impl InterventionSpec {
    /// Validates a bounded stage contract.
    ///
    /// # Errors
    ///
    /// Returns an error when the invocation bound is zero.
    pub fn validate(&self) -> Result<(), CoreError> {
        if self.max_invocations == 0 {
            return Err(CoreError::invalid(
                "diffusion intervention",
                "maximum invocations must be positive",
            ));
        }
        Ok(())
    }

    /// Returns the identity of this exact stage contract.
    ///
    /// # Errors
    ///
    /// Returns a validation or serialization error.
    pub fn digest(&self) -> Result<Digest, CoreError> {
        self.validate()?;
        Digest::of_serializable("diffusion-intervention-spec-v1", self)
    }
}

/// Ordered diffusion intervention contracts.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineSpec {
    /// Stages in exact execution order.
    pub stages: Vec<InterventionSpec>,
}

impl PipelineSpec {
    /// Validates stage bounds and their common tensor identity.
    ///
    /// # Errors
    ///
    /// Returns an error unless one to 32 compatible stages are present.
    pub fn validate(&self) -> Result<(), CoreError> {
        if !(1..=MAX_INTERVENTION_STAGES).contains(&self.stages.len()) {
            return Err(CoreError::invalid(
                "diffusion pipeline stages",
                format!("requires 1..={MAX_INTERVENTION_STAGES} entries"),
            ));
        }
        let tensor = &self.stages[0].tensor;
        for stage in &self.stages {
            stage.validate()?;
            if &stage.tensor != tensor {
                return Err(CoreError::invalid(
                    "diffusion pipeline stages",
                    "all stages must accept the same tensor contract",
                ));
            }
        }
        Ok(())
    }

    /// Returns the identity of this exact ordered pipeline.
    ///
    /// # Errors
    ///
    /// Returns a validation or serialization error.
    pub fn digest(&self) -> Result<Digest, CoreError> {
        self.validate()?;
        Digest::of_serializable("diffusion-pipeline-spec-v1", self)
    }
}

/// Bounded callback error or contained panic.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InterventionFailure {
    /// Whether the callback unwound.
    pub panicked: bool,
    /// Bounded human-readable detail.
    pub message: String,
}

impl InterventionFailure {
    /// Constructs a UTF-8-safe bounded failure.
    pub fn new(panicked: bool, message: impl Into<String>) -> Self {
        let mut message = message.into();
        if message.len() > MAX_RETAINED_FAILURE_BYTES {
            let mut end = MAX_RETAINED_FAILURE_BYTES;
            while !message.is_char_boundary(end) {
                end -= 1;
            }
            message.truncate(end);
        }
        Self { panicked, message }
    }
}

/// Mechanical accounting for one intervention stage.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageReceipt {
    /// Exact stage contract.
    pub specification: InterventionSpec,
    /// Reset calls.
    pub resets: u32,
    /// Apply calls, including a failed call.
    pub invocations: u32,
    /// State elements presented across calls.
    pub elements_seen: u64,
    /// Elements whose bit pattern changed.
    pub elements_changed: u64,
    /// First contained failure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<InterventionFailure>,
}

/// Aggregate transactional pipeline accounting.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineReceipt {
    /// Exact ordered stage contract.
    pub specification: PipelineSpec,
    /// Pipeline identity.
    pub pipeline: Digest,
    /// Begin/reset operations.
    pub begins: u32,
    /// Complete committed invocations.
    pub invocations: u32,
    /// Elements copied into the transactional boundary.
    pub elements_copied: u64,
    /// Elements committed after every stage succeeded.
    pub elements_committed: u64,
    /// First failed stage index.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failed_stage: Option<u32>,
    /// Per-stage accounting in execution order.
    pub stages: Vec<StageReceipt>,
}

impl PipelineReceipt {
    /// Validates aggregate and per-stage accounting.
    ///
    /// # Errors
    ///
    /// Returns an error for inconsistent identity, bounds, or attribution.
    pub fn validate(&self) -> Result<(), CoreError> {
        self.specification.validate()?;
        if self.pipeline != self.specification.digest()?
            || self.begins > 1
            || self.elements_committed > self.elements_copied
            || self.stages.len() != self.specification.stages.len()
        {
            return Err(CoreError::invalid(
                "diffusion pipeline receipt",
                "aggregate accounting is inconsistent",
            ));
        }
        let failed = self
            .failed_stage
            .map(usize::try_from)
            .transpose()
            .map_err(|_| {
                CoreError::invalid("diffusion pipeline receipt", "failed stage exceeds usize")
            })?;
        if failed.is_some_and(|index| index >= self.stages.len()) {
            return Err(CoreError::invalid(
                "diffusion pipeline receipt",
                "failed stage is outside the pipeline",
            ));
        }
        let mut failures = HashSet::new();
        for (index, (stage, specification)) in self
            .stages
            .iter()
            .zip(&self.specification.stages)
            .enumerate()
        {
            if stage.specification != *specification
                || stage.resets > self.begins
                || stage.invocations > stage.specification.max_invocations
                || stage.elements_changed > stage.elements_seen
            {
                return Err(CoreError::invalid(
                    "diffusion pipeline receipt",
                    "stage accounting is inconsistent",
                ));
            }
            if stage.failure.is_some() {
                failures.insert(index);
            }
        }
        if failures.len() > 1 || failures.iter().next().copied() != failed {
            return Err(CoreError::invalid(
                "diffusion pipeline receipt",
                "failure attribution is inconsistent",
            ));
        }
        Ok(())
    }

    /// Returns the identity of this exact mechanical receipt.
    ///
    /// # Errors
    ///
    /// Returns a validation or serialization error.
    pub fn digest(&self) -> Result<Digest, CoreError> {
        self.validate()?;
        Digest::of_serializable("diffusion-pipeline-receipt-v1", self)
    }
}

/// Conservative lineage for opaque adapter checkpoint bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiffusionCheckpointReceipt {
    /// Exact diffusion plan identity.
    pub plan: Digest,
    /// Adapter build identity.
    pub backend: Digest,
    /// Next zero-based schedule step after restore.
    pub next_step: u32,
    /// Identity of exact tensor state bytes.
    pub state: Digest,
    /// Identity of exact scheduler and RNG continuation state.
    pub continuation: Digest,
}

impl DiffusionCheckpointReceipt {
    /// Validates checkpoint identity and position against a plan.
    ///
    /// # Errors
    ///
    /// Returns an error when the plan differs or the position is out of bounds.
    pub fn validate_for(&self, plan: &DiffusionPlan) -> Result<(), CoreError> {
        plan.validate()?;
        if self.plan != plan.digest()?
            || usize::try_from(self.next_step).map_or(true, |step| step > plan.schedule.steps())
        {
            return Err(CoreError::invalid(
                "diffusion checkpoint",
                "plan identity or step position is incompatible",
            ));
        }
        Ok(())
    }

    /// Returns the identity of this exact checkpoint lineage.
    ///
    /// # Errors
    ///
    /// Returns a validation or serialization error.
    pub fn digest_for(&self, plan: &DiffusionPlan) -> Result<Digest, CoreError> {
        self.validate_for(plan)?;
        Digest::of_serializable("diffusion-checkpoint-receipt-v1", self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor() -> TensorSpec {
        TensorSpec::new(
            vec![8, 8, 3, 1],
            TensorDType::F32,
            TensorLayout::DimensionZeroFastest,
            "test0",
        )
        .unwrap()
    }

    #[test]
    fn tensor_bounds_reject_zero_and_overflow() {
        assert!(
            TensorSpec::new(
                vec![1, 0],
                TensorDType::F32,
                TensorLayout::DimensionZeroFastest,
                "test0"
            )
            .is_err()
        );
        assert!(
            TensorSpec::new(
                vec![u64::MAX, 2],
                TensorDType::F32,
                TensorLayout::DimensionZeroFastest,
                "test0"
            )
            .is_err()
        );
    }

    #[test]
    fn schedules_reject_nonfinite_and_increasing_values() {
        let implementation = Digest::of_bytes("test-scheduler", b"v1");
        assert!(DiffusionSchedule::new(implementation.clone(), vec![1.0, f32::NAN]).is_err());
        assert!(DiffusionSchedule::new(implementation, vec![0.0, 1.0]).is_err());
    }

    #[test]
    fn component_names_accept_catalog_periods_but_reject_uppercase() {
        let schedule =
            DiffusionSchedule::new(Digest::of_bytes("scheduler", b"v1"), vec![1.0, 0.0]).unwrap();
        let mut components = BTreeMap::new();
        components.insert(
            "wan-2.1-vae".to_owned(),
            Digest::of_bytes("artifact", b"vae"),
        );
        assert!(
            DiffusionPlan::new(
                components.clone(),
                Digest::of_bytes("conditioning", b"input"),
                Digest::of_bytes("rng", b"v1"),
                7,
                tensor(),
                schedule.clone(),
            )
            .is_ok()
        );
        components.insert(
            "Invalid".to_owned(),
            Digest::of_bytes("artifact", b"invalid"),
        );
        assert!(
            DiffusionPlan::new(
                components,
                Digest::of_bytes("conditioning", b"input"),
                Digest::of_bytes("rng", b"v1"),
                7,
                tensor(),
                schedule,
            )
            .is_err()
        );
    }

    #[test]
    fn step_context_binds_exact_tensor_and_sigmas() {
        let schedule =
            DiffusionSchedule::new(Digest::of_bytes("scheduler", b"v1"), vec![1.0, 0.0]).unwrap();
        let mut components = BTreeMap::new();
        components.insert("model".to_owned(), Digest::of_bytes("artifact", b"model"));
        let plan = DiffusionPlan::new(
            components,
            Digest::of_bytes("conditioning", b"input"),
            Digest::of_bytes("rng", b"v1"),
            7,
            tensor(),
            schedule,
        )
        .unwrap();
        let mut context = StepContext::for_plan(&plan, 0).unwrap();
        context.tensor.device = "other0".to_owned();
        assert!(context.validate_for(&plan).is_err());
    }
}

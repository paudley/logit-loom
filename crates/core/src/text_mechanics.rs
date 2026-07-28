// SPDX-License-Identifier: MIT OR Apache-2.0

//! Versioned aggregate text-mechanics plans and receipts.

use serde::{Deserialize, Serialize};

use crate::{
    ActivationProgramV1, ControlVectorSpec, CoreError, Digest, GenerationPlan, LoraSpec,
    MAX_ACTIVATION_OBSERVATIONS, SpeculationPlanV1, TextModelTopologyV1,
};

/// Maximum `LoRA` applications bound into one text-mechanics plan.
pub const MAX_TEXT_MECHANICS_LORAS: usize = 32;
/// Maximum speculative tokens requested from one draft boundary.
pub const MAX_SPECULATIVE_TOKENS: u32 = 4_096;
/// Maximum activation-capture plans bound into one aggregate operation.
pub const MAX_TEXT_MECHANICS_CAPTURES: usize = MAX_ACTIVATION_OBSERVATIONS;

/// Complete backend-neutral mechanics for one bounded text operation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextMechanicsPlanV1 {
    /// Complete sampler, grammar, stop, and logit-bias mechanics.
    pub generation: GenerationPlan,
    /// Exact requested controlled-prefill token bound.
    pub controlled_prefill_tokens: u32,
    /// Ordered transform pipeline identity, when installed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transform_pipeline: Option<Digest>,
    /// Ordered observer set identity, when installed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observer_set: Option<Digest>,
    /// Ordered scoped `LoRA` applications.
    #[serde(default)]
    pub loras: Vec<LoraSpec>,
    /// Optional in-memory control-vector application.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_vector: Option<ControlVectorSpec>,
    /// Exact parent checkpoint identity for a branch, when restoring one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branch_checkpoint: Option<Digest>,
    /// Exact draft implementation for bounded speculation, when enabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft_implementation: Option<Digest>,
    /// Maximum draft tokens per speculation boundary; zero disables it.
    pub maximum_speculative_tokens: u32,
}

impl TextMechanicsPlanV1 {
    /// Validates complete mechanics and returns the stable plan identity.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed generation, steering, or speculation
    /// bounds.
    pub fn digest(&self) -> Result<Digest, CoreError> {
        self.generation.validate()?;
        if self.controlled_prefill_tokens == 0 {
            return Err(CoreError::invalid(
                "text mechanics plan",
                "controlled prefill token bound must be nonzero",
            ));
        }
        if self.loras.len() > MAX_TEXT_MECHANICS_LORAS {
            return Err(CoreError::invalid(
                "text mechanics plan",
                "too many scoped LoRA applications",
            ));
        }
        for lora in &self.loras {
            lora.validate()?;
        }
        if let Some(control) = &self.control_vector {
            control.validate()?;
        }
        let speculation_is_consistent = match self.draft_implementation {
            Some(_) => (1..=MAX_SPECULATIVE_TOKENS).contains(&self.maximum_speculative_tokens),
            None => self.maximum_speculative_tokens == 0,
        };
        if !speculation_is_consistent {
            return Err(CoreError::invalid(
                "text mechanics plan",
                "draft implementation and speculative token bound must be selected together",
            ));
        }
        Digest::of_serializable("text-mechanics-plan-v1", self)
    }
}

/// Content-free terminal evidence for [`TextMechanicsPlanV1`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextMechanicsReceiptV1 {
    /// Exact plan identity.
    pub plan: Digest,
    /// Optional controlled-prefill receipt identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_receipt: Option<Digest>,
    /// Optional generation receipt identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation_receipt: Option<Digest>,
    /// Optional checkpoint captured after the operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint: Option<Digest>,
    /// Branch lineage identity, when this operation restored a checkpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branch_checkpoint: Option<Digest>,
    /// Number of draft tokens accepted by the target model.
    pub accepted_speculative_tokens: u32,
}

impl TextMechanicsReceiptV1 {
    /// Validates the receipt against its plan and returns its stable identity.
    ///
    /// # Errors
    ///
    /// Returns an error for a mismatched plan, omitted terminal evidence, or
    /// impossible speculation accounting.
    pub fn digest_for(&self, plan: &TextMechanicsPlanV1) -> Result<Digest, CoreError> {
        if self.plan != plan.digest()? {
            return Err(CoreError::invalid(
                "text mechanics receipt",
                "receipt plan does not match the supplied plan",
            ));
        }
        if self.prefill_receipt.is_none() && self.generation_receipt.is_none() {
            return Err(CoreError::invalid(
                "text mechanics receipt",
                "receipt requires prefill or generation terminal evidence",
            ));
        }
        if self.branch_checkpoint != plan.branch_checkpoint {
            return Err(CoreError::invalid(
                "text mechanics receipt",
                "branch checkpoint lineage differs from the plan",
            ));
        }
        if self.accepted_speculative_tokens > plan.maximum_speculative_tokens {
            return Err(CoreError::invalid(
                "text mechanics receipt",
                "accepted speculative tokens exceed the plan bound",
            ));
        }
        Digest::of_serializable("text-mechanics-receipt-v1", self)
    }
}

/// Aggregate text mechanics with topology-bound activation and speculation.
///
/// This is a new serialized contract. It does not reinterpret
/// [`TextMechanicsPlanV1`] or any V1 digest domain.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextMechanicsPlanV2 {
    /// Complete sampler, grammar, stop, and logit-bias mechanics.
    pub generation: GenerationPlan,
    /// Exact requested controlled-prefill token bound.
    pub controlled_prefill_tokens: u32,
    /// Ordered transform-pipeline identity, when installed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transform_pipeline: Option<Digest>,
    /// Ordered generated-token observer-set identity, when installed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observer_set: Option<Digest>,
    /// Ordered scoped `LoRA` applications.
    #[serde(default)]
    pub loras: Vec<LoraSpec>,
    /// Optional in-memory llama.cpp control-vector application.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_vector: Option<ControlVectorSpec>,
    /// Exact target model-topology identity.
    pub target_topology: Digest,
    /// Target activation program, when installed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_activation: Option<ActivationProgramV1>,
    /// Canonically ordered activation-capture plan identities available to the
    /// target activation runtime.
    #[serde(default)]
    pub activation_captures: Vec<Digest>,
    /// Exact target-authoritative speculation plan.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speculation: Option<SpeculationPlanV1>,
    /// Exact parent checkpoint identity for a restored branch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branch_checkpoint: Option<Digest>,
}

impl TextMechanicsPlanV2 {
    /// Validates complete V2 mechanics against target and optional draft
    /// topologies.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed generation, steering, topology,
    /// activation, capture, or speculation mechanics.
    pub fn digest_for(
        &self,
        target: &TextModelTopologyV1,
        draft: Option<&TextModelTopologyV1>,
    ) -> Result<Digest, CoreError> {
        self.generation.validate()?;
        if self.controlled_prefill_tokens == 0 {
            return Err(CoreError::invalid(
                "text mechanics plan v2",
                "controlled prefill token bound must be nonzero",
            ));
        }
        if self.loras.len() > MAX_TEXT_MECHANICS_LORAS {
            return Err(CoreError::invalid(
                "text mechanics plan v2",
                "too many scoped LoRA applications",
            ));
        }
        for lora in &self.loras {
            lora.validate()?;
        }
        if let Some(control) = &self.control_vector {
            control.validate()?;
        }
        let target_topology = target.digest()?;
        if self.target_topology != target_topology {
            return Err(CoreError::invalid(
                "text mechanics plan v2",
                "target topology identity does not match",
            ));
        }
        if self.activation_captures.len() > MAX_TEXT_MECHANICS_CAPTURES
            || self
                .activation_captures
                .windows(2)
                .any(|window| window[0] >= window[1])
        {
            return Err(CoreError::invalid(
                "text mechanics plan v2",
                "capture identities must be bounded, unique, and canonically ordered",
            ));
        }
        let target_activation_identity = self
            .target_activation
            .as_ref()
            .map(|program| program.digest_for(target))
            .transpose()?;
        if let Some(program) = &self.target_activation {
            if program.observations != self.activation_captures {
                return Err(CoreError::invalid(
                    "text mechanics plan v2",
                    "target program observations differ from available captures",
                ));
            }
        } else if !self.activation_captures.is_empty() {
            return Err(CoreError::invalid(
                "text mechanics plan v2",
                "capture plans require a target activation program",
            ));
        }

        match (&self.speculation, draft) {
            (None, None | Some(_)) => {}
            (Some(_), None) => {
                return Err(CoreError::invalid(
                    "text mechanics plan v2",
                    "speculation requires an exact draft topology",
                ));
            }
            (Some(speculation), Some(draft)) => {
                speculation.digest_for(target, draft)?;
                let declared_target = speculation.activation.target_program().cloned();
                if declared_target != target_activation_identity {
                    return Err(CoreError::invalid(
                        "text mechanics plan v2",
                        "target activation differs from the speculation policy",
                    ));
                }
            }
        }
        Digest::of_serializable("text-mechanics-plan-v2", self)
    }

    /// Returns the independently selected draft activation-program identity.
    pub fn draft_activation_identity(&self) -> Option<&Digest> {
        self.speculation
            .as_ref()
            .and_then(|plan| plan.activation.draft_program())
    }
}

/// Resource-release and quiescence evidence at aggregate operation completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "each flag records release of a distinct optional native resource"
)]
pub struct TextMechanicsCleanupReceiptV2 {
    /// Number of scoped `LoRA` applications removed.
    pub loras_removed: u32,
    /// Whether a selected control vector was removed.
    pub control_vector_removed: bool,
    /// Whether the target activation runtime was released.
    pub target_activation_released: bool,
    /// Whether a separately selected draft activation runtime was released.
    pub draft_activation_released: bool,
    /// Whether speculative state was quiescent before release.
    pub speculation_quiescent: bool,
}

impl TextMechanicsCleanupReceiptV2 {
    fn validate_for(self, plan: &TextMechanicsPlanV2) -> Result<(), CoreError> {
        let loras = u32::try_from(plan.loras.len())
            .map_err(|_| CoreError::invalid("text mechanics cleanup", "LoRA count exceeds u32"))?;
        let target_selected = plan.target_activation.is_some();
        let draft_selected = plan.draft_activation_identity().is_some();
        let speculation_selected = plan.speculation.is_some();
        if self.loras_removed != loras
            || self.control_vector_removed != plan.control_vector.is_some()
            || self.target_activation_released != target_selected
            || self.draft_activation_released != draft_selected
            || self.speculation_quiescent != speculation_selected
        {
            return Err(CoreError::invalid(
                "text mechanics cleanup",
                "resource-release evidence differs from the plan",
            ));
        }
        Ok(())
    }
}

/// Content-free terminal evidence for [`TextMechanicsPlanV2`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextMechanicsReceiptV2 {
    /// Exact V2 plan identity.
    pub plan: Digest,
    /// Controlled-prefill receipt identity, when prefill ran.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_receipt: Option<Digest>,
    /// Generation receipt identity, when generation ran.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation_receipt: Option<Digest>,
    /// Ordered completed activation-capture receipt identities.
    #[serde(default)]
    pub activation_captures: Vec<Digest>,
    /// Target activation-program receipt identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_activation: Option<Digest>,
    /// Draft activation-program receipt identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft_activation: Option<Digest>,
    /// Aggregate speculation receipt identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speculation: Option<Digest>,
    /// Quiescent checkpoint captured after the operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint: Option<Digest>,
    /// Branch lineage identity, when this operation restored a checkpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branch_checkpoint: Option<Digest>,
    /// Successful release and quiescence accounting.
    pub cleanup: TextMechanicsCleanupReceiptV2,
}

impl TextMechanicsReceiptV2 {
    /// Validates terminal evidence against its exact V2 plan and topologies.
    ///
    /// # Errors
    ///
    /// Returns an error for mismatched plan or branch lineage, omitted
    /// terminal evidence, capture/program/speculation evidence inconsistent
    /// with selected mechanics, failed cleanup accounting, or serialization
    /// failure.
    pub fn digest_for(
        &self,
        plan: &TextMechanicsPlanV2,
        target: &TextModelTopologyV1,
        draft: Option<&TextModelTopologyV1>,
    ) -> Result<Digest, CoreError> {
        if self.plan != plan.digest_for(target, draft)? {
            return Err(CoreError::invalid(
                "text mechanics receipt v2",
                "receipt plan does not match the supplied plan",
            ));
        }
        if self.prefill_receipt.is_none() && self.generation_receipt.is_none() {
            return Err(CoreError::invalid(
                "text mechanics receipt v2",
                "receipt requires prefill or generation terminal evidence",
            ));
        }
        if self.branch_checkpoint != plan.branch_checkpoint {
            return Err(CoreError::invalid(
                "text mechanics receipt v2",
                "branch checkpoint lineage differs from the plan",
            ));
        }
        if self.activation_captures.len() > plan.activation_captures.len()
            || self.target_activation.is_some() != plan.target_activation.is_some()
            || self.draft_activation.is_some() != plan.draft_activation_identity().is_some()
            || self.speculation.is_some() != plan.speculation.is_some()
        {
            return Err(CoreError::invalid(
                "text mechanics receipt v2",
                "activation, capture, or speculation evidence differs from the plan",
            ));
        }
        self.cleanup.validate_for(plan)?;
        Digest::of_serializable("text-mechanics-receipt-v2", self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SamplingPlan, TextSpeculativeMechanismV1};

    fn plan() -> TextMechanicsPlanV1 {
        TextMechanicsPlanV1 {
            generation: GenerationPlan {
                sampling: SamplingPlan::default(),
                max_tokens: 16,
                biases: Vec::new(),
                grammar: None,
                stops: Vec::new(),
            },
            controlled_prefill_tokens: 128,
            transform_pipeline: Some(Digest::of_bytes("test-transform", b"one")),
            observer_set: Some(Digest::of_bytes("test-observer", b"one")),
            loras: Vec::new(),
            control_vector: None,
            branch_checkpoint: Some(Digest::of_bytes("test-checkpoint", b"one")),
            draft_implementation: Some(Digest::of_bytes("test-draft", b"one")),
            maximum_speculative_tokens: 8,
        }
    }

    fn topology() -> TextModelTopologyV1 {
        TextModelTopologyV1 {
            model: Digest::of_bytes("test-model", b"one"),
            backend: Digest::of_bytes("test-backend", b"one"),
            architecture_implementation: Digest::of_bytes("test-architecture", b"one"),
            layers: 4,
            embedding_width: 8,
            experts: None,
            experts_used: None,
            nextn_layers: 1,
            supported_speculation: vec![TextSpeculativeMechanismV1::Mtp],
        }
    }

    fn plan_v2(topology: &TextModelTopologyV1) -> TextMechanicsPlanV2 {
        TextMechanicsPlanV2 {
            generation: GenerationPlan {
                sampling: SamplingPlan::default(),
                max_tokens: 16,
                biases: Vec::new(),
                grammar: None,
                stops: Vec::new(),
            },
            controlled_prefill_tokens: 128,
            transform_pipeline: None,
            observer_set: None,
            loras: Vec::new(),
            control_vector: None,
            target_topology: topology.digest().unwrap(),
            target_activation: None,
            activation_captures: Vec::new(),
            speculation: None,
            branch_checkpoint: None,
        }
    }

    #[test]
    fn aggregate_plan_binds_complete_mechanics() {
        assert!(plan().digest().is_ok());
        let mut inconsistent = plan();
        inconsistent.maximum_speculative_tokens = 0;
        assert!(inconsistent.digest().is_err());
    }

    #[test]
    fn receipt_binds_branch_and_speculation_accounting() {
        let plan = plan();
        let receipt = TextMechanicsReceiptV1 {
            plan: plan.digest().unwrap(),
            prefill_receipt: Some(Digest::of_bytes("test-prefill", b"one")),
            generation_receipt: None,
            checkpoint: None,
            branch_checkpoint: plan.branch_checkpoint.clone(),
            accepted_speculative_tokens: 8,
        };
        assert!(receipt.digest_for(&plan).is_ok());
        let mut malformed = receipt;
        malformed.accepted_speculative_tokens = 9;
        assert!(malformed.digest_for(&plan).is_err());
    }

    #[test]
    fn v2_plan_binds_topology_without_reinterpreting_v1() {
        let topology = topology();
        let plan = plan_v2(&topology);
        let identity = plan.digest_for(&topology, None).unwrap();
        assert_eq!(
            identity,
            Digest::of_serializable("text-mechanics-plan-v2", &plan).unwrap()
        );
        assert_ne!(identity, Digest::of_bytes("text-mechanics-plan-v1", b""));

        let mut mismatched = plan;
        mismatched.target_topology = Digest::of_bytes("test-topology", b"other");
        assert!(mismatched.digest_for(&topology, None).is_err());
    }

    #[test]
    fn v2_receipt_requires_exact_cleanup_evidence() {
        let topology = topology();
        let plan = plan_v2(&topology);
        let mut receipt = TextMechanicsReceiptV2 {
            plan: plan.digest_for(&topology, None).unwrap(),
            prefill_receipt: Some(Digest::of_bytes("test-prefill", b"one")),
            generation_receipt: None,
            activation_captures: Vec::new(),
            target_activation: None,
            draft_activation: None,
            speculation: None,
            checkpoint: None,
            branch_checkpoint: None,
            cleanup: TextMechanicsCleanupReceiptV2 {
                loras_removed: 0,
                control_vector_removed: false,
                target_activation_released: false,
                draft_activation_released: false,
                speculation_quiescent: false,
            },
        };
        assert!(receipt.digest_for(&plan, &topology, None).is_ok());
        receipt.cleanup.control_vector_removed = true;
        assert!(receipt.digest_for(&plan, &topology, None).is_err());
    }
}

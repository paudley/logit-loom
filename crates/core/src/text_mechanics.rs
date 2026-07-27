// SPDX-License-Identifier: MIT OR Apache-2.0

//! Versioned aggregate text-mechanics plans and receipts.

use serde::{Deserialize, Serialize};

use crate::{ControlVectorSpec, CoreError, Digest, GenerationPlan, LoraSpec};

/// Maximum `LoRA` applications bound into one text-mechanics plan.
pub const MAX_TEXT_MECHANICS_LORAS: usize = 32;
/// Maximum speculative tokens requested from one draft boundary.
pub const MAX_SPECULATIVE_TOKENS: u32 = 4_096;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SamplingPlan;

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
}

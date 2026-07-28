// SPDX-License-Identifier: MIT OR Apache-2.0

//! Versioned contracts and receipts for target-authoritative speculation.

use serde::{Deserialize, Serialize};

use crate::{
    ActivationTelemetryDispositionV1, CoreError, Digest, MAX_SPECULATIVE_TOKENS,
    TextModelTopologyV1, TextSpeculativeMechanismV1, TokenId,
};

/// Maximum sequence slots declared by one speculative plan.
pub const MAX_SPECULATION_SEQUENCES: u32 = 4_096;
/// Maximum completed proposal boundaries retained in one aggregate receipt.
pub const MAX_SPECULATION_BOUNDARIES: usize = 1_048_576;
/// Maximum activation telemetry resolutions retained at one proposal boundary.
pub const MAX_SPECULATION_TELEMETRY_RESOLUTIONS: usize = 65_536;
/// Maximum opaque speculative implementation-state bytes.
pub const MAX_SPECULATION_STATE_BYTES: u64 = 64 * 1024 * 1024;

/// Explicit target and draft activation-program selection.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SpeculationActivationPolicyV1 {
    /// Run no activation program in either context.
    None,
    /// Apply one program to target prefill, generation, and verification only.
    TargetOnly {
        /// Exact target activation-program identity.
        target_program: Digest,
    },
    /// Apply independently validated programs to target and draft contexts.
    SeparateDraftProgram {
        /// Exact target activation-program identity.
        target_program: Digest,
        /// Exact draft activation-program identity.
        draft_program: Digest,
    },
}

impl SpeculationActivationPolicyV1 {
    /// Returns the selected target-program identity.
    pub const fn target_program(&self) -> Option<&Digest> {
        match self {
            Self::None => None,
            Self::TargetOnly { target_program }
            | Self::SeparateDraftProgram { target_program, .. } => Some(target_program),
        }
    }

    /// Returns the independently selected draft-program identity.
    pub const fn draft_program(&self) -> Option<&Digest> {
        match self {
            Self::SeparateDraftProgram { draft_program, .. } => Some(draft_program),
            Self::None | Self::TargetOnly { .. } => None,
        }
    }
}

/// Complete backend-neutral configuration for one speculative text operation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeculationPlanV1 {
    /// Target model artifact identity.
    pub target_model: Digest,
    /// Target topology identity.
    pub target_topology: Digest,
    /// Draft model artifact identity.
    pub draft_model: Digest,
    /// Draft topology identity.
    pub draft_topology: Digest,
    /// Exact native and safe-wrapper implementation identity.
    pub implementation: Digest,
    /// Native proposal mechanism.
    pub mechanism: TextSpeculativeMechanismV1,
    /// Concurrent sequence slots.
    pub sequences: u32,
    /// Inclusive proposal-token bound at each boundary.
    pub maximum_draft_tokens: u32,
    /// Minimum proposal tokens requested from the draft implementation.
    pub minimum_draft_tokens: u32,
    /// Exact IEEE-754 draft probability-floor bits.
    pub probability_floor_bits: u32,
    /// Explicit target/draft activation selection.
    pub activation: SpeculationActivationPolicyV1,
}

impl SpeculationPlanV1 {
    /// Returns the finite draft probability floor.
    ///
    /// # Errors
    ///
    /// Returns an error when the encoded value is NaN, infinite, or outside
    /// the inclusive unit interval.
    pub fn probability_floor(&self) -> Result<f32, CoreError> {
        let value = f32::from_bits(self.probability_floor_bits);
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(CoreError::invalid(
                "speculation probability floor",
                "value must be finite and between zero and one",
            ));
        }
        Ok(value)
    }

    /// Validates model, topology, mechanism, and bound compatibility.
    ///
    /// # Errors
    ///
    /// Returns an error for mismatched topology lineage, an unavailable
    /// mechanism, incompatible MTP/EAGLE model relationships, or malformed
    /// sequence, proposal, probability, or activation bounds.
    pub fn digest_for(
        &self,
        target: &TextModelTopologyV1,
        draft: &TextModelTopologyV1,
    ) -> Result<Digest, CoreError> {
        let target_topology = target.digest()?;
        let draft_topology = draft.digest()?;
        if self.target_model != target.model
            || self.target_topology != target_topology
            || self.draft_model != draft.model
            || self.draft_topology != draft_topology
        {
            return Err(CoreError::invalid(
                "speculation plan",
                "model or topology lineage does not match",
            ));
        }
        if !target.supported_speculation.contains(&self.mechanism) {
            return Err(CoreError::invalid(
                "speculation plan",
                "target topology does not report the selected mechanism",
            ));
        }
        match self.mechanism {
            TextSpeculativeMechanismV1::Mtp => {
                if target.model != draft.model || target_topology != draft_topology {
                    return Err(CoreError::invalid(
                        "speculation plan",
                        "MTP requires target and draft contexts over one exact topology",
                    ));
                }
                if target.nextn_layers == 0 {
                    return Err(CoreError::invalid(
                        "speculation plan",
                        "MTP requires at least one reported NextN layer",
                    ));
                }
            }
            TextSpeculativeMechanismV1::Eagle3 => {
                if target.model == draft.model {
                    return Err(CoreError::invalid(
                        "speculation plan",
                        "EAGLE-3 requires a separately identified draft model",
                    ));
                }
            }
        }
        if self.sequences == 0 || self.sequences > MAX_SPECULATION_SEQUENCES {
            return Err(CoreError::invalid(
                "speculation plan",
                "sequence count is outside the supported bound",
            ));
        }
        if self.maximum_draft_tokens == 0
            || self.maximum_draft_tokens > MAX_SPECULATIVE_TOKENS
            || self.minimum_draft_tokens > self.maximum_draft_tokens
        {
            return Err(CoreError::invalid(
                "speculation plan",
                "draft token bounds are inconsistent or excessive",
            ));
        }
        self.probability_floor()?;
        Digest::of_serializable("speculation-plan-v1", self)
    }
}

/// Final disposition of one provisional activation telemetry record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeculationTelemetryResolutionV1 {
    /// Identity emitted while target verification was unresolved.
    pub provisional: Digest,
    /// Target-authoritative final disposition.
    pub disposition: ActivationTelemetryDispositionV1,
    /// Identity of the final admitted or rejected activation record.
    pub resolved: Digest,
}

impl SpeculationTelemetryResolutionV1 {
    fn validate(&self) -> Result<(), CoreError> {
        if self.disposition == ActivationTelemetryDispositionV1::Provisional {
            return Err(CoreError::invalid(
                "speculation telemetry resolution",
                "final disposition cannot remain provisional",
            ));
        }
        Ok(())
    }
}

/// Target-authoritative accounting for one completed proposal boundary.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeculationBoundaryReceiptV1 {
    /// Exact speculation-plan identity.
    pub plan: Digest,
    /// Zero-based completed boundary index.
    pub boundary: u64,
    /// Identity of proposed token IDs in exact draft order.
    pub proposal: Digest,
    /// Number of proposed token IDs.
    pub proposed: u32,
    /// Accepted proposal prefix length.
    pub accepted: u32,
    /// Rejected proposal suffix length.
    pub rejected: u32,
    /// Provisional telemetry resolved by this target decision.
    #[serde(default)]
    pub telemetry: Vec<SpeculationTelemetryResolutionV1>,
}

impl SpeculationBoundaryReceiptV1 {
    /// Constructs exact boundary accounting from one proposal and accepted
    /// prefix length.
    ///
    /// # Errors
    ///
    /// Returns an error for an excessive proposal, an accepted prefix beyond
    /// the proposal, or identity encoding failure. A zero-token proposal is a
    /// valid completed boundary when the accepted count is also zero.
    pub fn from_tokens(
        plan: &SpeculationPlanV1,
        plan_identity: Digest,
        boundary: u64,
        proposed_tokens: &[TokenId],
        accepted: u32,
        telemetry: Vec<SpeculationTelemetryResolutionV1>,
    ) -> Result<Self, CoreError> {
        let proposed = u32::try_from(proposed_tokens.len()).map_err(|_| {
            CoreError::invalid("speculation boundary", "proposal length exceeds u32")
        })?;
        if proposed > plan.maximum_draft_tokens || accepted > proposed {
            return Err(CoreError::invalid(
                "speculation boundary",
                "proposal or accepted-prefix count is invalid",
            ));
        }
        let rejected = proposed
            .checked_sub(accepted)
            .ok_or_else(|| CoreError::invalid("speculation boundary", "count underflowed"))?;
        let receipt = Self {
            plan: plan_identity,
            boundary,
            proposal: Digest::of_serializable(
                "speculation-proposal-token-ids-v1",
                proposed_tokens,
            )?,
            proposed,
            accepted,
            rejected,
            telemetry,
        };
        receipt.validate_for(plan)?;
        Ok(receipt)
    }

    /// Validates boundary accounting against a plan.
    ///
    /// The caller must first obtain the supplied plan identity from
    /// [`SpeculationPlanV1::digest_for`].
    ///
    /// # Errors
    ///
    /// Returns an error for mismatched lineage, impossible counts, excessive
    /// telemetry, or unresolved provisional dispositions.
    pub fn validate_for(&self, plan: &SpeculationPlanV1) -> Result<(), CoreError> {
        if self.plan != Digest::of_serializable("speculation-plan-v1", plan)?
            || self.proposed > plan.maximum_draft_tokens
            || self.accepted > self.proposed
            || self.rejected != self.proposed - self.accepted
            || self.telemetry.len() > MAX_SPECULATION_TELEMETRY_RESOLUTIONS
        {
            return Err(CoreError::invalid(
                "speculation boundary receipt",
                "plan lineage, counts, or telemetry bounds are invalid",
            ));
        }
        for resolution in &self.telemetry {
            resolution.validate()?;
        }
        Ok(())
    }

    /// Returns the stable completed-boundary identity.
    ///
    /// # Errors
    ///
    /// Returns a validation or serialization error.
    pub fn digest_for(&self, plan: &SpeculationPlanV1) -> Result<Digest, CoreError> {
        self.validate_for(plan)?;
        Digest::of_serializable("speculation-boundary-receipt-v1", self)
    }
}

/// Aggregate proposal and target-acceptance accounting.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeculationReceiptV1 {
    /// Exact speculation-plan identity.
    pub plan: Digest,
    /// Ordered completed-boundary identities.
    pub boundaries: Vec<Digest>,
    /// Total proposed tokens.
    pub proposed: u64,
    /// Total target-accepted proposal tokens.
    pub accepted: u64,
    /// Total rejected proposal tokens.
    pub rejected: u64,
}

impl SpeculationReceiptV1 {
    /// Builds aggregate accounting from exact ordered boundary receipts.
    ///
    /// # Errors
    ///
    /// Returns an error for excessive boundaries, non-contiguous indices,
    /// mismatched plans, arithmetic overflow, or malformed boundary
    /// accounting. An operation that terminates before its first proposal
    /// boundary has valid zeroed aggregate accounting.
    pub fn from_boundaries(
        plan: &SpeculationPlanV1,
        boundaries: &[SpeculationBoundaryReceiptV1],
    ) -> Result<Self, CoreError> {
        if boundaries.len() > MAX_SPECULATION_BOUNDARIES {
            return Err(CoreError::invalid(
                "speculation receipt",
                "boundary count exceeds the supported bound",
            ));
        }
        let plan_identity = Digest::of_serializable("speculation-plan-v1", plan)?;
        let mut identities = Vec::with_capacity(boundaries.len());
        let mut proposed = 0_u64;
        let mut accepted = 0_u64;
        let mut rejected = 0_u64;
        for (index, boundary) in boundaries.iter().enumerate() {
            let expected = u64::try_from(index).map_err(|_| {
                CoreError::invalid("speculation receipt", "boundary index exceeds u64")
            })?;
            if boundary.boundary != expected || boundary.plan != plan_identity {
                return Err(CoreError::invalid(
                    "speculation receipt",
                    "boundaries must be contiguous and plan-bound",
                ));
            }
            identities.push(boundary.digest_for(plan)?);
            proposed = proposed
                .checked_add(u64::from(boundary.proposed))
                .ok_or_else(|| {
                    CoreError::invalid("speculation receipt", "proposal count overflowed")
                })?;
            accepted = accepted
                .checked_add(u64::from(boundary.accepted))
                .ok_or_else(|| {
                    CoreError::invalid("speculation receipt", "acceptance count overflowed")
                })?;
            rejected = rejected
                .checked_add(u64::from(boundary.rejected))
                .ok_or_else(|| {
                    CoreError::invalid("speculation receipt", "rejection count overflowed")
                })?;
        }
        Ok(Self {
            plan: plan_identity,
            boundaries: identities,
            proposed,
            accepted,
            rejected,
        })
    }

    /// Validates aggregate accounting against exact boundary receipts.
    ///
    /// # Errors
    ///
    /// Returns an error when recomputed aggregate evidence differs.
    pub fn digest_for(
        &self,
        plan: &SpeculationPlanV1,
        boundaries: &[SpeculationBoundaryReceiptV1],
    ) -> Result<Digest, CoreError> {
        if *self != Self::from_boundaries(plan, boundaries)? {
            return Err(CoreError::invalid(
                "speculation receipt",
                "aggregate evidence does not match its boundary receipts",
            ));
        }
        Digest::of_serializable("speculation-receipt-v1", self)
    }
}

/// Serializable evidence for one quiescent speculative checkpoint envelope.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeculativeCheckpointReceiptV1 {
    /// Exact aggregate text-mechanics plan identity.
    pub mechanics: Digest,
    /// Exact speculation-plan identity.
    pub speculation: Digest,
    /// Exact target causal-state checkpoint receipt.
    pub target_state: Digest,
    /// Exact draft causal-state checkpoint receipt.
    pub draft_state: Digest,
    /// Identity of opaque speculative implementation state.
    pub implementation_state: Digest,
    /// Opaque speculative implementation-state byte count.
    pub implementation_state_bytes: u64,
    /// Target sampler clone lineage identity.
    pub target_sampler_lineage: Digest,
    /// Exact admitted target token-history identity.
    pub admitted_history: Digest,
    /// Next target causal position.
    pub position: u64,
    /// Number of completely resolved speculation boundaries.
    pub completed_boundaries: u64,
    /// Active target activation-program identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_activation: Option<Digest>,
    /// Active draft activation-program identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft_activation: Option<Digest>,
    /// Parent checkpoint identity for a restored branch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<Digest>,
}

impl SpeculativeCheckpointReceiptV1 {
    /// Validates quiescent checkpoint accounting and returns its identity.
    ///
    /// # Errors
    ///
    /// Returns an error for zero/excessive implementation state, activation
    /// lineage inconsistent with the speculation plan, or serialization
    /// failure.
    pub fn digest_for(&self, plan: &SpeculationPlanV1) -> Result<Digest, CoreError> {
        if self.speculation != Digest::of_serializable("speculation-plan-v1", plan)?
            || self.implementation_state_bytes == 0
            || self.implementation_state_bytes > MAX_SPECULATION_STATE_BYTES
            || self.target_activation != plan.activation.target_program().cloned()
            || self.draft_activation != plan.activation.draft_program().cloned()
        {
            return Err(CoreError::invalid(
                "speculative checkpoint receipt",
                "state bounds or plan lineage are invalid",
            ));
        }
        Digest::of_serializable("speculative-checkpoint-receipt-v1", self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn topology(model: &[u8], mtp: bool) -> TextModelTopologyV1 {
        TextModelTopologyV1 {
            model: Digest::of_bytes("test-model", model),
            backend: Digest::of_bytes("test-backend", b"one"),
            architecture_implementation: Digest::of_bytes("test-architecture", b"one"),
            layers: 4,
            embedding_width: 8,
            experts: None,
            experts_used: None,
            nextn_layers: u32::from(mtp),
            supported_speculation: if mtp {
                vec![
                    TextSpeculativeMechanismV1::Mtp,
                    TextSpeculativeMechanismV1::Eagle3,
                ]
            } else {
                vec![TextSpeculativeMechanismV1::Eagle3]
            },
        }
    }

    fn mtp_plan(topology: &TextModelTopologyV1) -> SpeculationPlanV1 {
        SpeculationPlanV1 {
            target_model: topology.model.clone(),
            target_topology: topology.digest().unwrap(),
            draft_model: topology.model.clone(),
            draft_topology: topology.digest().unwrap(),
            implementation: Digest::of_bytes("test-speculation", b"one"),
            mechanism: TextSpeculativeMechanismV1::Mtp,
            sequences: 1,
            maximum_draft_tokens: 4,
            minimum_draft_tokens: 0,
            probability_floor_bits: 0.0_f32.to_bits(),
            activation: SpeculationActivationPolicyV1::None,
        }
    }

    #[test]
    fn mtp_requires_one_exact_model_topology() {
        let target = topology(b"target", true);
        let mut plan = mtp_plan(&target);
        assert!(plan.digest_for(&target, &target).is_ok());

        let draft = topology(b"draft", true);
        plan.draft_model = draft.model.clone();
        plan.draft_topology = draft.digest().unwrap();
        assert!(plan.digest_for(&target, &draft).is_err());
    }

    #[test]
    fn eagle_requires_a_separate_draft_identity() {
        let target = topology(b"target", true);
        let draft = topology(b"draft", false);
        let plan = SpeculationPlanV1 {
            target_model: target.model.clone(),
            target_topology: target.digest().unwrap(),
            draft_model: draft.model.clone(),
            draft_topology: draft.digest().unwrap(),
            implementation: Digest::of_bytes("test-speculation", b"one"),
            mechanism: TextSpeculativeMechanismV1::Eagle3,
            sequences: 1,
            maximum_draft_tokens: 4,
            minimum_draft_tokens: 0,
            probability_floor_bits: 0.1_f32.to_bits(),
            activation: SpeculationActivationPolicyV1::None,
        };
        assert!(plan.digest_for(&target, &draft).is_ok());
    }

    #[test]
    fn boundaries_preserve_zero_partial_and_complete_acceptance() {
        let topology = topology(b"target", true);
        let plan = mtp_plan(&topology);
        let plan_identity = plan.digest_for(&topology, &topology).unwrap();
        let proposed = [
            TokenId::new(1).unwrap(),
            TokenId::new(2).unwrap(),
            TokenId::new(3).unwrap(),
        ];
        let boundaries = [0_u32, 2, 3]
            .into_iter()
            .enumerate()
            .map(|(index, accepted)| {
                SpeculationBoundaryReceiptV1::from_tokens(
                    &plan,
                    plan_identity.clone(),
                    u64::try_from(index).unwrap(),
                    &proposed,
                    accepted,
                    Vec::new(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let receipt = SpeculationReceiptV1::from_boundaries(&plan, &boundaries).unwrap();
        assert_eq!(receipt.proposed, 9);
        assert_eq!(receipt.accepted, 5);
        assert_eq!(receipt.rejected, 4);
        assert!(receipt.digest_for(&plan, &boundaries).is_ok());
    }

    #[test]
    fn pre_boundary_terminal_selection_has_zeroed_speculation_accounting() {
        let topology = topology(b"target", true);
        let plan = mtp_plan(&topology);
        let receipt = SpeculationReceiptV1::from_boundaries(&plan, &[]).unwrap();
        assert!(receipt.boundaries.is_empty());
        assert_eq!(
            (receipt.proposed, receipt.accepted, receipt.rejected),
            (0, 0, 0)
        );
        assert!(receipt.digest_for(&plan, &[]).is_ok());
    }

    #[test]
    fn provisional_telemetry_requires_a_final_disposition() {
        let resolution = SpeculationTelemetryResolutionV1 {
            provisional: Digest::of_bytes("test-provisional", b"one"),
            disposition: ActivationTelemetryDispositionV1::Provisional,
            resolved: Digest::of_bytes("test-resolved", b"one"),
        };
        assert!(resolution.validate().is_err());
    }
}

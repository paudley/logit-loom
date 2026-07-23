// SPDX-License-Identifier: MIT OR Apache-2.0

//! Logit-transform specifications and mechanical receipts.

use serde::{Deserialize, Serialize};

use crate::{CandidateMode, CoreError, Digest};

/// Maximum stages in one ordered transform pipeline.
pub const MAX_PIPELINE_STAGES: usize = 32;
/// Maximum UTF-8 bytes retained from a callback error or panic.
pub const MAX_RETAINED_FAILURE_BYTES: usize = 512;

/// One content-identified transform contract.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformSpec {
    /// Caller-defined implementation identity.
    pub implementation: Digest,
    /// Candidate exposure used for every invocation.
    pub mode: CandidateMode,
    /// Maximum transform invocations in one call.
    pub max_steps: u32,
}

impl TransformSpec {
    /// Creates and validates a transform contract.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid mode or zero step bound.
    pub fn new(
        implementation: Digest,
        mode: CandidateMode,
        max_steps: u32,
    ) -> Result<Self, CoreError> {
        let specification = Self {
            implementation,
            mode,
            max_steps,
        };
        specification.validate()?;
        Ok(specification)
    }

    /// Validates the bounded transform contract.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid mode or zero step bound.
    pub fn validate(&self) -> Result<(), CoreError> {
        self.mode.validate()?;
        if self.max_steps == 0 {
            return Err(CoreError::invalid(
                "transform max steps",
                "must be greater than zero",
            ));
        }
        Ok(())
    }

    /// Returns a content identity for this exact contract.
    ///
    /// # Errors
    ///
    /// Returns a validation or serialization error.
    pub fn digest(&self) -> Result<Digest, CoreError> {
        self.validate()?;
        Digest::of_serializable("transform-spec-v1", self)
    }
}

/// Ordered transform contracts sharing one candidate exposure.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineSpec {
    /// Stages in exact execution order.
    pub stages: Vec<TransformSpec>,
}

impl PipelineSpec {
    /// Validates the stage bound and common candidate mode.
    ///
    /// # Errors
    ///
    /// Returns an error unless one to 32 compatible stages are present.
    pub fn validate(&self) -> Result<(), CoreError> {
        if !(1..=MAX_PIPELINE_STAGES).contains(&self.stages.len()) {
            return Err(CoreError::invalid(
                "pipeline stages",
                format!("requires 1..={MAX_PIPELINE_STAGES} stages"),
            ));
        }
        let mode = self.stages[0].mode;
        for stage in &self.stages {
            stage.validate()?;
            if stage.mode != mode {
                return Err(CoreError::invalid(
                    "pipeline stages",
                    "all stages must use the same candidate mode",
                ));
            }
        }
        Ok(())
    }

    /// Returns the common candidate mode.
    ///
    /// # Errors
    ///
    /// Returns an error when the pipeline is invalid.
    pub fn mode(&self) -> Result<CandidateMode, CoreError> {
        self.validate()?;
        Ok(self.stages[0].mode)
    }

    /// Returns a content identity for this exact ordered pipeline.
    ///
    /// # Errors
    ///
    /// Returns a validation or serialization error.
    pub fn digest(&self) -> Result<Digest, CoreError> {
        self.validate()?;
        Digest::of_serializable("transform-pipeline-spec-v1", self)
    }
}

/// Callback phase that produced a contained failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CallbackPhase {
    /// Per-call implementation reset.
    Reset,
    /// Per-step logit transformation.
    Apply,
    /// Notification of a causally admitted token.
    Accept,
    /// Pre-sampling observer poll.
    Poll,
    /// Admitted-token observation.
    Observe,
    /// Prefill-boundary observation.
    Prefill,
}

/// Bounded error or panic isolated at a callback boundary.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallbackFailure {
    /// Callback phase.
    pub phase: CallbackPhase,
    /// Whether the callback unwound and was contained.
    pub panicked: bool,
    /// Bounded human-readable detail.
    pub message: String,
}

impl CallbackFailure {
    /// Creates a bounded failure description.
    pub fn new(phase: CallbackPhase, panicked: bool, message: impl Into<String>) -> Self {
        let mut message = message.into();
        if message.len() > MAX_RETAINED_FAILURE_BYTES {
            let mut end = MAX_RETAINED_FAILURE_BYTES;
            while !message.is_char_boundary(end) {
                end -= 1;
            }
            message.truncate(end);
        }
        Self {
            phase,
            panicked,
            message,
        }
    }
}

/// Mechanical accounting for one transform stage.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageReceipt {
    /// Exact stage contract.
    pub specification: TransformSpec,
    /// Reset invocations.
    pub resets: u32,
    /// Apply invocations.
    pub invocations: u32,
    /// Candidate values presented across all invocations.
    pub candidates_seen: u64,
    /// Candidate logits changed by this stage.
    pub logits_changed: u64,
    /// Causally admitted tokens reported to this stage.
    pub accepted_tokens: u32,
    /// First contained failure, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<CallbackFailure>,
}

/// Aggregate accounting for an ordered transform pipeline.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineReceipt {
    /// Exact ordered pipeline contract.
    pub specification: PipelineSpec,
    /// Pipeline identity.
    pub pipeline: Digest,
    /// Begin invocations.
    pub begins: u32,
    /// Complete pipeline apply invocations.
    pub invocations: u32,
    /// Candidate values copied into the pipeline boundary.
    pub candidates_copied: u64,
    /// Candidate values committed after every stage succeeded.
    pub candidates_committed: u64,
    /// First failed stage index, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failed_stage: Option<u32>,
    /// Per-stage accounting in execution order.
    pub stages: Vec<StageReceipt>,
}

impl PipelineReceipt {
    /// Validates structural accounting against the pipeline specification.
    ///
    /// # Errors
    ///
    /// Returns an error when identities, stage accounting, or failure
    /// attribution are inconsistent.
    pub fn validate(&self) -> Result<(), CoreError> {
        self.specification.validate()?;
        if self.pipeline != self.specification.digest()?
            || self.begins > 1
            || self.candidates_committed > self.candidates_copied
            || self.stages.len() != self.specification.stages.len()
        {
            return Err(CoreError::invalid(
                "pipeline receipt",
                "pipeline identity or aggregate accounting is inconsistent",
            ));
        }

        let maximum_stage_invocations = self.invocations.saturating_add(1);
        let failed_stage = self
            .failed_stage
            .map(usize::try_from)
            .transpose()
            .map_err(|_| CoreError::invalid("pipeline receipt", "failed stage exceeds usize"))?;
        if failed_stage.is_some_and(|index| index >= self.stages.len()) {
            return Err(CoreError::invalid(
                "pipeline receipt",
                "failed stage is outside the pipeline",
            ));
        }

        let mut recorded_failure = None;
        for (index, (stage, specification)) in self
            .stages
            .iter()
            .zip(&self.specification.stages)
            .enumerate()
        {
            if stage.specification != *specification
                || stage.resets > self.begins
                || stage.invocations < self.invocations
                || stage.invocations > maximum_stage_invocations
                || stage.invocations > stage.specification.max_steps
                || stage.accepted_tokens > self.invocations
                || stage.candidates_seen > self.candidates_copied
                || stage.logits_changed > stage.candidates_seen
            {
                return Err(CoreError::invalid(
                    "pipeline receipt",
                    "per-stage accounting is inconsistent",
                ));
            }
            if stage.failure.is_some() && recorded_failure.replace(index).is_some() {
                return Err(CoreError::invalid(
                    "pipeline receipt",
                    "more than one stage records a failure",
                ));
            }
        }
        if recorded_failure != failed_stage {
            return Err(CoreError::invalid(
                "pipeline receipt",
                "failure attribution is inconsistent",
            ));
        }
        if self.begins == 0
            && (self.invocations != 0
                || self.candidates_copied != 0
                || self.candidates_committed != 0)
        {
            return Err(CoreError::invalid(
                "pipeline receipt",
                "unbegun pipeline contains execution accounting",
            ));
        }
        Ok(())
    }

    /// Returns a content identity for validated mechanical accounting.
    ///
    /// # Errors
    ///
    /// Returns a validation or serialization error.
    pub fn digest(&self) -> Result<Digest, CoreError> {
        self.validate()?;
        Digest::of_serializable("transform-pipeline-receipt-v1", self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn callback_messages_are_utf8_safely_bounded() {
        let failure = CallbackFailure::new(
            CallbackPhase::Apply,
            false,
            "🧶".repeat(MAX_RETAINED_FAILURE_BYTES),
        );
        assert!(failure.message.len() <= MAX_RETAINED_FAILURE_BYTES);
        assert!(failure.message.is_char_boundary(failure.message.len()));
    }

    #[test]
    fn pipeline_receipts_validate_identity_and_stage_accounting() {
        let specification = PipelineSpec {
            stages: vec![
                TransformSpec::new(
                    Digest::of_bytes("test-transform", b"one"),
                    CandidateMode::FullVocabulary,
                    2,
                )
                .unwrap(),
            ],
        };
        let mut receipt = PipelineReceipt {
            pipeline: specification.digest().unwrap(),
            specification: specification.clone(),
            begins: 1,
            invocations: 1,
            candidates_copied: 2,
            candidates_committed: 2,
            failed_stage: None,
            stages: vec![StageReceipt {
                specification: specification.stages[0].clone(),
                resets: 1,
                invocations: 1,
                candidates_seen: 2,
                logits_changed: 1,
                accepted_tokens: 1,
                failure: None,
            }],
        };
        assert!(receipt.digest().is_ok());

        receipt.stages[0].accepted_tokens = 2;
        assert!(receipt.digest().is_err());
        receipt.stages[0].accepted_tokens = 1;
        receipt.pipeline = Digest::of_bytes("test-pipeline", b"wrong");
        assert!(receipt.digest().is_err());
    }
}

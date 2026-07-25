// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded construction of content-identified transform pipelines.

use std::collections::BTreeMap;

use logit_loom::{
    CandidateMode, CoreError, Digest, LogitTransform, MAX_LOGIT_BIASES, MAX_PIPELINE_STAGES,
    Pipeline, RankBias, Stage, TokenBias, TokenId, TransformSpec,
};
use serde::Serialize;

use crate::{Error, Result};

const RANK_BIAS_IDENTITY_DOMAIN: &str = "runtime-rank-bias-v1";
const TOKEN_BIAS_IDENTITY_DOMAIN: &str = "runtime-token-bias-v1";

#[derive(Serialize)]
struct RankBiasIdentity {
    rank: u64,
    bias_bits: u32,
}

#[derive(Serialize)]
struct TokenBiasIdentity {
    biases: Vec<(i32, u32)>,
}

/// Constructs an ordered [`Pipeline`] with one shared exposure mode and bound.
///
/// Built-in stages receive deterministic versioned implementation identities.
/// Custom stages require the caller to supply an identity that binds the
/// implementation and its configuration.
pub struct PipelineBuilder {
    mode: CandidateMode,
    max_steps: u32,
    stages: Vec<Stage>,
}

impl PipelineBuilder {
    /// Creates an empty builder with an explicit candidate view and step bound.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid sparse bound or zero steps.
    pub fn new(mode: CandidateMode, max_steps: u32) -> Result<Self> {
        mode.validate()?;
        if max_steps == 0 {
            return Err(
                CoreError::invalid("pipeline maximum steps", "must be greater than zero").into(),
            );
        }
        Ok(Self {
            mode,
            max_steps,
            stages: Vec::new(),
        })
    }

    /// Returns the candidate exposure shared by every stage.
    pub const fn mode(&self) -> CandidateMode {
        self.mode
    }

    /// Returns the maximum transform invocations permitted per generation call.
    pub const fn max_steps(&self) -> u32 {
        self.max_steps
    }

    /// Returns the number of stages currently declared.
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Appends a first-party dynamic rank bias.
    ///
    /// Its automatic implementation identity binds the zero-based rank and
    /// exact finite bias bits. Candidate mode and call bound remain bound by
    /// the enclosing [`TransformSpec`].
    ///
    /// # Errors
    ///
    /// Returns an error for an unsupported rank representation, invalid bias,
    /// or excessive stage count.
    pub fn rank_bias(mut self, rank: usize, bias: f32) -> Result<Self> {
        self.ensure_stage_capacity()?;
        let transform = RankBias::new(rank, bias)?;
        let rank = u64::try_from(rank).map_err(|_| {
            Error::from(CoreError::invalid(
                "rank bias",
                "rank exceeds the portable u64 identity representation",
            ))
        })?;
        let identity = Digest::of_serializable(
            RANK_BIAS_IDENTITY_DOMAIN,
            &RankBiasIdentity {
                rank,
                bias_bits: bias.to_bits(),
            },
        )?;
        self.push_stage(identity, transform)?;
        Ok(self)
    }

    /// Appends a first-party token-bias map.
    ///
    /// Repeated token IDs use the last value, matching [`TokenBias`]. The
    /// automatic identity binds that normalized map in ascending token order.
    ///
    /// # Errors
    ///
    /// Returns an error for a non-finite value, excessive input or stage count,
    /// or identity serialization failure.
    pub fn token_bias(mut self, biases: impl IntoIterator<Item = (TokenId, f32)>) -> Result<Self> {
        self.ensure_stage_capacity()?;
        let entries = biases
            .into_iter()
            .take(MAX_LOGIT_BIASES + 1)
            .collect::<Vec<_>>();
        if entries.len() > MAX_LOGIT_BIASES {
            return Err(CoreError::invalid(
                "token bias input",
                format!("requires at most {MAX_LOGIT_BIASES} entries"),
            )
            .into());
        }
        let transform = TokenBias::new(entries.iter().copied())?;
        let mut normalized = BTreeMap::new();
        for (token, bias) in entries {
            normalized.insert(token, bias.to_bits());
        }
        let identity = Digest::of_serializable(
            TOKEN_BIAS_IDENTITY_DOMAIN,
            &TokenBiasIdentity {
                biases: normalized
                    .into_iter()
                    .map(|(token, bits)| (token.get(), bits))
                    .collect(),
            },
        )?;
        self.push_stage(identity, transform)?;
        Ok(self)
    }

    /// Appends a custom transform with a caller-defined stable identity.
    ///
    /// # Errors
    ///
    /// Returns an error when the pipeline already contains the maximum number
    /// of stages or the generated stage contract is invalid.
    pub fn stage(
        mut self,
        implementation: Digest,
        transform: impl LogitTransform + 'static,
    ) -> Result<Self> {
        self.ensure_stage_capacity()?;
        self.push_stage(implementation, transform)?;
        Ok(self)
    }

    /// Builds the ordered pipeline.
    ///
    /// # Errors
    ///
    /// Returns an error unless at least one valid stage was declared.
    pub fn build(self) -> Result<Pipeline> {
        Pipeline::new(self.stages).map_err(Error::from)
    }

    fn push_stage(
        &mut self,
        implementation: Digest,
        transform: impl LogitTransform + 'static,
    ) -> Result<()> {
        self.ensure_stage_capacity()?;
        let specification = TransformSpec::new(implementation, self.mode, self.max_steps)?;
        self.stages.push(Stage::new(specification, transform)?);
        Ok(())
    }

    fn ensure_stage_capacity(&self) -> Result<()> {
        if self.stages.len() >= MAX_PIPELINE_STAGES {
            return Err(CoreError::invalid(
                "pipeline stages",
                format!("requires at most {MAX_PIPELINE_STAGES} stages"),
            )
            .into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn built_in_rank_identity_is_deterministic_and_configuration_bound() {
        let first = PipelineBuilder::new(CandidateMode::FullVocabulary, 2)
            .unwrap()
            .rank_bias(1, 2.0)
            .unwrap()
            .build()
            .unwrap();
        let same = PipelineBuilder::new(CandidateMode::FullVocabulary, 2)
            .unwrap()
            .rank_bias(1, 2.0)
            .unwrap()
            .build()
            .unwrap();
        let different = PipelineBuilder::new(CandidateMode::FullVocabulary, 2)
            .unwrap()
            .rank_bias(0, 2.0)
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(
            first.specification().stages[0].implementation,
            same.specification().stages[0].implementation
        );
        assert_ne!(
            first.specification().stages[0].implementation,
            different.specification().stages[0].implementation
        );
    }

    #[test]
    fn token_bias_identity_normalizes_duplicate_inputs() {
        let token = TokenId::new(7).unwrap();
        let duplicate = PipelineBuilder::new(CandidateMode::FullVocabulary, 1)
            .unwrap()
            .token_bias([(token, 1.0), (token, 2.0)])
            .unwrap()
            .build()
            .unwrap();
        let normalized = PipelineBuilder::new(CandidateMode::FullVocabulary, 1)
            .unwrap()
            .token_bias([(token, 2.0)])
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(
            duplicate.specification().stages[0].implementation,
            normalized.specification().stages[0].implementation
        );
    }

    #[test]
    fn rank_bias_pipeline_runs_against_an_in_memory_vocabulary() {
        let mut pipeline = PipelineBuilder::new(CandidateMode::FullVocabulary, 1)
            .unwrap()
            .rank_bias(1, 4.0)
            .unwrap()
            .build()
            .unwrap();
        pipeline.begin(&[]).unwrap();
        let mut logits = [3.0, 2.0, 1.0];
        pipeline.apply_to_vocabulary(0, &[], &mut logits).unwrap();
        assert_eq!(
            logits.map(f32::to_bits),
            [3.0_f32.to_bits(), 6.0_f32.to_bits(), 1.0_f32.to_bits()]
        );
    }

    #[test]
    fn empty_and_zero_bound_builders_are_rejected() {
        assert!(PipelineBuilder::new(CandidateMode::FullVocabulary, 0).is_err());
        assert!(
            PipelineBuilder::new(CandidateMode::FullVocabulary, 1)
                .unwrap()
                .build()
                .is_err()
        );
    }

    #[test]
    fn stage_bound_is_enforced_by_the_builder() {
        let mut builder = PipelineBuilder::new(CandidateMode::FullVocabulary, 1).unwrap();
        for rank in 0..MAX_PIPELINE_STAGES {
            builder = builder.rank_bias(rank, 1.0).unwrap();
        }
        assert_eq!(builder.stage_count(), MAX_PIPELINE_STAGES);
        assert!(builder.rank_bias(MAX_PIPELINE_STAGES, 1.0).is_err());
    }
}

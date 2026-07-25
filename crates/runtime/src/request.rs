// SPDX-License-Identifier: MIT OR Apache-2.0

//! Validated bounded generation requests with optional borrowed controls.

use logit_loom::{
    CoreError, GenerationPlan, Grammar, LogitBias, MAX_LOGIT_BIASES, MAX_STOP_SEQUENCE_BYTES,
    MAX_STOP_SEQUENCES, ObserverSet, Pipeline, SamplingPlan, TokenId,
};

use crate::Result;

/// One bounded generation request and its optional execution controls.
///
/// The serializable [`GenerationPlan`] remains the durable contract. Pipelines
/// and observers are borrowed only for the synchronous generation call.
pub struct GenerationRequest<'controls> {
    plan: GenerationPlan,
    pipeline: Option<&'controls mut Pipeline>,
    observers: Option<&'controls mut ObserverSet>,
}

impl<'controls> GenerationRequest<'controls> {
    /// Creates a request using the existing default native sampling plan.
    ///
    /// # Errors
    ///
    /// Returns an error when `max_tokens` is zero.
    pub fn new(max_tokens: u32) -> Result<Self> {
        Self::from_plan(GenerationPlan {
            sampling: SamplingPlan::default(),
            max_tokens,
            biases: Vec::new(),
            grammar: None,
            stops: Vec::new(),
        })
    }

    /// Creates a request from an exact backend-neutral plan.
    ///
    /// # Errors
    ///
    /// Returns an error when the plan is invalid.
    pub fn from_plan(plan: GenerationPlan) -> Result<Self> {
        plan.validate()?;
        Ok(Self {
            plan,
            pipeline: None,
            observers: None,
        })
    }

    /// Replaces the native sampling configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when the resulting plan is invalid.
    pub fn sampling(mut self, sampling: SamplingPlan) -> Result<Self> {
        self.plan.sampling = sampling;
        self.plan.validate()?;
        Ok(self)
    }

    /// Appends one unique finite additive token bias.
    ///
    /// # Errors
    ///
    /// Returns an error for a duplicate token, non-finite bias, or collection
    /// bound violation.
    pub fn bias(mut self, token: TokenId, bias: f32) -> Result<Self> {
        if self.plan.biases.len() >= MAX_LOGIT_BIASES || !bias.is_finite() {
            return Err(CoreError::invalid(
                "logit biases",
                "too many biases or a non-finite value",
            )
            .into());
        }
        if self.plan.biases.iter().any(|entry| entry.token == token) {
            return Err(
                CoreError::invalid("logit biases", "token identifiers must be unique").into(),
            );
        }
        self.plan.biases.push(LogitBias { token, bias });
        self.plan.validate()?;
        Ok(self)
    }

    /// Replaces the optional eager grammar.
    ///
    /// # Errors
    ///
    /// Returns an error when the grammar or resulting plan is invalid.
    pub fn grammar(mut self, grammar: Grammar) -> Result<Self> {
        self.plan.grammar = Some(grammar);
        self.plan.validate()?;
        Ok(self)
    }

    /// Appends one exact byte stop suffix.
    ///
    /// # Errors
    ///
    /// Returns an error when the suffix is empty or oversized, or the stop
    /// collection exceeds its public bound.
    pub fn stop_bytes(mut self, stop: impl AsRef<[u8]>) -> Result<Self> {
        let stop = stop.as_ref();
        if self.plan.stops.len() >= MAX_STOP_SEQUENCES
            || stop.is_empty()
            || stop.len() > MAX_STOP_SEQUENCE_BYTES
        {
            return Err(CoreError::invalid(
                "stop sequences",
                format!(
                    "requires at most {MAX_STOP_SEQUENCES} non-empty sequences of at most \
                     {MAX_STOP_SEQUENCE_BYTES} bytes"
                ),
            )
            .into());
        }
        self.plan.stops.push(stop.to_vec());
        self.plan.validate()?;
        Ok(self)
    }

    /// Attaches one mutable transform pipeline for the synchronous call.
    ///
    /// # Errors
    ///
    /// Returns an error if a pipeline was already attached.
    pub fn pipeline(mut self, pipeline: &'controls mut Pipeline) -> Result<Self> {
        if self.pipeline.is_some() {
            return Err(CoreError::invalid(
                "generation pipeline",
                "only one pipeline may be attached",
            )
            .into());
        }
        self.pipeline = Some(pipeline);
        Ok(self)
    }

    /// Attaches one mutable observer set for the synchronous call.
    ///
    /// # Errors
    ///
    /// Returns an error if an observer set was already attached.
    pub fn observers(mut self, observers: &'controls mut ObserverSet) -> Result<Self> {
        if self.observers.is_some() {
            return Err(CoreError::invalid(
                "generation observers",
                "only one observer set may be attached",
            )
            .into());
        }
        self.observers = Some(observers);
        Ok(self)
    }

    /// Returns the exact serializable generation plan.
    pub const fn plan(&self) -> &GenerationPlan {
        &self.plan
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        GenerationPlan,
        Option<&'controls mut Pipeline>,
        Option<&'controls mut ObserverSet>,
    ) {
        (self.plan, self.pipeline, self.observers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maximum_tokens_are_required() {
        assert!(GenerationRequest::new(0).is_err());
        assert_eq!(GenerationRequest::new(1).unwrap().plan().max_tokens, 1);
    }

    #[test]
    fn fluent_fields_preserve_exact_plan_mechanics() {
        let token = TokenId::new(7).unwrap();
        let request = GenerationRequest::new(12)
            .unwrap()
            .bias(token, 1.5)
            .unwrap()
            .stop_bytes([0xff, 0x00])
            .unwrap();
        assert_eq!(request.plan().biases, [LogitBias { token, bias: 1.5 }]);
        assert_eq!(request.plan().stops, [vec![0xff, 0x00]]);
    }

    #[test]
    fn duplicate_bias_and_empty_stop_fail_before_execution() {
        let token = TokenId::new(7).unwrap();
        assert!(
            GenerationRequest::new(1)
                .unwrap()
                .bias(token, 1.0)
                .unwrap()
                .bias(token, 2.0)
                .is_err()
        );
        assert!(
            GenerationRequest::new(1)
                .unwrap()
                .bias(token, f32::NAN)
                .is_err()
        );
        assert!(GenerationRequest::new(1).unwrap().stop_bytes([]).is_err());
        let oversized = vec![0; MAX_STOP_SEQUENCE_BYTES + 1];
        assert!(
            GenerationRequest::new(1)
                .unwrap()
                .stop_bytes(&oversized)
                .is_err()
        );
    }

    #[test]
    fn controls_cannot_be_attached_twice() {
        let mut first = crate::PipelineBuilder::new(crate::CandidateMode::FullVocabulary, 1)
            .unwrap()
            .rank_bias(0, 1.0)
            .unwrap()
            .build()
            .unwrap();
        let mut second = crate::PipelineBuilder::new(crate::CandidateMode::FullVocabulary, 1)
            .unwrap()
            .rank_bias(1, 1.0)
            .unwrap()
            .build()
            .unwrap();
        let request = GenerationRequest::new(1)
            .unwrap()
            .pipeline(&mut first)
            .unwrap();
        assert!(request.pipeline(&mut second).is_err());
    }
}

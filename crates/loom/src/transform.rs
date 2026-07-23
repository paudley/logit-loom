// SPDX-License-Identifier: MIT OR Apache-2.0

//! Ordered, transactional logit-transform pipelines.

use std::any::Any;
use std::collections::HashSet;
use std::panic::{AssertUnwindSafe, catch_unwind};

use logit_loom_core::{
    CallbackFailure, CallbackPhase, CandidateMode, MAX_PIPELINE_STAGES, PipelineReceipt,
    PipelineSpec, StageReceipt, TokenId, TransformSpec,
};

use crate::{PipelineError, TransformError};

/// Borrowed state supplied to one Rust logit-transform invocation.
pub struct TransformContext<'a> {
    step: u32,
    causal_tokens: &'a [TokenId],
    candidate_tokens: &'a [TokenId],
    logits: &'a mut [f32],
}

impl<'a> TransformContext<'a> {
    /// Creates a validated borrowed candidate view.
    ///
    /// # Errors
    ///
    /// Returns an error when token and logit lengths disagree.
    pub fn new(
        step: u32,
        causal_tokens: &'a [TokenId],
        candidate_tokens: &'a [TokenId],
        logits: &'a mut [f32],
    ) -> Result<Self, PipelineError> {
        if candidate_tokens.len() != logits.len() || candidate_tokens.is_empty() {
            return Err(PipelineError::InvalidCandidates(
                "token and logit arrays must be non-empty and equal in length".to_owned(),
            ));
        }
        Ok(Self {
            step,
            causal_tokens,
            candidate_tokens,
            logits,
        })
    }

    /// Zero-based sampling step.
    pub const fn step(&self) -> u32 {
        self.step
    }

    /// Exact prompt and previously admitted generation tokens.
    pub const fn causal_tokens(&self) -> &[TokenId] {
        self.causal_tokens
    }

    /// Immutable candidate token identifiers.
    pub const fn candidate_tokens(&self) -> &[TokenId] {
        self.candidate_tokens
    }

    /// Mutable candidate logits corresponding one-for-one with token IDs.
    pub fn logits_mut(&mut self) -> &mut [f32] {
        self.logits
    }

    /// Iterates token/logit pairs without permitting token-ID mutation.
    pub fn candidates_mut(&mut self) -> impl Iterator<Item = (TokenId, &mut f32)> {
        self.candidate_tokens
            .iter()
            .copied()
            .zip(self.logits.iter_mut())
    }
}

/// Safe, synchronous implementation of one per-step logit transformation.
pub trait LogitTransform {
    /// Resets implementation-local state for a fresh generation call.
    ///
    /// # Errors
    ///
    /// Returns an implementation-defined reset failure.
    fn reset(&mut self, _causal_tokens: &[TokenId]) -> Result<(), TransformError> {
        Ok(())
    }

    /// Mutates one candidate-logit view.
    ///
    /// # Errors
    ///
    /// Returns an implementation-defined transformation failure.
    fn apply(&mut self, context: TransformContext<'_>) -> Result<(), TransformError>;

    /// Observes a token after the backend has admitted it to causal state.
    ///
    /// # Errors
    ///
    /// Returns an implementation-defined token-admission failure.
    fn accept(&mut self, _token: TokenId) -> Result<(), TransformError> {
        Ok(())
    }
}

/// One identified implementation in an ordered pipeline.
pub struct Stage {
    specification: TransformSpec,
    implementation: Box<dyn LogitTransform>,
}

impl Stage {
    /// Wraps a safe Rust transform with its mechanical contract.
    ///
    /// # Errors
    ///
    /// Returns an error when the specification is invalid.
    pub fn new(
        specification: TransformSpec,
        implementation: impl LogitTransform + 'static,
    ) -> Result<Self, PipelineError> {
        specification.validate()?;
        Ok(Self {
            specification,
            implementation: Box::new(implementation),
        })
    }

    /// Returns the stage contract.
    pub const fn specification(&self) -> &TransformSpec {
        &self.specification
    }
}

/// Ordered, call-scoped transform pipeline.
///
/// Candidate changes are committed to the backend view only after every stage
/// succeeds. A returned error therefore cannot expose a partially transformed
/// vocabulary, although implementation-local state in stages that already ran
/// may have changed. Begin a fresh call before retrying.
pub struct Pipeline {
    specification: PipelineSpec,
    stages: Vec<Stage>,
    receipt: PipelineReceipt,
    begun: bool,
    failed: bool,
}

impl Pipeline {
    /// Creates an ordered pipeline of one to 32 compatible stages.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid stage count or incompatible modes.
    pub fn new(stages: Vec<Stage>) -> Result<Self, PipelineError> {
        let specification = PipelineSpec {
            stages: stages
                .iter()
                .take(MAX_PIPELINE_STAGES + 1)
                .map(|stage| stage.specification.clone())
                .collect(),
        };
        specification.validate()?;
        let receipt = empty_receipt(&specification)?;
        Ok(Self {
            specification,
            stages,
            receipt,
            begun: false,
            failed: false,
        })
    }

    /// Returns the immutable ordered specification.
    pub const fn specification(&self) -> &PipelineSpec {
        &self.specification
    }

    /// Returns current mechanical accounting.
    pub fn receipt(&self) -> PipelineReceipt {
        self.receipt.clone()
    }

    /// Returns whether a callback has failed since the last begin.
    pub const fn is_failed(&self) -> bool {
        self.failed
    }

    /// Resets every stage for a new generation call.
    ///
    /// # Errors
    ///
    /// Returns a contained callback failure. Calling `begin` again constructs
    /// fresh accounting and asks every stage to reset again.
    pub fn begin(&mut self, causal_tokens: &[TokenId]) -> Result<(), PipelineError> {
        self.receipt = empty_receipt(&self.specification)?;
        self.receipt.begins = 1;
        self.begun = true;
        self.failed = false;
        for index in 0..self.stages.len() {
            self.receipt.stages[index].resets = 1;
            let result = catch_unwind(AssertUnwindSafe(|| {
                self.stages[index].implementation.reset(causal_tokens)
            }));
            if let Some(failure) = callback_result(CallbackPhase::Reset, result) {
                return Err(self.record_failure(index, failure));
            }
        }
        Ok(())
    }

    /// Applies the pipeline to a complete vocabulary-logit slice.
    ///
    /// Sparse pipelines select the highest finite raw logits, breaking ties by
    /// ascending token ID. Full pipelines expose tokenizer order. Candidate
    /// output containing NaN is rejected before write-back; infinity remains a
    /// valid explicit suppression or promotion value. Except for a terminal
    /// selection, the prior successful step must be matched by [`Self::accept`]
    /// before the next step begins.
    ///
    /// # Errors
    ///
    /// Returns an error when the call has not begun, bounds are exceeded, the
    /// vocabulary is invalid, lifecycle order is violated, or a callback
    /// fails.
    pub fn apply_to_vocabulary(
        &mut self,
        step: u32,
        causal_tokens: &[TokenId],
        vocabulary_logits: &mut [f32],
    ) -> Result<(), PipelineError> {
        self.require_ready()?;
        let maximum_vocabulary = usize::try_from(i32::MAX).unwrap_or(usize::MAX);
        if vocabulary_logits.is_empty() || vocabulary_logits.len() > maximum_vocabulary {
            return Err(PipelineError::InvalidCandidates(
                "vocabulary must fit non-negative i32 token identifiers".to_owned(),
            ));
        }
        let mode = self.specification.mode()?;
        let indices = selected_indices(mode, vocabulary_logits);
        if indices.is_empty() {
            return Err(PipelineError::InvalidCandidates(
                "candidate exposure selected no finite logits".to_owned(),
            ));
        }
        let candidate_tokens = indices
            .iter()
            .map(|index| {
                TokenId::new(i32::try_from(*index).map_err(|_| {
                    PipelineError::InvalidCandidates("token identifier overflowed".to_owned())
                })?)
                .map_err(PipelineError::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut candidate_logits = indices
            .iter()
            .map(|index| vocabulary_logits[*index])
            .collect::<Vec<_>>();
        self.apply_working_candidates(
            step,
            causal_tokens,
            &candidate_tokens,
            &mut candidate_logits,
        )?;

        for (index, value) in indices.into_iter().zip(candidate_logits) {
            vocabulary_logits[index] = value;
        }
        Ok(())
    }

    /// Applies the pipeline to a candidate view selected by a backend adapter.
    ///
    /// This entry point is for adapters whose native sampler has already copied
    /// token IDs and logits into an owned scratch view. The adapter remains
    /// responsible for proving that a full-mode view covers the vocabulary and
    /// that sparse candidates were selected according to its declared native
    /// boundary. Logit Loom enforces shape, sparse bounds, stage limits, panic
    /// containment, NaN rejection, and transactional write-back to this slice.
    ///
    /// # Errors
    ///
    /// Returns an error when the call has not begun, candidate arrays are empty
    /// or mismatched, token identifiers repeat, a sparse view exceeds its
    /// limit, lifecycle order is violated, or a callback fails.
    pub fn apply_to_candidates(
        &mut self,
        step: u32,
        causal_tokens: &[TokenId],
        candidate_tokens: &[TokenId],
        logits: &mut [f32],
    ) -> Result<(), PipelineError> {
        self.require_ready()?;
        self.validate_candidate_view(candidate_tokens, logits)?;
        let mut working_logits = logits.to_vec();
        self.apply_working_candidates(step, causal_tokens, candidate_tokens, &mut working_logits)?;
        logits.copy_from_slice(&working_logits);
        Ok(())
    }

    fn apply_working_candidates(
        &mut self,
        step: u32,
        causal_tokens: &[TokenId],
        candidate_tokens: &[TokenId],
        candidate_logits: &mut [f32],
    ) -> Result<(), PipelineError> {
        self.require_ready()?;
        if self
            .receipt
            .stages
            .iter()
            .any(|stage| stage.accepted_tokens < self.receipt.invocations)
        {
            return Err(PipelineError::AdmissionPending);
        }
        if step != self.receipt.invocations {
            return Err(PipelineError::InvalidStep {
                expected: self.receipt.invocations,
                actual: step,
            });
        }
        let candidate_count = u64::try_from(candidate_logits.len())
            .map_err(|_| PipelineError::AccountingOverflow("candidate count"))?;
        self.receipt.candidates_copied = self
            .receipt
            .candidates_copied
            .checked_add(candidate_count)
            .ok_or(PipelineError::AccountingOverflow("candidates copied"))?;

        for index in 0..self.stages.len() {
            if step >= self.stages[index].specification.max_steps
                || self.receipt.stages[index].invocations
                    >= self.stages[index].specification.max_steps
            {
                let failure = CallbackFailure::new(
                    CallbackPhase::Apply,
                    false,
                    format!(
                        "step {step} exceeds stage bound {}",
                        self.stages[index].specification.max_steps
                    ),
                );
                return Err(self.record_failure(index, failure));
            }
            let before = candidate_logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>();
            let context =
                TransformContext::new(step, causal_tokens, candidate_tokens, candidate_logits)?;
            self.receipt.stages[index].invocations = self.receipt.stages[index]
                .invocations
                .checked_add(1)
                .ok_or(PipelineError::AccountingOverflow("stage invocations"))?;
            self.receipt.stages[index].candidates_seen = self.receipt.stages[index]
                .candidates_seen
                .checked_add(candidate_count)
                .ok_or(PipelineError::AccountingOverflow("stage candidates seen"))?;
            let result = catch_unwind(AssertUnwindSafe(|| {
                self.stages[index].implementation.apply(context)
            }));
            if let Some(failure) = callback_result(CallbackPhase::Apply, result) {
                return Err(self.record_failure(index, failure));
            }
            if candidate_logits.iter().any(|value| value.is_nan()) {
                let failure = CallbackFailure::new(
                    CallbackPhase::Apply,
                    false,
                    "transform produced a NaN logit",
                );
                return Err(self.record_failure(index, failure));
            }
            let changed = before
                .iter()
                .zip(candidate_logits.iter())
                .filter(|(before, after)| **before != after.to_bits())
                .count();
            let changed = u64::try_from(changed)
                .map_err(|_| PipelineError::AccountingOverflow("changed logits"))?;
            self.receipt.stages[index].logits_changed = self.receipt.stages[index]
                .logits_changed
                .checked_add(changed)
                .ok_or(PipelineError::AccountingOverflow("stage logits changed"))?;
        }

        self.receipt.invocations = self
            .receipt
            .invocations
            .checked_add(1)
            .ok_or(PipelineError::AccountingOverflow("pipeline invocations"))?;
        self.receipt.candidates_committed = self
            .receipt
            .candidates_committed
            .checked_add(candidate_count)
            .ok_or(PipelineError::AccountingOverflow("candidates committed"))?;
        Ok(())
    }

    fn validate_candidate_view(
        &self,
        candidate_tokens: &[TokenId],
        logits: &[f32],
    ) -> Result<(), PipelineError> {
        if candidate_tokens.is_empty() || candidate_tokens.len() != logits.len() {
            return Err(PipelineError::InvalidCandidates(
                "token and logit arrays must be non-empty and equal in length".to_owned(),
            ));
        }
        let maximum_candidates = usize::try_from(i32::MAX).unwrap_or(usize::MAX);
        if candidate_tokens.len() > maximum_candidates {
            return Err(PipelineError::InvalidCandidates(
                "candidate count exceeds non-negative i32 token space".to_owned(),
            ));
        }
        let unique_tokens = candidate_tokens.iter().copied().collect::<HashSet<_>>();
        if unique_tokens.len() != candidate_tokens.len() {
            return Err(PipelineError::InvalidCandidates(
                "candidate token identifiers must be unique".to_owned(),
            ));
        }
        if let CandidateMode::Sparse { limit } = self.specification.mode()?
            && candidate_tokens.len() > usize::try_from(limit).unwrap_or(usize::MAX)
        {
            return Err(PipelineError::InvalidCandidates(format!(
                "candidate view exceeds sparse limit {limit}"
            )));
        }
        Ok(())
    }

    /// Reports one token after causal admission.
    ///
    /// Each admission must correspond to an unmatched successful transform
    /// invocation from the current call.
    ///
    /// # Errors
    ///
    /// Returns a contained callback failure or a lifecycle error.
    pub fn accept(&mut self, token: TokenId) -> Result<(), PipelineError> {
        self.require_ready()?;
        if self
            .receipt
            .stages
            .iter()
            .any(|stage| stage.accepted_tokens >= self.receipt.invocations)
        {
            return Err(PipelineError::UnexpectedAdmission);
        }
        for index in 0..self.stages.len() {
            self.receipt.stages[index].accepted_tokens = self.receipt.stages[index]
                .accepted_tokens
                .checked_add(1)
                .ok_or(PipelineError::AccountingOverflow("accepted tokens"))?;
            let result = catch_unwind(AssertUnwindSafe(|| {
                self.stages[index].implementation.accept(token)
            }));
            if let Some(failure) = callback_result(CallbackPhase::Accept, result) {
                return Err(self.record_failure(index, failure));
            }
        }
        Ok(())
    }

    fn require_ready(&self) -> Result<(), PipelineError> {
        if !self.begun {
            return Err(PipelineError::NotBegun);
        }
        if self.failed {
            return Err(PipelineError::Failed);
        }
        Ok(())
    }

    fn record_failure(&mut self, index: usize, failure: CallbackFailure) -> PipelineError {
        self.failed = true;
        self.receipt.failed_stage = Some(u32::try_from(index).unwrap_or(u32::MAX));
        self.receipt.stages[index].failure = Some(failure.clone());
        PipelineError::Callback {
            stage: index,
            failure,
        }
    }
}

fn empty_receipt(specification: &PipelineSpec) -> Result<PipelineReceipt, PipelineError> {
    Ok(PipelineReceipt {
        specification: specification.clone(),
        pipeline: specification.digest()?,
        begins: 0,
        invocations: 0,
        candidates_copied: 0,
        candidates_committed: 0,
        failed_stage: None,
        stages: specification
            .stages
            .iter()
            .cloned()
            .map(|specification| StageReceipt {
                specification,
                resets: 0,
                invocations: 0,
                candidates_seen: 0,
                logits_changed: 0,
                accepted_tokens: 0,
                failure: None,
            })
            .collect(),
    })
}

fn selected_indices(mode: CandidateMode, logits: &[f32]) -> Vec<usize> {
    match mode {
        CandidateMode::FullVocabulary => (0..logits.len()).collect(),
        CandidateMode::Sparse { limit } => {
            let mut indices = logits
                .iter()
                .enumerate()
                .filter(|(_, value)| value.is_finite())
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            indices.sort_unstable_by(|left, right| {
                logits[*right]
                    .total_cmp(&logits[*left])
                    .then_with(|| left.cmp(right))
            });
            indices.truncate(usize::try_from(limit).unwrap_or(usize::MAX));
            indices
        }
    }
}

fn callback_result(
    phase: CallbackPhase,
    result: Result<Result<(), TransformError>, Box<dyn Any + Send>>,
) -> Option<CallbackFailure> {
    match result {
        Ok(Ok(())) => None,
        Ok(Err(error)) => Some(CallbackFailure::new(phase, false, error.to_string())),
        Err(payload) => Some(CallbackFailure::new(phase, true, panic_message(&*payload))),
    }
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    payload.downcast_ref::<&str>().map_or_else(
        || {
            payload
                .downcast_ref::<String>()
                .cloned()
                .unwrap_or_else(|| "non-string panic payload".to_owned())
        },
        |message| (*message).to_owned(),
    )
}

#[cfg(test)]
mod tests {
    use logit_loom_core::Digest;

    use super::*;

    struct Add(f32);

    impl LogitTransform for Add {
        fn apply(&mut self, mut context: TransformContext<'_>) -> Result<(), TransformError> {
            for value in context.logits_mut() {
                *value += self.0;
            }
            Ok(())
        }
    }

    struct Panic;

    impl LogitTransform for Panic {
        fn apply(&mut self, _context: TransformContext<'_>) -> Result<(), TransformError> {
            panic!("contained")
        }
    }

    struct MakeNan;

    impl LogitTransform for MakeNan {
        fn apply(&mut self, mut context: TransformContext<'_>) -> Result<(), TransformError> {
            context.logits_mut()[0] = f32::NAN;
            Ok(())
        }
    }

    struct ResetPanics;

    impl LogitTransform for ResetPanics {
        fn reset(&mut self, _causal_tokens: &[TokenId]) -> Result<(), TransformError> {
            panic!("reset panic is contained")
        }

        fn apply(&mut self, _context: TransformContext<'_>) -> Result<(), TransformError> {
            Ok(())
        }
    }

    struct AcceptFails;

    impl LogitTransform for AcceptFails {
        fn apply(&mut self, _context: TransformContext<'_>) -> Result<(), TransformError> {
            Ok(())
        }

        fn accept(&mut self, _token: TokenId) -> Result<(), TransformError> {
            Err(TransformError::new("accept failure"))
        }
    }

    fn spec(name: &str, mode: CandidateMode) -> TransformSpec {
        TransformSpec::new(Digest::of_bytes("test", name.as_bytes()), mode, 4).unwrap()
    }

    #[test]
    fn ordered_pipeline_commits_only_after_every_stage() {
        let mut pipeline = Pipeline::new(vec![
            Stage::new(spec("one", CandidateMode::FullVocabulary), Add(1.0)).unwrap(),
            Stage::new(spec("two", CandidateMode::FullVocabulary), Add(2.0)).unwrap(),
        ])
        .unwrap();
        pipeline.begin(&[]).unwrap();
        let mut logits = [0.0, 1.0];
        pipeline.apply_to_vocabulary(0, &[], &mut logits).unwrap();
        assert_eq!(
            logits.map(f32::to_bits),
            [3.0_f32.to_bits(), 4.0_f32.to_bits()]
        );
        assert_eq!(pipeline.receipt().stages[0].logits_changed, 2);
        assert_eq!(pipeline.receipt().stages[1].logits_changed, 2);
    }

    #[test]
    fn callback_panic_does_not_commit_partial_logits() {
        let mut pipeline = Pipeline::new(vec![
            Stage::new(spec("one", CandidateMode::FullVocabulary), Add(1.0)).unwrap(),
            Stage::new(spec("panic", CandidateMode::FullVocabulary), Panic).unwrap(),
        ])
        .unwrap();
        pipeline.begin(&[]).unwrap();
        let mut logits = [0.0, 1.0];
        assert!(pipeline.apply_to_vocabulary(0, &[], &mut logits).is_err());
        assert_eq!(
            logits.map(f32::to_bits),
            [0.0_f32.to_bits(), 1.0_f32.to_bits()]
        );
        assert!(pipeline.is_failed());
        assert!(
            pipeline.receipt().stages[1]
                .failure
                .as_ref()
                .unwrap()
                .panicked
        );
    }

    #[test]
    fn sparse_mode_exposes_ranked_candidates_and_writes_back_by_token() {
        let mut pipeline = Pipeline::new(vec![
            Stage::new(spec("sparse", CandidateMode::Sparse { limit: 2 }), Add(1.0)).unwrap(),
        ])
        .unwrap();
        pipeline.begin(&[]).unwrap();
        let mut logits = [0.0, 3.0, 2.0, 1.0];
        pipeline.apply_to_vocabulary(0, &[], &mut logits).unwrap();
        assert_eq!(
            logits.map(f32::to_bits),
            [
                0.0_f32.to_bits(),
                4.0_f32.to_bits(),
                3.0_f32.to_bits(),
                1.0_f32.to_bits(),
            ]
        );
    }

    #[test]
    fn sparse_ties_break_by_ascending_token_identifier() {
        let mut pipeline = Pipeline::new(vec![
            Stage::new(spec("ties", CandidateMode::Sparse { limit: 2 }), Add(1.0)).unwrap(),
        ])
        .unwrap();
        pipeline.begin(&[]).unwrap();
        let mut logits = [0.0, 2.0, 2.0, 2.0];
        pipeline.apply_to_vocabulary(0, &[], &mut logits).unwrap();
        assert_eq!(
            logits.map(f32::to_bits),
            [
                0.0_f32.to_bits(),
                3.0_f32.to_bits(),
                3.0_f32.to_bits(),
                2.0_f32.to_bits(),
            ]
        );
    }

    #[test]
    fn backend_selected_candidate_view_is_bounded_and_transactional() {
        let mut pipeline = Pipeline::new(vec![
            Stage::new(
                spec("selected", CandidateMode::Sparse { limit: 2 }),
                Add(1.0),
            )
            .unwrap(),
        ])
        .unwrap();
        let causal = [TokenId::new(7).unwrap()];
        let candidates = [TokenId::new(41).unwrap(), TokenId::new(3).unwrap()];
        pipeline.begin(&causal).unwrap();
        let mut logits = [2.0, 1.0];
        pipeline
            .apply_to_candidates(0, &causal, &candidates, &mut logits)
            .unwrap();
        assert_eq!(
            logits.map(f32::to_bits),
            [3.0_f32.to_bits(), 2.0_f32.to_bits()]
        );
        let receipt = pipeline.receipt();
        assert_eq!(receipt.candidates_copied, 2);
        assert_eq!(receipt.candidates_committed, 2);

        let mut oversized = [0.0, 1.0, 2.0];
        let oversized_tokens = [
            TokenId::new(1).unwrap(),
            TokenId::new(2).unwrap(),
            TokenId::new(3).unwrap(),
        ];
        assert!(
            pipeline
                .apply_to_candidates(1, &causal, &oversized_tokens, &mut oversized)
                .is_err()
        );
        assert_eq!(
            oversized.map(f32::to_bits),
            [0.0_f32.to_bits(), 1.0_f32.to_bits(), 2.0_f32.to_bits()]
        );
    }

    #[test]
    fn nan_output_fails_without_writeback() {
        let mut pipeline = Pipeline::new(vec![
            Stage::new(spec("nan", CandidateMode::FullVocabulary), MakeNan).unwrap(),
        ])
        .unwrap();
        pipeline.begin(&[]).unwrap();
        let mut logits = [1.0, 2.0];
        assert!(pipeline.apply_to_vocabulary(0, &[], &mut logits).is_err());
        assert_eq!(
            logits.map(f32::to_bits),
            [1.0_f32.to_bits(), 2.0_f32.to_bits()]
        );
        assert!(pipeline.is_failed());
    }

    #[test]
    fn sampling_steps_are_zero_based_and_sequential() {
        let mut pipeline = Pipeline::new(vec![
            Stage::new(spec("sequence", CandidateMode::FullVocabulary), Add(1.0)).unwrap(),
        ])
        .unwrap();
        pipeline.begin(&[]).unwrap();
        let mut logits = [0.0];
        assert!(matches!(
            pipeline.apply_to_vocabulary(1, &[], &mut logits),
            Err(PipelineError::InvalidStep {
                expected: 0,
                actual: 1
            })
        ));
        pipeline.apply_to_vocabulary(0, &[], &mut logits).unwrap();
        pipeline.accept(TokenId::new(1).unwrap()).unwrap();
        assert!(matches!(
            pipeline.apply_to_vocabulary(0, &[], &mut logits),
            Err(PipelineError::InvalidStep {
                expected: 1,
                actual: 0
            })
        ));
    }

    #[test]
    fn candidate_views_reject_duplicate_token_identifiers() {
        let mut pipeline = Pipeline::new(vec![
            Stage::new(spec("duplicates", CandidateMode::FullVocabulary), Add(1.0)).unwrap(),
        ])
        .unwrap();
        pipeline.begin(&[]).unwrap();
        let token = TokenId::new(3).unwrap();
        let mut logits = [1.0, 2.0];
        assert!(
            pipeline
                .apply_to_candidates(0, &[], &[token, token], &mut logits)
                .is_err()
        );
        assert_eq!(
            logits.map(f32::to_bits),
            [1.0_f32.to_bits(), 2.0_f32.to_bits()]
        );
    }

    #[test]
    fn invocation_bound_cannot_be_bypassed_with_repeated_steps() {
        let specification = TransformSpec::new(
            Digest::of_bytes("test", b"one-step"),
            CandidateMode::FullVocabulary,
            1,
        )
        .unwrap();
        let mut pipeline =
            Pipeline::new(vec![Stage::new(specification, Add(1.0)).unwrap()]).unwrap();
        pipeline.begin(&[]).unwrap();
        let mut logits = [0.0];
        pipeline.apply_to_vocabulary(0, &[], &mut logits).unwrap();
        assert!(pipeline.apply_to_vocabulary(0, &[], &mut logits).is_err());
        assert_eq!(pipeline.receipt().invocations, 1);
    }

    #[test]
    fn admission_requires_an_unmatched_successful_step() {
        let mut pipeline = Pipeline::new(vec![
            Stage::new(spec("admission", CandidateMode::FullVocabulary), Add(1.0)).unwrap(),
        ])
        .unwrap();
        pipeline.begin(&[]).unwrap();
        let token = TokenId::new(1).unwrap();
        assert!(matches!(
            pipeline.accept(token),
            Err(PipelineError::UnexpectedAdmission)
        ));

        let mut logits = [0.0];
        pipeline.apply_to_vocabulary(0, &[], &mut logits).unwrap();
        assert!(matches!(
            pipeline.apply_to_vocabulary(1, &[], &mut logits),
            Err(PipelineError::AdmissionPending)
        ));
        pipeline.accept(token).unwrap();
        assert!(matches!(
            pipeline.accept(token),
            Err(PipelineError::UnexpectedAdmission)
        ));
    }

    #[test]
    fn reset_and_admission_failures_are_contained_and_accounted() {
        let mut reset = Pipeline::new(vec![
            Stage::new(
                spec("reset-panic", CandidateMode::FullVocabulary),
                ResetPanics,
            )
            .unwrap(),
        ])
        .unwrap();
        assert!(reset.begin(&[]).is_err());
        let failure = reset.receipt().stages[0].failure.clone().unwrap();
        assert!(failure.panicked);
        assert_eq!(failure.phase, CallbackPhase::Reset);

        let mut accept = Pipeline::new(vec![
            Stage::new(
                spec("accept-error", CandidateMode::FullVocabulary),
                AcceptFails,
            )
            .unwrap(),
        ])
        .unwrap();
        accept.begin(&[]).unwrap();
        let mut logits = [0.0];
        accept.apply_to_vocabulary(0, &[], &mut logits).unwrap();
        assert!(accept.accept(TokenId::new(0).unwrap()).is_err());
        let failure = accept.receipt().stages[0].failure.clone().unwrap();
        assert!(!failure.panicked);
        assert_eq!(failure.phase, CallbackPhase::Accept);
    }
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! A thinking-token budget as a logit transform.
//!
//! Reasoning models open a block (`<think>`), think, and close it
//! (`</think>`), and nothing inside the model bounds how long the block runs.
//! [`ReasoningBudget`] counts tokens inside a block and, once `budget` of them
//! have been admitted without a close, masks every candidate except the next
//! token of a forced-close sequence until that sequence is complete.
//!
//! The state machine is llama.cpp's `common/reasoning-budget.cpp`, so a
//! backend that chains the native sampler and one that runs this transform
//! spend and force identically, and [`ReasoningBudget::accept`] alone serves
//! as an exact mirror of the native sampler for accounting. One divergence
//! is deliberate: the native sampler delays forcing until a split UTF-8
//! character completes, which needs token bytes this transform never sees.
//! It can therefore begin forcing a token or two earlier inside a
//! multi-byte character; forcing still begins after exactly `budget` tokens.

use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use logit_loom_core::TokenId;

use crate::{LogitTransform, TransformContext, TransformError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// Outside a block, watching for the opening tag.
    Idle,
    /// Inside a block with this many tokens spent.
    Counting(u32),
    /// Budget spent; this is the next index of the forced sequence.
    Forcing(usize),
    /// Block closed; a new opening tag re-arms a fresh budget.
    Done,
}

/// Where a [`ReasoningBudget`] starts each call, and restarts on reset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningStart {
    /// Outside any block, waiting for the opening tag.
    Outside,
    /// Inside a block the prompt opened, as templates that force thinking
    /// render; the model then never emits the opening tag itself.
    Inside,
    /// Inside a block that must close now: the forced sequence is emitted
    /// from the first token. A caller that ends a thought early (a loop
    /// correction) starts the next call here; the close is the caller's to
    /// record and is not counted in the report.
    Closing,
}

/// Shared count of blocks a [`ReasoningBudget`] forced closed.
///
/// A transform moves into its pipeline stage; this handle stays with the
/// caller and reads the count for the current or most recent call.
#[derive(Clone, Debug, Default)]
pub struct ReasoningBudgetReport(Arc<AtomicU32>);

impl ReasoningBudgetReport {
    /// Blocks forced closed since the transform last reset.
    pub fn forced_closes(&self) -> u32 {
        self.0.load(Ordering::Relaxed)
    }
}

/// Forces a reasoning block closed once it has spent its token budget.
#[derive(Clone, Debug)]
pub struct ReasoningBudget {
    budget: NonZeroU32,
    open: TokenId,
    close: TokenId,
    forced: Vec<TokenId>,
    start: ReasoningStart,
    phase: Phase,
    report: ReasoningBudgetReport,
}

impl ReasoningBudget {
    /// Creates a budget over single-token tags.
    ///
    /// `forced` is emitted token by token when the budget runs out and must
    /// contain `close`, so a forced block always ends closed. `start` says
    /// where each call begins (see [`ReasoningStart`]); a prompt that opened
    /// the block needs [`ReasoningStart::Inside`], or the budget never arms.
    ///
    /// # Errors
    ///
    /// Returns an error when `open` equals `close` or `forced` lacks `close`.
    pub fn new(
        budget: NonZeroU32,
        open: TokenId,
        close: TokenId,
        forced: Vec<TokenId>,
        start: ReasoningStart,
    ) -> Result<Self, TransformError> {
        if open == close {
            return Err(TransformError::new(
                "reasoning budget opening and closing tags must differ",
            ));
        }
        if !forced.contains(&close) {
            return Err(TransformError::new(
                "the forced sequence must contain the closing tag",
            ));
        }
        Ok(Self {
            budget,
            open,
            close,
            forced,
            start,
            phase: initial_phase(start),
            report: ReasoningBudgetReport::default(),
        })
    }

    /// A handle that keeps reading the forced-close count after the
    /// transform has moved into a pipeline.
    pub fn report(&self) -> ReasoningBudgetReport {
        self.report.clone()
    }

    /// The configured budget in tokens.
    pub const fn budget(&self) -> NonZeroU32 {
        self.budget
    }

    /// Whether a reasoning block is open and still counting.
    pub const fn in_block(&self) -> bool {
        matches!(self.phase, Phase::Counting(_))
    }

    /// Ends the open block now, as if its budget had run out.
    ///
    /// The port of llama.cpp's `common_reasoning_budget_force`: only a
    /// counting block can be forced. The report counts only closes the
    /// budget itself forced; the caller that forces a close here knows why
    /// and records it. Returns whether forcing began.
    pub fn force_close(&mut self) -> bool {
        if !self.in_block() {
            return false;
        }
        self.phase = Phase::Forcing(0);
        true
    }

    /// The token the budget currently forces, if it is forcing.
    pub fn forcing(&self) -> Option<TokenId> {
        match self.phase {
            Phase::Forcing(index) => self.forced.get(index).copied(),
            _ => None,
        }
    }
}

const fn initial_phase(start: ReasoningStart) -> Phase {
    match start {
        ReasoningStart::Outside => Phase::Idle,
        ReasoningStart::Inside => Phase::Counting(0),
        ReasoningStart::Closing => Phase::Forcing(0),
    }
}

impl LogitTransform for ReasoningBudget {
    fn reset(&mut self, _causal_tokens: &[TokenId]) -> Result<(), TransformError> {
        self.phase = initial_phase(self.start);
        self.report.0.store(0, Ordering::Relaxed);
        Ok(())
    }

    fn apply(&mut self, mut context: TransformContext<'_>) -> Result<(), TransformError> {
        let Some(forced) = self.forcing() else {
            return Ok(());
        };
        if !context.candidate_tokens().contains(&forced) {
            return Err(TransformError::new(
                "the forced reasoning-close token is outside the candidate view",
            ));
        }
        for (token, logit) in context.candidates_mut() {
            if token != forced {
                *logit = f32::NEG_INFINITY;
            }
        }
        Ok(())
    }

    fn accept(&mut self, token: TokenId) -> Result<(), TransformError> {
        self.phase = match self.phase {
            Phase::Idle | Phase::Done if token == self.open => Phase::Counting(0),
            Phase::Counting(_) if token == self.close => Phase::Done,
            Phase::Counting(spent) => {
                let spent = spent.saturating_add(1);
                if spent >= self.budget.get() {
                    self.report.0.fetch_add(1, Ordering::Relaxed);
                    Phase::Forcing(0)
                } else {
                    Phase::Counting(spent)
                }
            }
            Phase::Forcing(index) if index + 1 >= self.forced.len() => Phase::Done,
            Phase::Forcing(index) => Phase::Forcing(index + 1),
            unchanged => unchanged,
        };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token(raw: i32) -> TokenId {
        TokenId::new(raw).expect("fixture token")
    }

    const OPEN: i32 = 100;
    const CLOSE: i32 = 101;
    const NEWLINE: i32 = 7;

    fn budget(tokens: u32, open_at_start: bool) -> ReasoningBudget {
        started(
            tokens,
            if open_at_start {
                ReasoningStart::Inside
            } else {
                ReasoningStart::Outside
            },
        )
    }

    fn started(tokens: u32, start: ReasoningStart) -> ReasoningBudget {
        ReasoningBudget::new(
            NonZeroU32::new(tokens).expect("nonzero"),
            token(OPEN),
            token(CLOSE),
            vec![token(NEWLINE), token(CLOSE)],
            start,
        )
        .expect("valid budget")
    }

    fn feed(budget: &mut ReasoningBudget, tokens: impl IntoIterator<Item = i32>) {
        for raw in tokens {
            budget.accept(token(raw)).expect("accept");
        }
    }

    fn masked(budget: &mut ReasoningBudget) -> Vec<f32> {
        let tokens: Vec<TokenId> = (0..8).chain([OPEN, CLOSE]).map(token).collect();
        let mut logits = vec![0.5_f32; tokens.len()];
        budget
            .apply(TransformContext::new(0, &[], &tokens, &mut logits).expect("view"))
            .expect("apply");
        logits
    }

    #[test]
    fn construction_refuses_unclosable_sequences() {
        let one = NonZeroU32::MIN;
        assert!(
            ReasoningBudget::new(
                one,
                token(OPEN),
                token(OPEN),
                vec![token(OPEN)],
                ReasoningStart::Inside
            )
            .is_err()
        );
        assert!(
            ReasoningBudget::new(
                one,
                token(OPEN),
                token(CLOSE),
                vec![token(NEWLINE)],
                ReasoningStart::Inside
            )
            .is_err()
        );
    }

    #[test]
    fn a_block_closed_in_time_is_left_alone() {
        let mut transform = budget(4, true);
        let report = transform.report();
        feed(&mut transform, [1, 2, 3, CLOSE, 5, 6, 7, 8, 9]);
        assert_eq!(report.forced_closes(), 0);
        assert!(masked(&mut transform).iter().all(|logit| logit.is_finite()));
    }

    #[test]
    fn a_spent_budget_forces_the_close_sequence_token_by_token() {
        let mut transform = budget(3, true);
        let report = transform.report();
        feed(&mut transform, [1, 2, 3]);
        assert_eq!(report.forced_closes(), 1);
        assert_eq!(transform.forcing(), Some(token(NEWLINE)));
        let logits = masked(&mut transform);
        assert_eq!(logits.iter().filter(|logit| logit.is_finite()).count(), 1);
        assert!(logits[usize::try_from(NEWLINE).unwrap()].is_finite());
        feed(&mut transform, [NEWLINE]);
        assert_eq!(transform.forcing(), Some(token(CLOSE)));
        feed(&mut transform, [CLOSE]);
        assert_eq!(transform.forcing(), None);
        assert!(masked(&mut transform).iter().all(|logit| logit.is_finite()));
    }

    #[test]
    fn an_idle_budget_arms_on_a_generated_opening_tag_and_rearms_after_a_close() {
        let mut transform = budget(2, false);
        let report = transform.report();
        feed(&mut transform, [1, 2, 3, 4, 5]);
        assert_eq!(report.forced_closes(), 0, "no block was opened");
        feed(&mut transform, [OPEN, 1, CLOSE, OPEN, 1, 2, NEWLINE, CLOSE]);
        assert_eq!(report.forced_closes(), 1);
    }

    #[test]
    fn reset_restores_the_initial_phase_and_clears_the_report() {
        let mut transform = budget(1, true);
        let report = transform.report();
        feed(&mut transform, [1]);
        assert_eq!(report.forced_closes(), 1);
        transform.reset(&[]).expect("reset");
        assert_eq!(report.forced_closes(), 0);
        assert_eq!(transform.forcing(), None);
        feed(&mut transform, [1]);
        assert_eq!(transform.forcing(), Some(token(NEWLINE)));
    }

    #[test]
    fn force_close_ends_only_an_open_block() {
        let mut transform = budget(100, false);
        let report = transform.report();
        assert!(!transform.force_close(), "no block is open");
        feed(&mut transform, [OPEN, 1, 2]);
        assert!(transform.in_block());
        assert!(transform.force_close());
        assert_eq!(
            report.forced_closes(),
            0,
            "a caller-forced close is the caller's to record"
        );
        assert_eq!(transform.forcing(), Some(token(NEWLINE)));
        assert!(!transform.force_close(), "already forcing");
        feed(&mut transform, [NEWLINE, CLOSE]);
        assert!(!transform.in_block());
    }

    #[test]
    fn a_closing_start_forces_from_the_first_token_and_survives_reset() {
        let mut transform = started(4, ReasoningStart::Closing);
        let report = transform.report();
        transform.reset(&[]).expect("reset");
        assert_eq!(transform.forcing(), Some(token(NEWLINE)));
        feed(&mut transform, [NEWLINE, CLOSE]);
        assert_eq!(transform.forcing(), None);
        assert_eq!(
            report.forced_closes(),
            0,
            "the caller records its own close"
        );
        // After the close it budgets the next block normally.
        feed(&mut transform, [OPEN, 1, 2, 3, 4]);
        assert_eq!(report.forced_closes(), 1);
    }

    #[test]
    fn a_forced_token_outside_a_sparse_view_is_an_error() {
        let mut transform = budget(1, true);
        feed(&mut transform, [1]);
        let tokens = [token(1), token(2)];
        let mut logits = [0.0, 0.0];
        assert!(
            transform
                .apply(TransformContext::new(0, &[], &tokens, &mut logits).expect("view"))
                .is_err()
        );
    }
}

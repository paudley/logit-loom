// SPDX-License-Identifier: MIT OR Apache-2.0

//! Small first-party transforms used by examples and conformance tests.

use std::collections::BTreeMap;

use logit_loom_core::{MAX_LOGIT_BIASES, TokenId};

use crate::{LogitTransform, TransformContext, TransformError};

/// Adds finite per-token values to candidates present in the current view.
#[derive(Clone, Debug, Default)]
pub struct TokenBias {
    biases: BTreeMap<TokenId, f32>,
}

impl TokenBias {
    /// Creates a validated token-bias map.
    ///
    /// When an input repeats a token identifier, the last value wins.
    ///
    /// # Errors
    ///
    /// Returns an error for a non-finite bias or more than
    /// [`MAX_LOGIT_BIASES`] input entries.
    pub fn new(biases: impl IntoIterator<Item = (TokenId, f32)>) -> Result<Self, TransformError> {
        let mut values = BTreeMap::new();
        for (index, (token, bias)) in biases.into_iter().enumerate() {
            if index == MAX_LOGIT_BIASES {
                return Err(TransformError::new(format!(
                    "token bias input exceeds {MAX_LOGIT_BIASES} entries"
                )));
            }
            if !bias.is_finite() {
                return Err(TransformError::new("token biases must be finite"));
            }
            values.insert(token, bias);
        }
        Ok(Self { biases: values })
    }
}

impl LogitTransform for TokenBias {
    fn apply(&mut self, mut context: TransformContext<'_>) -> Result<(), TransformError> {
        for (token, logit) in context.candidates_mut() {
            if let Some(bias) = self.biases.get(&token) {
                *logit += bias;
            }
        }
        Ok(())
    }
}

/// Adds a finite value to one dynamically ranked candidate.
#[derive(Clone, Copy, Debug)]
pub struct RankBias {
    rank: usize,
    bias: f32,
}

impl RankBias {
    /// Creates a rank-based transform.
    ///
    /// # Errors
    ///
    /// Returns an error for a non-finite or zero bias.
    pub fn new(rank: usize, bias: f32) -> Result<Self, TransformError> {
        if !bias.is_finite() || bias == 0.0 {
            return Err(TransformError::new("rank bias must be finite and nonzero"));
        }
        Ok(Self { rank, bias })
    }
}

impl LogitTransform for RankBias {
    fn apply(&mut self, mut context: TransformContext<'_>) -> Result<(), TransformError> {
        let tokens = context.candidate_tokens().to_vec();
        let logits = context.logits_mut();
        let mut finite = logits
            .iter()
            .enumerate()
            .filter(|(_, value)| value.is_finite())
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        finite.sort_unstable_by(|left, right| {
            logits[*right]
                .total_cmp(&logits[*left])
                .then_with(|| tokens[*left].cmp(&tokens[*right]))
        });
        let selected = finite
            .get(self.rank)
            .copied()
            .ok_or_else(|| TransformError::new("candidate view does not contain that rank"))?;
        logits[selected] += self.bias;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_bias_changes_only_matching_candidates() {
        let target = TokenId::new(7).unwrap();
        let other = TokenId::new(9).unwrap();
        let mut transform = TokenBias::new([(target, 1.5)]).unwrap();
        let mut logits = [2.0, 3.0];
        transform
            .apply(TransformContext::new(0, &[], &[other, target], &mut logits).unwrap())
            .unwrap();
        assert_eq!(
            logits.map(f32::to_bits),
            [2.0_f32.to_bits(), 4.5_f32.to_bits()]
        );
    }

    #[test]
    fn token_bias_rejects_nonfinite_and_oversized_inputs() {
        let token = TokenId::new(1).unwrap();
        assert!(TokenBias::new([(token, f32::NAN)]).is_err());
        let oversized = (0..=MAX_LOGIT_BIASES).map(|_| (token, 1.0));
        assert!(TokenBias::new(oversized).is_err());
    }

    #[test]
    fn rank_bias_breaks_logit_ties_by_token_identifier() {
        let tokens = [
            TokenId::new(5).unwrap(),
            TokenId::new(2).unwrap(),
            TokenId::new(9).unwrap(),
        ];
        let mut logits = [1.0, 1.0, 0.0];
        let mut transform = RankBias::new(0, 0.5).unwrap();
        transform
            .apply(TransformContext::new(0, &[], &tokens, &mut logits).unwrap())
            .unwrap();
        assert_eq!(
            logits.map(f32::to_bits),
            [1.0_f32.to_bits(), 1.5_f32.to_bits(), 0.0_f32.to_bits()]
        );
    }

    #[test]
    fn rank_bias_rejects_missing_finite_rank() {
        let tokens = [TokenId::new(1).unwrap()];
        let mut logits = [f32::INFINITY];
        let mut transform = RankBias::new(0, 1.0).unwrap();
        assert!(
            transform
                .apply(TransformContext::new(0, &[], &tokens, &mut logits).unwrap())
                .is_err()
        );
    }
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shape of one raw next-token distribution.
//!
//! A backend measures the vocabulary logits it is about to sample from,
//! before any transform, grammar, penalty, or truncation touches them, so the
//! numbers describe what the model itself believed at that step. Nothing here
//! changes a logit.

use logit_loom_core::TokenId;
use wide::f32x8;

use crate::DistributionError;

/// Floor applied to `logit - max` before exponentiation.
///
/// `exp(-100)` is below the smallest normal `f32`, so clamping loses no
/// representable mass, and it keeps masked `-inf` logits from turning the
/// entropy term `d * exp(d)` into `-inf * 0 = NaN`.
const LOG_FLOOR: f32 = -100.0;

/// Raw next-token distribution evidence for one sampled token.
///
/// All quantities are in nats and are measured on the unmodified logits, so a
/// pipeline, grammar, or penalty that steered selection away from the model's
/// preference shows up as a sampled log-probability below the top one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenDistribution {
    /// Shannon entropy of the softmax over the full vocabulary.
    pub entropy: f32,
    /// Log-probability of the token actually sampled.
    pub sampled_log_probability: f32,
    /// Most probable token under the raw distribution.
    pub top_token: TokenId,
    /// Log-probability of [`Self::top_token`].
    pub top_log_probability: f32,
    /// Logit gap between the most and second-most probable tokens.
    ///
    /// Zero for a single-token vocabulary or a tie at the top.
    pub margin: f32,
}

impl TokenDistribution {
    /// Measures one full-vocabulary logit row and the token sampled from it.
    ///
    /// Two passes: a scalar scan for the maximum, runner-up, and argmax, then
    /// an eight-lane pass accumulating the softmax normalizer and the entropy
    /// numerator relative to the maximum. `-inf` logits (masked tokens)
    /// contribute nothing; any NaN, `+inf`, or all-masked row is refused.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty row, a sampled token outside it, or a row
    /// with no finite maximum or any NaN or positive-infinite logit.
    pub fn measure(logits: &[f32], sampled: TokenId) -> Result<Self, DistributionError> {
        let sampled_index = usize::try_from(sampled.get())
            .ok()
            .filter(|index| *index < logits.len())
            .ok_or_else(|| {
                DistributionError::new(format!(
                    "sampled token {} is outside the {}-entry row",
                    sampled.get(),
                    logits.len()
                ))
            })?;
        let (maximum, runner_up, top_index) = extremes(logits)?;
        let (normalizer, weighted) = accumulate(logits, maximum);
        // normalizer >= 1: the maximum itself contributes exp(0).
        let log_normalizer = normalizer.ln();
        let entropy = (log_normalizer - weighted / normalizer).max(0.0);
        let top_token = i32::try_from(top_index)
            .map_err(|_| DistributionError::new("vocabulary index exceeds i32"))
            .and_then(|raw| {
                TokenId::new(raw).map_err(|error| DistributionError::new(error.to_string()))
            })?;
        let top_log_probability = -log_normalizer;
        Ok(Self {
            entropy,
            sampled_log_probability: logits[sampled_index] - maximum - log_normalizer,
            top_token,
            top_log_probability,
            margin: if runner_up.is_finite() {
                maximum - runner_up
            } else {
                0.0
            },
        })
    }
}

/// Returns `(maximum, runner-up, argmax)`, refusing non-measurable rows.
///
/// Branch-free eight-lane passes: the maximum (with a NaN sweep), the first
/// index holding it, then the maximum of everything else. A tie at the top
/// therefore yields `runner-up == maximum` and a zero margin.
fn extremes(logits: &[f32]) -> Result<(f32, f32, usize), DistributionError> {
    if logits.is_empty() {
        return Err(DistributionError::new("logit row is empty"));
    }
    let (maximum, any_nan) = lane_maximum(logits);
    if any_nan {
        return Err(DistributionError::new("the row holds a NaN logit"));
    }
    if maximum == f32::INFINITY {
        return Err(DistributionError::new("the row holds a +inf logit"));
    }
    if maximum == f32::NEG_INFINITY {
        return Err(DistributionError::new("every logit in the row is masked"));
    }
    // Exact equality is the point: the maximum is one of these values, and a
    // tolerance could land on a near-tie instead.
    #[allow(clippy::float_cmp)]
    let top_index = logits
        .iter()
        .position(|&logit| logit == maximum)
        .unwrap_or_default();
    let runner_up = lane_maximum(&logits[..top_index])
        .0
        .max(lane_maximum(&logits[top_index + 1..]).0);
    Ok((maximum, runner_up, top_index))
}

/// Returns the maximum of `values` (`-inf` when empty) and whether any is NaN.
fn lane_maximum(values: &[f32]) -> (f32, bool) {
    let (lanes, remainder) = values.as_chunks::<8>();
    let mut maximum = f32x8::splat(f32::NEG_INFINITY);
    let mut nan = f32x8::ZERO;
    for &lane in lanes {
        let lane = f32x8::from(lane);
        maximum = maximum.max(lane);
        nan |= lane.is_nan();
    }
    let mut any_nan = nan.any();
    let mut scalar = maximum
        .to_array()
        .into_iter()
        .fold(f32::NEG_INFINITY, f32::max);
    for &value in remainder {
        any_nan |= value.is_nan();
        scalar = scalar.max(value);
    }
    (scalar, any_nan)
}

/// Returns `(sum exp(d), sum d * exp(d))` over `d = logit - maximum`.
fn accumulate(logits: &[f32], maximum: f32) -> (f32, f32) {
    let floor = f32x8::splat(LOG_FLOOR);
    let shift = f32x8::splat(maximum);
    let mut normalizer = f32x8::ZERO;
    let mut weighted = f32x8::ZERO;
    let (lanes, remainder) = logits.as_chunks::<8>();
    for &lane in lanes {
        let delta = (f32x8::from(lane) - shift).max(floor);
        let mass = delta.exp();
        normalizer += mass;
        weighted += delta * mass;
    }
    let mut normalizer = normalizer.reduce_add();
    let mut weighted = weighted.reduce_add();
    for &logit in remainder {
        let delta = (logit - maximum).max(LOG_FLOOR);
        let mass = delta.exp();
        normalizer += mass;
        weighted += delta * mass;
    }
    (normalizer, weighted)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token(raw: i32) -> TokenId {
        TokenId::new(raw).expect("fixture token is non-negative")
    }

    /// Straightforward f64 reference for comparison.
    fn reference(logits: &[f32], sampled: usize) -> (f64, f64, f64) {
        let maximum = f64::from(logits.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        let masses: Vec<f64> = logits
            .iter()
            .map(|&logit| (f64::from(logit) - maximum).exp())
            .collect();
        let total: f64 = masses.iter().sum();
        let entropy = -masses
            .iter()
            .filter(|mass| **mass > 0.0)
            .map(|mass| (mass / total) * (mass / total).ln())
            .sum::<f64>();
        let sampled_log_probability = (masses[sampled] / total).ln();
        (entropy, sampled_log_probability, -total.ln())
    }

    #[test]
    fn uniform_row_has_log_vocabulary_entropy_and_no_margin() {
        let logits = vec![0.25_f32; 1_000];
        let shape = TokenDistribution::measure(&logits, token(3)).expect("measurable");
        let expected = 1_000_f32.ln();
        assert!((shape.entropy - expected).abs() < 1e-4, "{shape:?}");
        assert!((shape.sampled_log_probability + expected).abs() < 1e-4);
        assert!((shape.top_log_probability + expected).abs() < 1e-4);
        assert_eq!(shape.margin, 0.0);
    }

    #[test]
    fn peaked_row_matches_an_f64_reference_including_the_remainder_lanes() {
        // 1_004 entries: not a multiple of eight, so the scalar tail runs.
        let logits: Vec<f32> = (0..1_003_u16)
            .map(|index| f32::from(index.wrapping_mul(7919) % 211) * -0.09)
            .chain(std::iter::once(6.5))
            .collect();
        let sampled = 17;
        let shape = TokenDistribution::measure(&logits, token(17)).expect("measurable");
        let (entropy, sampled_log_probability, top_log_probability) = reference(&logits, sampled);
        assert!(
            (f64::from(shape.entropy) - entropy).abs() < 1e-4,
            "{shape:?} vs {entropy}"
        );
        assert!((f64::from(shape.sampled_log_probability) - sampled_log_probability).abs() < 1e-4);
        assert!((f64::from(shape.top_log_probability) - top_log_probability).abs() < 1e-4);
        assert_eq!(shape.top_token, token(1_003));
        assert!((shape.margin - 6.5).abs() < 1e-6);
    }

    #[test]
    fn masked_logits_carry_no_mass_and_no_nan() {
        let mut logits = vec![f32::NEG_INFINITY; 64];
        logits[5] = 1.0;
        logits[9] = 1.0;
        let shape = TokenDistribution::measure(&logits, token(9)).expect("measurable");
        assert!((shape.entropy - 2_f32.ln()).abs() < 1e-5, "{shape:?}");
        assert!((shape.sampled_log_probability + 2_f32.ln()).abs() < 1e-5);
        assert_eq!(shape.margin, 0.0);
    }

    #[test]
    fn certain_row_has_zero_entropy_and_an_infinite_style_margin() {
        let mut logits = vec![-50.0_f32; 32];
        logits[0] = 50.0;
        let shape = TokenDistribution::measure(&logits, token(0)).expect("measurable");
        assert!(shape.entropy.abs() < 1e-6, "{shape:?}");
        assert!(shape.sampled_log_probability.abs() < 1e-6);
        assert!((shape.margin - 100.0).abs() < 1e-4);
    }

    #[test]
    fn a_token_sampled_away_from_the_top_reports_its_own_probability() {
        let logits = [2.0_f32, 0.0, -1.0];
        let shape = TokenDistribution::measure(&logits, token(2)).expect("measurable");
        let (_, sampled_log_probability, _) = reference(&logits, 2);
        assert!((f64::from(shape.sampled_log_probability) - sampled_log_probability).abs() < 1e-5);
        assert!(shape.sampled_log_probability < shape.top_log_probability);
        assert_eq!(shape.top_token, token(0));
    }

    #[test]
    fn unmeasurable_rows_are_refused() {
        assert!(TokenDistribution::measure(&[], token(0)).is_err());
        assert!(TokenDistribution::measure(&[0.0, 1.0], token(2)).is_err());
        assert!(TokenDistribution::measure(&[0.0, f32::NAN], token(0)).is_err());
        assert!(TokenDistribution::measure(&[0.0, f32::INFINITY], token(0)).is_err());
        assert!(TokenDistribution::measure(&[f32::NEG_INFINITY; 4], token(0)).is_err());
    }
}

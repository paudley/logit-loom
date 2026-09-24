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

/// Running record of one generation's raw distributions, for telemetry.
///
/// Keeps one entropy per measured token (four bytes each) so the closing
/// report can locate where the stream collapsed, and counts the tokens a
/// backend could not measure rather than inventing values for them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DistributionSummary {
    entropies: Vec<f32>,
    surprisal_sum: f64,
    steered: u32,
    unmeasured: u32,
}

/// Closing figures of a [`DistributionSummary`], all in nats.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DistributionReport {
    /// Sampled tokens with a measured distribution.
    pub measured: u32,
    /// Sampled tokens whose row could not be measured.
    pub unmeasured: u32,
    /// Mean entropy over measured tokens.
    pub mean_entropy: f32,
    /// Mean entropy over the trailing window.
    pub tail_entropy: f32,
    /// Lowest rolling-window mean entropy, or the overall mean when the
    /// stream is shorter than one window.
    pub min_window_entropy: f32,
    /// Measured-token index where the rolling mean fell under the floor and
    /// stayed under it to the end: the candidate onset of a collapse.
    pub collapse_at: Option<u32>,
    /// Tokens sampled away from the raw distribution's top choice.
    pub steered: u32,
    /// Mean negative log-probability of the sampled tokens.
    pub mean_surprisal: f32,
}

impl DistributionSummary {
    /// Records one measured token.
    pub fn push(&mut self, distribution: TokenDistribution) {
        self.entropies.push(distribution.entropy);
        self.surprisal_sum -= f64::from(distribution.sampled_log_probability);
        if distribution.sampled_log_probability < distribution.top_log_probability {
            self.steered = self.steered.saturating_add(1);
        }
    }

    /// Records one sampled token whose row could not be measured.
    pub fn push_unmeasured(&mut self) {
        self.unmeasured = self.unmeasured.saturating_add(1);
    }

    /// Closing report over rolling windows of `window` tokens, calling a
    /// window under `floor` nats collapsed. `None` before anything was
    /// measured.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    pub fn report(&self, window: usize, floor: f32) -> Option<DistributionReport> {
        let count = self.entropies.len();
        if count == 0 {
            return None;
        }
        let window = window.clamp(1, count);
        let mean = |values: &[f32]| {
            (values.iter().map(|value| f64::from(*value)).sum::<f64>() / values.len() as f64) as f32
        };
        let rolling: Vec<f32> = self.entropies.windows(window).map(mean).collect();
        let collapse_at = if rolling.last().is_some_and(|last| *last < floor) {
            let start = rolling
                .iter()
                .rposition(|value| *value >= floor)
                .map_or(0, |index| index + 1);
            u32::try_from(start).ok()
        } else {
            None
        };
        Some(DistributionReport {
            measured: u32::try_from(count).unwrap_or(u32::MAX),
            unmeasured: self.unmeasured,
            mean_entropy: mean(&self.entropies),
            tail_entropy: mean(&self.entropies[count - window..]),
            min_window_entropy: rolling.iter().copied().fold(f32::INFINITY, f32::min),
            collapse_at,
            steered: self.steered,
            mean_surprisal: (self.surprisal_sum / count as f64) as f32,
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

    fn shaped(entropy: f32, steered: bool) -> TokenDistribution {
        TokenDistribution {
            entropy,
            sampled_log_probability: if steered { -2.0 } else { -0.5 },
            top_token: token(0),
            top_log_probability: -0.5,
            margin: 1.0,
        }
    }

    #[test]
    fn summary_reports_means_steering_and_a_sustained_collapse() {
        let mut summary = DistributionSummary::default();
        assert!(summary.report(4, 0.1).is_none());
        for index in 0..20 {
            summary.push(shaped(if index < 12 { 2.0 } else { 0.01 }, index % 5 == 0));
        }
        summary.push_unmeasured();
        let report = summary.report(4, 0.1).expect("measured");
        assert_eq!(report.measured, 20);
        assert_eq!(report.unmeasured, 1);
        assert_eq!(report.steered, 4);
        assert!((report.tail_entropy - 0.01).abs() < 1e-6);
        assert!((report.min_window_entropy - 0.01).abs() < 1e-6);
        // Windows 12.. are all collapsed; window 11 still holds 2.0.
        assert_eq!(report.collapse_at, Some(12));
        assert!((report.mean_surprisal - 0.8).abs() < 1e-5, "{report:?}");
    }

    #[test]
    fn a_recovered_stream_reports_no_collapse() {
        let mut summary = DistributionSummary::default();
        for entropy in [2.0, 0.01, 0.01, 0.01, 0.01, 2.0, 2.0] {
            summary.push(shaped(entropy, false));
        }
        assert_eq!(summary.report(2, 0.1).expect("measured").collapse_at, None);
        // A stream shorter than the window reports over what it has.
        let mut short = DistributionSummary::default();
        short.push(shaped(1.0, false));
        let report = short.report(64, 0.1).expect("measured");
        assert!((report.min_window_entropy - 1.0).abs() < 1e-6);
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

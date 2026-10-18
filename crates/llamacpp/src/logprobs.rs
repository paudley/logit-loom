// SPDX-License-Identifier: MIT OR Apache-2.0

//! Raw-distribution log-probabilities of emitted tokens.
//!
//! Pure, model-free arithmetic over one candidate row. The native probe stage
//! in `native_stage` calls [`record_row`] from `accept`; nothing here touches
//! native state, so every rule below is tested without a model.
//!
//! All values are natural-log probabilities of the **raw** softmax: the
//! untouched logits, before any grammar, penalty, bias, temperature, or
//! truncation. They describe what the model believed, not the distribution the
//! chain sampled from.

use std::cmp::Ordering;

/// One alternative token and its raw log-probability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TopLogprob {
    /// Backend-native token id (the index into the logit row).
    pub token: i32,
    /// Natural-log probability under the raw softmax. `-inf` for a token whose
    /// logit was `-inf` (masked or banned by the model's own output).
    pub logprob: f64,
}

/// What the probe recorded for one emitted token.
///
/// Memory: one record is a fixed header plus `top.len()` entries of
/// [`TopLogprob`] (16 bytes each), so a generation of `T` tokens with top-`N`
/// recording holds about `T * (N * 16 + 48)` bytes until drained.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenLogprobs {
    /// The token the chain emitted, as passed to the chain's `accept`.
    pub token: i32,
    /// Raw log-probability of [`Self::token`], or `None` when the row could
    /// not be measured (see [`Self::measured`]).
    pub logprob: Option<f64>,
    /// The `top_n` most probable tokens of the raw row, most probable first;
    /// ties are ordered by lower token id. Empty when unmeasured or when the
    /// probe was built with `top_n == 0`. Holds `min(top_n, vocabulary)`
    /// entries, including `-inf` entries when fewer tokens are finite.
    pub top: Vec<TopLogprob>,
}

impl TokenLogprobs {
    /// Whether a raw row was measured for this token.
    ///
    /// `false` when `accept` ran without a preceding full identity-ordered
    /// `apply` (a sorted or partial view, a token accepted without sampling, or
    /// several `apply` calls before the matching `accept`s), or when the row
    /// held a NaN, a `+inf`, no finite logit, or did not contain the token.
    /// Such a record is a placeholder that keeps positions aligned with the
    /// emitted token stream; its numbers are never guessed.
    #[must_use]
    pub fn measured(&self) -> bool {
        self.logprob.is_some()
    }
}

/// Builds the record for `token` over `row`, keeping the `top_n` best.
///
/// `scratch` is reusable index storage so repeated calls do not reallocate.
pub(crate) fn record_row(
    row: &[f32],
    token: i32,
    top_n: usize,
    scratch: &mut Vec<u32>,
) -> TokenLogprobs {
    let unmeasured = TokenLogprobs {
        token,
        logprob: None,
        top: Vec::new(),
    };
    let Some(log_normalizer) = log_normalizer(row) else {
        return unmeasured;
    };
    let Some(&chosen) = usize::try_from(token).ok().and_then(|index| row.get(index)) else {
        return unmeasured;
    };
    let logprob_of = |logit: f32| f64::from(logit) - log_normalizer;
    TokenLogprobs {
        token,
        logprob: Some(logprob_of(chosen)),
        top: top_indices(row, top_n, scratch)
            .iter()
            .map(|&index| TopLogprob {
                token: i32::try_from(index).unwrap_or(i32::MAX),
                logprob: logprob_of(row[index]),
            })
            .collect(),
    }
}

/// `ln(sum(exp(logit)))` by max-shifted log-sum-exp in `f64`, or `None` for an
/// empty row, any NaN or `+inf`, or a row with no finite logit.
fn log_normalizer(row: &[f32]) -> Option<f64> {
    let mut maximum = f32::NEG_INFINITY;
    for &logit in row {
        if logit.is_nan() || logit == f32::INFINITY {
            return None;
        }
        maximum = maximum.max(logit);
    }
    if !maximum.is_finite() {
        return None;
    }
    let maximum = f64::from(maximum);
    let sum: f64 = row
        .iter()
        .map(|&logit| (f64::from(logit) - maximum).exp())
        .sum();
    Some(maximum + sum.ln())
}

/// Indices of the `n` largest logits, largest first, ties by lower index.
/// `row` must hold no NaN.
fn top_indices(row: &[f32], n: usize, scratch: &mut Vec<u32>) -> Vec<usize> {
    let n = n.min(row.len());
    if n == 0 {
        return Vec::new();
    }
    scratch.clear();
    // A row longer than u32::MAX cannot be a llama.cpp vocabulary (token ids
    // are i32); the cast cannot truncate for any measurable row.
    scratch.extend(0..u32::try_from(row.len()).unwrap_or(u32::MAX));
    let better = |a: &u32, b: &u32| {
        row[*b as usize]
            .partial_cmp(&row[*a as usize])
            .unwrap_or(Ordering::Equal)
            .then(a.cmp(b))
    };
    if n < scratch.len() {
        scratch.select_nth_unstable_by(n - 1, better);
        scratch.truncate(n);
    }
    scratch.sort_unstable_by(better);
    scratch.iter().map(|&index| index as usize).collect()
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
mod tests {
    use super::*;

    /// Independent reference: naive softmax in f64 (small magnitudes only).
    fn naive(row: &[f32]) -> Vec<f64> {
        let sum: f64 = row.iter().map(|&x| f64::from(x).exp()).sum();
        row.iter()
            .map(|&x| (f64::from(x).exp() / sum).ln())
            .collect()
    }

    fn record(row: &[f32], token: i32, n: usize) -> TokenLogprobs {
        record_row(row, token, n, &mut Vec::new())
    }

    #[test]
    fn chosen_logprob_matches_an_independent_softmax() {
        let row = [2.0_f32, 0.0, -1.0, 0.5, 3.25, -4.0];
        let reference = naive(&row);
        for (index, expected) in reference.iter().enumerate() {
            let got = record(&row, index as i32, 0).logprob.expect("measured");
            assert!(
                (got - expected).abs() < 1e-12,
                "{index}: {got} vs {expected}"
            );
        }
        let total: f64 = (0..row.len())
            .map(|i| record(&row, i as i32, 0).logprob.unwrap().exp())
            .sum();
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn top_n_is_exact_for_every_n_including_beyond_the_vocabulary() {
        let row = [0.5_f32, 2.0, -1.0, 2.0, 1.0];
        // Descending by logit, tie (ids 1 and 3) by lower id.
        let order = [1, 3, 4, 0, 2];
        let reference = naive(&row);
        for n in 0..=9 {
            let top = record(&row, 0, n).top;
            let expected = n.min(row.len());
            assert_eq!(top.len(), expected, "n={n}");
            for (rank, entry) in top.iter().enumerate() {
                assert_eq!(entry.token, order[rank], "n={n} rank={rank}");
                assert!((entry.logprob - reference[order[rank] as usize]).abs() < 1e-12);
            }
        }
        assert!(
            record(&row, 0, 0).top.is_empty(),
            "N = 0 records no alternatives"
        );
    }

    #[test]
    fn ties_are_deterministic_by_lower_token_id() {
        let row = [1.0_f32; 40];
        for n in [1, 5, 17, 40] {
            let ids: Vec<i32> = record(&row, 9, n).top.iter().map(|e| e.token).collect();
            assert_eq!(ids, (0..n as i32).collect::<Vec<_>>());
        }
        // Signed zeros are equal, not ordered by sign bit.
        let ids: Vec<i32> = record(&[0.0, -0.0, 0.0], 0, 3)
            .top
            .iter()
            .map(|e| e.token)
            .collect();
        assert_eq!(ids, [0, 1, 2]);
    }

    #[test]
    fn negative_infinity_is_a_minus_infinite_logprob() {
        let row = [f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY, 0.0];
        let banned = record(&row, 0, 4);
        assert_eq!(banned.logprob, Some(f64::NEG_INFINITY));
        let ids: Vec<i32> = banned.top.iter().map(|e| e.token).collect();
        assert_eq!(ids, [1, 3, 0, 2], "masked tokens trail, ties by id");
        assert_eq!(banned.top[2].logprob, f64::NEG_INFINITY);
        assert!(record(&row, 1, 0).logprob.unwrap().is_finite());
    }

    #[test]
    fn unmeasurable_rows_are_flagged_not_guessed() {
        for row in [
            vec![0.0, f32::NAN, 1.0],
            vec![0.0, f32::INFINITY],
            vec![f32::NEG_INFINITY; 3],
            vec![],
        ] {
            let got = record(&row, 0, 3);
            assert!(!got.measured() && got.top.is_empty(), "{row:?}");
        }
        assert!(!record(&[1.0, 2.0], 2, 1).measured(), "token outside row");
        assert!(!record(&[1.0, 2.0], -1, 1).measured(), "negative token");
    }

    #[test]
    fn large_magnitudes_stay_finite_and_normalized() {
        let row = [1.0e4_f32, 1.0e4 - 1.0, -1.0e4, 1.0e4];
        let got: Vec<f64> = (0..4)
            .map(|i| record(&row, i, 0).logprob.expect("finite"))
            .collect();
        assert!(got.iter().all(|x| x.is_finite() || *x < -1.0e3));
        // Two equal maxima and one at -1: closed form.
        let z = 2.0 + (-1.0_f64).exp();
        assert!((got[0] - (-z.ln())).abs() < 1e-12);
        assert!((got[1] - (-1.0 - z.ln())).abs() < 1e-12);
        assert!((got[2] - (-2.0e4 - z.ln())).abs() < 1e-6);
        let total: f64 = got.iter().map(|x| x.exp()).sum();
        assert!((total - 1.0).abs() < 1e-12);
    }
}

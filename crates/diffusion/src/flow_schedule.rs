// SPDX-License-Identifier: MIT OR Apache-2.0

//! Versioned flow-matching Euler schedules for non-Krea diffusion transformers.
//!
//! Each family reproduces, in `f32` and in the same operation order, the sigma
//! boundaries upstream stable-diffusion.cpp computes for that model family at
//! the pinned revision. The companion receives them as custom sigmas, which
//! bypass upstream's own scheduler, so these values are the whole schedule the
//! model sees. Krea keeps its own [`crate::KreaDiscreteFlowScheduleV1`].

use logit_loom_core::{CoreError, Digest};
use serde::{Deserialize, Serialize};

use crate::{Error, MAX_DIFFUSION_STEPS, Result};

/// Upstream's discrete training grid length (`TIMESTEPS`).
const TRAINING_TIMESTEPS: u32 = 1_000;

/// Schedule family and its exact parameters.
///
/// Floating-point parameters are kept as IEEE-754 bit patterns so the schedule
/// identity does not depend on text formatting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub enum FlowScheduleFamilyV1 {
    /// Upstream `DiscreteScheduler` over `DiscreteFlowDenoiser`: timesteps
    /// evenly spaced from 999 to 0, each mapped through the SD3 time shift.
    /// Z-Image and Qwen-Image use this with shift 3.
    DiscreteFlowShift {
        /// Exact time-shift bit pattern.
        shift_bits: u32,
    },
    /// Upstream `Flux2Scheduler`: a `1 - i/n` grid under the exponential
    /// time shift, with `mu` fitted from the image sequence length and the
    /// step count.
    Flux2Empirical {
        /// Latent image tokens, `(height / 16) * (width / 16)` for FLUX.2.
        image_seq_len: u32,
        /// Exact fitted `mu` bit pattern.
        mu_bits: u32,
    },
    /// Sigma boundaries published with a step-distilled model, used verbatim.
    /// The boundaries themselves, carried in the schedule's sigma bits, are
    /// the parameters; Qwen-Image 2.1 Turbo publishes its eight-step list.
    Explicit,
}

/// Exact flow-matching Euler schedule contract.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlowMatchScheduleV1 {
    family: FlowScheduleFamilyV1,
    steps: u32,
    sigma_bits: Vec<u32>,
}

impl FlowMatchScheduleV1 {
    /// Constructs upstream's discrete-flow schedule with an explicit shift.
    ///
    /// # Errors
    ///
    /// Returns an error for zero or excessive steps, or for a non-finite or
    /// non-positive shift.
    pub fn discrete_flow_shift(steps: u32, shift: f32) -> Result<Self> {
        let steps_u16 = bounded_steps(steps)?;
        if !shift.is_finite() || shift <= 0.0 {
            return Err(invalid("shift must be finite and positive"));
        }
        let mut sigmas = Vec::with_capacity(usize::from(steps_u16) + 1);
        let t_max = TRAINING_TIMESTEPS - 1;
        // Values are formed exactly as upstream forms them in `f32`:
        // DiscreteScheduler::get_sigmas and DiscreteFlowDenoiser::t_to_sigma.
        #[allow(
            clippy::cast_precision_loss,
            reason = "999 and the bounded step index are exactly representable in f32"
        )]
        let t_max_f32 = t_max as f32;
        if steps_u16 == 1 {
            sigmas.push(discrete_flow_sigma(shift, t_max_f32));
        } else {
            let step = t_max_f32 / f32::from(steps_u16 - 1);
            for index in 0..steps_u16 {
                let t = t_max_f32 - step * f32::from(index);
                sigmas.push(discrete_flow_sigma(shift, t));
            }
        }
        sigmas.push(0.0_f32);
        Ok(Self {
            family: FlowScheduleFamilyV1::DiscreteFlowShift {
                shift_bits: shift.to_bits(),
            },
            steps,
            sigma_bits: sigmas.into_iter().map(f32::to_bits).collect(),
        })
    }

    /// Constructs upstream's FLUX.2 schedule for an image sequence length.
    ///
    /// # Errors
    ///
    /// Returns an error for zero or excessive steps or a zero sequence length.
    pub fn flux2_empirical(steps: u32, image_seq_len: u32) -> Result<Self> {
        let steps_u16 = bounded_steps(steps)?;
        if image_seq_len == 0 {
            return Err(invalid("image sequence length must be positive"));
        }
        let mu = flux2_empirical_mu(image_seq_len, steps_u16);
        let exponent = mu.exp();
        let n = f32::from(steps_u16);
        let mut sigmas = Vec::with_capacity(usize::from(steps_u16) + 1);
        for index in 0..=steps_u16 {
            let t = 1.0_f32 - f32::from(index) / n;
            let sigma = if t <= 0.0 {
                0.0
            } else if t >= 1.0 {
                1.0
            } else {
                // flux_time_shift(mu, 1.0, t) with its unit exponent applied.
                exponent / (exponent + (1.0 / t - 1.0))
            };
            sigmas.push(sigma);
        }
        if let Some(last) = sigmas.last_mut() {
            *last = 0.0;
        }
        Ok(Self {
            family: FlowScheduleFamilyV1::Flux2Empirical {
                image_seq_len,
                mu_bits: mu.to_bits(),
            },
            steps,
            sigma_bits: sigmas.into_iter().map(f32::to_bits).collect(),
        })
    }

    /// Adopts published sigma boundaries verbatim, terminal zero included.
    ///
    /// # Errors
    ///
    /// Returns an error unless there are between two and the diffusion bound
    /// plus one boundaries, every boundary is finite and non-negative, the
    /// list never increases, and zero appears only as the final boundary.
    pub fn explicit(sigmas: &[f32]) -> Result<Self> {
        let steps = u32::try_from(sigmas.len().saturating_sub(1))
            .map_err(|_| invalid("explicit sigma count exceeds the diffusion bound"))?;
        bounded_steps(steps)?;
        let (last, interior) = sigmas
            .split_last()
            .ok_or_else(|| invalid("explicit sigmas must not be empty"))?;
        if last.to_bits() != 0.0_f32.to_bits() {
            return Err(invalid("explicit sigmas must end at positive zero"));
        }
        if interior
            .iter()
            .any(|sigma| !sigma.is_finite() || *sigma <= 0.0)
        {
            return Err(invalid(
                "explicit sigmas before the terminal zero must be finite and positive",
            ));
        }
        if sigmas.windows(2).any(|pair| pair[1] > pair[0]) {
            return Err(invalid("explicit sigmas must not increase"));
        }
        Ok(Self {
            family: FlowScheduleFamilyV1::Explicit,
            steps,
            sigma_bits: sigmas.iter().copied().map(f32::to_bits).collect(),
        })
    }

    /// Returns the schedule family and its parameters.
    #[must_use]
    pub const fn family(&self) -> FlowScheduleFamilyV1 {
        self.family
    }

    /// Returns the exact diffusion transition count.
    #[must_use]
    pub const fn steps(&self) -> u32 {
        self.steps
    }

    /// Returns the exact ordered sigma bit patterns, including terminal zero.
    #[must_use]
    pub fn sigma_bits(&self) -> &[u32] {
        &self.sigma_bits
    }

    /// Reconstructs the exact ordered sigma boundaries.
    #[must_use]
    pub fn sigmas(&self) -> Vec<f32> {
        self.sigma_bits
            .iter()
            .copied()
            .map(f32::from_bits)
            .collect()
    }

    /// Returns the complete versioned schedule identity.
    ///
    /// # Errors
    ///
    /// Returns a serialization error if the canonical contract cannot be
    /// encoded.
    pub fn identity(&self) -> Result<Digest> {
        Digest::of_serializable("flow-match-euler-schedule-v1", self).map_err(Error::from)
    }
}

fn bounded_steps(steps: u32) -> Result<u16> {
    if steps == 0 || usize::try_from(steps).map_or(true, |value| value > MAX_DIFFUSION_STEPS) {
        return Err(invalid("steps must be within the diffusion bound"));
    }
    u16::try_from(steps).map_err(|_| invalid("validated steps do not fit the schedule bound"))
}

/// `DiscreteFlowDenoiser::t_to_sigma`: `time_snr_shift(shift, (t + 1) / 1000)`.
fn discrete_flow_sigma(shift: f32, t: f32) -> f32 {
    let t = (t + 1.0) / 1000.0;
    #[allow(
        clippy::float_cmp,
        reason = "upstream short-circuits exactly this value and so must we"
    )]
    if shift == 1.0 {
        return t;
    }
    shift * t / (1.0 + (shift - 1.0) * t)
}

/// `Flux2Scheduler::compute_empirical_mu`, constants and order as upstream.
fn flux2_empirical_mu(image_seq_len: u32, steps: u16) -> f32 {
    const A1: f32 = 8.738_095e-5;
    const B1: f32 = 1.898_333_3;
    const A2: f32 = 0.000_169_27;
    const B2: f32 = 0.456_666_66;
    #[allow(
        clippy::cast_precision_loss,
        reason = "upstream converts the int sequence length to float the same way"
    )]
    let seq = image_seq_len as f32;
    if image_seq_len > 4_300 {
        return A2 * seq + B2;
    }
    let m_200 = A2 * seq + B2;
    let m_10 = A1 * seq + B1;
    let a = (m_200 - m_10) / 190.0;
    let b = m_200 - 200.0 * a;
    a * f32::from(steps) + b
}

fn invalid(reason: &'static str) -> Error {
    Error::Contract(CoreError::invalid("flow-match schedule", reason))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discrete_flow_shift_matches_upstream_endpoints() {
        let schedule = FlowMatchScheduleV1::discrete_flow_shift(8, 3.0).unwrap();
        let sigmas = schedule.sigmas();

        assert_eq!(sigmas.len(), 9);
        assert_eq!(sigmas[0].to_bits(), 1.0_f32.to_bits());
        assert_eq!(sigmas[8].to_bits(), 0.0_f32.to_bits());
        // t = 0 maps to (0 + 1) / 1000 before the shift: upstream's last
        // non-terminal sigma sits near 0.003, not near 1 / steps. The grid
        // point is `999 - (999 / 7) * 7` in f32, which is not exactly zero in
        // upstream either, so the value is compared analytically.
        let expected_last = 3.0_f64 * 0.001 / (1.0 + 2.0 * 0.001);
        assert!((f64::from(sigmas[7]) - expected_last).abs() < 1e-6);
        assert!(sigmas.windows(2).all(|pair| pair[0] > pair[1]));
    }

    #[test]
    fn discrete_flow_single_step_uses_the_top_of_the_grid() {
        let schedule = FlowMatchScheduleV1::discrete_flow_shift(1, 3.0).unwrap();
        assert_eq!(schedule.sigmas(), vec![1.0, 0.0]);
    }

    #[test]
    fn flux2_mu_matches_the_fitted_line_at_1024_square() {
        // 1024x1024 through the FLUX.2 VAE is 64 x 64 latent tokens.
        let schedule = FlowMatchScheduleV1::flux2_empirical(4, 4_096).unwrap();
        let FlowScheduleFamilyV1::Flux2Empirical { mu_bits, .. } = schedule.family() else {
            panic!("family must be FLUX.2");
        };
        let mu = f32::from_bits(mu_bits);
        // Independent f64 evaluation of upstream's piecewise fit.
        let seq = 4_096.0_f64;
        let m_200 = 0.000_169_27 * seq + 0.456_666_66;
        let m_10 = 8.738_095_24e-5 * seq + 1.898_333_33;
        let a = (m_200 - m_10) / 190.0;
        let expected = a * 4.0 + (m_200 - 200.0 * a);
        assert!((f64::from(mu) - expected).abs() < 1e-4);

        let sigmas = schedule.sigmas();
        assert_eq!(sigmas.len(), 5);
        assert_eq!(sigmas[0].to_bits(), 1.0_f32.to_bits());
        assert_eq!(sigmas[4].to_bits(), 0.0_f32.to_bits());
        assert!(sigmas.windows(2).all(|pair| pair[0] > pair[1]));
    }

    #[test]
    fn flux2_long_sequences_use_the_steps_independent_line() {
        let short = FlowMatchScheduleV1::flux2_empirical(4, 8_192).unwrap();
        let long = FlowMatchScheduleV1::flux2_empirical(28, 8_192).unwrap();
        assert_eq!(
            match short.family() {
                FlowScheduleFamilyV1::Flux2Empirical { mu_bits, .. } => mu_bits,
                FlowScheduleFamilyV1::DiscreteFlowShift { .. } | FlowScheduleFamilyV1::Explicit =>
                    0,
            },
            match long.family() {
                FlowScheduleFamilyV1::Flux2Empirical { mu_bits, .. } => mu_bits,
                FlowScheduleFamilyV1::DiscreteFlowShift { .. } | FlowScheduleFamilyV1::Explicit =>
                    1,
            }
        );
    }

    #[test]
    fn identity_binds_family_steps_and_parameters() {
        let base = FlowMatchScheduleV1::discrete_flow_shift(8, 3.0).unwrap();
        let other_shift = FlowMatchScheduleV1::discrete_flow_shift(8, 3.16).unwrap();
        let other_steps = FlowMatchScheduleV1::discrete_flow_shift(9, 3.0).unwrap();
        let flux = FlowMatchScheduleV1::flux2_empirical(8, 4_096).unwrap();
        let ids = [
            base.identity().unwrap(),
            other_shift.identity().unwrap(),
            other_steps.identity().unwrap(),
            flux.identity().unwrap(),
        ];
        for (left, right) in ids.iter().zip(ids.iter().skip(1)) {
            assert_ne!(left, right);
        }
        assert_ne!(ids[0], ids[3]);
    }

    /// Qwen-Image 2.1 Turbo's published eight-step boundaries.
    const QWEN_IMAGE_21_TURBO: [f32; 9] = [
        1.0, 0.978_453, 0.954_18, 0.926_626, 0.895_08, 0.845_148, 0.704_534, 0.414_568, 0.0,
    ];

    #[test]
    fn explicit_schedule_keeps_published_boundaries_bit_for_bit() {
        let schedule = FlowMatchScheduleV1::explicit(&QWEN_IMAGE_21_TURBO).unwrap();
        assert_eq!(schedule.family(), FlowScheduleFamilyV1::Explicit);
        assert_eq!(schedule.steps(), 8);
        let expected: Vec<u32> = QWEN_IMAGE_21_TURBO.iter().map(|s| s.to_bits()).collect();
        assert_eq!(schedule.sigma_bits(), expected.as_slice());
    }

    #[test]
    fn explicit_identity_binds_every_boundary() {
        let base = FlowMatchScheduleV1::explicit(&QWEN_IMAGE_21_TURBO).unwrap();
        let mut nudged = QWEN_IMAGE_21_TURBO;
        nudged[4] = f32::from_bits(nudged[4].to_bits() - 1);
        let other = FlowMatchScheduleV1::explicit(&nudged).unwrap();
        let shifted = FlowMatchScheduleV1::discrete_flow_shift(8, 3.0).unwrap();
        assert_ne!(base.identity().unwrap(), other.identity().unwrap());
        assert_ne!(base.identity().unwrap(), shifted.identity().unwrap());
    }

    #[test]
    fn explicit_schedule_refuses_malformed_boundaries() {
        assert!(FlowMatchScheduleV1::explicit(&[]).is_err());
        assert!(FlowMatchScheduleV1::explicit(&[0.0]).is_err());
        assert!(FlowMatchScheduleV1::explicit(&[1.0, 0.5]).is_err());
        assert!(FlowMatchScheduleV1::explicit(&[1.0, -0.0]).is_err());
        assert!(FlowMatchScheduleV1::explicit(&[0.5, 1.0, 0.0]).is_err());
        assert!(FlowMatchScheduleV1::explicit(&[1.0, 0.0, 0.0]).is_err());
        assert!(FlowMatchScheduleV1::explicit(&[f32::NAN, 0.0]).is_err());
        assert!(FlowMatchScheduleV1::explicit(&[f32::INFINITY, 0.0]).is_err());
        assert!(FlowMatchScheduleV1::explicit(&[1.0, 0.0]).is_ok());
    }

    #[test]
    fn invalid_controls_fail_before_allocation() {
        assert!(FlowMatchScheduleV1::discrete_flow_shift(0, 3.0).is_err());
        assert!(FlowMatchScheduleV1::discrete_flow_shift(8, 0.0).is_err());
        assert!(FlowMatchScheduleV1::discrete_flow_shift(8, f32::NAN).is_err());
        assert!(FlowMatchScheduleV1::flux2_empirical(8, 0).is_err());
        assert!(FlowMatchScheduleV1::flux2_empirical(u32::MAX, 4_096).is_err());
    }
}

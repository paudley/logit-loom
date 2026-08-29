// SPDX-License-Identifier: MIT OR Apache-2.0

//! Versioned Krea discrete-flow sampling schedules.

use logit_loom_core::{CoreError, Digest};
use serde::{Deserialize, Serialize};

use crate::{Error, MAX_DIFFUSION_STEPS, Result};

/// Default transition count published for Krea 2 Turbo.
pub const KREA_2_TURBO_DEFAULT_STEPS: u32 = 8;
/// Fixed exponential time-shift `mu` published for Krea 2 Turbo.
pub const KREA_2_TURBO_DEFAULT_FLOW_SHIFT: f32 = 1.15;

/// Exact Krea discrete-flow Euler schedule contract.
///
/// Floating-point values are retained as IEEE-754 bit patterns so the
/// schedule identity is independent of text formatting and two mechanically
/// different schedules cannot share a receipt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KreaDiscreteFlowScheduleV1 {
    steps: u32,
    flow_shift_bits: u32,
    sigma_bits: Vec<u32>,
}

impl KreaDiscreteFlowScheduleV1 {
    /// Constructs the version-one Krea discrete-flow schedule.
    ///
    /// # Errors
    ///
    /// Returns an error for zero or excessive steps, or for a non-finite or
    /// non-positive flow shift.
    pub fn new(steps: u32, flow_shift: f32) -> Result<Self> {
        if steps == 0
            || usize::try_from(steps).map_or(true, |value| value > MAX_DIFFUSION_STEPS)
            || !flow_shift.is_finite()
            || flow_shift <= 0.0
        {
            return Err(Error::Contract(CoreError::invalid(
                "Krea discrete-flow schedule",
                "steps must be within the diffusion bound and flow shift must be finite and positive",
            )));
        }

        let bounded_steps = u16::try_from(steps).map_err(|_| {
            Error::Contract(CoreError::invalid(
                "Krea discrete-flow schedule",
                "validated steps do not fit the bounded schedule representation",
            ))
        })?;
        let capacity = usize::from(bounded_steps).saturating_add(1);
        let mut sigmas = Vec::with_capacity(capacity);
        let exponent = flow_shift.exp();
        // Krea 2 supplies FlowMatchEulerDiscreteScheduler with
        // linspace(1, 1 / steps, steps), then applies its exponential
        // time-shift with sigma=1.  This is deliberately not a training-grid
        // linspace from timestep 999 down to zero: that old approximation
        // drove the last pre-terminal boundary near 0.001 instead of 1/steps.
        for index in 0..bounded_steps {
            let normalized = f32::from(bounded_steps - index) / f32::from(bounded_steps);
            let sigma = exponent / (exponent + (1.0 / normalized - 1.0));
            sigmas.push(sigma);
        }
        sigmas.push(0.0_f32);

        Ok(Self {
            steps,
            flow_shift_bits: flow_shift.to_bits(),
            sigma_bits: sigmas.into_iter().map(f32::to_bits).collect(),
        })
    }

    /// Returns the exact diffusion transition count.
    #[must_use]
    pub const fn steps(&self) -> u32 {
        self.steps
    }

    /// Returns the exact discrete-flow time shift.
    #[must_use]
    pub const fn flow_shift(&self) -> f32 {
        f32::from_bits(self.flow_shift_bits)
    }

    /// Returns the exact time-shift bit pattern.
    #[must_use]
    pub const fn flow_shift_bits(&self) -> u32 {
        self.flow_shift_bits
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
        Digest::of_serializable("krea-discrete-flow-euler-schedule-v1", self).map_err(Error::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turbo_schedule_is_exact_monotone_and_terminal() {
        let schedule = KreaDiscreteFlowScheduleV1::new(
            KREA_2_TURBO_DEFAULT_STEPS,
            KREA_2_TURBO_DEFAULT_FLOW_SHIFT,
        )
        .unwrap();
        let sigmas = schedule.sigmas();

        assert_eq!(schedule.steps(), 8);
        assert_eq!(schedule.flow_shift_bits(), 1.15_f32.to_bits());
        assert_eq!(sigmas.len(), 9);
        assert_eq!(sigmas[0].to_bits(), 1.0_f32.to_bits());
        assert_eq!(sigmas[8].to_bits(), 0.0_f32.to_bits());
        assert!(sigmas[7] > 0.25);
        assert!(sigmas[7] < 0.5);
        assert!(sigmas.windows(2).all(|pair| pair[0] > pair[1]));
    }

    #[test]
    fn turbo_grid_uses_one_over_steps_before_the_exponential_shift() {
        let schedule = KreaDiscreteFlowScheduleV1::new(8, 1.15).unwrap();
        let exponent = 1.15_f32.exp();
        let expected_last = exponent / (exponent + (1.0 / 0.125_f32 - 1.0));

        assert_eq!(schedule.sigmas()[7].to_bits(), expected_last.to_bits());
        assert_ne!(schedule.sigmas()[7].to_bits(), 0.001_f32.to_bits());
    }

    #[test]
    fn identity_binds_steps_shift_and_boundaries() {
        let default = KreaDiscreteFlowScheduleV1::new(8, 1.15).unwrap();
        let fewer_steps = KreaDiscreteFlowScheduleV1::new(7, 1.15).unwrap();
        let shifted = KreaDiscreteFlowScheduleV1::new(8, 1.0).unwrap();

        assert_ne!(default.identity().unwrap(), fewer_steps.identity().unwrap());
        assert_ne!(default.identity().unwrap(), shifted.identity().unwrap());
    }

    #[test]
    fn invalid_controls_fail_before_schedule_allocation() {
        assert!(KreaDiscreteFlowScheduleV1::new(0, 1.15).is_err());
        assert!(KreaDiscreteFlowScheduleV1::new(u32::MAX, 1.15).is_err());
        assert!(KreaDiscreteFlowScheduleV1::new(8, 0.0).is_err());
        assert!(KreaDiscreteFlowScheduleV1::new(8, f32::NAN).is_err());
    }
}

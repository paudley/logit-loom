// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded normalized-mono audio prefill contracts.

use serde::{Deserialize, Serialize};

use crate::{CoreError, Digest};

/// Largest projector-declared sampling rate accepted by this contract.
pub const MAX_AUDIO_SAMPLE_RATE_HZ: u32 = 192_000;
/// Largest number of mono frames accepted by one audio-prefill plan.
pub const MAX_AUDIO_FRAMES: u32 = 5_760_000;

/// Exact, model-free contract for one audio projector prefill.
///
/// Audio bytes are deliberately not serialized into this plan. The caller
/// passes a bounded normalized mono slice to [`Self::validate_samples`] and
/// binds the private bytes with an [`AudioPrefillReceiptV1`] source digest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioPrefillPlanV1 {
    /// Identity of the projector that declares the input sample rate.
    pub projector: Digest,
    /// Required mono PCM sample rate, in hertz.
    pub projector_sample_rate_hz: u32,
    /// Inclusive maximum number of normalized mono PCM frames.
    pub maximum_frames: u32,
}

impl AudioPrefillPlanV1 {
    /// Validates plan bounds and returns its stable identity.
    ///
    /// # Errors
    ///
    /// Returns an error when the declared rate or frame bound is unusable.
    pub fn digest(&self) -> Result<Digest, CoreError> {
        if self.projector_sample_rate_hz == 0
            || self.projector_sample_rate_hz > MAX_AUDIO_SAMPLE_RATE_HZ
        {
            return Err(CoreError::invalid(
                "audio prefill plan",
                "projector sample rate is outside the supported bound",
            ));
        }
        if self.maximum_frames == 0 || self.maximum_frames > MAX_AUDIO_FRAMES {
            return Err(CoreError::invalid(
                "audio prefill plan",
                "maximum frames is outside the supported bound",
            ));
        }
        Digest::of_serializable("audio-prefill-plan-v1", self)
    }

    /// Validates one caller-owned normalized mono PCM slice.
    ///
    /// This accepts only finite samples in the closed normalized interval
    /// `[-1.0, 1.0]`. The slice itself remains caller-owned and is never
    /// retained by the plan or its receipt.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty, over-bound, non-finite, or non-normalized
    /// sample sequence.
    pub fn validate_samples(&self, samples: &[f32]) -> Result<u32, CoreError> {
        self.digest()?;
        let frames = u32::try_from(samples.len())
            .map_err(|_| CoreError::invalid("audio prefill samples", "frame count exceeds u32"))?;
        if frames == 0 || frames > self.maximum_frames {
            return Err(CoreError::invalid(
                "audio prefill samples",
                "frame count is outside the declared bound",
            ));
        }
        if samples
            .iter()
            .any(|sample| !sample.is_finite() || !(-1.0..=1.0).contains(sample))
        {
            return Err(CoreError::invalid(
                "audio prefill samples",
                "samples must be finite normalized mono PCM values",
            ));
        }
        Ok(frames)
    }
}

/// Content-free terminal accounting for one projected audio prefill.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioPrefillReceiptV1 {
    /// Stable identity of the accepted audio-prefill plan.
    pub plan: Digest,
    /// Caller-provided private-audio identity; no PCM is retained here.
    pub audio: Digest,
    /// Number of validated normalized mono input frames.
    pub frames: u32,
    /// Projected tokens causally admitted by the model.
    pub admitted_tokens: u32,
    /// Causal position before projection and prefill.
    pub initial_position: u64,
    /// Causal position after projection and prefill.
    pub final_position: u64,
    /// Optional controlled-prefill receipt identity from the text runtime.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_receipt: Option<Digest>,
}

impl AudioPrefillReceiptV1 {
    /// Validates exact plan and causal accounting, then returns a stable receipt
    /// identity.
    ///
    /// # Errors
    ///
    /// Returns an error for a mismatched plan, invalid frame count, or invalid
    /// causal position progression.
    pub fn digest_for(&self, plan: &AudioPrefillPlanV1) -> Result<Digest, CoreError> {
        let plan_digest = plan.digest()?;
        if self.plan != plan_digest {
            return Err(CoreError::invalid(
                "audio prefill receipt",
                "receipt plan does not match the supplied plan",
            ));
        }
        if self.frames == 0 || self.frames > plan.maximum_frames {
            return Err(CoreError::invalid(
                "audio prefill receipt",
                "receipt frame count is outside the declared bound",
            ));
        }
        let expected_position = self
            .initial_position
            .checked_add(u64::from(self.admitted_tokens))
            .ok_or_else(|| CoreError::invalid("audio prefill receipt", "position overflowed"))?;
        if self.final_position != expected_position {
            return Err(CoreError::invalid(
                "audio prefill receipt",
                "final position does not match admitted tokens",
            ));
        }
        Digest::of_serializable("audio-prefill-receipt-v1", self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan() -> AudioPrefillPlanV1 {
        AudioPrefillPlanV1 {
            projector: Digest::of_bytes("test-projector", b"qwen3-asr"),
            projector_sample_rate_hz: 16_000,
            maximum_frames: 16_000,
        }
    }

    #[test]
    fn normalized_mono_samples_are_bounded_without_retention() {
        let plan = plan();
        assert_eq!(plan.validate_samples(&[-1.0, 0.0, 1.0]).unwrap(), 3);
        assert!(plan.validate_samples(&[]).is_err());
        assert!(plan.validate_samples(&[f32::NAN]).is_err());
        assert!(plan.validate_samples(&[1.01]).is_err());
    }

    #[test]
    fn receipt_binds_plan_and_exact_causal_progression() {
        let plan = plan();
        let receipt = AudioPrefillReceiptV1 {
            plan: plan.digest().unwrap(),
            audio: Digest::of_bytes("private-audio", b"identity-only"),
            frames: 8_000,
            admitted_tokens: 12,
            initial_position: 4,
            final_position: 16,
            prefill_receipt: None,
        };
        assert!(receipt.digest_for(&plan).is_ok());
        let mut malformed = receipt;
        malformed.final_position = 15;
        assert!(malformed.digest_for(&plan).is_err());
    }
}

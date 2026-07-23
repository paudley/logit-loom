// SPDX-License-Identifier: MIT OR Apache-2.0

//! Backend-neutral generation and native-sampler plans.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::{CoreError, Digest, TokenId};

/// Maximum exact sequence breakers accepted by one DRY sampler.
pub const MAX_DRY_SEQUENCE_BREAKERS: usize = 64;
/// Maximum UTF-8 bytes accepted in one DRY sequence breaker.
pub const MAX_DRY_SEQUENCE_BREAKER_BYTES: usize = 40;
/// Maximum additive token biases accepted by one generation plan.
pub const MAX_LOGIT_BIASES: usize = 65_536;
/// Maximum UTF-8 bytes accepted in an eager grammar source.
pub const MAX_GRAMMAR_SOURCE_BYTES: usize = 1024 * 1024;
/// Maximum UTF-8 bytes accepted in an eager grammar root name.
pub const MAX_GRAMMAR_ROOT_BYTES: usize = 256;
/// Maximum exact byte stop sequences accepted by one generation plan.
pub const MAX_STOP_SEQUENCES: usize = 64;
/// Maximum bytes accepted in one exact stop sequence.
pub const MAX_STOP_SEQUENCE_BYTES: usize = 4_096;

/// Stateful repetition, frequency, and presence penalties.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RepetitionSampler {
    /// Number of preceding tokens to retain; `-1` selects the context bound.
    pub last_n: i32,
    /// Multiplicative repeat penalty; `1.0` is neutral.
    pub repeat_penalty: f32,
    /// Additive frequency penalty per occurrence.
    pub frequency_penalty: f32,
    /// Additive penalty when a token is present.
    pub presence_penalty: f32,
}

/// Don't Repeat Yourself sampler configuration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DrySampler {
    /// Positive penalty multiplier.
    pub multiplier: f32,
    /// Exponential penalty base, at least `1.0`.
    pub base: f32,
    /// Prefix length allowed before penalties begin.
    pub allowed_length: i32,
    /// Number of preceding tokens to scan; `-1` selects the context bound.
    pub penalty_last_n: i32,
    /// Exact sequence breakers.
    pub sequence_breakers: Vec<String>,
}

/// Native Mirostat algorithm version.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MirostatVersion {
    /// Original Mirostat algorithm.
    V1,
    /// Mirostat 2.0.
    V2,
}

/// Stateful Mirostat terminal sampler configuration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MirostatSampler {
    /// Selected algorithm.
    pub version: MirostatVersion,
    /// Target surprisal.
    pub tau: f32,
    /// Adaptive learning rate.
    pub eta: f32,
    /// Positive candidate-estimation window for version one; zero for version
    /// two.
    pub m: i32,
}

/// Native sampler configuration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SamplingPlan {
    /// Deterministic sampler seed.
    pub seed: u32,
    /// Temperature; zero selects greedy decoding.
    pub temperature: f32,
    /// Top-k candidate count; zero disables the filter.
    pub top_k: i32,
    /// Nucleus probability.
    pub top_p: f32,
    /// Minimum probability relative to the most likely token.
    pub min_p: f32,
    /// Locally typical probability mass; `1.0` disables the filter.
    pub typical_p: f32,
    /// Optional repetition/frequency/presence penalties.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repetition: Option<RepetitionSampler>,
    /// Optional DRY sampler.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dry: Option<DrySampler>,
    /// Optional Mirostat terminal sampler.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mirostat: Option<MirostatSampler>,
}

impl Default for SamplingPlan {
    fn default() -> Self {
        Self {
            seed: 0,
            temperature: 0.0,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            typical_p: 1.0,
            repetition: None,
            dry: None,
            mirostat: None,
        }
    }
}

impl SamplingPlan {
    /// Validates finite values and native sampler bounds.
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported numeric values.
    pub fn validate(&self) -> Result<(), CoreError> {
        for (field, value) in [
            ("temperature", self.temperature),
            ("top p", self.top_p),
            ("min p", self.min_p),
            ("typical p", self.typical_p),
        ] {
            if !value.is_finite() {
                return Err(CoreError::NonFinite { field });
            }
        }
        if self.temperature < 0.0
            || self.top_k < 0
            || !(0.0..=1.0).contains(&self.top_p)
            || !(0.0..=1.0).contains(&self.min_p)
            || !(0.0..=1.0).contains(&self.typical_p)
        {
            return Err(CoreError::invalid(
                "sampling plan",
                "temperature/top-k must be non-negative and probabilities must be in 0..=1",
            ));
        }
        if let Some(repetition) = &self.repetition {
            for (field, value) in [
                ("repeat penalty", repetition.repeat_penalty),
                ("frequency penalty", repetition.frequency_penalty),
                ("presence penalty", repetition.presence_penalty),
            ] {
                if !value.is_finite() {
                    return Err(CoreError::NonFinite { field });
                }
            }
            if repetition.last_n < -1 || repetition.repeat_penalty <= 0.0 {
                return Err(CoreError::invalid(
                    "repetition sampler",
                    "last_n must be at least -1 and repeat_penalty must be positive",
                ));
            }
        }
        if let Some(dry) = &self.dry {
            for (field, value) in [("DRY multiplier", dry.multiplier), ("DRY base", dry.base)] {
                if !value.is_finite() {
                    return Err(CoreError::NonFinite { field });
                }
            }
            if dry.multiplier <= 0.0
                || dry.base < 1.0
                || dry.allowed_length < 0
                || dry.penalty_last_n < -1
                || dry.penalty_last_n == 0
            {
                return Err(CoreError::invalid(
                    "dry sampler",
                    "multiplier must be positive, base at least one, allowed length non-negative, and lookback -1 or positive",
                ));
            }
            if dry.sequence_breakers.len() > MAX_DRY_SEQUENCE_BREAKERS
                || dry.sequence_breakers.iter().any(|value| {
                    value.is_empty()
                        || value.len() > MAX_DRY_SEQUENCE_BREAKER_BYTES
                        || value.contains('\0')
                })
            {
                return Err(CoreError::invalid(
                    "dry sequence breakers",
                    format!(
                        "requires at most {MAX_DRY_SEQUENCE_BREAKERS} non-empty, NUL-free values of at most {MAX_DRY_SEQUENCE_BREAKER_BYTES} bytes"
                    ),
                ));
            }
        }
        if let Some(mirostat) = &self.mirostat {
            for (field, value) in [
                ("Mirostat tau", mirostat.tau),
                ("Mirostat eta", mirostat.eta),
            ] {
                if !value.is_finite() {
                    return Err(CoreError::NonFinite { field });
                }
            }
            let invalid_window = match mirostat.version {
                MirostatVersion::V1 => mirostat.m <= 0,
                MirostatVersion::V2 => mirostat.m != 0,
            };
            if self.temperature <= 0.0
                || mirostat.tau <= 0.0
                || mirostat.eta <= 0.0
                || invalid_window
            {
                return Err(CoreError::invalid(
                    "mirostat sampler",
                    "temperature, tau, and eta must be positive; m must be positive for v1 and zero for v2",
                ));
            }
        }
        Ok(())
    }
}

/// One additive token-logit bias.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LogitBias {
    /// Target token.
    pub token: TokenId,
    /// Finite additive bias.
    pub bias: f32,
}

/// An eager llama.cpp GBNF grammar.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Grammar {
    /// Complete GBNF source.
    pub source: String,
    /// Root rule name.
    pub root: String,
}

impl Grammar {
    /// Validates bounded strings before they cross a native C-string boundary.
    ///
    /// # Errors
    ///
    /// Returns an error for empty, oversized, or NUL-containing values.
    pub fn validate(&self) -> Result<(), CoreError> {
        if self.source.is_empty() || self.root.is_empty() {
            return Err(CoreError::Empty {
                field: "grammar source/root",
            });
        }
        if self.source.len() > MAX_GRAMMAR_SOURCE_BYTES || self.root.len() > MAX_GRAMMAR_ROOT_BYTES
        {
            return Err(CoreError::invalid(
                "grammar",
                format!(
                    "source must be at most {MAX_GRAMMAR_SOURCE_BYTES} bytes and root at most {MAX_GRAMMAR_ROOT_BYTES} bytes"
                ),
            ));
        }
        if self.source.contains('\0') || self.root.contains('\0') {
            return Err(CoreError::invalid(
                "grammar",
                "source and root must not contain NUL bytes",
            ));
        }
        Ok(())
    }
}

/// Complete bounded generation mechanics independent of prompt semantics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerationPlan {
    /// Native sampler configuration.
    pub sampling: SamplingPlan,
    /// Maximum causally admitted output tokens.
    pub max_tokens: u32,
    /// Additive biases with unique token IDs, applied after Rust transforms and
    /// before filters.
    #[serde(default)]
    pub biases: Vec<LogitBias>,
    /// Optional eager grammar.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grammar: Option<Grammar>,
    /// Exact byte suffixes that stop after causal admission.
    ///
    /// Matching bytes remain in output and causal state. When several suffixes
    /// match the same output, the lowest declared index wins.
    #[serde(default)]
    pub stops: Vec<Vec<u8>>,
}

impl GenerationPlan {
    /// Validates all bounded generation mechanics.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid samplers, bounds, biases, grammar, or stops.
    pub fn validate(&self) -> Result<(), CoreError> {
        self.sampling.validate()?;
        if self.max_tokens == 0 {
            return Err(CoreError::invalid(
                "maximum tokens",
                "must be greater than zero",
            ));
        }
        if self.biases.len() > MAX_LOGIT_BIASES
            || self.biases.iter().any(|bias| !bias.bias.is_finite())
        {
            return Err(CoreError::invalid(
                "logit biases",
                "too many biases or a non-finite value",
            ));
        }
        if self
            .biases
            .iter()
            .map(|bias| bias.token)
            .collect::<HashSet<_>>()
            .len()
            != self.biases.len()
        {
            return Err(CoreError::invalid(
                "logit biases",
                "token identifiers must be unique",
            ));
        }
        if let Some(grammar) = &self.grammar {
            grammar.validate()?;
        }
        if self.stops.len() > MAX_STOP_SEQUENCES
            || self
                .stops
                .iter()
                .any(|stop| stop.is_empty() || stop.len() > MAX_STOP_SEQUENCE_BYTES)
        {
            return Err(CoreError::invalid(
                "stop sequences",
                format!(
                    "requires at most {MAX_STOP_SEQUENCES} non-empty sequences of at most {MAX_STOP_SEQUENCE_BYTES} bytes"
                ),
            ));
        }
        Ok(())
    }

    /// Returns a content identity for exact generation mechanics.
    ///
    /// # Errors
    ///
    /// Returns a validation or serialization error.
    pub fn digest(&self) -> Result<Digest, CoreError> {
        self.validate()?;
        Digest::of_serializable("generation-plan-v1", self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan() -> GenerationPlan {
        GenerationPlan {
            sampling: SamplingPlan::default(),
            max_tokens: 8,
            biases: Vec::new(),
            grammar: None,
            stops: vec![b"stop".to_vec()],
        }
    }

    #[test]
    fn generation_plan_roundtrip_preserves_identity() {
        let original = plan();
        let encoded = serde_json::to_vec(&original).unwrap();
        let decoded: GenerationPlan = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(original.digest().unwrap(), decoded.digest().unwrap());
    }

    #[test]
    fn generation_plan_rejects_unbounded_or_nonfinite_mechanics() {
        let mut invalid = plan();
        invalid.max_tokens = 0;
        assert!(invalid.validate().is_err());
        invalid.max_tokens = 1;
        invalid.sampling.temperature = f32::NAN;
        assert!(invalid.validate().is_err());
        invalid.sampling.temperature = 0.0;
        invalid.stops = vec![Vec::new()];
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn stateful_sampler_bounds_are_explicit() {
        let mut sampling = SamplingPlan {
            temperature: 0.8,
            dry: Some(DrySampler {
                multiplier: 0.8,
                base: 1.75,
                allowed_length: 2,
                penalty_last_n: -1,
                sequence_breakers: vec!["\n".to_owned()],
            }),
            mirostat: Some(MirostatSampler {
                version: MirostatVersion::V2,
                tau: 5.0,
                eta: 0.1,
                m: 0,
            }),
            ..SamplingPlan::default()
        };
        assert!(sampling.validate().is_ok());

        sampling.dry.as_mut().unwrap().sequence_breakers = vec![String::new()];
        assert!(sampling.validate().is_err());
        sampling.dry.as_mut().unwrap().sequence_breakers = vec!["\n".to_owned()];
        sampling.dry.as_mut().unwrap().base = f32::NAN;
        assert!(matches!(
            sampling.validate(),
            Err(CoreError::NonFinite { field: "DRY base" })
        ));

        sampling.dry = None;
        sampling.temperature = 0.0;
        assert!(sampling.validate().is_err());

        sampling.temperature = 0.8;
        sampling.mirostat.as_mut().unwrap().m = 1;
        assert!(sampling.validate().is_err());
        sampling.mirostat = Some(MirostatSampler {
            version: MirostatVersion::V1,
            tau: 5.0,
            eta: 0.1,
            m: 0,
        });
        assert!(sampling.validate().is_err());
    }

    #[test]
    fn grammar_and_stop_inputs_have_explicit_byte_bounds() {
        let oversized_source = Grammar {
            source: "x".repeat(MAX_GRAMMAR_SOURCE_BYTES + 1),
            root: "root".to_owned(),
        };
        assert!(oversized_source.validate().is_err());

        let oversized_root = Grammar {
            source: "root ::= \"x\"".to_owned(),
            root: "x".repeat(MAX_GRAMMAR_ROOT_BYTES + 1),
        };
        assert!(oversized_root.validate().is_err());

        let mut invalid = plan();
        invalid.stops = vec![vec![b'x'; MAX_STOP_SEQUENCE_BYTES + 1]];
        assert!(invalid.validate().is_err());
        invalid.stops = vec![vec![b'x']; MAX_STOP_SEQUENCES + 1];
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn every_logit_bias_must_be_finite() {
        let mut invalid = plan();
        invalid.biases.push(LogitBias {
            token: TokenId::new(7).unwrap(),
            bias: f32::INFINITY,
        });
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn generation_logit_bias_tokens_must_be_unique() {
        let mut invalid = plan();
        invalid.biases = vec![
            LogitBias {
                token: TokenId::new(7).unwrap(),
                bias: 1.0,
            },
            LogitBias {
                token: TokenId::new(7).unwrap(),
                bias: 2.0,
            },
        ];
        assert!(invalid.validate().is_err());
    }
}

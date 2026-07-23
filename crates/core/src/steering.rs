// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reversible session-steering descriptors and lifecycle receipts.

use serde::{Deserialize, Serialize};

use crate::{CoreError, Digest};

/// One model-compatible `LoRA` adapter application.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoraSpec {
    /// Adapter artifact identity.
    pub artifact: Digest,
    /// Finite native scale.
    pub scale: f32,
}

impl LoraSpec {
    /// Validates a finite, nonzero scale.
    ///
    /// # Errors
    ///
    /// Returns an error for zero, NaN, or infinite scale.
    pub fn validate(&self) -> Result<(), CoreError> {
        if !self.scale.is_finite() {
            return Err(CoreError::NonFinite {
                field: "LoRA scale",
            });
        }
        if self.scale == 0.0 {
            return Err(CoreError::invalid("LoRA scale", "must be nonzero"));
        }
        Ok(())
    }

    /// Returns a content identity for this exact application.
    ///
    /// # Errors
    ///
    /// Returns a validation or serialization error.
    pub fn digest(&self) -> Result<Digest, CoreError> {
        self.validate()?;
        Digest::of_serializable("lora-spec-v1", self)
    }
}

/// One in-memory control-vector application.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlVectorSpec {
    /// Exact native `f32` data identity.
    pub data: Digest,
    /// Values per model layer.
    pub embedding_width: u32,
    /// Inclusive first model layer; layer zero is not steerable.
    pub layer_start: i32,
    /// Inclusive final model layer.
    pub layer_end: i32,
}

impl ControlVectorSpec {
    /// Validates width and the inclusive layer range.
    ///
    /// # Errors
    ///
    /// Returns an error for a zero width or malformed layer range.
    pub fn validate(&self) -> Result<(), CoreError> {
        if self.embedding_width == 0 || self.layer_start < 1 || self.layer_end < self.layer_start {
            return Err(CoreError::invalid(
                "control vector",
                "embedding width must be nonzero and inclusive layer range must start at one",
            ));
        }
        Ok(())
    }

    /// Returns a content identity for this exact application.
    ///
    /// # Errors
    ///
    /// Returns a validation or serialization error.
    pub fn digest(&self) -> Result<Digest, CoreError> {
        self.validate()?;
        Digest::of_serializable("control-vector-spec-v1", self)
    }
}

/// Session steering resource kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SteeringKind {
    /// A `LoRA` adapter.
    Lora,
    /// A control vector.
    ControlVector,
}

/// Lifecycle action recorded at the Rust/native boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SteeringAction {
    /// Resource became active.
    Applied,
    /// Resource was explicitly cleared.
    Cleared,
}

/// Mechanical receipt for one steering lifecycle transition.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SteeringReceipt {
    /// Resource kind.
    pub kind: SteeringKind,
    /// Resource descriptor identity.
    pub resource: Digest,
    /// Lifecycle action.
    pub action: SteeringAction,
    /// Causal position at the transition.
    pub position: u64,
}

impl SteeringReceipt {
    /// Returns a content identity for this transition.
    ///
    /// # Errors
    ///
    /// Returns a serialization error.
    pub fn digest(&self) -> Result<Digest, CoreError> {
        Digest::of_serializable("steering-receipt-v1", self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_vector_uses_an_inclusive_nonzero_layer_range() {
        let data = Digest::of_bytes("test-vector", b"one");
        let valid = ControlVectorSpec {
            data: data.clone(),
            embedding_width: 4,
            layer_start: 1,
            layer_end: 1,
        };
        assert!(valid.validate().is_ok());
        assert!(
            ControlVectorSpec {
                data,
                embedding_width: 4,
                layer_start: 0,
                layer_end: 1,
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn lora_scale_must_be_finite_and_nonzero() {
        let artifact = Digest::of_bytes("test-lora", b"one");
        for scale in [0.0, f32::NAN, f32::INFINITY] {
            assert!(
                LoraSpec {
                    artifact: artifact.clone(),
                    scale,
                }
                .validate()
                .is_err()
            );
        }
    }
}

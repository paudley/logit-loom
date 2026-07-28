// SPDX-License-Identifier: MIT OR Apache-2.0

//! Versioned topology, activation-capture, vector, and program contracts.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{CoreError, Digest};

/// Maximum transformer layers represented by one topology contract.
pub const MAX_TEXT_MODEL_LAYERS: u32 = 4_096;
/// Maximum elements in one reported model row.
pub const MAX_TEXT_TENSOR_ROW_ELEMENTS: u32 = 1_048_576;
/// Maximum experts represented by one topology contract.
pub const MAX_TEXT_MODEL_EXPERTS: u32 = 65_536;
/// Maximum exact tensor sites selected by one capture or operation.
pub const MAX_ACTIVATION_SITES: usize = 128;
/// Maximum inclusive causal ranges selected by one contract.
pub const MAX_ACTIVATION_POSITION_RANGES: usize = 128;
/// Maximum causal positions selected by one capture.
pub const MAX_ACTIVATION_CAPTURE_POSITIONS: u32 = 4_096;
/// Maximum `f32` elements copied or retained by one activation operation.
pub const MAX_ACTIVATION_ELEMENTS: u64 = 16_777_216;
/// Maximum retained activation bytes declared by one capture.
pub const MAX_ACTIVATION_RETAINED_BYTES: u64 = MAX_ACTIVATION_ELEMENTS * 4;
/// Maximum sparse vector rows in one bank.
pub const MAX_ACTIVATION_VECTOR_ROWS: usize = 4_096;
/// Maximum ordered activation operations in one program.
pub const MAX_ACTIVATION_OPERATIONS: usize = 256;
/// Maximum capture-plan observations named by one program.
pub const MAX_ACTIVATION_OBSERVATIONS: usize = 128;
/// Maximum input artifacts represented by vector provenance.
pub const MAX_ACTIVATION_VECTOR_INPUTS: usize = 65_536;
/// Maximum successful tensor transactions retained in one program receipt.
pub const MAX_ACTIVATION_INVOCATION_RECEIPTS: usize = 65_536;
/// Maximum logical rows represented by one tensor transaction.
pub const MAX_ACTIVATION_TRANSACTION_ROWS: usize = 4_096;
/// Maximum sequence identifiers attached to one logical tensor row.
pub const MAX_ACTIVATION_ROW_SEQUENCES: usize = 4_096;
/// Maximum UTF-8 bytes in a backend-profile tensor name.
pub const MAX_TEXT_TENSOR_NAME_BYTES: usize = 256;

/// Speculative mechanisms a loaded text topology can report.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextSpeculativeMechanismV1 {
    /// Same-model multi-token prediction heads.
    Mtp,
    /// A separately trained EAGLE-3 draft model.
    Eagle3,
}

/// Backend-neutral, content-bound model topology.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextModelTopologyV1 {
    /// Exact model artifact identity.
    pub model: Digest,
    /// Exact backend and safe-binding build identity.
    pub backend: Digest,
    /// Architecture-selector implementation identity.
    pub architecture_implementation: Digest,
    /// Number of transformer layers.
    pub layers: u32,
    /// Residual-stream row width.
    pub embedding_width: u32,
    /// Total `MoE` experts per applicable layer, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experts: Option<u32>,
    /// Experts selected per token, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experts_used: Option<u32>,
    /// Number of same-model NextN/MTP heads.
    pub nextn_layers: u32,
    /// Canonically ordered speculative mechanisms available to this topology.
    #[serde(default)]
    pub supported_speculation: Vec<TextSpeculativeMechanismV1>,
}

impl TextModelTopologyV1 {
    /// Validates topology bounds and returns its stable identity.
    ///
    /// # Errors
    ///
    /// Returns an error for unusable dimensions, inconsistent expert
    /// accounting, or a non-canonical mechanism set.
    pub fn digest(&self) -> Result<Digest, CoreError> {
        if self.layers == 0 || self.layers > MAX_TEXT_MODEL_LAYERS {
            return Err(CoreError::invalid(
                "text model topology",
                "layer count is outside the supported bound",
            ));
        }
        if self.embedding_width == 0 || self.embedding_width > MAX_TEXT_TENSOR_ROW_ELEMENTS {
            return Err(CoreError::invalid(
                "text model topology",
                "embedding width is outside the supported bound",
            ));
        }
        if let Some(experts) = self.experts
            && (experts == 0 || experts > MAX_TEXT_MODEL_EXPERTS)
        {
            return Err(CoreError::invalid(
                "text model topology",
                "expert count is outside the supported bound",
            ));
        }
        match (self.experts, self.experts_used) {
            (None, Some(_)) => {
                return Err(CoreError::invalid(
                    "text model topology",
                    "experts-used requires a reported expert count",
                ));
            }
            (Some(experts), Some(used)) if used == 0 || used > experts => {
                return Err(CoreError::invalid(
                    "text model topology",
                    "experts-used must be within the reported expert count",
                ));
            }
            _ => {}
        }
        if self.nextn_layers > self.layers {
            return Err(CoreError::invalid(
                "text model topology",
                "NextN layer count exceeds transformer layers",
            ));
        }
        validate_canonical_unique(
            &self.supported_speculation,
            "text model topology",
            "speculative mechanisms must be unique and canonically ordered",
        )?;
        let has_mtp = self
            .supported_speculation
            .contains(&TextSpeculativeMechanismV1::Mtp);
        if has_mtp != (self.nextn_layers > 0) {
            return Err(CoreError::invalid(
                "text model topology",
                "MTP support and NextN layer count disagree",
            ));
        }
        Digest::of_serializable("text-model-topology-v1", self)
    }
}

/// Tensor element representation expected at a selected graph site.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextTensorElementTypeV1 {
    /// Finite IEEE-754 single-precision values.
    F32,
    /// Signed 32-bit integer values.
    I32,
}

/// Stable family shared by compatible activation vector rows and tensor sites.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextTensorSiteFamilyV1 {
    /// Transformer residual layer output.
    LayerOutput,
    /// `MoE` router logits before probability normalization.
    RouterLogits,
    /// `MoE` router probabilities.
    RouterProbabilities,
    /// Integer selected-expert indices.
    SelectedExperts,
    /// A compatibility-profile-defined exact graph node.
    Named,
}

/// Exact graph tensor selected for capture or mutation.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TextTensorSiteV1 {
    /// Built-in transformer residual layer output.
    LayerOutput {
        /// Zero-based transformer layer.
        layer: u32,
    },
    /// Architecture-profile-bound `MoE` router logits.
    RouterLogits {
        /// Zero-based transformer layer.
        layer: u32,
        /// Exact graph selector implementation.
        selector: Digest,
    },
    /// Architecture-profile-bound `MoE` router probabilities.
    RouterProbabilities {
        /// Zero-based transformer layer.
        layer: u32,
        /// Exact graph selector implementation.
        selector: Digest,
    },
    /// Architecture-profile-bound selected-expert indices.
    SelectedExperts {
        /// Zero-based transformer layer.
        layer: u32,
        /// Exact graph selector implementation.
        selector: Digest,
    },
    /// Exact graph node defined by a compatibility profile.
    Named {
        /// Exact UTF-8 graph-node name.
        name: String,
        /// Exact graph selector implementation.
        selector: Digest,
        /// Required element representation.
        element_type: TextTensorElementTypeV1,
        /// Required elements in each selected causal row.
        row_elements: u32,
    },
}

impl TextTensorSiteV1 {
    /// Returns the stable site family.
    pub const fn family(&self) -> TextTensorSiteFamilyV1 {
        match self {
            Self::LayerOutput { .. } => TextTensorSiteFamilyV1::LayerOutput,
            Self::RouterLogits { .. } => TextTensorSiteFamilyV1::RouterLogits,
            Self::RouterProbabilities { .. } => TextTensorSiteFamilyV1::RouterProbabilities,
            Self::SelectedExperts { .. } => TextTensorSiteFamilyV1::SelectedExperts,
            Self::Named { .. } => TextTensorSiteFamilyV1::Named,
        }
    }

    /// Returns the selected layer for built-in layer-scoped sites.
    pub const fn layer(&self) -> Option<u32> {
        match self {
            Self::LayerOutput { layer }
            | Self::RouterLogits { layer, .. }
            | Self::RouterProbabilities { layer, .. }
            | Self::SelectedExperts { layer, .. } => Some(*layer),
            Self::Named { .. } => None,
        }
    }

    /// Returns the required element representation.
    pub const fn element_type(&self) -> TextTensorElementTypeV1 {
        match self {
            Self::SelectedExperts { .. } => TextTensorElementTypeV1::I32,
            Self::LayerOutput { .. }
            | Self::RouterLogits { .. }
            | Self::RouterProbabilities { .. } => TextTensorElementTypeV1::F32,
            Self::Named { element_type, .. } => *element_type,
        }
    }

    /// Returns the expected row width for the supplied topology.
    ///
    /// # Errors
    ///
    /// Returns an error when the site is incompatible with the topology or
    /// depends on expert metadata the topology does not report.
    pub fn row_elements(&self, topology: &TextModelTopologyV1) -> Result<u32, CoreError> {
        self.validate_for(topology)?;
        match self {
            Self::LayerOutput { .. } => Ok(topology.embedding_width),
            Self::RouterLogits { .. } | Self::RouterProbabilities { .. } => {
                topology.experts.ok_or_else(|| {
                    CoreError::invalid("text tensor site", "topology does not report experts")
                })
            }
            Self::SelectedExperts { .. } => topology.experts_used.ok_or_else(|| {
                CoreError::invalid(
                    "text tensor site",
                    "topology does not report selected experts",
                )
            }),
            Self::Named { row_elements, .. } => Ok(*row_elements),
        }
    }

    /// Validates the site against a loaded topology.
    ///
    /// # Errors
    ///
    /// Returns an error for an out-of-range layer, missing expert topology, an
    /// invalid named-node contract, or malformed topology.
    pub fn validate_for(&self, topology: &TextModelTopologyV1) -> Result<(), CoreError> {
        topology.digest()?;
        if let Some(layer) = self.layer()
            && layer >= topology.layers
        {
            return Err(CoreError::invalid(
                "text tensor site",
                "selected layer is outside the topology",
            ));
        }
        match self {
            Self::RouterLogits { .. } | Self::RouterProbabilities { .. } => {
                if topology.experts.is_none() {
                    return Err(CoreError::invalid(
                        "text tensor site",
                        "router site requires reported expert topology",
                    ));
                }
            }
            Self::SelectedExperts { .. } => {
                if topology.experts_used.is_none() {
                    return Err(CoreError::invalid(
                        "text tensor site",
                        "selected-expert site requires experts-used topology",
                    ));
                }
            }
            Self::Named {
                name, row_elements, ..
            } => {
                if name.is_empty()
                    || name.len() > MAX_TEXT_TENSOR_NAME_BYTES
                    || name.as_bytes().contains(&0)
                    || *row_elements == 0
                    || *row_elements > MAX_TEXT_TENSOR_ROW_ELEMENTS
                {
                    return Err(CoreError::invalid(
                        "text tensor site",
                        "named site requires a bounded non-NUL name and row width",
                    ));
                }
            }
            Self::LayerOutput { .. } => {}
        }
        Ok(())
    }

    /// Returns a topology-bound site identity.
    ///
    /// # Errors
    ///
    /// Returns a validation or serialization error.
    pub fn digest_for(&self, topology: &TextModelTopologyV1) -> Result<Digest, CoreError> {
        self.validate_for(topology)?;
        Digest::of_serializable("text-tensor-site-v1", &(topology.digest()?, self))
    }
}

/// Inclusive causal-position interval.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CausalPositionRangeV1 {
    /// Inclusive first causal position.
    pub start: u64,
    /// Inclusive final causal position.
    pub end: u64,
}

impl CausalPositionRangeV1 {
    /// Returns whether the range is reversed and therefore selects no
    /// positions.
    pub const fn is_empty(self) -> bool {
        self.start > self.end
    }

    /// Returns the number of positions in this inclusive range.
    ///
    /// # Errors
    ///
    /// Returns an error for a reversed or overflowing range.
    pub fn len(self) -> Result<u64, CoreError> {
        self.end
            .checked_sub(self.start)
            .and_then(|distance| distance.checked_add(1))
            .ok_or_else(|| {
                CoreError::invalid("causal position range", "range is reversed or overflowed")
            })
    }

    /// Returns whether the inclusive range contains a position.
    pub const fn contains(self, position: u64) -> bool {
        position >= self.start && position <= self.end
    }
}

/// Causal positions retained by one activation capture.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "ranges", rename_all = "snake_case")]
pub enum ActivationCapturePositionsV1 {
    /// Capture only the last token of a non-empty prefill.
    LastPrefillToken,
    /// Capture canonically ordered, disjoint inclusive causal ranges.
    InclusiveRanges(Vec<CausalPositionRangeV1>),
}

impl ActivationCapturePositionsV1 {
    /// Returns the exact explicit position count, or one for last-prefill mode.
    ///
    /// # Errors
    ///
    /// Returns an error for empty, unordered, overlapping, or over-bound
    /// ranges.
    pub fn position_count(&self) -> Result<u32, CoreError> {
        match self {
            Self::LastPrefillToken => Ok(1),
            Self::InclusiveRanges(ranges) => {
                validate_ranges(ranges, "activation capture positions")?;
                let count = ranges.iter().try_fold(0_u64, |total, range| {
                    total.checked_add(range.len()?).ok_or_else(|| {
                        CoreError::invalid(
                            "activation capture positions",
                            "position count overflowed",
                        )
                    })
                })?;
                let count = u32::try_from(count).map_err(|_| {
                    CoreError::invalid("activation capture positions", "position count exceeds u32")
                })?;
                if count > MAX_ACTIVATION_CAPTURE_POSITIONS {
                    return Err(CoreError::invalid(
                        "activation capture positions",
                        "position count exceeds the supported bound",
                    ));
                }
                Ok(count)
            }
        }
    }

    /// Returns whether an explicit range selection contains a position.
    ///
    /// Last-prefill selection is resolved by the adapter and returns `false`.
    pub fn contains_explicit(&self, position: u64) -> bool {
        match self {
            Self::LastPrefillToken => false,
            Self::InclusiveRanges(ranges) => ranges.iter().any(|range| range.contains(position)),
        }
    }
}

/// Data retained for each selected tensor row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivationCaptureRetentionV1 {
    /// Retain only exact tensor-byte identity and element count.
    Digest,
    /// Retain identity plus deterministic scalar statistics.
    Statistics,
    /// Retain the complete bounded typed row.
    Snapshot,
}

/// Bounded activation-capture request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationCapturePlanV1 {
    /// Required model-topology identity.
    pub topology: Digest,
    /// Canonically ordered exact tensor sites.
    pub sites: Vec<TextTensorSiteV1>,
    /// Causal positions to retain.
    pub positions: ActivationCapturePositionsV1,
    /// Per-row retention mode.
    pub retention: ActivationCaptureRetentionV1,
    /// Inclusive total element bound across all retained rows.
    pub maximum_elements: u64,
    /// Inclusive retained-byte bound across all retained records.
    pub maximum_retained_bytes: u64,
}

impl ActivationCapturePlanV1 {
    /// Validates this request against a topology and returns its identity.
    ///
    /// # Errors
    ///
    /// Returns an error for topology mismatch, empty or non-canonical sites,
    /// malformed positions, or inconsistent allocation bounds.
    pub fn digest_for(&self, topology: &TextModelTopologyV1) -> Result<Digest, CoreError> {
        let topology_identity = topology.digest()?;
        if self.topology != topology_identity {
            return Err(CoreError::invalid(
                "activation capture plan",
                "topology identity does not match",
            ));
        }
        if self.sites.is_empty() || self.sites.len() > MAX_ACTIVATION_SITES {
            return Err(CoreError::invalid(
                "activation capture plan",
                "site count is outside the supported bound",
            ));
        }
        validate_canonical_unique(
            &self.sites,
            "activation capture plan",
            "sites must be unique and canonically ordered",
        )?;
        let positions = u64::from(self.positions.position_count()?);
        let elements_per_position = self.sites.iter().try_fold(0_u64, |total, site| {
            total
                .checked_add(u64::from(site.row_elements(topology)?))
                .ok_or_else(|| {
                    CoreError::invalid("activation capture plan", "site width sum overflowed")
                })
        })?;
        let required_elements = positions
            .checked_mul(elements_per_position)
            .ok_or_else(|| {
                CoreError::invalid("activation capture plan", "element bound overflowed")
            })?;
        if self.maximum_elements == 0
            || self.maximum_elements > MAX_ACTIVATION_ELEMENTS
            || self.maximum_elements < required_elements
        {
            return Err(CoreError::invalid(
                "activation capture plan",
                "element bound is zero, excessive, or smaller than selected rows",
            ));
        }
        let minimum_retained_bytes = match self.retention {
            ActivationCaptureRetentionV1::Digest => 0,
            ActivationCaptureRetentionV1::Statistics => u64::try_from(self.sites.len())
                .ok()
                .and_then(|sites| sites.checked_mul(positions))
                .and_then(|rows| rows.checked_mul(4 * 4))
                .ok_or_else(|| {
                    CoreError::invalid(
                        "activation capture plan",
                        "statistics byte bound overflowed",
                    )
                })?,
            ActivationCaptureRetentionV1::Snapshot => {
                required_elements.checked_mul(4).ok_or_else(|| {
                    CoreError::invalid("activation capture plan", "snapshot byte bound overflowed")
                })?
            }
        };
        if self.maximum_retained_bytes > MAX_ACTIVATION_RETAINED_BYTES
            || self.maximum_retained_bytes < minimum_retained_bytes
        {
            return Err(CoreError::invalid(
                "activation capture plan",
                "retained-byte bound is excessive or smaller than selected retention",
            ));
        }
        Digest::of_serializable("activation-capture-plan-v1", self)
    }
}

/// Causal status of telemetry created during target or draft evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivationTelemetryDispositionV1 {
    /// Target verification has not yet resolved the proposal.
    Provisional,
    /// The row belongs to target-admitted causal state.
    Admitted,
    /// The row belongs only to a rejected proposal.
    Rejected,
}

/// Deterministic scalar statistics represented by exact `f32` bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationStatisticsV1 {
    /// Minimum value bits.
    pub minimum_bits: u32,
    /// Maximum value bits.
    pub maximum_bits: u32,
    /// Arithmetic mean bits.
    pub mean_bits: u32,
    /// Euclidean norm bits.
    pub l2_norm_bits: u32,
}

impl ActivationStatisticsV1 {
    /// Computes statistics in stable row order using `f64` accumulators.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty or non-finite row.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "bounded f32 rows are accumulated in f64 and intentionally represented as f32 bits"
    )]
    pub fn from_f32(values: &[f32]) -> Result<Self, CoreError> {
        validate_finite_row(values, "activation statistics")?;
        let mut minimum = values[0];
        let mut maximum = values[0];
        let mut sum = 0.0_f64;
        let mut squared = 0.0_f64;
        for value in values {
            minimum = minimum.min(*value);
            maximum = maximum.max(*value);
            let wide = f64::from(*value);
            sum += wide;
            squared += wide * wide;
        }
        let length = values.len() as f64;
        let mean = (sum / length) as f32;
        let norm = squared.sqrt() as f32;
        if !mean.is_finite() || !norm.is_finite() {
            return Err(CoreError::invalid(
                "activation statistics",
                "statistics overflowed finite f32",
            ));
        }
        Ok(Self {
            minimum_bits: minimum.to_bits(),
            maximum_bits: maximum.to_bits(),
            mean_bits: mean.to_bits(),
            l2_norm_bits: norm.to_bits(),
        })
    }

    fn validate(self) -> Result<(), CoreError> {
        let minimum = f32::from_bits(self.minimum_bits);
        let maximum = f32::from_bits(self.maximum_bits);
        let mean = f32::from_bits(self.mean_bits);
        let norm = f32::from_bits(self.l2_norm_bits);
        if !minimum.is_finite()
            || !maximum.is_finite()
            || !mean.is_finite()
            || !norm.is_finite()
            || minimum > maximum
            || mean < minimum
            || mean > maximum
            || norm < 0.0
        {
            return Err(CoreError::invalid(
                "activation statistics",
                "statistics must be finite and internally consistent",
            ));
        }
        Ok(())
    }
}

/// Retained data for one captured tensor row.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ActivationCapturedDataV1 {
    /// Exact byte identity only.
    Digest {
        /// Canonical typed-row byte identity.
        data: Digest,
        /// Number of elements in the row.
        elements: u32,
    },
    /// Exact byte identity and deterministic scalar statistics.
    Statistics {
        /// Canonical typed-row byte identity.
        data: Digest,
        /// Number of elements in the row.
        elements: u32,
        /// Stable scalar statistics.
        statistics: ActivationStatisticsV1,
    },
    /// Complete finite `f32` row.
    F32Snapshot {
        /// Canonical typed-row byte identity.
        data: Digest,
        /// Exact finite row values.
        values: Vec<f32>,
    },
    /// Complete signed-integer row.
    I32Snapshot {
        /// Canonical typed-row byte identity.
        data: Digest,
        /// Exact row values.
        values: Vec<i32>,
    },
}

impl ActivationCapturedDataV1 {
    /// Constructs retained data from one finite `f32` row.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty, non-finite, or over-bound row.
    pub fn from_f32(
        values: &[f32],
        retention: ActivationCaptureRetentionV1,
    ) -> Result<Self, CoreError> {
        validate_finite_row(values, "activation capture row")?;
        let elements = bounded_element_count(values.len(), "activation capture row")?;
        let data = digest_f32_row("activation-captured-f32-v1", values);
        Ok(match retention {
            ActivationCaptureRetentionV1::Digest => Self::Digest { data, elements },
            ActivationCaptureRetentionV1::Statistics => Self::Statistics {
                data,
                elements,
                statistics: ActivationStatisticsV1::from_f32(values)?,
            },
            ActivationCaptureRetentionV1::Snapshot => Self::F32Snapshot {
                data,
                values: values.to_vec(),
            },
        })
    }

    /// Constructs retained data from one signed-integer row.
    ///
    /// Statistics retention is not defined for selected-expert indices.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty or over-bound row, or for statistics
    /// retention.
    pub fn from_i32(
        values: &[i32],
        retention: ActivationCaptureRetentionV1,
    ) -> Result<Self, CoreError> {
        let elements = bounded_element_count(values.len(), "activation capture row")?;
        let data = digest_i32_row("activation-captured-i32-v1", values);
        match retention {
            ActivationCaptureRetentionV1::Digest => Ok(Self::Digest { data, elements }),
            ActivationCaptureRetentionV1::Snapshot => Ok(Self::I32Snapshot {
                data,
                values: values.to_vec(),
            }),
            ActivationCaptureRetentionV1::Statistics => Err(CoreError::invalid(
                "activation capture row",
                "statistics retention requires f32 elements",
            )),
        }
    }

    /// Returns the exact row identity.
    pub const fn data_identity(&self) -> &Digest {
        match self {
            Self::Digest { data, .. }
            | Self::Statistics { data, .. }
            | Self::F32Snapshot { data, .. }
            | Self::I32Snapshot { data, .. } => data,
        }
    }

    /// Returns the retained element count.
    pub fn elements(&self) -> u32 {
        match self {
            Self::Digest { elements, .. } | Self::Statistics { elements, .. } => *elements,
            Self::F32Snapshot { values, .. } => u32::try_from(values.len()).unwrap_or(u32::MAX),
            Self::I32Snapshot { values, .. } => u32::try_from(values.len()).unwrap_or(u32::MAX),
        }
    }

    fn validate(
        &self,
        element_type: TextTensorElementTypeV1,
        retention: ActivationCaptureRetentionV1,
        expected_elements: u32,
    ) -> Result<u64, CoreError> {
        let retained_bytes = match (self, element_type, retention) {
            (Self::Digest { data, elements }, _, ActivationCaptureRetentionV1::Digest)
                if *elements == expected_elements =>
            {
                let _ = data;
                0
            }
            (
                Self::Statistics {
                    data,
                    elements,
                    statistics,
                },
                TextTensorElementTypeV1::F32,
                ActivationCaptureRetentionV1::Statistics,
            ) if *elements == expected_elements => {
                let _ = data;
                statistics.validate()?;
                4 * 4
            }
            (
                Self::F32Snapshot { data, values },
                TextTensorElementTypeV1::F32,
                ActivationCaptureRetentionV1::Snapshot,
            ) if self.elements() == expected_elements => {
                validate_finite_row(values, "activation capture record")?;
                if *data != digest_f32_row("activation-captured-f32-v1", values) {
                    return Err(CoreError::invalid(
                        "activation capture record",
                        "f32 snapshot identity does not match its values",
                    ));
                }
                u64::from(expected_elements) * 4
            }
            (
                Self::I32Snapshot { data, values },
                TextTensorElementTypeV1::I32,
                ActivationCaptureRetentionV1::Snapshot,
            ) if self.elements() == expected_elements => {
                if *data != digest_i32_row("activation-captured-i32-v1", values) {
                    return Err(CoreError::invalid(
                        "activation capture record",
                        "i32 snapshot identity does not match its values",
                    ));
                }
                u64::from(expected_elements) * 4
            }
            _ => {
                return Err(CoreError::invalid(
                    "activation capture record",
                    "retained data does not match site type, retention, or row width",
                ));
            }
        };
        Ok(retained_bytes)
    }
}

/// One topology-bound captured tensor row.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationCaptureRecordV1 {
    /// Exact capture-plan identity.
    pub plan: Digest,
    /// Exact selected site.
    pub site: TextTensorSiteV1,
    /// Causal position represented by this row.
    pub position: u64,
    /// Causal status of the row.
    pub disposition: ActivationTelemetryDispositionV1,
    /// Retained typed data.
    pub retained: ActivationCapturedDataV1,
}

impl ActivationCaptureRecordV1 {
    /// Validates this row against a capture plan and topology.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible lineage, position, shape, type, or
    /// retained data.
    pub fn digest_for(
        &self,
        plan: &ActivationCapturePlanV1,
        topology: &TextModelTopologyV1,
    ) -> Result<Digest, CoreError> {
        if self.plan != plan.digest_for(topology)? {
            return Err(CoreError::invalid(
                "activation capture record",
                "capture-plan identity does not match",
            ));
        }
        if !plan.sites.contains(&self.site) {
            return Err(CoreError::invalid(
                "activation capture record",
                "site was not selected by the capture plan",
            ));
        }
        if matches!(
            plan.positions,
            ActivationCapturePositionsV1::InclusiveRanges(_)
        ) && !plan.positions.contains_explicit(self.position)
        {
            return Err(CoreError::invalid(
                "activation capture record",
                "position was not selected by the capture plan",
            ));
        }
        let expected = self.site.row_elements(topology)?;
        self.retained
            .validate(self.site.element_type(), plan.retention, expected)?;
        Digest::of_serializable("activation-capture-record-v1", self)
    }
}

/// Aggregate evidence for one completed activation capture.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationCaptureReceiptV1 {
    /// Exact capture-plan identity.
    pub plan: Digest,
    /// Canonically execution-ordered record identities.
    pub records: Vec<Digest>,
    /// Total selected elements.
    pub elements: u64,
    /// Total retained bytes, excluding digest/contract overhead.
    pub retained_bytes: u64,
}

impl ActivationCaptureReceiptV1 {
    /// Validates aggregate bounds and returns the receipt identity.
    ///
    /// # Errors
    ///
    /// Returns an error for empty evidence or plan-bound overrun.
    pub fn digest_for(&self, plan: &ActivationCapturePlanV1) -> Result<Digest, CoreError> {
        if self.plan != Digest::of_serializable("activation-capture-plan-v1", plan)?
            || self.records.is_empty()
            || self.elements == 0
            || self.elements > plan.maximum_elements
            || self.retained_bytes > plan.maximum_retained_bytes
        {
            return Err(CoreError::invalid(
                "activation capture receipt",
                "receipt is empty or exceeds capture bounds",
            ));
        }
        Digest::of_serializable("activation-capture-receipt-v1", self)
    }
}

/// Per-row vector normalization contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivationVectorNormalizationV1 {
    /// Preserve accumulator output without normalization.
    None,
    /// Require every sparse row to have unit Euclidean norm.
    UnitL2,
}

/// Content-free vector construction performed by an accumulator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ActivationVectorOperationV1 {
    /// Arithmetic mean of one caller-ordered group.
    Mean {
        /// Number of input samples.
        count: u32,
    },
    /// Left group mean minus right group mean.
    DifferenceOfMeans {
        /// Number of left-group samples.
        left_count: u32,
        /// Number of right-group samples.
        right_count: u32,
    },
}

/// Mechanical provenance for vector construction.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationVectorProvenanceV1 {
    /// Exact accumulator implementation.
    pub accumulator: Digest,
    /// Construction operation and group counts.
    pub operation: ActivationVectorOperationV1,
    /// Caller-ordered input row or sample identities.
    pub ordered_inputs: Vec<Digest>,
}

impl ActivationVectorProvenanceV1 {
    /// Validates count accounting and input bounds.
    ///
    /// # Errors
    ///
    /// Returns an error for empty, excessive, or inconsistent input lineage.
    pub fn validate(&self) -> Result<(), CoreError> {
        if self.ordered_inputs.is_empty()
            || self.ordered_inputs.len() > MAX_ACTIVATION_VECTOR_INPUTS
        {
            return Err(CoreError::invalid(
                "activation vector provenance",
                "input count is outside the supported bound",
            ));
        }
        let expected = match self.operation {
            ActivationVectorOperationV1::Mean { count } => {
                if count == 0 {
                    return Err(CoreError::invalid(
                        "activation vector provenance",
                        "mean count must be nonzero",
                    ));
                }
                u64::from(count)
            }
            ActivationVectorOperationV1::DifferenceOfMeans {
                left_count,
                right_count,
            } => {
                if left_count == 0 || right_count == 0 {
                    return Err(CoreError::invalid(
                        "activation vector provenance",
                        "both difference groups must be nonempty",
                    ));
                }
                u64::from(left_count) + u64::from(right_count)
            }
        };
        if expected != u64::try_from(self.ordered_inputs.len()).unwrap_or(u64::MAX) {
            return Err(CoreError::invalid(
                "activation vector provenance",
                "group counts do not match ordered input identities",
            ));
        }
        Ok(())
    }
}

/// One canonical sparse vector row.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationVectorRowV1 {
    /// Zero-based model layer.
    pub layer: u32,
    /// Exact finite row values.
    pub values: Vec<f32>,
}

/// Topology-bound sparse activation vectors.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationVectorBankV1 {
    /// Required model-topology identity.
    pub topology: Digest,
    /// Compatible mutable site family.
    pub site_family: TextTensorSiteFamilyV1,
    /// Elements in every sparse row.
    pub row_elements: u32,
    /// Required per-row normalization.
    pub normalization: ActivationVectorNormalizationV1,
    /// Canonically layer-ordered sparse rows.
    pub rows: Vec<ActivationVectorRowV1>,
    /// Exact canonical layer and little-endian `f32` byte identity.
    pub data: Digest,
    /// Content-free construction lineage.
    pub provenance: ActivationVectorProvenanceV1,
}

impl ActivationVectorBankV1 {
    /// Constructs a bank and binds its exact sparse `f32` bytes.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed topology, unsupported site family,
    /// non-canonical rows, invalid values, normalization, or bounds.
    pub fn new(
        topology: &TextModelTopologyV1,
        site_family: TextTensorSiteFamilyV1,
        row_elements: u32,
        normalization: ActivationVectorNormalizationV1,
        rows: Vec<ActivationVectorRowV1>,
        provenance: ActivationVectorProvenanceV1,
    ) -> Result<Self, CoreError> {
        let mut bank = Self {
            topology: topology.digest()?,
            site_family,
            row_elements,
            normalization,
            rows,
            data: Digest::of_bytes("activation-vector-bank-placeholder-v1", b""),
            provenance,
        };
        bank.data = bank.canonical_data_identity()?;
        bank.digest_for(topology)?;
        Ok(bank)
    }

    /// Validates this bank against a topology and returns its identity.
    ///
    /// # Errors
    ///
    /// Returns an error for topology mismatch, unsupported family, malformed
    /// sparse rows, non-finite values, incorrect normalization, or byte
    /// identity mismatch.
    pub fn digest_for(&self, topology: &TextModelTopologyV1) -> Result<Digest, CoreError> {
        if self.topology != topology.digest()? {
            return Err(CoreError::invalid(
                "activation vector bank",
                "topology identity does not match",
            ));
        }
        if !matches!(
            self.site_family,
            TextTensorSiteFamilyV1::LayerOutput | TextTensorSiteFamilyV1::RouterLogits
        ) {
            return Err(CoreError::invalid(
                "activation vector bank",
                "only residual outputs and router logits are mutable site families",
            ));
        }
        let expected_width = match self.site_family {
            TextTensorSiteFamilyV1::LayerOutput => topology.embedding_width,
            TextTensorSiteFamilyV1::RouterLogits => topology.experts.ok_or_else(|| {
                CoreError::invalid(
                    "activation vector bank",
                    "router vectors require reported expert topology",
                )
            })?,
            _ => unreachable!("family checked above"),
        };
        if self.row_elements == 0
            || self.row_elements != expected_width
            || self.rows.is_empty()
            || self.rows.len() > MAX_ACTIVATION_VECTOR_ROWS
        {
            return Err(CoreError::invalid(
                "activation vector bank",
                "row width or sparse row count is incompatible with topology",
            ));
        }
        self.provenance.validate()?;
        let mut prior_layer = None;
        let mut total_elements = 0_u64;
        for row in &self.rows {
            if row.layer >= topology.layers
                || prior_layer.is_some_and(|prior| row.layer <= prior)
                || row.values.len() != usize::try_from(self.row_elements).unwrap_or(usize::MAX)
            {
                return Err(CoreError::invalid(
                    "activation vector bank",
                    "rows must be width-matched and strictly layer-ordered",
                ));
            }
            validate_finite_row(&row.values, "activation vector bank")?;
            if self.normalization == ActivationVectorNormalizationV1::UnitL2 {
                validate_unit_row(&row.values)?;
            }
            prior_layer = Some(row.layer);
            total_elements = total_elements
                .checked_add(u64::from(self.row_elements))
                .ok_or_else(|| {
                    CoreError::invalid("activation vector bank", "element count overflowed")
                })?;
        }
        if total_elements > MAX_ACTIVATION_ELEMENTS {
            return Err(CoreError::invalid(
                "activation vector bank",
                "vector elements exceed the supported bound",
            ));
        }
        if self.data != self.canonical_data_identity()? {
            return Err(CoreError::invalid(
                "activation vector bank",
                "data identity does not match canonical sparse row bytes",
            ));
        }
        Digest::of_serializable("activation-vector-bank-v1", self)
    }

    /// Returns one sparse row by exact layer.
    pub fn row(&self, layer: u32) -> Option<&[f32]> {
        self.rows
            .binary_search_by_key(&layer, |row| row.layer)
            .ok()
            .map(|index| self.rows[index].values.as_slice())
    }

    fn canonical_data_identity(&self) -> Result<Digest, CoreError> {
        let row_count = u32::try_from(self.rows.len())
            .map_err(|_| CoreError::invalid("activation vector bank", "row count exceeds u32"))?;
        let capacity = self
            .rows
            .len()
            .checked_mul(
                usize::try_from(self.row_elements)
                    .unwrap_or(usize::MAX)
                    .saturating_mul(4)
                    .saturating_add(4),
            )
            .and_then(|bytes| bytes.checked_add(8))
            .ok_or_else(|| {
                CoreError::invalid("activation vector bank", "canonical byte size overflowed")
            })?;
        let mut bytes = Vec::with_capacity(capacity);
        bytes.extend_from_slice(&self.row_elements.to_le_bytes());
        bytes.extend_from_slice(&row_count.to_le_bytes());
        for row in &self.rows {
            bytes.extend_from_slice(&row.layer.to_le_bytes());
            for value in &row.values {
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        Ok(Digest::of_bytes("activation-vector-f32-le-v1", &bytes))
    }
}

/// Decode phase at which one activation operation executes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivationPhaseV1 {
    /// Prompt admission in the selected target or draft context.
    Prefill,
    /// Ordinary target generation.
    Generation,
    /// Target verification of speculative proposals.
    Verification,
    /// Draft-model proposal evaluation.
    Draft,
}

/// Causal positions selected by one activation operation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "ranges", rename_all = "snake_case")]
pub enum ActivationPositionScopeV1 {
    /// Every position evaluated in the declared phase.
    All,
    /// Canonically ordered, disjoint inclusive causal ranges.
    InclusiveRanges(Vec<CausalPositionRangeV1>),
}

impl ActivationPositionScopeV1 {
    /// Validates range ordering and bounds.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty, reversed, overlapping, or excessive
    /// explicit range set.
    pub fn validate(&self) -> Result<(), CoreError> {
        if let Self::InclusiveRanges(ranges) = self {
            validate_ranges(ranges, "activation position scope")?;
        }
        Ok(())
    }

    /// Returns whether this scope contains a causal position.
    pub fn contains(&self, position: u64) -> bool {
        match self {
            Self::All => true,
            Self::InclusiveRanges(ranges) => ranges.iter().any(|range| range.contains(position)),
        }
    }
}

/// Built-in transactional tensor operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivationOperatorV1 {
    /// `x' = x + alpha * v`.
    ScaledAdd,
    /// `x' = x - alpha * dot(x, v) * v`.
    ScaledProjectionRemoval,
}

/// One ordered, position-scoped activation operation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationOperationV1 {
    /// Canonically ordered exact tensor sites.
    pub sites: Vec<TextTensorSiteV1>,
    /// Decode phase.
    pub phase: ActivationPhaseV1,
    /// Selected causal positions.
    pub positions: ActivationPositionScopeV1,
    /// Exact vector-bank identity.
    pub vector_bank: Digest,
    /// Built-in operation.
    pub operator: ActivationOperatorV1,
    /// Exact IEEE-754 scale bits.
    pub scale_bits: u32,
}

impl ActivationOperationV1 {
    /// Returns the finite nonzero operation scale.
    ///
    /// # Errors
    ///
    /// Returns an error when the bound bits represent zero, NaN, or infinity.
    pub fn scale(&self) -> Result<f32, CoreError> {
        let scale = f32::from_bits(self.scale_bits);
        if !scale.is_finite() || scale == 0.0 {
            return Err(CoreError::invalid(
                "activation operation",
                "scale must be finite and nonzero",
            ));
        }
        Ok(scale)
    }

    fn validate_for(&self, topology: &TextModelTopologyV1) -> Result<(), CoreError> {
        if self.sites.is_empty() || self.sites.len() > MAX_ACTIVATION_SITES {
            return Err(CoreError::invalid(
                "activation operation",
                "site count is outside the supported bound",
            ));
        }
        validate_canonical_unique(
            &self.sites,
            "activation operation",
            "sites must be unique and canonically ordered",
        )?;
        self.positions.validate()?;
        self.scale()?;
        for site in &self.sites {
            site.validate_for(topology)?;
            if !matches!(
                site.family(),
                TextTensorSiteFamilyV1::LayerOutput | TextTensorSiteFamilyV1::RouterLogits
            ) || site.element_type() != TextTensorElementTypeV1::F32
            {
                return Err(CoreError::invalid(
                    "activation operation",
                    "selected site is observable but not a built-in mutable f32 site",
                ));
            }
        }
        let families = self
            .sites
            .iter()
            .map(TextTensorSiteV1::family)
            .collect::<BTreeSet<_>>();
        if families.len() != 1 {
            return Err(CoreError::invalid(
                "activation operation",
                "one operation cannot cross tensor-site families",
            ));
        }
        Ok(())
    }
}

/// Ordered activation operations and observation requests.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationProgramV1 {
    /// Required model-topology identity.
    pub topology: Digest,
    /// Ordered tensor operations.
    pub operations: Vec<ActivationOperationV1>,
    /// Canonically ordered capture-plan identities requested for observation.
    #[serde(default)]
    pub observations: Vec<Digest>,
}

impl ActivationProgramV1 {
    /// Validates the program against a topology and returns its identity.
    ///
    /// Vector-bank references are resolved separately by an execution runtime.
    ///
    /// # Errors
    ///
    /// Returns an error for topology mismatch, empty or excessive operations,
    /// malformed sites, positions, scales, or observation identities.
    pub fn digest_for(&self, topology: &TextModelTopologyV1) -> Result<Digest, CoreError> {
        if self.topology != topology.digest()? {
            return Err(CoreError::invalid(
                "activation program",
                "topology identity does not match",
            ));
        }
        if self.operations.len() > MAX_ACTIVATION_OPERATIONS
            || (self.operations.is_empty() && self.observations.is_empty())
        {
            return Err(CoreError::invalid(
                "activation program",
                "the program is empty or has too many operations",
            ));
        }
        if self.observations.len() > MAX_ACTIVATION_OBSERVATIONS {
            return Err(CoreError::invalid(
                "activation program",
                "observation count exceeds the supported bound",
            ));
        }
        validate_canonical_unique(
            &self.observations,
            "activation program",
            "observation identities must be unique and canonically ordered",
        )?;
        for operation in &self.operations {
            operation.validate_for(topology)?;
        }
        Digest::of_serializable("activation-program-v1", self)
    }
}

/// Evidence for one successful copy-transform-write transaction.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationInvocationRowV1 {
    /// Zero-based logical row within the submitted decode batch.
    pub batch_index: u32,
    /// Zero-based operation indices that ran, in declared order.
    pub operations: Vec<u32>,
    /// Canonically ordered native sequence IDs attached to this row.
    pub sequence_ids: Vec<u32>,
    /// Causal position.
    pub position: u64,
    /// Target-authoritative status of this evaluated row.
    pub disposition: ActivationTelemetryDispositionV1,
}

impl ActivationInvocationRowV1 {
    fn validate(&self) -> Result<(), CoreError> {
        if self.operations.is_empty()
            || self
                .operations
                .windows(2)
                .any(|window| window[0] >= window[1])
            || self.sequence_ids.is_empty()
            || self.sequence_ids.len() > MAX_ACTIVATION_ROW_SEQUENCES
            || self
                .sequence_ids
                .windows(2)
                .any(|window| window[0] >= window[1])
        {
            return Err(CoreError::invalid(
                "activation invocation row",
                "operation or sequence identifiers are empty, excessive, or non-canonical",
            ));
        }
        Ok(())
    }
}

/// Evidence for one successful native tensor write-back.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationInvocationReceiptV1 {
    /// Exact tensor site.
    pub site: TextTensorSiteV1,
    /// Decode phase.
    pub phase: ActivationPhaseV1,
    /// Changed logical rows in ascending batch order.
    pub rows: Vec<ActivationInvocationRowV1>,
    /// Number of finite `f32` elements copied for the complete tensor.
    pub elements: u32,
    /// Complete tensor identity before operations.
    pub input: Digest,
    /// Complete tensor identity committed after every operation succeeded.
    pub output: Digest,
}

impl ActivationInvocationReceiptV1 {
    /// Returns the stable transaction identity.
    ///
    /// # Errors
    ///
    /// Returns an error for empty or non-canonical row accounting, unchanged
    /// tensor identity, or a zero/excessive element count.
    pub fn digest(&self) -> Result<Digest, CoreError> {
        if self.rows.is_empty()
            || self.rows.len() > MAX_ACTIVATION_TRANSACTION_ROWS
            || self
                .rows
                .windows(2)
                .any(|window| window[0].batch_index >= window[1].batch_index)
            || self.elements == 0
            || u64::from(self.elements) > MAX_ACTIVATION_ELEMENTS
            || self.input == self.output
        {
            return Err(CoreError::invalid(
                "activation invocation receipt",
                "row accounting, element count, or tensor identities are invalid",
            ));
        }
        for row in &self.rows {
            row.validate()?;
        }
        Digest::of_serializable("activation-invocation-receipt-v1", self)
    }
}

/// Aggregate successful accounting for one activation program runtime.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationProgramReceiptV1 {
    /// Exact program identity.
    pub program: Digest,
    /// Required model-topology identity.
    pub topology: Digest,
    /// Ordered successful transaction identities.
    pub invocations: Vec<Digest>,
    /// Total elements copied into Rust-owned transactional storage.
    pub elements_copied: u64,
    /// Number of complete native write-backs.
    pub write_backs: u64,
}

impl ActivationProgramReceiptV1 {
    /// Validates aggregate accounting and returns its identity.
    ///
    /// # Errors
    ///
    /// Returns an error for excessive evidence or mismatched transaction and
    /// write-back counts.
    pub fn digest_for(&self, program: &ActivationProgramV1) -> Result<Digest, CoreError> {
        if self.program != Digest::of_serializable("activation-program-v1", program)?
            || self.topology != program.topology
            || self.invocations.len() > MAX_ACTIVATION_INVOCATION_RECEIPTS
            || self.write_backs != u64::try_from(self.invocations.len()).unwrap_or(u64::MAX)
            || self.elements_copied > MAX_ACTIVATION_ELEMENTS.saturating_mul(self.write_backs)
        {
            return Err(CoreError::invalid(
                "activation program receipt",
                "program lineage or transaction accounting is invalid",
            ));
        }
        Digest::of_serializable("activation-program-receipt-v1", self)
    }
}

pub(crate) fn digest_f32_row(domain: &str, values: &[f32]) -> Digest {
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(4));
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    Digest::of_bytes(domain, &bytes)
}

/// Returns the canonical identity of one bounded finite activation `f32` row.
///
/// # Errors
///
/// Returns an error for an empty, excessive, or non-finite row.
pub fn activation_f32_row_identity(values: &[f32]) -> Result<Digest, CoreError> {
    validate_finite_row(values, "activation f32 row identity")?;
    Ok(digest_f32_row("activation-runtime-f32-v1", values))
}

fn digest_i32_row(domain: &str, values: &[i32]) -> Digest {
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(4));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Digest::of_bytes(domain, &bytes)
}

fn validate_finite_row(values: &[f32], field: &'static str) -> Result<(), CoreError> {
    if values.is_empty()
        || u64::try_from(values.len()).unwrap_or(u64::MAX) > MAX_ACTIVATION_ELEMENTS
        || values.iter().any(|value| !value.is_finite())
    {
        return Err(CoreError::invalid(
            field,
            "row must be nonempty, bounded, and finite",
        ));
    }
    Ok(())
}

fn bounded_element_count(values: usize, field: &'static str) -> Result<u32, CoreError> {
    let elements = u32::try_from(values)
        .map_err(|_| CoreError::invalid(field, "element count exceeds u32"))?;
    if elements == 0 || u64::from(elements) > MAX_ACTIVATION_ELEMENTS {
        return Err(CoreError::invalid(
            field,
            "element count is outside the supported bound",
        ));
    }
    Ok(elements)
}

fn validate_unit_row(values: &[f32]) -> Result<(), CoreError> {
    let squared = values.iter().try_fold(0.0_f64, |total, value| {
        let wide = f64::from(*value);
        let next = total + wide * wide;
        next.is_finite()
            .then_some(next)
            .ok_or_else(|| CoreError::invalid("activation vector bank", "row norm overflowed"))
    })?;
    let norm = squared.sqrt();
    if (norm - 1.0).abs() > 1.0e-5 {
        return Err(CoreError::invalid(
            "activation vector bank",
            "unit-normalized row is outside tolerance",
        ));
    }
    Ok(())
}

fn validate_ranges(ranges: &[CausalPositionRangeV1], field: &'static str) -> Result<(), CoreError> {
    if ranges.is_empty() || ranges.len() > MAX_ACTIVATION_POSITION_RANGES {
        return Err(CoreError::invalid(
            field,
            "range count is outside the supported bound",
        ));
    }
    let mut prior_end = None;
    for range in ranges {
        range.len()?;
        if prior_end.is_some_and(|end| range.start <= end) {
            return Err(CoreError::invalid(
                field,
                "ranges must be strictly ordered and disjoint",
            ));
        }
        prior_end = Some(range.end);
    }
    Ok(())
}

fn validate_canonical_unique<T: Ord>(
    values: &[T],
    field: &'static str,
    reason: &'static str,
) -> Result<(), CoreError> {
    if values.windows(2).any(|window| window[0] >= window[1]) {
        return Err(CoreError::invalid(field, reason));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn topology() -> TextModelTopologyV1 {
        TextModelTopologyV1 {
            model: Digest::of_bytes("test-model", b"one"),
            backend: Digest::of_bytes("test-backend", b"one"),
            architecture_implementation: Digest::of_bytes("test-architecture", b"one"),
            layers: 4,
            embedding_width: 3,
            experts: Some(4),
            experts_used: Some(2),
            nextn_layers: 1,
            supported_speculation: vec![
                TextSpeculativeMechanismV1::Mtp,
                TextSpeculativeMechanismV1::Eagle3,
            ],
        }
    }

    fn provenance() -> ActivationVectorProvenanceV1 {
        ActivationVectorProvenanceV1 {
            accumulator: Digest::of_bytes("test-accumulator", b"one"),
            operation: ActivationVectorOperationV1::Mean { count: 1 },
            ordered_inputs: vec![Digest::of_bytes("test-input", b"one")],
        }
    }

    #[test]
    fn topology_rejects_inconsistent_mechanisms_and_experts() {
        let mut value = topology();
        assert!(value.digest().is_ok());
        value.supported_speculation.reverse();
        assert!(value.digest().is_err());
        let mut value = topology();
        value.experts_used = Some(5);
        assert!(value.digest().is_err());
        let mut value = topology();
        value.nextn_layers = 0;
        assert!(value.digest().is_err());
    }

    #[test]
    fn sites_are_topology_checked_and_content_bound() {
        let topology = topology();
        let layer = TextTensorSiteV1::LayerOutput { layer: 3 };
        assert_eq!(layer.row_elements(&topology).unwrap(), 3);
        assert!(layer.digest_for(&topology).is_ok());
        assert!(
            TextTensorSiteV1::LayerOutput { layer: 4 }
                .validate_for(&topology)
                .is_err()
        );
        assert!(
            TextTensorSiteV1::Named {
                name: "bad\0name".to_owned(),
                selector: Digest::of_bytes("selector", b"one"),
                element_type: TextTensorElementTypeV1::F32,
                row_elements: 3,
            }
            .validate_for(&topology)
            .is_err()
        );
    }

    #[test]
    fn capture_rejects_duplicate_unordered_and_underbounded_selections() {
        let topology = topology();
        let site_a = TextTensorSiteV1::LayerOutput { layer: 0 };
        let site_b = TextTensorSiteV1::LayerOutput { layer: 1 };
        let valid = ActivationCapturePlanV1 {
            topology: topology.digest().unwrap(),
            sites: vec![site_a.clone(), site_b.clone()],
            positions: ActivationCapturePositionsV1::InclusiveRanges(vec![CausalPositionRangeV1 {
                start: 2,
                end: 3,
            }]),
            retention: ActivationCaptureRetentionV1::Snapshot,
            maximum_elements: 6 * 2,
            maximum_retained_bytes: 6 * 2 * 4,
        };
        assert!(valid.digest_for(&topology).is_ok());

        let mut duplicate = valid.clone();
        duplicate.sites = vec![site_a.clone(), site_a];
        assert!(duplicate.digest_for(&topology).is_err());

        let mut unordered = valid.clone();
        unordered.sites = vec![site_b, TextTensorSiteV1::LayerOutput { layer: 0 }];
        assert!(unordered.digest_for(&topology).is_err());

        let mut underbounded = valid;
        underbounded.maximum_elements = 1;
        assert!(underbounded.digest_for(&topology).is_err());
    }

    #[test]
    fn capture_rows_preserve_exact_bits_and_retention_contract() {
        let values = [0.0_f32, -0.0, 2.0];
        let snapshot =
            ActivationCapturedDataV1::from_f32(&values, ActivationCaptureRetentionV1::Snapshot)
                .unwrap();
        if let ActivationCapturedDataV1::F32Snapshot {
            data,
            values: retained,
        } = snapshot
        {
            assert_ne!(retained[0].to_bits(), retained[1].to_bits());
            assert_eq!(data, digest_f32_row("activation-captured-f32-v1", &values));
        } else {
            panic!("expected f32 snapshot");
        }
        assert!(
            ActivationCapturedDataV1::from_f32(&[f32::NAN], ActivationCaptureRetentionV1::Digest)
                .is_err()
        );
        assert!(
            ActivationCapturedDataV1::from_i32(&[1, 2], ActivationCaptureRetentionV1::Statistics)
                .is_err()
        );
    }

    #[test]
    fn vector_bank_binds_sparse_rows_and_normalization() {
        let topology = topology();
        let bank = ActivationVectorBankV1::new(
            &topology,
            TextTensorSiteFamilyV1::LayerOutput,
            3,
            ActivationVectorNormalizationV1::UnitL2,
            vec![
                ActivationVectorRowV1 {
                    layer: 0,
                    values: vec![1.0, 0.0, 0.0],
                },
                ActivationVectorRowV1 {
                    layer: 2,
                    values: vec![0.0, 1.0, 0.0],
                },
            ],
            provenance(),
        )
        .unwrap();
        assert_eq!(bank.row(2), Some(&[0.0, 1.0, 0.0][..]));
        assert!(bank.digest_for(&topology).is_ok());

        let mut malformed = bank;
        malformed.rows[0].values[0] = 0.5;
        assert!(malformed.digest_for(&topology).is_err());
    }

    #[test]
    fn program_rejects_observable_only_mutation_sites() {
        let topology = topology();
        let bank = Digest::of_bytes("test-bank", b"one");
        let valid = ActivationProgramV1 {
            topology: topology.digest().unwrap(),
            operations: vec![ActivationOperationV1 {
                sites: vec![TextTensorSiteV1::LayerOutput { layer: 0 }],
                phase: ActivationPhaseV1::Generation,
                positions: ActivationPositionScopeV1::All,
                vector_bank: bank.clone(),
                operator: ActivationOperatorV1::ScaledAdd,
                scale_bits: 0.5_f32.to_bits(),
            }],
            observations: Vec::new(),
        };
        assert!(valid.digest_for(&topology).is_ok());

        let mut invalid = valid;
        invalid.operations[0].sites = vec![TextTensorSiteV1::RouterProbabilities {
            layer: 0,
            selector: Digest::of_bytes("selector", b"one"),
        }];
        assert!(invalid.digest_for(&topology).is_err());
    }

    #[test]
    fn invocation_rows_bind_provisional_and_final_causal_status() {
        let provisional = ActivationInvocationReceiptV1 {
            site: TextTensorSiteV1::LayerOutput { layer: 1 },
            phase: ActivationPhaseV1::Verification,
            rows: vec![ActivationInvocationRowV1 {
                batch_index: 0,
                operations: vec![0],
                sequence_ids: vec![0],
                position: 7,
                disposition: ActivationTelemetryDispositionV1::Provisional,
            }],
            elements: 3,
            input: Digest::of_bytes("test-activation-input", b"one"),
            output: Digest::of_bytes("test-activation-output", b"one"),
        };
        let mut admitted = provisional.clone();
        admitted.rows[0].disposition = ActivationTelemetryDispositionV1::Admitted;
        assert_ne!(provisional.digest().unwrap(), admitted.digest().unwrap());
    }
}

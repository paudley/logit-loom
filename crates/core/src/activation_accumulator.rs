// SPDX-License-Identifier: MIT OR Apache-2.0

//! Deterministic, content-free activation-vector accumulation.

use serde::{Deserialize, Serialize};

use crate::{
    ActivationVectorBankV1, ActivationVectorNormalizationV1, ActivationVectorOperationV1,
    ActivationVectorProvenanceV1, ActivationVectorRowV1, CoreError, Digest,
    MAX_ACTIVATION_ELEMENTS, MAX_ACTIVATION_VECTOR_INPUTS, MAX_ACTIVATION_VECTOR_ROWS,
    TextModelTopologyV1, TextTensorSiteFamilyV1,
};

/// One caller-ordered, content-bound sparse activation sample.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationVectorSampleV1 {
    /// Exact identity of the source capture or caller-defined sample artifact.
    pub source: Digest,
    /// Canonically layer-ordered finite rows.
    pub rows: Vec<ActivationVectorRowV1>,
}

/// Deterministic accumulator for sparse activation samples.
///
/// The accumulator assigns no semantic labels and performs no dataset
/// selection. It consumes caller-ordered numeric rows and records that exact
/// order in the resulting vector-bank provenance.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ActivationVectorAccumulatorV1;

impl ActivationVectorAccumulatorV1 {
    /// Returns the exact arithmetic implementation identity.
    pub fn implementation() -> Digest {
        Digest::of_bytes(
            "activation-vector-accumulator-implementation-v1",
            b"ordered-f64-sum;f32-output;unit-l2-f64",
        )
    }

    /// Computes an ordered arithmetic mean.
    ///
    /// # Errors
    ///
    /// Returns an error for empty or excessive input, incompatible sparse
    /// shapes, non-finite rows, arithmetic overflow, or an undefined requested
    /// normalization.
    pub fn mean(
        topology: &TextModelTopologyV1,
        site_family: TextTensorSiteFamilyV1,
        samples: &[ActivationVectorSampleV1],
        normalization: ActivationVectorNormalizationV1,
    ) -> Result<ActivationVectorBankV1, CoreError> {
        validate_sample_count(samples.len(), "activation mean")?;
        let rows = group_mean(topology, site_family, samples)?;
        let count = u32::try_from(samples.len())
            .map_err(|_| CoreError::invalid("activation mean", "sample count exceeds u32"))?;
        ActivationVectorBankV1::new(
            topology,
            site_family,
            expected_width(topology, site_family)?,
            normalization,
            normalize_rows(rows, normalization)?,
            ActivationVectorProvenanceV1 {
                accumulator: Self::implementation(),
                operation: ActivationVectorOperationV1::Mean { count },
                ordered_inputs: samples.iter().map(|sample| sample.source.clone()).collect(),
            },
        )
    }

    /// Computes the ordered left-group mean minus the ordered right-group
    /// mean.
    ///
    /// # Errors
    ///
    /// Returns an error for empty or excessive groups, incompatible sparse
    /// shapes, non-finite rows, arithmetic overflow, or an undefined requested
    /// normalization.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "finite f64 accumulator differences are intentionally represented as f32 rows"
    )]
    pub fn difference_of_means(
        topology: &TextModelTopologyV1,
        site_family: TextTensorSiteFamilyV1,
        left: &[ActivationVectorSampleV1],
        right: &[ActivationVectorSampleV1],
        normalization: ActivationVectorNormalizationV1,
    ) -> Result<ActivationVectorBankV1, CoreError> {
        validate_sample_count(left.len(), "activation difference left group")?;
        validate_sample_count(right.len(), "activation difference right group")?;
        let total = left.len().checked_add(right.len()).ok_or_else(|| {
            CoreError::invalid("activation difference", "sample count overflowed")
        })?;
        if total > MAX_ACTIVATION_VECTOR_INPUTS {
            return Err(CoreError::invalid(
                "activation difference",
                "combined sample count exceeds the supported bound",
            ));
        }
        let left_rows = group_mean(topology, site_family, left)?;
        let right_rows = group_mean(topology, site_family, right)?;
        if left_rows
            .iter()
            .map(|row| row.layer)
            .ne(right_rows.iter().map(|row| row.layer))
        {
            return Err(CoreError::invalid(
                "activation difference",
                "left and right sparse layer sets differ",
            ));
        }
        let mut rows = Vec::with_capacity(left_rows.len());
        for (left_row, right_row) in left_rows.into_iter().zip(right_rows) {
            let values = left_row
                .values
                .into_iter()
                .zip(right_row.values)
                .map(|(left_value, right_value)| {
                    let difference = f64::from(left_value) - f64::from(right_value);
                    let difference = difference as f32;
                    difference.is_finite().then_some(difference).ok_or_else(|| {
                        CoreError::invalid(
                            "activation difference",
                            "difference overflowed finite f32",
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            rows.push(ActivationVectorRowV1 {
                layer: left_row.layer,
                values,
            });
        }
        let left_count = u32::try_from(left.len()).map_err(|_| {
            CoreError::invalid("activation difference", "left sample count exceeds u32")
        })?;
        let right_count = u32::try_from(right.len()).map_err(|_| {
            CoreError::invalid("activation difference", "right sample count exceeds u32")
        })?;
        ActivationVectorBankV1::new(
            topology,
            site_family,
            expected_width(topology, site_family)?,
            normalization,
            normalize_rows(rows, normalization)?,
            ActivationVectorProvenanceV1 {
                accumulator: Self::implementation(),
                operation: ActivationVectorOperationV1::DifferenceOfMeans {
                    left_count,
                    right_count,
                },
                ordered_inputs: left
                    .iter()
                    .chain(right)
                    .map(|sample| sample.source.clone())
                    .collect(),
            },
        )
    }
}

fn validate_sample_count(count: usize, field: &'static str) -> Result<(), CoreError> {
    if count == 0 || count > MAX_ACTIVATION_VECTOR_INPUTS {
        return Err(CoreError::invalid(
            field,
            "sample count is outside the supported bound",
        ));
    }
    Ok(())
}

fn expected_width(
    topology: &TextModelTopologyV1,
    family: TextTensorSiteFamilyV1,
) -> Result<u32, CoreError> {
    topology.digest()?;
    match family {
        TextTensorSiteFamilyV1::LayerOutput => Ok(topology.embedding_width),
        TextTensorSiteFamilyV1::RouterLogits => topology.experts.ok_or_else(|| {
            CoreError::invalid(
                "activation accumulator",
                "router vectors require reported expert topology",
            )
        }),
        _ => Err(CoreError::invalid(
            "activation accumulator",
            "only residual outputs and router logits support vector accumulation",
        )),
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "bounded ordered samples are accumulated in f64 and intentionally represented as f32 rows"
)]
fn group_mean(
    topology: &TextModelTopologyV1,
    family: TextTensorSiteFamilyV1,
    samples: &[ActivationVectorSampleV1],
) -> Result<Vec<ActivationVectorRowV1>, CoreError> {
    validate_sample_count(samples.len(), "activation accumulator")?;
    let width = usize::try_from(expected_width(topology, family)?)
        .map_err(|_| CoreError::invalid("activation accumulator", "row width exceeds usize"))?;
    let first = samples
        .first()
        .ok_or_else(|| CoreError::invalid("activation accumulator", "samples are empty"))?;
    validate_sample(topology, width, first)?;
    let layers = first.rows.iter().map(|row| row.layer).collect::<Vec<_>>();
    let total_elements = layers
        .len()
        .checked_mul(width)
        .ok_or_else(|| CoreError::invalid("activation accumulator", "element count overflowed"))?;
    if u64::try_from(total_elements).unwrap_or(u64::MAX) > MAX_ACTIVATION_ELEMENTS {
        return Err(CoreError::invalid(
            "activation accumulator",
            "sparse rows exceed the supported element bound",
        ));
    }
    let mut sums = vec![0.0_f64; total_elements];
    for sample in samples {
        validate_sample(topology, width, sample)?;
        if sample
            .rows
            .iter()
            .map(|row| row.layer)
            .ne(layers.iter().copied())
        {
            return Err(CoreError::invalid(
                "activation accumulator",
                "every sample must contain the same sparse layer set",
            ));
        }
        for (sum, value) in sums
            .iter_mut()
            .zip(sample.rows.iter().flat_map(|row| row.values.iter()))
        {
            *sum += f64::from(*value);
            if !sum.is_finite() {
                return Err(CoreError::invalid(
                    "activation accumulator",
                    "ordered sum overflowed finite f64",
                ));
            }
        }
    }
    let divisor = samples.len() as f64;
    layers
        .into_iter()
        .enumerate()
        .map(|(row_index, layer)| {
            let start = row_index.checked_mul(width).ok_or_else(|| {
                CoreError::invalid("activation accumulator", "row offset overflowed")
            })?;
            let end = start.checked_add(width).ok_or_else(|| {
                CoreError::invalid("activation accumulator", "row end overflowed")
            })?;
            let values = sums[start..end]
                .iter()
                .map(|sum| {
                    let mean = (*sum / divisor) as f32;
                    mean.is_finite().then_some(mean).ok_or_else(|| {
                        CoreError::invalid("activation accumulator", "mean overflowed finite f32")
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(ActivationVectorRowV1 { layer, values })
        })
        .collect()
}

fn validate_sample(
    topology: &TextModelTopologyV1,
    width: usize,
    sample: &ActivationVectorSampleV1,
) -> Result<(), CoreError> {
    if sample.rows.is_empty() || sample.rows.len() > MAX_ACTIVATION_VECTOR_ROWS {
        return Err(CoreError::invalid(
            "activation sample",
            "sparse row count is outside the supported bound",
        ));
    }
    let mut prior = None;
    for row in &sample.rows {
        if row.layer >= topology.layers
            || prior.is_some_and(|layer| row.layer <= layer)
            || row.values.len() != width
            || row.values.iter().any(|value| !value.is_finite())
        {
            return Err(CoreError::invalid(
                "activation sample",
                "rows must be finite, width-matched, and strictly layer-ordered",
            ));
        }
        prior = Some(row.layer);
    }
    Ok(())
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "finite f64-normalized values are intentionally represented as f32 rows"
)]
fn normalize_rows(
    mut rows: Vec<ActivationVectorRowV1>,
    normalization: ActivationVectorNormalizationV1,
) -> Result<Vec<ActivationVectorRowV1>, CoreError> {
    if normalization == ActivationVectorNormalizationV1::None {
        return Ok(rows);
    }
    for row in &mut rows {
        let squared = row.values.iter().try_fold(0.0_f64, |total, value| {
            let value = f64::from(*value);
            let next = total + value * value;
            next.is_finite().then_some(next).ok_or_else(|| {
                CoreError::invalid("activation normalization", "row norm overflowed")
            })
        })?;
        let norm = squared.sqrt();
        if !norm.is_finite() || norm <= f64::EPSILON {
            return Err(CoreError::invalid(
                "activation normalization",
                "cannot unit-normalize a zero or non-finite row",
            ));
        }
        for value in &mut row.values {
            let normalized = (f64::from(*value) / norm) as f32;
            if !normalized.is_finite() {
                return Err(CoreError::invalid(
                    "activation normalization",
                    "normalized value is not finite",
                ));
            }
            *value = normalized;
        }
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TextSpeculativeMechanismV1;

    fn topology() -> TextModelTopologyV1 {
        TextModelTopologyV1 {
            model: Digest::of_bytes("test-model", b"one"),
            backend: Digest::of_bytes("test-backend", b"one"),
            architecture_implementation: Digest::of_bytes("test-architecture", b"one"),
            layers: 3,
            embedding_width: 2,
            experts: None,
            experts_used: None,
            nextn_layers: 1,
            supported_speculation: vec![TextSpeculativeMechanismV1::Mtp],
        }
    }

    fn sample(source: &[u8], first: [f32; 2], second: [f32; 2]) -> ActivationVectorSampleV1 {
        ActivationVectorSampleV1 {
            source: Digest::of_bytes("test-sample", source),
            rows: vec![
                ActivationVectorRowV1 {
                    layer: 0,
                    values: first.to_vec(),
                },
                ActivationVectorRowV1 {
                    layer: 2,
                    values: second.to_vec(),
                },
            ],
        }
    }

    #[test]
    fn mean_is_ordered_deterministic_and_content_free() {
        let topology = topology();
        let samples = [
            sample(b"one", [1.0, 3.0], [4.0, 0.0]),
            sample(b"two", [3.0, 1.0], [0.0, 4.0]),
        ];
        let bank = ActivationVectorAccumulatorV1::mean(
            &topology,
            TextTensorSiteFamilyV1::LayerOutput,
            &samples,
            ActivationVectorNormalizationV1::None,
        )
        .unwrap();
        assert_eq!(bank.row(0), Some(&[2.0, 2.0][..]));
        assert_eq!(bank.row(2), Some(&[2.0, 2.0][..]));
        assert_eq!(
            bank.provenance.ordered_inputs,
            samples
                .iter()
                .map(|sample| sample.source.clone())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn difference_normalizes_each_sparse_row() {
        let topology = topology();
        let left = [sample(b"left", [2.0, 0.0], [0.0, 3.0])];
        let right = [sample(b"right", [0.0, 0.0], [0.0, 0.0])];
        let bank = ActivationVectorAccumulatorV1::difference_of_means(
            &topology,
            TextTensorSiteFamilyV1::LayerOutput,
            &left,
            &right,
            ActivationVectorNormalizationV1::UnitL2,
        )
        .unwrap();
        assert_eq!(bank.row(0), Some(&[1.0, 0.0][..]));
        assert_eq!(bank.row(2), Some(&[0.0, 1.0][..]));
    }

    #[test]
    fn accumulator_rejects_mismatched_layers_and_zero_norms() {
        let topology = topology();
        let first = sample(b"one", [1.0, 0.0], [0.0, 1.0]);
        let mut second = sample(b"two", [1.0, 0.0], [0.0, 1.0]);
        second.rows[1].layer = 1;
        assert!(
            ActivationVectorAccumulatorV1::mean(
                &topology,
                TextTensorSiteFamilyV1::LayerOutput,
                &[first, second],
                ActivationVectorNormalizationV1::None,
            )
            .is_err()
        );
        assert!(
            ActivationVectorAccumulatorV1::mean(
                &topology,
                TextTensorSiteFamilyV1::LayerOutput,
                &[sample(b"zero", [0.0, 0.0], [0.0, 0.0])],
                ActivationVectorNormalizationV1::UnitL2,
            )
            .is_err()
        );
    }
}

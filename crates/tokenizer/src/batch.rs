// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded batching, length buckets, and stable scatter.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{MAX_BULK_ROWS, TokenizationError};

/// One offline-qualified discrete batch candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchCandidate {
    /// Maximum sequence count.
    pub sequences: u32,
    /// Maximum aggregate token count.
    pub tokens: u32,
    /// Maximum aggregate source bytes.
    pub bytes: u64,
}

impl BatchCandidate {
    /// Validates a nonzero bounded candidate.
    ///
    /// # Errors
    ///
    /// Returns an error for zero or excessive bounds.
    pub fn validate(self) -> Result<(), TokenizationError> {
        if self.sequences == 0
            || !usize::try_from(self.sequences).is_ok_and(|value| value <= MAX_BULK_ROWS)
            || self.tokens == 0
            || self.bytes == 0
        {
            return Err(TokenizationError::Invalid(
                "batch candidate is outside public bounds".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Runtime input bounds independent of a specific candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchLimits {
    /// Maximum rows considered by one plan.
    pub maximum_rows: u32,
    /// Maximum aggregate source bytes.
    pub maximum_bytes: u64,
    /// Maximum aggregate estimated tokens.
    pub maximum_tokens: u64,
    /// Bucket width in estimated tokens.
    pub bucket_width: u32,
}

impl BatchLimits {
    fn validate(self) -> Result<(), TokenizationError> {
        if self.maximum_rows == 0
            || !usize::try_from(self.maximum_rows).is_ok_and(|value| value <= MAX_BULK_ROWS)
            || self.maximum_bytes == 0
            || self.maximum_tokens == 0
            || self.bucket_width == 0
        {
            return Err(TokenizationError::Invalid(
                "batch limits are outside public bounds".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Content-free row estimate used by a planner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchRow {
    /// Original stable row index.
    pub index: u32,
    /// Source byte length.
    pub bytes: u64,
    /// Bounded token-count estimate.
    pub estimated_tokens: u64,
    /// Absolute deadline, where lower values are earlier.
    pub deadline_millis: u64,
}

/// One length-compatible bucket in execution order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LengthBucket {
    /// Bucket's lower inclusive token estimate.
    pub token_floor: u64,
    /// Stable row indices, ordered by deadline then input index.
    pub rows: Vec<u32>,
    /// Aggregate source bytes.
    pub bytes: u64,
    /// Aggregate estimated tokens.
    pub estimated_tokens: u64,
    /// Padding tokens implied by the bucket maximum.
    pub padding_tokens: u64,
}

/// Stable scatter from execution order back to input order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StableScatter {
    /// For each execution row, its original input row.
    execution_to_input: Vec<u32>,
    /// For each input row, its execution row.
    input_to_execution: Vec<u32>,
}

impl StableScatter {
    /// Builds a bijective scatter.
    ///
    /// # Errors
    ///
    /// Returns an error for duplicates, gaps, or excessive rows.
    pub fn new(execution_to_input: Vec<u32>) -> Result<Self, TokenizationError> {
        if execution_to_input.len() > MAX_BULK_ROWS {
            return Err(TokenizationError::Bound {
                field: "scatter rows",
                limit: MAX_BULK_ROWS,
            });
        }
        let mut input_to_execution = vec![u32::MAX; execution_to_input.len()];
        for (execution, input) in execution_to_input.iter().copied().enumerate() {
            let input = usize::try_from(input).map_err(|_| {
                TokenizationError::Invalid("scatter index exceeds usize".to_owned())
            })?;
            let Some(slot) = input_to_execution.get_mut(input) else {
                return Err(TokenizationError::Invalid(
                    "scatter index is outside row range".to_owned(),
                ));
            };
            if *slot != u32::MAX {
                return Err(TokenizationError::Invalid(
                    "scatter contains a duplicate row".to_owned(),
                ));
            }
            *slot = u32::try_from(execution).map_err(|_| {
                TokenizationError::Invalid("scatter execution index overflowed".to_owned())
            })?;
        }
        if input_to_execution.contains(&u32::MAX) {
            return Err(TokenizationError::Invalid(
                "scatter does not cover every input row".to_owned(),
            ));
        }
        Ok(Self {
            execution_to_input,
            input_to_execution,
        })
    }

    /// Returns execution-to-input indices.
    #[must_use]
    pub fn execution_to_input(&self) -> &[u32] {
        &self.execution_to_input
    }

    /// Returns input-to-execution indices.
    #[must_use]
    pub fn input_to_execution(&self) -> &[u32] {
        &self.input_to_execution
    }
}

/// Complete batch layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchPlan {
    /// Length buckets in execution order.
    pub buckets: Vec<LengthBucket>,
    /// Stable output scatter.
    pub scatter: StableScatter,
}

/// Plans bounded length buckets without inspecting source content.
///
/// Rows are first grouped by the configured token width, then ordered by
/// deadline and original row index. The resulting scatter restores the exact
/// caller row order.
///
/// # Errors
///
/// Returns an error for invalid limits, non-contiguous row indices, or
/// aggregate-bound overflow.
pub fn plan_length_buckets(
    rows: &[BatchRow],
    limits: BatchLimits,
) -> Result<BatchPlan, TokenizationError> {
    limits.validate()?;
    if rows.is_empty()
        || rows.len()
            > usize::try_from(limits.maximum_rows).map_err(|_| {
                TokenizationError::Invalid("maximum row count overflowed".to_owned())
            })?
    {
        return Err(TokenizationError::Invalid(
            "batch row count is outside configured bounds".to_owned(),
        ));
    }
    let mut seen = vec![false; rows.len()];
    let mut total_bytes = 0_u64;
    let mut total_tokens = 0_u64;
    let mut grouped: BTreeMap<u64, Vec<BatchRow>> = BTreeMap::new();
    for row in rows {
        let index = usize::try_from(row.index)
            .map_err(|_| TokenizationError::Invalid("row index overflowed".to_owned()))?;
        let Some(slot) = seen.get_mut(index) else {
            return Err(TokenizationError::Invalid(
                "row index is outside the batch".to_owned(),
            ));
        };
        if std::mem::replace(slot, true) || row.bytes == 0 {
            return Err(TokenizationError::Invalid(
                "batch rows are duplicated or empty".to_owned(),
            ));
        }
        total_bytes = total_bytes
            .checked_add(row.bytes)
            .ok_or_else(|| TokenizationError::Invalid("batch bytes overflowed".to_owned()))?;
        total_tokens = total_tokens
            .checked_add(row.estimated_tokens)
            .ok_or_else(|| TokenizationError::Invalid("batch tokens overflowed".to_owned()))?;
        let bucket = row.estimated_tokens / u64::from(limits.bucket_width);
        grouped.entry(bucket).or_default().push(*row);
    }
    if total_bytes > limits.maximum_bytes || total_tokens > limits.maximum_tokens {
        return Err(TokenizationError::Invalid(
            "batch aggregate exceeds configured bounds".to_owned(),
        ));
    }
    let mut buckets = Vec::with_capacity(grouped.len());
    let mut execution_to_input = Vec::with_capacity(rows.len());
    for (bucket, mut values) in grouped {
        values.sort_unstable_by_key(|row| (row.deadline_millis, row.index));
        let maximum = values
            .iter()
            .map(|row| row.estimated_tokens)
            .max()
            .unwrap_or(0);
        let estimated_tokens = values
            .iter()
            .try_fold(0_u64, |sum, row| sum.checked_add(row.estimated_tokens));
        let Some(estimated_tokens) = estimated_tokens else {
            return Err(TokenizationError::Invalid(
                "bucket token sum overflowed".to_owned(),
            ));
        };
        let padded = maximum
            .checked_mul(u64::try_from(values.len()).map_err(|_| {
                TokenizationError::Invalid("bucket row count overflowed".to_owned())
            })?)
            .ok_or_else(|| TokenizationError::Invalid("padding sum overflowed".to_owned()))?;
        let bytes = values
            .iter()
            .try_fold(0_u64, |sum, row| sum.checked_add(row.bytes))
            .ok_or_else(|| TokenizationError::Invalid("bucket bytes overflowed".to_owned()))?;
        let row_indices: Vec<u32> = values.iter().map(|row| row.index).collect();
        execution_to_input.extend(row_indices.iter().copied());
        buckets.push(LengthBucket {
            token_floor: bucket * u64::from(limits.bucket_width),
            rows: row_indices,
            bytes,
            estimated_tokens,
            padding_tokens: padded.saturating_sub(estimated_tokens),
        });
    }
    Ok(BatchPlan {
        buckets,
        scatter: StableScatter::new(execution_to_input)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_plan_is_deadline_ordered_and_bijective() {
        let rows = [
            BatchRow {
                index: 0,
                bytes: 10,
                estimated_tokens: 9,
                deadline_millis: 20,
            },
            BatchRow {
                index: 1,
                bytes: 10,
                estimated_tokens: 2,
                deadline_millis: 30,
            },
            BatchRow {
                index: 2,
                bytes: 10,
                estimated_tokens: 8,
                deadline_millis: 10,
            },
        ];
        let plan = plan_length_buckets(
            &rows,
            BatchLimits {
                maximum_rows: 3,
                maximum_bytes: 30,
                maximum_tokens: 19,
                bucket_width: 8,
            },
        )
        .unwrap();
        assert_eq!(plan.scatter.execution_to_input(), &[1, 2, 0]);
        assert_eq!(plan.scatter.input_to_execution(), &[2, 0, 1]);
        assert_eq!(plan.buckets[1].padding_tokens, 1);
    }

    #[test]
    fn scatter_rejects_duplicates_and_gaps() {
        assert!(StableScatter::new(vec![0, 0]).is_err());
        assert!(StableScatter::new(vec![1]).is_err());
    }
}

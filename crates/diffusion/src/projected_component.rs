// SPDX-License-Identifier: MIT OR Apache-2.0

//! Create-new projection transforms for floating-point `SafeTensors` components.

use std::collections::{BTreeMap, BTreeSet};

use logit_loom_core::{CoreError, Digest};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Maximum selected tensors in one derived-component transform.
pub const MAX_PROJECTED_COMPONENT_TENSORS: usize = 4_096;
/// Maximum UTF-8 bytes in one selected tensor name.
pub const MAX_PROJECTED_COMPONENT_TENSOR_NAME_BYTES: usize = 1_024;
/// Maximum basis rank accepted by the reference transform.
pub const MAX_PROJECTED_COMPONENT_BASIS_RANK: u32 = 4_096;
/// Maximum finite basis elements accepted by the reference transform.
pub const MAX_PROJECTED_COMPONENT_BASIS_ELEMENTS: u64 = 64 * 1024 * 1024;
/// Maximum `SafeTensors` JSON header accepted by the transform.
pub const MAX_PROJECTED_COMPONENT_HEADER_BYTES: usize = 64 * 1024 * 1024;

/// Matrix side on which an orthogonal projection is applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectedComponentFormulaV1 {
    /// `W' = (I - alpha U U^T) W`; the first tensor axis is the feature axis.
    LeftProjection,
    /// `W' = W (I - alpha U U^T)`; the second tensor axis is the feature axis.
    RightProjection,
}

/// Deterministic reduction and output-scalar policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectedComponentReductionV1 {
    /// Canonical row/rank order, `f64` accumulation, and one final `f32` cast.
    OrderedF64ToF32,
}

/// Exact orthonormal basis declaration.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectedComponentBasisV1 {
    /// Feature width of each rank-major basis row.
    pub feature_width: u32,
    /// Number of rank-major basis rows.
    pub rank: u32,
    /// Exact canonical little-endian `f32` basis identity.
    pub data: Digest,
}

/// Exact create-new component transform.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectedComponentPlanV1 {
    /// Exact runtime-compatible topology identity.
    pub topology: Digest,
    /// Exact source component bytes.
    pub source: Digest,
    /// Exact orthonormal basis declaration.
    pub basis: ProjectedComponentBasisV1,
    /// Matrix-side projection formula.
    pub formula: ProjectedComponentFormulaV1,
    /// Canonical deterministic reduction policy.
    pub reduction: ProjectedComponentReductionV1,
    /// Exact IEEE-754 projection strength bits in `0..=1`.
    pub strength_bits: u32,
    /// Canonically ordered exact `SafeTensors` parameter names.
    pub tensors: Vec<String>,
    /// Exact implementation identity.
    pub implementation: Digest,
}

impl ProjectedComponentPlanV1 {
    /// Returns the finite projection strength.
    ///
    /// # Errors
    ///
    /// Returns an error for NaN, infinity, or a value outside `0..=1`.
    pub fn strength(&self) -> Result<f32, CoreError> {
        let strength = f32::from_bits(self.strength_bits);
        if !strength.is_finite() || !(0.0..=1.0).contains(&strength) {
            return Err(CoreError::invalid(
                "projected component plan",
                "strength must be finite and within 0..=1",
            ));
        }
        Ok(strength)
    }

    /// Validates canonical plan metadata and returns its identity.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed basis dimensions, tensor selectors,
    /// strength, reduction, or implementation identity.
    pub fn digest(&self) -> Result<Digest, CoreError> {
        let elements = u64::from(self.basis.feature_width)
            .checked_mul(u64::from(self.basis.rank))
            .ok_or_else(|| {
                CoreError::invalid("projected component plan", "basis element count overflowed")
            })?;
        if self.basis.feature_width == 0
            || self.basis.rank == 0
            || self.basis.rank > MAX_PROJECTED_COMPONENT_BASIS_RANK
            || elements > MAX_PROJECTED_COMPONENT_BASIS_ELEMENTS
            || self.tensors.is_empty()
            || self.tensors.len() > MAX_PROJECTED_COMPONENT_TENSORS
            || self.tensors.iter().any(|name| {
                name.is_empty()
                    || name.len() > MAX_PROJECTED_COMPONENT_TENSOR_NAME_BYTES
                    || name.contains('\0')
            })
            || !self.tensors.windows(2).all(|pair| pair[0] < pair[1])
            || self.implementation != projected_component_implementation_v1()
        {
            return Err(CoreError::invalid(
                "projected component plan",
                "basis, tensors, or implementation are empty, excessive, or non-canonical",
            ));
        }
        self.strength()?;
        Digest::of_serializable("projected-component-plan-v1", self)
    }
}

/// Before/after identity for one transformed tensor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectedComponentTensorReceiptV1 {
    /// Exact tensor name.
    pub name: String,
    /// Tensor bytes before projection.
    pub before: Digest,
    /// Tensor bytes after projection.
    pub after: Digest,
    /// First matrix dimension.
    pub rows: u64,
    /// Second matrix dimension.
    pub columns: u64,
}

/// Complete create-new component mutation manifest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectedComponentManifestV1 {
    /// Exact plan identity.
    pub plan: Digest,
    /// Exact source bytes.
    pub source: Digest,
    /// Required topology identity.
    pub topology: Digest,
    /// Exact basis identity.
    pub basis: Digest,
    /// Applied formula.
    pub formula: ProjectedComponentFormulaV1,
    /// Applied implementation.
    pub implementation: Digest,
    /// Exact new component bytes.
    pub output: Digest,
    /// Canonically tensor-ordered mutation records.
    pub tensors: Vec<ProjectedComponentTensorReceiptV1>,
}

impl ProjectedComponentManifestV1 {
    /// Validates complete lineage and returns the manifest identity.
    ///
    /// # Errors
    ///
    /// Returns an error for inconsistent plan/source/basis/formula/output or
    /// incomplete/non-canonical selected-tensor coverage.
    pub fn digest_for(
        &self,
        plan: &ProjectedComponentPlanV1,
        output: &[u8],
    ) -> Result<Digest, CoreError> {
        if self.plan != plan.digest()?
            || self.source != plan.source
            || self.topology != plan.topology
            || self.basis != plan.basis.data
            || self.formula != plan.formula
            || self.implementation != plan.implementation
            || self.output != projected_component_output_v1(output)
            || self.tensors.len() != plan.tensors.len()
            || self
                .tensors
                .iter()
                .map(|tensor| &tensor.name)
                .ne(plan.tensors.iter())
            || self
                .tensors
                .iter()
                .any(|tensor| tensor.rows == 0 || tensor.columns == 0)
        {
            return Err(CoreError::invalid(
                "projected component manifest",
                "lineage, output, or tensor coverage differs from the plan",
            ));
        }
        let no_op = plan.strength()? == 0.0;
        if no_op
            && self
                .tensors
                .iter()
                .any(|tensor| tensor.before != tensor.after)
        {
            return Err(CoreError::invalid(
                "projected component manifest",
                "tensor mutation identities disagree with no-op status",
            ));
        }
        Digest::of_serializable("projected-component-manifest-v1", self)
    }
}

/// Successful create-new transformation output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectedComponentOutputV1 {
    /// Complete new `SafeTensors` bytes.
    pub bytes: Vec<u8>,
    /// Exact mutation manifest.
    pub manifest: ProjectedComponentManifestV1,
}

/// Returns the exact reference-transform implementation identity.
pub fn projected_component_implementation_v1() -> Digest {
    Digest::of_bytes(
        "projected-component-implementation-v1",
        b"safetensors-f32;preserve-header-and-unselected-bytes;ordered-f64-to-f32;create-new",
    )
}

/// Returns the exact source-component identity.
pub fn projected_component_source_v1(bytes: &[u8]) -> Digest {
    Digest::of_bytes("projected-component-source-v1", bytes)
}

/// Returns the exact canonical little-endian basis identity.
pub fn projected_component_basis_v1(bytes: &[u8]) -> Digest {
    Digest::of_bytes("projected-component-basis-f32-le-v1", bytes)
}

/// Returns the exact derived-component identity.
pub fn projected_component_output_v1(bytes: &[u8]) -> Digest {
    Digest::of_bytes("projected-component-output-v1", bytes)
}

/// Creates a new projected `SafeTensors` component without changing the source.
///
/// The exact source header, metadata, unselected tensor bytes, tensor offsets,
/// and file length are preserved. Only selected finite `F32` matrix payloads
/// are rewritten.
///
/// # Errors
///
/// Returns an error for substituted source/basis bytes, malformed or aliased
/// `SafeTensors` ranges, unsupported dtype/rank, missing selectors, incompatible
/// feature axes, non-finite data, a non-orthonormal basis, or arithmetic
/// overflow. No partial output is returned.
pub fn project_safetensors_component_v1(
    source: &[u8],
    basis_bytes: &[u8],
    plan: &ProjectedComponentPlanV1,
) -> Result<ProjectedComponentOutputV1, CoreError> {
    plan.digest()?;
    if plan.source != projected_component_source_v1(source)
        || plan.basis.data != projected_component_basis_v1(basis_bytes)
    {
        return Err(CoreError::invalid(
            "projected component input",
            "source or basis identity differs from the plan",
        ));
    }
    let basis = decode_f32(basis_bytes, "projected component basis")?;
    validate_basis(&basis, &plan.basis)?;
    let parsed = ParsedSafeTensors::parse(source)?;
    let selected = plan.tensors.iter().cloned().collect::<BTreeSet<_>>();
    if selected
        .iter()
        .any(|name| !parsed.tensors.contains_key(name))
    {
        return Err(CoreError::invalid(
            "projected component plan",
            "a selected tensor is absent from the source",
        ));
    }

    let mut output = source.to_vec();
    let mut receipts = Vec::with_capacity(plan.tensors.len());
    for name in &plan.tensors {
        let tensor = parsed.tensors.get(name).ok_or_else(|| {
            CoreError::invalid("projected component plan", "selected tensor is absent")
        })?;
        if tensor.dtype != "F32" || tensor.shape.len() != 2 {
            return Err(CoreError::invalid(
                "projected component tensor",
                "selected tensors must be rank-two F32 matrices",
            ));
        }
        let rows = tensor.shape[0];
        let columns = tensor.shape[1];
        let feature = match plan.formula {
            ProjectedComponentFormulaV1::LeftProjection => rows,
            ProjectedComponentFormulaV1::RightProjection => columns,
        };
        if feature != u64::from(plan.basis.feature_width) {
            return Err(CoreError::invalid(
                "projected component tensor",
                "declared feature axis is incompatible with the basis width",
            ));
        }
        let absolute_start = parsed
            .data_start
            .checked_add(tensor.start)
            .ok_or_else(|| CoreError::invalid("SafeTensors", "tensor offset overflowed"))?;
        let absolute_end = parsed
            .data_start
            .checked_add(tensor.end)
            .ok_or_else(|| CoreError::invalid("SafeTensors", "tensor end overflowed"))?;
        let source_tensor = source.get(absolute_start..absolute_end).ok_or_else(|| {
            CoreError::invalid("SafeTensors", "tensor range exceeds source bytes")
        })?;
        let values = decode_f32(source_tensor, "projected component tensor")?;
        let expected_elements = rows.checked_mul(columns).ok_or_else(|| {
            CoreError::invalid("projected component tensor", "matrix size overflowed")
        })?;
        if u64::try_from(values.len()).unwrap_or(u64::MAX) != expected_elements {
            return Err(CoreError::invalid(
                "projected component tensor",
                "matrix shape and byte count differ",
            ));
        }
        let transformed = transform_matrix(&values, rows, columns, &basis, plan)?;
        let transformed_bytes = encode_f32(&transformed);
        if transformed_bytes.len() != source_tensor.len() {
            return Err(CoreError::invalid(
                "projected component tensor",
                "projection changed the tensor byte length",
            ));
        }
        output[absolute_start..absolute_end].copy_from_slice(&transformed_bytes);
        receipts.push(ProjectedComponentTensorReceiptV1 {
            name: name.clone(),
            before: Digest::of_bytes("projected-component-tensor-v1", source_tensor),
            after: Digest::of_bytes("projected-component-tensor-v1", &transformed_bytes),
            rows,
            columns,
        });
    }

    let manifest = ProjectedComponentManifestV1 {
        plan: plan.digest()?,
        source: plan.source.clone(),
        topology: plan.topology.clone(),
        basis: plan.basis.data.clone(),
        formula: plan.formula,
        implementation: plan.implementation.clone(),
        output: projected_component_output_v1(&output),
        tensors: receipts,
    };
    manifest.digest_for(plan, &output)?;
    Ok(ProjectedComponentOutputV1 {
        bytes: output,
        manifest,
    })
}

/// Recomputes and verifies one complete create-new transform.
///
/// # Errors
///
/// Returns an error if the output or manifest differs from a fresh exact
/// transform of the supplied source and basis.
pub fn verify_projected_component_v1(
    source: &[u8],
    basis_bytes: &[u8],
    output: &[u8],
    plan: &ProjectedComponentPlanV1,
    manifest: &ProjectedComponentManifestV1,
) -> Result<(), CoreError> {
    manifest.digest_for(plan, output)?;
    let expected = project_safetensors_component_v1(source, basis_bytes, plan)?;
    if expected.bytes != output || expected.manifest != *manifest {
        return Err(CoreError::invalid(
            "projected component verification",
            "output or manifest differs from exact recomputation",
        ));
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct TensorEntry {
    dtype: String,
    shape: Vec<u64>,
    start: usize,
    end: usize,
}

struct ParsedSafeTensors {
    data_start: usize,
    tensors: BTreeMap<String, TensorEntry>,
}

impl ParsedSafeTensors {
    fn parse(source: &[u8]) -> Result<Self, CoreError> {
        let length_bytes: [u8; 8] = source
            .get(..8)
            .ok_or_else(|| CoreError::invalid("SafeTensors", "header length is absent"))?
            .try_into()
            .map_err(|_| CoreError::invalid("SafeTensors", "header length is malformed"))?;
        let header_len = usize::try_from(u64::from_le_bytes(length_bytes))
            .map_err(|_| CoreError::invalid("SafeTensors", "header length exceeds usize"))?;
        if header_len == 0 || header_len > MAX_PROJECTED_COMPONENT_HEADER_BYTES {
            return Err(CoreError::invalid(
                "SafeTensors",
                "header length is zero or excessive",
            ));
        }
        let data_start = 8_usize
            .checked_add(header_len)
            .ok_or_else(|| CoreError::invalid("SafeTensors", "data offset overflowed"))?;
        let header = source
            .get(8..data_start)
            .ok_or_else(|| CoreError::invalid("SafeTensors", "header exceeds source bytes"))?;
        let root: Value = serde_json::from_slice(header).map_err(|error| {
            CoreError::invalid("SafeTensors", format!("header JSON is invalid: {error}"))
        })?;
        let object = root
            .as_object()
            .ok_or_else(|| CoreError::invalid("SafeTensors", "header root must be an object"))?;
        let data_bytes = source
            .len()
            .checked_sub(data_start)
            .ok_or_else(|| CoreError::invalid("SafeTensors", "data offset exceeds source bytes"))?;
        let mut tensors = BTreeMap::new();
        let mut ranges = Vec::new();
        for (name, value) in object {
            if name == "__metadata__" {
                if !value.is_object() {
                    return Err(CoreError::invalid(
                        "SafeTensors",
                        "metadata must remain a JSON object",
                    ));
                }
                continue;
            }
            if name.is_empty()
                || name.len() > MAX_PROJECTED_COMPONENT_TENSOR_NAME_BYTES
                || name.contains('\0')
            {
                return Err(CoreError::invalid(
                    "SafeTensors",
                    "tensor name is empty, excessive, or contains NUL",
                ));
            }
            let entry = parse_tensor_entry(value)?;
            if entry.start >= entry.end || entry.end > data_bytes {
                return Err(CoreError::invalid(
                    "SafeTensors",
                    "tensor range is empty or outside the data section",
                ));
            }
            ranges.push((entry.start, entry.end, name.as_str()));
            tensors.insert(name.clone(), entry);
        }
        if tensors.is_empty() {
            return Err(CoreError::invalid("SafeTensors", "contains no tensors"));
        }
        ranges.sort_unstable_by_key(|range| (range.0, range.1));
        if ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err(CoreError::invalid(
                "SafeTensors",
                "aliased or overlapping tensor ranges are unsupported",
            ));
        }
        Ok(Self {
            data_start,
            tensors,
        })
    }
}

fn parse_tensor_entry(value: &Value) -> Result<TensorEntry, CoreError> {
    let object = value
        .as_object()
        .ok_or_else(|| CoreError::invalid("SafeTensors", "tensor entry must be an object"))?;
    if object.len() != 3
        || !object.contains_key("dtype")
        || !object.contains_key("shape")
        || !object.contains_key("data_offsets")
    {
        return Err(CoreError::invalid(
            "SafeTensors",
            "tensor entry must contain only dtype, shape, and data_offsets",
        ));
    }
    let dtype = object["dtype"]
        .as_str()
        .ok_or_else(|| CoreError::invalid("SafeTensors", "dtype must be a string"))?
        .to_owned();
    let shape = object["shape"]
        .as_array()
        .ok_or_else(|| CoreError::invalid("SafeTensors", "shape must be an array"))?
        .iter()
        .map(|dimension| {
            dimension
                .as_u64()
                .filter(|dimension| *dimension > 0)
                .ok_or_else(|| {
                    CoreError::invalid("SafeTensors", "shape dimensions must be positive integers")
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let offsets = object["data_offsets"]
        .as_array()
        .filter(|offsets| offsets.len() == 2)
        .ok_or_else(|| {
            CoreError::invalid("SafeTensors", "data_offsets must contain two integers")
        })?;
    let start = usize::try_from(
        offsets[0]
            .as_u64()
            .ok_or_else(|| CoreError::invalid("SafeTensors", "tensor start must be an integer"))?,
    )
    .map_err(|_| CoreError::invalid("SafeTensors", "tensor start exceeds usize"))?;
    let end = usize::try_from(
        offsets[1]
            .as_u64()
            .ok_or_else(|| CoreError::invalid("SafeTensors", "tensor end must be an integer"))?,
    )
    .map_err(|_| CoreError::invalid("SafeTensors", "tensor end exceeds usize"))?;
    Ok(TensorEntry {
        dtype,
        shape,
        start,
        end,
    })
}

fn transform_matrix(
    values: &[f32],
    rows: u64,
    columns: u64,
    basis: &[f32],
    plan: &ProjectedComponentPlanV1,
) -> Result<Vec<f32>, CoreError> {
    let rows = usize::try_from(rows)
        .map_err(|_| CoreError::invalid("projected component", "rows exceed usize"))?;
    let columns = usize::try_from(columns)
        .map_err(|_| CoreError::invalid("projected component", "columns exceed usize"))?;
    let width = usize::try_from(plan.basis.feature_width)
        .map_err(|_| CoreError::invalid("projected component", "basis width exceeds usize"))?;
    let rank = usize::try_from(plan.basis.rank)
        .map_err(|_| CoreError::invalid("projected component", "basis rank exceeds usize"))?;
    let strength = f64::from(plan.strength()?);
    if strength == 0.0 {
        return Ok(values.to_vec());
    }
    let mut output = values.to_vec();
    match plan.formula {
        ProjectedComponentFormulaV1::LeftProjection => {
            for column in 0..columns {
                for vector in 0..rank {
                    let basis_row = &basis[vector * width..(vector + 1) * width];
                    let coefficient =
                        basis_row
                            .iter()
                            .enumerate()
                            .fold(0.0_f64, |sum, (row, basis)| {
                                sum + f64::from(*basis) * f64::from(values[row * columns + column])
                            });
                    for (row, basis) in basis_row.iter().enumerate() {
                        let index = row * columns + column;
                        let next =
                            f64::from(output[index]) - strength * coefficient * f64::from(*basis);
                        output[index] = finite_f32(next)?;
                    }
                }
            }
        }
        ProjectedComponentFormulaV1::RightProjection => {
            for row in 0..rows {
                let offset = row * columns;
                for vector in 0..rank {
                    let basis_row = &basis[vector * width..(vector + 1) * width];
                    let coefficient =
                        basis_row
                            .iter()
                            .enumerate()
                            .fold(0.0_f64, |sum, (column, basis)| {
                                sum + f64::from(values[offset + column]) * f64::from(*basis)
                            });
                    for (column, basis) in basis_row.iter().enumerate() {
                        let index = offset + column;
                        let next =
                            f64::from(output[index]) - strength * coefficient * f64::from(*basis);
                        output[index] = finite_f32(next)?;
                    }
                }
            }
        }
    }
    Ok(output)
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "the declared output representation is finite f32"
)]
fn finite_f32(value: f64) -> Result<f32, CoreError> {
    let value = value as f32;
    if !value.is_finite() {
        return Err(CoreError::invalid(
            "projected component arithmetic",
            "result overflowed finite f32",
        ));
    }
    Ok(value)
}

fn validate_basis(values: &[f32], basis: &ProjectedComponentBasisV1) -> Result<(), CoreError> {
    const TOLERANCE: f64 = 1.0e-5;

    let width = usize::try_from(basis.feature_width)
        .map_err(|_| CoreError::invalid("projected component basis", "width exceeds usize"))?;
    let rank = usize::try_from(basis.rank)
        .map_err(|_| CoreError::invalid("projected component basis", "rank exceeds usize"))?;
    if values.len() != width.saturating_mul(rank) {
        return Err(CoreError::invalid(
            "projected component basis",
            "byte count differs from width and rank",
        ));
    }
    for left in 0..rank {
        for right in left..rank {
            let dot = values[left * width..(left + 1) * width]
                .iter()
                .zip(&values[right * width..(right + 1) * width])
                .fold(0.0_f64, |sum, (left, right)| {
                    sum + f64::from(*left) * f64::from(*right)
                });
            let expected = if left == right { 1.0 } else { 0.0 };
            if !dot.is_finite() || (dot - expected).abs() > TOLERANCE {
                return Err(CoreError::invalid(
                    "projected component basis",
                    "basis rows are not orthonormal",
                ));
            }
        }
    }
    Ok(())
}

fn decode_f32(bytes: &[u8], field: &'static str) -> Result<Vec<f32>, CoreError> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return Err(CoreError::invalid(
            field,
            "bytes must contain complete nonempty f32 elements",
        ));
    }
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let value =
                f32::from_bits(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            value
                .is_finite()
                .then_some(value)
                .ok_or_else(|| CoreError::invalid(field, "every value must be finite"))
        })
        .collect()
}

fn encode_f32(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        encode_f32(values)
    }

    fn safetensors(first: &[f32], second: &[f32]) -> Vec<u8> {
        let first_bytes = first.len() * 4;
        let second_end = first_bytes + second.len() * 4;
        let header = format!(
            "{{\"__metadata__\":{{\"kept\":\"exact\"}},\"a\":{{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,{first_bytes}]}},\"b\":{{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[{first_bytes},{second_end}]}}}}"
        );
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&u64::try_from(header.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&f32_bytes(first));
        bytes.extend_from_slice(&f32_bytes(second));
        bytes
    }

    fn plan(source: &[u8], basis: &[u8], strength: f32) -> ProjectedComponentPlanV1 {
        ProjectedComponentPlanV1 {
            topology: Digest::of_bytes("test-projection-topology", b"krea"),
            source: projected_component_source_v1(source),
            basis: ProjectedComponentBasisV1 {
                feature_width: 2,
                rank: 1,
                data: projected_component_basis_v1(basis),
            },
            formula: ProjectedComponentFormulaV1::LeftProjection,
            reduction: ProjectedComponentReductionV1::OrderedF64ToF32,
            strength_bits: strength.to_bits(),
            tensors: vec!["a".to_owned()],
            implementation: projected_component_implementation_v1(),
        }
    }

    #[test]
    fn zero_strength_copies_every_byte_exactly() {
        let source = safetensors(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
        let basis = f32_bytes(&[1.0, 0.0]);
        let plan = plan(&source, &basis, 0.0);
        let output = project_safetensors_component_v1(&source, &basis, &plan).unwrap();
        assert_eq!(output.bytes, source);
        assert_eq!(
            output.manifest.tensors[0].before,
            output.manifest.tensors[0].after
        );
        verify_projected_component_v1(&source, &basis, &output.bytes, &plan, &output.manifest)
            .unwrap();
    }

    #[test]
    fn left_projection_changes_only_selected_tensor_payload() {
        let source = safetensors(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
        let basis = f32_bytes(&[1.0, 0.0]);
        let plan = plan(&source, &basis, 1.0);
        let output = project_safetensors_component_v1(&source, &basis, &plan).unwrap();
        let parsed = ParsedSafeTensors::parse(&source).unwrap();
        let a = &parsed.tensors["a"];
        let b = &parsed.tensors["b"];
        let a_start = parsed.data_start + a.start;
        let a_end = parsed.data_start + a.end;
        let b_start = parsed.data_start + b.start;
        let b_end = parsed.data_start + b.end;
        assert_eq!(
            decode_f32(&output.bytes[a_start..a_end], "test").unwrap(),
            vec![0.0, 0.0, 3.0, 4.0]
        );
        assert_eq!(&output.bytes[..a_start], &source[..a_start]);
        assert_eq!(&output.bytes[b_start..b_end], &source[b_start..b_end]);
    }

    #[test]
    fn malformed_alias_basis_and_orientation_fail_without_output() {
        let source = safetensors(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
        let non_orthogonal = f32_bytes(&[2.0, 0.0]);
        let bad_basis_plan = plan(&source, &non_orthogonal, 1.0);
        assert!(
            project_safetensors_component_v1(&source, &non_orthogonal, &bad_basis_plan).is_err()
        );

        let basis = f32_bytes(&[1.0, 0.0]);
        let mut wrong_axis = plan(&source, &basis, 1.0);
        wrong_axis.basis.feature_width = 3;
        assert!(project_safetensors_component_v1(&source, &basis, &wrong_axis).is_err());

        let header = "{\"a\":{\"dtype\":\"F32\",\"shape\":[1,2],\"data_offsets\":[0,8]},\"b\":{\"dtype\":\"F32\",\"shape\":[1,2],\"data_offsets\":[4,12]}}";
        let mut aliased = Vec::new();
        aliased.extend_from_slice(&u64::try_from(header.len()).unwrap().to_le_bytes());
        aliased.extend_from_slice(header.as_bytes());
        aliased.extend_from_slice(&[0_u8; 12]);
        let alias_plan = plan(&aliased, &basis, 1.0);
        assert!(project_safetensors_component_v1(&aliased, &basis, &alias_plan).is_err());
    }
}

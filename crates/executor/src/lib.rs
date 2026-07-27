// SPDX-License-Identifier: MIT OR Apache-2.0

#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

use std::fmt::Debug;

use logit_loom_core::{CoreError, Digest};
use serde::{Deserialize, Serialize};

mod whole;

pub use whole::{
    ClassifiedWholeGenerationError, DurabilityReceipt, MAX_WHOLE_GENERATION_EVIDENCE,
    MAX_WHOLE_GENERATION_EVIDENCE_BYTES, MAX_WHOLE_GENERATION_INPUT_BYTES,
    MAX_WHOLE_GENERATION_OUTPUT_BYTES, PendingGeneration, ProviderOwnedGenerationPlan,
    ProviderOwnedSampler, WholeGenerationBackend, WholeGenerationCommitReceipt,
    WholeGenerationEvidence, WholeGenerationFailure, WholeGenerationOutput, WholeGenerationPlan,
    WholeGenerationReceipt, WholeGenerationRequest,
};

/// Maximum bytes in a buffer media-type label.
pub const MAX_MEDIA_TYPE_BYTES: usize = 128;
/// Maximum buffers presented to one local execution.
pub const MAX_EXECUTION_BUFFERS: usize = 64;

/// Mechanical state of one worker-local executor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ExecutorState {
    /// No backend state is resident.
    Unloaded,
    /// Exact artifacts are resident and no operation is running.
    Resident,
    /// One synchronous operation is running.
    Busy,
    /// Native or cleanup state is uncertain and must not be reused.
    Poisoned,
    /// Cleanup completed and the executor cannot be reused.
    Closed,
}

/// Downstream handling required after an execution error.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FailureDisposition {
    /// The request was rejected while resident state remained known.
    Rejected,
    /// Cooperative cancellation reached a declared safe boundary.
    Cancelled,
    /// Native state or cleanup is uncertain and the executor must be replaced.
    Poisoned,
}

/// An error whose effect on resident executor state is explicit.
pub trait ClassifiedExecutionError: std::error::Error {
    /// Returns the required downstream handling.
    fn disposition(&self) -> FailureDisposition;
}

/// Exact metadata for one borrowed or writable byte buffer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BufferSpec {
    /// Caller-defined content or allocation identity.
    pub identity: Digest,
    /// Exact accessible byte length.
    pub byte_length: u64,
    /// Bounded mechanical media type.
    pub media_type: String,
}

impl BufferSpec {
    /// Constructs and validates an exact buffer contract.
    ///
    /// # Errors
    ///
    /// Returns an error for a zero length or invalid media type.
    pub fn new(
        identity: Digest,
        byte_length: u64,
        media_type: impl Into<String>,
    ) -> Result<Self, CoreError> {
        let value = Self {
            identity,
            byte_length,
            media_type: media_type.into(),
        };
        value.validate()?;
        Ok(value)
    }

    /// Validates public bounds without touching storage.
    ///
    /// # Errors
    ///
    /// Returns the first invalid field.
    pub fn validate(&self) -> Result<(), CoreError> {
        if self.byte_length == 0 {
            return Err(CoreError::invalid(
                "executor buffer length",
                "must be positive",
            ));
        }
        if self.media_type.is_empty()
            || self.media_type.len() > MAX_MEDIA_TYPE_BYTES
            || self.media_type.contains('\0')
        {
            return Err(CoreError::invalid(
                "executor buffer media type",
                format!("must be non-empty, NUL-free, and at most {MAX_MEDIA_TYPE_BYTES} bytes"),
            ));
        }
        Ok(())
    }

    /// Returns the identity of this exact metadata.
    ///
    /// # Errors
    ///
    /// Returns a validation or serialization error.
    pub fn digest(&self) -> Result<Digest, CoreError> {
        self.validate()?;
        Digest::of_serializable("executor-buffer-spec-v1", self)
    }
}

/// One validated read-only buffer borrowed for a synchronous operation.
#[derive(Debug)]
pub struct InputBuffer<'a> {
    specification: &'a BufferSpec,
    bytes: &'a [u8],
}

impl<'a> InputBuffer<'a> {
    /// Binds exact metadata to readable bytes.
    ///
    /// Content hashing and seal verification remain the caller's
    /// responsibility; this constructor verifies the exact accessible length.
    ///
    /// # Errors
    ///
    /// Returns an error when metadata is invalid or the length differs.
    pub fn new(specification: &'a BufferSpec, bytes: &'a [u8]) -> Result<Self, CoreError> {
        specification.validate()?;
        let actual = u64::try_from(bytes.len())
            .map_err(|_| CoreError::invalid("executor input length", "exceeds u64"))?;
        if actual != specification.byte_length {
            return Err(CoreError::invalid(
                "executor input length",
                "does not match the exact buffer specification",
            ));
        }
        Ok(Self {
            specification,
            bytes,
        })
    }

    /// Returns exact validated metadata.
    pub const fn specification(&self) -> &BufferSpec {
        self.specification
    }

    /// Returns the readable bytes.
    pub const fn bytes(&self) -> &[u8] {
        self.bytes
    }
}

/// One exact caller-owned output allocation for a synchronous operation.
#[derive(Debug)]
pub struct OutputBuffer<'a> {
    specification: &'a BufferSpec,
    bytes: &'a mut [u8],
    written: usize,
}

impl<'a> OutputBuffer<'a> {
    /// Binds exact metadata to writable storage.
    ///
    /// # Errors
    ///
    /// Returns an error when metadata is invalid or storage has another size.
    pub fn new(specification: &'a BufferSpec, bytes: &'a mut [u8]) -> Result<Self, CoreError> {
        specification.validate()?;
        let actual = u64::try_from(bytes.len())
            .map_err(|_| CoreError::invalid("executor output length", "exceeds u64"))?;
        if actual != specification.byte_length {
            return Err(CoreError::invalid(
                "executor output length",
                "does not match the exact buffer specification",
            ));
        }
        Ok(Self {
            specification,
            bytes,
            written: 0,
        })
    }

    /// Returns exact validated metadata.
    pub const fn specification(&self) -> &BufferSpec {
        self.specification
    }

    /// Returns the complete writable allocation.
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        self.bytes
    }

    /// Records the exact initialized prefix.
    ///
    /// # Errors
    ///
    /// Returns an error when `written` exceeds the allocation.
    pub fn set_written(&mut self, written: usize) -> Result<(), CoreError> {
        if written > self.bytes.len() {
            return Err(CoreError::invalid(
                "executor output written length",
                "exceeds the output allocation",
            ));
        }
        self.written = written;
        Ok(())
    }

    /// Returns the exact initialized prefix length.
    pub const fn written(&self) -> usize {
        self.written
    }

    /// Returns the initialized prefix.
    pub fn initialized(&self) -> &[u8] {
        &self.bytes[..self.written]
    }
}

/// Cooperative cancellation state owned by the caller.
pub trait CancellationProbe {
    /// Reports whether cancellation is currently requested.
    fn is_cancelled(&self) -> bool;
}

/// Cancellation probe that never requests a stop.
#[derive(Clone, Copy, Debug, Default)]
pub struct NeverCancel;

impl CancellationProbe for NeverCancel {
    fn is_cancelled(&self) -> bool {
        false
    }
}

/// Evidence returned by explicit session cleanup or close.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CleanupReceipt {
    /// Backend identity whose state was cleared.
    pub backend: Digest,
    /// Session epoch that was invalidated.
    pub cleared_epoch: u64,
    /// Whether backend cleanup was confirmed.
    pub confirmed: bool,
}

impl CleanupReceipt {
    /// Validates cleanup evidence.
    ///
    /// # Errors
    ///
    /// Returns an error when an unconfirmed receipt is presented as success.
    pub fn validate(&self) -> Result<(), CoreError> {
        if !self.confirmed {
            return Err(CoreError::invalid(
                "executor cleanup receipt",
                "cleanup was not confirmed",
            ));
        }
        Ok(())
    }
}

/// Synchronous, single-owner executor over borrowed storage.
///
/// Implementations must not retain input or output slices after a call.
/// Transport, admission, queueing, and resource allocation remain outside this
/// trait.
pub trait LocalExecutor {
    /// Exact backend-specific plan.
    type Plan;
    /// Successful mechanical receipt.
    type Receipt;
    /// Classified executor failure.
    type Error: ClassifiedExecutionError;

    /// Returns current lifecycle state.
    fn state(&self) -> ExecutorState;

    /// Performs an explicit caller-selected warm operation.
    ///
    /// # Errors
    ///
    /// Returns a classified request, cancellation, native, or cleanup error.
    fn warm(
        &mut self,
        plan: &Self::Plan,
        cancellation: &dyn CancellationProbe,
    ) -> Result<Self::Receipt, Self::Error>;

    /// Executes over already verified borrowed inputs and caller-owned outputs.
    ///
    /// # Errors
    ///
    /// Returns a classified request, cancellation, native, or cleanup error.
    fn execute(
        &mut self,
        plan: &Self::Plan,
        inputs: &[InputBuffer<'_>],
        outputs: &mut [OutputBuffer<'_>],
        cancellation: &dyn CancellationProbe,
    ) -> Result<Self::Receipt, Self::Error>;

    /// Clears request-local mutable state and advances the backend epoch.
    ///
    /// # Errors
    ///
    /// Returns a poisoning error when cleanup cannot be confirmed.
    fn clear_session(&mut self) -> Result<CleanupReceipt, Self::Error>;

    /// Consumes the executor and explicitly releases resident state.
    ///
    /// # Errors
    ///
    /// Returns a poisoning error when cleanup cannot be confirmed.
    fn close(self) -> Result<CleanupReceipt, Self::Error>
    where
        Self: Sized;
}

/// Factory that loads one exact worker-local executor from verified artifacts.
pub trait LocalExecutorFactory {
    /// Exact backend-specific load plan.
    type LoadPlan;
    /// Loaded executor type.
    type Executor: LocalExecutor;
    /// Classified load failure.
    type Error: ClassifiedExecutionError;

    /// Loads exact artifacts without selecting transport or resource policy.
    ///
    /// # Errors
    ///
    /// Returns a classified validation, compatibility, native, or cleanup
    /// failure.
    fn load(
        &self,
        plan: &Self::LoadPlan,
        artifacts: &[InputBuffer<'_>],
    ) -> Result<Self::Executor, Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn specification(length: u64) -> BufferSpec {
        BufferSpec::new(
            Digest::of_bytes("test-buffer", b"bytes"),
            length,
            "application/octet-stream",
        )
        .unwrap()
    }

    #[test]
    fn borrowed_buffers_require_exact_lengths() {
        let specification = specification(3);
        assert!(InputBuffer::new(&specification, b"abc").is_ok());
        assert!(InputBuffer::new(&specification, b"ab").is_err());

        let mut exact = [0_u8; 3];
        assert!(OutputBuffer::new(&specification, &mut exact).is_ok());
        let mut short = [0_u8; 2];
        assert!(OutputBuffer::new(&specification, &mut short).is_err());
    }

    #[test]
    fn output_tracks_only_the_initialized_prefix() {
        let specification = specification(4);
        let mut storage = [0_u8; 4];
        let mut output = OutputBuffer::new(&specification, &mut storage).unwrap();
        output.bytes_mut()[..2].copy_from_slice(b"ok");
        output.set_written(2).unwrap();
        assert_eq!(output.initialized(), b"ok");
        assert!(output.set_written(5).is_err());
    }

    #[test]
    fn metadata_round_trips_and_rejects_nul() {
        let specification = specification(7);
        let json = serde_json::to_string(&specification).unwrap();
        assert_eq!(
            serde_json::from_str::<BufferSpec>(&json).unwrap(),
            specification
        );
        assert!(BufferSpec::new(Digest::of_bytes("test-buffer", b"x"), 1, "bad\0type").is_err());
    }
}

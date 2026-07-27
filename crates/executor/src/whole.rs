// SPDX-License-Identifier: MIT OR Apache-2.0

//! Backend-neutral whole-request execution contracts.

use logit_loom_core::{CoreError, Digest, GenerationPlan};
use serde::{Deserialize, Serialize};

use crate::{BufferSpec, CancellationProbe, MAX_MEDIA_TYPE_BYTES};

/// Maximum exact input bytes owned by one whole-generation request.
pub const MAX_WHOLE_GENERATION_INPUT_BYTES: usize = 16 * 1024 * 1024;
/// Maximum output bytes accepted from one whole-generation backend.
pub const MAX_WHOLE_GENERATION_OUTPUT_BYTES: usize = 16 * 1024 * 1024;
/// Maximum opaque evidence objects attached to one terminal result.
pub const MAX_WHOLE_GENERATION_EVIDENCE: usize = 16;
/// Maximum bytes in one opaque backend evidence object.
pub const MAX_WHOLE_GENERATION_EVIDENCE_BYTES: usize = 1024 * 1024;

/// Failure category for a whole-request backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WholeGenerationFailure {
    /// The public request was structurally invalid.
    Rejected,
    /// The request selected mechanics the backend cannot represent exactly.
    Unsupported,
    /// Cooperative cancellation reached a declared backend boundary.
    Cancelled,
    /// A caller or backend deadline expired.
    Deadline,
    /// Backend execution failed while cleanup remained known.
    Backend,
    /// Returned data or receipts violated the selected contract.
    Protocol,
    /// Terminal acknowledgement or cleanup could not be confirmed.
    CommitUncertain,
}

/// A whole-request error with an explicit failure category.
pub trait ClassifiedWholeGenerationError: std::error::Error {
    /// Returns the mechanical failure category.
    fn failure(&self) -> WholeGenerationFailure;
}

#[derive(Serialize)]
struct WholeGenerationRequestIdentity<'a> {
    input_specification: &'a BufferSpec,
    input_content: &'a Digest,
    generation_plan: &'a Digest,
    maximum_output_bytes: u64,
}

/// One validated, owned, backend-neutral generation request.
#[derive(Clone, Debug)]
pub struct WholeGenerationRequest {
    input_specification: BufferSpec,
    input_bytes: Vec<u8>,
    input_content: Digest,
    generation_plan: GenerationPlan,
    generation_plan_identity: Digest,
    maximum_output_bytes: usize,
    identity: Digest,
}

impl WholeGenerationRequest {
    /// Constructs a bounded request and binds its exact bytes and mechanics.
    ///
    /// # Errors
    ///
    /// Returns an error when the buffer metadata, exact length, generation
    /// plan, or input/output bounds are invalid.
    pub fn new(
        input_specification: BufferSpec,
        input_bytes: Vec<u8>,
        generation_plan: GenerationPlan,
        maximum_output_bytes: usize,
    ) -> Result<Self, CoreError> {
        input_specification.validate()?;
        generation_plan.validate()?;
        if input_bytes.is_empty() || input_bytes.len() > MAX_WHOLE_GENERATION_INPUT_BYTES {
            return Err(CoreError::invalid(
                "whole generation input",
                format!("must contain 1..={MAX_WHOLE_GENERATION_INPUT_BYTES} exact bytes"),
            ));
        }
        let input_length = u64::try_from(input_bytes.len())
            .map_err(|_| CoreError::invalid("whole generation input", "length exceeds u64"))?;
        if input_specification.byte_length != input_length {
            return Err(CoreError::invalid(
                "whole generation input",
                "length does not match the buffer specification",
            ));
        }
        if maximum_output_bytes == 0 || maximum_output_bytes > MAX_WHOLE_GENERATION_OUTPUT_BYTES {
            return Err(CoreError::invalid(
                "whole generation output bound",
                format!("must be in 1..={MAX_WHOLE_GENERATION_OUTPUT_BYTES}"),
            ));
        }
        let input_content = Digest::of_bytes("whole-generation-input-content-v1", &input_bytes);
        let generation_plan_identity = generation_plan.digest()?;
        let maximum_output_bytes_u64 = u64::try_from(maximum_output_bytes)
            .map_err(|_| CoreError::invalid("whole generation output bound", "exceeds u64"))?;
        let identity = Digest::of_serializable(
            "whole-generation-request-v1",
            &WholeGenerationRequestIdentity {
                input_specification: &input_specification,
                input_content: &input_content,
                generation_plan: &generation_plan_identity,
                maximum_output_bytes: maximum_output_bytes_u64,
            },
        )?;
        Ok(Self {
            input_specification,
            input_bytes,
            input_content,
            generation_plan,
            generation_plan_identity,
            maximum_output_bytes,
            identity,
        })
    }

    /// Returns the exact request identity.
    pub const fn identity(&self) -> &Digest {
        &self.identity
    }

    /// Returns the exact input buffer contract.
    pub const fn input_specification(&self) -> &BufferSpec {
        &self.input_specification
    }

    /// Returns the exact owned input bytes.
    pub fn input_bytes(&self) -> &[u8] {
        &self.input_bytes
    }

    /// Returns the input content identity.
    pub const fn input_content_identity(&self) -> &Digest {
        &self.input_content
    }

    /// Returns the validated generation mechanics.
    pub const fn generation_plan(&self) -> &GenerationPlan {
        &self.generation_plan
    }

    /// Returns the exact generation-plan identity.
    pub const fn generation_plan_identity(&self) -> &Digest {
        &self.generation_plan_identity
    }

    /// Returns the caller-selected maximum output bytes.
    pub const fn maximum_output_bytes(&self) -> usize {
        self.maximum_output_bytes
    }
}

/// Bounded opaque backend evidence retained with a terminal result.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WholeGenerationEvidence {
    /// Mechanical evidence media type.
    pub media_type: String,
    /// Exact content identity.
    pub identity: Digest,
    /// Exact evidence bytes.
    pub bytes: Vec<u8>,
}

impl WholeGenerationEvidence {
    /// Constructs bounded evidence and computes its content identity.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid media type or byte bound.
    pub fn new(media_type: impl Into<String>, bytes: Vec<u8>) -> Result<Self, CoreError> {
        let media_type = media_type.into();
        validate_media_type("whole generation evidence media type", &media_type)?;
        if bytes.is_empty() || bytes.len() > MAX_WHOLE_GENERATION_EVIDENCE_BYTES {
            return Err(CoreError::invalid(
                "whole generation evidence",
                format!("must contain 1..={MAX_WHOLE_GENERATION_EVIDENCE_BYTES} bytes"),
            ));
        }
        let identity = Digest::of_bytes("whole-generation-evidence-v1", &bytes);
        Ok(Self {
            media_type,
            identity,
            bytes,
        })
    }
}

/// Backend-neutral terminal receipt before caller durability is committed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WholeGenerationReceipt {
    /// Exact originating request.
    pub request: Digest,
    /// Exact input content.
    pub input: Digest,
    /// Exact generation mechanics.
    pub plan: Digest,
    /// Exact output bytes.
    pub output: Digest,
    /// Selected backend contract or implementation identity.
    pub backend: Digest,
    /// Exact opaque evidence identities in declared order.
    pub evidence: Vec<Digest>,
    /// The backend reported a verified terminal result.
    pub terminal_verified: bool,
}

/// Verified output held before terminal acknowledgement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WholeGenerationOutput {
    specification: BufferSpec,
    bytes: Vec<u8>,
    evidence: Vec<WholeGenerationEvidence>,
    receipt: WholeGenerationReceipt,
}

impl WholeGenerationOutput {
    /// Constructs a bounded verified terminal result for one request.
    ///
    /// # Errors
    ///
    /// Returns an error when output or evidence violates public bounds.
    pub fn new(
        request: &WholeGenerationRequest,
        backend: Digest,
        media_type: impl Into<String>,
        bytes: Vec<u8>,
        evidence: Vec<WholeGenerationEvidence>,
    ) -> Result<Self, CoreError> {
        let media_type = media_type.into();
        validate_media_type("whole generation output media type", &media_type)?;
        if bytes.is_empty() || bytes.len() > request.maximum_output_bytes {
            return Err(CoreError::invalid(
                "whole generation output",
                "is empty or exceeds the request output bound",
            ));
        }
        if evidence.len() > MAX_WHOLE_GENERATION_EVIDENCE {
            return Err(CoreError::invalid(
                "whole generation evidence",
                format!("exceeds {MAX_WHOLE_GENERATION_EVIDENCE} objects"),
            ));
        }
        let output_identity = Digest::of_bytes("whole-generation-output-v1", &bytes);
        let byte_length = u64::try_from(bytes.len())
            .map_err(|_| CoreError::invalid("whole generation output", "length exceeds u64"))?;
        let specification = BufferSpec::new(output_identity.clone(), byte_length, media_type)?;
        let receipt = WholeGenerationReceipt {
            request: request.identity.clone(),
            input: request.input_content.clone(),
            plan: request.generation_plan_identity.clone(),
            output: output_identity,
            backend,
            evidence: evidence.iter().map(|item| item.identity.clone()).collect(),
            terminal_verified: true,
        };
        Ok(Self {
            specification,
            bytes,
            evidence,
            receipt,
        })
    }

    /// Returns exact output metadata.
    pub const fn specification(&self) -> &BufferSpec {
        &self.specification
    }

    /// Returns exact output bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns bounded opaque backend evidence.
    pub fn evidence(&self) -> &[WholeGenerationEvidence] {
        &self.evidence
    }

    /// Returns the verified pre-commit receipt.
    pub const fn receipt(&self) -> &WholeGenerationReceipt {
        &self.receipt
    }
}

/// Caller evidence that a verified terminal result was durably persisted.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DurabilityReceipt {
    /// Caller-owned durable record identity.
    pub identity: Digest,
}

impl DurabilityReceipt {
    /// Wraps an exact caller-owned durable record identity.
    #[must_use]
    pub const fn new(identity: Digest) -> Self {
        Self { identity }
    }
}

/// Evidence that terminal acknowledgement completed after caller durability.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WholeGenerationCommitReceipt {
    /// Exact originating request.
    pub request: Digest,
    /// Exact output bytes.
    pub output: Digest,
    /// Caller-owned durable record.
    pub durability: Digest,
    /// Selected backend contract or implementation identity.
    pub backend: Digest,
    /// Whether terminal acknowledgement and required cleanup were confirmed.
    pub acknowledged: bool,
}

impl WholeGenerationCommitReceipt {
    /// Constructs confirmed commit evidence from verified output.
    #[must_use]
    pub fn confirmed(output: &WholeGenerationOutput, durability: &DurabilityReceipt) -> Self {
        Self {
            request: output.receipt.request.clone(),
            output: output.receipt.output.clone(),
            durability: durability.identity.clone(),
            backend: output.receipt.backend.clone(),
            acknowledged: true,
        }
    }

    /// Validates confirmed acknowledgement evidence.
    ///
    /// # Errors
    ///
    /// Returns an error when acknowledgement was not confirmed.
    pub fn validate(&self) -> Result<(), CoreError> {
        if !self.acknowledged {
            return Err(CoreError::invalid(
                "whole generation commit receipt",
                "terminal acknowledgement was not confirmed",
            ));
        }
        Ok(())
    }
}

/// A verified terminal result awaiting caller durability and acknowledgement.
///
/// Implementations must not acknowledge backend success before `commit`.
/// `abort` and `Drop` must make a best effort to release request-local state
/// without representing the operation as committed.
pub trait PendingGeneration: Send {
    /// Classified backend failure.
    type Error: ClassifiedWholeGenerationError;

    /// Returns verified output before acknowledgement.
    fn output(&self) -> &WholeGenerationOutput;

    /// Acknowledges the backend terminal result after caller durability.
    ///
    /// # Errors
    ///
    /// Returns a classified error when acknowledgement or required cleanup
    /// cannot be confirmed.
    fn commit(
        self,
        durability: DurabilityReceipt,
    ) -> Result<WholeGenerationCommitReceipt, Self::Error>
    where
        Self: Sized;

    /// Aborts without acknowledging successful completion.
    ///
    /// # Errors
    ///
    /// Returns a classified error when cancellation or cleanup is uncertain.
    fn abort(self) -> Result<(), Self::Error>
    where
        Self: Sized;
}

/// Synchronous whole-request generation backend.
///
/// Transport, admission, scheduling, and resource policy may exist behind
/// this boundary. The contract exposes no per-token callback, transform,
/// observer, or checkpoint semantics.
pub trait WholeGenerationBackend {
    /// Verified result awaiting caller durability.
    type Pending: PendingGeneration<Error = Self::Error>;
    /// Classified backend failure.
    type Error: ClassifiedWholeGenerationError;

    /// Runs one validated request through a verified terminal result.
    ///
    /// # Errors
    ///
    /// Returns a classified validation, support, cancellation, deadline,
    /// backend, or protocol error.
    fn generate(
        &mut self,
        request: WholeGenerationRequest,
        cancellation: &dyn CancellationProbe,
    ) -> Result<Self::Pending, Self::Error>;
}

fn validate_media_type(field: &'static str, media_type: &str) -> Result<(), CoreError> {
    if media_type.is_empty() || media_type.len() > MAX_MEDIA_TYPE_BYTES || media_type.contains('\0')
    {
        return Err(CoreError::invalid(
            field,
            format!("must be non-empty, NUL-free, and at most {MAX_MEDIA_TYPE_BYTES} bytes"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        error::Error,
        fmt,
        sync::{
            Arc, Mutex,
            atomic::{AtomicBool, Ordering},
        },
    };

    use logit_loom_core::SamplingPlan;

    use super::*;

    fn request(input: &[u8]) -> WholeGenerationRequest {
        WholeGenerationRequest::new(
            BufferSpec::new(
                Digest::of_bytes("caller-input", input),
                u64::try_from(input.len()).unwrap(),
                "application/json",
            )
            .unwrap(),
            input.to_vec(),
            GenerationPlan {
                sampling: SamplingPlan {
                    top_k: 0,
                    ..SamplingPlan::default()
                },
                max_tokens: 8,
                biases: Vec::new(),
                grammar: None,
                stops: Vec::new(),
            },
            128,
        )
        .unwrap()
    }

    #[test]
    fn request_identity_binds_exact_bytes_plan_and_output_bound() {
        let original = request(b"{\"value\":1}");
        let changed = request(b"{\"value\":2}");
        assert_ne!(original.identity(), changed.identity());

        let mut plan = original.generation_plan().clone();
        plan.max_tokens += 1;
        let changed_plan = WholeGenerationRequest::new(
            original.input_specification().clone(),
            original.input_bytes().to_vec(),
            plan,
            original.maximum_output_bytes(),
        )
        .unwrap();
        assert_ne!(original.identity(), changed_plan.identity());

        let changed_bound = WholeGenerationRequest::new(
            original.input_specification().clone(),
            original.input_bytes().to_vec(),
            original.generation_plan().clone(),
            original.maximum_output_bytes() + 1,
        )
        .unwrap();
        assert_ne!(original.identity(), changed_bound.identity());
    }

    #[test]
    fn request_rejects_mismatched_and_unbounded_storage() {
        let spec = BufferSpec::new(
            Digest::of_bytes("caller-input", b"x"),
            2,
            "application/json",
        )
        .unwrap();
        assert!(
            WholeGenerationRequest::new(
                spec,
                b"x".to_vec(),
                request(b"x").generation_plan().clone(),
                1,
            )
            .is_err()
        );
        let oversized = vec![0_u8; MAX_WHOLE_GENERATION_INPUT_BYTES + 1];
        let oversized_spec = BufferSpec::new(
            Digest::of_bytes("caller-input", &oversized),
            u64::try_from(oversized.len()).unwrap(),
            "application/octet-stream",
        )
        .unwrap();
        assert!(
            WholeGenerationRequest::new(
                oversized_spec,
                oversized,
                request(b"x").generation_plan().clone(),
                1,
            )
            .is_err()
        );
    }

    #[test]
    fn output_binds_bounded_bytes_and_evidence() {
        let request = request(b"input");
        let evidence =
            WholeGenerationEvidence::new("application/vnd.example.receipt+json", b"{}".to_vec())
                .unwrap();
        let output = WholeGenerationOutput::new(
            &request,
            Digest::of_bytes("backend", b"fake"),
            "application/json",
            b"{\"text\":\"ok\"}".to_vec(),
            vec![evidence.clone()],
        )
        .unwrap();
        assert_eq!(output.evidence(), &[evidence]);
        assert_eq!(output.specification().identity, output.receipt().output);
        assert!(output.receipt().terminal_verified);
    }

    #[derive(Debug)]
    struct FakeError(WholeGenerationFailure);

    impl fmt::Display for FakeError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "fake {:?}", self.0)
        }
    }

    impl Error for FakeError {}

    impl ClassifiedWholeGenerationError for FakeError {
        fn failure(&self) -> WholeGenerationFailure {
            self.0
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum FakeState {
        Pending,
        Committed,
        Aborted,
        Dropped,
    }

    struct FakePending {
        output: WholeGenerationOutput,
        state: Arc<Mutex<FakeState>>,
    }

    impl PendingGeneration for FakePending {
        type Error = FakeError;

        fn output(&self) -> &WholeGenerationOutput {
            &self.output
        }

        fn commit(
            mut self,
            durability: DurabilityReceipt,
        ) -> Result<WholeGenerationCommitReceipt, Self::Error> {
            *self.state.lock().unwrap() = FakeState::Committed;
            let receipt = WholeGenerationCommitReceipt::confirmed(&self.output, &durability);
            self.state = Arc::new(Mutex::new(FakeState::Committed));
            Ok(receipt)
        }

        fn abort(mut self) -> Result<(), Self::Error> {
            *self.state.lock().unwrap() = FakeState::Aborted;
            self.state = Arc::new(Mutex::new(FakeState::Aborted));
            Ok(())
        }
    }

    impl Drop for FakePending {
        fn drop(&mut self) {
            let mut state = self.state.lock().unwrap();
            if *state == FakeState::Pending {
                *state = FakeState::Dropped;
            }
        }
    }

    struct AtomicCancellation(AtomicBool);

    impl CancellationProbe for AtomicCancellation {
        fn is_cancelled(&self) -> bool {
            self.0.load(Ordering::Acquire)
        }
    }

    struct FakeBackend {
        state: Arc<Mutex<FakeState>>,
    }

    impl WholeGenerationBackend for FakeBackend {
        type Pending = FakePending;
        type Error = FakeError;

        fn generate(
            &mut self,
            request: WholeGenerationRequest,
            cancellation: &dyn CancellationProbe,
        ) -> Result<Self::Pending, Self::Error> {
            if cancellation.is_cancelled() {
                return Err(FakeError(WholeGenerationFailure::Cancelled));
            }
            let output = WholeGenerationOutput::new(
                &request,
                Digest::of_bytes("backend", b"fake"),
                "text/plain",
                b"ok".to_vec(),
                Vec::new(),
            )
            .unwrap();
            Ok(FakePending {
                output,
                state: Arc::clone(&self.state),
            })
        }
    }

    #[test]
    fn pending_result_commits_only_after_durability() {
        let state = Arc::new(Mutex::new(FakeState::Pending));
        let mut backend = FakeBackend {
            state: Arc::clone(&state),
        };
        let pending = backend
            .generate(request(b"input"), &crate::NeverCancel)
            .unwrap();
        assert_eq!(*state.lock().unwrap(), FakeState::Pending);
        let durability = DurabilityReceipt::new(Digest::of_bytes("durability", b"event-record"));
        let commit = pending.commit(durability.clone()).unwrap();
        assert_eq!(*state.lock().unwrap(), FakeState::Committed);
        assert_eq!(commit.durability, durability.identity);
        commit.validate().unwrap();
    }

    #[test]
    fn dropped_result_is_not_committed_and_cancellation_is_classified() {
        let state = Arc::new(Mutex::new(FakeState::Pending));
        let mut backend = FakeBackend {
            state: Arc::clone(&state),
        };
        drop(
            backend
                .generate(request(b"input"), &crate::NeverCancel)
                .unwrap(),
        );
        assert_eq!(*state.lock().unwrap(), FakeState::Dropped);

        let cancellation = AtomicCancellation(AtomicBool::new(true));
        let Err(error) = backend.generate(request(b"input"), &cancellation) else {
            panic!("cancelled request unexpectedly succeeded");
        };
        assert_eq!(error.failure(), WholeGenerationFailure::Cancelled);
    }
}

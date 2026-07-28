// SPDX-License-Identifier: MIT OR Apache-2.0

//! Target-authoritative MTP and EAGLE-3 generation.

use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;

use llama_cpp_4::context::params::LlamaContextType;
use llama_cpp_4::context::{LlamaContext, TensorTransactions, TransactionalTensorCapture};
use llama_cpp_4::eagle::{Eagle3Session, Eagle3SessionConfig};
use llama_cpp_4::llama_batch::LlamaBatch;
use llama_cpp_4::mtp::{MtpSession, MtpSessionConfig};
use llama_cpp_4::sampling::LlamaSampler;
use llama_cpp_4::token::LlamaToken;
use llama_cpp_4::token::data::LlamaTokenData;
use llama_cpp_4::token::data_array::LlamaTokenDataArray;
use logit_loom::{
    ActivationPhaseV1, ActivationTelemetryDispositionV1, ControlFlow, Digest, GenerationFinish,
    GenerationPlan, GenerationReceipt, ObservedToken, ObserverSet, Pipeline, PrefillFinish,
    PrefillMonitor, PrefillProgress, SpeculationActivationPolicyV1, SpeculationBoundaryReceiptV1,
    SpeculationPlanV1, SpeculationReceiptV1, SpeculativeCheckpointReceiptV1, SteeringReceipt,
    TextMechanicsPlanV2, TextSpeculativeMechanismV1, TokenId,
};

use crate::{
    ActivationCaptureOutput, ActivationConfiguration, ActivationProgramOutput, Error,
    GenerationOutput, LLAMA_CPP_BINDING_SOURCE_REVISION, LLAMA_CPP_BINDING_VERSION,
    LLAMA_CPP_REVISION, Model, PrefillOutput, Runtime, Session, SessionOptions, StateSnapshot,
    activation::ActivationController,
    error::native,
    sampler::build_sampler,
    steering::{
        ControlVector, LoraApplication, execute_with_steering, validate_steering_resources,
    },
};

const SPECULATION_IMPLEMENTATION_DOMAIN: &str = "llamacpp-speculation-implementation-v1";
const SPECULATIVE_VOCABULARY_CHECK_START: i32 = 5;
const MAX_SPECULATIVE_VOCABULARY_DIFFERENCE: u32 = 128;

/// Returns the exact native and safe-wrapper implementation identity.
///
/// This identity changes when the binding version or source, pinned llama.cpp
/// revision, or Logit Loom lowering profile changes.
#[must_use]
pub fn speculation_implementation_identity() -> Digest {
    Digest::of_bytes(
        SPECULATION_IMPLEMENTATION_DOMAIN,
        format!(
            "{LLAMA_CPP_BINDING_VERSION}|{LLAMA_CPP_BINDING_SOURCE_REVISION}|{LLAMA_CPP_REVISION}|target-authoritative-v1"
        )
        .as_bytes(),
    )
}

/// Target and draft context-allocation options for one speculative operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpeculativeSessionOptions {
    /// Target-model context options.
    pub target: SessionOptions,
    /// Draft-model context options.
    pub draft: SessionOptions,
}

impl Default for SpeculativeSessionOptions {
    fn default() -> Self {
        let options = SessionOptions::default();
        Self {
            target: options,
            draft: options,
        }
    }
}

/// Explicit activation runtimes supplied to one speculative operation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpeculativeActivations {
    target: Option<ActivationConfiguration>,
    draft: Option<ActivationConfiguration>,
}

impl SpeculativeActivations {
    /// Selects no activation runtime.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            target: None,
            draft: None,
        }
    }

    /// Selects one target-only activation runtime.
    #[must_use]
    pub fn target_only(target: ActivationConfiguration) -> Self {
        Self {
            target: Some(target),
            draft: None,
        }
    }

    /// Selects independently validated target and draft activation runtimes.
    #[must_use]
    pub fn separate(target: ActivationConfiguration, draft: ActivationConfiguration) -> Self {
        Self {
            target: Some(target),
            draft: Some(draft),
        }
    }
}

/// Borrowed mechanics and owned runtime configuration for one operation.
pub struct SpeculativeRequest<'a> {
    prompt: &'a [TokenId],
    generation: &'a GenerationPlan,
    speculation: &'a SpeculationPlanV1,
    options: SpeculativeSessionOptions,
    activations: SpeculativeActivations,
    target_loras: Vec<LoraApplication<'a>>,
    target_control_vector: Option<&'a ControlVector>,
    pipeline: Option<&'a mut Pipeline>,
    observers: Option<&'a mut ObserverSet>,
}

impl<'a> SpeculativeRequest<'a> {
    /// Constructs a request with default context options and no callbacks.
    #[must_use]
    pub fn new(
        prompt: &'a [TokenId],
        generation: &'a GenerationPlan,
        speculation: &'a SpeculationPlanV1,
    ) -> Self {
        Self {
            prompt,
            generation,
            speculation,
            options: SpeculativeSessionOptions::default(),
            activations: SpeculativeActivations::none(),
            target_loras: Vec::new(),
            target_control_vector: None,
            pipeline: None,
            observers: None,
        }
    }

    /// Sets target and draft context-allocation options.
    #[must_use]
    pub const fn with_options(mut self, options: SpeculativeSessionOptions) -> Self {
        self.options = options;
        self
    }

    /// Installs activation runtimes matching the speculation policy.
    #[must_use]
    pub fn with_activations(mut self, activations: SpeculativeActivations) -> Self {
        self.activations = activations;
        self
    }

    /// Applies exact ordered steering to the target context for the complete
    /// operation.
    #[must_use]
    pub fn with_target_steering(
        mut self,
        loras: Vec<LoraApplication<'a>>,
        control_vector: Option<&'a ControlVector>,
    ) -> Self {
        self.target_loras = loras;
        self.target_control_vector = control_vector;
        self
    }

    /// Installs one ordered logit-transform pipeline.
    #[must_use]
    pub fn with_pipeline(mut self, pipeline: &'a mut Pipeline) -> Self {
        self.pipeline = Some(pipeline);
        self
    }

    /// Installs one ordered admitted-token observer set.
    #[must_use]
    pub fn with_observers(mut self, observers: &'a mut ObserverSet) -> Self {
        self.observers = Some(observers);
        self
    }
}

/// Activation evidence produced by one context in a speculative operation.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeculativeActivationOutput {
    /// Final admitted/rejected capture records and aggregate capture receipts.
    pub captures: ActivationCaptureOutput,
    /// Final per-row write-back records and aggregate program receipt.
    pub program: ActivationProgramOutput,
}

/// Complete successful output of one target-authoritative operation.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeculativeGenerationOutput {
    /// Fresh-prompt prefill accounting. Restored continuations have no prefill.
    pub prefill: Option<PrefillOutput>,
    /// Causally admitted generated tokens, bytes, and generation receipt.
    pub generation: GenerationOutput,
    /// Exact completed proposal boundaries in execution order.
    pub boundaries: Vec<SpeculationBoundaryReceiptV1>,
    /// Aggregate proposal and target-acceptance accounting.
    pub speculation: SpeculationReceiptV1,
    /// Target activation evidence, when selected.
    pub target_activation: Option<SpeculativeActivationOutput>,
    /// Independently selected draft activation evidence.
    pub draft_activation: Option<SpeculativeActivationOutput>,
    /// Successful target-steering applications in exact native order.
    pub steering_applied: Vec<SteeringReceipt>,
    /// Successful target-steering cleanup in exact reverse native order.
    pub steering_cleared: Vec<SteeringReceipt>,
}

/// Successful cooperative stop at a complete speculative prefill boundary.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SpeculativePrefillStoppedOutput {
    pub(crate) prefill: PrefillOutput,
    pub(crate) target_activation: Option<SpeculativeActivationOutput>,
    pub(crate) draft_activation: Option<SpeculativeActivationOutput>,
    pub(crate) steering_applied: Vec<SteeringReceipt>,
    pub(crate) steering_cleared: Vec<SteeringReceipt>,
}

/// Aggregate-plan lineage selected for one quiescent speculative checkpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeculativeCheckpointRequest {
    mechanics: TextMechanicsPlanV2,
}

impl SpeculativeCheckpointRequest {
    /// Selects the complete aggregate text-mechanics plan for the captured
    /// boundary.
    #[must_use]
    pub const fn new(mechanics: TextMechanicsPlanV2) -> Self {
        Self { mechanics }
    }
}

#[derive(Clone, Debug)]
struct ValidatedCheckpointRequest {
    mechanics: Digest,
    plan: TextMechanicsPlanV2,
    parent: Option<Digest>,
}

/// Opaque process-local continuation state for one quiescent speculative
/// boundary.
///
/// Native target and draft context bytes are retained with an opaque native
/// target-sampler clone. The clone has no portable serialization contract, so
/// this value deliberately remains thread-affine and cannot be reconstructed
/// from its serializable receipt alone.
pub struct SpeculativeStateSnapshot {
    target: StateSnapshot,
    draft: StateSnapshot,
    implementation_state: Vec<u8>,
    sampler: LlamaSampler,
    generation: Digest,
    stop_tail: Vec<u8>,
    activations: SpeculativeActivations,
    options: SpeculativeSessionOptions,
    mechanics: TextMechanicsPlanV2,
    receipt: SpeculativeCheckpointReceiptV1,
    thread_affinity: PhantomData<Rc<()>>,
}

impl std::fmt::Debug for SpeculativeStateSnapshot {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SpeculativeStateSnapshot")
            .field("receipt", &self.receipt)
            .field("target_state_bytes", &self.target.receipt().state_bytes)
            .field("draft_state_bytes", &self.draft.receipt().state_bytes)
            .finish_non_exhaustive()
    }
}

impl SpeculativeStateSnapshot {
    /// Returns serializable, content-bound checkpoint accounting.
    pub const fn receipt(&self) -> &SpeculativeCheckpointReceiptV1 {
        &self.receipt
    }

    /// Returns the complete mechanics used to build this causal state.
    pub const fn mechanics(&self) -> &TextMechanicsPlanV2 {
        &self.mechanics
    }

    /// Returns the opaque target-context state.
    pub const fn target_state(&self) -> &StateSnapshot {
        &self.target
    }

    /// Returns the opaque draft-context state.
    pub const fn draft_state(&self) -> &StateSnapshot {
        &self.draft
    }

    /// Returns the opaque native speculative implementation state.
    pub fn implementation_state(&self) -> &[u8] {
        &self.implementation_state
    }

    /// Clones every opaque continuation component for an independent branch.
    ///
    /// # Errors
    ///
    /// Returns an error when the pinned native sampler cannot produce an
    /// independent in-process clone.
    pub fn try_clone(&self) -> Result<Self, Error> {
        Ok(Self {
            target: self.target.clone(),
            draft: self.draft.clone(),
            implementation_state: self.implementation_state.clone(),
            sampler: clone_sampler(&self.sampler)?,
            generation: self.generation.clone(),
            stop_tail: self.stop_tail.clone(),
            activations: self.activations.clone(),
            options: self.options,
            mechanics: self.mechanics.clone(),
            receipt: self.receipt.clone(),
            thread_affinity: PhantomData,
        })
    }
}

/// Successful generation plus a reusable quiescent continuation checkpoint.
#[derive(Debug)]
pub struct SpeculativeCheckpointOutput {
    /// Target-authoritative generation and exact operation evidence.
    pub generation: SpeculativeGenerationOutput,
    /// Opaque in-process state at the completed operation boundary.
    pub checkpoint: SpeculativeStateSnapshot,
}

/// Borrowed mechanics for continuation from a speculative checkpoint.
pub struct SpeculativeContinuationRequest<'a> {
    generation: &'a GenerationPlan,
    speculation: &'a SpeculationPlanV1,
    options: SpeculativeSessionOptions,
    target_loras: Vec<LoraApplication<'a>>,
    target_control_vector: Option<&'a ControlVector>,
    pipeline: Option<&'a mut Pipeline>,
    observers: Option<&'a mut ObserverSet>,
}

impl<'a> SpeculativeContinuationRequest<'a> {
    /// Constructs a continuation request with default context options and no
    /// callbacks.
    #[must_use]
    pub fn new(generation: &'a GenerationPlan, speculation: &'a SpeculationPlanV1) -> Self {
        Self {
            generation,
            speculation,
            options: SpeculativeSessionOptions::default(),
            target_loras: Vec::new(),
            target_control_vector: None,
            pipeline: None,
            observers: None,
        }
    }

    /// Sets target and draft context-allocation options.
    #[must_use]
    pub const fn with_options(mut self, options: SpeculativeSessionOptions) -> Self {
        self.options = options;
        self
    }

    /// Reapplies exact ordered target steering after checkpoint restoration
    /// and clears it before the next quiescent boundary is captured.
    #[must_use]
    pub fn with_target_steering(
        mut self,
        loras: Vec<LoraApplication<'a>>,
        control_vector: Option<&'a ControlVector>,
    ) -> Self {
        self.target_loras = loras;
        self.target_control_vector = control_vector;
        self
    }

    /// Installs one ordered logit-transform pipeline.
    #[must_use]
    pub fn with_pipeline(mut self, pipeline: &'a mut Pipeline) -> Self {
        self.pipeline = Some(pipeline);
        self
    }

    /// Installs one ordered admitted-token observer set.
    #[must_use]
    pub fn with_observers(mut self, observers: &'a mut ObserverSet) -> Self {
        self.observers = Some(observers);
        self
    }
}

/// Runs one bounded MTP or EAGLE-3 operation.
///
/// The target model is the sole causal authority. Draft tokens become visible
/// to transforms, observers, output, and causal lineage only after exact
/// target sampling accepts the proposal prefix. There is no fallback to
/// ordinary generation.
///
/// # Errors
///
/// Returns before context allocation for any model, topology,
/// implementation, activation-policy, sequence, capacity, or bound mismatch.
/// During execution, any callback, native decode, rollback, process, or
/// acceptance failure aborts the operation.
pub fn generate_speculative(
    runtime: &Runtime,
    target_model: &Model,
    draft_model: &Model,
    request: SpeculativeRequest<'_>,
) -> Result<SpeculativeGenerationOutput, Error> {
    match generate_speculative_inner(runtime, target_model, draft_model, request, None, None)? {
        SpeculativeExecutionOutcome::Generated(execution) => Ok(execution.generation),
        SpeculativeExecutionOutcome::PrefillStopped(_) => Err(Error::Poisoned(
            "unmonitored speculative generation stopped during prefill".to_owned(),
        )),
    }
}

/// Runs one bounded speculative operation and captures complete quiescent
/// continuation state.
///
/// The returned checkpoint owns opaque target, draft, implementation, grammar,
/// sampling, stop-prefix, activation, and causal-lineage state. It is reusable
/// for independent in-process branches through
/// [`resume_speculative_checkpointed`].
///
/// # Errors
///
/// Returns the same validation and execution errors as
/// [`generate_speculative`], plus exact state-capture or sampler-clone errors.
pub fn generate_speculative_checkpointed(
    runtime: &Runtime,
    target_model: &Model,
    draft_model: &Model,
    request: SpeculativeRequest<'_>,
    checkpoint: &SpeculativeCheckpointRequest,
) -> Result<SpeculativeCheckpointOutput, Error> {
    let execution = generate_speculative_inner(
        runtime,
        target_model,
        draft_model,
        request,
        Some(checkpoint),
        None,
    )?;
    let SpeculativeExecutionOutcome::Generated(execution) = execution else {
        return Err(Error::Poisoned(
            "unmonitored speculative generation stopped during prefill".to_owned(),
        ));
    };
    Ok(SpeculativeCheckpointOutput {
        generation: execution.generation,
        checkpoint: execution.checkpoint.ok_or_else(|| {
            Error::Poisoned("checkpoint capture completed without checkpoint state".to_owned())
        })?,
    })
}

pub(crate) fn generate_speculative_controlled(
    runtime: &Runtime,
    target_model: &Model,
    draft_model: &Model,
    request: SpeculativeRequest<'_>,
    checkpoint: Option<&SpeculativeCheckpointRequest>,
    monitor: &mut PrefillMonitor,
) -> Result<SpeculativeExecutionOutcome, Error> {
    generate_speculative_inner(
        runtime,
        target_model,
        draft_model,
        request,
        checkpoint,
        Some(monitor),
    )
}

/// Continues one compatible quiescent checkpoint without capturing a
/// successor boundary.
///
/// The input checkpoint remains reusable because its opaque target sampler is
/// cloned before execution.
///
/// # Errors
///
/// Returns before context allocation for any plan, model, topology,
/// activation, option, state, or capacity mismatch. Restore, sampler cloning,
/// and generation also fail closed.
pub fn resume_speculative(
    runtime: &Runtime,
    target_model: &Model,
    draft_model: &Model,
    checkpoint: &SpeculativeStateSnapshot,
    request: SpeculativeContinuationRequest<'_>,
) -> Result<SpeculativeGenerationOutput, Error> {
    Ok(resume_speculative_inner(
        runtime,
        target_model,
        draft_model,
        checkpoint,
        request,
        None,
    )?
    .generation)
}

/// Continues one compatible quiescent checkpoint and captures the next
/// quiescent boundary.
///
/// The input checkpoint remains reusable. The opaque native sampler is cloned
/// before execution, so multiple calls create independent branches from the
/// same exact parent.
///
/// # Errors
///
/// Returns before context allocation for any plan, model, topology,
/// activation, option, parent-lineage, state, or capacity mismatch. Native
/// restore, sampler cloning, generation, and next-checkpoint capture also fail
/// closed.
pub fn resume_speculative_checkpointed(
    runtime: &Runtime,
    target_model: &Model,
    draft_model: &Model,
    checkpoint: &SpeculativeStateSnapshot,
    request: SpeculativeContinuationRequest<'_>,
    next_checkpoint: &SpeculativeCheckpointRequest,
) -> Result<SpeculativeCheckpointOutput, Error> {
    let execution = resume_speculative_inner(
        runtime,
        target_model,
        draft_model,
        checkpoint,
        request,
        Some(next_checkpoint),
    )?;
    Ok(SpeculativeCheckpointOutput {
        generation: execution.generation,
        checkpoint: execution.checkpoint.ok_or_else(|| {
            Error::Poisoned("continuation completed without checkpoint state".to_owned())
        })?,
    })
}

#[allow(
    clippy::too_many_lines,
    reason = "restore order and the two native mechanisms remain adjacent for state-audit review"
)]
fn resume_speculative_inner(
    runtime: &Runtime,
    target_model: &Model,
    draft_model: &Model,
    checkpoint: &SpeculativeStateSnapshot,
    mut request: SpeculativeContinuationRequest<'_>,
    next_checkpoint: Option<&SpeculativeCheckpointRequest>,
) -> Result<SpeculativeExecution, Error> {
    let parent =
        validate_continuation_request(runtime, target_model, draft_model, checkpoint, &request)?;
    let next_checkpoint = next_checkpoint
        .map(|next_checkpoint| {
            validate_checkpoint_request(
                next_checkpoint,
                target_model,
                draft_model,
                request.generation,
                request.speculation,
                &checkpoint.activations,
                &request.target_loras,
                request.target_control_vector,
                request.pipeline.as_deref(),
                request.observers.as_deref(),
                Some(&parent),
            )
        })
        .transpose()?;
    let sampler = clone_sampler(&checkpoint.sampler)?;
    let plan = request.speculation;
    let recurrent_slots = plan.maximum_draft_tokens;
    let checkpoint_activations = checkpoint.activations.clone();
    let (target_activation, draft_activation) =
        validate_activations(plan, checkpoint.activations.clone())?;
    let mut target = Session::new_speculative(
        target_model,
        runtime,
        request.options.target,
        LlamaContextType::Default,
        recurrent_slots,
        target_activation,
    )?;
    let draft_context_type = match plan.mechanism {
        TextSpeculativeMechanismV1::Mtp => LlamaContextType::Mtp,
        TextSpeculativeMechanismV1::Eagle3 => LlamaContextType::Default,
    };
    let mut draft = Session::new_speculative(
        draft_model,
        runtime,
        request.options.draft,
        draft_context_type,
        recurrent_slots,
        draft_activation,
    )?;
    target.restore_envelope_state(&checkpoint.target, Some(ActivationPhaseV1::Verification))?;
    draft.restore_envelope_state(&checkpoint.draft, None)?;

    let maximum_draft = i32::try_from(plan.maximum_draft_tokens)
        .map_err(|_| Error::Invalid("maximum draft tokens exceed i32".to_owned()))?;
    let minimum_draft = i32::try_from(plan.minimum_draft_tokens)
        .map_err(|_| Error::Invalid("minimum draft tokens exceed i32".to_owned()))?;
    let probability_floor = plan.probability_floor()?;
    let loras = std::mem::take(&mut request.target_loras);
    let control_vector = request.target_control_vector;
    let ((mut run, implementation_state), steering_applied, steering_cleared) =
        execute_with_steering(&mut target, loras, control_vector, |target| {
            run_restored_native(
                target,
                &mut draft,
                checkpoint,
                sampler,
                request.generation,
                plan,
                request.pipeline.as_deref_mut(),
                request.observers.as_deref_mut(),
                next_checkpoint.is_some(),
                maximum_draft,
                minimum_draft,
                probability_floor,
            )
        })?;
    run.output.steering_applied = steering_applied;
    run.output.steering_cleared = steering_cleared;
    finish_speculative_execution(
        &mut target,
        &mut draft,
        run,
        implementation_state,
        next_checkpoint,
        checkpoint_activations,
        request.generation,
        plan,
        checkpoint.receipt.completed_boundaries,
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "native continuation lowering receives every prevalidated mechanic explicitly"
)]
fn run_restored_native<'model>(
    target: &mut Session<'model>,
    draft: &mut Session<'model>,
    checkpoint: &SpeculativeStateSnapshot,
    sampler: LlamaSampler,
    generation: &GenerationPlan,
    plan: &SpeculationPlanV1,
    pipeline: Option<&mut Pipeline>,
    observers: Option<&mut ObserverSet>,
    capture_state: bool,
    maximum_draft: i32,
    minimum_draft: i32,
    probability_floor: f32,
) -> Result<(SpeculativeRunState, Option<Vec<u8>>), Error> {
    target.refresh_restored_logits_with_active_steering(Some(ActivationPhaseV1::Verification))?;
    let start = SpeculativeRunStart::Restored {
        sampler,
        stop_tail: checkpoint.stop_tail.clone(),
    };
    match plan.mechanism {
        TextSpeculativeMechanismV1::Mtp => {
            let config = MtpSessionConfig::new(plan.sequences, maximum_draft)
                .with_n_min(minimum_draft)
                .with_p_min(probability_floor);
            let (target_context, target_side) = speculative_parts(target);
            let (draft_context, draft_side) = speculative_parts(draft);
            let mut backend = MtpSession::new_with_config(target_context, draft_context, config)
                .map_err(native)?;
            backend.restore_implementation_state(&checkpoint.implementation_state)?;
            let run = run_backend(
                &mut backend,
                target_side,
                draft_side,
                start,
                generation,
                plan,
                None,
                pipeline,
                observers,
            )?;
            let SpeculativeRunOutcome::Generated(run) = run else {
                return Err(Error::Poisoned(
                    "restored speculation stopped during nonexistent prefill".to_owned(),
                ));
            };
            let implementation_state = capture_state
                .then(|| backend.capture_implementation_state())
                .transpose()?;
            Ok((*run, implementation_state))
        }
        TextSpeculativeMechanismV1::Eagle3 => {
            let config = Eagle3SessionConfig::new(plan.sequences, maximum_draft)
                .with_n_min(minimum_draft)
                .with_p_min(probability_floor);
            let (target_context, target_side) = speculative_parts(target);
            let (draft_context, draft_side) = speculative_parts(draft);
            let mut backend = Eagle3Session::new_with_config(target_context, draft_context, config)
                .map_err(native)?;
            backend.restore_implementation_state(&checkpoint.implementation_state)?;
            let run = run_backend(
                &mut backend,
                target_side,
                draft_side,
                start,
                generation,
                plan,
                None,
                pipeline,
                observers,
            )?;
            let SpeculativeRunOutcome::Generated(run) = run else {
                return Err(Error::Poisoned(
                    "restored speculation stopped during nonexistent prefill".to_owned(),
                ));
            };
            let implementation_state = capture_state
                .then(|| backend.capture_implementation_state())
                .transpose()?;
            Ok((*run, implementation_state))
        }
    }
}

pub(crate) struct SpeculativeExecution {
    pub(crate) generation: SpeculativeGenerationOutput,
    pub(crate) checkpoint: Option<SpeculativeStateSnapshot>,
}

pub(crate) enum SpeculativeExecutionOutcome {
    Generated(Box<SpeculativeExecution>),
    PrefillStopped(Box<SpeculativePrefillStoppedOutput>),
}

#[allow(
    clippy::too_many_lines,
    reason = "fresh execution and state capture keep both native mechanisms structurally identical"
)]
fn generate_speculative_inner(
    runtime: &Runtime,
    target_model: &Model,
    draft_model: &Model,
    mut request: SpeculativeRequest<'_>,
    checkpoint: Option<&SpeculativeCheckpointRequest>,
    prefill_monitor: Option<&mut PrefillMonitor>,
) -> Result<SpeculativeExecutionOutcome, Error> {
    validate_request(runtime, target_model, draft_model, &request)?;
    let plan = request.speculation;
    let checkpoint = checkpoint
        .map(|checkpoint| {
            validate_checkpoint_request(
                checkpoint,
                target_model,
                draft_model,
                request.generation,
                plan,
                &request.activations,
                &request.target_loras,
                request.target_control_vector,
                request.pipeline.as_deref(),
                request.observers.as_deref(),
                None,
            )
        })
        .transpose()?;
    let recurrent_slots = plan.maximum_draft_tokens;
    let checkpoint_activations = request.activations.clone();
    let (target_activation, draft_activation) = validate_activations(plan, request.activations)?;
    let mut target = Session::new_speculative(
        target_model,
        runtime,
        request.options.target,
        LlamaContextType::Default,
        recurrent_slots,
        target_activation,
    )?;
    let draft_context_type = match plan.mechanism {
        TextSpeculativeMechanismV1::Mtp => LlamaContextType::Mtp,
        TextSpeculativeMechanismV1::Eagle3 => LlamaContextType::Default,
    };
    let mut draft = Session::new_speculative(
        draft_model,
        runtime,
        request.options.draft,
        draft_context_type,
        recurrent_slots,
        draft_activation,
    )?;

    let maximum_draft = i32::try_from(plan.maximum_draft_tokens)
        .map_err(|_| Error::Invalid("maximum draft tokens exceed i32".to_owned()))?;
    let minimum_draft = i32::try_from(plan.minimum_draft_tokens)
        .map_err(|_| Error::Invalid("minimum draft tokens exceed i32".to_owned()))?;
    let probability_floor = plan.probability_floor()?;
    let loras = std::mem::take(&mut request.target_loras);
    let control_vector = request.target_control_vector;
    let ((run, implementation_state), steering_applied, steering_cleared) =
        execute_with_steering(&mut target, loras, control_vector, |target| {
            run_fresh_native(
                target,
                &mut draft,
                request.prompt,
                request.generation,
                plan,
                prefill_monitor,
                request.pipeline.as_deref_mut(),
                request.observers.as_deref_mut(),
                checkpoint.is_some(),
                maximum_draft,
                minimum_draft,
                probability_floor,
            )
        })?;
    match run {
        SpeculativeRunOutcome::Generated(mut run) => {
            run.output.steering_applied = steering_applied;
            run.output.steering_cleared = steering_cleared;
            Ok(SpeculativeExecutionOutcome::Generated(Box::new(
                finish_speculative_execution(
                    &mut target,
                    &mut draft,
                    *run,
                    implementation_state,
                    checkpoint,
                    checkpoint_activations,
                    request.generation,
                    plan,
                    0,
                )?,
            )))
        }
        SpeculativeRunOutcome::PrefillStopped(mut output) => {
            if implementation_state.is_some() {
                return Err(Error::Poisoned(
                    "controlled prefill stop unexpectedly captured speculative state".to_owned(),
                ));
            }
            output.steering_applied = steering_applied;
            output.steering_cleared = steering_cleared;
            Ok(SpeculativeExecutionOutcome::PrefillStopped(output))
        }
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "native speculative lowering receives every prevalidated mechanic explicitly"
)]
fn run_fresh_native<'model>(
    target: &mut Session<'model>,
    draft: &mut Session<'model>,
    prompt: &[TokenId],
    generation: &GenerationPlan,
    plan: &SpeculationPlanV1,
    prefill_monitor: Option<&mut PrefillMonitor>,
    pipeline: Option<&mut Pipeline>,
    observers: Option<&mut ObserverSet>,
    capture_state: bool,
    maximum_draft: i32,
    minimum_draft: i32,
    probability_floor: f32,
) -> Result<(SpeculativeRunOutcome, Option<Vec<u8>>), Error> {
    match plan.mechanism {
        TextSpeculativeMechanismV1::Mtp => {
            let config = MtpSessionConfig::new(plan.sequences, maximum_draft)
                .with_n_min(minimum_draft)
                .with_p_min(probability_floor);
            let (target_context, target_side) = speculative_parts(target);
            let (draft_context, draft_side) = speculative_parts(draft);
            let mut backend = MtpSession::new_with_config(target_context, draft_context, config)
                .map_err(native)?;
            let run = run_backend(
                &mut backend,
                target_side,
                draft_side,
                SpeculativeRunStart::Fresh(prompt),
                generation,
                plan,
                prefill_monitor,
                pipeline,
                observers,
            )?;
            let implementation_state = (capture_state
                && matches!(run, SpeculativeRunOutcome::Generated(_)))
            .then(|| backend.capture_implementation_state())
            .transpose()?;
            Ok((run, implementation_state))
        }
        TextSpeculativeMechanismV1::Eagle3 => {
            let config = Eagle3SessionConfig::new(plan.sequences, maximum_draft)
                .with_n_min(minimum_draft)
                .with_p_min(probability_floor);
            let (target_context, target_side) = speculative_parts(target);
            let (draft_context, draft_side) = speculative_parts(draft);
            let mut backend = Eagle3Session::new_with_config(target_context, draft_context, config)
                .map_err(native)?;
            let run = run_backend(
                &mut backend,
                target_side,
                draft_side,
                SpeculativeRunStart::Fresh(prompt),
                generation,
                plan,
                prefill_monitor,
                pipeline,
                observers,
            )?;
            let implementation_state = (capture_state
                && matches!(run, SpeculativeRunOutcome::Generated(_)))
            .then(|| backend.capture_implementation_state())
            .transpose()?;
            Ok((run, implementation_state))
        }
    }
}

fn speculative_parts<'borrow, 'model>(
    session: &'borrow mut Session<'model>,
) -> (
    &'borrow mut LlamaContext<'model>,
    SpeculativeSide<'borrow, 'model>,
) {
    let Session {
        context,
        model,
        options,
        token_history,
        position,
        activation,
        poison_reason,
        ..
    } = session;
    (
        context,
        SpeculativeSide {
            model,
            options: *options,
            history: token_history,
            position,
            activation,
            poison: poison_reason,
        },
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "aggregate checkpoint validation binds every independently supplied mechanic"
)]
fn validate_checkpoint_request(
    request: &SpeculativeCheckpointRequest,
    target: &Model,
    draft: &Model,
    generation: &GenerationPlan,
    speculation: &SpeculationPlanV1,
    activations: &SpeculativeActivations,
    target_loras: &[LoraApplication<'_>],
    target_control_vector: Option<&ControlVector>,
    pipeline: Option<&Pipeline>,
    observers: Option<&ObserverSet>,
    expected_parent: Option<&Digest>,
) -> Result<ValidatedCheckpointRequest, Error> {
    let mechanics = request
        .mechanics
        .digest_for(target.topology(), Some(draft.topology()))?;
    if request.mechanics.generation != *generation
        || request.mechanics.speculation.as_ref() != Some(speculation)
        || request.mechanics.branch_checkpoint.as_ref() != expected_parent
    {
        return Err(Error::Incompatible(
            "checkpoint aggregate plan differs from generation, speculation, or parent lineage"
                .to_owned(),
        ));
    }
    let loras = validate_steering_resources(target, target_loras, target_control_vector)?;
    let control_vector = target_control_vector.map(ControlVector::specification);
    if loras != request.mechanics.loras
        || control_vector != request.mechanics.control_vector.as_ref()
    {
        return Err(Error::Incompatible(
            "checkpoint target steering differs from the aggregate plan".to_owned(),
        ));
    }
    let pipeline_identity = pipeline
        .map(|pipeline| pipeline.specification().digest())
        .transpose()?;
    if pipeline_identity != request.mechanics.transform_pipeline {
        return Err(Error::Incompatible(
            "checkpoint transform pipeline differs from the aggregate plan".to_owned(),
        ));
    }
    let observer_identity = observers.map(ObserverSet::identity).transpose()?;
    if observer_identity != request.mechanics.observer_set {
        return Err(Error::Incompatible(
            "checkpoint observer set differs from the aggregate plan".to_owned(),
        ));
    }
    let target_activation = activations
        .target
        .as_ref()
        .map(ActivationConfiguration::program_identity);
    let draft_activation = activations
        .draft
        .as_ref()
        .map(ActivationConfiguration::program_identity);
    let declared_target = request
        .mechanics
        .target_activation
        .as_ref()
        .map(|program| program.digest_for(target.topology()))
        .transpose()?;
    if target_activation != declared_target.as_ref()
        || draft_activation != request.mechanics.draft_activation_identity()
    {
        return Err(Error::Incompatible(
            "checkpoint activation configurations differ from the aggregate plan".to_owned(),
        ));
    }
    Ok(ValidatedCheckpointRequest {
        mechanics,
        plan: request.mechanics.clone(),
        parent: request.mechanics.branch_checkpoint.clone(),
    })
}

#[allow(
    clippy::too_many_arguments,
    reason = "checkpoint construction binds every independent continuation component"
)]
fn finish_speculative_execution(
    target: &mut Session<'_>,
    draft: &mut Session<'_>,
    run: SpeculativeRunState,
    implementation_state: Option<Vec<u8>>,
    checkpoint: Option<ValidatedCheckpointRequest>,
    activations: SpeculativeActivations,
    generation: &GenerationPlan,
    speculation: &SpeculationPlanV1,
    prior_boundaries: u64,
) -> Result<SpeculativeExecution, Error> {
    let Some(checkpoint) = checkpoint else {
        return Ok(SpeculativeExecution {
            generation: run.output,
            checkpoint: None,
        });
    };
    let implementation_state = implementation_state.ok_or_else(|| {
        Error::Poisoned("checkpoint capture omitted speculative implementation state".to_owned())
    })?;
    let implementation_state_bytes = u64::try_from(implementation_state.len())
        .map_err(|_| Error::Native("speculative implementation state exceeds u64".to_owned()))?;
    let target_state = target.capture_envelope_state()?;
    let draft_state = draft.capture_envelope_state()?;
    if target_state.tokens() != draft_state.tokens()
        || target_state.receipt().position != draft_state.receipt().position
    {
        return Err(Error::Poisoned(
            "target and draft checkpoints have different causal lineage".to_owned(),
        ));
    }
    let completed = prior_boundaries
        .checked_add(
            u64::try_from(run.output.boundaries.len())
                .map_err(|_| Error::Invalid("speculation boundary count exceeds u64".to_owned()))?,
        )
        .ok_or_else(|| Error::Invalid("completed speculation boundaries overflowed".to_owned()))?;
    let generation_identity = generation.digest()?;
    let stop_tail_identity =
        Digest::of_bytes("speculative-checkpoint-stop-tail-v1", &run.stop_tail);
    let target_sampler_lineage = Digest::of_serializable(
        "speculative-target-sampler-lineage-v1",
        &(
            &checkpoint.mechanics,
            &generation_identity,
            &target_state.receipt().token_history,
            target_state.receipt().position,
            completed,
            &stop_tail_identity,
            &checkpoint.parent,
        ),
    )?;
    let receipt = SpeculativeCheckpointReceiptV1 {
        mechanics: checkpoint.mechanics,
        speculation: speculation.digest_for(target.model.topology(), draft.model.topology())?,
        target_state: target_state.receipt().digest()?,
        draft_state: draft_state.receipt().digest()?,
        implementation_state: Digest::of_bytes(
            "speculative-implementation-state-v1",
            &implementation_state,
        ),
        implementation_state_bytes,
        target_sampler_lineage,
        admitted_history: target_state.receipt().token_history.clone(),
        position: target_state.receipt().position,
        completed_boundaries: completed,
        target_activation: activations
            .target
            .as_ref()
            .map(ActivationConfiguration::program_identity)
            .cloned(),
        draft_activation: activations
            .draft
            .as_ref()
            .map(ActivationConfiguration::program_identity)
            .cloned(),
        parent: checkpoint.parent,
    };
    receipt.digest_for(speculation)?;
    let snapshot = SpeculativeStateSnapshot {
        target: target_state,
        draft: draft_state,
        implementation_state,
        sampler: run.sampler,
        generation: generation_identity,
        stop_tail: run.stop_tail,
        activations,
        options: SpeculativeSessionOptions {
            target: target.options,
            draft: draft.options,
        },
        mechanics: checkpoint.plan,
        receipt,
        thread_affinity: PhantomData,
    };
    Ok(SpeculativeExecution {
        generation: run.output,
        checkpoint: Some(snapshot),
    })
}

#[allow(
    clippy::too_many_lines,
    reason = "fail-before-allocation validation audits every checkpoint component in one sequence"
)]
fn validate_continuation_request(
    runtime: &Runtime,
    target: &Model,
    draft: &Model,
    checkpoint: &SpeculativeStateSnapshot,
    request: &SpeculativeContinuationRequest<'_>,
) -> Result<Digest, Error> {
    request.generation.validate()?;
    let target_loras =
        validate_steering_resources(target, &request.target_loras, request.target_control_vector)?;
    let target_control = request
        .target_control_vector
        .map(ControlVector::specification);
    let pipeline = request
        .pipeline
        .as_deref()
        .map(|pipeline| pipeline.specification().digest())
        .transpose()?;
    let observers = request
        .observers
        .as_deref()
        .map(ObserverSet::identity)
        .transpose()?;
    if request.generation != &checkpoint.mechanics.generation
        || checkpoint.mechanics.speculation.as_ref() != Some(request.speculation)
        || target_loras != checkpoint.mechanics.loras
        || target_control != checkpoint.mechanics.control_vector.as_ref()
        || pipeline != checkpoint.mechanics.transform_pipeline
        || observers != checkpoint.mechanics.observer_set
    {
        return Err(Error::Incompatible(
            "continuation mechanics differ from the checkpoint mechanics".to_owned(),
        ));
    }
    request
        .speculation
        .digest_for(target.topology(), draft.topology())?;
    if request.speculation.implementation != speculation_implementation_identity()
        || request.speculation.sequences != 1
    {
        return Err(Error::Incompatible(
            "continuation requires the installed one-sequence speculation implementation"
                .to_owned(),
        ));
    }
    if target.topology().backend != *runtime.identity()
        || draft.topology().backend != *runtime.identity()
    {
        return Err(Error::Incompatible(
            "target or draft model belongs to another runtime identity".to_owned(),
        ));
    }
    validate_native_model_pair(target, draft, request.speculation.mechanism)?;
    validate_speculative_context_capacities(
        request.speculation.maximum_draft_tokens,
        request.options,
        target.native.is_recurrent() || target.native.is_hybrid(),
        draft.native.is_recurrent() || draft.native.is_hybrid(),
    )?;
    if request.options != checkpoint.options {
        return Err(Error::Incompatible(
            "continuation context options differ from the checkpoint".to_owned(),
        ));
    }
    checkpoint.target.validate_contents()?;
    checkpoint.draft.validate_contents()?;
    if checkpoint
        .mechanics
        .digest_for(target.topology(), Some(draft.topology()))?
        != checkpoint.receipt.mechanics
        || checkpoint.target.tokens() != checkpoint.draft.tokens()
        || checkpoint.target.receipt().position != checkpoint.draft.receipt().position
        || checkpoint.receipt.target_state != checkpoint.target.receipt().digest()?
        || checkpoint.receipt.draft_state != checkpoint.draft.receipt().digest()?
        || checkpoint.receipt.admitted_history != checkpoint.target.receipt().token_history
        || checkpoint.receipt.position != checkpoint.target.receipt().position
    {
        return Err(Error::Incompatible(
            "speculative checkpoint context lineage is inconsistent".to_owned(),
        ));
    }
    if checkpoint.target.receipt().model != *target.artifact_digest()
        || checkpoint.draft.receipt().model != *draft.artifact_digest()
    {
        return Err(Error::Incompatible(
            "speculative checkpoint names another target or draft model".to_owned(),
        ));
    }
    target.validate_tokens(checkpoint.target.tokens())?;
    draft.validate_tokens(checkpoint.draft.tokens())?;
    let implementation_state_bytes = u64::try_from(checkpoint.implementation_state.len())
        .map_err(|_| Error::Incompatible("speculative state size exceeds u64".to_owned()))?;
    if checkpoint.implementation_state.is_empty()
        || checkpoint.receipt.implementation_state_bytes != implementation_state_bytes
        || checkpoint.receipt.implementation_state
            != Digest::of_bytes(
                "speculative-implementation-state-v1",
                &checkpoint.implementation_state,
            )
    {
        return Err(Error::Incompatible(
            "speculative implementation-state accounting is inconsistent".to_owned(),
        ));
    }
    let generation = request.generation.digest()?;
    if generation != checkpoint.generation {
        return Err(Error::Incompatible(
            "continuation generation mechanics differ from the checkpoint".to_owned(),
        ));
    }
    validate_stop_tail(request.generation, &checkpoint.stop_tail)?;
    let stop_tail = Digest::of_bytes("speculative-checkpoint-stop-tail-v1", &checkpoint.stop_tail);
    let sampler_lineage = Digest::of_serializable(
        "speculative-target-sampler-lineage-v1",
        &(
            &checkpoint.receipt.mechanics,
            &generation,
            &checkpoint.target.receipt().token_history,
            checkpoint.target.receipt().position,
            checkpoint.receipt.completed_boundaries,
            &stop_tail,
            &checkpoint.receipt.parent,
        ),
    )?;
    if sampler_lineage != checkpoint.receipt.target_sampler_lineage {
        return Err(Error::Incompatible(
            "speculative target-sampler lineage is inconsistent".to_owned(),
        ));
    }
    let activations = checkpoint.activations.clone();
    validate_activations(request.speculation, activations)?;
    let parent = checkpoint.receipt.digest_for(request.speculation)?;
    let required_context = checkpoint
        .receipt
        .position
        .checked_add(u64::from(request.generation.max_tokens))
        .and_then(|value| value.checked_add(u64::from(request.speculation.maximum_draft_tokens)))
        .ok_or_else(|| Error::Invalid("speculative continuation bound overflowed".to_owned()))?;
    if required_context > u64::from(request.options.target.context_size.get())
        || required_context > u64::from(request.options.draft.context_size.get())
    {
        return Err(Error::Invalid(format!(
            "target and draft contexts must each hold restored history + generation + draft headroom ({required_context} tokens)"
        )));
    }
    Ok(parent)
}

fn validate_request(
    runtime: &Runtime,
    target: &Model,
    draft: &Model,
    request: &SpeculativeRequest<'_>,
) -> Result<(), Error> {
    request.generation.validate()?;
    validate_steering_resources(target, &request.target_loras, request.target_control_vector)?;
    if request.prompt.is_empty() {
        return Err(Error::Invalid(
            "speculative prefill tokens must not be empty".to_owned(),
        ));
    }
    target.validate_tokens(request.prompt)?;
    draft.validate_tokens(request.prompt)?;
    request
        .speculation
        .digest_for(target.topology(), draft.topology())?;
    if request.speculation.implementation != speculation_implementation_identity() {
        return Err(Error::Incompatible(
            "speculation plan names another native implementation".to_owned(),
        ));
    }
    if request.speculation.sequences != 1 {
        return Err(Error::Invalid(
            "this target-authoritative operation currently requires exactly one sequence"
                .to_owned(),
        ));
    }
    if target.topology().backend != *runtime.identity()
        || draft.topology().backend != *runtime.identity()
    {
        return Err(Error::Incompatible(
            "target or draft model belongs to another runtime identity".to_owned(),
        ));
    }
    validate_native_model_pair(target, draft, request.speculation.mechanism)?;
    validate_speculative_context_capacities(
        request.speculation.maximum_draft_tokens,
        request.options,
        target.native.is_recurrent() || target.native.is_hybrid(),
        draft.native.is_recurrent() || draft.native.is_hybrid(),
    )?;
    let prompt = u64::try_from(request.prompt.len())
        .map_err(|_| Error::Invalid("prompt length exceeds u64".to_owned()))?;
    let required_context = prompt
        .checked_add(u64::from(request.generation.max_tokens))
        .and_then(|value| value.checked_add(u64::from(request.speculation.maximum_draft_tokens)))
        .ok_or_else(|| Error::Invalid("speculative context bound overflowed".to_owned()))?;
    if required_context > u64::from(request.options.target.context_size.get())
        || required_context > u64::from(request.options.draft.context_size.get())
    {
        return Err(Error::Invalid(format!(
            "target and draft contexts must each hold prompt + generation + draft headroom ({required_context} tokens)"
        )));
    }
    Ok(())
}

fn validate_speculative_context_capacities(
    maximum_draft_tokens: u32,
    options: SpeculativeSessionOptions,
    target_recurrent_or_hybrid: bool,
    draft_recurrent_or_hybrid: bool,
) -> Result<(), Error> {
    let verify_rows = maximum_draft_tokens
        .checked_add(1)
        .ok_or_else(|| Error::Invalid("verification batch bound overflowed".to_owned()))?;
    if options.target.batch_size < verify_rows || options.draft.batch_size < verify_rows {
        return Err(Error::Invalid(format!(
            "target and draft batch sizes must each hold {verify_rows} verification rows"
        )));
    }
    if (target_recurrent_or_hybrid && options.target.micro_batch_size < verify_rows)
        || (draft_recurrent_or_hybrid && options.draft.micro_batch_size < verify_rows)
    {
        return Err(Error::Invalid(format!(
            "recurrent target or draft micro-batches must each hold {verify_rows} rollback rows"
        )));
    }
    Ok(())
}

fn validate_native_model_pair(
    target: &Model,
    draft: &Model,
    mechanism: TextSpeculativeMechanismV1,
) -> Result<(), Error> {
    validate_speculative_vocabulary(target, draft)?;
    match mechanism {
        TextSpeculativeMechanismV1::Mtp => {
            if draft.native.n_embd_out() != target.native.n_embd() {
                return Err(Error::Incompatible(
                    "MTP draft output width differs from the target hidden width".to_owned(),
                ));
            }
        }
        TextSpeculativeMechanismV1::Eagle3 => {
            if draft.architecture() != "eagle3" {
                return Err(Error::Incompatible(format!(
                    "EAGLE-3 requires an eagle3 draft architecture, found {}",
                    draft.architecture()
                )));
            }
            validate_eagle_layer_ids(
                draft.native.target_layer_ids(),
                target.topology().layers,
                target.architecture() == "gpt-oss",
            )?;
        }
    }
    Ok(())
}

fn validate_eagle_layer_ids(
    layer_ids: &[i32],
    target_layers: u32,
    terminal_nextn_site: bool,
) -> Result<(), Error> {
    if layer_ids.len() != 3 {
        return Err(Error::Incompatible(format!(
            "EAGLE-3 draft metadata must name exactly 3 target extraction layers, found {}",
            layer_ids.len()
        )));
    }
    for &layer in layer_ids {
        let layer = u32::try_from(layer).map_err(|_| {
            Error::Incompatible(
                "EAGLE-3 draft metadata contains a negative target layer".to_owned(),
            )
        })?;
        if layer > target_layers || (layer == target_layers && !terminal_nextn_site) {
            return Err(Error::Incompatible(format!(
                "EAGLE-3 target extraction site {layer} is unsupported by the target's {target_layers}-layer architecture"
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct SpeculativeVocabularyMetadata {
    kind: u32,
    size: i32,
    add_bos: bool,
    bos: LlamaToken,
    add_eos: bool,
    eos: LlamaToken,
}

fn validate_speculative_vocabulary(target: &Model, draft: &Model) -> Result<(), Error> {
    let target_vocab = target.native.get_vocab();
    let draft_vocab = draft.native.get_vocab();
    let target_metadata = SpeculativeVocabularyMetadata {
        kind: target_vocab.vocab_type(),
        size: target_vocab.n_tokens(),
        add_bos: target_vocab.get_add_bos(),
        bos: target_vocab.bos(),
        add_eos: target_vocab.get_add_eos(),
        eos: target_vocab.eos(),
    };
    let draft_metadata = SpeculativeVocabularyMetadata {
        kind: draft_vocab.vocab_type(),
        size: draft_vocab.n_tokens(),
        add_bos: draft_vocab.get_add_bos(),
        bos: draft_vocab.bos(),
        add_eos: draft_vocab.get_add_eos(),
        eos: draft_vocab.eos(),
    };
    validate_speculative_vocabulary_metadata(target_metadata, draft_metadata)?;

    let common_size = target_metadata.size.min(draft_metadata.size);
    for token in SPECULATIVE_VOCABULARY_CHECK_START..common_size {
        let token = LlamaToken::new(token);
        let target_text = target_vocab.get_text_bytes(token).map_err(native)?;
        let draft_text = draft_vocab.get_text_bytes(token).map_err(native)?;
        if target_text != draft_text {
            return Err(Error::Incompatible(format!(
                "target and draft vocabulary text differs at token {}",
                token.0
            )));
        }
    }
    Ok(())
}

fn validate_speculative_vocabulary_metadata(
    target: SpeculativeVocabularyMetadata,
    draft: SpeculativeVocabularyMetadata,
) -> Result<(), Error> {
    if target.kind != draft.kind {
        return Err(Error::Incompatible(
            "target and draft vocabulary types differ".to_owned(),
        ));
    }
    if target.add_bos != draft.add_bos || (target.add_bos && target.bos != draft.bos) {
        return Err(Error::Incompatible(
            "target and draft beginning-of-sequence behavior differs".to_owned(),
        ));
    }
    if target.add_eos != draft.add_eos || (target.add_eos && target.eos != draft.eos) {
        return Err(Error::Incompatible(
            "target and draft end-of-sequence behavior differs".to_owned(),
        ));
    }
    if target.size.abs_diff(draft.size) > MAX_SPECULATIVE_VOCABULARY_DIFFERENCE {
        return Err(Error::Incompatible(format!(
            "target and draft vocabulary sizes differ by more than {MAX_SPECULATIVE_VOCABULARY_DIFFERENCE} tokens"
        )));
    }
    Ok(())
}

fn validate_activations(
    plan: &SpeculationPlanV1,
    activations: SpeculativeActivations,
) -> Result<
    (
        Option<ActivationConfiguration>,
        Option<ActivationConfiguration>,
    ),
    Error,
> {
    let target_identity = activations
        .target
        .as_ref()
        .map(ActivationConfiguration::program_identity);
    let draft_identity = activations
        .draft
        .as_ref()
        .map(ActivationConfiguration::program_identity);
    let valid = match &plan.activation {
        SpeculationActivationPolicyV1::None => {
            target_identity.is_none() && draft_identity.is_none()
        }
        SpeculationActivationPolicyV1::TargetOnly { target_program } => {
            target_identity == Some(target_program) && draft_identity.is_none()
        }
        SpeculationActivationPolicyV1::SeparateDraftProgram {
            target_program,
            draft_program,
        } => target_identity == Some(target_program) && draft_identity == Some(draft_program),
    };
    if !valid {
        return Err(Error::Incompatible(
            "supplied activation runtimes differ from the speculation policy".to_owned(),
        ));
    }
    Ok((activations.target, activations.draft))
}

struct SpeculativeSide<'borrow, 'model> {
    model: &'model Model,
    options: SessionOptions,
    history: &'borrow mut Vec<TokenId>,
    position: &'borrow mut u64,
    activation: &'borrow mut Option<ActivationController>,
    poison: &'borrow mut Option<String>,
}

impl SpeculativeSide<'_, '_> {
    fn poison(&mut self, error: &Error) {
        self.poison.get_or_insert_with(|| error.to_string());
    }
}

trait NativeSpeculation {
    fn begin(&mut self, prompt: &[LlamaToken]) -> Result<(), Error>;
    fn draft(&mut self, n_past: i32, last: LlamaToken) -> Result<Vec<LlamaToken>, Error>;
    fn decode_target(&mut self, batch: &mut LlamaBatch) -> Result<(), Error>;
    fn process(&mut self, batch: &LlamaBatch) -> Result<(), Error>;
    fn accept(&mut self, accepted: u16) -> Result<(), Error>;
    fn clear_target(&mut self, from: u32) -> Result<bool, Error>;
    fn clear_draft(&mut self, from: u32) -> Result<bool, Error>;
    fn is_quiescent(&self) -> bool;
    fn capture_implementation_state(&self) -> Result<Vec<u8>, Error>;
    fn restore_implementation_state(&mut self, state: &[u8]) -> Result<(), Error>;
    fn target_context(&self) -> &LlamaContext<'_>;
    fn target_failure(&self) -> Option<Error>;
    fn draft_failure(&self) -> Option<Error>;
    fn take_target_captures(&mut self) -> Result<Vec<TransactionalTensorCapture>, Error>;
    fn take_draft_captures(&mut self) -> Result<Vec<TransactionalTensorCapture>, Error>;
}

impl NativeSpeculation for MtpSession<'_, '_> {
    fn begin(&mut self, prompt: &[LlamaToken]) -> Result<(), Error> {
        MtpSession::begin(self, 0, prompt).map_err(native)
    }

    fn draft(&mut self, n_past: i32, last: LlamaToken) -> Result<Vec<LlamaToken>, Error> {
        MtpSession::draft(self, 0, n_past, last).map_err(native)
    }

    fn decode_target(&mut self, batch: &mut LlamaBatch) -> Result<(), Error> {
        MtpSession::decode_target(self, batch).map_err(native)
    }

    fn process(&mut self, batch: &LlamaBatch) -> Result<(), Error> {
        MtpSession::process(self, batch).map_err(native)
    }

    fn accept(&mut self, accepted: u16) -> Result<(), Error> {
        MtpSession::accept(self, 0, accepted).map_err(native)
    }

    fn clear_target(&mut self, from: u32) -> Result<bool, Error> {
        MtpSession::clear_target_kv_cache_seq(self, Some(0), Some(from), None).map_err(native)
    }

    fn clear_draft(&mut self, from: u32) -> Result<bool, Error> {
        MtpSession::clear_draft_kv_cache_seq(self, Some(0), Some(from), None).map_err(native)
    }

    fn is_quiescent(&self) -> bool {
        MtpSession::is_quiescent(self)
    }

    fn capture_implementation_state(&self) -> Result<Vec<u8>, Error> {
        MtpSession::speculative_state(self, 0).map_err(native)
    }

    fn restore_implementation_state(&mut self, state: &[u8]) -> Result<(), Error> {
        MtpSession::restore_speculative_state(self, 0, state).map_err(native)
    }

    fn target_context(&self) -> &LlamaContext<'_> {
        MtpSession::target_context(self)
    }

    fn target_failure(&self) -> Option<Error> {
        transaction_failure(MtpSession::target_context(self))
    }

    fn draft_failure(&self) -> Option<Error> {
        transaction_failure(MtpSession::draft_context(self))
    }

    fn take_target_captures(&mut self) -> Result<Vec<TransactionalTensorCapture>, Error> {
        take_native_captures(MtpSession::target_context_mut(self))
    }

    fn take_draft_captures(&mut self) -> Result<Vec<TransactionalTensorCapture>, Error> {
        take_native_captures(MtpSession::draft_context_mut(self))
    }
}

impl NativeSpeculation for Eagle3Session<'_, '_, '_> {
    fn begin(&mut self, prompt: &[LlamaToken]) -> Result<(), Error> {
        Eagle3Session::begin(self, 0, prompt).map_err(native)
    }

    fn draft(&mut self, n_past: i32, last: LlamaToken) -> Result<Vec<LlamaToken>, Error> {
        Eagle3Session::draft(self, 0, n_past, last).map_err(native)
    }

    fn decode_target(&mut self, batch: &mut LlamaBatch) -> Result<(), Error> {
        Eagle3Session::decode_target(self, batch).map_err(native)
    }

    fn process(&mut self, batch: &LlamaBatch) -> Result<(), Error> {
        Eagle3Session::process(self, batch).map_err(native)
    }

    fn accept(&mut self, accepted: u16) -> Result<(), Error> {
        Eagle3Session::accept(self, 0, accepted).map_err(native)
    }

    fn clear_target(&mut self, from: u32) -> Result<bool, Error> {
        Eagle3Session::clear_target_kv_cache_seq(self, Some(0), Some(from), None).map_err(native)
    }

    fn clear_draft(&mut self, from: u32) -> Result<bool, Error> {
        Eagle3Session::clear_draft_kv_cache_seq(self, Some(0), Some(from), None).map_err(native)
    }

    fn is_quiescent(&self) -> bool {
        Eagle3Session::is_quiescent(self)
    }

    fn capture_implementation_state(&self) -> Result<Vec<u8>, Error> {
        Eagle3Session::speculative_state(self, 0).map_err(native)
    }

    fn restore_implementation_state(&mut self, state: &[u8]) -> Result<(), Error> {
        Eagle3Session::restore_speculative_state(self, 0, state).map_err(native)
    }

    fn target_context(&self) -> &LlamaContext<'_> {
        Eagle3Session::target_context(self)
    }

    fn target_failure(&self) -> Option<Error> {
        transaction_failure(Eagle3Session::target_context(self))
    }

    fn draft_failure(&self) -> Option<Error> {
        transaction_failure(Eagle3Session::draft_context(self))
    }

    fn take_target_captures(&mut self) -> Result<Vec<TransactionalTensorCapture>, Error> {
        take_native_captures(Eagle3Session::target_context_mut(self))
    }

    fn take_draft_captures(&mut self) -> Result<Vec<TransactionalTensorCapture>, Error> {
        take_native_captures(Eagle3Session::draft_context_mut(self))
    }
}

enum SpeculativeRunStart<'a> {
    Fresh(&'a [TokenId]),
    Restored {
        sampler: LlamaSampler,
        stop_tail: Vec<u8>,
    },
}

struct SpeculativeRunState {
    output: SpeculativeGenerationOutput,
    sampler: LlamaSampler,
    stop_tail: Vec<u8>,
}

enum SpeculativeRunOutcome {
    Generated(Box<SpeculativeRunState>),
    PrefillStopped(Box<SpeculativePrefillStoppedOutput>),
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the operation keeps target authority, callback order, rollback, and receipt commit in one auditable loop"
)]
fn run_backend(
    backend: &mut impl NativeSpeculation,
    mut target: SpeculativeSide<'_, '_>,
    mut draft: SpeculativeSide<'_, '_>,
    start: SpeculativeRunStart<'_>,
    generation: &GenerationPlan,
    speculation: &SpeculationPlanV1,
    mut prefill_monitor: Option<&mut PrefillMonitor>,
    mut pipeline: Option<&mut Pipeline>,
    mut observers: Option<&mut ObserverSet>,
) -> Result<SpeculativeRunOutcome, Error> {
    let (restored_sampler, mut stop_window, prefill_output) = match start {
        SpeculativeRunStart::Fresh(prompt) => {
            let prefill = prefill(
                backend,
                &mut target,
                &mut draft,
                prompt,
                prefill_monitor.as_deref_mut(),
            )?;
            if prefill
                .receipt
                .as_ref()
                .is_some_and(|receipt| receipt.finish == PrefillFinish::Stopped)
            {
                if !backend.is_quiescent() {
                    let error = Error::Poisoned(
                        "speculation was not quiescent after controlled prefill stop".to_owned(),
                    );
                    target.poison(&error);
                    draft.poison(&error);
                    return Err(error);
                }
                return Ok(SpeculativeRunOutcome::PrefillStopped(Box::new(
                    SpeculativePrefillStoppedOutput {
                        prefill,
                        target_activation: take_activation_output(target.activation)?,
                        draft_activation: take_activation_output(draft.activation)?,
                        steering_applied: Vec::new(),
                        steering_cleared: Vec::new(),
                    },
                )));
            }
            let native_prompt = prompt
                .iter()
                .map(|token| LlamaToken::new(token.get()))
                .collect::<Vec<_>>();
            backend.begin(&native_prompt)?;
            (None, Vec::new(), Some(prefill))
        }
        SpeculativeRunStart::Restored { sampler, stop_tail } => {
            if prefill_monitor.is_some() {
                return Err(Error::Invalid(
                    "a restored speculative boundary has no prefill to monitor".to_owned(),
                ));
            }
            if target.history.is_empty()
                || target.history != draft.history
                || target.position != draft.position
            {
                return Err(Error::Incompatible(
                    "restored target and draft causal lineage differs".to_owned(),
                ));
            }
            validate_stop_tail(generation, &stop_tail)?;
            (Some(sampler), stop_tail, None)
        }
    };

    let initial_position = *target.position;
    if let Some(active) = pipeline.as_deref_mut() {
        active.begin(target.history)?;
    }
    if let Some(active) = observers.as_deref_mut() {
        active.begin(initial_position, generation.max_tokens)?;
    }
    let mut target_sampler = restored_sampler.map_or_else(
        || build_sampler(&target.model.native, generation, target.history),
        Ok,
    )?;
    let plan_identity = speculation.digest_for(target.model.topology(), draft.model.topology())?;
    let mut output_tokens = Vec::new();
    let mut output_bytes = Vec::new();
    let mut boundaries = Vec::new();
    let mut finish = GenerationFinish::TokenLimit;

    let mut pending = match poll_observers(observers.as_deref_mut())? {
        ControlFlow::Stop => None,
        ControlFlow::Continue => {
            let sampled = sample_candidate(
                backend.target_context(),
                target.model,
                0,
                target.history,
                &target_sampler,
                pipeline.as_deref_mut(),
                -1,
            )?;
            if target.model.is_end_of_generation(sampled.token) {
                finish = GenerationFinish::EndOfGeneration {
                    token: sampled.token,
                };
                None
            } else {
                let piece = target.model.token_piece(sampled.token)?;
                Some(sampled.with_piece(piece))
            }
        }
    };
    if pending.is_none() && matches!(finish, GenerationFinish::TokenLimit) {
        finish = GenerationFinish::ObserverStop;
    }

    while let Some(current) = pending.take() {
        let boundary_index = u64::try_from(boundaries.len())
            .map_err(|_| Error::Invalid("speculation boundary count exceeds u64".to_owned()))?;
        let start_position = *target.position;
        let n_past = i32::try_from(start_position)
            .map_err(|_| Error::Invalid("target position exceeds i32".to_owned()))?;
        begin_activation(
            draft.activation.as_ref(),
            ActivationPhaseV1::Draft,
            false,
            ActivationTelemetryDispositionV1::Provisional,
        )?;
        let remaining = generation
            .max_tokens
            .checked_sub(
                u32::try_from(output_tokens.len())
                    .map_err(|_| Error::Invalid("generated token count exceeds u32".to_owned()))?,
            )
            .ok_or_else(|| Error::Invalid("generated token accounting underflowed".to_owned()))?;
        let native_drafts = if remaining > 1 {
            match backend.draft(n_past, LlamaToken::new(current.token.get())) {
                Ok(tokens) => {
                    if let Some(error) = backend.draft_failure() {
                        abort_activation(draft.activation, || backend.take_draft_captures());
                        draft.poison(&error);
                        return Err(error);
                    }
                    tokens
                }
                Err(error) => {
                    abort_activation(draft.activation, || backend.take_draft_captures());
                    draft.poison(&error);
                    return Err(error);
                }
            }
        } else {
            Vec::new()
        };
        let proposed = native_drafts
            .into_iter()
            .map(|token| TokenId::new(token.0).map_err(Error::from))
            .collect::<Result<Vec<_>, _>>()?;
        target.model.validate_tokens(&proposed)?;
        draft.model.validate_tokens(&proposed)?;
        let proposal_pieces = proposed
            .iter()
            .copied()
            .map(|token| target.model.token_piece(token))
            .collect::<Result<Vec<_>, _>>()?;

        let mut verify = build_verify_batch(start_position, current.token, &proposed)?;
        begin_activation(
            target.activation.as_ref(),
            ActivationPhaseV1::Verification,
            false,
            ActivationTelemetryDispositionV1::Provisional,
        )?;
        let target_result = backend.decode_target(&mut verify);
        let target_result = callback_result(target_result, backend.target_failure());
        let target_activation_result =
            finish_activation(target.activation, target_result.is_ok(), || {
                backend.take_target_captures()
            });
        if let Err(error) = target_result.and(target_activation_result) {
            abort_activation(draft.activation, || backend.take_draft_captures());
            target.poison(&error);
            draft.poison(&error);
            return Err(error);
        }

        if !proposed.is_empty() {
            let rollback_position = u32::try_from(start_position)
                .map_err(|_| Error::Invalid("draft rollback position exceeds u32".to_owned()))?;
            match backend.clear_draft(rollback_position) {
                Ok(true) => {}
                Ok(false) => {
                    let error = Error::Poisoned(
                        "draft context refused rollback before target-state processing".to_owned(),
                    );
                    abort_activation(draft.activation, || backend.take_draft_captures());
                    draft.poison(&error);
                    return Err(error);
                }
                Err(error) => {
                    abort_activation(draft.activation, || backend.take_draft_captures());
                    draft.poison(&error);
                    return Err(error);
                }
            }
        }
        let process_result = backend.process(&verify);
        let process_result = callback_result(process_result, backend.draft_failure());
        let draft_activation_result =
            finish_activation(draft.activation, process_result.is_ok(), || {
                backend.take_draft_captures()
            });
        if let Err(error) = process_result.and(draft_activation_result) {
            target.poison(&error);
            draft.poison(&error);
            return Err(error);
        }

        let mut accepted = 0_usize;
        let mut next_pending = None;
        let mut boundary_finish = None;
        let mut decision_error = None;

        target_sampler = current.sampler;
        target_sampler.accept(LlamaToken::new(current.token.get()));
        append_causal_token(&mut target, &mut draft, current.token)?;
        if let Some(active) = pipeline.as_deref_mut()
            && let Err(error) = active.accept(current.token)
        {
            decision_error = Some(Error::from(error));
        }
        output_tokens.push(current.token);
        output_bytes.extend_from_slice(&current.piece);
        extend_stop_window(generation, &mut stop_window, &current.piece);
        if decision_error.is_none() {
            match observe_admission(
                observers.as_deref_mut(),
                current.token,
                &current.piece,
                *target.position,
            ) {
                Ok(ControlFlow::Stop) => {
                    boundary_finish = Some(GenerationFinish::ObserverStop);
                }
                Ok(ControlFlow::Continue) => {}
                Err(error) => decision_error = Some(error),
            }
        }
        if decision_error.is_none() && boundary_finish.is_none() {
            boundary_finish = stop_finish(generation, &stop_window)?;
        }

        while decision_error.is_none() && boundary_finish.is_none() {
            let generated = u32::try_from(output_tokens.len())
                .map_err(|_| Error::Invalid("generated token count exceeds u32".to_owned()))?;
            if generated >= generation.max_tokens {
                boundary_finish = Some(GenerationFinish::TokenLimit);
                break;
            }
            match poll_observers(observers.as_deref_mut()) {
                Ok(ControlFlow::Stop) => {
                    boundary_finish = Some(GenerationFinish::ObserverStop);
                    break;
                }
                Ok(ControlFlow::Continue) => {}
                Err(error) => {
                    decision_error = Some(error);
                    break;
                }
            }
            let logits_index = i32::try_from(accepted)
                .map_err(|_| Error::Invalid("verification row exceeds i32".to_owned()))?;
            let step = generated;
            let sampled = match sample_candidate(
                backend.target_context(),
                target.model,
                step,
                target.history,
                &target_sampler,
                pipeline.as_deref_mut(),
                logits_index,
            ) {
                Ok(sampled) => sampled,
                Err(error) => {
                    decision_error = Some(error);
                    break;
                }
            };
            if target.model.is_end_of_generation(sampled.token) {
                boundary_finish = Some(GenerationFinish::EndOfGeneration {
                    token: sampled.token,
                });
                break;
            }
            if proposed.get(accepted) != Some(&sampled.token) {
                match target.model.token_piece(sampled.token) {
                    Ok(piece) => next_pending = Some(sampled.with_piece(piece)),
                    Err(error) => decision_error = Some(error),
                }
                break;
            }

            let piece = &proposal_pieces[accepted];
            accepted = accepted
                .checked_add(1)
                .ok_or_else(|| Error::Invalid("accepted token count overflowed".to_owned()))?;
            target_sampler = sampled.sampler;
            target_sampler.accept(LlamaToken::new(sampled.token.get()));
            append_causal_token(&mut target, &mut draft, sampled.token)?;
            if let Some(active) = pipeline.as_deref_mut()
                && let Err(error) = active.accept(sampled.token)
            {
                decision_error = Some(Error::from(error));
            }
            output_tokens.push(sampled.token);
            output_bytes.extend_from_slice(piece);
            extend_stop_window(generation, &mut stop_window, piece);
            if decision_error.is_none() {
                match observe_admission(
                    observers.as_deref_mut(),
                    sampled.token,
                    piece,
                    *target.position,
                ) {
                    Ok(ControlFlow::Stop) => {
                        boundary_finish = Some(GenerationFinish::ObserverStop);
                    }
                    Ok(ControlFlow::Continue) => {}
                    Err(error) => decision_error = Some(error),
                }
            }
            if decision_error.is_none() && boundary_finish.is_none() {
                boundary_finish = stop_finish(generation, &stop_window)?;
            }
        }

        let final_position = *target.position;
        let proposed_count = u64::try_from(proposed.len())
            .map_err(|_| Error::Invalid("proposed token count exceeds u64".to_owned()))?;
        let evaluated_end = start_position
            .checked_add(1)
            .and_then(|value| value.checked_add(proposed_count))
            .ok_or_else(|| Error::Invalid("verification position overflowed".to_owned()))?;
        if final_position < evaluated_end {
            let rollback_position = u32::try_from(final_position)
                .map_err(|_| Error::Invalid("rollback position exceeds u32".to_owned()))?;
            let target_rollback = backend.clear_target(rollback_position);
            let draft_rollback = backend.clear_draft(rollback_position);
            if !matches!(target_rollback, Ok(true)) || !matches!(draft_rollback, Ok(true)) {
                let error = Error::Poisoned(format!(
                    "target or draft context refused rejected-suffix rollback \
                     (target={target_rollback:?}, draft={draft_rollback:?})"
                ));
                target.poison(&error);
                draft.poison(&error);
                return Err(error);
            }
        }
        if !proposed.is_empty() {
            let accepted_native = u16::try_from(accepted)
                .map_err(|_| Error::Invalid("accepted token count exceeds u16".to_owned()))?;
            if let Err(error) = backend.accept(accepted_native) {
                target.poison(&error);
                draft.poison(&error);
                return Err(error);
            }
        }
        if !backend.is_quiescent() {
            let error = Error::Poisoned(
                "native speculation remained non-quiescent after accept".to_owned(),
            );
            target.poison(&error);
            draft.poison(&error);
            return Err(error);
        }

        let mut telemetry = Vec::new();
        if let Some(active) = target.activation.as_mut() {
            telemetry.extend(active.resolve_provisional(final_position)?);
        }
        if let Some(active) = draft.activation.as_mut() {
            telemetry.extend(active.resolve_provisional(final_position)?);
        }
        let accepted_u32 = u32::try_from(accepted)
            .map_err(|_| Error::Invalid("accepted token count exceeds u32".to_owned()))?;
        boundaries.push(SpeculationBoundaryReceiptV1::from_tokens(
            speculation,
            plan_identity.clone(),
            boundary_index,
            &proposed,
            accepted_u32,
            telemetry,
        )?);

        if let Some(error) = decision_error {
            return Err(error);
        }
        if let Some(done) = boundary_finish {
            finish = done;
            break;
        }
        pending = next_pending;
        if pending.is_none() {
            return Err(Error::Poisoned(
                "verification completed without a terminal boundary or next token".to_owned(),
            ));
        }
    }

    if !backend.is_quiescent() {
        let error =
            Error::Poisoned("speculation was not quiescent at operation completion".to_owned());
        target.poison(&error);
        draft.poison(&error);
        return Err(error);
    }
    let transform_receipt = pipeline
        .as_deref()
        .map(|active| active.receipt().digest())
        .transpose()?;
    let observer_receipts = observers
        .as_deref()
        .map_or_else(Vec::new, ObserverSet::receipts)
        .into_iter()
        .map(|receipt| receipt.digest())
        .collect::<Result<Vec<_>, _>>()?;
    let generation_receipt = GenerationReceipt {
        plan: generation.digest()?,
        initial_position,
        admitted_tokens: u32::try_from(output_tokens.len())
            .map_err(|_| Error::Invalid("generated token count exceeds u32".to_owned()))?,
        admitted_bytes: u64::try_from(output_bytes.len())
            .map_err(|_| Error::Invalid("generated byte count exceeds u64".to_owned()))?,
        final_position: *target.position,
        finish,
        transform_receipt,
        observer_receipts,
    };
    generation_receipt.digest()?;
    let speculation_receipt = SpeculationReceiptV1::from_boundaries(speculation, &boundaries)?;
    speculation_receipt.digest_for(speculation, &boundaries)?;
    let target_activation = take_activation_output(target.activation)?;
    let draft_activation = take_activation_output(draft.activation)?;
    let stop_tail = checkpoint_stop_tail(generation, &stop_window);
    Ok(SpeculativeRunOutcome::Generated(Box::new(
        SpeculativeRunState {
            output: SpeculativeGenerationOutput {
                prefill: prefill_output,
                generation: GenerationOutput {
                    bytes: output_bytes,
                    tokens: output_tokens,
                    receipt: generation_receipt,
                },
                boundaries,
                speculation: speculation_receipt,
                target_activation,
                draft_activation,
                steering_applied: Vec::new(),
                steering_cleared: Vec::new(),
            },
            sampler: target_sampler,
            stop_tail,
        },
    )))
}

#[allow(
    clippy::too_many_lines,
    reason = "target and draft chunk admission plus cooperative accounting stay adjacent"
)]
fn prefill(
    backend: &mut impl NativeSpeculation,
    target: &mut SpeculativeSide<'_, '_>,
    draft: &mut SpeculativeSide<'_, '_>,
    prompt: &[TokenId],
    mut monitor: Option<&mut PrefillMonitor>,
) -> Result<PrefillOutput, Error> {
    let requested_tokens = u64::try_from(prompt.len())
        .map_err(|_| Error::Invalid("speculative prefill token count exceeds u64".to_owned()))?;
    let mut progress = PrefillProgress {
        initial_position: *target.position,
        requested_tokens,
        admitted_tokens: 0,
        admitted_chunks: 0,
        position: *target.position,
    };
    if let Some(active) = monitor.as_deref_mut() {
        active.begin(progress)?;
    }
    let target_isolates = target
        .activation
        .as_ref()
        .is_some_and(ActivationController::has_last_prefill_capture);
    let draft_isolates = draft
        .activation
        .as_ref()
        .is_some_and(ActivationController::has_last_prefill_capture);
    let isolate_last = target_isolates || draft_isolates;
    let chunk_limit = usize::try_from(target.options.batch_size.min(draft.options.batch_size))
        .map_err(|_| Error::Invalid("prefill batch size exceeds usize".to_owned()))?;
    let prefix_end = if isolate_last {
        prompt.len().saturating_sub(1)
    } else {
        prompt.len()
    };
    let mut offset = 0_usize;
    while offset < prompt.len() {
        let end = if offset < prefix_end {
            offset.saturating_add(chunk_limit).min(prefix_end)
        } else {
            prompt.len()
        };
        let chunk = &prompt[offset..end];
        if let Some(active) = monitor.as_deref_mut()
            && active.poll(progress)? == ControlFlow::Stop
        {
            return Ok(PrefillOutput {
                admitted_tokens: progress.admitted_tokens,
                position: *target.position,
                receipt: Some(active.finish(PrefillFinish::Stopped)?),
            });
        }
        let mut batch = build_prefill_batch(*target.position, chunk, end == prompt.len())?;
        begin_activation(
            target.activation.as_ref(),
            ActivationPhaseV1::Prefill,
            isolate_last && end == prompt.len(),
            ActivationTelemetryDispositionV1::Admitted,
        )?;
        let target_result = backend.decode_target(&mut batch);
        let target_result = callback_result(target_result, backend.target_failure());
        let target_activation_result =
            finish_activation(target.activation, target_result.is_ok(), || {
                backend.take_target_captures()
            });
        if let Err(error) = target_result.and(target_activation_result) {
            target.poison(&error);
            return Err(error);
        }

        begin_activation(
            draft.activation.as_ref(),
            ActivationPhaseV1::Prefill,
            isolate_last && end == prompt.len(),
            ActivationTelemetryDispositionV1::Admitted,
        )?;
        let process_result = backend.process(&batch);
        let process_result = callback_result(process_result, backend.draft_failure());
        let draft_activation_result =
            finish_activation(draft.activation, process_result.is_ok(), || {
                backend.take_draft_captures()
            });
        if let Err(error) = process_result.and(draft_activation_result) {
            draft.poison(&error);
            return Err(error);
        }
        append_causal_slice(target, draft, chunk)?;
        let chunk_tokens = u64::try_from(chunk.len())
            .map_err(|_| Error::Invalid("speculative prefill chunk exceeds u64".to_owned()))?;
        progress.admitted_tokens = progress
            .admitted_tokens
            .checked_add(chunk_tokens)
            .ok_or_else(|| {
                Error::Invalid("speculative prefill token accounting overflowed".to_owned())
            })?;
        progress.admitted_chunks = progress.admitted_chunks.checked_add(1).ok_or_else(|| {
            Error::Invalid("speculative prefill chunk accounting overflowed".to_owned())
        })?;
        progress.position = *target.position;
        if let Some(active) = monitor.as_deref_mut()
            && active.observe_chunk(progress)? == ControlFlow::Stop
            && progress.admitted_tokens < progress.requested_tokens
        {
            return Ok(PrefillOutput {
                admitted_tokens: progress.admitted_tokens,
                position: *target.position,
                receipt: Some(active.finish(PrefillFinish::Stopped)?),
            });
        }
        offset = end;
    }
    let receipt = monitor
        .map(|active| active.finish(PrefillFinish::Complete))
        .transpose()?;
    Ok(PrefillOutput {
        admitted_tokens: progress.admitted_tokens,
        position: *target.position,
        receipt,
    })
}

fn build_prefill_batch(
    position: u64,
    tokens: &[TokenId],
    final_chunk: bool,
) -> Result<LlamaBatch, Error> {
    let mut batch = LlamaBatch::new(tokens.len(), 1);
    for (offset, token) in tokens.iter().enumerate() {
        let offset_u64 = u64::try_from(offset)
            .map_err(|_| Error::Invalid("prefill offset exceeds u64".to_owned()))?;
        let native_position = position
            .checked_add(offset_u64)
            .ok_or_else(|| Error::Invalid("prefill position overflowed".to_owned()))?;
        let native_position = i32::try_from(native_position)
            .map_err(|_| Error::Invalid("prefill position exceeds i32".to_owned()))?;
        batch
            .add(
                LlamaToken::new(token.get()),
                native_position,
                &[0],
                final_chunk && offset + 1 == tokens.len(),
            )
            .map_err(native)?;
    }
    Ok(batch)
}

fn build_verify_batch(
    position: u64,
    pending: TokenId,
    proposed: &[TokenId],
) -> Result<LlamaBatch, Error> {
    let capacity = proposed
        .len()
        .checked_add(1)
        .ok_or_else(|| Error::Invalid("verification batch capacity overflowed".to_owned()))?;
    let mut batch = LlamaBatch::new(capacity, 1);
    for (offset, token) in std::iter::once(&pending).chain(proposed).enumerate() {
        let offset = u64::try_from(offset)
            .map_err(|_| Error::Invalid("verification offset exceeds u64".to_owned()))?;
        let position = position
            .checked_add(offset)
            .ok_or_else(|| Error::Invalid("verification position overflowed".to_owned()))?;
        let position = i32::try_from(position)
            .map_err(|_| Error::Invalid("verification position exceeds i32".to_owned()))?;
        batch
            .add(LlamaToken::new(token.get()), position, &[0], true)
            .map_err(native)?;
    }
    Ok(batch)
}

struct SampledToken {
    token: TokenId,
    sampler: LlamaSampler,
}

impl SampledToken {
    fn with_piece(self, piece: Vec<u8>) -> PendingToken {
        PendingToken {
            token: self.token,
            sampler: self.sampler,
            piece,
        }
    }
}

struct PendingToken {
    token: TokenId,
    sampler: LlamaSampler,
    piece: Vec<u8>,
}

fn sample_candidate(
    context: &LlamaContext<'_>,
    model: &Model,
    step: u32,
    history: &[TokenId],
    sampler: &LlamaSampler,
    pipeline: Option<&mut Pipeline>,
    logits_index: i32,
) -> Result<SampledToken, Error> {
    let mut sampled_sampler = clone_sampler(sampler)?;
    let mut logits = catch_unwind(AssertUnwindSafe(|| {
        context.get_logits_ith(logits_index).to_vec()
    }))
    .map_err(|_| Error::Native("llama.cpp logits row was unavailable".to_owned()))?;
    if let Some(active) = pipeline {
        active.apply_to_vocabulary(step, history, &mut logits)?;
    }
    let data = logits
        .into_iter()
        .enumerate()
        .map(|(index, logit)| {
            let token = i32::try_from(index)
                .map_err(|_| Error::Native("vocabulary token ID overflowed".to_owned()))?;
            Ok(LlamaTokenData::new(LlamaToken::new(token), logit, 0.0))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let mut candidates = LlamaTokenDataArray::new(data, false);
    sampled_sampler.apply(&mut candidates);
    let selected = candidates
        .selected_token()
        .ok_or_else(|| Error::Native("native sampler did not select a token".to_owned()))?;
    let token = TokenId::new(selected.0)?;
    model.validate_tokens(std::slice::from_ref(&token))?;
    Ok(SampledToken {
        token,
        sampler: sampled_sampler,
    })
}

pub(crate) fn clone_sampler(sampler: &LlamaSampler) -> Result<LlamaSampler, Error> {
    catch_unwind(AssertUnwindSafe(|| sampler.clone_sampler()))
        .map_err(|_| Error::Native("native sampler clone panicked".to_owned()))
}

fn begin_activation(
    activation: Option<&ActivationController>,
    phase: ActivationPhaseV1,
    last_prefill: bool,
    disposition: ActivationTelemetryDispositionV1,
) -> Result<(), Error> {
    if let Some(active) = activation {
        active.begin_decode(phase, last_prefill, disposition)?;
    }
    Ok(())
}

fn finish_activation(
    activation: &mut Option<ActivationController>,
    succeeded: bool,
    take_captures: impl FnOnce() -> Result<Vec<TransactionalTensorCapture>, Error>,
) -> Result<(), Error> {
    let Some(active) = activation.as_mut() else {
        return Ok(());
    };
    let decode = active.end_decode()?;
    let captures = take_captures()?;
    if succeeded {
        active.consume_captures(captures, decode)
    } else {
        Ok(())
    }
}

fn abort_activation(
    activation: &mut Option<ActivationController>,
    take_captures: impl FnOnce() -> Result<Vec<TransactionalTensorCapture>, Error>,
) {
    if let Some(active) = activation.as_mut() {
        let _ = active.end_decode();
        let _ = take_captures();
    }
}

fn transaction_failure(context: &LlamaContext<'_>) -> Option<Error> {
    context
        .tensor_transactions()
        .and_then(|transactions| transactions.failure())
        .map(|failure| Error::Native(failure.to_string()))
}

fn take_native_captures(
    context: &mut LlamaContext<'_>,
) -> Result<Vec<TransactionalTensorCapture>, Error> {
    context
        .tensor_transactions_mut()
        .ok_or_else(|| {
            Error::Poisoned("activation context lost its tensor transaction state".to_owned())
        })
        .map(TensorTransactions::take_captures)
}

fn callback_result(result: Result<(), Error>, failure: Option<Error>) -> Result<(), Error> {
    failure.map_or(result, Err)
}

fn append_causal_slice(
    target: &mut SpeculativeSide<'_, '_>,
    draft: &mut SpeculativeSide<'_, '_>,
    tokens: &[TokenId],
) -> Result<(), Error> {
    let count = u64::try_from(tokens.len())
        .map_err(|_| Error::Invalid("causal token count exceeds u64".to_owned()))?;
    let target_position = target
        .position
        .checked_add(count)
        .ok_or_else(|| Error::Invalid("target position overflowed".to_owned()))?;
    let draft_position = draft
        .position
        .checked_add(count)
        .ok_or_else(|| Error::Invalid("draft position overflowed".to_owned()))?;
    target.history.extend_from_slice(tokens);
    draft.history.extend_from_slice(tokens);
    *target.position = target_position;
    *draft.position = draft_position;
    Ok(())
}

fn append_causal_token(
    target: &mut SpeculativeSide<'_, '_>,
    draft: &mut SpeculativeSide<'_, '_>,
    token: TokenId,
) -> Result<(), Error> {
    append_causal_slice(target, draft, std::slice::from_ref(&token))
}

fn poll_observers(observers: Option<&mut ObserverSet>) -> Result<ControlFlow, Error> {
    observers.map_or(Ok(ControlFlow::Continue), |active| {
        active.poll().map_err(Error::from)
    })
}

fn observe_admission(
    observers: Option<&mut ObserverSet>,
    token: TokenId,
    piece: &[u8],
    position: u64,
) -> Result<ControlFlow, Error> {
    observers.map_or(Ok(ControlFlow::Continue), |active| {
        active
            .observe(ObservedToken {
                token,
                piece,
                position,
            })
            .map_err(Error::from)
    })
}

fn stop_finish(plan: &GenerationPlan, output: &[u8]) -> Result<Option<GenerationFinish>, Error> {
    let Some(index) = plan.stops.iter().position(|stop| output.ends_with(stop)) else {
        return Ok(None);
    };
    Ok(Some(GenerationFinish::StopSequence {
        index: u32::try_from(index)
            .map_err(|_| Error::Invalid("stop sequence index exceeds u32".to_owned()))?,
    }))
}

fn validate_stop_tail(plan: &GenerationPlan, tail: &[u8]) -> Result<(), Error> {
    let maximum = maximum_stop_bytes(plan).saturating_sub(1);
    if tail.len() > maximum {
        return Err(Error::Incompatible(
            "checkpoint stop-prefix state exceeds the generation plan bound".to_owned(),
        ));
    }
    Ok(())
}

fn extend_stop_window(plan: &GenerationPlan, window: &mut Vec<u8>, piece: &[u8]) {
    let maximum = maximum_stop_bytes(plan);
    if maximum == 0 {
        window.clear();
        return;
    }
    if piece.len() >= maximum {
        window.clear();
        window.extend_from_slice(&piece[piece.len() - maximum..]);
        return;
    }
    window.extend_from_slice(piece);
    if window.len() > maximum {
        window.drain(..window.len() - maximum);
    }
}

fn checkpoint_stop_tail(plan: &GenerationPlan, window: &[u8]) -> Vec<u8> {
    let maximum = maximum_stop_bytes(plan).saturating_sub(1);
    window[window.len().saturating_sub(maximum)..].to_vec()
}

fn maximum_stop_bytes(plan: &GenerationPlan) -> usize {
    plan.stops.iter().map(Vec::len).max().unwrap_or(0)
}

fn take_activation_output(
    activation: &mut Option<ActivationController>,
) -> Result<Option<SpeculativeActivationOutput>, Error> {
    let Some(active) = activation.as_mut() else {
        return Ok(None);
    };
    let captures = active.take_capture_output()?;
    let program = active.take_program_output()?;
    Ok(Some(SpeculativeActivationOutput { captures, program }))
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;

    #[test]
    fn implementation_identity_binds_the_pinned_native_revision() {
        assert_eq!(
            speculation_implementation_identity(),
            Digest::of_bytes(
                SPECULATION_IMPLEMENTATION_DOMAIN,
                format!(
                    "{LLAMA_CPP_BINDING_VERSION}|{LLAMA_CPP_BINDING_SOURCE_REVISION}|{LLAMA_CPP_REVISION}|target-authoritative-v1"
                )
                .as_bytes(),
            )
        );
    }

    #[test]
    fn default_speculative_contexts_are_nonzero() {
        let options = SpeculativeSessionOptions::default();
        assert_eq!(options.target.context_size, NonZeroU32::new(4_096).unwrap());
        assert_eq!(options.target, options.draft);
    }

    #[test]
    fn speculative_decode_capacity_is_validated_before_context_allocation() {
        let options = SpeculativeSessionOptions {
            target: SessionOptions {
                batch_size: 4,
                micro_batch_size: 1,
                ..SessionOptions::default()
            },
            draft: SessionOptions {
                batch_size: 4,
                micro_batch_size: 4,
                ..SessionOptions::default()
            },
        };
        assert!(validate_speculative_context_capacities(3, options, false, false).is_ok());
        assert!(validate_speculative_context_capacities(3, options, true, false).is_err());
        assert!(
            validate_speculative_context_capacities(
                3,
                SpeculativeSessionOptions {
                    target: SessionOptions {
                        batch_size: 3,
                        ..options.target
                    },
                    ..options
                },
                false,
                false,
            )
            .is_err()
        );
        assert!(validate_speculative_context_capacities(u32::MAX, options, false, false).is_err());
    }

    #[test]
    fn eagle_layer_metadata_accepts_only_supported_sites() {
        assert!(validate_eagle_layer_ids(&[1, 4, 7], 8, false).is_ok());
        assert!(validate_eagle_layer_ids(&[1, 4], 8, false).is_err());
        assert!(validate_eagle_layer_ids(&[1, -1, 7], 8, false).is_err());
        assert!(validate_eagle_layer_ids(&[1, 4, 8], 8, false).is_err());
        assert!(validate_eagle_layer_ids(&[1, 4, 8], 8, true).is_ok());
        assert!(validate_eagle_layer_ids(&[1, 4, 9], 8, true).is_err());
    }

    #[test]
    fn speculative_vocabulary_metadata_matches_native_rules() {
        let target = SpeculativeVocabularyMetadata {
            kind: 1,
            size: 128,
            add_bos: true,
            bos: LlamaToken::new(1),
            add_eos: true,
            eos: LlamaToken::new(2),
        };
        assert!(validate_speculative_vocabulary_metadata(target, target).is_ok());
        assert!(
            validate_speculative_vocabulary_metadata(
                target,
                SpeculativeVocabularyMetadata { kind: 2, ..target }
            )
            .is_err()
        );
        assert!(
            validate_speculative_vocabulary_metadata(
                target,
                SpeculativeVocabularyMetadata {
                    bos: LlamaToken::new(3),
                    ..target
                }
            )
            .is_err()
        );
        assert!(
            validate_speculative_vocabulary_metadata(
                target,
                SpeculativeVocabularyMetadata {
                    size: target.size + 129,
                    ..target
                }
            )
            .is_err()
        );
    }

    #[test]
    fn checkpoint_stop_tail_preserves_cross_operation_matches() {
        let plan = GenerationPlan {
            sampling: logit_loom::SamplingPlan::default(),
            max_tokens: 8,
            biases: Vec::new(),
            stops: vec![b"abc".to_vec(), b"xy".to_vec()],
            grammar: None,
        };
        let mut window = b"ab".to_vec();
        validate_stop_tail(&plan, &window).unwrap();
        extend_stop_window(&plan, &mut window, b"c");
        assert_eq!(
            stop_finish(&plan, &window).unwrap(),
            Some(GenerationFinish::StopSequence { index: 0 })
        );
        assert_eq!(checkpoint_stop_tail(&plan, &window), b"bc");
        assert!(validate_stop_tail(&plan, b"abc").is_err());
    }

    #[test]
    fn checkpoint_stop_tail_is_empty_without_stop_mechanics() {
        let plan = GenerationPlan {
            sampling: logit_loom::SamplingPlan::default(),
            max_tokens: 8,
            biases: Vec::new(),
            grammar: None,
            stops: Vec::new(),
        };
        let mut window = b"prior".to_vec();
        extend_stop_window(&plan, &mut window, b"new");
        assert!(window.is_empty());
        assert!(checkpoint_stop_tail(&plan, &window).is_empty());
        assert!(validate_stop_tail(&plan, b"x").is_err());
    }
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! Whole-operation lowering for aggregate text mechanics.

use std::marker::PhantomData;
use std::rc::Rc;

use llama_cpp_4::sampling::LlamaSampler;
use logit_loom::{
    Digest, ObserverSet, Pipeline, PrefillFinish, PrefillMonitor, SpeculationReceiptV1,
    SteeringAction, SteeringKind, SteeringReceipt, TextMechanicsCheckpointReceiptV2,
    TextMechanicsCleanupReceiptV2, TextMechanicsPlanV2, TextMechanicsReceiptV2, TokenId,
};

use crate::{
    ActivationConfiguration, ControlVector, Error, GenerationOutput, LoraApplication, Model,
    PrefillOutput, Runtime, SpeculativeActivationOutput, SpeculativeActivations,
    SpeculativeCheckpointRequest, SpeculativeContinuationRequest, SpeculativeGenerationOutput,
    SpeculativeRequest, SpeculativeSessionOptions, SpeculativeStateSnapshot, StateSnapshot,
    generate_speculative, generate_speculative_checkpointed, resume_speculative,
    resume_speculative_checkpointed,
    session::{Session, StatefulGenerationOutput},
    speculation::{
        SpeculativeExecutionOutcome, SpeculativePrefillStoppedOutput, clone_sampler,
        generate_speculative_controlled,
    },
    steering::{execute_with_steering, validate_steering_resources},
};

/// Opaque ordinary causal state plus its complete aggregate mechanics.
pub struct OrdinaryTextMechanicsSnapshot {
    state: StateSnapshot,
    mechanics: TextMechanicsPlanV2,
    generation: Digest,
    stop_tail: Vec<u8>,
    sampler: LlamaSampler,
    receipt: TextMechanicsCheckpointReceiptV2,
    thread_affinity: PhantomData<Rc<()>>,
}

impl std::fmt::Debug for OrdinaryTextMechanicsSnapshot {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OrdinaryTextMechanicsSnapshot")
            .field("receipt", &self.receipt)
            .field("state_bytes", &self.state.receipt().state_bytes)
            .finish_non_exhaustive()
    }
}

impl OrdinaryTextMechanicsSnapshot {
    /// Returns the opaque target-context state.
    pub const fn state(&self) -> &StateSnapshot {
        &self.state
    }

    /// Returns the complete mechanics used to build this causal state.
    pub const fn mechanics(&self) -> &TextMechanicsPlanV2 {
        &self.mechanics
    }

    /// Returns serializable aggregate checkpoint accounting.
    pub const fn receipt(&self) -> &TextMechanicsCheckpointReceiptV2 {
        &self.receipt
    }

    /// Clones opaque sampler and causal state for an independent branch.
    ///
    /// # Errors
    ///
    /// Returns an error when the pinned native sampler cannot produce an
    /// independent in-process clone.
    pub fn try_clone(&self) -> Result<Self, Error> {
        Ok(Self {
            state: self.state.clone(),
            mechanics: self.mechanics.clone(),
            generation: self.generation.clone(),
            stop_tail: self.stop_tail.clone(),
            sampler: clone_sampler(&self.sampler)?,
            receipt: self.receipt.clone(),
            thread_affinity: PhantomData,
        })
    }
}

/// Exact causal state selected as the parent of one aggregate operation.
#[derive(Clone, Copy, Debug)]
pub enum TextMechanicsResume<'a> {
    /// Ordinary single-context state.
    Ordinary(&'a OrdinaryTextMechanicsSnapshot),
    /// Complete target-authoritative speculative state.
    Speculative(&'a SpeculativeStateSnapshot),
}

/// Opaque causal state captured at a successful aggregate boundary.
#[derive(Debug)]
pub enum TextMechanicsCheckpoint {
    /// Ordinary single-context state.
    Ordinary(Box<OrdinaryTextMechanicsSnapshot>),
    /// Complete target-authoritative speculative state.
    Speculative(Box<SpeculativeStateSnapshot>),
}

/// Mechanic-specific output retained beside the aggregate receipt.
#[derive(Clone, Debug, PartialEq)]
pub enum TextMechanicsExecutionOutput {
    /// Ordinary target generation.
    Ordinary {
        /// Fresh-prompt prefill accounting. Restored branches have no prefill.
        prefill: Option<PrefillOutput>,
        /// Causally admitted generation.
        generation: Option<GenerationOutput>,
        /// Target activation evidence, when selected.
        target_activation: Option<SpeculativeActivationOutput>,
    },
    /// Target-authoritative speculative generation.
    Speculative(Box<SpeculativeGenerationOutput>),
    /// Cooperative cancellation at a complete target/draft prefill boundary.
    SpeculativePrefillStopped {
        /// Exact admitted prefill accounting.
        prefill: PrefillOutput,
        /// Target activation evidence, when selected.
        target_activation: Option<SpeculativeActivationOutput>,
        /// Draft activation evidence, when independently selected.
        draft_activation: Option<SpeculativeActivationOutput>,
    },
}

impl TextMechanicsExecutionOutput {
    /// Returns the target-authoritative generation output.
    #[must_use]
    pub const fn generation(&self) -> Option<&GenerationOutput> {
        match self {
            Self::Ordinary { generation, .. } => generation.as_ref(),
            Self::Speculative(output) => Some(&output.generation),
            Self::SpeculativePrefillStopped { .. } => None,
        }
    }
}

/// Successful whole-operation output and content-free terminal evidence.
#[derive(Debug)]
pub struct TextMechanicsOutput {
    /// Ordinary or target-authoritative speculative mechanics.
    pub execution: TextMechanicsExecutionOutput,
    /// Optional quiescent continuation state.
    pub checkpoint: Option<TextMechanicsCheckpoint>,
    /// Successful steering applications in exact native order.
    pub steering_applied: Vec<SteeringReceipt>,
    /// Successful steering cleanup in exact reverse native order.
    pub steering_cleared: Vec<SteeringReceipt>,
    /// Aggregate plan, terminal, lineage, and release evidence.
    pub receipt: TextMechanicsReceiptV2,
}

/// Borrowed callbacks/resources and owned runtime configuration for one
/// aggregate text operation.
pub struct TextMechanicsRequest<'a> {
    plan: &'a TextMechanicsPlanV2,
    prompt: &'a [TokenId],
    options: SpeculativeSessionOptions,
    target_activation: Option<ActivationConfiguration>,
    draft_activation: Option<ActivationConfiguration>,
    target_loras: Vec<LoraApplication<'a>>,
    target_control_vector: Option<&'a ControlVector>,
    prefill_monitor: Option<&'a mut PrefillMonitor>,
    pipeline: Option<&'a mut Pipeline>,
    observers: Option<&'a mut ObserverSet>,
    resume: Option<TextMechanicsResume<'a>>,
    capture_checkpoint: bool,
}

impl<'a> TextMechanicsRequest<'a> {
    /// Constructs a fresh aggregate operation with default context options.
    #[must_use]
    pub fn new(plan: &'a TextMechanicsPlanV2, prompt: &'a [TokenId]) -> Self {
        Self {
            plan,
            prompt,
            options: SpeculativeSessionOptions::default(),
            target_activation: None,
            draft_activation: None,
            target_loras: Vec::new(),
            target_control_vector: None,
            prefill_monitor: None,
            pipeline: None,
            observers: None,
            resume: None,
            capture_checkpoint: false,
        }
    }

    /// Sets target and optional draft context-allocation options.
    #[must_use]
    pub const fn with_options(mut self, options: SpeculativeSessionOptions) -> Self {
        self.options = options;
        self
    }

    /// Installs exact target and independently selected draft activation
    /// runtimes.
    #[must_use]
    pub fn with_activations(
        mut self,
        target: Option<ActivationConfiguration>,
        draft: Option<ActivationConfiguration>,
    ) -> Self {
        self.target_activation = target;
        self.draft_activation = draft;
        self
    }

    /// Applies ordered target `LoRA`s and an optional target control vector for
    /// the complete prefill/generation operation.
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

    /// Installs one exact ordered logit-transform pipeline.
    #[must_use]
    pub fn with_pipeline(mut self, pipeline: &'a mut Pipeline) -> Self {
        self.pipeline = Some(pipeline);
        self
    }

    /// Installs one exact ordered admitted-token observer set.
    #[must_use]
    pub fn with_observers(mut self, observers: &'a mut ObserverSet) -> Self {
        self.observers = Some(observers);
        self
    }

    /// Installs cooperative controlled-prefill polling and chunk observation.
    #[must_use]
    pub fn with_prefill_monitor(mut self, monitor: &'a mut PrefillMonitor) -> Self {
        self.prefill_monitor = Some(monitor);
        self
    }

    /// Selects an exact ordinary or speculative parent checkpoint.
    #[must_use]
    pub const fn with_resume(mut self, resume: TextMechanicsResume<'a>) -> Self {
        self.resume = Some(resume);
        self
    }

    /// Requests capture of a quiescent successor checkpoint.
    #[must_use]
    pub const fn with_checkpoint_capture(mut self) -> Self {
        self.capture_checkpoint = true;
        self
    }
}

/// Executes one complete topology-bound text-mechanics plan.
///
/// Every plan, model, prompt bound, checkpoint lineage, activation,
/// transform, observer, and steering identity is validated before context
/// allocation. Steering is applied only to the target context, remains active
/// through the complete operation, and is explicitly cleared before any
/// checkpoint capture. There is no ordinary/speculative fallback.
///
/// # Errors
///
/// Returns before allocation for a malformed or mismatched request. Native
/// execution, callback, restore, cleanup, or capture uncertainty fails closed.
pub fn execute_text_mechanics(
    runtime: &Runtime,
    target: &Model,
    draft: Option<&Model>,
    request: TextMechanicsRequest<'_>,
) -> Result<TextMechanicsOutput, Error> {
    validate_request(runtime, target, draft, &request)?;
    if request.plan.speculation.is_some() {
        let draft = draft.ok_or_else(|| {
            Error::Invalid("validated aggregate speculation lost its draft model".to_owned())
        })?;
        execute_speculative(runtime, target, draft, request)
    } else {
        execute_ordinary(runtime, target, request)
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "restore, stateful sampler continuation, cleanup, and checkpoint receipt stay adjacent"
)]
fn execute_ordinary(
    runtime: &Runtime,
    target: &Model,
    mut request: TextMechanicsRequest<'_>,
) -> Result<TextMechanicsOutput, Error> {
    let activation_selected = request.target_activation.is_some();
    let mut continuation = match request.resume {
        Some(TextMechanicsResume::Ordinary(snapshot)) => Some((
            clone_sampler(&snapshot.sampler)?,
            snapshot.stop_tail.clone(),
        )),
        _ => None,
    };
    let resume = request.resume;
    let mut session = Session::new_ordinary_text_mechanics(
        target,
        runtime,
        request.options.target,
        request.capture_checkpoint || resume.is_some(),
        request.target_activation.take(),
    )?;
    if let Some(TextMechanicsResume::Ordinary(snapshot)) = resume {
        session.restore_envelope_state(snapshot.state(), None)?;
    }
    let loras = std::mem::take(&mut request.target_loras);
    let control_vector = request.target_control_vector;
    let steering_selected = !loras.is_empty() || control_vector.is_some();
    let ((prefill, generation, target_activation), steering_applied, steering_cleared) =
        execute_with_steering(&mut session, loras, control_vector, |session| {
            if resume.is_some() && steering_selected {
                session.refresh_restored_logits_with_active_steering(Some(
                    logit_loom::ActivationPhaseV1::Generation,
                ))?;
            }
            let prefill = if request.prompt.is_empty() {
                None
            } else if let Some(monitor) = request.prefill_monitor.as_deref_mut() {
                Some(session.prefill_observed(request.prompt, resume.is_none(), monitor)?)
            } else {
                Some(session.prefill(request.prompt, resume.is_none())?)
            };
            let prefill_stopped = prefill
                .as_ref()
                .and_then(|prefill| prefill.receipt.as_ref())
                .is_some_and(|receipt| receipt.finish == PrefillFinish::Stopped);
            let generation = if prefill_stopped {
                None
            } else {
                Some(session.generate_stateful(
                    &request.plan.generation,
                    continuation.take(),
                    request.pipeline.as_deref_mut(),
                    request.observers.as_deref_mut(),
                )?)
            };
            let target_activation = if activation_selected {
                Some(SpeculativeActivationOutput {
                    captures: session.take_activation_captures()?,
                    program: session.take_activation_program_output()?,
                })
            } else {
                None
            };
            Ok((prefill, generation, target_activation))
        })?;
    validate_steering_evidence(request.plan, &steering_applied, &steering_cleared)?;

    let (generation, sampler_state) = match generation {
        Some(StatefulGenerationOutput {
            output,
            sampler,
            stop_tail,
        }) => (Some(output), Some((sampler, stop_tail))),
        None => (None, None),
    };
    let checkpoint = if request.capture_checkpoint
        && let Some((sampler, stop_tail)) = sampler_state
    {
        let state = session.capture_envelope_state()?;
        let mechanics = request.plan.digest_for(target.topology(), None)?;
        let generation_identity = request.plan.generation.digest()?;
        let stop_tail_identity = Digest::of_bytes("ordinary-checkpoint-stop-tail-v2", &stop_tail);
        let sampler_lineage = Digest::of_serializable(
            "ordinary-text-sampler-lineage-v2",
            &(
                &mechanics,
                &generation_identity,
                &state.receipt().token_history,
                state.receipt().position,
                &stop_tail_identity,
                &request.plan.branch_checkpoint,
            ),
        )?;
        let receipt = TextMechanicsCheckpointReceiptV2 {
            mechanics,
            state: state.receipt().digest()?,
            generation: generation_identity.clone(),
            stop_tail: stop_tail_identity,
            stop_tail_bytes: u32::try_from(stop_tail.len()).map_err(|_| {
                Error::Invalid("ordinary checkpoint stop-prefix exceeds u32".to_owned())
            })?,
            sampler_lineage,
            position: state.receipt().position,
            parent: request.plan.branch_checkpoint.clone(),
        };
        receipt.digest_for(request.plan, target.topology(), state.receipt(), &stop_tail)?;
        Some(TextMechanicsCheckpoint::Ordinary(Box::new(
            OrdinaryTextMechanicsSnapshot {
                state,
                mechanics: request.plan.clone(),
                generation: generation_identity,
                stop_tail,
                sampler,
                receipt,
                thread_affinity: PhantomData,
            },
        )))
    } else {
        None
    };
    let receipt = ordinary_receipt(
        request.plan,
        target,
        prefill.as_ref(),
        generation.as_ref(),
        target_activation.as_ref(),
        checkpoint.as_ref(),
    )?;
    Ok(TextMechanicsOutput {
        execution: TextMechanicsExecutionOutput::Ordinary {
            prefill,
            generation,
            target_activation,
        },
        checkpoint,
        steering_applied,
        steering_cleared,
        receipt,
    })
}

#[allow(
    clippy::too_many_lines,
    reason = "fresh, restored, monitored, and checkpointed variants share one exact receipt exit"
)]
fn execute_speculative(
    runtime: &Runtime,
    target: &Model,
    draft: &Model,
    mut request: TextMechanicsRequest<'_>,
) -> Result<TextMechanicsOutput, Error> {
    let plan = request.plan.speculation.as_ref().ok_or_else(|| {
        Error::Invalid("speculative aggregate execution requires a speculation plan".to_owned())
    })?;
    let resume = request.resume;
    let loras = std::mem::take(&mut request.target_loras);
    let control_vector = request.target_control_vector;

    let (execution, checkpoint) = match resume {
        None => {
            let activations = supplied_speculative_activations(
                request.target_activation.take(),
                request.draft_activation.take(),
            )?;
            let mut native =
                SpeculativeRequest::new(request.prompt, &request.plan.generation, plan)
                    .with_options(request.options)
                    .with_activations(activations)
                    .with_target_steering(loras, control_vector);
            if let Some(pipeline) = request.pipeline {
                native = native.with_pipeline(pipeline);
            }
            if let Some(observers) = request.observers {
                native = native.with_observers(observers);
            }
            if let Some(monitor) = request.prefill_monitor {
                let checkpoint_request = request
                    .capture_checkpoint
                    .then(|| SpeculativeCheckpointRequest::new(request.plan.clone()));
                match generate_speculative_controlled(
                    runtime,
                    target,
                    draft,
                    native,
                    checkpoint_request.as_ref(),
                    monitor,
                )? {
                    SpeculativeExecutionOutcome::Generated(output) => {
                        let checkpoint = output.checkpoint.map(|checkpoint| {
                            TextMechanicsCheckpoint::Speculative(Box::new(checkpoint))
                        });
                        (output.generation, checkpoint)
                    }
                    SpeculativeExecutionOutcome::PrefillStopped(output) => {
                        return finish_speculative_prefill_stop(
                            request.plan,
                            target,
                            draft,
                            *output,
                        );
                    }
                }
            } else if request.capture_checkpoint {
                let checkpoint_request = SpeculativeCheckpointRequest::new(request.plan.clone());
                let output = generate_speculative_checkpointed(
                    runtime,
                    target,
                    draft,
                    native,
                    &checkpoint_request,
                )?;
                (
                    output.generation,
                    Some(TextMechanicsCheckpoint::Speculative(Box::new(
                        output.checkpoint,
                    ))),
                )
            } else {
                (generate_speculative(runtime, target, draft, native)?, None)
            }
        }
        Some(TextMechanicsResume::Speculative(parent)) => {
            let mut native = SpeculativeContinuationRequest::new(&request.plan.generation, plan)
                .with_options(request.options)
                .with_target_steering(loras, control_vector);
            if let Some(pipeline) = request.pipeline {
                native = native.with_pipeline(pipeline);
            }
            if let Some(observers) = request.observers {
                native = native.with_observers(observers);
            }
            if request.capture_checkpoint {
                let checkpoint_request = SpeculativeCheckpointRequest::new(request.plan.clone());
                let output = resume_speculative_checkpointed(
                    runtime,
                    target,
                    draft,
                    parent,
                    native,
                    &checkpoint_request,
                )?;
                (
                    output.generation,
                    Some(TextMechanicsCheckpoint::Speculative(Box::new(
                        output.checkpoint,
                    ))),
                )
            } else {
                (
                    resume_speculative(runtime, target, draft, parent, native)?,
                    None,
                )
            }
        }
        Some(TextMechanicsResume::Ordinary(_)) => {
            return Err(Error::Incompatible(
                "ordinary checkpoint reached speculative execution".to_owned(),
            ));
        }
    };

    let steering_applied = execution.steering_applied.clone();
    let steering_cleared = execution.steering_cleared.clone();
    validate_steering_evidence(request.plan, &steering_applied, &steering_cleared)?;
    let receipt =
        speculative_receipt(request.plan, target, draft, &execution, checkpoint.as_ref())?;
    Ok(TextMechanicsOutput {
        execution: TextMechanicsExecutionOutput::Speculative(Box::new(execution)),
        checkpoint,
        steering_applied,
        steering_cleared,
        receipt,
    })
}

fn finish_speculative_prefill_stop(
    plan: &TextMechanicsPlanV2,
    target: &Model,
    draft: &Model,
    output: SpeculativePrefillStoppedOutput,
) -> Result<TextMechanicsOutput, Error> {
    validate_steering_evidence(plan, &output.steering_applied, &output.steering_cleared)?;
    let (activation_captures, target_activation) =
        activation_evidence(output.target_activation.as_ref())?;
    let (_, draft_activation) = activation_evidence(output.draft_activation.as_ref())?;
    let speculation_plan = plan.speculation.as_ref().ok_or_else(|| {
        Error::Poisoned("speculative prefill stop lost its speculation plan".to_owned())
    })?;
    let speculation = SpeculationReceiptV1::from_boundaries(speculation_plan, &[])?;
    let receipt = TextMechanicsReceiptV2 {
        plan: plan.digest_for(target.topology(), Some(draft.topology()))?,
        prefill_receipt: Some(
            output
                .prefill
                .receipt
                .as_ref()
                .ok_or_else(|| {
                    Error::Poisoned(
                        "controlled speculative prefill stop omitted its receipt".to_owned(),
                    )
                })?
                .digest()?,
        ),
        generation_receipt: None,
        activation_captures,
        target_activation,
        draft_activation,
        speculation: Some(speculation.digest_for(speculation_plan, &[])?),
        checkpoint: None,
        branch_checkpoint: plan.branch_checkpoint.clone(),
        cleanup: cleanup_receipt(plan),
    };
    receipt.digest_for(plan, target.topology(), Some(draft.topology()))?;
    Ok(TextMechanicsOutput {
        execution: TextMechanicsExecutionOutput::SpeculativePrefillStopped {
            prefill: output.prefill,
            target_activation: output.target_activation,
            draft_activation: output.draft_activation,
        },
        checkpoint: None,
        steering_applied: output.steering_applied,
        steering_cleared: output.steering_cleared,
        receipt,
    })
}

fn supplied_speculative_activations(
    target: Option<ActivationConfiguration>,
    draft: Option<ActivationConfiguration>,
) -> Result<SpeculativeActivations, Error> {
    match (target, draft) {
        (None, None) => Ok(SpeculativeActivations::none()),
        (Some(target), None) => Ok(SpeculativeActivations::target_only(target)),
        (Some(target), Some(draft)) => Ok(SpeculativeActivations::separate(target, draft)),
        (None, Some(_)) => Err(Error::Invalid(
            "draft activation requires an independently selected target activation".to_owned(),
        )),
    }
}

fn validate_request(
    runtime: &Runtime,
    target: &Model,
    draft: Option<&Model>,
    request: &TextMechanicsRequest<'_>,
) -> Result<(), Error> {
    let plan = request.plan;
    plan.digest_for(target.topology(), draft.map(Model::topology))?;
    if target.topology().backend != *runtime.identity()
        || draft.is_some_and(|draft| draft.topology().backend != *runtime.identity())
    {
        return Err(Error::Incompatible(
            "aggregate model belongs to another runtime identity".to_owned(),
        ));
    }
    if plan.speculation.is_some() != draft.is_some() {
        return Err(Error::Invalid(
            "aggregate speculation and an exact draft model must be selected together".to_owned(),
        ));
    }

    let parent = validate_parent(target, draft, plan, request.resume)?;
    if parent != plan.branch_checkpoint {
        return Err(Error::Incompatible(
            "aggregate parent checkpoint differs from plan lineage".to_owned(),
        ));
    }
    let prompt_tokens = u32::try_from(request.prompt.len())
        .map_err(|_| Error::Invalid("aggregate prompt token count exceeds u32".to_owned()))?;
    match request.resume {
        None => {
            if prompt_tokens == 0 || prompt_tokens > plan.controlled_prefill_tokens {
                return Err(Error::Invalid(format!(
                    "fresh aggregate prompt must contain 1..={} tokens",
                    plan.controlled_prefill_tokens
                )));
            }
        }
        Some(TextMechanicsResume::Ordinary(_) | TextMechanicsResume::Speculative(_))
            if prompt_tokens != 0 =>
        {
            return Err(Error::Invalid(
                "a restored aggregate branch must not supply a second prompt".to_owned(),
            ));
        }
        Some(TextMechanicsResume::Ordinary(_) | TextMechanicsResume::Speculative(_)) => {}
    }
    if request.prefill_monitor.is_some() && request.prompt.is_empty() {
        return Err(Error::Invalid(
            "a controlled-prefill monitor requires non-empty prefill tokens".to_owned(),
        ));
    }
    let loras =
        validate_steering_resources(target, &request.target_loras, request.target_control_vector)?;
    if loras != plan.loras
        || request
            .target_control_vector
            .map(ControlVector::specification)
            != plan.control_vector.as_ref()
    {
        return Err(Error::Incompatible(
            "supplied target steering differs from the aggregate plan".to_owned(),
        ));
    }
    let pipeline = request
        .pipeline
        .as_deref()
        .map(|pipeline| pipeline.specification().digest())
        .transpose()?;
    if pipeline != plan.transform_pipeline {
        return Err(Error::Incompatible(
            "supplied transform pipeline differs from the aggregate plan".to_owned(),
        ));
    }
    let observers = request
        .observers
        .as_deref()
        .map(ObserverSet::identity)
        .transpose()?;
    if observers != plan.observer_set {
        return Err(Error::Incompatible(
            "supplied observer set differs from the aggregate plan".to_owned(),
        ));
    }
    validate_activation_selection(target, draft, request)?;
    validate_context_bound(request)?;
    Ok(())
}

fn validate_parent(
    target: &Model,
    draft: Option<&Model>,
    plan: &TextMechanicsPlanV2,
    resume: Option<TextMechanicsResume<'_>>,
) -> Result<Option<Digest>, Error> {
    match (plan.speculation.as_ref(), resume) {
        (None | Some(_), None) => Ok(None),
        (None, Some(TextMechanicsResume::Ordinary(snapshot))) => {
            validate_successor_mechanics(snapshot.mechanics(), plan, target, None)?;
            if snapshot.generation != plan.generation.digest()? {
                return Err(Error::Incompatible(
                    "ordinary checkpoint sampler belongs to another generation plan".to_owned(),
                ));
            }
            Ok(Some(snapshot.receipt().digest_for(
                snapshot.mechanics(),
                target.topology(),
                snapshot.state().receipt(),
                &snapshot.stop_tail,
            )?))
        }
        (Some(speculation), Some(TextMechanicsResume::Speculative(snapshot))) => {
            validate_successor_mechanics(snapshot.mechanics(), plan, target, draft)?;
            Ok(Some(snapshot.receipt().digest_for(speculation)?))
        }
        (None, Some(TextMechanicsResume::Speculative(_)))
        | (Some(_), Some(TextMechanicsResume::Ordinary(_))) => Err(Error::Incompatible(
            "aggregate checkpoint kind differs from the selected mechanics".to_owned(),
        )),
    }
}

fn validate_successor_mechanics(
    parent: &TextMechanicsPlanV2,
    successor: &TextMechanicsPlanV2,
    target: &Model,
    draft: Option<&Model>,
) -> Result<(), Error> {
    let mut parent = parent.clone();
    let mut successor = successor.clone();
    parent.branch_checkpoint = None;
    successor.branch_checkpoint = None;
    if parent.digest_for(target.topology(), draft.map(Model::topology))?
        != successor.digest_for(target.topology(), draft.map(Model::topology))?
    {
        return Err(Error::Incompatible(
            "successor mechanics differ from the checkpoint mechanics".to_owned(),
        ));
    }
    Ok(())
}

fn validate_activation_selection(
    target: &Model,
    draft: Option<&Model>,
    request: &TextMechanicsRequest<'_>,
) -> Result<(), Error> {
    let declared_target = request
        .plan
        .target_activation
        .as_ref()
        .map(|program| program.digest_for(target.topology()))
        .transpose()?;
    if matches!(request.resume, Some(TextMechanicsResume::Speculative(_))) {
        if request.target_activation.is_some() || request.draft_activation.is_some() {
            return Err(Error::Invalid(
                "a speculative checkpoint owns its activation runtimes".to_owned(),
            ));
        }
        if let Some(TextMechanicsResume::Speculative(snapshot)) = request.resume
            && (snapshot.receipt().target_activation != declared_target
                || snapshot.receipt().draft_activation
                    != request.plan.draft_activation_identity().cloned())
        {
            return Err(Error::Incompatible(
                "checkpoint activation lineage differs from the aggregate plan".to_owned(),
            ));
        }
        return Ok(());
    }

    let supplied_target = request
        .target_activation
        .as_ref()
        .map(ActivationConfiguration::program_identity)
        .cloned();
    let supplied_draft = request
        .draft_activation
        .as_ref()
        .map(ActivationConfiguration::program_identity)
        .cloned();
    if supplied_target != declared_target
        || supplied_draft != request.plan.draft_activation_identity().cloned()
    {
        return Err(Error::Incompatible(
            "supplied activation runtimes differ from the aggregate plan".to_owned(),
        ));
    }
    if request.draft_activation.is_some() && draft.is_none() {
        return Err(Error::Invalid(
            "draft activation requires an exact draft model".to_owned(),
        ));
    }
    Ok(())
}

fn validate_context_bound(request: &TextMechanicsRequest<'_>) -> Result<(), Error> {
    let prompt = u64::try_from(request.prompt.len())
        .map_err(|_| Error::Invalid("aggregate prompt length exceeds u64".to_owned()))?;
    let initial = match request.resume {
        None => prompt,
        Some(TextMechanicsResume::Ordinary(snapshot)) => snapshot.receipt().position,
        Some(TextMechanicsResume::Speculative(snapshot)) => snapshot.receipt().position,
    };
    let required = initial
        .checked_add(u64::from(request.plan.generation.max_tokens))
        .ok_or_else(|| Error::Invalid("aggregate context bound overflowed".to_owned()))?;
    if required > u64::from(request.options.target.context_size.get()) {
        return Err(Error::Invalid(format!(
            "target context must hold aggregate history and generation ({required} tokens)"
        )));
    }
    Ok(())
}

fn validate_steering_evidence(
    plan: &TextMechanicsPlanV2,
    applied: &[SteeringReceipt],
    cleared: &[SteeringReceipt],
) -> Result<(), Error> {
    let mut resources = plan
        .loras
        .iter()
        .map(|lora| lora.digest().map(|digest| (SteeringKind::Lora, digest)))
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(vector) = &plan.control_vector {
        resources.push((SteeringKind::ControlVector, vector.digest()?));
    }
    if applied.len() != resources.len()
        || applied
            .iter()
            .zip(&resources)
            .any(|(receipt, (kind, resource))| {
                receipt.action != SteeringAction::Applied
                    || receipt.kind != *kind
                    || receipt.resource != *resource
            })
        || cleared.len() != resources.len()
        || cleared
            .iter()
            .zip(resources.iter().rev())
            .any(|(receipt, (kind, resource))| {
                receipt.action != SteeringAction::Cleared
                    || receipt.kind != *kind
                    || receipt.resource != *resource
            })
    {
        return Err(Error::Poisoned(
            "aggregate steering lifecycle evidence is incomplete or out of order".to_owned(),
        ));
    }
    Ok(())
}

fn ordinary_receipt(
    plan: &TextMechanicsPlanV2,
    target: &Model,
    prefill: Option<&PrefillOutput>,
    generation: Option<&GenerationOutput>,
    activation: Option<&SpeculativeActivationOutput>,
    checkpoint: Option<&TextMechanicsCheckpoint>,
) -> Result<TextMechanicsReceiptV2, Error> {
    let (activation_captures, target_activation) = activation_evidence(activation)?;
    let receipt = TextMechanicsReceiptV2 {
        plan: plan.digest_for(target.topology(), None)?,
        prefill_receipt: prefill
            .and_then(|prefill| prefill.receipt.as_ref())
            .map(logit_loom::PrefillReceipt::digest)
            .transpose()?,
        generation_receipt: generation
            .map(|generation| generation.receipt.digest())
            .transpose()?,
        activation_captures,
        target_activation,
        draft_activation: None,
        speculation: None,
        checkpoint: checkpoint.map(checkpoint_identity).transpose()?,
        branch_checkpoint: plan.branch_checkpoint.clone(),
        cleanup: cleanup_receipt(plan),
    };
    receipt.digest_for(plan, target.topology(), None)?;
    Ok(receipt)
}

fn speculative_receipt(
    plan: &TextMechanicsPlanV2,
    target: &Model,
    draft: &Model,
    output: &SpeculativeGenerationOutput,
    checkpoint: Option<&TextMechanicsCheckpoint>,
) -> Result<TextMechanicsReceiptV2, Error> {
    let (activation_captures, target_activation) =
        activation_evidence(output.target_activation.as_ref())?;
    let (_, draft_activation) = activation_evidence(output.draft_activation.as_ref())?;
    let speculation_plan = plan.speculation.as_ref().ok_or_else(|| {
        Error::Poisoned("speculative receipt lost its speculation plan".to_owned())
    })?;
    let receipt = TextMechanicsReceiptV2 {
        plan: plan.digest_for(target.topology(), Some(draft.topology()))?,
        prefill_receipt: None,
        generation_receipt: Some(output.generation.receipt.digest()?),
        activation_captures,
        target_activation,
        draft_activation,
        speculation: Some(
            output
                .speculation
                .digest_for(speculation_plan, &output.boundaries)?,
        ),
        checkpoint: checkpoint.map(checkpoint_identity).transpose()?,
        branch_checkpoint: plan.branch_checkpoint.clone(),
        cleanup: cleanup_receipt(plan),
    };
    receipt.digest_for(plan, target.topology(), Some(draft.topology()))?;
    Ok(receipt)
}

fn activation_evidence(
    output: Option<&SpeculativeActivationOutput>,
) -> Result<(Vec<Digest>, Option<Digest>), Error> {
    let Some(output) = output else {
        return Ok((Vec::new(), None));
    };
    let captures = output
        .captures
        .receipts
        .iter()
        .map(|receipt| {
            Digest::of_serializable("activation-capture-receipt-v1", receipt).map_err(Error::from)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let program =
        Digest::of_serializable("activation-program-receipt-v1", &output.program.receipt)?;
    Ok((captures, Some(program)))
}

fn checkpoint_identity(checkpoint: &TextMechanicsCheckpoint) -> Result<Digest, Error> {
    match checkpoint {
        TextMechanicsCheckpoint::Ordinary(snapshot) => Ok(Digest::of_serializable(
            "text-mechanics-checkpoint-receipt-v2",
            snapshot.receipt(),
        )?),
        TextMechanicsCheckpoint::Speculative(snapshot) => Ok(Digest::of_serializable(
            "speculative-checkpoint-receipt-v1",
            snapshot.receipt(),
        )?),
    }
}

fn cleanup_receipt(plan: &TextMechanicsPlanV2) -> TextMechanicsCleanupReceiptV2 {
    TextMechanicsCleanupReceiptV2 {
        loras_removed: u32::try_from(plan.loras.len()).unwrap_or(u32::MAX),
        control_vector_removed: plan.control_vector.is_some(),
        target_activation_released: plan.target_activation.is_some(),
        draft_activation_released: plan.draft_activation_identity().is_some(),
        speculation_quiescent: plan.speculation.is_some(),
    }
}

#[cfg(test)]
mod tests {
    use logit_loom::{
        ControlVectorSpec, GenerationPlan, LoraSpec, SamplingPlan, TextModelTopologyV1,
    };

    use super::*;

    fn plan() -> TextMechanicsPlanV2 {
        let topology = TextModelTopologyV1 {
            model: Digest::of_bytes("test-model", b"one"),
            backend: Digest::of_bytes("test-backend", b"one"),
            architecture_implementation: Digest::of_bytes("test-architecture", b"one"),
            layers: 4,
            embedding_width: 2,
            experts: None,
            experts_used: None,
            nextn_layers: 0,
            supported_speculation: Vec::new(),
        };
        TextMechanicsPlanV2 {
            generation: GenerationPlan {
                sampling: SamplingPlan::default(),
                max_tokens: 4,
                biases: Vec::new(),
                grammar: None,
                stops: Vec::new(),
            },
            controlled_prefill_tokens: 8,
            transform_pipeline: None,
            observer_set: None,
            loras: vec![
                LoraSpec {
                    artifact: Digest::of_bytes("test-lora", b"one"),
                    scale: 0.5,
                },
                LoraSpec {
                    artifact: Digest::of_bytes("test-lora", b"two"),
                    scale: 1.0,
                },
            ],
            control_vector: Some(ControlVectorSpec {
                data: Digest::of_bytes("test-vector", b"one"),
                embedding_width: 2,
                layer_start: 1,
                layer_end: 3,
            }),
            target_topology: topology.digest().unwrap(),
            target_activation: None,
            activation_captures: Vec::new(),
            speculation: None,
            branch_checkpoint: None,
        }
    }

    #[test]
    fn aggregate_steering_evidence_requires_apply_and_reverse_cleanup_order() {
        let plan = plan();
        let mut resources = plan
            .loras
            .iter()
            .map(|lora| (SteeringKind::Lora, lora.digest().unwrap()))
            .collect::<Vec<_>>();
        resources.push((
            SteeringKind::ControlVector,
            plan.control_vector.as_ref().unwrap().digest().unwrap(),
        ));
        let applied = resources
            .iter()
            .map(|(kind, resource)| SteeringReceipt {
                kind: *kind,
                resource: resource.clone(),
                action: SteeringAction::Applied,
                position: 0,
            })
            .collect::<Vec<_>>();
        let mut cleared = resources
            .iter()
            .rev()
            .map(|(kind, resource)| SteeringReceipt {
                kind: *kind,
                resource: resource.clone(),
                action: SteeringAction::Cleared,
                position: 4,
            })
            .collect::<Vec<_>>();
        assert!(validate_steering_evidence(&plan, &applied, &cleared).is_ok());
        cleared.swap(0, 1);
        assert!(validate_steering_evidence(&plan, &applied, &cleared).is_err());
    }
}

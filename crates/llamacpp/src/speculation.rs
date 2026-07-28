// SPDX-License-Identifier: MIT OR Apache-2.0

//! Target-authoritative MTP and EAGLE-3 generation.

use std::panic::{AssertUnwindSafe, catch_unwind};

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
    GenerationPlan, GenerationReceipt, ObservedToken, ObserverSet, Pipeline,
    SpeculationActivationPolicyV1, SpeculationBoundaryReceiptV1, SpeculationPlanV1,
    SpeculationReceiptV1, TextSpeculativeMechanismV1, TokenId,
};

use crate::{
    ActivationCaptureOutput, ActivationConfiguration, ActivationProgramOutput, Error,
    GenerationOutput, LLAMA_CPP_BINDING_VERSION, LLAMA_CPP_REVISION, Model, Runtime, Session,
    SessionOptions, activation::ActivationController, error::native, sampler::build_sampler,
};

const SPECULATION_IMPLEMENTATION_DOMAIN: &str = "llamacpp-speculation-implementation-v1";
const SPECULATIVE_VOCABULARY_CHECK_START: i32 = 5;
const MAX_SPECULATIVE_VOCABULARY_DIFFERENCE: u32 = 128;

/// Returns the exact native and safe-wrapper implementation identity.
///
/// This identity changes when the binding version, pinned llama.cpp revision,
/// or Logit Loom lowering profile changes.
#[must_use]
pub fn speculation_implementation_identity() -> Digest {
    Digest::of_bytes(
        SPECULATION_IMPLEMENTATION_DOMAIN,
        format!("{LLAMA_CPP_BINDING_VERSION}|{LLAMA_CPP_REVISION}|target-authoritative-v1")
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
    mut request: SpeculativeRequest<'_>,
) -> Result<SpeculativeGenerationOutput, Error> {
    validate_request(runtime, target_model, draft_model, &request)?;
    let plan = request.speculation;
    let recurrent_slots = plan.maximum_draft_tokens;
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

    let target_side = SpeculativeSide {
        model: target.model,
        options: target.options,
        history: &mut target.token_history,
        position: &mut target.position,
        activation: &mut target.activation,
        poison: &mut target.poison_reason,
    };
    let draft_side = SpeculativeSide {
        model: draft.model,
        options: draft.options,
        history: &mut draft.token_history,
        position: &mut draft.position,
        activation: &mut draft.activation,
        poison: &mut draft.poison_reason,
    };

    match plan.mechanism {
        TextSpeculativeMechanismV1::Mtp => {
            let config = MtpSessionConfig::new(plan.sequences, maximum_draft)
                .with_n_min(minimum_draft)
                .with_p_min(probability_floor);
            let mut backend =
                MtpSession::new_with_config(&mut target.context, &mut draft.context, config)
                    .map_err(native)?;
            run_backend(
                &mut backend,
                target_side,
                draft_side,
                request.prompt,
                request.generation,
                plan,
                request.pipeline.as_deref_mut(),
                request.observers.as_deref_mut(),
            )
        }
        TextSpeculativeMechanismV1::Eagle3 => {
            let config = Eagle3SessionConfig::new(plan.sequences, maximum_draft)
                .with_n_min(minimum_draft)
                .with_p_min(probability_floor);
            let mut backend =
                Eagle3Session::new_with_config(&mut target.context, &mut draft.context, config)
                    .map_err(native)?;
            run_backend(
                &mut backend,
                target_side,
                draft_side,
                request.prompt,
                request.generation,
                plan,
                request.pipeline.as_deref_mut(),
                request.observers.as_deref_mut(),
            )
        }
    }
}

fn validate_request(
    runtime: &Runtime,
    target: &Model,
    draft: &Model,
    request: &SpeculativeRequest<'_>,
) -> Result<(), Error> {
    request.generation.validate()?;
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
            validate_eagle_layer_ids(draft.native.target_layer_ids(), target.topology().layers)?;
        }
    }
    Ok(())
}

fn validate_eagle_layer_ids(layer_ids: &[i32], target_layers: u32) -> Result<(), Error> {
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
        if layer >= target_layers {
            return Err(Error::Incompatible(format!(
                "EAGLE-3 target extraction layer {layer} is outside the target's {target_layers} layers"
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

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the operation keeps target authority, callback order, rollback, and receipt commit in one auditable loop"
)]
fn run_backend(
    backend: &mut impl NativeSpeculation,
    mut target: SpeculativeSide<'_, '_>,
    mut draft: SpeculativeSide<'_, '_>,
    prompt: &[TokenId],
    generation: &GenerationPlan,
    speculation: &SpeculationPlanV1,
    mut pipeline: Option<&mut Pipeline>,
    mut observers: Option<&mut ObserverSet>,
) -> Result<SpeculativeGenerationOutput, Error> {
    prefill(backend, &mut target, &mut draft, prompt)?;
    let native_prompt = prompt
        .iter()
        .map(|token| LlamaToken::new(token.get()))
        .collect::<Vec<_>>();
    backend.begin(&native_prompt)?;

    let initial_position = *target.position;
    if let Some(active) = pipeline.as_deref_mut() {
        active.begin(target.history)?;
    }
    if let Some(active) = observers.as_deref_mut() {
        active.begin(initial_position, generation.max_tokens)?;
    }
    let mut target_sampler = build_sampler(&target.model.native, generation, target.history)?;
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
            boundary_finish = stop_finish(generation, &output_bytes)?;
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
                boundary_finish = stop_finish(generation, &output_bytes)?;
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
    Ok(SpeculativeGenerationOutput {
        generation: GenerationOutput {
            bytes: output_bytes,
            tokens: output_tokens,
            receipt: generation_receipt,
        },
        boundaries,
        speculation: speculation_receipt,
        target_activation,
        draft_activation,
    })
}

fn prefill(
    backend: &mut impl NativeSpeculation,
    target: &mut SpeculativeSide<'_, '_>,
    draft: &mut SpeculativeSide<'_, '_>,
    prompt: &[TokenId],
) -> Result<(), Error> {
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
        offset = end;
    }
    Ok(())
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
    let mut sampled_sampler = catch_unwind(AssertUnwindSafe(|| sampler.clone_sampler()))
        .map_err(|_| Error::Native("native sampler clone panicked".to_owned()))?;
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
                format!("{LLAMA_CPP_BINDING_VERSION}|{LLAMA_CPP_REVISION}|target-authoritative-v1")
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
    fn eagle_layer_metadata_is_exact_and_in_range() {
        assert!(validate_eagle_layer_ids(&[1, 4, 7], 8).is_ok());
        assert!(validate_eagle_layer_ids(&[1, 4], 8).is_err());
        assert!(validate_eagle_layer_ids(&[1, -1, 7], 8).is_err());
        assert!(validate_eagle_layer_ids(&[1, 4, 8], 8).is_err());
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
}

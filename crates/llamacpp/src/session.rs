// SPDX-License-Identifier: MIT OR Apache-2.0

//! Single-owner causal sessions, generation, and checkpoints.

use std::marker::PhantomData;
use std::num::NonZeroU32;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;

use llama_cpp_4::context::LlamaContext;
use llama_cpp_4::context::params::LlamaContextParams;
use llama_cpp_4::llama_batch::LlamaBatch;
use llama_cpp_4::sampling::LlamaSampler;
use llama_cpp_4::token::LlamaToken;
use llama_cpp_4::token::data::LlamaTokenData;
use llama_cpp_4::token::data_array::LlamaTokenDataArray;
use logit_loom::{
    CheckpointReceipt, ControlFlow, Digest, GenerationFinish, GenerationPlan, GenerationReceipt,
    ObservedToken, ObserverSet, Pipeline, PrefillFinish, PrefillMonitor, PrefillProgress,
    PrefillReceipt, SteeringKind, TokenId,
};

use crate::{Error, Model, Runtime, error::native, sampler::build_sampler};

const STATE_BYTES_DOMAIN: &str = "llamacpp-state-bytes-v1";
const TOKEN_HISTORY_DOMAIN: &str = "llamacpp-state-token-history-v1";

/// Context-allocation options.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionOptions {
    /// Maximum causal context size.
    pub context_size: NonZeroU32,
    /// Maximum logical tokens per decode batch.
    pub batch_size: u32,
    /// Maximum physical tokens per decode micro-batch.
    pub micro_batch_size: u32,
    /// Host orchestration threads used by llama.cpp.
    pub threads: i32,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            context_size: NonZeroU32::new(4_096).expect("nonzero constant"),
            batch_size: 512,
            micro_batch_size: 512,
            threads: 4,
        }
    }
}

impl SessionOptions {
    fn validate(self) -> Result<(), Error> {
        let maximum_native_context = u32::try_from(i32::MAX).unwrap_or(u32::MAX);
        if self.batch_size == 0
            || self.micro_batch_size == 0
            || self.micro_batch_size > self.batch_size
            || self.batch_size > self.context_size.get()
            || self.context_size.get() > maximum_native_context
            || self.threads <= 0
        {
            return Err(Error::Invalid(
                "session requires 0 < micro_batch <= batch <= context and positive threads"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_options_reject_native_position_overflow() {
        let options = SessionOptions {
            context_size: NonZeroU32::new(u32::MAX).unwrap(),
            ..SessionOptions::default()
        };
        assert!(options.validate().is_err());
    }

    #[test]
    fn session_options_enforce_batch_and_thread_bounds() {
        let defaults = SessionOptions::default();
        assert!(defaults.validate().is_ok());
        assert!(
            SessionOptions {
                micro_batch_size: defaults.batch_size + 1,
                ..defaults
            }
            .validate()
            .is_err()
        );
        assert!(
            SessionOptions {
                threads: 0,
                ..defaults
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn checkpoint_compatibility_binds_session_options() {
        let runtime = Digest::of_bytes("test-runtime", b"one");
        let defaults = SessionOptions::default();
        let different_context = SessionOptions {
            context_size: NonZeroU32::new(defaults.context_size.get() + 1).unwrap(),
            ..defaults
        };
        let different_threads = SessionOptions {
            threads: defaults.threads + 1,
            ..defaults
        };
        let identity = session_compatibility_identity(&runtime, defaults);
        assert_ne!(
            identity,
            session_compatibility_identity(&runtime, different_context)
        );
        assert_ne!(
            identity,
            session_compatibility_identity(&runtime, different_threads)
        );
    }

    #[test]
    fn generation_output_preserves_arbitrary_bytes() {
        let output = GenerationOutput {
            bytes: vec![0xff],
            tokens: vec![TokenId::new(1).unwrap()],
            receipt: GenerationReceipt {
                plan: Digest::of_bytes("test-plan", b"bytes"),
                initial_position: 0,
                admitted_tokens: 1,
                admitted_bytes: 1,
                final_position: 1,
                finish: GenerationFinish::TokenLimit,
                transform_receipt: None,
                observer_receipts: Vec::new(),
            },
        };
        assert!(output.text().is_err());
        assert_eq!(output.bytes, [0xff]);
    }

    #[test]
    fn checkpoint_parts_validate_before_native_restore() {
        let bytes = vec![1, 2, 3];
        let tokens = vec![TokenId::new(4).unwrap()];
        let receipt = CheckpointReceipt {
            model: Digest::of_bytes("test-model", b"one"),
            backend: Digest::of_bytes("test-backend", b"one"),
            state: Digest::of_bytes(STATE_BYTES_DOMAIN, &bytes),
            state_bytes: 3,
            token_history: Digest::of_serializable(TOKEN_HISTORY_DOMAIN, &tokens).unwrap(),
            position: 1,
        };
        let snapshot =
            StateSnapshot::from_parts(bytes.clone(), tokens.clone(), receipt.clone()).unwrap();
        assert_eq!(snapshot.bytes(), bytes);
        assert_eq!(snapshot.tokens(), tokens);
        assert_eq!(snapshot.receipt(), &receipt);
        assert_eq!(snapshot.into_parts(), (bytes, tokens, receipt));
    }

    #[test]
    fn checkpoint_parts_reject_tampered_accounting() {
        let bytes = vec![1, 2, 3];
        let tokens = vec![TokenId::new(4).unwrap()];
        let mut receipt = CheckpointReceipt {
            model: Digest::of_bytes("test-model", b"one"),
            backend: Digest::of_bytes("test-backend", b"one"),
            state: Digest::of_bytes(STATE_BYTES_DOMAIN, &bytes),
            state_bytes: 4,
            token_history: Digest::of_serializable(TOKEN_HISTORY_DOMAIN, &tokens).unwrap(),
            position: 1,
        };
        assert!(StateSnapshot::from_parts(bytes.clone(), tokens.clone(), receipt.clone()).is_err());

        receipt.state_bytes = 3;
        receipt.position = 2;
        assert!(StateSnapshot::from_parts(bytes, tokens, receipt).is_err());
    }

    #[test]
    fn overlapping_stops_choose_the_lowest_declared_index() {
        let stops = vec![b"bc".to_vec(), b"abc".to_vec(), b"c".to_vec()];
        assert_eq!(matching_stop(&stops, b"zabc"), Some(0));
        assert_eq!(matching_stop(&stops, b"zab"), None);
    }
}

/// Result of one plain or controlled prefill call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefillOutput {
    /// Tokens admitted by this call.
    pub admitted_tokens: u64,
    /// Final causal position.
    pub position: u64,
    /// Controlled-prefill accounting when an observer was supplied.
    pub receipt: Option<PrefillReceipt>,
}

/// Successful bounded generation output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationOutput {
    /// Exact, potentially non-UTF-8 generated bytes.
    pub bytes: Vec<u8>,
    /// Causally admitted generated token IDs.
    pub tokens: Vec<TokenId>,
    /// Complete mechanical receipt.
    pub receipt: GenerationReceipt,
}

impl GenerationOutput {
    /// Returns generated bytes as UTF-8 when the complete output is valid.
    ///
    /// # Errors
    ///
    /// Returns the standard UTF-8 error for arbitrary model bytes.
    pub fn text(&self) -> Result<&str, std::str::Utf8Error> {
        std::str::from_utf8(&self.bytes)
    }
}

/// Opaque llama.cpp causal state with compatibility-bound identity metadata.
#[derive(Clone, Debug)]
pub struct StateSnapshot {
    bytes: Vec<u8>,
    tokens: Vec<TokenId>,
    receipt: CheckpointReceipt,
}

impl StateSnapshot {
    /// Reconstructs a checkpoint from caller-owned storage.
    ///
    /// This validates internal byte count, state identity, token lineage, and
    /// position. [`Session::restore_state`] separately verifies the model,
    /// backend build, and session allocation identities before passing bytes to
    /// llama.cpp.
    ///
    /// # Errors
    ///
    /// Returns an error for empty state bytes or inconsistent metadata.
    pub fn from_parts(
        bytes: Vec<u8>,
        tokens: Vec<TokenId>,
        receipt: CheckpointReceipt,
    ) -> Result<Self, Error> {
        let snapshot = Self {
            bytes,
            tokens,
            receipt,
        };
        snapshot.validate_contents()?;
        Ok(snapshot)
    }

    /// Returns opaque native state bytes covered by the checkpoint receipt.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns the exact causal tokens represented by the state.
    pub fn tokens(&self) -> &[TokenId] {
        &self.tokens
    }

    /// Returns serializable, compatibility-bound checkpoint metadata.
    pub const fn receipt(&self) -> &CheckpointReceipt {
        &self.receipt
    }

    /// Returns all caller-owned parts for application-defined persistence.
    pub fn into_parts(self) -> (Vec<u8>, Vec<TokenId>, CheckpointReceipt) {
        (self.bytes, self.tokens, self.receipt)
    }

    fn validate_contents(&self) -> Result<(), Error> {
        if self.bytes.is_empty() {
            return Err(Error::Incompatible(
                "checkpoint state bytes must not be empty".to_owned(),
            ));
        }
        let state_bytes = u64::try_from(self.bytes.len())
            .map_err(|_| Error::Incompatible("checkpoint state size exceeds u64".to_owned()))?;
        let token_count = u64::try_from(self.tokens.len())
            .map_err(|_| Error::Incompatible("checkpoint token count exceeds u64".to_owned()))?;
        if self.receipt.state != Digest::of_bytes(STATE_BYTES_DOMAIN, &self.bytes)
            || self.receipt.state_bytes != state_bytes
            || self.receipt.token_history
                != Digest::of_serializable(TOKEN_HISTORY_DOMAIN, &self.tokens)?
            || self.receipt.position != token_count
        {
            return Err(Error::Incompatible(
                "checkpoint metadata does not match state bytes or token lineage".to_owned(),
            ));
        }
        self.receipt.digest()?;
        Ok(())
    }
}

/// Single-owner llama.cpp context with explicit causal lineage.
///
/// `Session` is deliberately neither `Send` nor `Sync`. Create it inside the
/// thread that will own all mutation.
pub struct Session<'model> {
    pub(crate) context: LlamaContext<'model>,
    pub(crate) model: &'model Model,
    options: SessionOptions,
    backend: Digest,
    token_history: Vec<TokenId>,
    position: u64,
    active_steering: Option<SteeringKind>,
    poison_reason: Option<String>,
    thread_affinity: PhantomData<Rc<()>>,
}

impl std::fmt::Debug for Session<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Session")
            .field("options", &self.options)
            .field("position", &self.position)
            .field("tokens", &self.token_history.len())
            .field("active_steering", &self.active_steering)
            .field("healthy", &self.poison_reason.is_none())
            .finish_non_exhaustive()
    }
}

impl<'model> Session<'model> {
    pub(crate) fn new(
        model: &'model Model,
        runtime: &Runtime,
        options: SessionOptions,
    ) -> Result<Self, Error> {
        options.validate()?;
        let params = LlamaContextParams::default()
            .with_n_ctx(Some(options.context_size))
            .with_n_batch(options.batch_size)
            .with_n_ubatch(options.micro_batch_size)
            .with_n_threads(options.threads)
            .with_n_threads_batch(options.threads)
            .with_offload_kqv(true);
        let context = model
            .native
            .new_context(&runtime.native, params)
            .map_err(native)?;
        Ok(Self {
            context,
            model,
            options,
            backend: session_compatibility_identity(runtime.identity(), options),
            token_history: Vec::new(),
            position: 0,
            active_steering: None,
            poison_reason: None,
            thread_affinity: PhantomData,
        })
    }

    /// Returns the context-allocation options.
    pub const fn options(&self) -> SessionOptions {
        self.options
    }

    /// Returns the current causal position.
    pub const fn position(&self) -> u64 {
        self.position
    }

    /// Returns exact admitted token history.
    pub fn token_history(&self) -> &[TokenId] {
        &self.token_history
    }

    /// Returns whether no operation has left native state uncertain.
    pub const fn is_healthy(&self) -> bool {
        self.poison_reason.is_none()
    }

    /// Returns the retained reason native state became uncertain, if any.
    pub fn poison_reason(&self) -> Option<&str> {
        self.poison_reason.as_deref()
    }

    /// Returns the session-scoped steering resource currently active.
    pub const fn active_steering(&self) -> Option<SteeringKind> {
        self.active_steering
    }

    /// Clears native causal memory and local lineage.
    ///
    /// # Errors
    ///
    /// Returns the retained reason when the session is poisoned.
    pub fn clear(&mut self) -> Result<(), Error> {
        self.ensure_healthy()?;
        self.context.clear_kv_cache();
        self.token_history.clear();
        self.position = 0;
        Ok(())
    }

    /// Admits exact tokens in bounded native batches.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty or overflowing request or native decode
    /// failure. Complete chunks admitted before a native failure remain causal.
    pub fn prefill(
        &mut self,
        tokens: &[TokenId],
        clear_first: bool,
    ) -> Result<PrefillOutput, Error> {
        self.prefill_with_monitor(tokens, clear_first, None)
    }

    /// Admits exact tokens with polling and observation at complete chunks.
    ///
    /// Cooperative stop retains every previously completed chunk. Callback
    /// failure likewise leaves already admitted native work visible so callers
    /// can inspect or checkpoint the exact prefix.
    ///
    /// # Errors
    ///
    /// Returns a validation, observer, or native decode error.
    pub fn prefill_observed(
        &mut self,
        tokens: &[TokenId],
        clear_first: bool,
        monitor: &mut PrefillMonitor,
    ) -> Result<PrefillOutput, Error> {
        self.prefill_with_monitor(tokens, clear_first, Some(monitor))
    }

    fn prefill_with_monitor(
        &mut self,
        tokens: &[TokenId],
        clear_first: bool,
        mut monitor: Option<&mut PrefillMonitor>,
    ) -> Result<PrefillOutput, Error> {
        self.ensure_healthy()?;
        if tokens.is_empty() {
            return Err(Error::Invalid(
                "prefill tokens must not be empty".to_owned(),
            ));
        }
        self.model.validate_tokens(tokens)?;
        let initial_position = if clear_first { 0 } else { self.position };
        let requested_tokens = u64::try_from(tokens.len())
            .map_err(|_| Error::Invalid("prefill token count overflowed".to_owned()))?;
        let final_position = initial_position
            .checked_add(requested_tokens)
            .ok_or_else(|| Error::Invalid("prefill position overflowed".to_owned()))?;
        if final_position > u64::from(self.options.context_size.get()) {
            return Err(Error::Invalid(format!(
                "prefill would exceed context size {}",
                self.options.context_size
            )));
        }
        let mut progress = PrefillProgress {
            initial_position,
            requested_tokens,
            admitted_tokens: 0,
            admitted_chunks: 0,
            position: initial_position,
        };
        if let Some(active) = monitor.as_deref_mut() {
            active.begin(progress)?;
        }
        if clear_first {
            self.clear()?;
        }

        let chunk_limit = usize::try_from(self.options.batch_size)
            .map_err(|_| Error::Invalid("batch size exceeds usize".to_owned()))?;
        for chunk in tokens.chunks(chunk_limit) {
            if let Some(active) = monitor.as_deref_mut()
                && active.poll(progress)? == ControlFlow::Stop
            {
                let receipt = active.finish(PrefillFinish::Stopped)?;
                return Ok(PrefillOutput {
                    admitted_tokens: progress.admitted_tokens,
                    position: self.position,
                    receipt: Some(receipt),
                });
            }
            self.decode_tokens(chunk)?;
            let chunk_tokens = u64::try_from(chunk.len())
                .map_err(|_| Error::Invalid("prefill chunk size overflowed".to_owned()))?;
            progress.admitted_tokens = progress
                .admitted_tokens
                .checked_add(chunk_tokens)
                .ok_or_else(|| Error::Invalid("prefill token accounting overflowed".to_owned()))?;
            progress.admitted_chunks = progress
                .admitted_chunks
                .checked_add(1)
                .ok_or_else(|| Error::Invalid("prefill chunk accounting overflowed".to_owned()))?;
            progress.position = self.position;
            if let Some(active) = monitor.as_deref_mut()
                && active.observe_chunk(progress)? == ControlFlow::Stop
                && progress.admitted_tokens < progress.requested_tokens
            {
                let receipt = active.finish(PrefillFinish::Stopped)?;
                return Ok(PrefillOutput {
                    admitted_tokens: progress.admitted_tokens,
                    position: self.position,
                    receipt: Some(receipt),
                });
            }
        }
        let receipt = monitor
            .map(|active| active.finish(PrefillFinish::Complete))
            .transpose()?;
        Ok(PrefillOutput {
            admitted_tokens: progress.admitted_tokens,
            position: self.position,
            receipt,
        })
    }

    /// Generates through optional Logit Loom transforms and observers.
    ///
    /// Transform stages see raw candidate logits before native grammar, bias,
    /// filtering, temperature, and terminal sampling. The selected token is
    /// decoded before `accept` and observer callbacks, so those callbacks see
    /// only causal admissions. An end-of-generation token is not decoded.
    ///
    /// # Errors
    ///
    /// Returns a plan, pipeline, observer, detokenization, sampling, capacity,
    /// or decode error. Native state retains every token successfully decoded
    /// before the error.
    pub fn generate(
        &mut self,
        plan: &GenerationPlan,
        mut pipeline: Option<&mut Pipeline>,
        mut observers: Option<&mut ObserverSet>,
    ) -> Result<GenerationOutput, Error> {
        self.ensure_healthy()?;
        plan.validate()?;
        if self.token_history.is_empty() {
            return Err(Error::Invalid(
                "generation requires at least one prefilled token".to_owned(),
            ));
        }
        let plan_identity = plan.digest()?;
        let initial_position = self.position;
        if let Some(active) = pipeline.as_deref_mut() {
            active.begin(&self.token_history)?;
        }
        if let Some(active) = observers.as_deref_mut() {
            active.begin(initial_position, plan.max_tokens)?;
        }
        let mut sampler = build_sampler(&self.model.native, plan, &self.token_history)?;
        let mut output_tokens = Vec::new();
        let mut output_bytes = Vec::new();
        let mut finish = GenerationFinish::TokenLimit;

        for step in 0..plan.max_tokens {
            if let Some(active) = observers.as_deref_mut()
                && active.poll()? == ControlFlow::Stop
            {
                finish = GenerationFinish::ObserverStop;
                break;
            }
            if self.position >= u64::from(self.options.context_size.get()) {
                return Err(Error::Invalid(
                    "generation reached the configured context bound".to_owned(),
                ));
            }
            let token = self.sample_token(step, &mut sampler, pipeline.as_deref_mut())?;
            if self.model.is_end_of_generation(token) {
                finish = GenerationFinish::EndOfGeneration { token };
                break;
            }

            let piece = self.model.token_piece(token)?;
            self.decode_tokens(std::slice::from_ref(&token))?;
            sampler.accept(LlamaToken::new(token.get()));
            if let Some(active) = pipeline.as_deref_mut() {
                active.accept(token)?;
            }
            output_tokens.push(token);
            output_bytes.extend_from_slice(&piece);
            let mut observer_stop = false;
            if let Some(active) = observers.as_deref_mut()
                && active.observe(ObservedToken {
                    token,
                    piece: &piece,
                    position: self.position,
                })? == ControlFlow::Stop
            {
                observer_stop = true;
            }
            if observer_stop {
                finish = GenerationFinish::ObserverStop;
                break;
            }
            if let Some(index) = matching_stop(&plan.stops, &output_bytes) {
                let index = u32::try_from(index)
                    .map_err(|_| Error::Invalid("stop sequence index overflowed".to_owned()))?;
                finish = GenerationFinish::StopSequence { index };
                break;
            }
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
        let receipt = GenerationReceipt {
            plan: plan_identity,
            initial_position,
            admitted_tokens: u32::try_from(output_tokens.len())
                .map_err(|_| Error::Invalid("generated token accounting overflowed".to_owned()))?,
            admitted_bytes: u64::try_from(output_bytes.len())
                .map_err(|_| Error::Invalid("generated byte accounting overflowed".to_owned()))?,
            final_position: self.position,
            finish,
            transform_receipt,
            observer_receipts,
        };
        receipt.digest()?;
        Ok(GenerationOutput {
            bytes: output_bytes,
            tokens: output_tokens,
            receipt,
        })
    }

    fn sample_token(
        &mut self,
        step: u32,
        sampler: &mut LlamaSampler,
        pipeline: Option<&mut Pipeline>,
    ) -> Result<TokenId, Error> {
        let mut logits = catch_unwind(AssertUnwindSafe(|| self.context.get_logits().to_vec()))
            .map_err(|_| Error::Native("llama.cpp logits were unavailable".to_owned()))?;
        if let Some(active) = pipeline {
            active.apply_to_vocabulary(step, &self.token_history, &mut logits)?;
        }
        let data = logits
            .into_iter()
            .enumerate()
            .map(|(index, logit)| {
                let token = i32::try_from(index).map_err(|_| {
                    Error::Native("vocabulary token identifier overflowed".to_owned())
                })?;
                Ok(LlamaTokenData::new(LlamaToken::new(token), logit, 0.0))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let mut candidates = LlamaTokenDataArray::new(data, false);
        sampler.apply(&mut candidates);
        let selected = candidates
            .selected_token()
            .ok_or_else(|| Error::Native("native sampler did not select a token".to_owned()))?;
        TokenId::new(selected.0).map_err(Error::from)
    }

    /// Captures opaque native state and exact causal lineage.
    ///
    /// # Errors
    ///
    /// Returns an error when native state capture or identity encoding fails.
    pub fn capture_state(&mut self) -> Result<StateSnapshot, Error> {
        self.ensure_healthy()?;
        if self.active_steering.is_some() {
            return Err(Error::Invalid(
                "clear active steering before capturing a checkpoint".to_owned(),
            ));
        }
        let size = self.context.state_get_size();
        if size == 0 {
            return Err(Error::Native(
                "llama.cpp returned a zero-sized state".to_owned(),
            ));
        }
        let mut bytes = vec![0_u8; size];
        let written = self.context.state_get_data(&mut bytes);
        if written == 0 || written > bytes.len() {
            return Err(Error::Native(
                "llama.cpp returned invalid state accounting".to_owned(),
            ));
        }
        bytes.truncate(written);
        let receipt = CheckpointReceipt {
            model: self.model.artifact_digest().clone(),
            backend: self.backend.clone(),
            state: Digest::of_bytes(STATE_BYTES_DOMAIN, &bytes),
            state_bytes: u64::try_from(bytes.len())
                .map_err(|_| Error::Native("native state size exceeds u64".to_owned()))?,
            token_history: Digest::of_serializable(TOKEN_HISTORY_DOMAIN, &self.token_history)?,
            position: self.position,
        };
        StateSnapshot::from_parts(bytes, self.token_history.clone(), receipt)
    }

    /// Restores an exact compatible checkpoint.
    ///
    /// # Errors
    ///
    /// Returns an identity or native restore error. Local lineage changes only
    /// after llama.cpp accepts the complete state bytes.
    pub fn restore_state(&mut self, snapshot: &StateSnapshot) -> Result<(), Error> {
        self.ensure_healthy()?;
        if self.active_steering.is_some() {
            return Err(Error::Invalid(
                "clear active steering before restoring a checkpoint".to_owned(),
            ));
        }
        snapshot.validate_contents()?;
        if snapshot.receipt.model != *self.model.artifact_digest()
            || snapshot.receipt.backend != self.backend
        {
            return Err(Error::Incompatible(
                "checkpoint model or backend identity does not match".to_owned(),
            ));
        }
        if snapshot.receipt.position > u64::from(self.options.context_size.get()) {
            return Err(Error::Incompatible(
                "checkpoint position exceeds the destination context".to_owned(),
            ));
        }
        self.model.validate_tokens(&snapshot.tokens)?;
        let read = self.context.state_set_data(&snapshot.bytes);
        if read != snapshot.bytes.len() {
            let error = Error::Native(format!(
                "llama.cpp restored {read} of {} state bytes",
                snapshot.bytes.len()
            ));
            self.record_poison(&error);
            return Err(error);
        }
        self.token_history.clone_from(&snapshot.tokens);
        self.position = snapshot.receipt.position;
        Ok(())
    }

    fn decode_tokens(&mut self, tokens: &[TokenId]) -> Result<(), Error> {
        self.model.validate_tokens(tokens)?;
        let token_count = u64::try_from(tokens.len())
            .map_err(|_| Error::Invalid("native batch token count overflowed".to_owned()))?;
        let mut batch = LlamaBatch::new(tokens.len(), 1);
        for (offset, token) in tokens.iter().enumerate() {
            let offset = u64::try_from(offset)
                .map_err(|_| Error::Invalid("native batch offset overflowed".to_owned()))?;
            let position = self
                .position
                .checked_add(offset)
                .ok_or_else(|| Error::Invalid("native position overflowed".to_owned()))?;
            let position = i32::try_from(position)
                .map_err(|_| Error::Invalid("native position exceeds i32".to_owned()))?;
            batch
                .add(
                    LlamaToken::new(token.get()),
                    position,
                    &[0],
                    offset + 1 == token_count,
                )
                .map_err(native)?;
        }
        self.context.decode(&mut batch).map_err(native)?;
        self.token_history.extend_from_slice(tokens);
        self.position = self
            .position
            .checked_add(token_count)
            .ok_or_else(|| Error::Invalid("causal position overflowed".to_owned()))?;
        Ok(())
    }

    pub(crate) fn ensure_steering_available(&self) -> Result<(), Error> {
        self.ensure_healthy()?;
        if self.active_steering.is_some() {
            return Err(Error::Invalid(
                "only one steering resource may be active per session".to_owned(),
            ));
        }
        Ok(())
    }

    pub(crate) fn mark_steering_active(&mut self, kind: SteeringKind) {
        self.active_steering = Some(kind);
    }

    pub(crate) fn mark_steering_cleared(&mut self) {
        self.active_steering = None;
    }

    pub(crate) fn record_cleanup_failure(&mut self, error: &Error) {
        self.record_poison(error);
    }

    fn record_poison(&mut self, error: &Error) {
        self.poison_reason.get_or_insert_with(|| error.to_string());
    }

    fn ensure_healthy(&self) -> Result<(), Error> {
        if let Some(reason) = &self.poison_reason {
            return Err(Error::Poisoned(reason.clone()));
        }
        Ok(())
    }
}

fn matching_stop(stops: &[Vec<u8>], output: &[u8]) -> Option<usize> {
    stops.iter().position(|stop| output.ends_with(stop))
}

fn session_compatibility_identity(runtime: &Digest, options: SessionOptions) -> Digest {
    let mut bytes = Vec::with_capacity(runtime.as_str().len() + 16);
    bytes.extend_from_slice(runtime.as_str().as_bytes());
    bytes.extend_from_slice(&options.context_size.get().to_le_bytes());
    bytes.extend_from_slice(&options.batch_size.to_le_bytes());
    bytes.extend_from_slice(&options.micro_batch_size.to_le_bytes());
    bytes.extend_from_slice(&options.threads.to_le_bytes());
    Digest::of_bytes("llamacpp-session-compatibility-v1", &bytes)
}

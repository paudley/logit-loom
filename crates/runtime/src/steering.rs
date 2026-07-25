// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed high-level sessions retaining explicit steering cleanup.

use logit_loom::{PrefillMonitor, SteeringReceipt, TokenId};
use logit_loom_llamacpp::{
    ControlVector, ControlVectorScope, GenerationOutput, LoraAdapter, LoraScope, Model, Session,
    Tokenization,
};

use crate::{
    AdmissionOutput, GenerationRequest, LoomSession, Result,
    session::{admit_text, generate},
};

impl<'model> LoomSession<'model> {
    /// Applies a `LoRA` and borrows this session until explicit or drop cleanup.
    ///
    /// # Errors
    ///
    /// Returns a contract or native application error.
    pub fn lora<'scope>(
        &'scope mut self,
        adapter: &'scope mut LoraAdapter,
        scale: f32,
    ) -> Result<LoraSession<'scope, 'model>> {
        let inner = self.inner.lora_scope(adapter, scale)?;
        Ok(LoraSession {
            model: self.model,
            inner,
        })
    }

    /// Applies a control vector and borrows this session until cleanup.
    ///
    /// # Errors
    ///
    /// Returns a contract, model-compatibility, or native application error.
    pub fn control_vector<'scope>(
        &'scope mut self,
        vector: &'scope ControlVector,
    ) -> Result<ControlVectorSession<'scope, 'model>> {
        let inner = self.inner.control_vector_scope(vector)?;
        Ok(ControlVectorSession {
            model: self.model,
            inner,
        })
    }
}

/// High-level session while one `LoRA` remains active.
#[must_use = "retain the scope while LoRA should remain active and call clear to observe cleanup"]
pub struct LoraSession<'scope, 'model> {
    model: &'model Model,
    inner: LoraScope<'scope, 'model>,
}

impl<'model> LoraSession<'_, 'model> {
    /// Returns the successful application receipt.
    pub const fn applied_receipt(&self) -> &SteeringReceipt {
        self.inner.applied_receipt()
    }

    /// Replaces causal state with tokenized exact text under active steering.
    ///
    /// # Errors
    ///
    /// Returns a tokenization, capacity, validation, or native prefill error.
    pub fn replace_text(
        &mut self,
        text: &str,
        tokenization: Tokenization,
    ) -> Result<AdmissionOutput> {
        admit_text(
            self.inner.session_mut(),
            self.model,
            text,
            tokenization,
            true,
            None,
        )
    }

    /// Appends tokenized exact text under active steering.
    ///
    /// # Errors
    ///
    /// Returns a tokenization, capacity, validation, or native prefill error.
    pub fn append_text(
        &mut self,
        text: &str,
        tokenization: Tokenization,
    ) -> Result<AdmissionOutput> {
        admit_text(
            self.inner.session_mut(),
            self.model,
            text,
            tokenization,
            false,
            None,
        )
    }

    /// Replaces causal state while observing complete prefill chunks.
    ///
    /// # Errors
    ///
    /// Returns a tokenization, observer, capacity, or native prefill error.
    pub fn replace_text_observed(
        &mut self,
        text: &str,
        tokenization: Tokenization,
        monitor: &mut PrefillMonitor,
    ) -> Result<AdmissionOutput> {
        admit_text(
            self.inner.session_mut(),
            self.model,
            text,
            tokenization,
            true,
            Some(monitor),
        )
    }

    /// Appends exact text while observing complete prefill chunks.
    ///
    /// # Errors
    ///
    /// Returns a tokenization, observer, capacity, or native prefill error.
    pub fn append_text_observed(
        &mut self,
        text: &str,
        tokenization: Tokenization,
        monitor: &mut PrefillMonitor,
    ) -> Result<AdmissionOutput> {
        admit_text(
            self.inner.session_mut(),
            self.model,
            text,
            tokenization,
            false,
            Some(monitor),
        )
    }

    /// Executes one bounded generation request under active steering.
    ///
    /// # Errors
    ///
    /// Returns a plan, callback, sampling, capacity, or native error.
    pub fn generate(&mut self, request: GenerationRequest<'_>) -> Result<GenerationOutput> {
        generate(self.inner.session_mut(), request)
    }

    /// Returns the current causal position.
    pub fn position(&mut self) -> u64 {
        self.inner.session_mut().position()
    }

    /// Returns exact admitted token history.
    pub fn token_history(&mut self) -> &[TokenId] {
        self.inner.session_mut().token_history()
    }

    /// Returns the underlying mutable low-level session while steering remains active.
    pub fn raw_session_mut(&mut self) -> &mut Session<'model> {
        self.inner.session_mut()
    }

    /// Explicitly removes the `LoRA` and returns cleanup accounting.
    ///
    /// # Errors
    ///
    /// Returns a native cleanup or receipt error and poisons the session.
    pub fn clear(self) -> Result<SteeringReceipt> {
        self.inner.clear().map_err(Into::into)
    }
}

/// High-level session while one control vector remains active.
#[must_use = "retain the scope while the vector should remain active and call clear to observe cleanup"]
pub struct ControlVectorSession<'scope, 'model> {
    model: &'model Model,
    inner: ControlVectorScope<'scope, 'model>,
}

impl<'model> ControlVectorSession<'_, 'model> {
    /// Returns the successful application receipt.
    pub const fn applied_receipt(&self) -> &SteeringReceipt {
        self.inner.applied_receipt()
    }

    /// Replaces causal state with tokenized exact text under active steering.
    ///
    /// # Errors
    ///
    /// Returns a tokenization, capacity, validation, or native prefill error.
    pub fn replace_text(
        &mut self,
        text: &str,
        tokenization: Tokenization,
    ) -> Result<AdmissionOutput> {
        admit_text(
            self.inner.session_mut(),
            self.model,
            text,
            tokenization,
            true,
            None,
        )
    }

    /// Appends tokenized exact text under active steering.
    ///
    /// # Errors
    ///
    /// Returns a tokenization, capacity, validation, or native prefill error.
    pub fn append_text(
        &mut self,
        text: &str,
        tokenization: Tokenization,
    ) -> Result<AdmissionOutput> {
        admit_text(
            self.inner.session_mut(),
            self.model,
            text,
            tokenization,
            false,
            None,
        )
    }

    /// Replaces causal state while observing complete prefill chunks.
    ///
    /// # Errors
    ///
    /// Returns a tokenization, observer, capacity, or native prefill error.
    pub fn replace_text_observed(
        &mut self,
        text: &str,
        tokenization: Tokenization,
        monitor: &mut PrefillMonitor,
    ) -> Result<AdmissionOutput> {
        admit_text(
            self.inner.session_mut(),
            self.model,
            text,
            tokenization,
            true,
            Some(monitor),
        )
    }

    /// Appends exact text while observing complete prefill chunks.
    ///
    /// # Errors
    ///
    /// Returns a tokenization, observer, capacity, or native prefill error.
    pub fn append_text_observed(
        &mut self,
        text: &str,
        tokenization: Tokenization,
        monitor: &mut PrefillMonitor,
    ) -> Result<AdmissionOutput> {
        admit_text(
            self.inner.session_mut(),
            self.model,
            text,
            tokenization,
            false,
            Some(monitor),
        )
    }

    /// Executes one bounded generation request under active steering.
    ///
    /// # Errors
    ///
    /// Returns a plan, callback, sampling, capacity, or native error.
    pub fn generate(&mut self, request: GenerationRequest<'_>) -> Result<GenerationOutput> {
        generate(self.inner.session_mut(), request)
    }

    /// Returns the current causal position.
    pub fn position(&mut self) -> u64 {
        self.inner.session_mut().position()
    }

    /// Returns exact admitted token history.
    pub fn token_history(&mut self) -> &[TokenId] {
        self.inner.session_mut().token_history()
    }

    /// Returns the underlying mutable low-level session while steering remains active.
    pub fn raw_session_mut(&mut self) -> &mut Session<'model> {
        self.inner.session_mut()
    }

    /// Explicitly neutralizes the vector and returns cleanup accounting.
    ///
    /// # Errors
    ///
    /// Returns a native cleanup or receipt error and poisons the session.
    pub fn clear(self) -> Result<SteeringReceipt> {
        self.inner.clear().map_err(Into::into)
    }
}

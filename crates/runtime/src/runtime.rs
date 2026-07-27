// SPDX-License-Identifier: MIT OR Apache-2.0

//! Loaded model/runtime ownership and one-shot generation.

use std::path::Path;

use logit_loom::Digest;
use logit_loom_llamacpp::{
    LoraAdapter, Model, ModelOptions, Runtime, SessionOptions, Tokenization,
};
use logit_loom_models::{
    ArtifactReceipt, Catalog, QWEN3_SMALL_ARTIFACT_PATH, QWEN3_SMALL_PROFILE_ID,
    QWEN3_SMALL_SOURCE_ID,
};

use crate::{
    CompletionOutput, GenerationRequest, LoomSession, Result, session::admit_text,
    session::generate,
};

/// Native log handling selected before model loading.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NativeLogPolicy {
    /// Preserve llama.cpp's configured native logging.
    #[default]
    Preserve,
    /// Suppress llama.cpp log output for this process runtime.
    Silence,
}

/// Model loading, default session allocation, and native log options.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoomOptions {
    /// Native model placement and device requirements.
    pub model: ModelOptions,
    /// Allocation used by [`Loom::session`] and one-shot completion.
    pub session: SessionOptions,
    /// Native logging behavior applied before model loading.
    pub native_logs: NativeLogPolicy,
}

/// One loaded local model and its process-wide llama.cpp runtime.
///
/// The model field intentionally precedes the runtime field so native model
/// destruction occurs before backend shutdown.
pub struct Loom {
    model: Model,
    runtime: Runtime,
    options: LoomOptions,
    profile_artifact: Option<ArtifactReceipt>,
}

impl std::fmt::Debug for Loom {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Loom")
            .field("model", &self.model)
            .field("runtime", &self.runtime.compatibility_label())
            .field("options", &self.options)
            .field("profile_artifact", &self.profile_artifact)
            .finish()
    }
}

impl Loom {
    /// Initializes llama.cpp and loads one caller-supplied GGUF model.
    ///
    /// This method never downloads a model or retries a rejected placement as
    /// CPU-only inference.
    ///
    /// # Errors
    ///
    /// Returns an initialization, I/O, native load, identity, or device-policy
    /// error.
    pub fn load(path: impl AsRef<Path>, options: LoomOptions) -> Result<Self> {
        Self::load_inner(path.as_ref(), options, None)
    }

    /// Verifies and loads the catalogued Qwen3 0.6B `Q8_0` GGUF.
    ///
    /// The caller still selects the exact local path and every [`LoomOptions`]
    /// placement and allocation field. This method does not format a chat
    /// prompt, choose tokenization flags, allocate a session, or download
    /// artifacts.
    ///
    /// # Errors
    ///
    /// Returns a catalog, size, SHA-256, initialization, native load, identity,
    /// or device-policy error.
    pub fn load_qwen3_small(path: impl AsRef<Path>, options: LoomOptions) -> Result<Self> {
        let catalog = Catalog::embedded()?;
        let catalog_sha256 = catalog.packaged_sha256();
        let profile = catalog
            .find_profile(QWEN3_SMALL_PROFILE_ID)
            .ok_or_else(|| logit_loom_models::ArtifactError::Unknown {
                profile: QWEN3_SMALL_PROFILE_ID.to_owned(),
                source_id: QWEN3_SMALL_SOURCE_ID.to_owned(),
                path: QWEN3_SMALL_ARTIFACT_PATH.to_owned(),
            })?;
        let verified = profile.verify_artifact(
            &catalog_sha256,
            QWEN3_SMALL_SOURCE_ID,
            QWEN3_SMALL_ARTIFACT_PATH,
            path,
        )?;
        let loaded = Self::load_inner(verified.path(), options, Some(verified.receipt().clone()))?;
        profile.verify_artifact(
            &catalog_sha256,
            QWEN3_SMALL_SOURCE_ID,
            QWEN3_SMALL_ARTIFACT_PATH,
            verified.path(),
        )?;
        Ok(loaded)
    }

    fn load_inner(
        path: &Path,
        options: LoomOptions,
        profile_artifact: Option<ArtifactReceipt>,
    ) -> Result<Self> {
        let mut runtime = Runtime::initialize()?;
        if options.native_logs == NativeLogPolicy::Silence {
            runtime.silence_native_logs();
        }
        let model = Model::load(&runtime, path, options.model)?;
        Ok(Self {
            model,
            runtime,
            options,
            profile_artifact,
        })
    }

    /// Returns the exact options used for model loading and default sessions.
    pub const fn options(&self) -> LoomOptions {
        self.options
    }

    /// Returns the model's reported native device descriptions.
    pub fn devices(&self) -> &[String] {
        self.model.devices()
    }

    /// Returns the content identity of the exact loaded GGUF bytes.
    pub const fn model_identity(&self) -> &Digest {
        self.model.artifact_digest()
    }

    /// Returns the native build identity bound into checkpoints.
    pub const fn backend_identity(&self) -> &Digest {
        self.runtime.identity()
    }

    /// Returns the readable native build label bound into checkpoints.
    pub fn backend_compatibility(&self) -> &str {
        self.runtime.compatibility_label()
    }

    /// Returns exact profile verification evidence when loaded through a
    /// profile-specific constructor.
    pub const fn profile_artifact(&self) -> Option<&ArtifactReceipt> {
        self.profile_artifact.as_ref()
    }

    /// Returns the underlying loaded model.
    pub const fn raw_model(&self) -> &Model {
        &self.model
    }

    /// Returns the underlying process runtime.
    pub const fn raw_runtime(&self) -> &Runtime {
        &self.runtime
    }

    /// Creates a borrowing session with the configured default allocation.
    ///
    /// # Errors
    ///
    /// Returns an allocation or native context error.
    pub fn session(&self) -> Result<LoomSession<'_>> {
        self.session_with(self.options.session)
    }

    /// Creates a borrowing session with an explicit allocation contract.
    ///
    /// # Errors
    ///
    /// Returns a validation, allocation, or native context error.
    pub fn session_with(&self, options: SessionOptions) -> Result<LoomSession<'_>> {
        let session = self.model.session(&self.runtime, options)?;
        Ok(LoomSession::new(&self.model, session))
    }

    /// Loads a model-compatible `LoRA` without applying it.
    ///
    /// # Errors
    ///
    /// Returns an I/O, identity, or native compatibility error.
    pub fn load_lora(&self, path: impl AsRef<Path>) -> Result<LoraAdapter> {
        self.model.load_lora(path).map_err(Into::into)
    }

    /// Runs one fresh prompt replacement and bounded generation call.
    ///
    /// # Errors
    ///
    /// Returns a tokenization, prefill, request-control, sampling, callback, or
    /// native execution error.
    pub fn complete(
        &self,
        text: &str,
        tokenization: Tokenization,
        request: GenerationRequest<'_>,
    ) -> Result<CompletionOutput> {
        let mut session = self.model.session(&self.runtime, self.options.session)?;
        let admission = admit_text(&mut session, &self.model, text, tokenization, true, None)?;
        let generation = generate(&mut session, request)?;
        Ok(CompletionOutput {
            admission,
            generation,
        })
    }
}

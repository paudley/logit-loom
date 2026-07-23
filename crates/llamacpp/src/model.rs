// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native backend and model ownership.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use llama_cpp_4::llama_backend::LlamaBackend;
use llama_cpp_4::model::params::LlamaModelParams;
use llama_cpp_4::model::{AddBos, LlamaBackendDeviceType, LlamaModel, Special};
use llama_cpp_4::token::LlamaToken;
use logit_loom::{Digest, TokenId};

use crate::{Error, LLAMA_CPP_BINDING_VERSION, Session, SessionOptions, error::native};

const MODEL_ARTIFACT_DOMAIN: &str = "llamacpp-model-file-blake3-v1";
pub(crate) const LORA_ARTIFACT_DOMAIN: &str = "llamacpp-lora-file-blake3-v1";

/// Maximum UTF-8 bytes accepted by one native tokenization call.
pub const MAX_TOKENIZATION_BYTES: usize = 16 * 1024 * 1024;

/// Process-wide initialized llama.cpp backend.
#[derive(Debug)]
pub struct Runtime {
    pub(crate) native: LlamaBackend,
    identity: Digest,
    compatibility: String,
}

impl Runtime {
    /// Initializes llama.cpp once for the process.
    ///
    /// # Errors
    ///
    /// Returns an error if another owner already initialized the binding or
    /// llama.cpp initialization fails.
    pub fn initialize() -> Result<Self, Error> {
        let native = LlamaBackend::init().map_err(native)?;
        let compatibility = compatibility_label();
        Ok(Self {
            native,
            identity: Digest::of_bytes("llamacpp-binding-identity-v1", compatibility.as_bytes()),
            compatibility,
        })
    }

    /// Returns the native binding compatibility identity.
    pub const fn identity(&self) -> &Digest {
        &self.identity
    }

    /// Returns the readable build compatibility label bound into checkpoints.
    pub fn compatibility_label(&self) -> &str {
        &self.compatibility
    }

    /// Suppresses llama.cpp log output for this process.
    pub fn silence_native_logs(&mut self) {
        self.native.void_logs();
    }
}

/// Runtime placement requirement checked after model loading.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DevicePolicy {
    /// Reject a model with no accelerator-backed tensors.
    #[default]
    RequireAccelerator,
    /// Permit any placement selected by llama.cpp.
    Any,
}

/// Model-loading options exposed by the first adapter release.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelOptions {
    /// Number of repeating layers requested for GPU offload.
    pub gpu_layers: u32,
    /// Primary accelerator index.
    pub main_gpu: i32,
    /// Post-load placement requirement.
    pub device_policy: DevicePolicy,
}

impl Default for ModelOptions {
    fn default() -> Self {
        Self {
            gpu_layers: u32::MAX,
            main_gpu: 0,
            device_policy: DevicePolicy::RequireAccelerator,
        }
    }
}

/// Tokenization flags for exact text admission.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tokenization {
    /// Add the model's beginning-of-sequence token.
    pub add_bos: bool,
}

/// Immutable loaded model with content and placement identity.
pub struct Model {
    pub(crate) native: LlamaModel,
    artifact: Digest,
    devices: Vec<String>,
}

impl std::fmt::Debug for Model {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Model")
            .field("artifact", &self.artifact)
            .field("devices", &self.devices)
            .finish_non_exhaustive()
    }
}

impl Model {
    /// Loads one GGUF model and verifies the selected device policy.
    ///
    /// # Errors
    ///
    /// Returns an I/O, native load, or placement error.
    pub fn load(
        runtime: &Runtime,
        path: impl AsRef<Path>,
        options: ModelOptions,
    ) -> Result<Self, Error> {
        if options.main_gpu < 0 {
            return Err(Error::Invalid(
                "main GPU index must be non-negative".to_owned(),
            ));
        }
        let path = path.as_ref();
        let artifact_before_load = digest_file(path, MODEL_ARTIFACT_DOMAIN)?;
        let params = LlamaModelParams::default()
            .with_n_gpu_layers(options.gpu_layers)
            .with_main_gpu(options.main_gpu);
        let native_model =
            LlamaModel::load_from_file(&runtime.native, path, &params).map_err(native)?;
        validate_vocabulary_size(native_model.n_vocab())?;
        let artifact = digest_file(path, MODEL_ARTIFACT_DOMAIN)?;
        if artifact != artifact_before_load {
            return Err(Error::Incompatible(
                "model file changed while llama.cpp was loading it".to_owned(),
            ));
        }
        let devices = native_model
            .devices()
            .map(|device| {
                let name = device.name().unwrap_or("unknown");
                format!("{name}:{:?}", device.device_type())
            })
            .collect::<Vec<_>>();
        if options.device_policy == DevicePolicy::RequireAccelerator
            && !native_model.devices().any(|device| {
                matches!(
                    device.device_type(),
                    LlamaBackendDeviceType::Gpu
                        | LlamaBackendDeviceType::IntegratedGpu
                        | LlamaBackendDeviceType::Accel
                )
            })
        {
            return Err(Error::Native(
                "model placement contains no accelerator device".to_owned(),
            ));
        }
        Ok(Self {
            native: native_model,
            artifact,
            devices,
        })
    }

    /// Returns the content identity of the exact GGUF bytes.
    pub const fn artifact_digest(&self) -> &Digest {
        &self.artifact
    }

    /// Returns observed llama.cpp device descriptions.
    pub fn devices(&self) -> &[String] {
        &self.devices
    }

    /// Returns the model vocabulary size.
    ///
    /// # Errors
    ///
    /// Returns an error if llama.cpp reports a non-positive size.
    pub fn vocabulary_size(&self) -> Result<u32, Error> {
        validate_vocabulary_size(self.native.n_vocab())?;
        u32::try_from(self.native.n_vocab())
            .map_err(|_| Error::Native("model vocabulary size exceeds u32".to_owned()))
    }

    /// Tokenizes exact UTF-8 input with the model tokenizer.
    ///
    /// # Errors
    ///
    /// Returns an error when the input is oversized or contains NUL, or when
    /// native tokenization fails or emits a negative ID.
    pub fn tokenize(&self, text: &str, options: Tokenization) -> Result<Vec<TokenId>, Error> {
        validate_tokenization_input(text)?;
        let add_bos = if options.add_bos {
            AddBos::Always
        } else {
            AddBos::Never
        };
        let tokens = self
            .native
            .str_to_token(text, add_bos)
            .map_err(native)?
            .into_iter()
            .map(|token| TokenId::new(token.0).map_err(Error::from))
            .collect::<Result<Vec<_>, _>>()?;
        self.validate_tokens(&tokens)?;
        Ok(tokens)
    }

    /// Returns one exact, potentially non-UTF-8 token piece.
    ///
    /// # Errors
    ///
    /// Returns an error when the token is outside the loaded vocabulary or
    /// native detokenization fails.
    pub fn token_piece(&self, token: TokenId) -> Result<Vec<u8>, Error> {
        self.validate_tokens(std::slice::from_ref(&token))?;
        self.native
            .token_to_raw_bytes(LlamaToken::new(token.get()), Special::Tokenize)
            .map_err(native)
    }

    /// Returns whether a token terminates generation.
    pub fn is_end_of_generation(&self, token: TokenId) -> bool {
        self.native.is_eog_token(LlamaToken::new(token.get()))
    }

    pub(crate) fn validate_tokens(&self, tokens: &[TokenId]) -> Result<(), Error> {
        validate_token_ids(self.native.n_vocab(), tokens)
    }

    /// Creates one single-owner causal session.
    ///
    /// # Errors
    ///
    /// Returns a validation or native context-allocation error.
    pub fn session<'model>(
        &'model self,
        runtime: &Runtime,
        options: SessionOptions,
    ) -> Result<Session<'model>, Error> {
        Session::new(self, runtime, options)
    }
}

fn validate_token_ids(n_vocab: i32, tokens: &[TokenId]) -> Result<(), Error> {
    validate_vocabulary_size(n_vocab)?;
    if let Some(token) = tokens.iter().find(|token| token.get() >= n_vocab) {
        return Err(Error::Invalid(format!(
            "token {} is outside vocabulary size {n_vocab}",
            token.get()
        )));
    }
    Ok(())
}

pub(crate) fn digest_file(path: &Path, domain: &str) -> Result<Digest, Error> {
    let mut file = File::open(path)?;
    digest_reader(&mut file, domain)
}

fn digest_reader(reader: &mut impl Read, domain: &str) -> Result<Digest, Error> {
    let mut hasher = blake3::Hasher::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(Digest::of_bytes(domain, hasher.finalize().as_bytes()))
}

fn validate_tokenization_input(text: &str) -> Result<(), Error> {
    validate_tokenization_shape(text.len(), text.as_bytes().contains(&0))
}

fn validate_tokenization_shape(bytes: usize, contains_nul: bool) -> Result<(), Error> {
    if bytes > MAX_TOKENIZATION_BYTES {
        return Err(Error::Invalid(format!(
            "tokenization input exceeds {MAX_TOKENIZATION_BYTES} UTF-8 bytes"
        )));
    }
    if contains_nul {
        return Err(Error::Invalid(
            "tokenization input must not contain NUL bytes".to_owned(),
        ));
    }
    Ok(())
}

fn compatibility_label() -> String {
    let features = [
        ("blas", cfg!(feature = "blas")),
        ("cuda", cfg!(feature = "cuda")),
        ("dynamic-link", cfg!(feature = "dynamic-link")),
        ("hip", cfg!(feature = "hip")),
        ("metal", cfg!(feature = "metal")),
        ("native-cpu", cfg!(feature = "native-cpu")),
        ("opencl", cfg!(feature = "opencl")),
        ("openmp", cfg!(feature = "openmp")),
        ("prebuilt", cfg!(feature = "prebuilt")),
        ("rpc", cfg!(feature = "rpc")),
        ("vulkan", cfg!(feature = "vulkan")),
        ("webgpu", cfg!(feature = "webgpu")),
    ]
    .into_iter()
    .filter_map(|(name, enabled)| enabled.then_some(name))
    .collect::<Vec<_>>();
    let features = if features.is_empty() {
        "baseline".to_owned()
    } else {
        features.join(",")
    };
    format!(
        "{LLAMA_CPP_BINDING_VERSION};logit-loom={};target={}-{}-{};features={features}",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::ARCH,
        std::env::consts::OS,
        if cfg!(target_endian = "little") {
            "little"
        } else {
            "big"
        },
    )
}

fn validate_vocabulary_size(n_vocab: i32) -> Result<(), Error> {
    if n_vocab <= 0 {
        return Err(Error::Native(
            "model returned a non-positive vocabulary size".to_owned(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn compatibility_label_binds_version_target_and_features() {
        let label = compatibility_label();
        assert!(label.contains(LLAMA_CPP_BINDING_VERSION));
        assert!(label.contains(std::env::consts::ARCH));
        assert!(label.contains(std::env::consts::OS));
        assert!(label.contains("features="));
    }

    #[test]
    fn artifact_domains_separate_models_and_lora_adapters() {
        let bytes = b"same artifact bytes";
        let model = digest_reader(&mut Cursor::new(bytes), MODEL_ARTIFACT_DOMAIN).unwrap();
        let lora = digest_reader(&mut Cursor::new(bytes), LORA_ARTIFACT_DOMAIN).unwrap();
        assert_ne!(model, lora);
    }

    #[test]
    fn tokenization_inputs_are_bounded_before_native_calls() {
        assert!(validate_tokenization_input("hello").is_ok());
        assert!(validate_tokenization_input("hello\0world").is_err());
        assert!(validate_tokenization_shape(MAX_TOKENIZATION_BYTES + 1, false).is_err());
    }

    #[test]
    fn vocabulary_size_must_be_positive() {
        assert!(validate_vocabulary_size(1).is_ok());
        assert!(validate_vocabulary_size(0).is_err());
        assert!(validate_vocabulary_size(-1).is_err());
    }

    #[test]
    fn token_ids_must_fit_the_loaded_vocabulary() {
        let inside = TokenId::new(9).unwrap();
        let outside = TokenId::new(10).unwrap();
        assert!(validate_token_ids(10, &[inside]).is_ok());
        assert!(validate_token_ids(10, &[outside]).is_err());
    }
}

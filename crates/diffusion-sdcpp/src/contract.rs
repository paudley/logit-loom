// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use logit_loom_diffusion::{
    ControlFlow, DiffusionCheckpointReceipt, DiffusionPlan, DiffusionSchedule, Digest,
    MAX_DIFFUSION_STEPS, MAX_TENSOR_ELEMENTS, StepContext,
};
use logit_loom_models::ArtifactReceipt;
use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// Maximum prompt bytes passed to the native conditioner.
pub const MAX_PROMPT_BYTES: usize = 64 * 1024;
/// Maximum output image dimension.
pub const MAX_IMAGE_DIMENSION: u32 = 4_096;
/// Maximum native backend label bytes.
pub const MAX_BACKEND_LABEL_BYTES: usize = 128;

/// Exact catalogued image profile understood by the companion ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Profile {
    /// MiniT2I-B/16 with FLAN-T5-Large.
    MiniT2iB16,
    /// Krea 2 Turbo with Qwen3-VL and Wan 2.1 VAE.
    Krea2Turbo,
}

impl Profile {
    /// Returns the catalog profile ID.
    pub const fn id(self) -> &'static str {
        match self {
            Self::MiniT2iB16 => "minit2i-b16",
            Self::Krea2Turbo => "krea-2-turbo",
        }
    }

    pub(crate) const fn native_id(self) -> i32 {
        match self {
            Self::MiniT2iB16 => crate::ffi::PROFILE_MINIT2I,
            Self::Krea2Turbo => crate::ffi::PROFILE_KREA2,
        }
    }

    pub(crate) fn validate_dimensions(self, width: u32, height: u32) -> Result<()> {
        let alignment = match self {
            Self::MiniT2iB16 => 16,
            Self::Krea2Turbo => 64,
        };
        if width < alignment
            || height < alignment
            || width > MAX_IMAGE_DIMENSION
            || height > MAX_IMAGE_DIMENSION
            || !width.is_multiple_of(alignment)
            || !height.is_multiple_of(alignment)
        {
            return Err(Error::Invalid(format!(
                "{} dimensions must be aligned to {alignment} and within \
                 {alignment}..={MAX_IMAGE_DIMENSION}",
                self.id()
            )));
        }
        let elements = match self {
            Self::MiniT2iB16 => u64::from(width)
                .checked_mul(u64::from(height))
                .and_then(|pixels| pixels.checked_mul(3)),
            Self::Krea2Turbo => u64::from(width / 8)
                .checked_mul(u64::from(height / 8))
                .and_then(|pixels| pixels.checked_mul(16)),
        }
        .ok_or_else(|| Error::Invalid("profile state element count overflowed".to_owned()))?;
        if elements > MAX_TENSOR_ELEMENTS {
            return Err(Error::Invalid(format!(
                "{} state has {elements} elements; maximum is {MAX_TENSOR_ELEMENTS}",
                self.id()
            )));
        }
        Ok(())
    }
}

/// Caller-selected paths for exact catalogued model components.
#[derive(Clone, Debug)]
pub enum ProfileArtifacts {
    /// `MiniT2I` diffusion transformer and FLAN-T5-Large weights.
    MiniT2i {
        /// `diffusion_pytorch_model.safetensors`.
        diffusion_model: PathBuf,
        /// FLAN-T5-Large `model.safetensors`.
        text_encoder: PathBuf,
    },
    /// Krea diffusion GGUF, Qwen3-VL GGUF, Wan VAE, and gated license.
    Krea2 {
        /// `Krea-2-Turbo-Q6_K.gguf`.
        diffusion_model: PathBuf,
        /// `Qwen3VL-4B-Instruct-Q4_K_M.gguf`.
        text_encoder: PathBuf,
        /// `wan_2.1_vae.safetensors`.
        vae: PathBuf,
        /// Official gated `LICENSE.pdf`.
        license: PathBuf,
    },
}

impl ProfileArtifacts {
    /// Creates the exact `MiniT2I` component layout.
    pub fn minit2i(diffusion_model: impl Into<PathBuf>, text_encoder: impl Into<PathBuf>) -> Self {
        Self::MiniT2i {
            diffusion_model: diffusion_model.into(),
            text_encoder: text_encoder.into(),
        }
    }

    /// Creates the exact Krea 2 component layout.
    pub fn krea2(
        diffusion_model: impl Into<PathBuf>,
        text_encoder: impl Into<PathBuf>,
        vae: impl Into<PathBuf>,
        license: impl Into<PathBuf>,
    ) -> Self {
        Self::Krea2 {
            diffusion_model: diffusion_model.into(),
            text_encoder: text_encoder.into(),
            vae: vae.into(),
            license: license.into(),
        }
    }

    /// Returns the profile selected by this component shape.
    pub const fn profile(&self) -> Profile {
        match self {
            Self::MiniT2i { .. } => Profile::MiniT2iB16,
            Self::Krea2 { .. } => Profile::Krea2Turbo,
        }
    }

    pub(crate) fn diffusion_model(&self) -> &Path {
        match self {
            Self::MiniT2i {
                diffusion_model, ..
            }
            | Self::Krea2 {
                diffusion_model, ..
            } => diffusion_model,
        }
    }

    pub(crate) fn text_encoder(&self) -> &Path {
        match self {
            Self::MiniT2i { text_encoder, .. } | Self::Krea2 { text_encoder, .. } => text_encoder,
        }
    }

    pub(crate) fn vae(&self) -> Option<&Path> {
        match self {
            Self::MiniT2i { .. } => None,
            Self::Krea2 { vae, .. } => Some(vae),
        }
    }
}

/// Explicit native placement and loading options.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SdcppOptions {
    /// Accelerator backend used for model evaluation.
    pub backend: String,
    /// Backend used to retain model parameters.
    pub params_backend: String,
    /// Positive native host-orchestration thread count.
    pub threads: u32,
    /// Permit memory mapping of supported artifacts.
    pub enable_mmap: bool,
    /// Enable general native flash attention.
    pub flash_attention: bool,
    /// Enable diffusion-model flash attention.
    pub diffusion_flash_attention: bool,
}

impl SdcppOptions {
    /// Creates explicit native placement options.
    ///
    /// # Errors
    ///
    /// Returns an error for a CPU-named, empty, oversized, or NUL-containing
    /// backend or a zero thread count.
    pub fn new(
        backend: impl Into<String>,
        params_backend: impl Into<String>,
        threads: u32,
    ) -> Result<Self> {
        let options = Self {
            backend: backend.into(),
            params_backend: params_backend.into(),
            threads,
            enable_mmap: true,
            flash_attention: false,
            diffusion_flash_attention: false,
        };
        options.validate()?;
        Ok(options)
    }

    /// Validates explicit placement fields.
    ///
    /// # Errors
    ///
    /// Returns the first invalid field.
    pub fn validate(&self) -> Result<()> {
        validate_backend("backend", &self.backend)?;
        validate_backend("parameter backend", &self.params_backend)?;
        if self.threads == 0 || self.threads > i32::MAX as u32 {
            return Err(Error::Invalid(
                "native thread count must be within 1..=i32::MAX".to_owned(),
            ));
        }
        Ok(())
    }
}

/// One exact image-generation request.
#[derive(Clone, Debug)]
pub struct ImageRequest {
    prompt: String,
    width: u32,
    height: u32,
    seed: u64,
    cfg_scale: f32,
    schedule: DiffusionSchedule,
}

impl ImageRequest {
    /// Constructs one bounded request with a caller-defined exact schedule.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty/oversized/NUL prompt, dimensions outside
    /// adapter bounds, seed outside the native range, invalid guidance, or an
    /// invalid Euler schedule.
    pub fn new(
        prompt: impl Into<String>,
        width: u32,
        height: u32,
        seed: u64,
        cfg_scale: f32,
        schedule: DiffusionSchedule,
    ) -> Result<Self> {
        let request = Self {
            prompt: prompt.into(),
            width,
            height,
            seed,
            cfg_scale,
            schedule,
        };
        request.validate_common()?;
        Ok(request)
    }

    /// Creates a linearly descending custom Euler sigma schedule.
    ///
    /// This is a mechanical convenience, not a quality-tuned scheduler.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid request or step count.
    pub fn linear_euler(
        prompt: impl Into<String>,
        width: u32,
        height: u32,
        seed: u64,
        cfg_scale: f32,
        steps: u32,
    ) -> Result<Self> {
        if steps == 0 || usize::try_from(steps).map_or(true, |steps| steps > MAX_DIFFUSION_STEPS) {
            return Err(Error::Invalid(format!(
                "Euler step count must be within 1..={MAX_DIFFUSION_STEPS}"
            )));
        }
        let mut sigmas = Vec::with_capacity(
            usize::try_from(steps)
                .map_err(|_| Error::Invalid("step count exceeds usize".to_owned()))?
                .saturating_add(1),
        );
        let divisor = f32::from(
            u16::try_from(steps)
                .map_err(|_| Error::Invalid("Euler step count exceeds u16".to_owned()))?,
        );
        for step in 0..=steps {
            let numerator = f32::from(
                u16::try_from(steps - step)
                    .map_err(|_| Error::Invalid("Euler step count exceeds u16".to_owned()))?,
            );
            sigmas.push(numerator / divisor);
        }
        let implementation =
            Digest::of_bytes("sdcpp-linear-euler-schedule-v1", &steps.to_le_bytes());
        let schedule = DiffusionSchedule::new(implementation, sigmas)
            .map_err(logit_loom_diffusion::Error::from)?;
        Self::new(prompt, width, height, seed, cfg_scale, schedule)
    }

    /// Returns the exact prompt.
    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    /// Returns output width.
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Returns output height.
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Returns the exact random seed.
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Returns text guidance scale.
    pub const fn cfg_scale(&self) -> f32 {
        self.cfg_scale
    }

    /// Returns the exact custom sigma schedule.
    pub const fn schedule(&self) -> &DiffusionSchedule {
        &self.schedule
    }

    /// Returns a path-free, prompt-free receipt for this exact request.
    ///
    /// # Errors
    ///
    /// Returns a request validation or schedule identity error.
    pub fn receipt(&self) -> Result<ImageRequestReceipt> {
        self.validate_common()?;
        Ok(ImageRequestReceipt {
            prompt: Digest::of_bytes("sdcpp-prompt-bytes-v1", self.prompt.as_bytes()),
            width: self.width,
            height: self.height,
            seed: self.seed,
            cfg_scale_bits: self.cfg_scale.to_bits(),
            schedule: self
                .schedule
                .digest()
                .map_err(logit_loom_diffusion::Error::from)?,
        })
    }

    pub(crate) fn validate_for(&self, profile: Profile) -> Result<()> {
        self.validate_common()?;
        profile.validate_dimensions(self.width, self.height)
    }

    pub(crate) fn validate_common(&self) -> Result<()> {
        if self.prompt.is_empty()
            || self.prompt.len() > MAX_PROMPT_BYTES
            || self.prompt.contains('\0')
        {
            return Err(Error::Invalid(format!(
                "prompt must contain 1..={MAX_PROMPT_BYTES} bytes without NUL"
            )));
        }
        if self.width == 0
            || self.height == 0
            || self.width > MAX_IMAGE_DIMENSION
            || self.height > MAX_IMAGE_DIMENSION
        {
            return Err(Error::Invalid(format!(
                "image dimensions must be within 1..={MAX_IMAGE_DIMENSION}"
            )));
        }
        if self.seed > i64::MAX as u64 {
            return Err(Error::Invalid(
                "seed exceeds the native i64 range".to_owned(),
            ));
        }
        if !self.cfg_scale.is_finite() || !(0.0..=64.0).contains(&self.cfg_scale) {
            return Err(Error::Invalid(
                "guidance scale must be finite and within 0..=64".to_owned(),
            ));
        }
        self.schedule
            .validate()
            .map_err(logit_loom_diffusion::Error::from)?;
        if self.schedule.sigmas[..self.schedule.sigmas.len() - 1].contains(&0.0) {
            return Err(Error::Invalid(
                "Euler sigma may reach zero only at the final boundary".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Path-free, prompt-free identity of one exact image request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageRequestReceipt {
    /// Content identity of the exact UTF-8 prompt bytes.
    pub prompt: Digest,
    /// Requested output width.
    pub width: u32,
    /// Requested output height.
    pub height: u32,
    /// Exact seed interpreted by the identified native RNG.
    pub seed: u64,
    /// Exact IEEE 754 bit pattern of the guidance scale.
    pub cfg_scale_bits: u32,
    /// Identity of the exact custom sigma schedule.
    pub schedule: Digest,
}

/// Synchronous behavior at exact diffusion boundaries.
///
/// `begin` runs once immediately before the first state callback.
/// `intervene` receives a private transactional copy. `observe` receives the
/// finite post-intervention copy. The native state is changed only after both
/// methods succeed.
pub trait StepProgram {
    /// Returns the implementation/configuration identity.
    fn implementation(&self) -> &Digest;

    /// Initializes program state against the exact runtime plan.
    ///
    /// # Errors
    ///
    /// Returns an error when the program rejects the plan or cannot initialize
    /// its per-run state.
    fn begin(&mut self, _plan: &DiffusionPlan) -> std::result::Result<(), String> {
        Ok(())
    }

    /// Mutates one private post-Euler state copy.
    ///
    /// # Errors
    ///
    /// Returns an error when the intervention rejects the boundary or cannot
    /// produce a complete finite state.
    fn intervene(
        &mut self,
        _context: &StepContext,
        _state: &mut [f32],
    ) -> std::result::Result<(), String> {
        Ok(())
    }

    /// Observes the complete finite state and may request cooperative stop.
    ///
    /// # Errors
    ///
    /// Returns an error when the observer cannot account for the complete
    /// post-intervention state.
    fn observe(
        &mut self,
        _context: &StepContext,
        _state: &[f32],
    ) -> std::result::Result<ControlFlow, String> {
        Ok(ControlFlow::Continue)
    }
}

/// Lightweight synchronous control at exact post-Euler boundaries.
///
/// Unlike [`StepProgram`], this contract never exposes, copies, hashes, or
/// scans the scheduler state. It is intended for worker-local cancellation and
/// boundary accounting when no latent intervention or observation is needed.
pub trait BoundaryControl {
    /// Returns the implementation/configuration identity.
    fn implementation(&self) -> &Digest;

    /// Initializes control state against the exact runtime plan.
    ///
    /// # Errors
    ///
    /// Returns an error when the controller rejects the plan.
    fn begin(&mut self, _plan: &DiffusionPlan) -> std::result::Result<(), String> {
        Ok(())
    }

    /// Observes one completed transition and may request cooperative stop.
    ///
    /// # Errors
    ///
    /// Returns an error when boundary accounting fails.
    fn boundary(&mut self, _context: &StepContext) -> std::result::Result<ControlFlow, String> {
        Ok(ControlFlow::Continue)
    }
}

/// Caller-owned destination for one complete native image.
///
/// The adapter calls [`ImageOutputSink::write_image`] exactly once after
/// validating native geometry. Implementations may write directly to mapped
/// storage, a descriptor, or an in-memory slice. Panics are contained.
pub trait ImageOutputSink {
    /// Returns the exact byte length accepted by the destination.
    fn expected_len(&self) -> usize;

    /// Retains one complete image.
    ///
    /// # Errors
    ///
    /// Returns a bounded caller-defined failure.
    fn write_image(&mut self, bytes: &[u8]) -> std::result::Result<(), String>;
}

/// Boundary controller that always continues.
#[derive(Clone, Debug)]
pub struct ContinueControl {
    implementation: Digest,
}

impl Default for ContinueControl {
    fn default() -> Self {
        Self {
            implementation: Digest::of_bytes("sdcpp-continue-boundary-control-v1", b""),
        }
    }
}

impl BoundaryControl for ContinueControl {
    fn implementation(&self) -> &Digest {
        &self.implementation
    }
}

/// A program that leaves state unchanged and continues every step.
#[derive(Clone, Debug)]
pub struct NoopProgram {
    implementation: Digest,
}

impl Default for NoopProgram {
    fn default() -> Self {
        Self {
            implementation: Digest::of_bytes("sdcpp-noop-step-program-v1", b""),
        }
    }
}

impl StepProgram for NoopProgram {
    fn implementation(&self) -> &Digest {
        &self.implementation
    }
}

impl<T> StepProgram for Box<T>
where
    T: StepProgram + ?Sized,
{
    fn implementation(&self) -> &Digest {
        self.as_ref().implementation()
    }

    fn begin(&mut self, plan: &DiffusionPlan) -> std::result::Result<(), String> {
        self.as_mut().begin(plan)
    }

    fn intervene(
        &mut self,
        context: &StepContext,
        state: &mut [f32],
    ) -> std::result::Result<(), String> {
        self.as_mut().intervene(context, state)
    }

    fn observe(
        &mut self,
        context: &StepContext,
        state: &[f32],
    ) -> std::result::Result<ControlFlow, String> {
        self.as_mut().observe(context, state)
    }
}

/// Path-free receipt for the exact verified component set.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileReceipt {
    /// Catalog profile ID.
    pub profile_id: String,
    /// Exact catalog JSON digest.
    pub catalog_sha256: String,
    /// Exact verified files in stable component order.
    pub artifacts: Vec<ArtifactReceipt>,
}

/// Model-free evidence for one exact dynamically loaded companion library.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompanionReceipt {
    /// SHA-256 of the exact shared-library bytes.
    pub library_sha256: String,
    /// Companion ABI version.
    pub companion_abi: u32,
    /// Exact upstream source revision.
    pub upstream_commit: String,
    /// Native device report lines, which may include CPU-only devices.
    pub devices: Vec<String>,
}

/// Exact dynamically loaded runtime and placement evidence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeRuntimeReceipt {
    /// SHA-256 of the exact shared-library bytes.
    pub library_sha256: String,
    /// Companion ABI version.
    pub companion_abi: u32,
    /// Exact upstream source revision.
    pub upstream_commit: String,
    /// Requested evaluation backend.
    pub backend: String,
    /// Requested parameter backend.
    pub params_backend: String,
    /// Native host-orchestration thread count.
    pub threads: u32,
    /// Whether supported artifact mapping was enabled.
    pub enable_mmap: bool,
    /// Whether general native flash attention was enabled.
    pub flash_attention: bool,
    /// Whether diffusion-model flash attention was enabled.
    pub diffusion_flash_attention: bool,
    /// Native device report lines.
    pub devices: Vec<String>,
    /// Conservative adapter build identity.
    pub identity: Digest,
}

/// Accounting for one exact post-Euler callback.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StepReceipt {
    /// Zero-based completed transition.
    pub step_index: u32,
    /// Identity before the step program.
    pub native_state: Digest,
    /// Identity committed after the program.
    pub committed_state: Digest,
    /// Number of changed `f32` bit patterns.
    pub elements_changed: u64,
    /// Whether an observer requested cooperative stop.
    pub stop_requested: bool,
}

/// Accounting for one control-only post-Euler boundary.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundaryReceipt {
    /// Zero-based completed transition.
    pub step_index: u32,
    /// Whether the controller requested cooperative stop.
    pub stop_requested: bool,
}

/// Path-free mechanical record for one native generation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationReceipt {
    /// Exact profile artifacts.
    pub profile: ProfileReceipt,
    /// Native runtime and placement identity.
    pub native: NativeRuntimeReceipt,
    /// Session epoch used by this operation.
    pub session_epoch: u64,
    /// Exact request mechanics without raw prompt bytes.
    pub request: ImageRequestReceipt,
    /// Exact diffusion plan after native conditioning.
    pub plan: DiffusionPlan,
    /// Exact step-program identity.
    pub program: Digest,
    /// Number of condition tensors hashed.
    pub condition_tensors: u32,
    /// Total condition-tensor bytes hashed.
    pub condition_bytes: u64,
    /// Post-step accounting in order.
    pub steps: Vec<StepReceipt>,
    /// Whether native sampling stopped cooperatively.
    pub stopped: bool,
    /// Exact output pixel-byte identity.
    pub image: Digest,
    /// Output width.
    pub width: u32,
    /// Output height.
    pub height: u32,
    /// Output channels.
    pub channels: u32,
}

/// One copied native image and its path-free receipt.
#[derive(Clone, Debug)]
pub struct GenerationOutput {
    /// Exact interleaved image bytes.
    pub bytes: Vec<u8>,
    /// Mechanical execution receipt.
    pub receipt: GenerationReceipt,
    /// Non-deterministic deployment measurements excluded from identities.
    pub measurements: GenerationMeasurements,
}

/// Path-free mechanical record for control-only native generation.
///
/// This receipt deliberately contains no scheduler-state identity because the
/// control-only path never reads, copies, hashes, or mutates that state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlledGenerationReceipt {
    /// Exact profile artifacts.
    pub profile: ProfileReceipt,
    /// Native runtime and placement identity.
    pub native: NativeRuntimeReceipt,
    /// Session epoch used by this operation.
    pub session_epoch: u64,
    /// Exact request mechanics without raw prompt bytes.
    pub request: ImageRequestReceipt,
    /// Exact diffusion plan after native conditioning.
    pub plan: DiffusionPlan,
    /// Exact boundary-controller identity.
    pub control: Digest,
    /// Number of condition tensors hashed.
    pub condition_tensors: u32,
    /// Total condition-tensor bytes hashed.
    pub condition_bytes: u64,
    /// Completed control boundaries in order.
    pub boundaries: Vec<BoundaryReceipt>,
    /// Whether native sampling stopped cooperatively.
    pub stopped: bool,
    /// Exact output pixel-byte identity.
    pub image: Digest,
    /// Output width.
    pub width: u32,
    /// Output height.
    pub height: u32,
    /// Output channels.
    pub channels: u32,
}

/// Control-only generation written into caller-owned storage.
#[derive(Clone, Debug)]
pub struct ControlledGenerationOutput {
    /// Number of initialized destination bytes.
    pub bytes_written: usize,
    /// Mechanical control-only execution receipt.
    pub receipt: ControlledGenerationReceipt,
    /// Non-deterministic deployment measurements excluded from identities.
    pub measurements: GenerationMeasurements,
}

/// Non-deterministic deployment facts for one native generation.
///
/// These values are deliberately separate from [`GenerationReceipt`] and its
/// content identities. They describe one execution environment and are not
/// replay invariants.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GenerationMeasurements {
    /// Native denoiser-plus-Euler-update latency for each completed step.
    pub step_latency_milliseconds: Vec<f64>,
}

/// Exact restorable post-step state plus conservative lineage.
#[derive(Clone, Debug)]
pub struct DiffusionCheckpoint {
    receipt: DiffusionCheckpointReceipt,
    state_le_bytes: Vec<u8>,
}

impl DiffusionCheckpoint {
    /// Returns the serializable checkpoint lineage.
    pub const fn receipt(&self) -> &DiffusionCheckpointReceipt {
        &self.receipt
    }

    /// Returns exact little-endian finite `f32` state bytes.
    pub fn state_bytes(&self) -> &[u8] {
        &self.state_le_bytes
    }

    /// Reconstructs authenticated checkpoint parts.
    ///
    /// # Errors
    ///
    /// Returns an error when the byte length or state identity differs.
    pub fn from_parts(
        receipt: DiffusionCheckpointReceipt,
        state_le_bytes: Vec<u8>,
    ) -> Result<Self> {
        if state_le_bytes.is_empty() || !state_le_bytes.len().is_multiple_of(4) {
            return Err(Error::Invalid(
                "checkpoint state must contain complete f32 bytes".to_owned(),
            ));
        }
        let state = Digest::of_bytes("sdcpp-checkpoint-f32-le-v1", &state_le_bytes);
        if state != receipt.state {
            return Err(Error::Incompatible(
                "checkpoint state bytes do not match their receipt".to_owned(),
            ));
        }
        Ok(Self {
            receipt,
            state_le_bytes,
        })
    }

    pub(crate) fn capture(
        plan: &DiffusionPlan,
        backend: &Digest,
        context: &StepContext,
        state: &[f32],
    ) -> Result<Self> {
        let mut state_le_bytes = Vec::with_capacity(state.len().saturating_mul(4));
        for value in state {
            state_le_bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        let state_digest = Digest::of_bytes("sdcpp-checkpoint-f32-le-v1", &state_le_bytes);
        let next_step = context
            .step_index
            .checked_add(1)
            .ok_or_else(|| Error::Invalid("checkpoint step overflowed".to_owned()))?;
        let continuation = Digest::of_serializable(
            "sdcpp-deterministic-prefix-replay-v1",
            &(
                plan.schedule
                    .digest()
                    .map_err(logit_loom_diffusion::Error::from)?,
                &plan.rng,
                plan.seed,
                next_step,
            ),
        )
        .map_err(logit_loom_diffusion::Error::from)?;
        Ok(Self {
            receipt: DiffusionCheckpointReceipt {
                plan: plan.digest().map_err(logit_loom_diffusion::Error::from)?,
                backend: backend.clone(),
                next_step,
                state: state_digest,
                continuation,
            },
            state_le_bytes,
        })
    }

    pub(crate) fn restore(
        &self,
        plan: &DiffusionPlan,
        backend: &Digest,
        context: &StepContext,
        state: &mut [f32],
    ) -> Result<()> {
        self.receipt
            .validate_for(plan)
            .map_err(logit_loom_diffusion::Error::from)?;
        if &self.receipt.backend != backend
            || self.receipt.next_step != context.step_index.saturating_add(1)
            || self.state_le_bytes.len() != state.len().saturating_mul(4)
        {
            return Err(Error::Incompatible(
                "checkpoint backend, step, or tensor length differs".to_owned(),
            ));
        }
        for (destination, bytes) in state.iter_mut().zip(self.state_le_bytes.chunks_exact(4)) {
            let bits = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            let value = f32::from_bits(bits);
            if !value.is_finite() {
                return Err(Error::Incompatible(
                    "checkpoint contains a non-finite state value".to_owned(),
                ));
            }
            *destination = value;
        }
        Ok(())
    }
}

pub(crate) fn component_map(
    profile: &ProfileReceipt,
    native: &NativeRuntimeReceipt,
) -> Result<BTreeMap<String, Digest>> {
    let mut components = BTreeMap::new();
    for receipt in &profile.artifacts {
        let identity = Digest::of_serializable("sdcpp-profile-artifact-v1", receipt)
            .map_err(logit_loom_diffusion::Error::from)?;
        if components
            .insert(receipt.source_id.clone(), identity)
            .is_some()
        {
            return Err(Error::Incompatible(
                "profile repeats a component source ID".to_owned(),
            ));
        }
    }
    components.insert("native-runtime".to_owned(), native.identity.clone());
    Ok(components)
}

fn validate_backend(label: &str, value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > MAX_BACKEND_LABEL_BYTES
        || value.contains('\0')
        || value.to_ascii_lowercase().contains("cpu")
    {
        return Err(Error::Invalid(format!(
            "{label} must be a bounded non-CPU label without NUL"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_rejects_internal_zero_sigma_and_oversized_seed() {
        let schedule = DiffusionSchedule::new(
            Digest::of_bytes("test-schedule", b"one"),
            vec![1.0, 0.0, 0.0],
        )
        .expect("backend-neutral schedule permits a plateau");
        assert!(ImageRequest::new("test", 512, 512, 1, 6.0, schedule).is_err());

        let request = ImageRequest::linear_euler("test", 512, 512, i64::MAX as u64 + 1, 6.0, 4);
        assert!(request.is_err());
    }

    #[test]
    fn profile_dimensions_are_explicit() {
        assert!(Profile::MiniT2iB16.validate_dimensions(512, 512).is_ok());
        assert!(Profile::MiniT2iB16.validate_dimensions(510, 512).is_err());
        assert!(
            Profile::MiniT2iB16
                .validate_dimensions(4_096, 4_096)
                .is_err()
        );
        assert!(Profile::Krea2Turbo.validate_dimensions(1024, 1024).is_ok());
        assert!(Profile::Krea2Turbo.validate_dimensions(1000, 1024).is_err());
    }

    #[test]
    fn cpu_named_backends_are_rejected() {
        assert!(SdcppOptions::new("cpu", "vulkan", 8).is_err());
        assert!(SdcppOptions::new("vulkan", "cpu", 8).is_err());
    }

    #[test]
    fn linear_schedule_rejects_extreme_steps_before_allocation() {
        assert!(ImageRequest::linear_euler("test", 512, 512, 7, 6.0, u32::MAX).is_err());
    }

    #[test]
    fn request_receipt_binds_but_does_not_expose_prompt_bytes() {
        let request = ImageRequest::linear_euler("private prompt", 512, 512, 7, 1.0, 4)
            .expect("valid request");
        let receipt = request.receipt().expect("request receipt");
        let json = serde_json::to_string(&receipt).expect("serialize receipt");
        assert!(!json.contains("private prompt"));
        assert_eq!(receipt.cfg_scale_bits, 1.0_f32.to_bits());
        assert_eq!(receipt.seed, 7);
    }
}

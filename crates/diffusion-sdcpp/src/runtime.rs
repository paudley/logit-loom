// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    any::Any,
    collections::BTreeMap,
    ffi::{CStr, CString, c_void},
    fs::File,
    io::Read as _,
    marker::PhantomData,
    mem::align_of,
    path::Path,
    ptr::NonNull,
    rc::Rc,
    slice,
    time::Duration,
};

use logit_loom_diffusion::{
    ControlFlow, DiffusionPlan, Digest, StepContext, TensorDType, TensorLayout, TensorSpec,
};
use logit_loom_executor::{
    CancellationProbe, ClassifiedExecutionError, CleanupReceipt, ExecutorState, FailureDisposition,
    InputBuffer, LocalExecutor, OutputBuffer,
};
use logit_loom_models::Catalog;
use serde::Serialize;
use sha2::{Digest as _, Sha256};

use crate::{
    ADAPTER_CONTRACT_VERSION, AdvancedGenerationOutput, AdvancedGenerationReceipt,
    AdvancedImageRequest, AdvancedProgramGenerationOutput, AdvancedProgramGenerationReceipt,
    BoundaryControl, BoundaryReceipt, COMPANION_ABI_VERSION, CompanionReceipt,
    ControlledGenerationOutput, ControlledGenerationReceipt, Error, GenerationMeasurements,
    GenerationOutput, GenerationReceipt, IMAGE_ABI_VERSION, ImageExecutionBindings,
    ImageOutputSink, ImagePixels, ImageRequest, NativeRuntimeReceipt, Profile, ProfileArtifacts,
    ProfileReceipt, Result, SdcppOptions, StepProgram, StepReceipt, UPSTREAM_COMMIT,
    VaeImageOutput, VaeOperationReceipt, VaeTensor, VaeTensorOutput,
    contract::component_map,
    ffi::{
        self, ConditionTensor, ContextParams, ImageParams, ImageParamsV2, ImageViewV2, LoraV2,
        NativeApi, OwnedTensorV2, Step, TensorViewV2,
    },
};

const HASH_BUFFER_BYTES: usize = 1024 * 1024;
const MAX_PATH_BYTES: usize = 4_096;
const MAX_CONDITION_TENSORS: u32 = 256;
const MAX_CONDITION_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_CONDITION_LABEL_BYTES: usize = 256;

/// Verifies and inspects one exact companion shared library without loading a
/// model or requiring an accelerator.
///
/// The device report may contain CPU-only devices. Model execution through
/// [`Sdcpp::load`] separately requires caller-selected non-CPU backends.
///
/// # Errors
///
/// Returns a file, digest, dynamic-loader, symbol, ABI, revision, or device
/// report error.
pub fn probe_companion(path: impl AsRef<Path>) -> Result<CompanionReceipt> {
    let (_, library_sha256, devices) = open_companion(path.as_ref())?;
    Ok(CompanionReceipt {
        library_sha256,
        companion_abi: COMPANION_ABI_VERSION,
        upstream_commit: UPSTREAM_COMMIT.to_owned(),
        devices,
    })
}

/// One loaded exact diffusion profile and companion runtime.
///
/// This owner is deliberately neither `Send` nor `Sync`. Model context
/// destruction occurs before the dynamic library is unloaded.
///
/// ```compile_fail
/// use logit_loom_diffusion_sdcpp::Sdcpp;
///
/// fn require_send<T: Send>() {}
/// require_send::<Sdcpp>();
/// ```
///
/// ```compile_fail
/// use logit_loom_diffusion_sdcpp::Sdcpp;
///
/// fn require_sync<T: Sync>() {}
/// require_sync::<Sdcpp>();
/// ```
pub struct Sdcpp {
    pub(crate) context: NonNull<c_void>,
    pub(crate) api: NativeApi,
    pub(crate) profile: Profile,
    pub(crate) profile_receipt: ProfileReceipt,
    pub(crate) native_receipt: NativeRuntimeReceipt,
    options: SdcppOptions,
    pub(crate) state: ExecutorState,
    pub(crate) session_epoch: u64,
    pub(crate) krea_activation: Option<crate::krea_activation::InstalledKreaActivation>,
    _single_owner: PhantomData<Rc<()>>,
}

impl std::fmt::Debug for Sdcpp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Sdcpp")
            .field("profile", &self.profile)
            .field("profile_receipt", &self.profile_receipt)
            .field("native_receipt", &self.native_receipt)
            .field("options", &self.options)
            .field("state", &self.state)
            .field("session_epoch", &self.session_epoch)
            .field("krea_activation", &self.krea_activation.is_some())
            .finish_non_exhaustive()
    }
}

struct AdvancedNativeInputs {
    prompt: CString,
    negative_prompt: CString,
    width: i32,
    height: i32,
    seed: i64,
    operation: i32,
    init_image: ImageViewV2,
    mask_image: ImageViewV2,
    reference_images: Vec<ImageViewV2>,
    _lora_paths: Vec<CString>,
    loras: Vec<LoraV2>,
}

impl AdvancedNativeInputs {
    fn new(request: &AdvancedImageRequest<'_>) -> Result<Self> {
        let base = request.base();
        let lora_paths = request
            .loras()
            .iter()
            .map(|lora| path_c_string(lora.path()))
            .collect::<Result<Vec<_>>>()?;
        let loras = request
            .loras()
            .iter()
            .zip(&lora_paths)
            .map(|(lora, path)| LoraV2 {
                path: path.as_ptr(),
                multiplier: lora.scale(),
                is_high_noise: lora.is_high_noise(),
            })
            .collect();
        let operation = match request.operation() {
            logit_loom_diffusion::ImageOperation::TextToImage => ffi::OPERATION_TEXT_TO_IMAGE,
            logit_loom_diffusion::ImageOperation::ImageToImage => ffi::OPERATION_IMAGE_TO_IMAGE,
            logit_loom_diffusion::ImageOperation::Inpaint => ffi::OPERATION_INPAINT,
            logit_loom_diffusion::ImageOperation::Outpaint => ffi::OPERATION_OUTPAINT,
            _ => {
                return Err(Error::Invalid(
                    "image ABI v2 generation requires a diffusion operation".to_owned(),
                ));
            }
        };
        Ok(Self {
            prompt: bounded_c_string("prompt", base.prompt())?,
            negative_prompt: bounded_c_string("negative prompt", request.negative_prompt())?,
            width: i32::try_from(base.width())
                .map_err(|_| Error::Invalid("image width exceeds i32".to_owned()))?,
            height: i32::try_from(base.height())
                .map_err(|_| Error::Invalid("image height exceeds i32".to_owned()))?,
            seed: i64::try_from(base.seed())
                .map_err(|_| Error::Invalid("seed exceeds i64".to_owned()))?,
            operation,
            init_image: request
                .source()
                .map_or(ImageViewV2::EMPTY, native_image_view),
            mask_image: request.mask().map_or(ImageViewV2::EMPTY, native_image_view),
            reference_images: request
                .references()
                .iter()
                .copied()
                .map(native_image_view)
                .collect(),
            _lora_paths: lora_paths,
            loras,
        })
    }

    fn params(&self, request: &AdvancedImageRequest<'_>) -> ImageParamsV2 {
        let base = request.base();
        ImageParamsV2 {
            abi_version: IMAGE_ABI_VERSION,
            operation: self.operation,
            prompt: self.prompt.as_ptr(),
            negative_prompt: self.negative_prompt.as_ptr(),
            width: self.width,
            height: self.height,
            seed: self.seed,
            cfg_scale: base.cfg_scale(),
            strength: request.strength(),
            sigmas: base.schedule().sigmas.as_ptr(),
            sigma_count: base.schedule().sigmas.len(),
            init_image: self.init_image,
            mask_image: self.mask_image,
            reference_images: pointer_or_null(&self.reference_images),
            reference_image_count: self.reference_images.len(),
            loras: pointer_or_null(&self.loras),
            lora_count: self.loras.len(),
        }
    }
}

impl Sdcpp {
    /// Verifies exact artifacts, loads the companion library, requires the
    /// requested accelerator devices, and creates one native context.
    ///
    /// The method never downloads an artifact or retries on CPU.
    ///
    /// # Errors
    ///
    /// Returns a catalog, artifact, library, ABI, placement, path, or native
    /// context error.
    pub fn load(
        library_path: impl AsRef<Path>,
        artifacts: &ProfileArtifacts,
        options: SdcppOptions,
    ) -> Result<Self> {
        options.validate()?;
        let profile = artifacts.profile();
        let catalog = Catalog::embedded()?;
        let profile_receipt = verify_profile_artifacts(&catalog, artifacts)?;
        let library_path = library_path.as_ref();
        let (api, library_sha256, devices) = open_companion(library_path)?;
        require_backend_device(&options.backend, &devices)?;
        require_backend_device(&options.params_backend, &devices)?;

        let native_identity = native_identity(&library_sha256, &devices, &options)?;
        let native_receipt = NativeRuntimeReceipt {
            library_sha256,
            companion_abi: COMPANION_ABI_VERSION,
            upstream_commit: UPSTREAM_COMMIT.to_owned(),
            backend: options.backend.clone(),
            params_backend: options.params_backend.clone(),
            threads: options.threads,
            enable_mmap: options.enable_mmap,
            flash_attention: options.flash_attention,
            diffusion_flash_attention: options.diffusion_flash_attention,
            devices,
            identity: native_identity,
        };

        let diffusion_path = path_c_string(artifacts.diffusion_model())?;
        let text_encoder_path = path_c_string(artifacts.text_encoder())?;
        let vae_path = artifacts.vae().map(path_c_string).transpose()?;
        let backend = bounded_c_string("backend", &options.backend)?;
        let params_backend = bounded_c_string("parameter backend", &options.params_backend)?;
        let native_params = ContextParams {
            abi_version: COMPANION_ABI_VERSION,
            profile: profile.native_id(),
            diffusion_model_path: diffusion_path.as_ptr(),
            text_encoder_path: text_encoder_path.as_ptr(),
            vae_path: vae_path
                .as_ref()
                .map_or(std::ptr::null(), |path| path.as_ptr()),
            backend: backend.as_ptr(),
            params_backend: params_backend.as_ptr(),
            n_threads: i32::try_from(options.threads)
                .map_err(|_| Error::Invalid("thread count exceeds i32".to_owned()))?,
            enable_mmap: options.enable_mmap,
            flash_attn: options.flash_attention,
            diffusion_flash_attn: options.diffusion_flash_attention,
        };
        // SAFETY: Every C string above remains live for this synchronous call,
        // and the exact ABI/commit has already been checked.
        let context = unsafe { api.new_context(&native_params) };
        let context = NonNull::new(context).ok_or_else(|| {
            Error::Native(format!(
                "{} context construction rejected the requested artifacts or placement",
                profile.id()
            ))
        })?;

        let loaded = Self {
            context,
            api,
            profile,
            profile_receipt,
            native_receipt,
            options,
            state: ExecutorState::Resident,
            session_epoch: 0,
            krea_activation: None,
            _single_owner: PhantomData,
        };
        let after_load = verify_profile_artifacts(&catalog, artifacts)?;
        if after_load != loaded.profile_receipt {
            return Err(Error::Incompatible(
                "profile artifacts changed while the native context was loading".to_owned(),
            ));
        }
        Ok(loaded)
    }

    /// Returns the exact selected profile.
    pub const fn profile(&self) -> Profile {
        self.profile
    }

    /// Returns path-free verified model-component evidence.
    pub const fn profile_receipt(&self) -> &ProfileReceipt {
        &self.profile_receipt
    }

    /// Returns exact native library and placement evidence.
    pub const fn native_receipt(&self) -> &NativeRuntimeReceipt {
        &self.native_receipt
    }

    /// Returns the current worker-local lifecycle state.
    pub const fn state(&self) -> ExecutorState {
        self.state
    }

    /// Returns the epoch to which any backend-local handle would be bound.
    pub const fn session_epoch(&self) -> u64 {
        self.session_epoch
    }

    /// Returns the exact safe-backend, profile, load, RNG, and placement
    /// identities required by a backend-neutral whole-image plan for this
    /// resident owner.
    ///
    /// # Errors
    ///
    /// Returns an error only when deterministic identity encoding fails.
    pub fn execution_bindings(&self) -> Result<ImageExecutionBindings> {
        let backend = image_execution_backend_identity(&self.native_receipt)?;
        let profile =
            Digest::of_serializable("sdcpp-image-execution-profile-v1", &self.profile_receipt)
                .map_err(logit_loom_diffusion::Error::from)?;
        let load = Digest::of_serializable("sdcpp-image-execution-load-v1", &(&profile, &backend))
            .map_err(logit_loom_diffusion::Error::from)?;
        let placement = Digest::of_serializable(
            "sdcpp-image-execution-placement-v1",
            &(
                &self.native_receipt.backend,
                &self.native_receipt.params_backend,
                &self.native_receipt.devices,
            ),
        )
        .map_err(logit_loom_diffusion::Error::from)?;
        Ok(ImageExecutionBindings {
            backend,
            profile,
            load,
            rng: rng_identity(&self.native_receipt, &self.profile_receipt)?,
            placement,
        })
    }

    /// Runs one exact custom-sigma Euler generation.
    ///
    /// Native conditioning tensors are hashed before `StepProgram::begin`.
    /// Each subsequent post-Euler state is copied transactionally, passed to
    /// `intervene`, validated, passed immutably to `observe`, and only then
    /// committed to native state. Callback errors and panics are contained.
    ///
    /// # Errors
    ///
    /// Returns a request, conditioning, step-boundary, callback, native-output,
    /// or accounting error.
    pub fn generate(
        &mut self,
        request: &ImageRequest,
        program: &mut dyn StepProgram,
    ) -> Result<GenerationOutput> {
        self.run_operation(|runtime| runtime.generate_inner(request, program))
    }

    fn generate_inner(
        &mut self,
        request: &ImageRequest,
        program: &mut dyn StepProgram,
    ) -> Result<GenerationOutput> {
        request.validate_for(self.profile)?;
        let prompt = bounded_c_string("prompt", request.prompt())?;
        let width = i32::try_from(request.width())
            .map_err(|_| Error::Invalid("image width exceeds i32".to_owned()))?;
        let height = i32::try_from(request.height())
            .map_err(|_| Error::Invalid("image height exceeds i32".to_owned()))?;
        let seed = i64::try_from(request.seed())
            .map_err(|_| Error::Invalid("seed exceeds i64".to_owned()))?;
        let native_params = ImageParams {
            abi_version: COMPANION_ABI_VERSION,
            prompt: prompt.as_ptr(),
            width,
            height,
            seed,
            cfg_scale: request.cfg_scale(),
            sigmas: request.schedule().sigmas.as_ptr(),
            sigma_count: request.schedule().sigmas.len(),
        };
        let components = component_map(&self.profile_receipt, &self.native_receipt)?;
        let request_receipt = request.receipt()?;
        let program_identity = program.implementation().clone();
        let mut callbacks = CallbackState::new_full(
            self.profile,
            &self.profile_receipt,
            &self.native_receipt,
            request,
            components,
            program,
        )?;
        let callback_pointer = (&raw mut callbacks).cast::<c_void>();
        let mut image = std::ptr::null_mut();
        // SAFETY: Context ownership is exclusive, parameters and callback
        // state remain live for this synchronous call, and callbacks validate
        // native descriptors before forming slices.
        let status = unsafe {
            self.api.generate_image(
                self.context.as_ptr(),
                &native_params,
                condition_callback,
                callback_pointer,
                step_callback,
                &mut image,
            )
        };
        if let Some(error) = callbacks.error.take() {
            if !image.is_null() {
                // SAFETY: A non-null result is one native image allocation
                // transferred by this exact API.
                unsafe { self.api.free_images(image, 1) };
            }
            return Err(Error::Callback(error));
        }
        if !matches!(status, ffi::STATUS_OK | ffi::STATUS_STOPPED) {
            if !image.is_null() {
                // SAFETY: Same ownership rule as above.
                unsafe { self.api.free_images(image, 1) };
            }
            return Err(native_status_error(status));
        }
        let image = NonNull::new(image)
            .ok_or_else(|| Error::Native("native success returned a null image".to_owned()))?;
        let image_guard = NativeImageGuard {
            api: &self.api,
            image,
        };
        let (bytes, width, height, channels) = copy_image(image_guard.image.as_ptr(), request)?;
        let plan = callbacks
            .plan
            .take()
            .ok_or_else(|| Error::Native("native generation produced no step plan".to_owned()))?;
        if callbacks.steps.is_empty() {
            return Err(Error::Native(
                "native generation produced no post-Euler steps".to_owned(),
            ));
        }
        let image_digest = Digest::of_bytes("sdcpp-image-u8-v1", &bytes);
        Ok(GenerationOutput {
            bytes,
            receipt: GenerationReceipt {
                profile: self.profile_receipt.clone(),
                native: self.native_receipt.clone(),
                session_epoch: self.session_epoch,
                request: request_receipt,
                plan,
                program: program_identity,
                condition_tensors: callbacks.condition_tensors,
                condition_bytes: callbacks.condition_bytes,
                steps: callbacks.steps,
                stopped: status == ffi::STATUS_STOPPED,
                image: image_digest,
                width,
                height,
                channels,
            },
            measurements: GenerationMeasurements {
                step_latency_milliseconds: callbacks.step_latency_milliseconds,
            },
        })
    }

    /// Runs one exact generation with control-only post-Euler callbacks and
    /// writes pixels directly into caller-owned storage.
    ///
    /// The destination must have the exact native image size. This path
    /// validates boundary metadata but never reads, copies, hashes, or mutates
    /// scheduler-state elements.
    ///
    /// # Errors
    ///
    /// Returns a request, boundary, callback, native-output, destination, or
    /// accounting error.
    pub fn generate_controlled_into(
        &mut self,
        request: &ImageRequest,
        control: &mut dyn BoundaryControl,
        destination: &mut [u8],
    ) -> Result<ControlledGenerationOutput> {
        let mut sink = SliceImageSink { destination };
        self.generate_controlled_to(request, control, &mut sink)
    }

    /// Runs one exact control-only generation into a caller-owned sink.
    ///
    /// The sink is invoked exactly once with the complete validated RGB image.
    /// A descriptor-backed sink can therefore avoid an intermediate Rust
    /// image allocation.
    ///
    /// # Errors
    ///
    /// Returns a request, boundary, callback, native-output, sink, or
    /// accounting error.
    pub fn generate_controlled_to(
        &mut self,
        request: &ImageRequest,
        control: &mut dyn BoundaryControl,
        sink: &mut dyn ImageOutputSink,
    ) -> Result<ControlledGenerationOutput> {
        self.run_operation(|runtime| runtime.generate_controlled_to_inner(request, control, sink))
    }

    fn generate_controlled_to_inner(
        &mut self,
        request: &ImageRequest,
        control: &mut dyn BoundaryControl,
        sink: &mut dyn ImageOutputSink,
    ) -> Result<ControlledGenerationOutput> {
        request.validate_for(self.profile)?;
        let expected = expected_rgb_bytes(request)?;
        if sink.expected_len() != expected {
            return Err(Error::Invalid(format!(
                "control-only output has {} bytes; expected {expected}",
                sink.expected_len()
            )));
        }
        let prompt = bounded_c_string("prompt", request.prompt())?;
        let width = i32::try_from(request.width())
            .map_err(|_| Error::Invalid("image width exceeds i32".to_owned()))?;
        let height = i32::try_from(request.height())
            .map_err(|_| Error::Invalid("image height exceeds i32".to_owned()))?;
        let seed = i64::try_from(request.seed())
            .map_err(|_| Error::Invalid("seed exceeds i64".to_owned()))?;
        let native_params = ImageParams {
            abi_version: COMPANION_ABI_VERSION,
            prompt: prompt.as_ptr(),
            width,
            height,
            seed,
            cfg_scale: request.cfg_scale(),
            sigmas: request.schedule().sigmas.as_ptr(),
            sigma_count: request.schedule().sigmas.len(),
        };
        let components = component_map(&self.profile_receipt, &self.native_receipt)?;
        let request_receipt = request.receipt()?;
        let control_identity = control.implementation().clone();
        let mut callbacks = CallbackState::new_control(
            self.profile,
            &self.profile_receipt,
            &self.native_receipt,
            request,
            components,
            control,
        )?;
        let callback_pointer = (&raw mut callbacks).cast::<c_void>();
        let mut image = std::ptr::null_mut();
        // SAFETY: Context ownership is exclusive, parameters and callback
        // state remain live for this synchronous call, and callbacks validate
        // descriptors without forming a scheduler-state slice in this mode.
        let status = unsafe {
            self.api.generate_image(
                self.context.as_ptr(),
                &native_params,
                condition_callback,
                callback_pointer,
                step_callback,
                &mut image,
            )
        };
        if let Some(error) = callbacks.error.take() {
            if !image.is_null() {
                // SAFETY: A non-null result is one native image allocation
                // transferred by this exact API.
                unsafe { self.api.free_images(image, 1) };
            }
            return Err(Error::Callback(error));
        }
        if !matches!(status, ffi::STATUS_OK | ffi::STATUS_STOPPED) {
            if !image.is_null() {
                // SAFETY: Same ownership rule as above.
                unsafe { self.api.free_images(image, 1) };
            }
            return Err(native_status_error(status));
        }
        let image = NonNull::new(image)
            .ok_or_else(|| Error::Native("native success returned a null image".to_owned()))?;
        let image_guard = NativeImageGuard {
            api: &self.api,
            image,
        };
        let (bytes_written, width, height, channels, image_digest) =
            write_image_to(image_guard.image.as_ptr(), request, sink)?;
        let plan = callbacks
            .plan
            .take()
            .ok_or_else(|| Error::Native("native generation produced no step plan".to_owned()))?;
        if callbacks.boundaries.is_empty() {
            return Err(Error::Native(
                "native generation produced no post-Euler boundaries".to_owned(),
            ));
        }
        Ok(ControlledGenerationOutput {
            bytes_written,
            receipt: ControlledGenerationReceipt {
                profile: self.profile_receipt.clone(),
                native: self.native_receipt.clone(),
                session_epoch: self.session_epoch,
                request: request_receipt,
                plan,
                control: control_identity,
                condition_tensors: callbacks.condition_tensors,
                condition_bytes: callbacks.condition_bytes,
                boundaries: callbacks.boundaries,
                stopped: status == ffi::STATUS_STOPPED,
                image: image_digest,
                width,
                height,
                channels,
            },
            measurements: GenerationMeasurements {
                step_latency_milliseconds: callbacks.step_latency_milliseconds,
            },
        })
    }

    /// Runs text-to-image, image-to-image, inpaint, or outpaint through image
    /// ABI v2 with a full transactional scheduler-state program and writes one
    /// RGB image to caller-owned storage.
    ///
    /// Unlike [`Self::generate_advanced_controlled_to`], this path copies each
    /// post-Euler state into Rust, applies [`StepProgram::intervene`] and
    /// [`StepProgram::observe`] transactionally, and records exact
    /// scheduler-state lineage. Source, mask, reference, negative
    /// conditioning, and fixed request-local `LoRA` mechanics remain part of
    /// the same synchronous native request.
    ///
    /// # Errors
    ///
    /// Returns a request, binding, callback, native-output, scoped-cleanup,
    /// sink, or accounting error.
    pub fn generate_advanced_program_to(
        &mut self,
        request: &AdvancedImageRequest<'_>,
        program: &mut dyn StepProgram,
        sink: &mut dyn ImageOutputSink,
    ) -> Result<AdvancedProgramGenerationOutput> {
        self.run_operation(|runtime| {
            runtime.generate_advanced_program_to_inner(request, program, sink)
        })
    }

    fn generate_advanced_program_to_inner(
        &mut self,
        request: &AdvancedImageRequest<'_>,
        program: &mut dyn StepProgram,
        sink: &mut dyn ImageOutputSink,
    ) -> Result<AdvancedProgramGenerationOutput> {
        request.validate_for(self.profile)?;
        let base = request.base();
        let expected = expected_rgb_bytes(base)?;
        if sink.expected_len() != expected {
            return Err(Error::Invalid(format!(
                "image ABI v2 output has {} bytes; expected {expected}",
                sink.expected_len()
            )));
        }

        let request_receipt = request.receipt()?;
        let request_identity = request_receipt.digest()?;
        let native_inputs = AdvancedNativeInputs::new(request)?;
        let native_params = native_inputs.params(request);
        let mut components = component_map(&self.profile_receipt, &self.native_receipt)?;
        if components
            .insert("image-request-v2".to_owned(), request_identity)
            .is_some()
        {
            return Err(Error::Incompatible(
                "runtime component map repeats image-request-v2".to_owned(),
            ));
        }
        let program_identity = program.implementation().clone();
        let mut callbacks = CallbackState::new_full(
            self.profile,
            &self.profile_receipt,
            &self.native_receipt,
            base,
            components,
            program,
        )?;
        let callback_pointer = (&raw mut callbacks).cast::<c_void>();
        let mut image = std::ptr::null_mut();
        // SAFETY: Every borrowed byte view and C string remains live for this
        // synchronous call. Callback state is exclusively borrowed, and the
        // exact v2 symbol validates all descriptors before reading them.
        let status = unsafe {
            self.api.generate_image_v2(
                self.context.as_ptr(),
                &native_params,
                condition_callback,
                callback_pointer,
                step_callback,
                &mut image,
            )
        };
        if let Some(error) = callbacks.error.take() {
            if !image.is_null() {
                // SAFETY: A non-null result is one allocation transferred by
                // this exact companion.
                unsafe { self.api.free_images(image, 1) };
            }
            return Err(Error::Callback(error));
        }
        if !matches!(status, ffi::STATUS_OK | ffi::STATUS_STOPPED) {
            if !image.is_null() {
                // SAFETY: Same ownership rule as above.
                unsafe { self.api.free_images(image, 1) };
            }
            return Err(native_status_error(status));
        }
        let image = NonNull::new(image)
            .ok_or_else(|| Error::Native("native success returned a null image".to_owned()))?;
        let image_guard = NativeImageGuard {
            api: &self.api,
            image,
        };
        let (bytes_written, width, height, channels, image_digest) =
            write_image_to(image_guard.image.as_ptr(), base, sink)?;
        let plan = callbacks
            .plan
            .take()
            .ok_or_else(|| Error::Native("native generation produced no step plan".to_owned()))?;
        if callbacks.steps.is_empty() {
            return Err(Error::Native(
                "native generation produced no post-Euler steps".to_owned(),
            ));
        }
        Ok(AdvancedProgramGenerationOutput {
            bytes_written,
            receipt: AdvancedProgramGenerationReceipt {
                request: request_receipt,
                generation: GenerationReceipt {
                    profile: self.profile_receipt.clone(),
                    native: self.native_receipt.clone(),
                    session_epoch: self.session_epoch,
                    request: base.receipt()?,
                    plan,
                    program: program_identity,
                    condition_tensors: callbacks.condition_tensors,
                    condition_bytes: callbacks.condition_bytes,
                    steps: callbacks.steps,
                    stopped: status == ffi::STATUS_STOPPED,
                    image: image_digest,
                    width,
                    height,
                    channels,
                },
            },
            measurements: GenerationMeasurements {
                step_latency_milliseconds: callbacks.step_latency_milliseconds,
            },
        })
    }

    /// Runs text-to-image, image-to-image, inpaint, or outpaint through image
    /// ABI v2 and writes one RGB image to caller-owned storage.
    ///
    /// Source, mask, reference, negative-conditioning, and fixed `LoRA`
    /// identities are bound into the exact runtime plan. `LoRA` entries are applied
    /// only for this request and the native companion clears them before
    /// returning.
    ///
    /// # Errors
    ///
    /// Returns a request, binding, callback, native-output, scoped-cleanup,
    /// sink, or accounting error.
    pub fn generate_advanced_controlled_to(
        &mut self,
        request: &AdvancedImageRequest<'_>,
        control: &mut dyn BoundaryControl,
        sink: &mut dyn ImageOutputSink,
    ) -> Result<AdvancedGenerationOutput> {
        self.run_operation(|runtime| {
            runtime.generate_advanced_controlled_to_inner(request, control, sink)
        })
    }

    fn generate_advanced_controlled_to_inner(
        &mut self,
        request: &AdvancedImageRequest<'_>,
        control: &mut dyn BoundaryControl,
        sink: &mut dyn ImageOutputSink,
    ) -> Result<AdvancedGenerationOutput> {
        request.validate_for(self.profile)?;
        let base = request.base();
        let expected = expected_rgb_bytes(base)?;
        if sink.expected_len() != expected {
            return Err(Error::Invalid(format!(
                "image ABI v2 output has {} bytes; expected {expected}",
                sink.expected_len()
            )));
        }

        let request_receipt = request.receipt()?;
        let request_identity = request_receipt.digest()?;
        let native_inputs = AdvancedNativeInputs::new(request)?;
        let native_params = native_inputs.params(request);
        let mut components = component_map(&self.profile_receipt, &self.native_receipt)?;
        if components
            .insert("image-request-v2".to_owned(), request_identity)
            .is_some()
        {
            return Err(Error::Incompatible(
                "runtime component map repeats image-request-v2".to_owned(),
            ));
        }
        let control_identity = control.implementation().clone();
        let mut callbacks = CallbackState::new_control(
            self.profile,
            &self.profile_receipt,
            &self.native_receipt,
            base,
            components,
            control,
        )?;
        let callback_pointer = (&raw mut callbacks).cast::<c_void>();
        let mut image = std::ptr::null_mut();
        // SAFETY: Every borrowed byte view and C string remains live for this
        // synchronous call. Callback state is exclusively borrowed, and the
        // exact v2 symbol validates all descriptors before reading them.
        let status = unsafe {
            self.api.generate_image_v2(
                self.context.as_ptr(),
                &native_params,
                condition_callback,
                callback_pointer,
                step_callback,
                &mut image,
            )
        };
        if let Some(error) = callbacks.error.take() {
            if !image.is_null() {
                // SAFETY: A non-null result is one allocation transferred by
                // this exact companion.
                unsafe { self.api.free_images(image, 1) };
            }
            return Err(Error::Callback(error));
        }
        if !matches!(status, ffi::STATUS_OK | ffi::STATUS_STOPPED) {
            if !image.is_null() {
                // SAFETY: Same ownership rule as above.
                unsafe { self.api.free_images(image, 1) };
            }
            return Err(native_status_error(status));
        }
        let image = NonNull::new(image)
            .ok_or_else(|| Error::Native("native success returned a null image".to_owned()))?;
        let image_guard = NativeImageGuard {
            api: &self.api,
            image,
        };
        let (bytes_written, width, height, channels, image_digest) =
            write_image_to(image_guard.image.as_ptr(), base, sink)?;
        let plan = callbacks
            .plan
            .take()
            .ok_or_else(|| Error::Native("native generation produced no step plan".to_owned()))?;
        if callbacks.boundaries.is_empty() {
            return Err(Error::Native(
                "native generation produced no post-Euler boundaries".to_owned(),
            ));
        }
        Ok(AdvancedGenerationOutput {
            bytes_written,
            receipt: AdvancedGenerationReceipt {
                request: request_receipt,
                generation: ControlledGenerationReceipt {
                    profile: self.profile_receipt.clone(),
                    native: self.native_receipt.clone(),
                    session_epoch: self.session_epoch,
                    request: base.receipt()?,
                    plan,
                    control: control_identity,
                    condition_tensors: callbacks.condition_tensors,
                    condition_bytes: callbacks.condition_bytes,
                    boundaries: callbacks.boundaries,
                    stopped: status == ffi::STATUS_STOPPED,
                    image: image_digest,
                    width,
                    height,
                    channels,
                },
            },
            measurements: GenerationMeasurements {
                step_latency_milliseconds: callbacks.step_latency_milliseconds,
            },
        })
    }

    /// Encodes one exact image through the resident Krea VAE.
    ///
    /// # Errors
    ///
    /// Returns a profile, geometry, native tensor, or accounting error.
    pub fn vae_encode(&mut self, image: ImagePixels<'_>) -> Result<VaeTensorOutput> {
        self.run_operation(|runtime| runtime.vae_encode_inner(image))
    }

    fn vae_encode_inner(&mut self, image: ImagePixels<'_>) -> Result<VaeTensorOutput> {
        self.require_vae_profile()?;
        image.validate_color()?;
        self.profile
            .validate_dimensions(image.width(), image.height())?;
        let native_image = native_image_view(image);
        let mut tensor = std::ptr::null_mut();
        // SAFETY: The borrowed pixels remain live for this synchronous call,
        // and `tensor` accepts one native allocation on success.
        let status = unsafe {
            self.api
                .vae_encode_v2(self.context.as_ptr(), &native_image, &mut tensor)
        };
        if status != ffi::STATUS_OK {
            if !tensor.is_null() {
                // SAFETY: A non-null result belongs to this exact API.
                unsafe { self.api.free_tensor_v2(tensor) };
            }
            return Err(native_status_error(status));
        }
        let tensor = NonNull::new(tensor)
            .ok_or_else(|| Error::Native("VAE encode returned a null tensor".to_owned()))?;
        let tensor_guard = NativeTensorGuard {
            api: &self.api,
            tensor,
        };
        let (values, shape) = copy_native_tensor(tensor_guard.tensor.as_ptr())?;
        let tensor = VaeTensor::from_parts(values, shape.clone())?;
        let input = Digest::of_bytes("sdcpp-vae-encode-image-u8-v1", image.bytes());
        let output = tensor.digest();
        Ok(VaeTensorOutput {
            tensor,
            receipt: VaeOperationReceipt {
                profile: self.profile_receipt.clone(),
                backend: self.native_receipt.identity.clone(),
                session_epoch: self.session_epoch,
                input,
                output,
                tensor_shape: shape,
                width: 0,
                height: 0,
                channels: 0,
            },
        })
    }

    /// Decodes one exact finite native-layout tensor through the resident Krea
    /// VAE.
    ///
    /// # Errors
    ///
    /// Returns a profile, tensor, native image, or accounting error.
    pub fn vae_decode(&mut self, tensor: &VaeTensor) -> Result<VaeImageOutput> {
        self.run_operation(|runtime| runtime.vae_decode_inner(tensor))
    }

    fn vae_decode_inner(&mut self, tensor: &VaeTensor) -> Result<VaeImageOutput> {
        self.require_vae_profile()?;
        let native_tensor = TensorViewV2 {
            abi_version: IMAGE_ABI_VERSION,
            data: tensor.values().as_ptr(),
            element_count: tensor.values().len(),
            shape: tensor.shape().as_ptr(),
            rank: tensor.shape().len(),
        };
        let mut image = std::ptr::null_mut();
        // SAFETY: Tensor slices remain live for this synchronous call and
        // `image` accepts one native allocation on success.
        let status = unsafe {
            self.api
                .vae_decode_v2(self.context.as_ptr(), &native_tensor, &mut image)
        };
        if status != ffi::STATUS_OK {
            if !image.is_null() {
                // SAFETY: A non-null result belongs to this exact API.
                unsafe { self.api.free_images(image, 1) };
            }
            return Err(native_status_error(status));
        }
        let image = NonNull::new(image)
            .ok_or_else(|| Error::Native("VAE decode returned a null image".to_owned()))?;
        let image_guard = NativeImageGuard {
            api: &self.api,
            image,
        };
        let (bytes, width, height, channels) = copy_native_image(image_guard.image.as_ptr(), None)?;
        let output = Digest::of_bytes("sdcpp-vae-decode-image-u8-v1", &bytes);
        Ok(VaeImageOutput {
            bytes,
            receipt: VaeOperationReceipt {
                profile: self.profile_receipt.clone(),
                backend: self.native_receipt.identity.clone(),
                session_epoch: self.session_epoch,
                input: tensor.digest(),
                output,
                tensor_shape: tensor.shape().to_vec(),
                width,
                height,
                channels,
            },
        })
    }

    fn require_vae_profile(&self) -> Result<()> {
        if self.profile != Profile::Krea2Turbo {
            return Err(Error::Invalid(
                "direct VAE operations require a profile with an explicit VAE".to_owned(),
            ));
        }
        Ok(())
    }

    /// Clears request-local state and advances the session epoch.
    ///
    /// Image ABI v2 clears the native request-local `LoRA` stack and pending
    /// cancellation state before the epoch advances.
    ///
    /// # Errors
    ///
    /// Returns a poisoning error if the session is already uncertain.
    pub fn clear_session(&mut self) -> Result<CleanupReceipt> {
        if self.state == ExecutorState::Poisoned {
            return Err(Error::Poisoned(
                "cannot confirm cleanup after a poisoning failure".to_owned(),
            ));
        }
        if self.state != ExecutorState::Resident {
            return Err(Error::Invalid(format!(
                "cannot clear stable-diffusion.cpp while in {:?}",
                self.state
            )));
        }
        if let Some(activation) = self.krea_activation.take() {
            activation.release(self)?;
        }
        // SAFETY: This value exclusively owns the live context and no
        // operation is running while the executor is resident.
        let status = unsafe { self.api.clear_session_v2(self.context.as_ptr()) };
        if status != ffi::STATUS_OK {
            self.state = ExecutorState::Poisoned;
            return Err(Error::Poisoned(format!(
                "native session cleanup failed: {}",
                native_status_error(status)
            )));
        }
        let cleared_epoch = self.session_epoch;
        self.session_epoch = self
            .session_epoch
            .checked_add(1)
            .ok_or_else(|| Error::Poisoned("session epoch overflowed".to_owned()))?;
        Ok(CleanupReceipt {
            backend: self.native_receipt.identity.clone(),
            cleared_epoch,
            confirmed: true,
        })
    }

    /// Explicitly releases the loaded context.
    ///
    /// Native request-local state is cleared synchronously before context
    /// destruction. A previously poisoned session is consumed but cannot
    /// produce a confirmed cleanup receipt.
    ///
    /// # Errors
    ///
    /// Returns a poisoning error when the session was already uncertain.
    pub fn close(mut self) -> Result<CleanupReceipt> {
        if self.state == ExecutorState::Poisoned {
            return Err(Error::Poisoned(
                "native cleanup cannot be confirmed for a poisoned session".to_owned(),
            ));
        }
        if self.state != ExecutorState::Resident {
            return Err(Error::Invalid(format!(
                "cannot close stable-diffusion.cpp while in {:?}",
                self.state
            )));
        }
        if let Some(activation) = self.krea_activation.take() {
            activation.release(&mut self)?;
        }
        // SAFETY: This value exclusively owns the live context and consumes
        // itself immediately after this synchronous cleanup.
        let status = unsafe { self.api.clear_session_v2(self.context.as_ptr()) };
        if status != ffi::STATUS_OK {
            self.state = ExecutorState::Poisoned;
            return Err(Error::Poisoned(format!(
                "native close cleanup failed: {}",
                native_status_error(status)
            )));
        }
        Ok(CleanupReceipt {
            backend: self.native_receipt.identity.clone(),
            cleared_epoch: self.session_epoch,
            confirmed: true,
        })
    }

    fn run_operation<T>(&mut self, operation: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        match self.state {
            ExecutorState::Resident => {}
            ExecutorState::Poisoned => {
                return Err(Error::Poisoned(
                    "session cannot continue after a poisoning failure".to_owned(),
                ));
            }
            state => {
                return Err(Error::Invalid(format!(
                    "cannot execute stable-diffusion.cpp while in {state:?}"
                )));
            }
        }
        self.state = ExecutorState::Busy;
        let result = operation(self);
        self.state = match &result {
            Err(error) if error.disposition() == FailureDisposition::Poisoned => {
                ExecutorState::Poisoned
            }
            Ok(_) | Err(_) => ExecutorState::Resident,
        };
        result
    }
}

struct SliceImageSink<'a> {
    destination: &'a mut [u8],
}

impl ImageOutputSink for SliceImageSink<'_> {
    fn expected_len(&self) -> usize {
        self.destination.len()
    }

    fn write_image(&mut self, bytes: &[u8]) -> std::result::Result<(), String> {
        if bytes.len() != self.destination.len() {
            return Err("image length differs from the destination".to_owned());
        }
        self.destination.copy_from_slice(bytes);
        Ok(())
    }
}

fn open_companion(path: &Path) -> Result<(NativeApi, String, Vec<String>)> {
    let library_before = sha256_file(path)?;
    let api = NativeApi::open(path)?;
    let devices = api.devices()?;
    let library_after = sha256_file(path)?;
    if library_after != library_before {
        return Err(Error::Incompatible(
            "companion library changed while it was being loaded".to_owned(),
        ));
    }
    Ok((api, library_before, devices))
}

impl Drop for Sdcpp {
    fn drop(&mut self) {
        if let Some(activation) = self.krea_activation.take() {
            let _ = activation.release(self);
        }
        // SAFETY: `context` is owned by this value and is released exactly once
        // while the corresponding `api` library remains loaded.
        unsafe { self.api.free_context(self.context.as_ptr()) };
    }
}

struct CancellationBoundary<'a> {
    probe: &'a dyn CancellationProbe,
    implementation: Digest,
}

impl<'a> CancellationBoundary<'a> {
    fn new(probe: &'a dyn CancellationProbe) -> Self {
        Self {
            probe,
            implementation: Digest::of_bytes(
                "sdcpp-cancellation-boundary-control-v1",
                b"post-euler",
            ),
        }
    }
}

impl BoundaryControl for CancellationBoundary<'_> {
    fn implementation(&self) -> &Digest {
        &self.implementation
    }

    fn boundary(&mut self, _context: &StepContext) -> std::result::Result<ControlFlow, String> {
        Ok(if self.probe.is_cancelled() {
            ControlFlow::Stop
        } else {
            ControlFlow::Continue
        })
    }
}

impl LocalExecutor for Sdcpp {
    type Plan = ImageRequest;
    type Receipt = ControlledGenerationReceipt;
    type Error = Error;

    fn state(&self) -> ExecutorState {
        Sdcpp::state(self)
    }

    fn warm(
        &mut self,
        plan: &Self::Plan,
        cancellation: &dyn CancellationProbe,
    ) -> Result<Self::Receipt> {
        if cancellation.is_cancelled() {
            return Err(Error::Cancelled);
        }
        let mut destination = vec![0_u8; expected_rgb_bytes(plan)?];
        let mut control = CancellationBoundary::new(cancellation);
        let generated =
            self.generate_controlled_into(plan, &mut control, destination.as_mut_slice())?;
        if generated.receipt.stopped {
            Err(Error::Cancelled)
        } else {
            Ok(generated.receipt)
        }
    }

    fn execute(
        &mut self,
        plan: &Self::Plan,
        inputs: &[InputBuffer<'_>],
        outputs: &mut [OutputBuffer<'_>],
        cancellation: &dyn CancellationProbe,
    ) -> Result<Self::Receipt> {
        if !inputs.is_empty() {
            return Err(Error::Invalid(
                "baseline stable-diffusion.cpp execution accepts no input buffers".to_owned(),
            ));
        }
        if outputs.len() != 1 {
            return Err(Error::Invalid(
                "baseline stable-diffusion.cpp execution requires one RGB output".to_owned(),
            ));
        }
        if cancellation.is_cancelled() {
            return Err(Error::Cancelled);
        }
        let mut control = CancellationBoundary::new(cancellation);
        let generated =
            self.generate_controlled_into(plan, &mut control, outputs[0].bytes_mut())?;
        if generated.receipt.stopped {
            return Err(Error::Cancelled);
        }
        outputs[0]
            .set_written(generated.bytes_written)
            .map_err(|error| Error::Invalid(error.to_string()))?;
        Ok(generated.receipt)
    }

    fn clear_session(&mut self) -> Result<CleanupReceipt> {
        Sdcpp::clear_session(self)
    }

    fn close(self) -> Result<CleanupReceipt> {
        Sdcpp::close(self)
    }
}

pub(crate) struct CallbackState<'a> {
    profile: Profile,
    profile_receipt: &'a ProfileReceipt,
    native_receipt: &'a NativeRuntimeReceipt,
    request: &'a ImageRequest,
    components: BTreeMap<String, Digest>,
    program: CallbackProgram<'a>,
    condition_hasher: blake3::Hasher,
    condition_tensors: u32,
    condition_bytes: u64,
    plan: Option<DiffusionPlan>,
    steps: Vec<StepReceipt>,
    boundaries: Vec<BoundaryReceipt>,
    step_latency_milliseconds: Vec<f64>,
    next_step: u32,
    error: Option<String>,
}

enum CallbackProgram<'a> {
    Full(&'a mut dyn StepProgram),
    Control(&'a mut dyn BoundaryControl),
}

impl CallbackProgram<'_> {
    fn begin(&mut self, plan: &DiffusionPlan) -> Result<()> {
        match self {
            Self::Full(program) => contained("step program begin", || program.begin(plan)),
            Self::Control(control) => {
                contained("boundary controller begin", || control.begin(plan))
            }
        }
    }
}

impl<'a> CallbackState<'a> {
    pub(crate) fn new_full(
        profile: Profile,
        profile_receipt: &'a ProfileReceipt,
        native_receipt: &'a NativeRuntimeReceipt,
        request: &'a ImageRequest,
        components: BTreeMap<String, Digest>,
        program: &'a mut dyn StepProgram,
    ) -> Result<Self> {
        let mut condition_hasher = blake3::Hasher::new();
        condition_hasher.update(b"logit-loom\0sdcpp-conditioning-tensors-v1\0");
        hash_length_prefixed(&mut condition_hasher, request.prompt().as_bytes())?;
        condition_hasher.update(&request.width().to_le_bytes());
        condition_hasher.update(&request.height().to_le_bytes());
        condition_hasher.update(&request.cfg_scale().to_bits().to_le_bytes());
        Ok(Self {
            profile,
            profile_receipt,
            native_receipt,
            request,
            components,
            program: CallbackProgram::Full(program),
            condition_hasher,
            condition_tensors: 0,
            condition_bytes: 0,
            plan: None,
            steps: Vec::with_capacity(request.schedule().steps()),
            boundaries: Vec::new(),
            step_latency_milliseconds: Vec::with_capacity(request.schedule().steps()),
            next_step: 0,
            error: None,
        })
    }

    pub(crate) fn take_error(&mut self) -> Option<String> {
        self.error.take()
    }

    pub(crate) fn plan(&self) -> Option<&DiffusionPlan> {
        self.plan.as_ref()
    }

    pub(crate) fn last_completed_step(&self) -> Option<u32> {
        self.steps.last().map(|step| step.step_index)
    }

    pub(crate) fn native_time_ns(&self) -> Option<u64> {
        self.step_latency_milliseconds
            .iter()
            .try_fold(0_u64, |total, value| {
                let duration = Duration::try_from_secs_f64(*value / 1_000.0).ok()?;
                let nanoseconds = u64::try_from(duration.as_nanos()).ok()?;
                total.checked_add(nanoseconds)
            })
    }

    fn new_control(
        profile: Profile,
        profile_receipt: &'a ProfileReceipt,
        native_receipt: &'a NativeRuntimeReceipt,
        request: &'a ImageRequest,
        components: BTreeMap<String, Digest>,
        control: &'a mut dyn BoundaryControl,
    ) -> Result<Self> {
        let mut value = Self::new_common(
            profile,
            profile_receipt,
            native_receipt,
            request,
            components,
            CallbackProgram::Control(control),
        )?;
        value.boundaries = Vec::with_capacity(request.schedule().steps());
        Ok(value)
    }

    fn new_common(
        profile: Profile,
        profile_receipt: &'a ProfileReceipt,
        native_receipt: &'a NativeRuntimeReceipt,
        request: &'a ImageRequest,
        components: BTreeMap<String, Digest>,
        program: CallbackProgram<'a>,
    ) -> Result<Self> {
        let mut condition_hasher = blake3::Hasher::new();
        condition_hasher.update(b"logit-loom\0sdcpp-conditioning-tensors-v1\0");
        hash_length_prefixed(&mut condition_hasher, request.prompt().as_bytes())?;
        condition_hasher.update(&request.width().to_le_bytes());
        condition_hasher.update(&request.height().to_le_bytes());
        condition_hasher.update(&request.cfg_scale().to_bits().to_le_bytes());
        Ok(Self {
            profile,
            profile_receipt,
            native_receipt,
            request,
            components,
            program,
            condition_hasher,
            condition_tensors: 0,
            condition_bytes: 0,
            plan: None,
            steps: Vec::new(),
            boundaries: Vec::new(),
            step_latency_milliseconds: Vec::with_capacity(request.schedule().steps()),
            next_step: 0,
            error: None,
        })
    }

    fn condition(&mut self, raw: &ConditionTensor) -> Result<()> {
        if self.plan.is_some() {
            return Err(Error::Incompatible(
                "condition tensor arrived after sampling began".to_owned(),
            ));
        }
        if raw.abi_version != COMPANION_ABI_VERSION {
            return Err(Error::Incompatible(
                "condition callback ABI differs".to_owned(),
            ));
        }
        self.condition_tensors = self
            .condition_tensors
            .checked_add(1)
            .ok_or_else(|| Error::Incompatible("condition tensor count overflowed".to_owned()))?;
        if self.condition_tensors > MAX_CONDITION_TENSORS {
            return Err(Error::Incompatible(format!(
                "condition tensor count exceeds {MAX_CONDITION_TENSORS}"
            )));
        }
        let bytes_u64 = u64::try_from(raw.bytes)
            .map_err(|_| Error::Incompatible("condition bytes exceed u64".to_owned()))?;
        self.condition_bytes = self
            .condition_bytes
            .checked_add(bytes_u64)
            .ok_or_else(|| Error::Incompatible("condition byte count overflowed".to_owned()))?;
        if self.condition_bytes > MAX_CONDITION_BYTES {
            return Err(Error::Incompatible(format!(
                "condition bytes exceed {MAX_CONDITION_BYTES}"
            )));
        }
        let label = unsafe { bounded_native_label(raw.label)? };
        let element_bytes = match raw.dtype {
            ffi::TENSOR_F32 | ffi::TENSOR_I32 => 4_usize,
            dtype => {
                return Err(Error::Incompatible(format!(
                    "unknown condition tensor dtype {dtype}"
                )));
            }
        };
        let shape = unsafe { validated_shape(raw.shape, raw.rank, raw.bytes, element_bytes)? };
        if (raw.bytes == 0) != raw.data.is_null() {
            return Err(Error::Incompatible(
                "condition data pointer and byte count disagree".to_owned(),
            ));
        }
        let bytes = if raw.bytes == 0 {
            &[][..]
        } else {
            if !(raw.data as usize).is_multiple_of(element_bytes) {
                return Err(Error::Incompatible(
                    "condition data pointer is misaligned".to_owned(),
                ));
            }
            // SAFETY: The exact companion ABI guarantees `bytes` readable
            // bytes for this synchronous callback; bounds and nullness were
            // validated above.
            unsafe { slice::from_raw_parts(raw.data.cast::<u8>(), raw.bytes) }
        };
        if raw.dtype == ffi::TENSOR_F32
            && bytes.chunks_exact(4).any(|chunk| {
                !f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]).is_finite()
            })
        {
            return Err(Error::Incompatible(
                "condition tensor contains a non-finite value".to_owned(),
            ));
        }

        hash_length_prefixed(&mut self.condition_hasher, label.as_bytes())?;
        self.condition_hasher.update(&raw.dtype.to_le_bytes());
        self.condition_hasher.update(
            &u64::try_from(shape.len())
                .map_err(|_| Error::Incompatible("condition rank exceeds u64".to_owned()))?
                .to_le_bytes(),
        );
        for dimension in shape {
            self.condition_hasher.update(&dimension.to_le_bytes());
        }
        hash_length_prefixed(&mut self.condition_hasher, bytes)?;
        Ok(())
    }

    fn tensor_for_step(&self, raw: &Step) -> Result<TensorSpec> {
        let shape =
            // SAFETY: The callback descriptor is live for this synchronous
            // call; the helper validates rank, pointer, dimensions, and byte
            // agreement before copying the shape.
            unsafe { validated_shape(raw.shape, raw.rank, raw.state_len.saturating_mul(4), 4)? };
        let dimensions = shape
            .iter()
            .map(|dimension| {
                u64::try_from(*dimension).map_err(|_| {
                    Error::Incompatible("native tensor dimension is negative".to_owned())
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let tensor = TensorSpec::new(
            dimensions,
            TensorDType::F32,
            TensorLayout::DimensionZeroFastest,
            format!("host-f32:{}", self.native_receipt.backend),
        )
        .map_err(logit_loom_diffusion::Error::from)?;
        validate_profile_tensor(self.profile, self.request, &tensor)?;
        Ok(tensor)
    }

    fn initialize_plan(&mut self, tensor: &TensorSpec) -> Result<()> {
        if self.plan.is_some() {
            return Ok(());
        }
        let conditioning = Digest::of_bytes(
            "sdcpp-conditioning-tensors-v1",
            self.condition_hasher.clone().finalize().as_bytes(),
        );
        let rng = rng_identity(self.native_receipt, self.profile_receipt)?;
        let plan = DiffusionPlan::new(
            self.components.clone(),
            conditioning,
            rng,
            self.request.seed(),
            tensor.clone(),
            self.request.schedule().clone(),
        )
        .map_err(logit_loom_diffusion::Error::from)?;
        self.program.begin(&plan)?;
        self.plan = Some(plan);
        Ok(())
    }

    fn prepare_step_context(&mut self, raw: &Step) -> Result<StepContext> {
        if raw.abi_version != COMPANION_ABI_VERSION {
            return Err(Error::Incompatible("step callback ABI differs".to_owned()));
        }
        if !raw.elapsed_milliseconds.is_finite() || raw.elapsed_milliseconds < 0.0 {
            return Err(Error::Incompatible(
                "native step latency is non-finite or negative".to_owned(),
            ));
        }
        if self.condition_tensors == 0 {
            return Err(Error::Incompatible(
                "sampling began without condition tensors".to_owned(),
            ));
        }
        let tensor = self.tensor_for_step(raw)?;
        self.initialize_plan(&tensor)?;
        let plan = self
            .plan
            .as_ref()
            .ok_or_else(|| Error::Incompatible("step plan was not initialized".to_owned()))?;
        let expected_steps = u32::try_from(plan.schedule.steps())
            .map_err(|_| Error::Incompatible("schedule steps exceed u32".to_owned()))?;
        if raw.index != self.next_step || raw.count != expected_steps {
            return Err(Error::Incompatible(
                "native step index or total is out of sequence".to_owned(),
            ));
        }
        let context = StepContext::for_plan(
            plan,
            usize::try_from(raw.index)
                .map_err(|_| Error::Incompatible("step index exceeds usize".to_owned()))?,
        )
        .map_err(logit_loom_diffusion::Error::from)?;
        if raw.sigma_from.to_bits() != context.sigma_from.to_bits()
            || raw.sigma_to.to_bits() != context.sigma_to.to_bits()
            || tensor != context.tensor
        {
            return Err(Error::Incompatible(
                "native sigma or tensor differs from the exact plan".to_owned(),
            ));
        }
        if raw.state.is_null() || !(raw.state as usize).is_multiple_of(align_of::<f32>()) {
            return Err(Error::Incompatible(
                "native step state pointer is null or misaligned".to_owned(),
            ));
        }
        let expected_len = usize::try_from(
            tensor
                .elements()
                .map_err(logit_loom_diffusion::Error::from)?,
        )
        .map_err(|_| Error::Incompatible("tensor elements exceed usize".to_owned()))?;
        if raw.state_len != expected_len {
            return Err(Error::Incompatible(format!(
                "native state has {} elements; expected {expected_len}",
                raw.state_len
            )));
        }
        Ok(context)
    }

    fn step(&mut self, raw: &Step) -> Result<ControlFlow> {
        let context = self.prepare_step_context(raw)?;
        let control = match &mut self.program {
            CallbackProgram::Full(program) => {
                // SAFETY: The exact ABI guarantees readable/writable
                // contiguous f32 state for this synchronous callback. Pointer,
                // alignment, and length were validated before forming the
                // slice.
                let native_state = unsafe { slice::from_raw_parts_mut(raw.state, raw.state_len) };
                if native_state.iter().any(|value| !value.is_finite()) {
                    return Err(Error::Incompatible(
                        "native step state contains a non-finite value".to_owned(),
                    ));
                }
                let before_bits = native_state
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>();
                let native_digest = state_digest(native_state);
                let mut working = native_state.to_vec();
                contained("step intervention", || {
                    program.intervene(&context, &mut working)
                })?;
                if working.iter().any(|value| !value.is_finite()) {
                    return Err(Error::Callback(
                        "step intervention produced a non-finite value".to_owned(),
                    ));
                }
                let control = contained("step observer", || program.observe(&context, &working))?;
                let committed_digest = state_digest(&working);
                let changed = before_bits
                    .iter()
                    .zip(&working)
                    .filter(|(before, after)| **before != after.to_bits())
                    .count();
                native_state.copy_from_slice(&working);
                self.steps.push(StepReceipt {
                    step_index: raw.index,
                    native_state: native_digest,
                    committed_state: committed_digest,
                    elements_changed: u64::try_from(changed).map_err(|_| {
                        Error::Incompatible("changed elements exceed u64".to_owned())
                    })?,
                    stop_requested: control == ControlFlow::Stop,
                });
                control
            }
            CallbackProgram::Control(controller) => {
                let control = contained("boundary controller", || controller.boundary(&context))?;
                self.boundaries.push(BoundaryReceipt {
                    step_index: raw.index,
                    stop_requested: control == ControlFlow::Stop,
                });
                control
            }
        };
        self.step_latency_milliseconds
            .push(raw.elapsed_milliseconds);
        self.next_step = self
            .next_step
            .checked_add(1)
            .ok_or_else(|| Error::Incompatible("step accounting overflowed".to_owned()))?;
        Ok(control)
    }
}

pub(crate) unsafe extern "C" fn condition_callback(
    raw: *const ConditionTensor,
    data: *mut c_void,
) -> i32 {
    if raw.is_null() || data.is_null() {
        return ffi::CALLBACK_ERROR;
    }
    // SAFETY: The companion receives this pointer from `generate` and calls it
    // synchronously while the state and descriptor are live.
    let state = unsafe { &mut *data.cast::<CallbackState<'_>>() };
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: Nullness was checked and the exact ABI owns the descriptor.
        state.condition(unsafe { &*raw })
    }));
    match outcome {
        Ok(Ok(())) => ffi::CALLBACK_CONTINUE,
        Ok(Err(error)) => {
            state.error = Some(error.to_string());
            ffi::CALLBACK_ERROR
        }
        Err(payload) => {
            state.error = Some(format!(
                "condition callback panicked: {}",
                panic_message(&payload)
            ));
            ffi::CALLBACK_ERROR
        }
    }
}

pub(crate) unsafe extern "C" fn step_callback(raw: *const Step, data: *mut c_void) -> i32 {
    if raw.is_null() || data.is_null() {
        return ffi::CALLBACK_ERROR;
    }
    // SAFETY: Same synchronous callback ownership as `condition_callback`.
    let state = unsafe { &mut *data.cast::<CallbackState<'_>>() };
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: Nullness was checked and the exact ABI owns the descriptor.
        state.step(unsafe { &*raw })
    }));
    match outcome {
        Ok(Ok(ControlFlow::Continue)) => ffi::CALLBACK_CONTINUE,
        Ok(Ok(ControlFlow::Stop)) => ffi::CALLBACK_STOP,
        Ok(Err(error)) => {
            state.error = Some(error.to_string());
            ffi::CALLBACK_ERROR
        }
        Err(payload) => {
            state.error = Some(format!(
                "step callback panicked: {}",
                panic_message(&payload)
            ));
            ffi::CALLBACK_ERROR
        }
    }
}

struct NativeImageGuard<'a> {
    api: &'a NativeApi,
    image: NonNull<ffi::Image>,
}

impl Drop for NativeImageGuard<'_> {
    fn drop(&mut self) {
        // SAFETY: This guard exclusively owns one native image allocation.
        unsafe { self.api.free_images(self.image.as_ptr(), 1) };
    }
}

struct NativeTensorGuard<'a> {
    api: &'a NativeApi,
    tensor: NonNull<OwnedTensorV2>,
}

impl Drop for NativeTensorGuard<'_> {
    fn drop(&mut self) {
        // SAFETY: This guard exclusively owns one native tensor allocation.
        unsafe { self.api.free_tensor_v2(self.tensor.as_ptr()) };
    }
}

fn native_image_view(pixels: ImagePixels<'_>) -> ImageViewV2 {
    ImageViewV2 {
        data: pixels.bytes().as_ptr(),
        bytes: pixels.bytes().len(),
        width: pixels.width(),
        height: pixels.height(),
        channels: pixels.channels(),
    }
}

fn pointer_or_null<T>(values: &[T]) -> *const T {
    if values.is_empty() {
        std::ptr::null()
    } else {
        values.as_ptr()
    }
}

fn copy_native_tensor(pointer: *mut OwnedTensorV2) -> Result<(Vec<f32>, Vec<i64>)> {
    // SAFETY: The non-null pointer is owned by a live guard for this call.
    let tensor = unsafe { &*pointer };
    if tensor.abi_version != IMAGE_ABI_VERSION
        || tensor.element_count == 0
        || tensor.rank == 0
        || tensor.rank > crate::MAX_VAE_TENSOR_RANK
        || tensor.data.is_null()
        || tensor.shape.is_null()
        || !(tensor.data as usize).is_multiple_of(align_of::<f32>())
        || !(tensor.shape as usize).is_multiple_of(align_of::<i64>())
    {
        return Err(Error::Incompatible(
            "native VAE tensor descriptor is invalid".to_owned(),
        ));
    }
    // SAFETY: Image ABI v2 owns exactly these readable arrays until the guard
    // is dropped. Rank and element count were bounded before slice creation.
    let shape = unsafe { slice::from_raw_parts(tensor.shape, tensor.rank) }.to_vec();
    let expected = shape.iter().try_fold(1_usize, |elements, dimension| {
        usize::try_from(*dimension)
            .ok()
            .and_then(|dimension| elements.checked_mul(dimension))
    });
    let within_public_bound = u64::try_from(tensor.element_count)
        .is_ok_and(|elements| elements <= logit_loom_diffusion::MAX_TENSOR_ELEMENTS);
    if expected != Some(tensor.element_count) || !within_public_bound {
        return Err(Error::Incompatible(
            "native VAE tensor shape or element count is invalid".to_owned(),
        ));
    }
    // SAFETY: Same allocation contract as the shape slice.
    let values = unsafe { slice::from_raw_parts(tensor.data, tensor.element_count) }.to_vec();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Error::Incompatible(
            "native VAE tensor contains a non-finite value".to_owned(),
        ));
    }
    Ok((values, shape))
}

fn verify_profile_artifacts(
    catalog: &Catalog,
    artifacts: &ProfileArtifacts,
) -> Result<ProfileReceipt> {
    let profile_id = artifacts.profile().id();
    let profile = catalog
        .find_profile(profile_id)
        .ok_or_else(|| Error::Incompatible(format!("catalog has no profile {profile_id:?}")))?;
    let catalog_sha256 = catalog.packaged_sha256();
    let mut receipts = Vec::new();
    match artifacts {
        ProfileArtifacts::MiniT2i {
            diffusion_model,
            text_encoder,
        } => {
            receipts.push(
                profile
                    .verify_artifact(
                        &catalog_sha256,
                        "minit2i",
                        "minit2i-b-16/transformer/diffusion_pytorch_model.safetensors",
                        diffusion_model,
                    )?
                    .receipt()
                    .clone(),
            );
            receipts.push(
                profile
                    .verify_artifact(
                        &catalog_sha256,
                        "flan-t5-large",
                        "model.safetensors",
                        text_encoder,
                    )?
                    .receipt()
                    .clone(),
            );
        }
        ProfileArtifacts::Krea2 {
            diffusion_model,
            text_encoder,
            vae,
        } => {
            receipts.push(
                profile
                    .verify_artifact(
                        &catalog_sha256,
                        "krea-2-turbo-q6-k",
                        "TURBO/Krea-2-Turbo-Q6_K.gguf",
                        diffusion_model,
                    )?
                    .receipt()
                    .clone(),
            );
            receipts.push(
                profile
                    .verify_artifact(
                        &catalog_sha256,
                        "qwen3-vl-text-encoder",
                        "Qwen3VL-4B-Instruct-Q4_K_M.gguf",
                        text_encoder,
                    )?
                    .receipt()
                    .clone(),
            );
            receipts.push(
                profile
                    .verify_artifact(
                        &catalog_sha256,
                        "wan-2.1-vae",
                        "split_files/vae/wan_2.1_vae.safetensors",
                        vae,
                    )?
                    .receipt()
                    .clone(),
            );
        }
    }
    Ok(ProfileReceipt {
        profile_id: profile_id.to_owned(),
        catalog_sha256,
        artifacts: receipts,
    })
}

fn native_identity(
    library_sha256: &str,
    devices: &[String],
    options: &SdcppOptions,
) -> Result<Digest> {
    #[derive(Serialize)]
    struct Identity<'a> {
        library_sha256: &'a str,
        companion_abi: u32,
        upstream_commit: &'a str,
        backend: &'a str,
        params_backend: &'a str,
        devices: &'a [String],
        threads: u32,
        enable_mmap: bool,
        flash_attention: bool,
        diffusion_flash_attention: bool,
    }
    Digest::of_serializable(
        "sdcpp-native-runtime-v1",
        &Identity {
            library_sha256,
            companion_abi: COMPANION_ABI_VERSION,
            upstream_commit: UPSTREAM_COMMIT,
            backend: &options.backend,
            params_backend: &options.params_backend,
            devices,
            threads: options.threads,
            enable_mmap: options.enable_mmap,
            flash_attention: options.flash_attention,
            diffusion_flash_attention: options.diffusion_flash_attention,
        },
    )
    .map_err(logit_loom_diffusion::Error::from)
    .map_err(Into::into)
}

fn rng_identity(native: &NativeRuntimeReceipt, profile: &ProfileReceipt) -> Result<Digest> {
    Digest::of_serializable(
        "sdcpp-cpu-rng-v1",
        &(&native.identity, "CPU_RNG", "CPU_RNG", profile),
    )
    .map_err(logit_loom_diffusion::Error::from)
    .map_err(Into::into)
}

fn image_execution_backend_identity(native: &NativeRuntimeReceipt) -> Result<Digest> {
    Digest::of_serializable(
        "sdcpp-image-plan-backend-v1",
        &(
            &native.identity,
            env!("CARGO_PKG_VERSION"),
            ADAPTER_CONTRACT_VERSION,
            std::env::consts::ARCH,
            std::env::consts::OS,
            cfg!(target_endian = "little"),
        ),
    )
    .map_err(logit_loom_diffusion::Error::from)
    .map_err(Into::into)
}

fn require_backend_device(backend: &str, devices: &[String]) -> Result<()> {
    if !devices.iter().any(|device| {
        device
            .split_once('\t')
            .is_some_and(|(name, _)| name.eq_ignore_ascii_case(backend))
    }) {
        return Err(Error::Incompatible(format!(
            "requested backend {backend:?} is absent from the native device report"
        )));
    }
    Ok(())
}

pub(crate) fn path_c_string(path: &Path) -> Result<CString> {
    let value = path
        .to_str()
        .ok_or_else(|| Error::Invalid("native paths must be valid UTF-8".to_owned()))?;
    if value.is_empty() || value.len() > MAX_PATH_BYTES {
        return Err(Error::Invalid(format!(
            "native path must contain 1..={MAX_PATH_BYTES} bytes"
        )));
    }
    CString::new(value).map_err(|_| Error::Invalid("native path must not contain NUL".to_owned()))
}

fn bounded_c_string(label: &str, value: &str) -> Result<CString> {
    CString::new(value).map_err(|_| Error::Invalid(format!("{label} must not contain NUL")))
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; HASH_BUFFER_BYTES];
    loop {
        let count = file.read(&mut buffer).map_err(|source| Error::Io {
            path: path.to_owned(),
            source,
        })?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

unsafe fn bounded_native_label(pointer: *const i8) -> Result<String> {
    if pointer.is_null() {
        return Err(Error::Incompatible(
            "condition tensor label is null".to_owned(),
        ));
    }
    // SAFETY: The exact ABI promises a callback-lifetime NUL-terminated label.
    let label = unsafe { CStr::from_ptr(pointer) };
    let bytes = label.to_bytes();
    if bytes.is_empty() || bytes.len() > MAX_CONDITION_LABEL_BYTES {
        return Err(Error::Incompatible(
            "condition tensor label exceeds its bound".to_owned(),
        ));
    }
    label
        .to_str()
        .map(str::to_owned)
        .map_err(|_| Error::Incompatible("condition tensor label is not UTF-8".to_owned()))
}

unsafe fn validated_shape(
    pointer: *const i64,
    rank: usize,
    bytes: usize,
    element_bytes: usize,
) -> Result<Vec<i64>> {
    if rank == 0 {
        if !pointer.is_null() || bytes != 0 {
            return Err(Error::Incompatible(
                "empty native tensor shape is inconsistent".to_owned(),
            ));
        }
        return Ok(Vec::new());
    }
    if rank > logit_loom_diffusion::MAX_TENSOR_DIMENSIONS || pointer.is_null() {
        return Err(Error::Incompatible(
            "native tensor rank or shape pointer is invalid".to_owned(),
        ));
    }
    if !(pointer as usize).is_multiple_of(align_of::<i64>()) {
        return Err(Error::Incompatible(
            "native tensor shape pointer is misaligned".to_owned(),
        ));
    }
    // SAFETY: The exact ABI promises `rank` readable i64 dimensions for this
    // callback. Rank and pointer were checked before forming the slice.
    let shape = unsafe { slice::from_raw_parts(pointer, rank) };
    let elements = shape.iter().try_fold(1_usize, |product, dimension| {
        let dimension = usize::try_from(*dimension).map_err(|_| {
            Error::Incompatible("native tensor dimension is not positive".to_owned())
        })?;
        if dimension == 0 {
            return Err(Error::Incompatible(
                "native tensor dimension is zero".to_owned(),
            ));
        }
        product
            .checked_mul(dimension)
            .ok_or_else(|| Error::Incompatible("native tensor shape overflowed".to_owned()))
    })?;
    let expected_bytes = elements
        .checked_mul(element_bytes)
        .ok_or_else(|| Error::Incompatible("native tensor byte count overflowed".to_owned()))?;
    if expected_bytes != bytes {
        return Err(Error::Incompatible(format!(
            "native tensor shape represents {expected_bytes} bytes; callback reported {bytes}"
        )));
    }
    Ok(shape.to_vec())
}

fn validate_profile_tensor(
    profile: Profile,
    request: &ImageRequest,
    tensor: &TensorSpec,
) -> Result<()> {
    let expected = match profile {
        Profile::MiniT2iB16 => {
            vec![
                u64::from(request.width()),
                u64::from(request.height()),
                3,
                1,
            ]
        }
        Profile::Krea2Turbo => vec![
            u64::from(request.width() / 8),
            u64::from(request.height() / 8),
            16,
            1,
        ],
    };
    if tensor.shape != expected {
        return Err(Error::Incompatible(format!(
            "{} state shape is {:?}; expected {expected:?}",
            profile.id(),
            tensor.shape
        )));
    }
    Ok(())
}

fn copy_image(
    pointer: *mut ffi::Image,
    request: &ImageRequest,
) -> Result<(Vec<u8>, u32, u32, u32)> {
    copy_native_image(pointer, Some((request.width(), request.height())))
}

fn copy_native_image(
    pointer: *mut ffi::Image,
    expected_dimensions: Option<(u32, u32)>,
) -> Result<(Vec<u8>, u32, u32, u32)> {
    // SAFETY: The non-null pointer is owned by a live guard for this call.
    let image = unsafe { &*pointer };
    if !matches!(image.channel, 3 | 4)
        || expected_dimensions
            .is_some_and(|(width, height)| image.width != width || image.height != height)
    {
        let expected = expected_dimensions.map_or_else(
            || "positive dimensions with 3 or 4 channels".to_owned(),
            |(width, height)| format!("{width}x{height} with 3 or 4 channels"),
        );
        return Err(Error::Incompatible(format!(
            "native image is {}x{}x{}; expected {expected}",
            image.width, image.height, image.channel
        )));
    }
    let bytes = usize::try_from(image.width)
        .ok()
        .and_then(|width| {
            usize::try_from(image.height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| {
            usize::try_from(image.channel)
                .ok()
                .and_then(|channels| pixels.checked_mul(channels))
        })
        .ok_or_else(|| Error::Incompatible("native image byte count overflowed".to_owned()))?;
    if bytes == 0 || image.data.is_null() {
        return Err(Error::Incompatible(
            "native image data is empty or null".to_owned(),
        ));
    }
    // SAFETY: The exact ABI owns `width * height * channel` readable bytes
    // until the native image guard is dropped.
    let output = unsafe { slice::from_raw_parts(image.data, bytes) }.to_vec();
    Ok((output, image.width, image.height, image.channel))
}

fn expected_rgb_bytes(request: &ImageRequest) -> Result<usize> {
    usize::try_from(request.width())
        .ok()
        .and_then(|width| {
            usize::try_from(request.height())
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| Error::Invalid("requested RGB image byte count overflowed".to_owned()))
}

fn write_image_to(
    pointer: *mut ffi::Image,
    request: &ImageRequest,
    sink: &mut dyn ImageOutputSink,
) -> Result<(usize, u32, u32, u32, Digest)> {
    // SAFETY: The non-null pointer is owned by a live guard for this call.
    let image = unsafe { &*pointer };
    if image.width != request.width() || image.height != request.height() || image.channel != 3 {
        return Err(Error::Incompatible(format!(
            "native image is {}x{}x{}; expected {}x{}x3 for direct output",
            image.width,
            image.height,
            image.channel,
            request.width(),
            request.height()
        )));
    }
    let bytes = expected_rgb_bytes(request)?;
    if sink.expected_len() != bytes || image.data.is_null() {
        return Err(Error::Incompatible(
            "native image data or direct destination is inconsistent".to_owned(),
        ));
    }
    // SAFETY: The exact ABI owns `width * height * 3` readable bytes until the
    // native image guard is dropped. Destination has the same validated size
    // and cannot overlap the native allocation.
    let source = unsafe { slice::from_raw_parts(image.data, bytes) };
    let image_digest = Digest::of_bytes("sdcpp-image-u8-v1", source);
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sink.write_image(source))) {
        Ok(Ok(())) => Ok((
            bytes,
            image.width,
            image.height,
            image.channel,
            image_digest,
        )),
        Ok(Err(message)) => Err(Error::Output(message)),
        Err(payload) => Err(Error::Output(format!(
            "image sink panicked: {}",
            panic_message(&payload)
        ))),
    }
}

fn state_digest(state: &[f32]) -> Digest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"logit-loom\0sdcpp-step-state-f32-le-v1\0");
    hasher.update(&u64::try_from(state.len()).unwrap_or(u64::MAX).to_le_bytes());
    for value in state {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    Digest::of_bytes("sdcpp-step-state-f32-le-v1", hasher.finalize().as_bytes())
}

fn hash_length_prefixed(hasher: &mut blake3::Hasher, bytes: &[u8]) -> Result<()> {
    hasher.update(
        &u64::try_from(bytes.len())
            .map_err(|_| Error::Incompatible("hashed byte length exceeds u64".to_owned()))?
            .to_le_bytes(),
    );
    hasher.update(bytes);
    Ok(())
}

fn contained<T>(
    label: &str,
    callback: impl FnOnce() -> std::result::Result<T, String>,
) -> Result<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(callback)) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(message)) => Err(Error::Callback(format!("{label}: {message}"))),
        Err(payload) => Err(Error::Callback(format!(
            "{label} panicked: {}",
            panic_message(&payload)
        ))),
    }
}

fn panic_message(payload: &Box<dyn Any + Send>) -> String {
    payload.downcast_ref::<&str>().map_or_else(
        || {
            payload
                .downcast_ref::<String>()
                .map_or_else(|| "callback panicked".to_owned(), Clone::clone)
        },
        |message| (*message).to_owned(),
    )
}

pub(crate) fn native_status_error(status: i32) -> Error {
    match status {
        ffi::STATUS_INVALID_ARGUMENT => {
            Error::Invalid("companion rejected the bounded native arguments".to_owned())
        }
        ffi::STATUS_UNSUPPORTED => Error::Incompatible(
            "companion cannot implement the requested mechanic exactly".to_owned(),
        ),
        ffi::STATUS_CALLBACK_ERROR => {
            Error::Callback("companion reported a callback failure without Rust detail".to_owned())
        }
        ffi::STATUS_NATIVE_ERROR => Error::Native("image generation failed".to_owned()),
        other => Error::Native(format!("unknown native generation status {other}")),
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, ffi::CString};

    use super::*;

    struct MutateProgram {
        implementation: Digest,
        panic: bool,
        fail: bool,
    }

    impl StepProgram for MutateProgram {
        fn implementation(&self) -> &Digest {
            &self.implementation
        }

        fn intervene(
            &mut self,
            _context: &StepContext,
            state: &mut [f32],
        ) -> std::result::Result<(), String> {
            state[0] = 9.0;
            assert!(!self.panic, "contained mutation panic");
            if self.fail {
                Err("contained mutation error".to_owned())
            } else {
                Ok(())
            }
        }
    }

    struct StopControl {
        implementation: Digest,
        calls: u32,
    }

    impl BoundaryControl for StopControl {
        fn implementation(&self) -> &Digest {
            &self.implementation
        }

        fn boundary(&mut self, _context: &StepContext) -> std::result::Result<ControlFlow, String> {
            self.calls = self.calls.saturating_add(1);
            Ok(ControlFlow::Stop)
        }
    }

    fn callback_receipts() -> (ProfileReceipt, NativeRuntimeReceipt) {
        (
            ProfileReceipt {
                profile_id: "minit2i-b16".to_owned(),
                catalog_sha256: "catalog".to_owned(),
                artifacts: Vec::new(),
            },
            NativeRuntimeReceipt {
                library_sha256: "library".to_owned(),
                companion_abi: COMPANION_ABI_VERSION,
                upstream_commit: UPSTREAM_COMMIT.to_owned(),
                backend: "vulkan0".to_owned(),
                params_backend: "vulkan0".to_owned(),
                threads: 1,
                enable_mmap: true,
                flash_attention: false,
                diffusion_flash_attention: false,
                devices: vec!["vulkan0\ttest".to_owned()],
                identity: Digest::of_bytes("native", b"test"),
            },
        )
    }

    fn call_condition(callbacks: &mut CallbackState<'_>) -> i32 {
        let label = CString::new("cond.c_crossattn").expect("valid label");
        let data = [1.0_f32];
        let shape = [1_i64];
        let tensor = ConditionTensor {
            abi_version: COMPANION_ABI_VERSION,
            label: label.as_ptr(),
            dtype: ffi::TENSOR_F32,
            data: data.as_ptr().cast::<c_void>(),
            bytes: size_of_val(&data),
            shape: shape.as_ptr(),
            rank: shape.len(),
        };
        // SAFETY: Every descriptor pointer remains live for this synchronous
        // callback invocation.
        unsafe { condition_callback(&raw const tensor, (&raw mut *callbacks).cast::<c_void>()) }
    }

    fn call_step(
        callbacks: &mut CallbackState<'_>,
        state: &mut [f32],
        elapsed_milliseconds: f64,
    ) -> i32 {
        let shape = [16_i64, 16, 3, 1];
        let step = Step {
            abi_version: COMPANION_ABI_VERSION,
            index: 0,
            count: 2,
            sigma_from: 1.0,
            sigma_to: 0.5,
            state: state.as_mut_ptr(),
            state_len: state.len(),
            shape: shape.as_ptr(),
            rank: shape.len(),
            elapsed_milliseconds,
        };
        // SAFETY: Every descriptor pointer and the mutable state remain live
        // and exclusively borrowed for this synchronous callback invocation.
        unsafe { step_callback(&raw const step, (&raw mut *callbacks).cast::<c_void>()) }
    }

    fn run_mutation_callback(
        panic: bool,
        fail: bool,
        elapsed_milliseconds: f64,
    ) -> (i32, Vec<f32>, Option<String>, Vec<f64>) {
        let (profile, native) = callback_receipts();
        let request = ImageRequest::linear_euler("test", 16, 16, 7, 1.0, 2).expect("valid request");
        let mut components = BTreeMap::new();
        components.insert("model".to_owned(), Digest::of_bytes("model", b"test"));
        let mut program = MutateProgram {
            implementation: Digest::of_bytes("program", b"test"),
            panic,
            fail,
        };
        let mut callbacks = CallbackState::new_full(
            Profile::MiniT2iB16,
            &profile,
            &native,
            &request,
            components,
            &mut program,
        )
        .expect("callback state");
        assert_eq!(call_condition(&mut callbacks), ffi::CALLBACK_CONTINUE);
        let mut state = vec![0.0; 16 * 16 * 3];
        let status = call_step(&mut callbacks, &mut state, elapsed_milliseconds);
        (
            status,
            state,
            callbacks.error,
            callbacks.step_latency_milliseconds,
        )
    }

    #[test]
    fn malformed_shapes_are_rejected_before_slices_escape() {
        let shape = [2_i64, 3];
        // SAFETY: The test supplies a valid two-element shape pointer.
        assert!(unsafe { validated_shape(shape.as_ptr(), 2, 24, 4) }.is_ok());
        // SAFETY: The function rejects the byte mismatch without reading data.
        assert!(unsafe { validated_shape(shape.as_ptr(), 2, 20, 4) }.is_err());
        // SAFETY: Nullness is handled before dereference.
        assert!(unsafe { validated_shape(std::ptr::null(), 2, 24, 4) }.is_err());
    }

    #[test]
    fn contained_callbacks_report_errors_and_panics() {
        let error = contained::<()>("test", || Err("no".to_owned()))
            .expect_err("error should be contained");
        assert!(error.to_string().contains("test: no"));

        let panic =
            contained::<()>("test", || panic!("boom")).expect_err("panic should be contained");
        assert!(panic.to_string().contains("panicked: boom"));
    }

    #[test]
    fn callback_error_and_panic_do_not_write_back() {
        for (panic, fail) in [(false, true), (true, false)] {
            let (status, state, error, measurements) = run_mutation_callback(panic, fail, 12.5);
            assert_eq!(status, ffi::CALLBACK_ERROR);
            assert!(state.iter().all(|value| *value == 0.0));
            assert!(error.is_some());
            assert!(measurements.is_empty());
        }
    }

    #[test]
    fn invalid_native_step_latency_is_rejected_before_write_back() {
        let (status, state, error, measurements) = run_mutation_callback(false, false, f64::NAN);
        assert_eq!(status, ffi::CALLBACK_ERROR);
        assert!(state.iter().all(|value| *value == 0.0));
        assert!(error.is_some_and(|message| message.contains("latency")));
        assert!(measurements.is_empty());
    }

    #[test]
    fn successful_callback_commits_complete_state_and_receipt() {
        let (status, state, error, measurements) = run_mutation_callback(false, false, 12.5);
        assert_eq!(status, ffi::CALLBACK_CONTINUE);
        assert_eq!(state[0].to_bits(), 9.0_f32.to_bits());
        assert!(
            state[1..]
                .iter()
                .all(|value| value.to_bits() == 0.0_f32.to_bits())
        );
        assert!(error.is_none());
        assert_eq!(measurements, [12.5]);
    }

    #[test]
    fn control_only_boundary_does_not_read_copy_hash_or_mutate_state() {
        let (profile, native) = callback_receipts();
        let request = ImageRequest::linear_euler("test", 16, 16, 7, 1.0, 2).unwrap();
        let mut components = BTreeMap::new();
        components.insert("model".to_owned(), Digest::of_bytes("model", b"test"));
        let mut control = StopControl {
            implementation: Digest::of_bytes("control", b"stop"),
            calls: 0,
        };
        let mut callbacks = CallbackState::new_control(
            Profile::MiniT2iB16,
            &profile,
            &native,
            &request,
            components,
            &mut control,
        )
        .unwrap();
        assert_eq!(call_condition(&mut callbacks), ffi::CALLBACK_CONTINUE);
        let mut state = vec![f32::NAN; 16 * 16 * 3];
        let before = state
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        assert_eq!(
            call_step(&mut callbacks, &mut state, 1.25),
            ffi::CALLBACK_STOP
        );
        assert_eq!(
            state
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            before
        );
        assert!(callbacks.steps.is_empty());
        assert_eq!(
            callbacks.boundaries,
            [BoundaryReceipt {
                step_index: 0,
                stop_requested: true
            }]
        );
        assert_eq!(callbacks.step_latency_milliseconds, [1.25]);
        drop(callbacks);
        assert_eq!(control.calls, 1);
    }

    #[test]
    fn executor_errors_expose_reuse_disposition() {
        assert_eq!(
            Error::Invalid("bad request".to_owned()).disposition(),
            FailureDisposition::Rejected
        );
        assert_eq!(
            Error::Cancelled.disposition(),
            FailureDisposition::Cancelled
        );
        assert_eq!(
            Error::Native("device lost".to_owned()).disposition(),
            FailureDisposition::Poisoned
        );
        assert_eq!(
            native_status_error(ffi::STATUS_INVALID_ARGUMENT).disposition(),
            FailureDisposition::Rejected
        );
        assert_eq!(
            native_status_error(ffi::STATUS_UNSUPPORTED).disposition(),
            FailureDisposition::Rejected
        );
        assert_eq!(
            native_status_error(ffi::STATUS_CALLBACK_ERROR).disposition(),
            FailureDisposition::Poisoned
        );
    }

    #[test]
    fn profile_tensor_shape_is_exact() {
        let request =
            ImageRequest::linear_euler("test", 512, 512, 7, 6.0, 4).expect("valid request");
        let mini = TensorSpec::new(
            vec![512, 512, 3, 1],
            TensorDType::F32,
            TensorLayout::DimensionZeroFastest,
            "host-f32:vulkan",
        )
        .expect("valid tensor");
        assert!(validate_profile_tensor(Profile::MiniT2iB16, &request, &mini).is_ok());

        let wrong = TensorSpec::new(
            vec![64, 64, 16, 1],
            TensorDType::F32,
            TensorLayout::DimensionZeroFastest,
            "host-f32:vulkan",
        )
        .expect("valid tensor");
        assert!(validate_profile_tensor(Profile::MiniT2iB16, &request, &wrong).is_err());
    }

    #[test]
    fn image_copy_validates_shape_and_copies_owned_bytes() {
        let request = ImageRequest::linear_euler("test", 16, 16, 7, 1.0, 2).expect("valid request");
        let mut pixels = vec![3_u8; 16 * 16 * 3];
        let mut image = ffi::Image {
            width: 16,
            height: 16,
            channel: 3,
            data: pixels.as_mut_ptr(),
        };
        let (copy, width, height, channels) =
            copy_image(&raw mut image, &request).expect("valid image");
        assert_eq!(copy, pixels);
        assert_eq!((width, height, channels), (16, 16, 3));

        image.width = 15;
        assert!(copy_image(&raw mut image, &request).is_err());
    }
}

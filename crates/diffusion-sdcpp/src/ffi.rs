// SPDX-License-Identifier: MIT OR Apache-2.0

//! Private dynamic companion ABI.

use std::{
    ffi::{CStr, c_char, c_void},
    fmt::Write as _,
    path::Path,
    ptr,
    sync::Mutex,
};

use libloading::Library;

use crate::{
    COMPANION_ABI_VERSION, Error, IMAGE_ABI_VERSION, KREA_ACTIVATION_ABI_VERSION,
    MODEL_BLOCK_ABI_VERSION, PROGRAM_ABI_VERSION, Result, UPSTREAM_COMMIT,
};

pub(crate) const PROFILE_MINIT2I: i32 = 1;
pub(crate) const PROFILE_KREA2: i32 = 2;

pub(crate) const STATUS_OK: i32 = 0;
pub(crate) const STATUS_STOPPED: i32 = 1;
pub(crate) const STATUS_INVALID_ARGUMENT: i32 = 2;
pub(crate) const STATUS_UNSUPPORTED: i32 = 3;
pub(crate) const STATUS_CALLBACK_ERROR: i32 = 4;
pub(crate) const STATUS_NATIVE_ERROR: i32 = 5;

pub(crate) const CALLBACK_CONTINUE: i32 = 0;
pub(crate) const CALLBACK_STOP: i32 = 1;
pub(crate) const CALLBACK_ERROR: i32 = 2;

pub(crate) const TENSOR_F32: i32 = 1;
pub(crate) const TENSOR_I32: i32 = 2;

pub(crate) const OPERATION_TEXT_TO_IMAGE: i32 = 1;
pub(crate) const OPERATION_IMAGE_TO_IMAGE: i32 = 2;
pub(crate) const OPERATION_INPAINT: i32 = 3;
pub(crate) const OPERATION_OUTPAINT: i32 = 4;

pub(crate) const VALUE_BYTES_V3: i32 = 1;
pub(crate) const VALUE_IMAGE_V3: i32 = 2;
pub(crate) const VALUE_TENSOR_V3: i32 = 3;
pub(crate) const VALUE_LORA_V3: i32 = 4;
pub(crate) const VALUE_CHECKPOINT_STATE_V3: i32 = 5;
pub(crate) const VALUE_PNG_V3: i32 = 6;

pub(crate) const PROGRAM_RGB8_V3: i32 = 1;
pub(crate) const PROGRAM_RGBA8_V3: i32 = 2;
pub(crate) const PROGRAM_PNG_RGB8_V3: i32 = 3;
pub(crate) const PROGRAM_PNG_RGBA8_V3: i32 = 4;

pub(crate) const VALUE_HOST_V3: i32 = 1;
pub(crate) const VALUE_MIXED_V3: i32 = 2;

pub(crate) const MODEL_COMPONENT_KREA2_V5: i32 = 1;
pub(crate) const MODEL_BLOCK_RESIDUAL_V5: i32 = 1;
pub(crate) const STEP_ALL_V5: i32 = 1;
pub(crate) const STEP_EXACT_V5: i32 = 2;

pub(crate) const KREA_CONDITIONER_LAYER_V6: i32 = 1;
pub(crate) const KREA_POST_FUSION_V6: i32 = 2;
pub(crate) const KREA_POST_PROJECTION_V6: i32 = 3;
pub(crate) const KREA_TEXT_RESIDUAL_V6: i32 = 4;
pub(crate) const KREA_TRANSFORMER_RESIDUAL_V6: i32 = 5;
pub(crate) const KREA_PRE_DENOISER_V6: i32 = 1;
pub(crate) const KREA_TRANSITION_V6: i32 = 2;
pub(crate) const KREA_TEXT_V6: i32 = 1;
pub(crate) const KREA_IMAGE_V6: i32 = 2;
pub(crate) const KREA_REFERENCE_V6: i32 = 3;
pub(crate) const KREA_CONDITIONAL_V6: i32 = 1;
pub(crate) const KREA_UNCONDITIONAL_V6: i32 = 2;
pub(crate) const KREA_CAPTURE_DIGEST_V6: i32 = 1;
pub(crate) const KREA_CAPTURE_STATISTICS_V6: i32 = 2;
pub(crate) const KREA_CAPTURE_DEVICE_SNAPSHOT_V6: i32 = 3;
pub(crate) const KREA_DONOR_F32_ROWS_V6: i32 = 1;
pub(crate) const KREA_VECTOR_F32_ROWS_V6: i32 = 2;
pub(crate) const KREA_ORTHONORMAL_F32_ROWS_V6: i32 = 3;
pub(crate) const KREA_DONOR_TRANSPLANT_V6: i32 = 1;
pub(crate) const KREA_SCALED_VECTOR_ADD_V6: i32 = 2;
pub(crate) const KREA_SCALED_VECTOR_SUBTRACT_V6: i32 = 3;
pub(crate) const KREA_PROJECTION_REMOVAL_V6: i32 = 4;
pub(crate) const KREA_ONE_SIDED_REMOVAL_V6: i32 = 5;
pub(crate) const KREA_RESIDENT_INPUT_V6: i32 = 1;
pub(crate) const KREA_CAPTURE_INPUT_V6: i32 = 2;
pub(crate) const KREA_ALL_TOKENS_V6: i32 = 1;
pub(crate) const KREA_TOKEN_RANGES_V6: i32 = 2;
pub(crate) const KREA_CAPTURE_EVENT_V6: i32 = 1;
pub(crate) const KREA_APPLICATION_BEFORE_EVENT_V6: i32 = 2;
pub(crate) const KREA_APPLICATION_AFTER_EVENT_V6: i32 = 3;

const MAX_DEVICE_REPORT_BYTES: usize = 64 * 1024;
const MAX_DEVICE_LINES: usize = 64;
const MAX_DEVICE_LINE_BYTES: usize = 512;
const MAX_NATIVE_IDENTITY_BYTES: usize = 256;

#[repr(C)]
pub(crate) struct ContextParams {
    pub abi_version: u32,
    pub profile: i32,
    pub diffusion_model_path: *const c_char,
    pub text_encoder_path: *const c_char,
    pub vae_path: *const c_char,
    pub backend: *const c_char,
    pub params_backend: *const c_char,
    pub n_threads: i32,
    pub enable_mmap: bool,
    pub flash_attn: bool,
    pub diffusion_flash_attn: bool,
}

#[repr(C)]
pub(crate) struct ImageParams {
    pub abi_version: u32,
    pub prompt: *const c_char,
    pub width: i32,
    pub height: i32,
    pub seed: i64,
    pub cfg_scale: f32,
    pub sigmas: *const f32,
    pub sigma_count: usize,
    pub resume_state: *const f32,
    pub resume_state_len: usize,
    pub resume_next_step: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct ImageViewV2 {
    pub data: *const u8,
    pub bytes: usize,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
}

impl ImageViewV2 {
    pub(crate) const EMPTY: Self = Self {
        data: ptr::null(),
        bytes: 0,
        width: 0,
        height: 0,
        channels: 0,
    };
}

#[repr(C)]
pub(crate) struct LoraV2 {
    pub path: *const c_char,
    pub multiplier: f32,
    pub is_high_noise: bool,
}

#[repr(C)]
pub(crate) struct ImageParamsV2 {
    pub abi_version: u32,
    pub operation: i32,
    pub prompt: *const c_char,
    pub negative_prompt: *const c_char,
    pub width: i32,
    pub height: i32,
    pub seed: i64,
    pub cfg_scale: f32,
    pub strength: f32,
    pub sigmas: *const f32,
    pub sigma_count: usize,
    pub init_image: ImageViewV2,
    pub mask_image: ImageViewV2,
    pub reference_images: *const ImageViewV2,
    pub reference_image_count: usize,
    pub loras: *const LoraV2,
    pub lora_count: usize,
    pub resume_state: *const f32,
    pub resume_state_len: usize,
    pub resume_next_step: u32,
}

#[repr(C)]
pub(crate) struct OwnedTensorV2 {
    pub abi_version: u32,
    pub data: *mut f32,
    pub element_count: usize,
    pub shape: *mut i64,
    pub rank: usize,
}

#[repr(C)]
pub(crate) struct TensorViewV2 {
    pub abi_version: u32,
    pub data: *const f32,
    pub element_count: usize,
    pub shape: *const i64,
    pub rank: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct ValueHandleV3 {
    pub generation: u64,
    pub slot: u32,
    pub reserved: u32,
}

impl ValueHandleV3 {
    pub(crate) const EMPTY: Self = Self {
        generation: 0,
        slot: u32::MAX,
        reserved: 0,
    };

    pub(crate) const fn is_empty(self) -> bool {
        self.generation == 0 && self.slot == u32::MAX && self.reserved == 0
    }
}

#[derive(Default)]
#[repr(C)]
pub(crate) struct ValueDescriptorV3 {
    pub abi_version: u32,
    pub kind: i32,
    pub bytes: u64,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub dtype: i32,
    pub element_count: u64,
    pub rank: u32,
    pub shape: [i64; 8],
    pub placement: i32,
    pub host_to_device_transfers: u64,
    pub host_to_device_bytes: u64,
    pub device_to_host_transfers: u64,
    pub device_to_host_bytes: u64,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct LoraScalePointV3 {
    pub step: u32,
    pub scale: f32,
}

#[repr(C)]
pub(crate) struct LoraScheduleV3 {
    pub lora: ValueHandleV3,
    pub points: *const LoraScalePointV3,
    pub point_count: usize,
}

#[repr(C)]
pub(crate) struct ProgramImageParamsV3 {
    pub abi_version: u32,
    pub operation: i32,
    pub positive_conditioning: ValueHandleV3,
    pub negative_conditioning: ValueHandleV3,
    pub width: i32,
    pub height: i32,
    pub output_format: i32,
    pub maximum_output_bytes: u64,
    pub seed: i64,
    pub cfg_scale: f32,
    pub strength: f32,
    pub sigmas: *const f32,
    pub sigma_count: usize,
    pub init_image: ValueHandleV3,
    pub mask_image: ValueHandleV3,
    pub reference_images: *const ValueHandleV3,
    pub reference_image_count: usize,
    pub loras: *const LoraScheduleV3,
    pub lora_count: usize,
    pub checkpoint_after_step: u32,
    pub snapshot_after_steps: *const u32,
    pub snapshot_count: usize,
}

#[repr(C)]
pub(crate) struct ModelBlockOperatorV5 {
    pub operator_index: u32,
    pub component: i32,
    pub block: u32,
    pub site: i32,
    pub residual_scale: f32,
    pub step_selection: i32,
    pub steps: *const u32,
    pub step_count: usize,
}

#[repr(C)]
pub(crate) struct ProgramImageParamsV5 {
    pub abi_version: u32,
    pub image: ProgramImageParamsV3,
    pub model_block_operators: *const ModelBlockOperatorV5,
    pub model_block_operator_count: usize,
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub(crate) struct NativeModelBlockApplicationV5 {
    pub operator_index: u32,
    pub loaded_model_blocks: u32,
    pub block: u32,
    pub residual_scale: f32,
    pub graph_applications: u32,
    pub ordinary_graphs: u32,
    pub bypassed_graphs: u32,
    pub scaled_residual_graphs: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct KreaInputHandleV6 {
    pub generation: u64,
    pub slot: u32,
    pub reserved: u32,
}

impl KreaInputHandleV6 {
    pub(crate) const EMPTY: Self = Self {
        generation: 0,
        slot: u32::MAX,
        reserved: 0,
    };
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub(crate) struct KreaSiteV6 {
    pub site: u32,
    pub kind: i32,
    pub index: u32,
    pub width: u32,
    pub boundary_mask: u32,
    pub domain_mask: u32,
    pub branch_mask: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct KreaTopologyV6 {
    pub abi_version: u32,
    pub conditioner_layers: u32,
    pub transformer_blocks: u32,
    pub site_count: usize,
}

impl Default for KreaTopologyV6 {
    fn default() -> Self {
        Self {
            abi_version: KREA_ACTIVATION_ABI_VERSION,
            conditioner_layers: 0,
            transformer_blocks: 0,
            site_count: 0,
        }
    }
}

#[repr(C)]
pub(crate) struct KreaInputV6 {
    pub abi_version: u32,
    pub site: u32,
    pub rows: u32,
    pub representation: i32,
    pub values: *const f32,
    pub element_count: usize,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct KreaInputDescriptionV6 {
    pub abi_version: u32,
    pub handle: KreaInputHandleV6,
    pub site: u32,
    pub width: u32,
    pub rows: u32,
    pub representation: i32,
    pub bytes: u64,
    pub host_to_device_transfers: u64,
    pub host_to_device_bytes: u64,
}

impl Default for KreaInputDescriptionV6 {
    fn default() -> Self {
        Self {
            abi_version: 0,
            handle: KreaInputHandleV6::EMPTY,
            site: 0,
            width: 0,
            rows: 0,
            representation: 0,
            bytes: 0,
            host_to_device_transfers: 0,
            host_to_device_bytes: 0,
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct KreaTokenRangeV6 {
    pub start: u32,
    pub end: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct KreaTokenSelectionV6 {
    pub domain: i32,
    pub selection: i32,
    pub ranges: *const KreaTokenRangeV6,
    pub range_count: usize,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct KreaBoundarySelectionV6 {
    pub boundary: i32,
    pub step_selection: i32,
    pub steps: *const u32,
    pub step_count: usize,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct KreaCaptureV6 {
    pub capture_index: u32,
    pub site: u32,
    pub tokens: KreaTokenSelectionV6,
    pub boundary: KreaBoundarySelectionV6,
    pub branch: i32,
    pub retention: i32,
    pub maximum_elements: u64,
    pub maximum_host_bytes: u64,
    pub maximum_device_bytes: u64,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct KreaOperationV6 {
    pub operation_index: u32,
    pub site: u32,
    pub tokens: KreaTokenSelectionV6,
    pub boundary: KreaBoundarySelectionV6,
    pub branch: i32,
    pub operation: i32,
    pub input_source: i32,
    pub resident_input: KreaInputHandleV6,
    pub capture_input: u32,
    pub vector: u32,
    pub strength: f32,
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub(crate) struct KreaCaptureResultV6 {
    pub capture_index: u32,
    pub reached: u64,
    pub elements: u64,
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub(crate) struct KreaApplicationResultV6 {
    pub operation_index: u32,
    pub reached: u64,
    pub applied: u64,
    pub unchanged: u64,
}

#[repr(C)]
pub(crate) struct ProgramImageParamsV6 {
    pub abi_version: u32,
    pub image: ProgramImageParamsV5,
    pub captures: *const KreaCaptureV6,
    pub capture_count: usize,
    pub operations: *const KreaOperationV6,
    pub operation_count: usize,
    pub maximum_host_bytes: u64,
    pub maximum_device_bytes: u64,
    pub maximum_applications: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ProgramOutputV3 {
    pub width: u32,
    pub height: u32,
    pub format: i32,
    pub maximum_bytes: u64,
}

#[repr(C)]
pub(crate) struct ProgramImageResultV3 {
    pub abi_version: u32,
    pub primary: ValueHandleV3,
    pub checkpoint_state: ValueHandleV3,
    pub snapshot_count: usize,
}

impl Default for ProgramImageResultV3 {
    fn default() -> Self {
        Self {
            abi_version: PROGRAM_ABI_VERSION,
            primary: ValueHandleV3::EMPTY,
            checkpoint_state: ValueHandleV3::EMPTY,
            snapshot_count: 0,
        }
    }
}

#[repr(C)]
pub(crate) struct ProgramImageResultV5 {
    pub abi_version: u32,
    pub image: ProgramImageResultV3,
    pub model_block_application_count: usize,
    pub transition_words_per_operator: usize,
    pub controls_cleared: u32,
}

impl Default for ProgramImageResultV5 {
    fn default() -> Self {
        Self {
            abi_version: MODEL_BLOCK_ABI_VERSION,
            image: ProgramImageResultV3::default(),
            model_block_application_count: 0,
            transition_words_per_operator: 0,
            controls_cleared: 0,
        }
    }
}

#[repr(C)]
pub(crate) struct ProgramImageResultV6 {
    pub abi_version: u32,
    pub image: ProgramImageResultV5,
    pub capture_count: usize,
    pub operation_count: usize,
    pub activation_controls_cleared: u32,
    pub peak_host_bytes: u64,
    pub peak_device_bytes: u64,
}

impl Default for ProgramImageResultV6 {
    fn default() -> Self {
        Self {
            abi_version: KREA_ACTIVATION_ABI_VERSION,
            image: ProgramImageResultV5::default(),
            capture_count: 0,
            operation_count: 0,
            activation_controls_cleared: 0,
            peak_host_bytes: 0,
            peak_device_bytes: 0,
        }
    }
}

#[repr(C)]
pub(crate) struct Step {
    pub abi_version: u32,
    pub index: u32,
    pub count: u32,
    pub sigma_from: f32,
    pub sigma_to: f32,
    pub state: *mut f32,
    pub state_len: usize,
    pub shape: *const i64,
    pub rank: usize,
    pub elapsed_milliseconds: f64,
}

#[repr(C)]
pub(crate) struct ConditionTensor {
    pub abi_version: u32,
    pub label: *const c_char,
    pub dtype: i32,
    pub data: *const c_void,
    pub bytes: usize,
    pub shape: *const i64,
    pub rank: usize,
}

#[repr(C)]
pub(crate) struct Image {
    pub width: u32,
    pub height: u32,
    pub channel: u32,
    pub data: *mut u8,
}

pub(crate) type ConditionCallback =
    unsafe extern "C" fn(*const ConditionTensor, *mut c_void) -> i32;
pub(crate) type StepCallback = unsafe extern "C" fn(*const Step, *mut c_void) -> i32;
pub(crate) type ValueReadCallback = unsafe extern "C" fn(*const u8, usize, *mut c_void) -> i32;
pub(crate) type KreaEventCallback =
    unsafe extern "C" fn(i32, u32, u64, *const f32, usize, *mut c_void) -> i32;
type NativeLogCallback = unsafe extern "C" fn(i32, *const c_char, *mut c_void);
type SetLogCallback = unsafe extern "C" fn(Option<NativeLogCallback>, *mut c_void);

const NATIVE_LOG_ERROR: i32 = 3;

static NATIVE_ERROR_LOGS: Mutex<Vec<String>> = Mutex::new(Vec::new());

fn native_error_logs() -> std::sync::MutexGuard<'static, Vec<String>> {
    NATIVE_ERROR_LOGS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

unsafe extern "C" fn native_log_callback(level: i32, text: *const c_char, _data: *mut c_void) {
    if level != NATIVE_LOG_ERROR {
        return;
    }
    let message = if text.is_null() {
        "native logger emitted a null error message".to_owned()
    } else {
        // SAFETY: stable-diffusion.cpp promises a live NUL-terminated string
        // for the duration of this synchronous callback.
        let bytes = unsafe { CStr::from_ptr(text) }.to_bytes();
        if let Ok(message) = std::str::from_utf8(bytes) {
            message.to_owned()
        } else {
            let mut encoded = String::with_capacity(bytes.len().saturating_mul(2));
            for byte in bytes {
                let _ = write!(encoded, "{byte:02x}");
            }
            format!("native logger emitted non-UTF-8 error bytes: {encoded}")
        }
    };
    native_error_logs().push(message);
}

pub(crate) fn take_native_error_logs() -> Vec<String> {
    std::mem::take(&mut *native_error_logs())
}

fn clear_native_error_logs() {
    native_error_logs().clear();
}

type AbiVersion = unsafe extern "C" fn() -> u32;
type UpstreamCommit = unsafe extern "C" fn() -> *const c_char;
type NewContext = unsafe extern "C" fn(*const ContextParams) -> *mut c_void;
type FreeContext = unsafe extern "C" fn(*mut c_void);
type GenerateImage = unsafe extern "C" fn(
    *mut c_void,
    *const ImageParams,
    Option<ConditionCallback>,
    *mut c_void,
    Option<StepCallback>,
    *mut c_void,
    *mut *mut Image,
) -> i32;
type GenerateImageV2 = unsafe extern "C" fn(
    *mut c_void,
    *const ImageParamsV2,
    Option<ConditionCallback>,
    *mut c_void,
    Option<StepCallback>,
    *mut c_void,
    *mut *mut Image,
) -> i32;
type VaeEncodeV2 =
    unsafe extern "C" fn(*mut c_void, *const ImageViewV2, *mut *mut OwnedTensorV2) -> i32;
type VaeDecodeV2 = unsafe extern "C" fn(*mut c_void, *const TensorViewV2, *mut *mut Image) -> i32;
type FreeTensorV2 = unsafe extern "C" fn(*mut OwnedTensorV2);
type ClearSessionV2 = unsafe extern "C" fn(*mut c_void) -> i32;
type ProgramBeginV3 = unsafe extern "C" fn(*mut c_void, usize, u64, *mut *mut c_void) -> i32;
type ProgramImportBytesV3 =
    unsafe extern "C" fn(*mut c_void, *const u8, usize, *mut ValueHandleV3) -> i32;
type ProgramImportImageV3 =
    unsafe extern "C" fn(*mut c_void, *const ImageViewV2, *mut ValueHandleV3) -> i32;
type ProgramImportTensorV3 =
    unsafe extern "C" fn(*mut c_void, *const TensorViewV2, bool, *mut ValueHandleV3) -> i32;
type ProgramImportLoraV3 =
    unsafe extern "C" fn(*mut c_void, *const c_char, bool, *mut ValueHandleV3) -> i32;
type ProgramMaskBlendV3 = unsafe extern "C" fn(
    *mut c_void,
    ValueHandleV3,
    ValueHandleV3,
    ValueHandleV3,
    *mut ValueHandleV3,
) -> i32;
type ProgramVaeEncodeV3 =
    unsafe extern "C" fn(*mut c_void, ValueHandleV3, *mut ValueHandleV3) -> i32;
type ProgramVaeDecodeV3 =
    unsafe extern "C" fn(*mut c_void, ValueHandleV3, u32, u32, i32, u64, *mut ValueHandleV3) -> i32;
type ProgramGenerateImageV3 = unsafe extern "C" fn(
    *mut c_void,
    *const ProgramImageParamsV3,
    Option<ConditionCallback>,
    *mut c_void,
    Option<StepCallback>,
    *mut c_void,
    *mut ValueHandleV3,
    usize,
    *mut ProgramImageResultV3,
) -> i32;
type ProgramGenerateImageV5 = unsafe extern "C" fn(
    *mut c_void,
    *const ProgramImageParamsV5,
    Option<ConditionCallback>,
    *mut c_void,
    Option<StepCallback>,
    *mut c_void,
    *mut ValueHandleV3,
    usize,
    *mut NativeModelBlockApplicationV5,
    usize,
    *mut u64,
    usize,
    *mut ProgramImageResultV5,
) -> i32;
type KreaTopologyFnV6 =
    unsafe extern "C" fn(*const c_void, *mut KreaTopologyV6, *mut KreaSiteV6, usize) -> i32;
type KreaImportInputV6 =
    unsafe extern "C" fn(*mut c_void, *const KreaInputV6, *mut KreaInputDescriptionV6) -> i32;
type KreaDescribeInputV6 =
    unsafe extern "C" fn(*const c_void, KreaInputHandleV6, *mut KreaInputDescriptionV6) -> i32;
type KreaReleaseInputV6 = unsafe extern "C" fn(*mut c_void, KreaInputHandleV6) -> i32;
type KreaClearInputsV6 = unsafe extern "C" fn(*mut c_void) -> i32;
type ProgramGenerateImageV6 = unsafe extern "C" fn(
    *mut c_void,
    *const ProgramImageParamsV6,
    Option<ConditionCallback>,
    *mut c_void,
    Option<StepCallback>,
    *mut c_void,
    Option<KreaEventCallback>,
    *mut c_void,
    *mut ValueHandleV3,
    usize,
    *mut NativeModelBlockApplicationV5,
    usize,
    *mut u64,
    usize,
    *mut KreaCaptureResultV6,
    usize,
    *mut KreaApplicationResultV6,
    usize,
    *mut ProgramImageResultV6,
) -> i32;
type ProgramDescribeV3 =
    unsafe extern "C" fn(*const c_void, ValueHandleV3, *mut ValueDescriptorV3) -> i32;
type ProgramReadV3 = unsafe extern "C" fn(
    *const c_void,
    ValueHandleV3,
    Option<ValueReadCallback>,
    *mut c_void,
) -> i32;
type ProgramCopyV3 =
    unsafe extern "C" fn(*const c_void, ValueHandleV3, *mut u8, usize, *mut usize) -> i32;
type ProgramReleaseV3 = unsafe extern "C" fn(*mut c_void, ValueHandleV3) -> i32;
type ProgramFinishV3 = unsafe extern "C" fn(*mut c_void, bool, *mut u64) -> i32;
type FreeImages = unsafe extern "C" fn(*mut Image, i32);
type ListDevices = unsafe extern "C" fn(*mut c_char, usize) -> usize;

#[derive(Clone, Copy)]
struct Functions {
    set_log_callback: SetLogCallback,
    new_context: NewContext,
    free_context: FreeContext,
    generate_image: GenerateImage,
    generate_image_v2: GenerateImageV2,
    vae_encode_v2: VaeEncodeV2,
    vae_decode_v2: VaeDecodeV2,
    free_tensor_v2: FreeTensorV2,
    clear_session_v2: ClearSessionV2,
    program_begin_v3: ProgramBeginV3,
    program_import_bytes_v3: ProgramImportBytesV3,
    program_import_image_v3: ProgramImportImageV3,
    program_import_tensor_v3: ProgramImportTensorV3,
    program_import_lora_v3: ProgramImportLoraV3,
    program_mask_blend_v3: ProgramMaskBlendV3,
    program_vae_encode_v3: ProgramVaeEncodeV3,
    program_vae_decode_v3: ProgramVaeDecodeV3,
    program_generate_image_v3: ProgramGenerateImageV3,
    program_generate_image_v5: ProgramGenerateImageV5,
    krea_topology_v6: KreaTopologyFnV6,
    krea_import_input_v6: KreaImportInputV6,
    krea_describe_input_v6: KreaDescribeInputV6,
    krea_release_input_v6: KreaReleaseInputV6,
    krea_clear_inputs_v6: KreaClearInputsV6,
    program_generate_image_v6: ProgramGenerateImageV6,
    program_describe_v3: ProgramDescribeV3,
    program_read_v3: ProgramReadV3,
    program_copy_v3: ProgramCopyV3,
    program_release_v3: ProgramReleaseV3,
    program_finish_v3: ProgramFinishV3,
    free_images: FreeImages,
    list_devices: ListDevices,
}

pub(crate) struct NativeApi {
    functions: Functions,
    _library: Library,
}

impl NativeApi {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        // SAFETY: Loading executable code is this module's explicit boundary.
        // The exact ABI and commit are checked before any model call.
        let library = unsafe { Library::new(path) }.map_err(|error| Error::Library {
            path: path.to_owned(),
            message: error.to_string(),
        })?;
        // SAFETY: Each requested symbol has the signature in the versioned
        // companion header. ABI/commit queries are called before retaining the
        // remaining table as usable.
        let abi_version = unsafe { load_symbol::<AbiVersion>(&library, b"sd_loom_abi_version\0")? };
        // SAFETY: Version query takes no pointers and returns a value.
        let actual_abi = unsafe { abi_version() };
        if actual_abi != COMPANION_ABI_VERSION {
            return Err(Error::Incompatible(format!(
                "ABI version is {actual_abi}; expected {COMPANION_ABI_VERSION}"
            )));
        }
        // SAFETY: See the symbol contract above.
        let upstream_commit =
            unsafe { load_symbol::<UpstreamCommit>(&library, b"sd_loom_upstream_commit\0")? };
        // SAFETY: The matching ABI promises a static NUL-terminated string.
        let commit = unsafe { bounded_c_string(upstream_commit(), MAX_NATIVE_IDENTITY_BYTES)? };
        if commit != UPSTREAM_COMMIT {
            return Err(Error::Incompatible(format!(
                "upstream commit is {commit:?}; expected {UPSTREAM_COMMIT}"
            )));
        }

        // SAFETY: The ABI and commit are now exact, and function pointers stay
        // valid because `library` is retained in this value.
        let functions = unsafe {
            Functions {
                set_log_callback: load_symbol(&library, b"sd_set_log_callback\0")?,
                new_context: load_symbol(&library, b"sd_loom_new_ctx_v1\0")?,
                free_context: load_symbol(&library, b"free_sd_ctx\0")?,
                generate_image: load_symbol(&library, b"sd_loom_generate_image_v1\0")?,
                generate_image_v2: load_symbol(&library, b"sd_loom_generate_image_v2\0")?,
                vae_encode_v2: load_symbol(&library, b"sd_loom_vae_encode_v2\0")?,
                vae_decode_v2: load_symbol(&library, b"sd_loom_vae_decode_v2\0")?,
                free_tensor_v2: load_symbol(&library, b"sd_loom_free_tensor_v2\0")?,
                clear_session_v2: load_symbol(&library, b"sd_loom_clear_session_v2\0")?,
                program_begin_v3: load_symbol(&library, b"sd_loom_program_begin_v3\0")?,
                program_import_bytes_v3: load_symbol(
                    &library,
                    b"sd_loom_program_import_bytes_v3\0",
                )?,
                program_import_image_v3: load_symbol(
                    &library,
                    b"sd_loom_program_import_image_v3\0",
                )?,
                program_import_tensor_v3: load_symbol(
                    &library,
                    b"sd_loom_program_import_tensor_v3\0",
                )?,
                program_import_lora_v3: load_symbol(&library, b"sd_loom_program_import_lora_v3\0")?,
                program_mask_blend_v3: load_symbol(&library, b"sd_loom_program_mask_blend_v3\0")?,
                program_vae_encode_v3: load_symbol(&library, b"sd_loom_program_vae_encode_v3\0")?,
                program_vae_decode_v3: load_symbol(&library, b"sd_loom_program_vae_decode_v3\0")?,
                program_generate_image_v3: load_symbol(
                    &library,
                    b"sd_loom_program_generate_image_v3\0",
                )?,
                program_generate_image_v5: load_symbol(
                    &library,
                    b"sd_loom_program_generate_image_v5\0",
                )?,
                krea_topology_v6: load_symbol(&library, b"sd_loom_krea_topology_v6\0")?,
                krea_import_input_v6: load_symbol(&library, b"sd_loom_krea_import_input_v6\0")?,
                krea_describe_input_v6: load_symbol(&library, b"sd_loom_krea_describe_input_v6\0")?,
                krea_release_input_v6: load_symbol(&library, b"sd_loom_krea_release_input_v6\0")?,
                krea_clear_inputs_v6: load_symbol(&library, b"sd_loom_krea_clear_inputs_v6\0")?,
                program_generate_image_v6: load_symbol(
                    &library,
                    b"sd_loom_program_generate_image_v6\0",
                )?,
                program_describe_v3: load_symbol(&library, b"sd_loom_program_describe_v3\0")?,
                program_read_v3: load_symbol(&library, b"sd_loom_program_read_v3\0")?,
                program_copy_v3: load_symbol(&library, b"sd_loom_program_copy_v3\0")?,
                program_release_v3: load_symbol(&library, b"sd_loom_program_release_v3\0")?,
                program_finish_v3: load_symbol(&library, b"sd_loom_program_finish_v3\0")?,
                free_images: load_symbol(&library, b"free_sd_images\0")?,
                list_devices: load_symbol(&library, b"sd_list_devices\0")?,
            }
        };
        // SAFETY: The callback has the exact upstream signature, retains no
        // borrowed native data. The process-wide synchronized collector also
        // retains errors emitted by native backend threads.
        unsafe {
            (functions.set_log_callback)(Some(native_log_callback), ptr::null_mut());
        }
        Ok(Self {
            functions,
            _library: library,
        })
    }

    pub(crate) fn devices(&self) -> Result<Vec<String>> {
        // SAFETY: Null/zero is the documented size query.
        let required = unsafe { (self.functions.list_devices)(ptr::null_mut(), 0) };
        if required == 0 || required > MAX_DEVICE_REPORT_BYTES {
            return Err(Error::Incompatible(format!(
                "native device report requires {required} bytes"
            )));
        }
        let capacity = required.checked_add(1).ok_or_else(|| {
            Error::Incompatible("native device report size overflowed".to_owned())
        })?;
        let mut bytes = vec![0_u8; capacity];
        // SAFETY: The buffer has `required + 1` writable bytes.
        let reported = unsafe {
            (self.functions.list_devices)(bytes.as_mut_ptr().cast::<c_char>(), bytes.len())
        };
        if reported != required || bytes[required] != 0 {
            return Err(Error::Incompatible(
                "native device report changed between bounded reads".to_owned(),
            ));
        }
        bytes.truncate(required);
        let report = String::from_utf8(bytes).map_err(|_| {
            Error::Incompatible("native device report is not valid UTF-8".to_owned())
        })?;
        parse_devices(&report)
    }

    /// Creates one native context.
    ///
    /// # Safety
    ///
    /// Every pointer in `params` must refer to a live NUL-terminated string
    /// for the duration of this synchronous call.
    pub(crate) unsafe fn new_context(&self, params: &ContextParams) -> *mut c_void {
        clear_native_error_logs();
        // SAFETY: Forwarded from this method's caller contract.
        unsafe { (self.functions.new_context)(params) }
    }

    /// Releases one context previously returned by this exact API.
    ///
    /// # Safety
    ///
    /// `context` must be non-null, owned, live, and released only once.
    pub(crate) unsafe fn free_context(&self, context: *mut c_void) {
        // SAFETY: Forwarded from this method's caller contract.
        unsafe { (self.functions.free_context)(context) };
    }

    /// Runs one synchronous image generation.
    ///
    /// # Safety
    ///
    /// Context, parameter, callback, and callback-data pointers must satisfy
    /// the versioned companion ABI for the complete call.
    pub(crate) unsafe fn generate_image(
        &self,
        context: *mut c_void,
        params: &ImageParams,
        condition_callback: ConditionCallback,
        callback_data: *mut c_void,
        step_callback: StepCallback,
        image_out: &mut *mut Image,
    ) -> i32 {
        clear_native_error_logs();
        // SAFETY: Forwarded from this method's caller contract.
        unsafe {
            (self.functions.generate_image)(
                context,
                params,
                Some(condition_callback),
                callback_data,
                Some(step_callback),
                callback_data,
                image_out,
            )
        }
    }

    /// Runs one image ABI v2 request.
    ///
    /// # Safety
    ///
    /// Every pointer in `params` and both callback states must remain valid
    /// for the synchronous call. `image_out` must accept exactly one
    /// transferred native allocation on success.
    pub(crate) unsafe fn generate_image_v2(
        &self,
        context: *mut c_void,
        params: &ImageParamsV2,
        condition_callback: ConditionCallback,
        callback_data: *mut c_void,
        step_callback: StepCallback,
        image_out: &mut *mut Image,
    ) -> i32 {
        clear_native_error_logs();
        debug_assert_eq!(params.abi_version, IMAGE_ABI_VERSION);
        // SAFETY: Forwarded from this method's caller contract.
        unsafe {
            (self.functions.generate_image_v2)(
                context,
                params,
                Some(condition_callback),
                callback_data,
                Some(step_callback),
                callback_data,
                image_out,
            )
        }
    }

    /// Encodes one validated image into an owned native tensor.
    ///
    /// # Safety
    ///
    /// The image view must remain live for the synchronous call and
    /// `tensor_out` must accept one transferred native allocation.
    pub(crate) unsafe fn vae_encode_v2(
        &self,
        context: *mut c_void,
        image: &ImageViewV2,
        tensor_out: &mut *mut OwnedTensorV2,
    ) -> i32 {
        clear_native_error_logs();
        // SAFETY: Forwarded from this method's caller contract.
        unsafe { (self.functions.vae_encode_v2)(context, image, tensor_out) }
    }

    /// Decodes one validated tensor view into an owned native image.
    ///
    /// # Safety
    ///
    /// The tensor view must remain live for the synchronous call and
    /// `image_out` must accept one transferred native allocation.
    pub(crate) unsafe fn vae_decode_v2(
        &self,
        context: *mut c_void,
        tensor: &TensorViewV2,
        image_out: &mut *mut Image,
    ) -> i32 {
        clear_native_error_logs();
        debug_assert_eq!(tensor.abi_version, IMAGE_ABI_VERSION);
        // SAFETY: Forwarded from this method's caller contract.
        unsafe { (self.functions.vae_decode_v2)(context, tensor, image_out) }
    }

    /// Releases one native image-ABI tensor.
    ///
    /// # Safety
    ///
    /// `tensor` must be null or an allocation returned by this exact library.
    pub(crate) unsafe fn free_tensor_v2(&self, tensor: *mut OwnedTensorV2) {
        // SAFETY: Forwarded from this method's caller contract.
        unsafe { (self.functions.free_tensor_v2)(tensor) };
    }

    /// Clears every request-scoped image ABI v2 state.
    ///
    /// # Safety
    ///
    /// `context` must be the exclusively owned live context associated with
    /// this exact function table.
    pub(crate) unsafe fn clear_session_v2(&self, context: *mut c_void) -> i32 {
        clear_native_error_logs();
        // SAFETY: Forwarded from this method's caller contract.
        unsafe { (self.functions.clear_session_v2)(context) }
    }

    /// Creates one request-scoped native value arena.
    ///
    /// # Safety
    ///
    /// `context` must be exclusively owned and live. `program_out` must accept
    /// one opaque program allocation and that allocation must be finished
    /// exactly once.
    pub(crate) unsafe fn program_begin_v3(
        &self,
        context: *mut c_void,
        maximum_values: usize,
        maximum_bytes: u64,
        program_out: &mut *mut c_void,
    ) -> i32 {
        clear_native_error_logs();
        // SAFETY: Forwarded from this method's caller contract.
        unsafe {
            (self.functions.program_begin_v3)(context, maximum_values, maximum_bytes, program_out)
        }
    }

    /// Imports exact bytes into one live v3 arena.
    ///
    /// # Safety
    ///
    /// `program` must be the live allocation returned by this API. `bytes`
    /// remains readable for the synchronous call and `value_out` is writable.
    pub(crate) unsafe fn program_import_bytes_v3(
        &self,
        program: *mut c_void,
        bytes: &[u8],
        value_out: &mut ValueHandleV3,
    ) -> i32 {
        clear_native_error_logs();
        // SAFETY: Forwarded from this method's caller contract.
        unsafe {
            (self.functions.program_import_bytes_v3)(
                program,
                bytes.as_ptr(),
                bytes.len(),
                value_out,
            )
        }
    }

    /// Imports one exact image view into a live v3 arena.
    ///
    /// # Safety
    ///
    /// The program and image view must remain valid for the synchronous call.
    pub(crate) unsafe fn program_import_image_v3(
        &self,
        program: *mut c_void,
        image: &ImageViewV2,
        value_out: &mut ValueHandleV3,
    ) -> i32 {
        clear_native_error_logs();
        // SAFETY: Forwarded from this method's caller contract.
        unsafe { (self.functions.program_import_image_v3)(program, image, value_out) }
    }

    /// Imports one exact finite tensor into a live v3 arena.
    ///
    /// # Safety
    ///
    /// The program and tensor view must remain valid for the synchronous call.
    pub(crate) unsafe fn program_import_tensor_v3(
        &self,
        program: *mut c_void,
        tensor: &TensorViewV2,
        checkpoint_state: bool,
        value_out: &mut ValueHandleV3,
    ) -> i32 {
        clear_native_error_logs();
        // SAFETY: Forwarded from this method's caller contract.
        unsafe {
            (self.functions.program_import_tensor_v3)(program, tensor, checkpoint_state, value_out)
        }
    }

    /// Imports one caller-retained descriptor path as a v3 `LoRA` value.
    ///
    /// # Safety
    ///
    /// `path` must remain NUL-terminated and readable for the synchronous call.
    pub(crate) unsafe fn program_import_lora_v3(
        &self,
        program: *mut c_void,
        path: *const c_char,
        high_noise: bool,
        value_out: &mut ValueHandleV3,
    ) -> i32 {
        clear_native_error_logs();
        // SAFETY: Forwarded from this method's caller contract.
        unsafe { (self.functions.program_import_lora_v3)(program, path, high_noise, value_out) }
    }

    /// Executes one deterministic arena-local RGB8 mask blend.
    ///
    /// # Safety
    ///
    /// Every handle must belong to the live program generation.
    pub(crate) unsafe fn program_mask_blend_v3(
        &self,
        program: *mut c_void,
        base: ValueHandleV3,
        overlay: ValueHandleV3,
        mask: ValueHandleV3,
        value_out: &mut ValueHandleV3,
    ) -> i32 {
        clear_native_error_logs();
        // SAFETY: Forwarded from this method's caller contract.
        unsafe { (self.functions.program_mask_blend_v3)(program, base, overlay, mask, value_out) }
    }

    /// Executes one arena-local direct VAE encode.
    ///
    /// # Safety
    ///
    /// The handle must identify one live compatible image in `program`.
    pub(crate) unsafe fn program_vae_encode_v3(
        &self,
        program: *mut c_void,
        image: ValueHandleV3,
        value_out: &mut ValueHandleV3,
    ) -> i32 {
        clear_native_error_logs();
        // SAFETY: Forwarded from this method's caller contract.
        unsafe { (self.functions.program_vae_encode_v3)(program, image, value_out) }
    }

    /// Executes one arena-local direct VAE decode.
    ///
    /// # Safety
    ///
    /// The handle must identify one live compatible tensor in `program`.
    pub(crate) unsafe fn program_vae_decode_v3(
        &self,
        program: *mut c_void,
        tensor: ValueHandleV3,
        output: &ProgramOutputV3,
        value_out: &mut ValueHandleV3,
    ) -> i32 {
        clear_native_error_logs();
        // SAFETY: Forwarded from this method's caller contract.
        unsafe {
            (self.functions.program_vae_decode_v3)(
                program,
                tensor,
                output.width,
                output.height,
                output.format,
                output.maximum_bytes,
                value_out,
            )
        }
    }

    /// Executes one resident diffusion operation.
    ///
    /// # Safety
    ///
    /// All handles, arrays, callbacks, and callback states must remain live for
    /// the complete synchronous call. Output storage must match the declared
    /// snapshot count.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn program_generate_image_v3(
        &self,
        program: *mut c_void,
        params: &ProgramImageParamsV3,
        condition_callback: ConditionCallback,
        condition_callback_data: *mut c_void,
        step_callback: StepCallback,
        step_callback_data: *mut c_void,
        snapshots: &mut [ValueHandleV3],
        result_out: &mut ProgramImageResultV3,
    ) -> i32 {
        clear_native_error_logs();
        debug_assert_eq!(params.abi_version, PROGRAM_ABI_VERSION);
        // SAFETY: Forwarded from this method's caller contract.
        unsafe {
            (self.functions.program_generate_image_v3)(
                program,
                params,
                Some(condition_callback),
                condition_callback_data,
                Some(step_callback),
                step_callback_data,
                snapshots.as_mut_ptr(),
                snapshots.len(),
                result_out,
            )
        }
    }

    /// Executes one resident diffusion operation with typed model-block
    /// controls.
    ///
    /// # Safety
    ///
    /// All handles, arrays, nested step arrays, callbacks, and callback states
    /// must remain live for the complete synchronous call. Output storage must
    /// match the declared snapshot count.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn program_generate_image_v5(
        &self,
        program: *mut c_void,
        params: &ProgramImageParamsV5,
        condition_callback: ConditionCallback,
        condition_callback_data: *mut c_void,
        step_callback: StepCallback,
        step_callback_data: *mut c_void,
        snapshots: &mut [ValueHandleV3],
        applications: &mut [NativeModelBlockApplicationV5],
        transition_masks: &mut [u64],
        result_out: &mut ProgramImageResultV5,
    ) -> i32 {
        clear_native_error_logs();
        debug_assert_eq!(params.abi_version, MODEL_BLOCK_ABI_VERSION);
        debug_assert_eq!(params.image.abi_version, PROGRAM_ABI_VERSION);
        // SAFETY: Forwarded from this method's caller contract.
        unsafe {
            (self.functions.program_generate_image_v5)(
                program,
                params,
                Some(condition_callback),
                condition_callback_data,
                Some(step_callback),
                step_callback_data,
                snapshots.as_mut_ptr(),
                snapshots.len(),
                applications.as_mut_ptr(),
                applications.len(),
                transition_masks.as_mut_ptr(),
                transition_masks.len(),
                result_out,
            )
        }
    }

    /// Queries the exact loaded Krea topology. Passing an empty site slice is
    /// the count-only phase of the native two-call contract.
    ///
    /// # Safety
    ///
    /// `context` must identify the live context that owns this API table.
    pub(crate) unsafe fn krea_topology_v6(
        &self,
        context: *const c_void,
        topology: &mut KreaTopologyV6,
        sites: &mut [KreaSiteV6],
    ) -> i32 {
        clear_native_error_logs();
        // SAFETY: Forwarded from this method's caller contract.
        unsafe {
            (self.functions.krea_topology_v6)(
                context,
                topology,
                if sites.is_empty() {
                    ptr::null_mut()
                } else {
                    sites.as_mut_ptr()
                },
                sites.len(),
            )
        }
    }

    /// Imports one finite Krea input into native resident device storage.
    ///
    /// # Safety
    ///
    /// The context and input pointer must remain live for the synchronous call.
    pub(crate) unsafe fn krea_import_input_v6(
        &self,
        context: *mut c_void,
        input: &KreaInputV6,
        description: &mut KreaInputDescriptionV6,
    ) -> i32 {
        clear_native_error_logs();
        // SAFETY: Forwarded from this method's caller contract.
        unsafe { (self.functions.krea_import_input_v6)(context, input, description) }
    }

    /// Re-reads native placement evidence for one resident Krea input.
    ///
    /// # Safety
    ///
    /// The context and handle must belong to the same live session.
    pub(crate) unsafe fn krea_describe_input_v6(
        &self,
        context: *const c_void,
        input: KreaInputHandleV6,
        description: &mut KreaInputDescriptionV6,
    ) -> i32 {
        clear_native_error_logs();
        // SAFETY: Forwarded from this method's caller contract.
        unsafe { (self.functions.krea_describe_input_v6)(context, input, description) }
    }

    /// Releases one native resident Krea input.
    ///
    /// # Safety
    ///
    /// The context and handle must belong to the same live session.
    pub(crate) unsafe fn krea_release_input_v6(
        &self,
        context: *mut c_void,
        input: KreaInputHandleV6,
    ) -> i32 {
        clear_native_error_logs();
        // SAFETY: Forwarded from this method's caller contract.
        unsafe { (self.functions.krea_release_input_v6)(context, input) }
    }

    /// Clears every native resident Krea input and advances its handle epoch.
    ///
    /// # Safety
    ///
    /// `context` must identify the live context that owns this API table.
    pub(crate) unsafe fn krea_clear_inputs_v6(&self, context: *mut c_void) -> i32 {
        clear_native_error_logs();
        // SAFETY: Forwarded from this method's caller contract.
        unsafe { (self.functions.krea_clear_inputs_v6)(context) }
    }

    /// Executes one resident image operation with native Krea activation
    /// controls and exact event callbacks.
    ///
    /// # Safety
    ///
    /// Every nested array, callback state, handle, and output slice must stay
    /// live for the complete synchronous call.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn program_generate_image_v6(
        &self,
        program: *mut c_void,
        params: &ProgramImageParamsV6,
        condition_callback: ConditionCallback,
        condition_callback_data: *mut c_void,
        step_callback: StepCallback,
        step_callback_data: *mut c_void,
        activation_callback: KreaEventCallback,
        activation_callback_data: *mut c_void,
        snapshots: &mut [ValueHandleV3],
        model_blocks: &mut [NativeModelBlockApplicationV5],
        transition_masks: &mut [u64],
        captures: &mut [KreaCaptureResultV6],
        applications: &mut [KreaApplicationResultV6],
        result: &mut ProgramImageResultV6,
    ) -> i32 {
        clear_native_error_logs();
        debug_assert_eq!(params.abi_version, KREA_ACTIVATION_ABI_VERSION);
        // SAFETY: Forwarded from this method's caller contract.
        unsafe {
            (self.functions.program_generate_image_v6)(
                program,
                params,
                Some(condition_callback),
                condition_callback_data,
                Some(step_callback),
                step_callback_data,
                Some(activation_callback),
                activation_callback_data,
                snapshots.as_mut_ptr(),
                snapshots.len(),
                model_blocks.as_mut_ptr(),
                model_blocks.len(),
                transition_masks.as_mut_ptr(),
                transition_masks.len(),
                captures.as_mut_ptr(),
                captures.len(),
                applications.as_mut_ptr(),
                applications.len(),
                result,
            )
        }
    }

    /// Describes one live v3 arena value.
    ///
    /// # Safety
    ///
    /// `program` and `value` must identify one live program value.
    pub(crate) unsafe fn program_describe_v3(
        &self,
        program: *const c_void,
        value: ValueHandleV3,
        descriptor_out: &mut ValueDescriptorV3,
    ) -> i32 {
        clear_native_error_logs();
        // SAFETY: Forwarded from this method's caller contract.
        unsafe { (self.functions.program_describe_v3)(program, value, descriptor_out) }
    }

    /// Exposes one native-owned value synchronously to a bounded callback.
    ///
    /// # Safety
    ///
    /// The callback must not retain the native byte pointer.
    pub(crate) unsafe fn program_read_v3(
        &self,
        program: *const c_void,
        value: ValueHandleV3,
        callback: ValueReadCallback,
        callback_data: *mut c_void,
    ) -> i32 {
        clear_native_error_logs();
        // SAFETY: Forwarded from this method's caller contract.
        unsafe { (self.functions.program_read_v3)(program, value, Some(callback), callback_data) }
    }

    /// Copies one explicitly materialized v3 value.
    ///
    /// # Safety
    ///
    /// `output` must be writable and exactly bounded for the call.
    pub(crate) unsafe fn program_copy_v3(
        &self,
        program: *const c_void,
        value: ValueHandleV3,
        output: &mut [u8],
        bytes_written: &mut usize,
    ) -> i32 {
        clear_native_error_logs();
        // SAFETY: Forwarded from this method's caller contract.
        unsafe {
            (self.functions.program_copy_v3)(
                program,
                value,
                output.as_mut_ptr(),
                output.len(),
                bytes_written,
            )
        }
    }

    /// Releases one live v3 arena handle.
    ///
    /// # Safety
    ///
    /// The handle must belong to `program` and be released exactly once.
    pub(crate) unsafe fn program_release_v3(
        &self,
        program: *mut c_void,
        value: ValueHandleV3,
    ) -> i32 {
        clear_native_error_logs();
        // SAFETY: Forwarded from this method's caller contract.
        unsafe { (self.functions.program_release_v3)(program, value) }
    }

    /// Invalidates and frees one v3 arena.
    ///
    /// # Safety
    ///
    /// `program` must be the exclusively owned live arena and this must be its
    /// only finish call.
    pub(crate) unsafe fn program_finish_v3(
        &self,
        program: *mut c_void,
        clear_model_session: bool,
        peak_arena_bytes: &mut u64,
    ) -> i32 {
        clear_native_error_logs();
        // SAFETY: Forwarded from this method's caller contract.
        unsafe {
            (self.functions.program_finish_v3)(program, clear_model_session, peak_arena_bytes)
        }
    }

    /// Releases native image memory.
    ///
    /// # Safety
    ///
    /// `images` must be a live allocation returned by this exact library and
    /// `count` must be its exact image count.
    pub(crate) unsafe fn free_images(&self, images: *mut Image, count: i32) {
        // SAFETY: Forwarded from this method's caller contract.
        unsafe { (self.functions.free_images)(images, count) };
    }
}

impl Drop for NativeApi {
    fn drop(&mut self) {
        // SAFETY: This removes the process-global callback before unloading
        // the library that owns the callback registration.
        unsafe {
            (self.functions.set_log_callback)(None, ptr::null_mut());
        }
    }
}

fn parse_devices(report: &str) -> Result<Vec<String>> {
    let devices = report
        .lines()
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if devices.is_empty() {
        return Err(Error::Incompatible(
            "native runtime reports no devices".to_owned(),
        ));
    }
    if devices.len() > MAX_DEVICE_LINES {
        return Err(Error::Incompatible(format!(
            "native device report exceeds {MAX_DEVICE_LINES} lines"
        )));
    }
    for device in &devices {
        let Some((name, description)) = device.split_once('\t') else {
            return Err(Error::Incompatible(
                "native device report line has no tab separator".to_owned(),
            ));
        };
        if name.is_empty()
            || description.is_empty()
            || device.len() > MAX_DEVICE_LINE_BYTES
            || device.contains('\0')
        {
            return Err(Error::Incompatible(
                "native device report line is empty, oversized, or contains NUL".to_owned(),
            ));
        }
    }
    Ok(devices)
}

unsafe fn load_symbol<T: Copy>(library: &Library, symbol: &[u8]) -> Result<T> {
    // SAFETY: The caller supplies the exact companion symbol signature.
    let loaded = unsafe { library.get::<T>(symbol) }.map_err(|error| {
        Error::Incompatible(format!(
            "missing companion symbol {:?}: {error}",
            String::from_utf8_lossy(symbol.strip_suffix(&[0]).unwrap_or(symbol))
        ))
    })?;
    Ok(*loaded)
}

unsafe fn bounded_c_string(pointer: *const c_char, maximum: usize) -> Result<String> {
    if pointer.is_null() {
        return Err(Error::Incompatible(
            "native identity string is null".to_owned(),
        ));
    }
    // SAFETY: The matching companion ABI promises a static NUL-terminated
    // string. The post-read length bound rejects malformed identities.
    let value = unsafe { CStr::from_ptr(pointer) };
    let bytes = value.to_bytes();
    if bytes.is_empty() || bytes.len() > maximum {
        return Err(Error::Incompatible(
            "native identity string exceeds its bound".to_owned(),
        ));
    }
    value
        .to_str()
        .map(str::to_owned)
        .map_err(|_| Error::Incompatible("native identity string is not UTF-8".to_owned()))
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;

    use super::*;

    #[test]
    fn device_report_requires_bounded_named_lines() {
        assert_eq!(
            parse_devices("Vulkan0\tAMD GPU\n").expect("valid report"),
            ["Vulkan0\tAMD GPU"]
        );
        assert!(parse_devices("").is_err());
        assert!(parse_devices("missing separator").is_err());
        assert!(parse_devices("Vulkan0\t").is_err());
    }

    #[test]
    fn native_error_logs_are_retained_complete() {
        clear_native_error_logs();
        let message = "native failure detail ".repeat(8_192);
        let native = CString::new(message.clone()).expect("message contains no NUL");
        // SAFETY: `native` is a live NUL-terminated string for the synchronous
        // callback, and the callback retains an owned copy only.
        unsafe {
            native_log_callback(NATIVE_LOG_ERROR, native.as_ptr(), ptr::null_mut());
        }
        assert_eq!(take_native_error_logs(), [message]);
    }
}

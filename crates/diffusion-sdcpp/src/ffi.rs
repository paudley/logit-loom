// SPDX-License-Identifier: MIT OR Apache-2.0

//! Private dynamic companion ABI.

use std::{
    ffi::{CStr, c_char, c_void},
    path::Path,
    ptr,
};

use libloading::Library;

use crate::{COMPANION_ABI_VERSION, Error, IMAGE_ABI_VERSION, Result, UPSTREAM_COMMIT};

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
type FreeImages = unsafe extern "C" fn(*mut Image, i32);
type ListDevices = unsafe extern "C" fn(*mut c_char, usize) -> usize;

#[derive(Clone, Copy)]
struct Functions {
    new_context: NewContext,
    free_context: FreeContext,
    generate_image: GenerateImage,
    generate_image_v2: GenerateImageV2,
    vae_encode_v2: VaeEncodeV2,
    vae_decode_v2: VaeDecodeV2,
    free_tensor_v2: FreeTensorV2,
    clear_session_v2: ClearSessionV2,
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
                new_context: load_symbol(&library, b"sd_loom_new_ctx_v1\0")?,
                free_context: load_symbol(&library, b"free_sd_ctx\0")?,
                generate_image: load_symbol(&library, b"sd_loom_generate_image_v1\0")?,
                generate_image_v2: load_symbol(&library, b"sd_loom_generate_image_v2\0")?,
                vae_encode_v2: load_symbol(&library, b"sd_loom_vae_encode_v2\0")?,
                vae_decode_v2: load_symbol(&library, b"sd_loom_vae_decode_v2\0")?,
                free_tensor_v2: load_symbol(&library, b"sd_loom_free_tensor_v2\0")?,
                clear_session_v2: load_symbol(&library, b"sd_loom_clear_session_v2\0")?,
                free_images: load_symbol(&library, b"free_sd_images\0")?,
                list_devices: load_symbol(&library, b"sd_list_devices\0")?,
            }
        };
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
        // SAFETY: Forwarded from this method's caller contract.
        unsafe { (self.functions.clear_session_v2)(context) }
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
}

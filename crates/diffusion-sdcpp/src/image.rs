// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native image-ABI bindings that remain outside backend-neutral plans.

use std::path::{Path, PathBuf};

use logit_loom_diffusion::{Digest, ImageOperation};
use serde::{Deserialize, Serialize};

use crate::{
    ControlledGenerationReceipt, Error, GenerationMeasurements, ImageRequest, ImageRequestReceipt,
    ProfileReceipt, Result,
};

/// Maximum reference images in one native request.
pub const MAX_REFERENCE_IMAGES: usize = 16;
/// Maximum fixed-scale `LoRA` entries in one native request.
pub const MAX_REQUEST_LORAS: usize = 32;
/// Maximum rank accepted for an explicit VAE tensor.
pub const MAX_VAE_TENSOR_RANK: usize = 8;

/// Tightly packed borrowed image bytes.
#[derive(Clone, Copy, Debug)]
pub struct ImagePixels<'a> {
    bytes: &'a [u8],
    width: u32,
    height: u32,
    channels: u32,
}

impl<'a> ImagePixels<'a> {
    /// Binds tightly packed RGB8 bytes.
    ///
    /// # Errors
    ///
    /// Returns an error when geometry or byte length differs.
    pub fn rgb8(bytes: &'a [u8], width: u32, height: u32) -> Result<Self> {
        Self::new(bytes, width, height, 3)
    }

    /// Binds tightly packed RGBA8 bytes.
    ///
    /// # Errors
    ///
    /// Returns an error when geometry or byte length differs.
    pub fn rgba8(bytes: &'a [u8], width: u32, height: u32) -> Result<Self> {
        Self::new(bytes, width, height, 4)
    }

    /// Binds tightly packed Gray8 bytes.
    ///
    /// # Errors
    ///
    /// Returns an error when geometry or byte length differs.
    pub fn gray8(bytes: &'a [u8], width: u32, height: u32) -> Result<Self> {
        Self::new(bytes, width, height, 1)
    }

    fn new(bytes: &'a [u8], width: u32, height: u32, channels: u32) -> Result<Self> {
        let value = Self {
            bytes,
            width,
            height,
            channels,
        };
        value.validate_length()?;
        Ok(value)
    }

    /// Returns exact borrowed bytes.
    pub const fn bytes(self) -> &'a [u8] {
        self.bytes
    }

    /// Returns pixel width.
    pub const fn width(self) -> u32 {
        self.width
    }

    /// Returns pixel height.
    pub const fn height(self) -> u32 {
        self.height
    }

    /// Returns interleaved channel count.
    pub const fn channels(self) -> u32 {
        self.channels
    }

    pub(crate) fn validate_color(self) -> Result<()> {
        self.validate_length()?;
        if !matches!(self.channels, 3 | 4) {
            return Err(Error::Invalid(
                "color image must contain three or four channels".to_owned(),
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_mask(self) -> Result<()> {
        self.validate_length()?;
        if self.channels != 1 {
            return Err(Error::Invalid(
                "mask image must contain exactly one channel".to_owned(),
            ));
        }
        Ok(())
    }

    fn validate_length(self) -> Result<()> {
        if self.width == 0
            || self.height == 0
            || self.width > crate::MAX_IMAGE_DIMENSION
            || self.height > crate::MAX_IMAGE_DIMENSION
        {
            return Err(Error::Invalid(format!(
                "image pixel dimensions must be within 1..={}",
                crate::MAX_IMAGE_DIMENSION
            )));
        }
        let expected = usize::try_from(self.width)
            .ok()
            .and_then(|width| {
                usize::try_from(self.height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .and_then(|pixels| {
                usize::try_from(self.channels)
                    .ok()
                    .and_then(|channels| pixels.checked_mul(channels))
            })
            .ok_or_else(|| Error::Invalid("image pixel length overflowed".to_owned()))?;
        if self.bytes.len() != expected {
            return Err(Error::Invalid(format!(
                "image has {} bytes; expected {expected}",
                self.bytes.len()
            )));
        }
        Ok(())
    }

    fn receipt(self, domain: &'static str) -> PixelReceipt {
        PixelReceipt {
            bytes: Digest::of_bytes(domain, self.bytes),
            width: self.width,
            height: self.height,
            channels: self.channels,
        }
    }
}

/// One caller-verified `LoRA` path and fixed request-local scale.
#[derive(Clone, Debug)]
pub struct LoraBinding {
    path: PathBuf,
    identity: Digest,
    scale: f32,
    high_noise: bool,
}

impl LoraBinding {
    /// Creates one fixed-scale `LoRA` binding.
    ///
    /// `identity` must identify the exact caller-verified bytes at `path`.
    /// The adapter reopens no network source and records no path in receipts.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty path or a non-finite/out-of-range scale.
    pub fn new(path: impl Into<PathBuf>, identity: Digest, scale: f32) -> Result<Self> {
        let value = Self {
            path: path.into(),
            identity,
            scale,
            high_noise: false,
        };
        value.validate()?;
        Ok(value)
    }

    /// Marks this binding for an upstream high-noise model where supported.
    #[must_use]
    pub const fn with_high_noise(mut self, high_noise: bool) -> Self {
        self.high_noise = high_noise;
        self
    }

    /// Returns the caller-managed local path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the exact artifact identity.
    pub const fn identity(&self) -> &Digest {
        &self.identity
    }

    /// Returns the exact fixed scale.
    pub const fn scale(&self) -> f32 {
        self.scale
    }

    /// Returns the upstream high-noise selector.
    pub const fn is_high_noise(&self) -> bool {
        self.high_noise
    }

    fn validate(&self) -> Result<()> {
        if self.path.as_os_str().is_empty() {
            return Err(Error::Invalid("LoRA path must be non-empty".to_owned()));
        }
        if !self.scale.is_finite() || !(-64.0..=64.0).contains(&self.scale) {
            return Err(Error::Invalid(
                "LoRA scale must be finite and within -64..=64".to_owned(),
            ));
        }
        Ok(())
    }

    fn receipt(&self) -> LoraBindingReceipt {
        LoraBindingReceipt {
            identity: self.identity.clone(),
            scale_bits: self.scale.to_bits(),
            high_noise: self.high_noise,
        }
    }
}

/// One image ABI v2 generation request.
#[derive(Clone, Debug)]
pub struct AdvancedImageRequest<'a> {
    base: ImageRequest,
    operation: ImageOperation,
    negative_prompt: String,
    strength: f32,
    source: Option<ImagePixels<'a>>,
    mask: Option<ImagePixels<'a>>,
    references: Vec<ImagePixels<'a>>,
    loras: Vec<LoraBinding>,
}

impl<'a> AdvancedImageRequest<'a> {
    /// Creates a text-to-image request.
    ///
    /// # Errors
    ///
    /// Returns an error when the baseline request is invalid.
    pub fn text_to_image(base: ImageRequest) -> Result<Self> {
        let value = Self {
            base,
            operation: ImageOperation::TextToImage,
            negative_prompt: String::new(),
            strength: 1.0,
            source: None,
            mask: None,
            references: Vec::new(),
            loras: Vec::new(),
        };
        value.validate()?;
        Ok(value)
    }

    /// Selects image-to-image over an exact source canvas.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid source geometry or strength.
    pub fn image_to_image(mut self, source: ImagePixels<'a>, strength: f32) -> Result<Self> {
        self.operation = ImageOperation::ImageToImage;
        self.source = Some(source);
        self.mask = None;
        self.strength = strength;
        self.validate()?;
        Ok(self)
    }

    /// Selects masked inpaint over an exact source canvas.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid source/mask geometry or strength.
    pub fn inpaint(
        mut self,
        source: ImagePixels<'a>,
        mask: ImagePixels<'a>,
        strength: f32,
    ) -> Result<Self> {
        self.operation = ImageOperation::Inpaint;
        self.source = Some(source);
        self.mask = Some(mask);
        self.strength = strength;
        self.validate()?;
        Ok(self)
    }

    /// Selects outpaint over a caller-expanded source canvas and mask.
    ///
    /// The adapter performs no implicit resizing or canvas expansion.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid source/mask geometry or strength.
    pub fn outpaint(
        mut self,
        expanded_source: ImagePixels<'a>,
        mask: ImagePixels<'a>,
        strength: f32,
    ) -> Result<Self> {
        self.operation = ImageOperation::Outpaint;
        self.source = Some(expanded_source);
        self.mask = Some(mask);
        self.strength = strength;
        self.validate()?;
        Ok(self)
    }

    /// Adds exact negative-conditioning UTF-8.
    ///
    /// # Errors
    ///
    /// Returns an error for an oversized or NUL-containing value.
    pub fn with_negative_prompt(mut self, prompt: impl Into<String>) -> Result<Self> {
        self.negative_prompt = prompt.into();
        self.validate()?;
        Ok(self)
    }

    /// Adds one exact reference image in declared order.
    ///
    /// # Errors
    ///
    /// Returns an error when the reference bound is exceeded or bytes differ
    /// from their geometry.
    pub fn with_reference(mut self, reference: ImagePixels<'a>) -> Result<Self> {
        self.references.push(reference);
        self.validate()?;
        Ok(self)
    }

    /// Replaces the exact ordered request-local `LoRA` stack.
    ///
    /// Every binding uses one fixed scale for the complete request. Scheduled
    /// scales and model-specific target selectors are intentionally rejected
    /// by this ABI version.
    ///
    /// # Errors
    ///
    /// Returns an error for an oversized or invalid stack.
    pub fn with_loras(mut self, loras: Vec<LoraBinding>) -> Result<Self> {
        self.loras = loras;
        self.validate()?;
        Ok(self)
    }

    /// Returns the baseline schedule, prompt, dimensions, seed, and guidance.
    pub const fn base(&self) -> &ImageRequest {
        &self.base
    }

    /// Returns the exact operation.
    pub const fn operation(&self) -> ImageOperation {
        self.operation
    }

    /// Returns exact negative-conditioning UTF-8.
    pub fn negative_prompt(&self) -> &str {
        &self.negative_prompt
    }

    /// Returns image-to-image/inpaint/outpaint strength.
    pub const fn strength(&self) -> f32 {
        self.strength
    }

    /// Returns the exact source canvas, if any.
    pub const fn source(&self) -> Option<ImagePixels<'a>> {
        self.source
    }

    /// Returns the exact mask, if any.
    pub const fn mask(&self) -> Option<ImagePixels<'a>> {
        self.mask
    }

    /// Returns reference images in declared order.
    pub fn references(&self) -> &[ImagePixels<'a>] {
        &self.references
    }

    /// Returns request-local `LoRA` entries in declared order.
    pub fn loras(&self) -> &[LoraBinding] {
        &self.loras
    }

    /// Validates operation requirements and all public bounds.
    ///
    /// # Errors
    ///
    /// Returns the first invalid field.
    pub fn validate(&self) -> Result<()> {
        self.base.validate_common()?;
        if self.negative_prompt.len() > crate::MAX_PROMPT_BYTES
            || self.negative_prompt.contains('\0')
        {
            return Err(Error::Invalid(format!(
                "negative prompt must contain at most {} bytes without NUL",
                crate::MAX_PROMPT_BYTES
            )));
        }
        if !self.strength.is_finite() || !(0.0..=1.0).contains(&self.strength) {
            return Err(Error::Invalid(
                "image strength must be finite and within 0..=1".to_owned(),
            ));
        }
        if self.references.len() > MAX_REFERENCE_IMAGES || self.loras.len() > MAX_REQUEST_LORAS {
            return Err(Error::Invalid(
                "image reference or LoRA collection exceeds the ABI bound".to_owned(),
            ));
        }
        let dimensions_match = |pixels: ImagePixels<'_>| {
            pixels.width == self.base.width() && pixels.height == self.base.height()
        };
        for reference in &self.references {
            reference.validate_color()?;
        }
        for lora in &self.loras {
            lora.validate()?;
        }
        match (self.operation, self.source, self.mask) {
            (ImageOperation::TextToImage, None, None)
                if self.strength.to_bits() == 1.0_f32.to_bits() =>
            {
                Ok(())
            }
            (ImageOperation::ImageToImage, Some(source), None)
                if dimensions_match(source) && self.strength > 0.0 =>
            {
                source.validate_color()
            }
            (ImageOperation::Inpaint | ImageOperation::Outpaint, Some(source), Some(mask))
                if dimensions_match(source) && dimensions_match(mask) && self.strength > 0.0 =>
            {
                source.validate_color()?;
                mask.validate_mask()
            }
            _ => Err(Error::Invalid(
                "operation, source, mask, strength, or canvas geometry is inconsistent".to_owned(),
            )),
        }
    }

    /// Returns a path-free identity of every exact request input.
    ///
    /// # Errors
    ///
    /// Returns a validation or serialization error.
    pub fn receipt(&self) -> Result<AdvancedImageRequestReceipt> {
        self.validate()?;
        Ok(AdvancedImageRequestReceipt {
            base: self.base.receipt()?,
            operation: self.operation,
            negative_prompt: Digest::of_bytes(
                "sdcpp-negative-prompt-bytes-v1",
                self.negative_prompt.as_bytes(),
            ),
            strength_bits: self.strength.to_bits(),
            source: self
                .source
                .map(|pixels| pixels.receipt("sdcpp-source-image-u8-v1")),
            mask: self
                .mask
                .map(|pixels| pixels.receipt("sdcpp-mask-image-u8-v1")),
            references: self
                .references
                .iter()
                .copied()
                .map(|pixels| pixels.receipt("sdcpp-reference-image-u8-v1"))
                .collect(),
            loras: self.loras.iter().map(LoraBinding::receipt).collect(),
        })
    }

    pub(crate) fn validate_for(&self, profile: crate::Profile) -> Result<()> {
        self.validate()?;
        self.base.validate_for(profile)
    }
}

/// Path-free identity of tightly packed image bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PixelReceipt {
    /// Exact pixel-byte identity.
    pub bytes: Digest,
    /// Pixel width.
    pub width: u32,
    /// Pixel height.
    pub height: u32,
    /// Interleaved channel count.
    pub channels: u32,
}

/// Path-free identity of one request-local `LoRA`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoraBindingReceipt {
    /// Exact caller-verified artifact identity.
    pub identity: Digest,
    /// Exact IEEE-754 fixed scale bits.
    pub scale_bits: u32,
    /// Upstream high-noise selector.
    pub high_noise: bool,
}

/// Path-free identity of one image ABI v2 request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdvancedImageRequestReceipt {
    /// Baseline prompt/schedule request.
    pub base: ImageRequestReceipt,
    /// Exact whole-image operation.
    pub operation: ImageOperation,
    /// Exact negative-conditioning byte identity.
    pub negative_prompt: Digest,
    /// Exact IEEE-754 source strength bits.
    pub strength_bits: u32,
    /// Source-canvas identity.
    pub source: Option<PixelReceipt>,
    /// Mask identity.
    pub mask: Option<PixelReceipt>,
    /// Reference identities in declared order.
    pub references: Vec<PixelReceipt>,
    /// Request-local `LoRA` entries in declared order.
    pub loras: Vec<LoraBindingReceipt>,
}

impl AdvancedImageRequestReceipt {
    /// Returns the complete advanced-request identity.
    ///
    /// # Errors
    ///
    /// Returns a serialization error.
    pub fn digest(&self) -> Result<Digest> {
        Digest::of_serializable("sdcpp-image-request-v2", self)
            .map_err(logit_loom_diffusion::Error::from)
            .map_err(Error::from)
    }
}

/// Mechanical receipt for one image ABI v2 generation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdvancedGenerationReceipt {
    /// Exact source/mask/reference/LoRA request identity.
    pub request: AdvancedImageRequestReceipt,
    /// Exact conditioning, schedule, boundaries, backend, and image identity.
    pub generation: ControlledGenerationReceipt,
}

/// Image ABI v2 generation written to caller-owned storage.
#[derive(Clone, Debug)]
pub struct AdvancedGenerationOutput {
    /// Number of initialized destination bytes.
    pub bytes_written: usize,
    /// Complete mechanical receipt.
    pub receipt: AdvancedGenerationReceipt,
    /// Non-deterministic deployment measurements excluded from identities.
    pub measurements: GenerationMeasurements,
}

/// Mechanical receipt for one image ABI v2 generation with full scheduler
/// state accounting.
///
/// This is a distinct contract from [`AdvancedGenerationReceipt`]. The
/// control-only receipt deliberately omits scheduler-state identities, while
/// this receipt records the transactional [`crate::StepProgram`] lineage in
/// its nested generation receipt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdvancedProgramGenerationReceipt {
    /// Exact source/mask/reference/LoRA request identity.
    pub request: AdvancedImageRequestReceipt,
    /// Exact conditioning, scheduler-state, program, backend, and image
    /// lineage.
    pub generation: crate::GenerationReceipt,
}

/// Image ABI v2 generation with a full scheduler-state program, written to
/// caller-owned storage.
#[derive(Clone, Debug)]
pub struct AdvancedProgramGenerationOutput {
    /// Number of initialized destination bytes.
    pub bytes_written: usize,
    /// Complete mechanical receipt.
    pub receipt: AdvancedProgramGenerationReceipt,
    /// Non-deterministic deployment measurements excluded from identities.
    pub measurements: GenerationMeasurements,
}

/// Exact finite host tensor produced or consumed by direct VAE operations.
#[derive(Clone, Debug)]
pub struct VaeTensor {
    values: Vec<f32>,
    shape: Vec<i64>,
}

impl VaeTensor {
    /// Reconstructs one exact finite VAE tensor.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid rank, dimensions, element count, or
    /// non-finite values.
    pub fn from_parts(values: Vec<f32>, shape: Vec<i64>) -> Result<Self> {
        let value = Self { values, shape };
        value.validate()?;
        Ok(value)
    }

    /// Returns exact native-layout values.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Returns exact native-layout dimensions.
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    /// Returns little-endian `f32` bytes.
    pub fn to_le_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.values.len().saturating_mul(4));
        for value in &self.values {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        bytes
    }

    /// Returns the exact tensor-byte identity.
    pub fn digest(&self) -> Digest {
        let bytes = self.to_le_bytes();
        Digest::of_bytes("sdcpp-vae-tensor-f32-le-v1", &bytes)
    }

    fn validate(&self) -> Result<()> {
        if self.shape.is_empty()
            || self.shape.len() > MAX_VAE_TENSOR_RANK
            || self.shape.iter().any(|dimension| *dimension <= 0)
            || self.values.is_empty()
            || u64::try_from(self.values.len()).map_or(true, |elements| {
                elements > logit_loom_diffusion::MAX_TENSOR_ELEMENTS
            })
            || self.values.iter().any(|value| !value.is_finite())
        {
            return Err(Error::Invalid(
                "VAE tensor must have bounded positive shape, element count, and finite values"
                    .to_owned(),
            ));
        }
        let expected = self.shape.iter().try_fold(1_usize, |elements, dimension| {
            usize::try_from(*dimension)
                .ok()
                .and_then(|dimension| elements.checked_mul(dimension))
        });
        if expected != Some(self.values.len()) {
            return Err(Error::Invalid(
                "VAE tensor shape does not match its element count".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Mechanical direct-VAE operation receipt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VaeOperationReceipt {
    /// Exact verified model profile.
    pub profile: ProfileReceipt,
    /// Exact backend build/placement identity.
    pub backend: Digest,
    /// Session epoch used by the operation.
    pub session_epoch: u64,
    /// Exact input bytes.
    pub input: Digest,
    /// Exact output bytes.
    pub output: Digest,
    /// Native tensor shape for encode/decode.
    pub tensor_shape: Vec<i64>,
    /// Output image width, or zero for encode.
    pub width: u32,
    /// Output image height, or zero for encode.
    pub height: u32,
    /// Output image channels, or zero for encode.
    pub channels: u32,
}

/// Direct VAE encode result.
#[derive(Clone, Debug)]
pub struct VaeTensorOutput {
    /// Exact finite native-layout tensor.
    pub tensor: VaeTensor,
    /// Mechanical lineage.
    pub receipt: VaeOperationReceipt,
}

/// Direct VAE decode result.
#[derive(Clone, Debug)]
pub struct VaeImageOutput {
    /// Tightly packed output bytes.
    pub bytes: Vec<u8>,
    /// Mechanical lineage.
    pub receipt: VaeOperationReceipt,
}

#[cfg(test)]
mod tests {
    use super::*;
    use logit_loom_diffusion::DiffusionSchedule;

    fn request() -> ImageRequest {
        ImageRequest::new(
            "mechanical fixture",
            64,
            64,
            7,
            1.0,
            DiffusionSchedule::new(Digest::of_bytes("test-schedule", b"v2"), vec![1.0, 0.0])
                .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn advanced_operations_require_exact_tight_canvases() {
        let source_bytes = vec![0_u8; 64 * 64 * 3];
        let mask_bytes = vec![255_u8; 64 * 64];
        let source = ImagePixels::rgb8(&source_bytes, 64, 64).unwrap();
        let mask = ImagePixels::gray8(&mask_bytes, 64, 64).unwrap();
        let value = AdvancedImageRequest::text_to_image(request())
            .unwrap()
            .outpaint(source, mask, 0.75)
            .unwrap();
        assert_eq!(value.operation(), ImageOperation::Outpaint);
        assert_eq!(value.receipt().unwrap().references.len(), 0);
    }

    #[test]
    fn request_receipt_omits_lora_path() {
        let lora = LoraBinding::new(
            "/private/model-store/adapter.safetensors",
            Digest::of_bytes("test-lora", b"exact"),
            0.5,
        )
        .unwrap();
        let receipt = AdvancedImageRequest::text_to_image(request())
            .unwrap()
            .with_loras(vec![lora])
            .unwrap()
            .receipt()
            .unwrap();
        let encoded = serde_json::to_string(&receipt).unwrap();
        assert!(!encoded.contains("private"));
        assert!(!encoded.contains("safetensors"));
    }

    #[test]
    fn vae_tensor_rejects_non_finite_or_wrong_shape() {
        assert!(VaeTensor::from_parts(vec![0.0, f32::NAN], vec![2]).is_err());
        assert!(VaeTensor::from_parts(vec![0.0], vec![2]).is_err());
        assert!(VaeTensor::from_parts(vec![0.0, 1.0], vec![2]).is_ok());
    }
}

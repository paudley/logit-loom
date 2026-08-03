// SPDX-License-Identifier: MIT OR Apache-2.0

#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod acquisition;

use std::{
    collections::HashSet,
    fs::File,
    io::{self, Read as _},
    path::{Component, Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use thiserror::Error;

pub use acquisition::{
    AcquisitionMethod, AcquisitionReport, AcquisitionReportError, AcquisitionStatus,
    PendingArtifact, PendingReason, ProfileAcquisition,
};

/// Current serialized catalog schema.
pub const CATALOG_SCHEMA_VERSION: u32 = 1;
/// Identity domain for the interpretation of the embedded catalog.
pub const CATALOG_IDENTITY_DOMAIN: &str = "logit-loom-model-catalog-v1";
/// Stable ID of the small default Qwen text profile.
pub const QWEN3_SMALL_PROFILE_ID: &str = "qwen3-0.6b-q8-0";
/// Source ID containing the small Qwen GGUF.
pub const QWEN3_SMALL_SOURCE_ID: &str = "qwen3-gguf";
/// Source-relative path of the small Qwen `Q8_0` GGUF.
pub const QWEN3_SMALL_ARTIFACT_PATH: &str = "Qwen3-0.6B-Q8_0.gguf";

const CATALOG_BYTES: &[u8] = include_bytes!("../profiles.json");
const DEFAULT_PROFILE_COUNT: usize = 3;
const MAX_PROFILES: usize = 32;
const MAX_SOURCES_PER_PROFILE: usize = 8;
const MAX_FILES_PER_SOURCE: usize = 64;
const MAX_TEXT_BYTES: usize = 2_048;
const MAX_PATH_BYTES: usize = 512;
const MAX_FILE_BYTES: u64 = 256 * 1024 * 1024 * 1024;
const MAX_PROFILE_BYTES: u64 = 512 * 1024 * 1024 * 1024;
const HASH_BUFFER_BYTES: usize = 1024 * 1024;
const WEIGHT_SUFFIXES: &[&str] = &[
    ".gguf",
    ".safetensors",
    ".onnx",
    ".pt",
    ".pth",
    ".bin",
    ".ckpt",
];

/// One validated model acquisition catalog.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Catalog {
    schema_version: u32,
    identity_domain: String,
    default_profiles: Vec<String>,
    profiles: Vec<Profile>,
}

impl Catalog {
    /// Parses and validates the catalog embedded in this crate.
    ///
    /// # Errors
    ///
    /// Returns a parse or validation error if the packaged catalog is invalid.
    pub fn embedded() -> Result<Self, CatalogError> {
        Self::from_slice(CATALOG_BYTES)
    }

    /// Parses and validates a bounded catalog value.
    ///
    /// # Errors
    ///
    /// Returns a parse or validation error if the input is malformed.
    pub fn from_slice(bytes: &[u8]) -> Result<Self, CatalogError> {
        if bytes.len() > MAX_TEXT_BYTES * MAX_FILES_PER_SOURCE * MAX_PROFILES {
            return Err(CatalogError::Invalid(
                "serialized catalog exceeds the input bound".to_owned(),
            ));
        }
        let catalog = serde_json::from_slice::<Self>(bytes).map_err(CatalogError::Parse)?;
        catalog.validate()?;
        Ok(catalog)
    }

    /// Returns the catalog schema version.
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Returns the interpretation domain for catalog identities.
    pub fn identity_domain(&self) -> &str {
        &self.identity_domain
    }

    /// Returns the SHA-256 identity of the exact packaged JSON bytes.
    pub fn packaged_sha256(&self) -> String {
        sha256_bytes(CATALOG_BYTES)
    }

    /// Returns the profile IDs selected as default experiment profiles.
    pub fn default_profiles(&self) -> &[String] {
        &self.default_profiles
    }

    /// Returns every catalog profile in declared order.
    pub fn profiles(&self) -> &[Profile] {
        &self.profiles
    }

    /// Finds one exact profile ID.
    pub fn find_profile(&self, id: &str) -> Option<&Profile> {
        self.profiles.iter().find(|profile| profile.id == id)
    }

    /// Validates catalog bounds, identities, and cross-field relationships.
    ///
    /// # Errors
    ///
    /// Returns the first invalid catalog invariant.
    pub fn validate(&self) -> Result<(), CatalogError> {
        if self.schema_version != CATALOG_SCHEMA_VERSION {
            return invalid(format!(
                "schema_version must be {CATALOG_SCHEMA_VERSION}, got {}",
                self.schema_version
            ));
        }
        if self.identity_domain != CATALOG_IDENTITY_DOMAIN {
            return invalid(format!(
                "identity_domain must be {CATALOG_IDENTITY_DOMAIN:?}"
            ));
        }
        if self.profiles.is_empty() || self.profiles.len() > MAX_PROFILES {
            return invalid(format!(
                "profiles must contain between 1 and {MAX_PROFILES} entries"
            ));
        }
        if self.default_profiles.len() != DEFAULT_PROFILE_COUNT {
            return invalid(format!(
                "default_profiles must contain exactly {DEFAULT_PROFILE_COUNT} entries"
            ));
        }

        let mut profile_ids = HashSet::new();
        for profile in &self.profiles {
            if !profile_ids.insert(profile.id.as_str()) {
                return invalid(format!("duplicate profile id {:?}", profile.id));
            }
            validate_profile(profile)?;
        }

        let mut default_ids = HashSet::new();
        for id in &self.default_profiles {
            if !default_ids.insert(id.as_str()) {
                return invalid(format!("duplicate default profile id {id:?}"));
            }
            if !profile_ids.contains(id.as_str()) {
                return invalid(format!("default profile {id:?} does not exist"));
            }
        }
        Ok(())
    }
}

/// One optional model and its exact acquisition contract.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    id: String,
    display_name: String,
    modality: Modality,
    role: Role,
    integration_status: IntegrationStatus,
    acceptance_status: AcceptanceStatus,
    adapter_target: AdapterTarget,
    requires_accelerator: bool,
    remote_code_policy: RemoteCodePolicy,
    notes: String,
    sources: Vec<Source>,
}

impl Profile {
    /// Returns the stable profile ID.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the human-readable profile name.
    pub fn display_name(&self) -> &str {
        &self.display_name
    }

    /// Returns the profile modality.
    pub const fn modality(&self) -> Modality {
        self.modality
    }

    /// Returns the intended catalog role.
    pub const fn role(&self) -> Role {
        self.role
    }

    /// Returns the checked-in integration status.
    pub const fn integration_status(&self) -> IntegrationStatus {
        self.integration_status
    }

    /// Returns the checked-in model-backed acceptance status.
    pub const fn acceptance_status(&self) -> AcceptanceStatus {
        self.acceptance_status
    }

    /// Returns the maintained adapter family.
    pub const fn adapter_target(&self) -> AdapterTarget {
        self.adapter_target
    }

    /// Returns whether model-backed execution requires accelerator placement.
    pub const fn requires_accelerator(&self) -> bool {
        self.requires_accelerator
    }

    /// Returns the policy for executable code supplied by a model repository.
    pub const fn remote_code_policy(&self) -> RemoteCodePolicy {
        self.remote_code_policy
    }

    /// Returns the profile's mechanics-only description.
    pub fn notes(&self) -> &str {
        &self.notes
    }

    /// Returns exact acquisition sources in declared order.
    pub fn sources(&self) -> &[Source] {
        &self.sources
    }

    /// Returns the sum of catalogued file sizes.
    pub fn total_bytes(&self) -> u64 {
        self.sources
            .iter()
            .flat_map(|source| &source.files)
            .fold(0_u64, |total, file| total.saturating_add(file.bytes))
    }

    /// Returns the number of exact files in this profile.
    pub fn file_count(&self) -> usize {
        self.sources.iter().map(|source| source.files.len()).sum()
    }

    /// Finds an exact source and artifact path.
    pub fn find_artifact(&self, source_id: &str, path: &str) -> Option<(&Source, &ArtifactFile)> {
        self.sources
            .iter()
            .find(|source| source.id == source_id)
            .and_then(|source| {
                source
                    .files
                    .iter()
                    .find(|artifact| artifact.path == path)
                    .map(|artifact| (source, artifact))
            })
    }

    /// Verifies one caller-supplied local file against an exact catalog entry.
    ///
    /// The returned receipt deliberately excludes the local filesystem path.
    ///
    /// # Errors
    ///
    /// Returns an unknown-artifact, I/O, size, or digest error.
    pub fn verify_artifact(
        &self,
        catalog_sha256: &str,
        source_id: &str,
        artifact_path: &str,
        local_path: impl AsRef<Path>,
    ) -> Result<VerifiedArtifact, ArtifactError> {
        let (source, artifact) = self
            .find_artifact(source_id, artifact_path)
            .ok_or_else(|| ArtifactError::Unknown {
                profile: self.id.clone(),
                source_id: source_id.to_owned(),
                path: artifact_path.to_owned(),
            })?;
        let local_path = local_path.as_ref();
        let actual_bytes = std::fs::metadata(local_path)
            .map_err(|source| ArtifactError::Metadata {
                path: local_path.to_owned(),
                source,
            })?
            .len();
        if actual_bytes != artifact.bytes {
            return Err(ArtifactError::Size {
                path: local_path.to_owned(),
                actual: actual_bytes,
                expected: artifact.bytes,
            });
        }
        let actual_sha256 = match &artifact.sha256 {
            Some(expected) => {
                let actual = sha256_file(local_path)?;
                if &actual != expected {
                    return Err(ArtifactError::Digest {
                        path: local_path.to_owned(),
                        actual,
                        expected: expected.clone(),
                    });
                }
                Some(actual)
            }
            None => None,
        };
        Ok(VerifiedArtifact {
            local_path: local_path.to_owned(),
            receipt: ArtifactReceipt {
                catalog_domain: CATALOG_IDENTITY_DOMAIN.to_owned(),
                catalog_sha256: catalog_sha256.to_owned(),
                profile_id: self.id.clone(),
                source_id: source.id.clone(),
                repository: source.repository.clone(),
                revision: source.revision.clone(),
                artifact_path: artifact.path.clone(),
                bytes: artifact.bytes,
                sha256: actual_sha256,
            },
        })
    }
}

/// Model modality.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Modality {
    /// Autoregressive text generation.
    Text,
    /// Iterative image generation.
    Image,
}

impl Modality {
    /// Returns the serialized spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Image => "image",
        }
    }
}

/// Intended profile complexity.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Role {
    /// Small profile intended for short experiments.
    Toy,
    /// Larger profile exercising advanced mechanics.
    Advanced,
}

impl Role {
    /// Returns the serialized spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Toy => "toy",
            Self::Advanced => "advanced",
        }
    }
}

/// Maintained adapter family.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdapterTarget {
    /// The llama.cpp text adapter.
    Llamacpp,
    /// The diffusion image adapter.
    Diffusion,
}

/// Checked-in integration status.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum IntegrationStatus {
    /// Exact acquisition metadata exists; maintained execution is not accepted.
    Catalogued,
    /// Maintained adapter, runbook, and acceptance evidence exist.
    FirstClass,
}

impl IntegrationStatus {
    /// Returns the serialized spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Catalogued => "catalogued",
            Self::FirstClass => "first-class",
        }
    }
}

/// Checked-in model-backed acceptance status.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AcceptanceStatus {
    /// No retained acceptance report exists for this revision.
    NotRun,
    /// A retained mechanics report passed its schema checks.
    Passed,
}

impl AcceptanceStatus {
    /// Returns the serialized spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NotRun => "not-run",
            Self::Passed => "passed",
        }
    }
}

/// Policy for executable code in remote model repositories.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RemoteCodePolicy {
    /// Remote repository code must not execute.
    Forbidden,
}

/// One immutable repository source.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    id: String,
    repository: String,
    revision: String,
    local_subdir: String,
    files: Vec<ArtifactFile>,
}

impl Source {
    /// Returns the source ID within its profile.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the Hugging Face repository name.
    pub fn repository(&self) -> &str {
        &self.repository
    }

    /// Returns the exact immutable source revision.
    pub fn revision(&self) -> &str {
        &self.revision
    }

    /// Returns the profile-relative destination directory.
    pub fn local_subdir(&self) -> &str {
        &self.local_subdir
    }

    /// Returns exact files in declared order.
    pub fn files(&self) -> &[ArtifactFile] {
        &self.files
    }
}

/// One exact file in an acquisition source.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactFile {
    path: String,
    bytes: u64,
    sha256: Option<String>,
}

impl ArtifactFile {
    /// Returns the source-relative path.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Returns the exact expected byte count.
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Returns the expected SHA-256 digest, when recorded.
    pub fn sha256(&self) -> Option<&str> {
        self.sha256.as_deref()
    }
}

/// Serializable evidence that one exact local artifact was verified.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactReceipt {
    /// Catalog interpretation domain.
    pub catalog_domain: String,
    /// SHA-256 of the exact packaged catalog JSON.
    pub catalog_sha256: String,
    /// Stable profile ID.
    pub profile_id: String,
    /// Stable source ID within the profile.
    pub source_id: String,
    /// Upstream repository.
    pub repository: String,
    /// Exact immutable upstream revision.
    pub revision: String,
    /// Source-relative artifact path.
    pub artifact_path: String,
    /// Exact artifact size.
    pub bytes: u64,
    /// Verified SHA-256 digest, when the catalog declares one.
    pub sha256: Option<String>,
}

/// A verified artifact and its caller-local path.
#[derive(Clone, Debug)]
pub struct VerifiedArtifact {
    local_path: PathBuf,
    receipt: ArtifactReceipt,
}

impl VerifiedArtifact {
    /// Returns the verified caller-local path.
    pub fn path(&self) -> &Path {
        &self.local_path
    }

    /// Returns the path-free verification receipt.
    pub const fn receipt(&self) -> &ArtifactReceipt {
        &self.receipt
    }
}

/// Catalog parsing or invariant failure.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CatalogError {
    /// JSON parsing failed.
    #[error("failed to parse model catalog: {0}")]
    Parse(serde_json::Error),
    /// A bounded catalog invariant failed.
    #[error("invalid model catalog: {0}")]
    Invalid(String),
}

/// Local artifact verification failure.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ArtifactError {
    /// The profile does not declare the requested source and artifact.
    #[error("profile {profile:?} has no artifact {source_id:?}/{path:?}")]
    Unknown {
        /// Profile ID.
        profile: String,
        /// Source ID.
        source_id: String,
        /// Source-relative file path.
        path: String,
    },
    /// File metadata could not be read.
    #[error("failed to inspect artifact {path}: {source}")]
    Metadata {
        /// Caller-local file path.
        path: PathBuf,
        /// Underlying I/O failure.
        source: io::Error,
    },
    /// File size differs from the catalog.
    #[error("artifact {path} has {actual} bytes; expected {expected}")]
    Size {
        /// Caller-local file path.
        path: PathBuf,
        /// Observed bytes.
        actual: u64,
        /// Catalogued bytes.
        expected: u64,
    },
    /// File contents could not be read.
    #[error("failed to read artifact {path}: {source}")]
    Read {
        /// Caller-local file path.
        path: PathBuf,
        /// Underlying I/O failure.
        source: io::Error,
    },
    /// File digest differs from the catalog.
    #[error("artifact {path} has SHA-256 {actual}; expected {expected}")]
    Digest {
        /// Caller-local file path.
        path: PathBuf,
        /// Observed lowercase SHA-256.
        actual: String,
        /// Catalogued lowercase SHA-256.
        expected: String,
    },
}

fn validate_profile(profile: &Profile) -> Result<(), CatalogError> {
    validate_slug("profile id", &profile.id)?;
    validate_text("display_name", &profile.display_name)?;
    validate_text("notes", &profile.notes)?;
    if profile.integration_status == IntegrationStatus::FirstClass
        && profile.acceptance_status != AcceptanceStatus::Passed
    {
        return invalid(format!(
            "first-class profile {:?} must have passed acceptance status",
            profile.id
        ));
    }
    match (profile.modality, profile.adapter_target) {
        (Modality::Text, AdapterTarget::Llamacpp) | (Modality::Image, AdapterTarget::Diffusion) => {
        }
        _ => {
            return invalid(format!(
                "profile {:?} has an adapter target inconsistent with its modality",
                profile.id
            ));
        }
    }
    if profile.sources.is_empty() || profile.sources.len() > MAX_SOURCES_PER_PROFILE {
        return invalid(format!(
            "profile {:?} must contain between 1 and {MAX_SOURCES_PER_PROFILE} sources",
            profile.id
        ));
    }

    let mut source_ids = HashSet::new();
    let mut local_subdirs = HashSet::new();
    let mut total_bytes = 0_u64;
    for source in &profile.sources {
        if !source_ids.insert(source.id.as_str()) {
            return invalid(format!(
                "profile {:?} has duplicate source id {:?}",
                profile.id, source.id
            ));
        }
        if !local_subdirs.insert(source.local_subdir.as_str()) {
            return invalid(format!(
                "profile {:?} has duplicate source local_subdir {:?}",
                profile.id, source.local_subdir
            ));
        }
        total_bytes = total_bytes
            .checked_add(validate_source(&profile.id, source)?)
            .ok_or_else(|| {
                CatalogError::Invalid(format!("profile {:?} size overflow", profile.id))
            })?;
    }
    if total_bytes > MAX_PROFILE_BYTES {
        return invalid(format!(
            "profile {:?} exceeds the {MAX_PROFILE_BYTES}-byte bound",
            profile.id
        ));
    }
    Ok(())
}

fn validate_source(profile_id: &str, source: &Source) -> Result<u64, CatalogError> {
    validate_slug("source id", &source.id)?;
    validate_repository(&source.repository)?;
    validate_revision(&source.revision)?;
    validate_slug("local_subdir", &source.local_subdir)?;
    if source.files.is_empty() || source.files.len() > MAX_FILES_PER_SOURCE {
        return invalid(format!(
            "source {:?} in profile {profile_id:?} must contain between 1 and \
             {MAX_FILES_PER_SOURCE} files",
            source.id
        ));
    }

    let mut file_paths = HashSet::new();
    let mut total_bytes = 0_u64;
    for file in &source.files {
        validate_relative_path(&file.path)?;
        if !file_paths.insert(file.path.as_str()) {
            return invalid(format!(
                "source {:?} in profile {profile_id:?} has duplicate file path {:?}",
                source.id, file.path
            ));
        }
        if file.bytes == 0 || file.bytes > MAX_FILE_BYTES {
            return invalid(format!(
                "file {:?} in source {:?} has an invalid byte count",
                file.path, source.id
            ));
        }
        total_bytes = total_bytes.checked_add(file.bytes).ok_or_else(|| {
            CatalogError::Invalid(format!("source {:?} size overflow", source.id))
        })?;
        if let Some(digest) = &file.sha256 {
            validate_sha256(&file.path, digest)?;
        } else if is_weight_file(&file.path) {
            return invalid(format!(
                "weight file {:?} in source {:?} requires a SHA-256 digest",
                file.path, source.id
            ));
        }
    }
    Ok(total_bytes)
}

fn validate_slug(label: &str, value: &str) -> Result<(), CatalogError> {
    if value.is_empty()
        || value.len() > 64
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'.')
        })
        || !value
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphanumeric)
        || !value
            .as_bytes()
            .last()
            .is_some_and(u8::is_ascii_alphanumeric)
    {
        return invalid(format!("{label} {value:?} is not a bounded lowercase slug"));
    }
    Ok(())
}

fn validate_text(label: &str, value: &str) -> Result<(), CatalogError> {
    if value.is_empty() || value.len() > MAX_TEXT_BYTES || value.contains('\0') {
        return invalid(format!("{label} must be nonempty and bounded"));
    }
    Ok(())
}

fn validate_repository(value: &str) -> Result<(), CatalogError> {
    if value.len() > 160 {
        return invalid(format!("repository {value:?} is too long"));
    }
    let Some((owner, name)) = value.split_once('/') else {
        return invalid(format!("repository {value:?} must contain one slash"));
    };
    if name.contains('/')
        || owner.is_empty()
        || name.is_empty()
        || !owner.bytes().all(is_repository_byte)
        || !name.bytes().all(is_repository_byte)
    {
        return invalid(format!("repository {value:?} has an invalid shape"));
    }
    Ok(())
}

const fn is_repository_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.')
}

fn validate_revision(value: &str) -> Result<(), CatalogError> {
    if value.len() != 40
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return invalid(format!(
            "revision {value:?} must be a 40-character lowercase commit hash"
        ));
    }
    Ok(())
}

fn validate_relative_path(value: &str) -> Result<(), CatalogError> {
    if value.is_empty()
        || value.len() > MAX_PATH_BYTES
        || value.contains('\\')
        || Path::new(value)
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return invalid(format!(
            "artifact path {value:?} must be a bounded relative path"
        ));
    }
    Ok(())
}

fn validate_sha256(path: &str, value: &str) -> Result<(), CatalogError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return invalid(format!("file {path:?} has an invalid SHA-256 digest"));
    }
    Ok(())
}

fn is_weight_file(path: &str) -> bool {
    let path = path.to_ascii_lowercase();
    WEIGHT_SUFFIXES.iter().any(|suffix| path.ends_with(suffix))
}

fn sha256_file(path: &Path) -> Result<String, ArtifactError> {
    let mut file = File::open(path).map_err(|source| ArtifactError::Read {
        path: path.to_owned(),
        source,
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; HASH_BUFFER_BYTES];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|source| ArtifactError::Read {
                path: path.to_owned(),
                source,
            })?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    let digest = hasher.finalize();
    Ok(encode_lower_hex(digest.as_ref()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    encode_lower_hex(digest.as_ref())
}

fn encode_lower_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";

    let mut encoded = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        encoded.push(char::from(HEX[usize::from(*byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(*byte & 0x0f)]));
    }
    encoded
}

fn invalid<T>(message: String) -> Result<T, CatalogError> {
    Err(CatalogError::Invalid(message))
}

#[cfg(test)]
mod tests {
    use super::{
        Catalog, CatalogError, QWEN3_SMALL_ARTIFACT_PATH, QWEN3_SMALL_PROFILE_ID,
        QWEN3_SMALL_SOURCE_ID, sha256_bytes,
    };

    #[test]
    fn sha256_encoding_is_exact_lowercase_hex() {
        assert_eq!(
            sha256_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn packaged_catalog_is_valid() {
        let catalog = Catalog::embedded().expect("packaged catalog should be valid");
        assert_eq!(catalog.schema_version(), 1);
        assert_eq!(catalog.default_profiles().len(), 3);
        assert_eq!(catalog.profiles().len(), 3);
        assert!(
            catalog
                .find_profile(QWEN3_SMALL_PROFILE_ID)
                .and_then(|profile| {
                    profile.find_artifact(QWEN3_SMALL_SOURCE_ID, QWEN3_SMALL_ARTIFACT_PATH)
                })
                .is_some()
        );
    }

    #[test]
    fn duplicate_default_is_rejected() {
        let mut catalog = Catalog::embedded().expect("fixture should load");
        catalog.default_profiles[1] = catalog.default_profiles[0].clone();
        assert!(matches!(
            catalog.validate(),
            Err(CatalogError::Invalid(message)) if message.contains("duplicate default")
        ));
    }

    #[test]
    fn parent_path_is_rejected() {
        let mut catalog = Catalog::embedded().expect("fixture should load");
        catalog.profiles[0].sources[0].files[0].path = "../model.gguf".to_owned();
        assert!(matches!(
            catalog.validate(),
            Err(CatalogError::Invalid(message)) if message.contains("relative path")
        ));
    }

    #[test]
    fn weight_without_digest_is_rejected() {
        let mut catalog = Catalog::embedded().expect("fixture should load");
        catalog.profiles[0].sources[0].files[0].sha256 = None;
        assert!(matches!(
            catalog.validate(),
            Err(CatalogError::Invalid(message)) if message.contains("requires a SHA-256")
        ));
    }

    #[test]
    fn first_class_profile_requires_passed_acceptance_status() {
        let mut catalog = Catalog::embedded().expect("fixture should load");
        catalog.profiles[0].acceptance_status = super::AcceptanceStatus::NotRun;
        assert!(matches!(
            catalog.validate(),
            Err(CatalogError::Invalid(message)) if message.contains("must have passed")
        ));
    }
}

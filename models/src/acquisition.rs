// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{ArtifactReceipt, Catalog};

/// Current serialized acquisition-report schema.
pub const ACQUISITION_REPORT_SCHEMA_VERSION: u32 = 1;
/// Identity domain for acquisition-report interpretation.
pub const ACQUISITION_REPORT_IDENTITY_DOMAIN: &str = "logit-loom-model-acquisition-report-v1";

const REPORT_BYTES: &[u8] = include_bytes!("../reports/acquisition-2026-07-25.json");
const MAX_REPORT_BYTES: usize = 1024 * 1024;
const MAX_PROFILES: usize = 32;
const MAX_ARTIFACTS_PER_PROFILE: usize = 128;
const MAX_TEXT_BYTES: usize = 256;

/// A path-free record of one model-acquisition verification session.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AcquisitionReport {
    schema_version: u32,
    identity_domain: String,
    catalog_sha256: String,
    recorded_at_utc: String,
    hf_cli_version: String,
    filesystem_available_bytes: u64,
    profiles: Vec<ProfileAcquisition>,
}

impl AcquisitionReport {
    /// Parses and validates the acquisition report packaged with this crate.
    ///
    /// # Errors
    ///
    /// Returns a parse or validation error when the packaged report is invalid
    /// or no longer matches the packaged catalog.
    pub fn embedded(catalog: &Catalog) -> Result<Self, AcquisitionReportError> {
        Self::from_slice(REPORT_BYTES, catalog)
    }

    /// Parses and validates a bounded, path-free acquisition report.
    ///
    /// # Errors
    ///
    /// Returns a parse or invariant error when the report is malformed,
    /// incomplete, internally inconsistent, or bound to another catalog.
    pub fn from_slice(bytes: &[u8], catalog: &Catalog) -> Result<Self, AcquisitionReportError> {
        if bytes.len() > MAX_REPORT_BYTES {
            return invalid("serialized acquisition report exceeds the input bound");
        }
        let report =
            serde_json::from_slice::<Self>(bytes).map_err(AcquisitionReportError::Parse)?;
        report.validate(catalog)?;
        Ok(report)
    }

    /// Validates report bounds and exact correspondence with a catalog.
    ///
    /// A `verified` profile must partition every catalogued artifact into its
    /// verified set. A `partial` profile must partition every artifact between
    /// verified and pending sets without overlap. Local filesystem paths are
    /// absent from this serialized shape by construction.
    ///
    /// # Errors
    ///
    /// Returns the first violated report invariant.
    pub fn validate(&self, catalog: &Catalog) -> Result<(), AcquisitionReportError> {
        if self.schema_version != ACQUISITION_REPORT_SCHEMA_VERSION {
            return invalid(format!(
                "schema_version must be {ACQUISITION_REPORT_SCHEMA_VERSION}"
            ));
        }
        if self.identity_domain != ACQUISITION_REPORT_IDENTITY_DOMAIN {
            return invalid(format!(
                "identity_domain must be {ACQUISITION_REPORT_IDENTITY_DOMAIN:?}"
            ));
        }
        let catalog_sha256 = catalog.packaged_sha256();
        if self.catalog_sha256 != catalog_sha256 {
            return invalid(format!(
                "catalog_sha256 must match the packaged catalog {catalog_sha256}"
            ));
        }
        validate_timestamp(&self.recorded_at_utc)?;
        validate_version(&self.hf_cli_version)?;
        if self.filesystem_available_bytes == 0 {
            return invalid("filesystem_available_bytes must be nonzero");
        }
        if self.profiles.is_empty() || self.profiles.len() > MAX_PROFILES {
            return invalid(format!(
                "profiles must contain between 1 and {MAX_PROFILES} entries"
            ));
        }

        let mut profile_ids = HashSet::new();
        for acquisition in &self.profiles {
            if !profile_ids.insert(acquisition.profile_id.as_str()) {
                return invalid(format!(
                    "duplicate acquisition profile {:?}",
                    acquisition.profile_id
                ));
            }
            validate_profile(acquisition, catalog, &catalog_sha256)?;
        }
        Ok(())
    }

    /// Returns the catalog digest this report is bound to.
    pub fn catalog_sha256(&self) -> &str {
        &self.catalog_sha256
    }

    /// Returns the UTC capture timestamp.
    pub fn recorded_at_utc(&self) -> &str {
        &self.recorded_at_utc
    }

    /// Returns the observed Hugging Face CLI version.
    pub fn hf_cli_version(&self) -> &str {
        &self.hf_cli_version
    }

    /// Returns available bytes on the model-store filesystem at capture time.
    pub const fn filesystem_available_bytes(&self) -> u64 {
        self.filesystem_available_bytes
    }

    /// Returns profile records in declared order.
    pub fn profiles(&self) -> &[ProfileAcquisition] {
        &self.profiles
    }
}

/// How the verified files reached their caller-managed store.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AcquisitionMethod {
    /// Exact files were fetched by the repository's `hf`-based command.
    HfFetch,
    /// Exact files already existed in a separate caller-managed store.
    ExistingLocalArtifacts,
    /// Exact files were verified across repository fetches and caller-managed stores.
    Mixed,
}

/// Completeness of one profile acquisition.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AcquisitionStatus {
    /// Every catalogued file was verified.
    Verified,
    /// Verified and pending files form a complete, disjoint partition.
    Partial,
}

/// One profile's path-free acquisition evidence.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileAcquisition {
    profile_id: String,
    method: AcquisitionMethod,
    status: AcquisitionStatus,
    catalog_artifact_bytes: u64,
    verified_artifact_bytes: u64,
    measured_profile_storage_bytes: Option<u64>,
    verified_artifacts: Vec<ArtifactReceipt>,
    pending_artifacts: Vec<PendingArtifact>,
}

impl ProfileAcquisition {
    /// Returns the profile ID.
    pub fn profile_id(&self) -> &str {
        &self.profile_id
    }

    /// Returns how artifacts reached the local store.
    pub const fn method(&self) -> AcquisitionMethod {
        self.method
    }

    /// Returns whether all catalogued artifacts were verified.
    pub const fn status(&self) -> AcquisitionStatus {
        self.status
    }

    /// Returns the catalogued total bytes for the profile.
    pub const fn catalog_artifact_bytes(&self) -> u64 {
        self.catalog_artifact_bytes
    }

    /// Returns the total bytes represented by verified receipts.
    pub const fn verified_artifact_bytes(&self) -> u64 {
        self.verified_artifact_bytes
    }

    /// Returns measured directory storage, when all files shared one root.
    pub const fn measured_profile_storage_bytes(&self) -> Option<u64> {
        self.measured_profile_storage_bytes
    }

    /// Returns exact path-free artifact receipts.
    pub fn verified_artifacts(&self) -> &[ArtifactReceipt] {
        &self.verified_artifacts
    }

    /// Returns exact artifacts that were not verified.
    pub fn pending_artifacts(&self) -> &[PendingArtifact] {
        &self.pending_artifacts
    }
}

/// One exact catalog artifact that was deliberately not verified.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PendingArtifact {
    /// Source ID within the profile.
    pub source_id: String,
    /// Source-relative artifact path.
    pub artifact_path: String,
    /// Explicit reason this exact artifact remains pending.
    pub reason: PendingReason,
}

/// Why an acquisition report did not verify an exact artifact.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum PendingReason {
    /// The operator did not assert acceptance of upstream gated terms.
    GatedTermsNotAcknowledged,
    /// The exact artifact was not present in the selected caller store.
    ArtifactNotPresent,
}

/// Acquisition-report parsing or invariant failure.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AcquisitionReportError {
    /// JSON parsing failed.
    #[error("failed to parse model acquisition report: {0}")]
    Parse(serde_json::Error),
    /// A bounded report invariant failed.
    #[error("invalid model acquisition report: {0}")]
    Invalid(String),
}

fn validate_profile(
    acquisition: &ProfileAcquisition,
    catalog: &Catalog,
    catalog_sha256: &str,
) -> Result<(), AcquisitionReportError> {
    let profile = catalog
        .find_profile(&acquisition.profile_id)
        .ok_or_else(|| {
            AcquisitionReportError::Invalid(format!(
                "unknown acquisition profile {:?}",
                acquisition.profile_id
            ))
        })?;
    if acquisition.catalog_artifact_bytes != profile.total_bytes() {
        return invalid(format!(
            "profile {:?} catalog_artifact_bytes does not match the catalog",
            acquisition.profile_id
        ));
    }
    if acquisition.verified_artifacts.len() + acquisition.pending_artifacts.len()
        != profile.file_count()
        || acquisition.verified_artifacts.len() > MAX_ARTIFACTS_PER_PROFILE
        || acquisition.pending_artifacts.len() > MAX_ARTIFACTS_PER_PROFILE
    {
        return invalid(format!(
            "profile {:?} artifacts do not form a bounded catalog partition",
            acquisition.profile_id
        ));
    }

    let mut identities = HashSet::new();
    let mut verified_bytes = 0_u64;
    for receipt in &acquisition.verified_artifacts {
        let identity = (receipt.source_id.as_str(), receipt.artifact_path.as_str());
        if !identities.insert(identity) {
            return invalid(format!(
                "profile {:?} repeats artifact {:?}/{:?}",
                acquisition.profile_id, receipt.source_id, receipt.artifact_path
            ));
        }
        validate_receipt(acquisition, receipt, profile, catalog_sha256)?;
        verified_bytes = verified_bytes.checked_add(receipt.bytes).ok_or_else(|| {
            AcquisitionReportError::Invalid(format!(
                "profile {:?} verified byte count overflows",
                acquisition.profile_id
            ))
        })?;
    }
    for pending in &acquisition.pending_artifacts {
        let identity = (pending.source_id.as_str(), pending.artifact_path.as_str());
        if !identities.insert(identity) {
            return invalid(format!(
                "profile {:?} repeats pending artifact {:?}/{:?}",
                acquisition.profile_id, pending.source_id, pending.artifact_path
            ));
        }
        let (source, _) = profile
            .find_artifact(&pending.source_id, &pending.artifact_path)
            .ok_or_else(|| {
                AcquisitionReportError::Invalid(format!(
                    "profile {:?} has unknown pending artifact {:?}/{:?}",
                    acquisition.profile_id, pending.source_id, pending.artifact_path
                ))
            })?;
        if pending.reason == PendingReason::GatedTermsNotAcknowledged && !source.gated() {
            return invalid(format!(
                "profile {:?} marks an ungated artifact as awaiting terms",
                acquisition.profile_id
            ));
        }
    }
    if identities.len() != profile.file_count() {
        return invalid(format!(
            "profile {:?} does not cover every catalog artifact",
            acquisition.profile_id
        ));
    }
    if verified_bytes != acquisition.verified_artifact_bytes {
        return invalid(format!(
            "profile {:?} verified_artifact_bytes does not match its receipts",
            acquisition.profile_id
        ));
    }
    if acquisition
        .measured_profile_storage_bytes
        .is_some_and(|measured| measured < verified_bytes)
    {
        return invalid(format!(
            "profile {:?} measured storage is smaller than verified artifacts",
            acquisition.profile_id
        ));
    }
    match acquisition.status {
        AcquisitionStatus::Verified
            if acquisition.pending_artifacts.is_empty()
                && verified_bytes == profile.total_bytes() => {}
        AcquisitionStatus::Partial if !acquisition.pending_artifacts.is_empty() => {}
        _ => {
            return invalid(format!(
                "profile {:?} status is inconsistent with its artifact partition",
                acquisition.profile_id
            ));
        }
    }
    Ok(())
}

fn validate_receipt(
    acquisition: &ProfileAcquisition,
    receipt: &ArtifactReceipt,
    profile: &crate::Profile,
    catalog_sha256: &str,
) -> Result<(), AcquisitionReportError> {
    if receipt.catalog_domain != crate::CATALOG_IDENTITY_DOMAIN
        || receipt.catalog_sha256 != catalog_sha256
        || receipt.profile_id != acquisition.profile_id
    {
        return invalid(format!(
            "profile {:?} has a receipt bound to another catalog or profile",
            acquisition.profile_id
        ));
    }
    let (source, artifact) = profile
        .find_artifact(&receipt.source_id, &receipt.artifact_path)
        .ok_or_else(|| {
            AcquisitionReportError::Invalid(format!(
                "profile {:?} has unknown receipt artifact {:?}/{:?}",
                acquisition.profile_id, receipt.source_id, receipt.artifact_path
            ))
        })?;
    if receipt.repository != source.repository()
        || receipt.revision != source.revision()
        || receipt.bytes != artifact.bytes()
        || receipt.sha256.as_deref() != artifact.sha256()
    {
        return invalid(format!(
            "profile {:?} receipt {:?}/{:?} does not match the catalog",
            acquisition.profile_id, receipt.source_id, receipt.artifact_path
        ));
    }
    Ok(())
}

fn validate_timestamp(value: &str) -> Result<(), AcquisitionReportError> {
    if value.is_empty()
        || value.len() > MAX_TEXT_BYTES
        || !value.is_ascii()
        || !value.contains('T')
        || !value.ends_with('Z')
    {
        return invalid("recorded_at_utc must be a bounded UTC timestamp");
    }
    Ok(())
}

fn validate_version(value: &str) -> Result<(), AcquisitionReportError> {
    if value.is_empty()
        || value.len() > 32
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || byte == b'.' || byte == b'-')
    {
        return invalid("hf_cli_version must be a bounded version string");
    }
    Ok(())
}

fn invalid<T>(message: impl Into<String>) -> Result<T, AcquisitionReportError> {
    Err(AcquisitionReportError::Invalid(message.into()))
}

#[cfg(test)]
mod tests {
    use super::{AcquisitionMethod, AcquisitionReport, AcquisitionStatus};
    use crate::Catalog;

    #[test]
    fn packaged_report_is_path_free_and_catalog_bound() {
        let catalog = Catalog::embedded().expect("catalog should parse");
        let report = AcquisitionReport::embedded(&catalog).expect("report should validate");
        assert_eq!(report.profiles().len(), 3);
        assert_eq!(report.profiles()[0].status(), AcquisitionStatus::Verified);
        assert_eq!(report.profiles()[2].status(), AcquisitionStatus::Verified);
        assert_eq!(report.profiles()[2].method(), AcquisitionMethod::Mixed);

        let encoded = serde_json::to_string(&report).expect("report should serialize");
        let home_path_marker = ["/", "home", "/"].concat();
        assert!(!encoded.contains(&home_path_marker));
        assert!(!encoded.contains("model-store"));
    }

    #[test]
    fn changed_catalog_digest_is_rejected() {
        let catalog = Catalog::embedded().expect("catalog should parse");
        let mut value =
            serde_json::from_slice::<serde_json::Value>(super::REPORT_BYTES).expect("valid JSON");
        value["catalog_sha256"] = serde_json::Value::String("0".repeat(64));
        let bytes = serde_json::to_vec(&value).expect("JSON should serialize");
        let error =
            AcquisitionReport::from_slice(&bytes, &catalog).expect_err("digest must be checked");
        assert!(error.to_string().contains("catalog_sha256"));
    }
}

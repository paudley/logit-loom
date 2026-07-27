// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded validation for retained model-backed acceptance reports.

use std::{
    collections::HashSet,
    fs,
    path::{Component, Path},
};

use logit_loom_models::{AcceptanceStatus, Catalog};
use serde::Deserialize;

const SCHEMA_VERSION: u64 = 1;
const IDENTITY_DOMAIN: &str = "logit-loom-model-acceptance-report-v1";
const MAX_REPORT_BYTES: u64 = 1024 * 1024;
const MAX_REPORTS: usize = 64;
const MAX_ARTIFACTS: usize = 32;
const MAX_DEVICES: usize = 64;
const MAX_RUNS: usize = 16;
const MAX_ASSERTIONS: usize = 64;
const MAX_BLOCKERS: usize = 32;
const MAX_STEP_MEASUREMENTS: usize = 4_096;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AcceptanceReport {
    schema_version: u64,
    identity_domain: String,
    profile_id: String,
    modality: String,
    adapter: String,
    experiment: String,
    status: String,
    catalog_sha256: String,
    artifacts: Vec<Artifact>,
    runtime: Runtime,
    runs: Vec<Run>,
    assertions: Vec<Assertion>,
    measurements: Option<Measurements>,
    blockers: Vec<Blocker>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    source_id: String,
    repository: String,
    revision: String,
    #[serde(rename = "artifact_path")]
    path: String,
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Runtime {
    adapter_build_identity: String,
    backend: String,
    parameter_backend: Option<String>,
    devices: Vec<String>,
    accelerator_required: bool,
    cpu_fallback: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Run {
    label: String,
    plan_identity: String,
    receipt_identity: String,
    output_identity: String,
    completed_steps: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Assertion {
    id: String,
    status: String,
    evidence: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Measurements {
    peak_host_bytes: Option<u64>,
    peak_device_bytes: Option<u64>,
    step_latency_milliseconds: Option<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Blocker {
    code: String,
    detail: String,
}

pub fn check_reports(repository: &Path, catalog: &Catalog) -> Result<usize, String> {
    let directory = repository.join("docs/acceptance");
    let mut paths = fs::read_dir(&directory)
        .map_err(|error| format!("failed to read {}: {error}", directory.display()))?
        .map(|entry| entry.map(|value| value.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("failed to inspect {}: {error}", directory.display()))?;
    paths.retain(|path| {
        path.extension().and_then(|value| value.to_str()) == Some("json")
            && path.file_name().and_then(|value| value.to_str()) != Some("model-run.schema.json")
    });
    paths.sort();
    if paths.is_empty() || paths.len() > MAX_REPORTS {
        return Err(format!(
            "expected 1..={MAX_REPORTS} retained JSON reports, found {}",
            paths.len()
        ));
    }

    let mut passed_profiles = HashSet::new();
    for path in &paths {
        let metadata = fs::metadata(path)
            .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?;
        if metadata.len() > MAX_REPORT_BYTES {
            return Err(format!(
                "{} exceeds the {MAX_REPORT_BYTES}-byte report bound",
                path.display()
            ));
        }
        let bytes = fs::read(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let report = serde_json::from_slice::<AcceptanceReport>(&bytes)
            .map_err(|error| format!("{} is not a closed report: {error}", path.display()))?;
        report
            .validate(catalog)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        if report.status == "passed" {
            passed_profiles.insert(report.profile_id.clone());
        }
    }
    check_promoted_profiles(catalog, &passed_profiles)?;
    Ok(paths.len())
}

fn check_promoted_profiles(
    catalog: &Catalog,
    passed_profiles: &HashSet<String>,
) -> Result<(), String> {
    for profile in catalog.profiles() {
        if profile.acceptance_status() == AcceptanceStatus::Passed
            && !passed_profiles.contains(profile.id())
        {
            return invalid(format!(
                "profile {:?} claims passed acceptance without a retained passed report",
                profile.id()
            ));
        }
    }
    Ok(())
}

impl AcceptanceReport {
    fn validate(&self, catalog: &Catalog) -> Result<(), String> {
        if self.schema_version != SCHEMA_VERSION || self.identity_domain != IDENTITY_DOMAIN {
            return invalid("schema version or identity domain differs");
        }
        identifier("profile_id", &self.profile_id)?;
        identifier("adapter", &self.adapter)?;
        identifier("experiment", &self.experiment)?;
        if !matches!(self.modality.as_str(), "text" | "image") {
            return invalid("modality must be text or image");
        }
        if !matches!(self.status.as_str(), "passed" | "failed" | "blocked") {
            return invalid("status must be passed, failed, or blocked");
        }
        digest("catalog_sha256", &self.catalog_sha256)?;
        let expected_catalog = catalog.packaged_sha256();
        if self.catalog_sha256 != expected_catalog {
            return invalid(format!(
                "catalog_sha256 must match the packaged catalog {expected_catalog}"
            ));
        }
        let profile = catalog
            .find_profile(&self.profile_id)
            .ok_or_else(|| format!("unknown profile_id {:?}", self.profile_id))?;
        if self.modality != profile.modality().as_str() {
            return invalid("modality differs from the catalogued profile");
        }

        if self.artifacts.is_empty() || self.artifacts.len() > MAX_ARTIFACTS {
            return invalid(format!(
                "artifacts must contain 1..={MAX_ARTIFACTS} entries"
            ));
        }
        let mut artifact_keys = HashSet::new();
        for artifact in &self.artifacts {
            artifact.validate(profile)?;
            if !artifact_keys.insert((&artifact.source_id, &artifact.path)) {
                return invalid("artifact source/path pairs must be unique");
            }
        }
        self.runtime.validate()?;
        if self.runs.is_empty() || self.runs.len() > MAX_RUNS {
            return invalid(format!("runs must contain 1..={MAX_RUNS} entries"));
        }
        let mut run_labels = HashSet::new();
        for run in &self.runs {
            run.validate()?;
            if !run_labels.insert(&run.label) {
                return invalid("run labels must be unique");
            }
        }
        if self.assertions.is_empty() || self.assertions.len() > MAX_ASSERTIONS {
            return invalid(format!(
                "assertions must contain 1..={MAX_ASSERTIONS} entries"
            ));
        }
        let mut assertion_ids = HashSet::new();
        for assertion in &self.assertions {
            assertion.validate()?;
            if !assertion_ids.insert(&assertion.id) {
                return invalid("assertion IDs must be unique");
            }
        }
        if let Some(measurements) = &self.measurements {
            measurements.validate()?;
        }
        if self.blockers.len() > MAX_BLOCKERS {
            return invalid(format!(
                "blockers must contain at most {MAX_BLOCKERS} entries"
            ));
        }
        let mut blocker_codes = HashSet::new();
        for blocker in &self.blockers {
            blocker.validate()?;
            if !blocker_codes.insert(&blocker.code) {
                return invalid("blocker codes must be unique");
            }
        }

        match self.status.as_str() {
            "passed"
                if self.blockers.is_empty()
                    && self
                        .assertions
                        .iter()
                        .all(|assertion| assertion.status == "passed") => {}
            "failed"
                if self
                    .assertions
                    .iter()
                    .any(|assertion| assertion.status == "failed") => {}
            "blocked" if !self.blockers.is_empty() => {}
            "passed" => {
                return invalid("passed reports require all assertions passed and no blockers");
            }
            "failed" => return invalid("failed reports require a failed assertion"),
            "blocked" => return invalid("blocked reports require at least one blocker"),
            _ => unreachable!("status spelling was validated above"),
        }
        Ok(())
    }
}

impl Artifact {
    fn validate(&self, profile: &logit_loom_models::Profile) -> Result<(), String> {
        identifier("artifact source_id", &self.source_id)?;
        repository(&self.repository)?;
        hexadecimal("artifact revision", &self.revision, 40)?;
        relative_path(&self.path)?;
        if self.bytes == 0 {
            return invalid("artifact bytes must be positive");
        }
        digest("artifact sha256", &self.sha256)?;

        let source = profile
            .sources()
            .iter()
            .find(|source| source.id() == self.source_id)
            .ok_or_else(|| format!("unknown artifact source_id {:?}", self.source_id))?;
        let file = source
            .files()
            .iter()
            .find(|file| file.path() == self.path)
            .ok_or_else(|| format!("unknown artifact path {:?}", self.path))?;
        if self.repository != source.repository()
            || self.revision != source.revision()
            || self.bytes != file.bytes()
            || file.sha256() != Some(self.sha256.as_str())
        {
            return invalid("artifact metadata differs from the exact catalog entry");
        }
        Ok(())
    }
}

impl Runtime {
    fn validate(&self) -> Result<(), String> {
        digest("adapter_build_identity", &self.adapter_build_identity)?;
        bounded_text("backend", &self.backend)?;
        if let Some(parameter_backend) = &self.parameter_backend {
            bounded_text("parameter_backend", parameter_backend)?;
        }
        if self.devices.is_empty() || self.devices.len() > MAX_DEVICES {
            return invalid(format!("devices must contain 1..={MAX_DEVICES} entries"));
        }
        for device in &self.devices {
            bounded_text("device", device)?;
        }
        if !self.accelerator_required || self.cpu_fallback {
            return invalid("runtime must require an accelerator and forbid CPU fallback");
        }
        if self.backend.to_ascii_lowercase().contains("cpu")
            || self
                .parameter_backend
                .as_ref()
                .is_some_and(|backend| backend.to_ascii_lowercase().contains("cpu"))
            || !self
                .devices
                .iter()
                .any(|device| !device.to_ascii_lowercase().starts_with("cpu"))
        {
            return invalid("runtime must identify a selected non-CPU backend and device");
        }
        Ok(())
    }
}

impl Run {
    fn validate(&self) -> Result<(), String> {
        identifier("run label", &self.label)?;
        digest("plan_identity", &self.plan_identity)?;
        digest("receipt_identity", &self.receipt_identity)?;
        digest("output_identity", &self.output_identity)?;
        if self.completed_steps > 4_096 {
            return invalid("completed_steps exceeds 4096");
        }
        Ok(())
    }
}

impl Assertion {
    fn validate(&self) -> Result<(), String> {
        identifier("assertion id", &self.id)?;
        if !matches!(self.status.as_str(), "passed" | "failed" | "not-run") {
            return invalid("assertion status must be passed, failed, or not-run");
        }
        bounded_text("assertion evidence", &self.evidence)
    }
}

impl Measurements {
    fn validate(&self) -> Result<(), String> {
        let _ = self.peak_host_bytes;
        let _ = self.peak_device_bytes;
        if let Some(latencies) = &self.step_latency_milliseconds
            && (latencies.len() > MAX_STEP_MEASUREMENTS
                || latencies
                    .iter()
                    .any(|latency| !latency.is_finite() || *latency < 0.0))
        {
            return invalid(format!(
                "step latencies must contain at most {MAX_STEP_MEASUREMENTS} finite nonnegative values"
            ));
        }
        Ok(())
    }
}

impl Blocker {
    fn validate(&self) -> Result<(), String> {
        identifier("blocker code", &self.code)?;
        bounded_text("blocker detail", &self.detail)
    }
}

fn identifier(label: &str, value: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .next()
            .is_some_and(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'.' | b'_' | b'-')
        })
    {
        return invalid(format!("{label} is not a bounded lowercase identifier"));
    }
    Ok(())
}

fn bounded_text(label: &str, value: &str) -> Result<(), String> {
    if value.is_empty() || value.len() > 512 {
        return invalid(format!("{label} must contain 1..=512 UTF-8 bytes"));
    }
    Ok(())
}

fn digest(label: &str, value: &str) -> Result<(), String> {
    hexadecimal(label, value, 64)
}

fn hexadecimal(label: &str, value: &str, bytes: usize) -> Result<(), String> {
    if value.len() != bytes
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return invalid(format!(
            "{label} must contain {bytes} lowercase hexadecimal bytes"
        ));
    }
    Ok(())
}

fn repository(value: &str) -> Result<(), String> {
    let Some((owner, name)) = value.split_once('/') else {
        return invalid("artifact repository must have owner/name form");
    };
    if owner.is_empty()
        || name.is_empty()
        || name.contains('/')
        || value.len() > 256
        || value.chars().any(char::is_whitespace)
    {
        return invalid("artifact repository must have bounded owner/name form");
    }
    Ok(())
}

fn relative_path(value: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > 512
        || value.starts_with('/')
        || Path::new(value)
            .components()
            .any(|component| matches!(component, Component::ParentDir))
    {
        return invalid("artifact_path must be a bounded relative path without parent traversal");
    }
    Ok(())
}

fn invalid<T>(message: impl Into<String>) -> Result<T, String> {
    Err(message.into())
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, path::Path};

    use logit_loom_models::Catalog;

    #[test]
    fn retained_reports_pass_bounded_validation() {
        let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask should have a workspace parent");
        let catalog = Catalog::embedded().expect("catalog should parse");
        assert_eq!(
            super::check_reports(repository, &catalog).expect("reports should validate"),
            3
        );
    }

    #[test]
    fn cpu_fallback_is_rejected() {
        let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask should have a workspace parent");
        let bytes = std::fs::read(
            repository.join("docs/acceptance/qwen3-0.6b-q8-0-vulkan-2026-07-25.json"),
        )
        .expect("report should be readable");
        let mut report =
            serde_json::from_slice::<super::AcceptanceReport>(&bytes).expect("report should parse");
        report.runtime.cpu_fallback = true;
        let catalog = Catalog::embedded().expect("catalog should parse");
        let error = report
            .validate(&catalog)
            .expect_err("CPU fallback must fail");
        assert!(error.contains("forbid CPU fallback"));
    }

    #[test]
    fn promoted_profile_requires_a_retained_passed_report() {
        let catalog = Catalog::embedded().expect("catalog should parse");
        let passed_profiles = HashSet::from(["minit2i-b16".to_owned()]);
        let error = super::check_promoted_profiles(&catalog, &passed_profiles)
            .expect_err("the promoted Qwen profile has no report in this set");
        assert!(error.contains("qwen3-0.6b-q8-0"));
    }
}

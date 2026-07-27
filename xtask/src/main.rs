// SPDX-License-Identifier: MIT OR Apache-2.0

//! Repository-local maintenance commands.

mod acceptance;
mod acquire;

use std::{
    env, fs,
    path::{Component, Path, PathBuf},
    process::ExitCode,
};

use thiserror::Error;

use crate::acquire::{AcquireError, fetch_profile, verify_profile};
use logit_loom_models::{
    AcquisitionReport, AcquisitionReportError, ArtifactError, Catalog, CatalogError, Profile,
};

const USAGE: &str = "\
usage:
  cargo run --quiet -p logit-loom-xtask -- models check
  cargo run --quiet -p logit-loom-xtask -- models list
  cargo run --quiet -p logit-loom-xtask -- models fetch <profile> --dir <path> [--dry-run] [--accept-license]
  cargo run --quiet -p logit-loom-xtask -- models verify <profile> --dir <path>
  cargo run --quiet -p logit-loom-xtask -- models verify-artifact <profile> <source> <artifact> --path <file>
";
const MAX_ACCEPTANCE_SCHEMA_BYTES: usize = 1024 * 1024;
const ACCEPTANCE_SCHEMA_VERSION: u64 = 1;
const ACCEPTANCE_IDENTITY_DOMAIN: &str = "logit-loom-model-acceptance-report-v1";

#[derive(Debug, Error)]
enum Error {
    #[error("{0}\n\n{USAGE}")]
    Usage(String),
    #[error(transparent)]
    Catalog(#[from] CatalogError),
    #[error(transparent)]
    Acquire(#[from] AcquireError),
    #[error(transparent)]
    Artifact(#[from] ArtifactError),
    #[error(transparent)]
    AcquisitionReport(#[from] AcquisitionReportError),
    #[error("invalid retained acceptance report: {0}")]
    AcceptanceReport(String),
}

fn main() -> ExitCode {
    match run(env::args().skip(1)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: impl IntoIterator<Item = String>) -> Result<(), Error> {
    let args = args.into_iter().collect::<Vec<_>>();
    match args.first().map(String::as_str) {
        Some("models") => run_models(&args[1..]),
        Some("-h" | "--help" | "help") => {
            print!("{USAGE}");
            Ok(())
        }
        Some(command) => Err(Error::Usage(format!("unknown command {command:?}"))),
        None => Err(Error::Usage("missing command".to_owned())),
    }
}

fn run_models(args: &[String]) -> Result<(), Error> {
    match args.first().map(String::as_str) {
        Some("check") if args.len() == 1 => check_catalog(),
        Some("list") if args.len() == 1 => list_profiles(),
        Some("fetch") => {
            let options = parse_fetch_options(&args[1..])?;
            let catalog = Catalog::embedded()?;
            let profile = find_profile(&catalog, &options.profile)?;
            fetch_profile(
                profile,
                &options.destination,
                options.accept_license,
                options.dry_run,
            )?;
            if options.dry_run {
                println!("dry run complete; no directories were created and `hf` was not invoked");
            } else {
                println!(
                    "fetched and verified profile {:?} under {}",
                    profile.id(),
                    options.destination.display()
                );
            }
            Ok(())
        }
        Some("verify") => {
            let options = parse_verify_options(&args[1..])?;
            let catalog = Catalog::embedded()?;
            let profile = find_profile(&catalog, &options.profile)?;
            verify_profile(profile, &options.destination)?;
            println!(
                "verified profile {:?} under {}",
                profile.id(),
                options.destination.display()
            );
            Ok(())
        }
        Some("verify-artifact") => {
            let options = parse_verify_artifact_options(&args[1..])?;
            let catalog = Catalog::embedded()?;
            let profile = find_profile(&catalog, &options.profile)?;
            let verified = profile.verify_artifact(
                &catalog.packaged_sha256(),
                &options.source,
                &options.artifact,
                &options.path,
            )?;
            serde_json::to_writer_pretty(std::io::stdout().lock(), verified.receipt())
                .map_err(|error| Error::Usage(format!("failed to write receipt: {error}")))?;
            println!();
            Ok(())
        }
        Some("-h" | "--help" | "help") => {
            print!("{USAGE}");
            Ok(())
        }
        Some(command) => Err(Error::Usage(format!("unknown models command {command:?}"))),
        None => Err(Error::Usage("missing models command".to_owned())),
    }
}

fn check_catalog() -> Result<(), Error> {
    let catalog = Catalog::embedded()?;
    let report = AcquisitionReport::embedded(&catalog)?;
    check_acceptance_schema()?;
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or_else(|| Error::Usage("failed to resolve repository root".to_owned()))?;
    let acceptance_reports =
        acceptance::check_reports(repository, &catalog).map_err(Error::AcceptanceReport)?;
    let file_count = catalog
        .profiles()
        .iter()
        .map(Profile::file_count)
        .sum::<usize>();
    println!(
        "model catalog valid: {} profiles, {} default profiles, {} exact files; \
         acquisition report valid for {} profiles; acceptance schema and {} retained reports valid",
        catalog.profiles().len(),
        catalog.default_profiles().len(),
        file_count,
        report.profiles().len(),
        acceptance_reports
    );
    Ok(())
}

fn check_acceptance_schema() -> Result<(), Error> {
    let schema_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or_else(|| Error::Usage("failed to resolve repository root".to_owned()))?
        .join("docs/acceptance/model-run.schema.json");
    let bytes = fs::read(&schema_path).map_err(|error| {
        Error::Usage(format!(
            "failed to read acceptance schema {}: {error}",
            schema_path.display()
        ))
    })?;
    if bytes.len() > MAX_ACCEPTANCE_SCHEMA_BYTES {
        return Err(Error::Usage(
            "acceptance schema exceeds its input bound".to_owned(),
        ));
    }
    let schema = serde_json::from_slice::<serde_json::Value>(&bytes)
        .map_err(|error| Error::Usage(format!("acceptance schema is invalid JSON: {error}")))?;
    if schema
        .pointer("/properties/schema_version/const")
        .and_then(serde_json::Value::as_u64)
        != Some(ACCEPTANCE_SCHEMA_VERSION)
        || schema
            .pointer("/properties/identity_domain/const")
            .and_then(serde_json::Value::as_str)
            != Some(ACCEPTANCE_IDENTITY_DOMAIN)
        || schema
            .get("additionalProperties")
            .and_then(serde_json::Value::as_bool)
            != Some(false)
    {
        return Err(Error::Usage(
            "acceptance schema version, identity domain, or closed shape differs".to_owned(),
        ));
    }
    Ok(())
}

fn list_profiles() -> Result<(), Error> {
    let catalog = Catalog::embedded()?;
    println!(
        "{:<20} {:<7} {:<9} {:<11} {:<10} {:>10}",
        "PROFILE", "MODE", "ROLE", "INTEGRATION", "ACCEPTANCE", "DOWNLOAD"
    );
    for profile in catalog.profiles() {
        println!(
            "{:<20} {:<7} {:<9} {:<11} {:<10} {:>10}",
            profile.id(),
            profile.modality().as_str(),
            profile.role().as_str(),
            profile.integration_status().as_str(),
            profile.acceptance_status().as_str(),
            format_bytes(profile.total_bytes())
        );
    }
    Ok(())
}

fn find_profile<'a>(catalog: &'a Catalog, id: &str) -> Result<&'a Profile, Error> {
    catalog
        .find_profile(id)
        .ok_or_else(|| Error::Usage(format!("unknown model profile {id:?}")))
}

#[derive(Debug)]
struct FetchOptions {
    profile: String,
    destination: PathBuf,
    accept_license: bool,
    dry_run: bool,
}

fn parse_fetch_options(args: &[String]) -> Result<FetchOptions, Error> {
    let Some(profile) = args.first() else {
        return Err(Error::Usage("missing model profile".to_owned()));
    };
    if profile.starts_with('-') {
        return Err(Error::Usage(
            "model profile must precede options".to_owned(),
        ));
    }

    let mut destination = None;
    let mut accept_license = false;
    let mut dry_run = false;
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--dir" => {
                if destination.is_some() {
                    return Err(Error::Usage("--dir may only be supplied once".to_owned()));
                }
                index += 1;
                let Some(path) = args.get(index) else {
                    return Err(Error::Usage("--dir requires a path".to_owned()));
                };
                destination = Some(PathBuf::from(path));
            }
            "--accept-license" if !accept_license => accept_license = true,
            "--dry-run" if !dry_run => dry_run = true,
            option => {
                return Err(Error::Usage(format!(
                    "unknown or duplicate option {option:?}"
                )));
            }
        }
        index += 1;
    }

    let destination =
        destination.ok_or_else(|| Error::Usage("fetch requires --dir <path>".to_owned()))?;
    validate_destination(&destination)?;
    Ok(FetchOptions {
        profile: profile.clone(),
        destination,
        accept_license,
        dry_run,
    })
}

#[derive(Debug)]
struct VerifyOptions {
    profile: String,
    destination: PathBuf,
}

#[derive(Debug)]
struct VerifyArtifactOptions {
    profile: String,
    source: String,
    artifact: String,
    path: PathBuf,
}

fn parse_verify_options(args: &[String]) -> Result<VerifyOptions, Error> {
    if args.len() != 3 || args.get(1).map(String::as_str) != Some("--dir") {
        return Err(Error::Usage(
            "verify requires <profile> --dir <path>".to_owned(),
        ));
    }
    let profile = args[0].clone();
    if profile.starts_with('-') {
        return Err(Error::Usage(
            "model profile must precede options".to_owned(),
        ));
    }
    let destination = PathBuf::from(&args[2]);
    validate_destination(&destination)?;
    Ok(VerifyOptions {
        profile,
        destination,
    })
}

fn parse_verify_artifact_options(args: &[String]) -> Result<VerifyArtifactOptions, Error> {
    if args.len() != 5 || args.get(3).map(String::as_str) != Some("--path") {
        return Err(Error::Usage(
            "verify-artifact requires <profile> <source> <artifact> --path <file>".to_owned(),
        ));
    }
    if args[..3].iter().any(|value| value.starts_with('-')) {
        return Err(Error::Usage(
            "profile, source, and artifact must precede --path".to_owned(),
        ));
    }
    let path = PathBuf::from(&args[4]);
    validate_destination(&path)?;
    Ok(VerifyArtifactOptions {
        profile: args[0].clone(),
        source: args[1].clone(),
        artifact: args[2].clone(),
        path,
    })
}

fn validate_destination(path: &Path) -> Result<(), Error> {
    if path.as_os_str().is_empty() {
        return Err(Error::Usage(
            "model destination must not be empty".to_owned(),
        ));
    }
    let current_dir = env::current_dir()
        .map_err(|error| Error::Usage(format!("failed to resolve current directory: {error}")))?;
    let absolute = if path.is_absolute() {
        path.to_owned()
    } else {
        current_dir.join(path)
    };
    let destination = resolve_existing_prefix(&normalize_path(&absolute)).map_err(|error| {
        Error::Usage(format!(
            "failed to resolve model destination {}: {error}",
            path.display()
        ))
    })?;
    let repository = fs::canonicalize(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .ok_or_else(|| Error::Usage("failed to resolve repository root".to_owned()))?,
    )
    .map_err(|error| Error::Usage(format!("failed to resolve repository root: {error}")))?;
    if destination.starts_with(repository) {
        return Err(Error::Usage(
            "model destination must be outside the repository checkout".to_owned(),
        ));
    }
    Ok(())
}

fn resolve_existing_prefix(path: &Path) -> std::io::Result<PathBuf> {
    let mut existing = path;
    let mut missing = Vec::new();
    while !existing.exists() {
        let Some(file_name) = existing.file_name() else {
            return fs::canonicalize(existing);
        };
        missing.push(file_name.to_owned());
        let Some(parent) = existing.parent() else {
            return fs::canonicalize(existing);
        };
        existing = parent;
    }

    let mut resolved = fs::canonicalize(existing)?;
    for component in missing.iter().rev() {
        resolved.push(component);
    }
    Ok(normalize_path(&resolved))
}

fn normalize_path(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            _ => normalized.push(component.as_os_str()),
        }
    }
    normalized
}

fn format_bytes(bytes: u64) -> String {
    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * MIB;
    if bytes >= GIB {
        let hundredths = bytes.saturating_mul(100).saturating_add(GIB / 2) / GIB;
        format!("{}.{:02} GiB", hundredths / 100, hundredths % 100)
    } else {
        let tenths = bytes.saturating_mul(10).saturating_add(MIB / 2) / MIB;
        format!("{}.{:01} MiB", tenths / 10, tenths % 10)
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::{format_bytes, parse_fetch_options};

    #[test]
    fn fetch_dry_run_options_are_explicit() {
        let destination = std::env::temp_dir().join("logit-loom-models-cache");
        let options = parse_fetch_options(&[
            "minit2i-b16".to_owned(),
            "--dir".to_owned(),
            destination.to_string_lossy().into_owned(),
            "--dry-run".to_owned(),
        ])
        .expect("options should parse");
        assert_eq!(options.profile, "minit2i-b16");
        assert_eq!(options.destination, destination);
        assert!(options.dry_run);
        assert!(!options.accept_license);
    }

    #[test]
    fn byte_format_uses_binary_units() {
        assert_eq!(format_bytes(639_458_232), "609.8 MiB");
        assert_eq!(format_bytes(35_678_257_629), "33.23 GiB");
    }

    #[test]
    fn model_destination_inside_checkout_is_rejected() {
        let destination = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask should have a workspace parent")
            .join("local-models");
        let error = parse_fetch_options(&[
            "qwen3-0.6b-q8-0".to_owned(),
            "--dir".to_owned(),
            destination.to_string_lossy().into_owned(),
        ])
        .expect_err("repository-local model destinations must be rejected");
        assert!(error.to_string().contains("outside the repository"));
    }

    #[test]
    fn path_normalization_removes_parent_components() {
        assert_eq!(
            super::normalize_path(Path::new("/tmp/one/../two")),
            PathBuf::from("/tmp/two")
        );
    }
}

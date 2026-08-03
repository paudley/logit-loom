// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    ffi::{OsStr, OsString},
    fs::{self, File},
    io::{self, Read},
    path::{Path, PathBuf},
    process::Command,
};

use sha2::{Digest as _, Sha256};
use thiserror::Error;

use logit_loom_models::{Profile, Source};

const HASH_BUFFER_BYTES: usize = 1024 * 1024;

#[derive(Debug, Error)]
pub(crate) enum AcquireError {
    #[error("failed to create model directory {path}: {source}")]
    CreateDirectory { path: PathBuf, source: io::Error },
    #[error("failed to invoke the `hf` CLI: {0}")]
    Invoke(io::Error),
    #[error("`hf` failed while downloading {repository} with status {status}")]
    Download { repository: String, status: String },
    #[error("expected artifact is missing: {0}")]
    Missing(PathBuf),
    #[error("failed to inspect artifact {path}: {source}")]
    Metadata { path: PathBuf, source: io::Error },
    #[error("artifact {path} has {actual} bytes; expected {expected}")]
    Size {
        path: PathBuf,
        actual: u64,
        expected: u64,
    },
    #[error("failed to read artifact {path}: {source}")]
    Read { path: PathBuf, source: io::Error },
    #[error("artifact {path} has SHA-256 {actual}; expected {expected}")]
    Digest {
        path: PathBuf,
        actual: String,
        expected: String,
    },
}

pub(crate) fn fetch_profile(
    profile: &Profile,
    destination: &Path,
    dry_run: bool,
) -> Result<(), AcquireError> {
    let profile_root = destination.join(profile.id());
    for source in profile.sources() {
        let source_root = profile_root.join(source.local_subdir());
        let args = hf_download_args(source, &source_root);
        eprintln!("{}", display_command(&args));
        if dry_run {
            continue;
        }

        fs::create_dir_all(&source_root).map_err(|source| AcquireError::CreateDirectory {
            path: source_root,
            source,
        })?;
        let status = Command::new("hf")
            .args(&args)
            .status()
            .map_err(AcquireError::Invoke)?;
        if !status.success() {
            return Err(AcquireError::Download {
                repository: source.repository().to_owned(),
                status: status.code().map_or_else(
                    || "terminated by signal".to_owned(),
                    |code| code.to_string(),
                ),
            });
        }
    }

    if !dry_run {
        verify_profile(profile, destination)?;
    }
    Ok(())
}

pub(crate) fn verify_profile(profile: &Profile, destination: &Path) -> Result<(), AcquireError> {
    let profile_root = destination.join(profile.id());
    for source in profile.sources() {
        let source_root = profile_root.join(source.local_subdir());
        for artifact in source.files() {
            let path = source_root.join(artifact.path());
            if !path.is_file() {
                return Err(AcquireError::Missing(path));
            }
            let actual_bytes = fs::metadata(&path)
                .map_err(|source| AcquireError::Metadata {
                    path: path.clone(),
                    source,
                })?
                .len();
            if actual_bytes != artifact.bytes() {
                return Err(AcquireError::Size {
                    path,
                    actual: actual_bytes,
                    expected: artifact.bytes(),
                });
            }
            if let Some(expected) = artifact.sha256() {
                eprintln!("verifying SHA-256: {}", path.display());
                let actual = sha256_file(&path)?;
                if actual != expected {
                    return Err(AcquireError::Digest {
                        path,
                        actual,
                        expected: expected.to_owned(),
                    });
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn hf_download_args(source: &Source, destination: &Path) -> Vec<OsString> {
    let mut args = Vec::with_capacity(source.files().len() + 6);
    args.push("download".into());
    args.push(source.repository().into());
    args.extend(source.files().iter().map(|artifact| artifact.path().into()));
    args.push("--revision".into());
    args.push(source.revision().into());
    args.push("--local-dir".into());
    args.push(destination.as_os_str().to_owned());
    args
}

fn sha256_file(path: &Path) -> Result<String, AcquireError> {
    let mut file = File::open(path).map_err(|source| AcquireError::Read {
        path: path.to_owned(),
        source,
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; HASH_BUFFER_BYTES];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|source| AcquireError::Read {
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

fn encode_lower_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";

    let mut encoded = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        encoded.push(char::from(HEX[usize::from(*byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(*byte & 0x0f)]));
    }
    encoded
}

fn display_command(args: &[OsString]) -> String {
    std::iter::once(OsString::from("hf"))
        .chain(args.iter().cloned())
        .map(|argument| display_argument(&argument))
        .collect::<Vec<_>>()
        .join(" ")
}

fn display_argument(argument: &OsStr) -> String {
    let value = argument.to_string_lossy();
    if !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"-_./:@".contains(&byte))
    {
        return value.into_owned();
    }
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use logit_loom_models::Catalog;

    use super::{encode_lower_hex, hf_download_args};

    #[test]
    fn digest_bytes_encode_as_exact_lowercase_hex() {
        assert_eq!(encode_lower_hex(&[0x00, 0xab, 0xff]), "00abff");
    }

    #[test]
    fn download_command_uses_exact_files_and_revision_without_a_token() {
        let catalog = Catalog::embedded().expect("catalog should load");
        let profile = catalog
            .find_profile("qwen3-0.6b-q8-0")
            .expect("Qwen profile should exist");
        let source = &profile.sources()[0];
        let args = hf_download_args(source, Path::new("/models/qwen"));
        let args = args
            .iter()
            .map(|argument| argument.to_string_lossy())
            .collect::<Vec<_>>();

        assert_eq!(args[0], "download");
        assert_eq!(args[1], source.repository());
        assert!(
            source
                .files()
                .iter()
                .all(|file| { args.iter().any(|argument| argument.as_ref() == file.path()) })
        );
        assert_eq!(
            args.iter()
                .position(|argument| argument == "--revision")
                .and_then(|position| args.get(position + 1))
                .map(AsRef::as_ref),
            Some(source.revision())
        );
        assert!(!args.iter().any(|argument| argument == "--token"));
    }
}

#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0

set -euo pipefail

allow_dirty=false
if [[ "${1:-}" == "--allow-dirty" ]]; then
    allow_dirty=true
elif [[ $# -ne 0 ]]; then
    echo "usage: $0 [--allow-dirty]" >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_root}"

if [[ "${allow_dirty}" == false ]] && [[ -n "$(git status --porcelain)" ]]; then
    echo "release check requires a clean checkout" >&2
    exit 1
fi

cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --all-targets --locked
cargo test --workspace --doc --locked
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked

package_flags=(--locked)
if [[ "${allow_dirty}" == true ]]; then
    package_flags+=(--allow-dirty)
fi

cargo package -p logit-loom-core "${package_flags[@]}"
cargo package -p logit-loom "${package_flags[@]}" --list >/dev/null
cargo package -p logit-loom-llamacpp "${package_flags[@]}" --list >/dev/null

if rg -n --hidden --glob '!target/**' --glob '!.git/**' \
    --glob '!scripts/release-check.sh' \
    '(/home/|Active/|lillith|gmeow|math_gdrive)' .; then
    echo "possible internal reference found" >&2
    exit 1
fi

if rg -n --hidden --glob '!target/**' --glob '!.git/**' \
    --glob '!scripts/release-check.sh' \
    '(BEGIN (((RSA|OPENSSH|EC|PGP) )?PRIVATE KEY|PGP PRIVATE KEY BLOCK)|AKIA[0-9A-Z]{16}|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{36,}|glpat-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{30,}|npm_[A-Za-z0-9]{30,}|pypi-[A-Za-z0-9_-]{30,}|cio[A-Za-z0-9_-]{20,})' .; then
    echo "possible credential material found" >&2
    exit 1
fi

if rg --files --hidden --glob '!target/**' --glob '!.git/**' \
    --glob '*.gguf' --glob '*.safetensors' --glob '*.onnx' \
    --glob '*.pt' --glob '*.pth' --glob '*.bin' --glob '*.ckpt' | rg -q .; then
    echo "model artifact found in release checkout" >&2
    exit 1
fi

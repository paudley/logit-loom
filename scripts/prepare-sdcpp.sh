#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0

set -euo pipefail

upstream_url="https://github.com/leejet/stable-diffusion.cpp.git"
upstream_commit="ea4e566ccffa10f853ecc3f29e74b1820bc91beb"

usage() {
    echo "usage: $0 --source DIR --build DIR --backend vulkan|hip|cuda|metal" >&2
}

source_dir=""
build_dir=""
backend=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)
            source_dir="${2:-}"
            shift 2
            ;;
        --build)
            build_dir="${2:-}"
            shift 2
            ;;
        --backend)
            backend="${2:-}"
            shift 2
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

if [[ -z "${source_dir}" || -z "${build_dir}" || -z "${backend}" ]]; then
    usage
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
patch_files=(
    "${repo_root}/native/stable-diffusion.cpp/logit-loom-step-v1.patch"
    "${repo_root}/native/stable-diffusion.cpp/logit-loom-image-v2.patch"
)
source_dir="$(realpath -m -- "${source_dir}")"
build_dir="$(realpath -m -- "${build_dir}")"

case "${source_dir}/" in
    "${repo_root}/"*)
        echo "native source must be outside the Logit Loom checkout" >&2
        exit 2
        ;;
esac
case "${build_dir}/" in
    "${repo_root}/"*)
        echo "native build output must be outside the Logit Loom checkout" >&2
        exit 2
        ;;
esac
if [[ "${source_dir}" == "${build_dir}" ]]; then
    echo "native source and build directories must differ" >&2
    exit 2
fi

case "${backend}" in
    vulkan)
        backend_flag="-DSD_VULKAN=ON"
        ;;
    hip)
        backend_flag="-DSD_HIPBLAS=ON"
        ;;
    cuda)
        backend_flag="-DSD_CUDA=ON"
        ;;
    metal)
        backend_flag="-DSD_METAL=ON"
        ;;
    *)
        usage
        exit 2
        ;;
esac

created_checkout=false
if [[ ! -e "${source_dir}" ]]; then
    mkdir -p -- "$(dirname -- "${source_dir}")"
    git clone --filter=blob:none --no-checkout "${upstream_url}" "${source_dir}"
    created_checkout=true
fi
if [[ ! -d "${source_dir}/.git" ]]; then
    echo "native source is not a Git checkout: ${source_dir}" >&2
    exit 1
fi

actual_commit="$(git -C "${source_dir}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${upstream_commit}" ]]; then
    if [[ "${created_checkout}" == false ]] &&
        [[ -n "$(git -C "${source_dir}" status --porcelain)" ]]; then
        echo "refusing to change a dirty native checkout at ${actual_commit}" >&2
        exit 1
    fi
    git -C "${source_dir}" fetch origin "${upstream_commit}"
    git -C "${source_dir}" checkout --detach "${upstream_commit}"
fi

for patch_file in "${patch_files[@]}"; do
    if git -C "${source_dir}" apply -p0 --whitespace=nowarn --check "${patch_file}"; then
        git -C "${source_dir}" apply -p0 --whitespace=nowarn "${patch_file}"
    elif git -C "${source_dir}" apply -p0 --whitespace=nowarn --reverse --check "${patch_file}"; then
        echo "Logit Loom companion patch is already applied: $(basename -- "${patch_file}")"
    else
        echo "native checkout has changes incompatible with $(basename -- "${patch_file}")" >&2
        exit 1
    fi
done

git -C "${source_dir}" submodule update --init --depth 1 ggml
mkdir -p -- "${build_dir}"

cmake \
    -S "${source_dir}" \
    -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CCACHE=OFF \
    -DSD_BUILD_EXAMPLES=OFF \
    -DSD_BUILD_SHARED_LIBS=ON \
    -DSD_WEBM=OFF \
    -DSD_WEBP=OFF \
    "${backend_flag}"
cmake --build "${build_dir}" --target stable-diffusion --parallel

library="${build_dir}/bin/libstable-diffusion.so"
if [[ "${backend}" == "metal" ]]; then
    library="${build_dir}/bin/libstable-diffusion.dylib"
fi
if [[ ! -f "${library}" ]]; then
    echo "native build completed without the expected shared library: ${library}" >&2
    exit 1
fi

echo "built Logit Loom stable-diffusion.cpp companion: ${library}"

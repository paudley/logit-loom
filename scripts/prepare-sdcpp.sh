#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0

set -euo pipefail

# The upstream sources are vendored as plain files under
# native/stable-diffusion.cpp/upstream/ (see its README.md): no clone, no
# checkout, no submodule. Each tree is pinned by the SHA-256 of its sorted
# per-file sha256sum listing; the build refuses a tree that differs.
upstream_commit="2988060a4338a5dcd94721c964a8e87003c1ac8b"
upstream_tree_sha256="04e2de27a19f4924314d3ee48e23634fb4fd96e353a5769e76b85e262ac93a3f"

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
    "${repo_root}/native/stable-diffusion.cpp/logit-loom-program-v3.patch"
    "${repo_root}/native/stable-diffusion.cpp/logit-loom-model-block-v4.patch"
    "${repo_root}/native/stable-diffusion.cpp/logit-loom-model-block-application-v5.patch"
    "${repo_root}/native/stable-diffusion.cpp/logit-loom-krea-activation-v6.patch"
    "${repo_root}/native/stable-diffusion.cpp/logit-loom-resume-v7.patch"
    "${repo_root}/native/stable-diffusion.cpp/logit-loom-native-errors-v9.patch"
    "${repo_root}/native/stable-diffusion.cpp/logit-loom-image-strength-window-v15.patch"
    "${repo_root}/native/stable-diffusion.cpp/logit-loom-krea-reference-guard-v16.patch"
    "${repo_root}/native/stable-diffusion.cpp/logit-loom-vae-tiling-v20.patch"
    "${repo_root}/native/stable-diffusion.cpp/logit-loom-eager-load-v21.patch"
    "${repo_root}/native/stable-diffusion.cpp/logit-loom-upstream-parity-v23.patch"
    "${repo_root}/native/stable-diffusion.cpp/logit-loom-sealed-sources-v23.patch"
    "${repo_root}/native/stable-diffusion.cpp/logit-loom-igpu-capacity-v23.patch"
)
ggml_patches=(
    "${repo_root}/native/stable-diffusion.cpp/logit-loom-vulkan-budget-v8.patch"
    "${repo_root}/native/stable-diffusion.cpp/logit-loom-vulkan-errors-v10.patch"
    "${repo_root}/native/stable-diffusion.cpp/logit-loom-vulkan-strix-halo-v11.patch"
    "${repo_root}/native/stable-diffusion.cpp/logit-loom-vulkan-strix-halo-v12.patch"
    "${repo_root}/native/stable-diffusion.cpp/logit-loom-vulkan-strix-halo-v13.patch"
    "${repo_root}/native/stable-diffusion.cpp/logit-loom-vulkan-strix-halo-v14.patch"
    "${repo_root}/native/stable-diffusion.cpp/logit-loom-vulkan-dispatch-bounds-v18.patch"
    "${repo_root}/native/stable-diffusion.cpp/logit-loom-vulkan-uma-prefer-host-v19.patch"
)
ggml_commit="d25b121ce1d6ae0c7e62ad080460d9fc1e46dd93"
ggml_tree_sha256="0612d476324262787f1bf0eef37959cda29df93643a004805808be1333202549"
vendored_root="${repo_root}/native/stable-diffusion.cpp/upstream/stable-diffusion.cpp"
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

# SHA-256 over the sorted sha256sum listing of every regular file below $1,
# skipping the nested ggml tree when $2 is "without-ggml".
tree_sha256() {
    local root="$1"
    local scope="$2"
    (
        cd -- "${root}"
        if [[ "${scope}" == "without-ggml" ]]; then
            find . -path ./ggml -prune -o -type f -print0
        else
            find . -type f -print0
        fi | LC_ALL=C sort -z | xargs -0 sha256sum
    ) | sha256sum | cut -d' ' -f1
}

require_tree() {
    local label="$1"
    local root="$2"
    local scope="$3"
    local expected="$4"
    local observed
    if [[ ! -d "${root}" ]] || [[ -n "$(find "${root}" -type l -print -quit)" ]]; then
        echo "${label} tree is absent or contains a symbolic link: ${root}" >&2
        exit 1
    fi
    observed="$(tree_sha256 "${root}" "${scope}")"
    if [[ "${observed}" != "${expected}" ]]; then
        echo "${label} tree identity differs: expected ${expected}, observed ${observed}" >&2
        exit 1
    fi
}

require_tree "vendored stable-diffusion.cpp ${upstream_commit}" "${vendored_root}" without-ggml "${upstream_tree_sha256}"
require_tree "vendored ggml ${ggml_commit}" "${vendored_root}/ggml" all "${ggml_tree_sha256}"

# The source directory is rebuilt from the vendored trees on every run, so it
# never carries state from an earlier build. Only a directory this script
# created (marked) is ever removed.
marker=".logit-loom-sdcpp-source"
if [[ -e "${source_dir}" ]]; then
    if [[ ! -f "${source_dir}/${marker}" ]]; then
        echo "refusing to replace a native source directory this script did not create: ${source_dir}" >&2
        exit 1
    fi
    rm -rf -- "${source_dir}"
fi
mkdir -p -- "$(dirname -- "${source_dir}")"
# Outside any Git work tree, so git apply works on plain files and CMake's
# revision queries find no repository: the build reads nothing but these files.
if git -C "$(dirname -- "${source_dir}")" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "native source must be outside every Git work tree: ${source_dir}" >&2
    exit 1
fi
cp -R --preserve=mode -- "${vendored_root}" "${source_dir}"
require_tree "copied stable-diffusion.cpp" "${source_dir}" without-ggml "${upstream_tree_sha256}"
require_tree "copied ggml" "${source_dir}/ggml" all "${ggml_tree_sha256}"

for patch_file in "${patch_files[@]}"; do
    if ! (cd -- "${source_dir}" && git apply -p0 --whitespace=nowarn "${patch_file}"); then
        echo "vendored stable-diffusion.cpp does not accept $(basename -- "${patch_file}")" >&2
        exit 1
    fi
done
for ggml_patch in "${ggml_patches[@]}"; do
    if ! (cd -- "${source_dir}/ggml" && git apply -p0 --whitespace=nowarn "${ggml_patch}"); then
        echo "vendored ggml does not accept $(basename -- "${ggml_patch}")" >&2
        exit 1
    fi
done
touch -- "${source_dir}/${marker}"
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

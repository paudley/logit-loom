<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Vendored upstream sources

Plain-file exports, unmodified, that `scripts/prepare-sdcpp.sh` copies outside
this checkout, patches, and builds. No clone, checkout or Git submodule is
involved, and the Logit Loom patches are never applied here.

| Tree | Upstream | Commit | Export | Files | Tree SHA-256 |
| --- | --- | --- | --- | ---: | --- |
| `stable-diffusion.cpp/` (excluding `ggml/`) | https://github.com/leejet/stable-diffusion.cpp | `2988060a4338a5dcd94721c964a8e87003c1ac8b` | `git archive <commit> CMakeLists.txt LICENSE cmake include src thirdparty` | 250 | `04e2de27a19f4924314d3ee48e23634fb4fd96e353a5769e76b85e262ac93a3f` |
| `stable-diffusion.cpp/ggml/` | https://github.com/leejet/ggml (the submodule upstream pins at that commit) | `d25b121ce1d6ae0c7e62ad080460d9fc1e46dd93` | `git archive <commit> CMakeLists.txt LICENSE ggml.pc.in cmake include src ':(exclude)src/ggml-virtgpu/regenerate_remoting.py'` | 1401 | `0612d476324262787f1bf0eef37959cda29df93643a004805808be1333202549` |

The filter keeps exactly what the library build reads with examples, tests,
WebP and WebM off; the `libwebp` and `libwebm` submodules are not vendored.
ggml's `src/ggml-virtgpu/regenerate_remoting.py`, a code generator the build
never runs, is excluded because its header carries an agent attribution line
that this repository's publication guard refuses.
The tree SHA-256 is the SHA-256 of the `sha256sum` listing of every regular
file below the tree root, sorted bytewise by path (`find . -type f`, run from
the tree root, `LC_ALL=C sort`), which `prepare-sdcpp.sh` recomputes before
and after copying. Both projects are MIT licensed (`LICENSE` in each tree).

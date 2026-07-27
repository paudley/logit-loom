<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Optional model profiles

Logit Loom does not redistribute model weights. This directory contains an
exact, machine-checked acquisition catalog for three maintained experiment
profiles:

| Profile | Intended role | Exact download | Current integration status |
| --- | --- | ---: | --- |
| `qwen3-0.6b-q8-0` | Small text mechanics | 609.8 MiB | First-class; retained Vulkan acceptance passed |
| `minit2i-b16` | Small direct-RGB image mechanics | 3.88 GiB | First-class; retained Vulkan acceptance passed |
| `krea-2-turbo` | Advanced latent image mechanics | 12.36 GiB | First-class; retained Vulkan acceptance passed |

“Catalogued” means the repository pins exact upstream commits, file names,
byte counts, weight digests, license locations, and local layout. It does not
mean that a model-backed acceptance run has passed. That stronger status
requires the adapter, opt-in accelerator execution, and captured mechanical
receipts described in the
[model integration plan](https://github.com/paudley/logit-loom/blob/main/NEXT_STEPS.md).
The retained Qwen, `MiniT2I`, and Krea reports passed that execution gate and
their catalog entries are first-class. The checker requires every profile that
claims passed acceptance to have a matching retained passed report.

## Inspect and acquire

The repository-local command uses the current `hf` CLI. It never accepts a
token on the command line; normal Hugging Face authentication, including
`HF_TOKEN`, remains the CLI's responsibility.

```sh
cargo run --quiet -p logit-loom-xtask -- models check
cargo run --quiet -p logit-loom-xtask -- models list
cargo run --quiet -p logit-loom-xtask -- models fetch \
  qwen3-0.6b-q8-0 --dir /path/to/model-store --dry-run
cargo run --quiet -p logit-loom-xtask -- models fetch \
  qwen3-0.6b-q8-0 --dir /path/to/model-store
cargo run --quiet -p logit-loom-xtask -- models verify \
  qwen3-0.6b-q8-0 --dir /path/to/model-store
cargo run --quiet -p logit-loom-xtask -- models verify-artifact krea-2-turbo \
  krea-2-turbo-q6-k TURBO/Krea-2-Turbo-Q6_K.gguf \
  --path /path/to/an/existing/Krea-2-Turbo-Q6_K.gguf
```

`--dry-run` validates the catalog and prints exact `hf download` commands. It
does not invoke `hf` or create a directory. A real fetch downloads only the
listed files at the pinned 40-character revision, then verifies every byte
count and every available SHA-256 digest. An interrupted fetch can be resumed
by repeating the same command. The destination must resolve outside the
repository checkout so model artifacts cannot accidentally enter a release.

`verify-artifact` is the no-network path for a file already managed by another
local model store. It applies the same byte-count and SHA-256 checks and emits
a path-free JSON receipt.

The retained acquisition report in
[`reports/acquisition-2026-07-25.json`](reports/acquisition-2026-07-25.json)
uses a bounded, versioned Rust contract. It records the `hf` CLI version,
available filesystem bytes, catalog and verified artifact bytes, and exact
path-free receipts. All three profiles are fully verified there. Krea uses the
`mixed` acquisition method because its exact official license bytes and its
three runtime components were verified across a repository fetch and a
caller-managed model store rather than one shared directory.

Artifacts are placed under `<destination>/<profile>/<source>/`. For example,
the text GGUF is:

```text
<destination>/qwen3-0.6b-q8-0/model/Qwen3-0.6B-Q8_0.gguf
```

Krea 2 Turbo is gated and uses the upstream Krea 2 Community License. Read and
accept those terms on the
[model page](https://huggingface.co/krea/Krea-2-Turbo), authenticate `hf`, and
then acknowledge that prior action locally:

```sh
cargo run --quiet -p logit-loom-xtask -- models fetch krea-2-turbo \
  --dir /path/to/model-store \
  --accept-license
```

The flag does not accept upstream terms on the user's behalf.

The advanced profile uses a pinned `Q6_K` diffusion GGUF, a pinned `Q4_K_M`
Qwen3-VL text encoder, and the pinned Wan 2.1 VAE expected by the maintained
stable-diffusion.cpp integration. The exact Krea license PDF is verified as a
separate required artifact. Quantization changes storage and runtime mechanics;
this catalog does not make a quality comparison with the original Diffusers
weights.

## Trust and execution boundary

- No fetch runs during tests, CI, packaging, or documentation builds.
- The catalog permits only exact file paths and commit hashes; wildcard
  downloads are not used.
- Weight files require SHA-256 identities. Git-backed configuration files are
  bound by exact repository revision and byte count.
- Remote model code is forbidden. In particular, `MiniT2I`'s upstream custom
  Python pipeline is not downloaded or executed by this tooling. The
  maintained adapter uses the reviewed, pinned stable-diffusion.cpp companion
  described in
  [ADR 0001](https://github.com/paudley/logit-loom/blob/main/docs/adr/0001-stable-diffusion-runtime.md).
- Model files remain untrusted inputs to native or tensor runtimes. A matching
  digest establishes artifact identity, not safety, quality, or efficacy.
- Model and dependency licenses remain upstream terms. The Logit Loom dual
  license does not relicense them.

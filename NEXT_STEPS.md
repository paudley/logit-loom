<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Next steps: first-class model experiments

## Outcome

This phase makes three optional model profiles easy to acquire, mechanically
inspect, and use in complete experiments:

1. **Qwen3 0.6B Q8_0** for small text-token experiments.
2. **MiniT2I-B/16** for compact, direct-RGB image-generation experiments.
3. **Krea 2 Turbo** for advanced latent, conditioning, and schedule
   experiments.

“Offered” has a narrow definition: Logit Loom supplies a pinned model profile,
download instructions and tooling, a maintained adapter, at least one
snout-to-tail runbook, and an opt-in acceptance procedure. It does not bundle
weights, silently download them, promise support for arbitrary sibling
checkpoints, or claim a semantic or quality outcome.

The plan is complete. Qwen, MiniT2I, and Krea have passed their live
mechanical gates and are checked in with
`integration_status = "first-class"` and `acceptance_status = "passed"`.
Their path-free acquisition and acceptance reports bind the final catalog
identity without retaining model weights, prompts, outputs, or caller-local
paths.

## Why these three

| Profile | Mechanical value | Catalog footprint | License boundary |
| --- | --- | ---: | --- |
| Qwen3 0.6B Q8_0 | A small text model for exact token bytes, ordered logit transforms, stopping, checkpoints, and replay | 609.8 MiB | Apache-2.0 |
| MiniT2I-B/16 | A compact direct-RGB flow model whose iterative state can make image-step interventions relatively inspectable | 3.88 GiB including its FLAN-T5-Large encoder | MIT model profile plus Apache-2.0 encoder |
| Krea 2 Turbo Q6_K | A substantially larger multi-component stable-diffusion.cpp pipeline for conditioning, latent, scheduler, and checkpoint mechanics | 12.36 GiB | Gated Krea 2 Community License |

The exact footprints are catalog contents, not measured VRAM requirements.
Model-backed acceptance records device placement and supported tensor dtypes
for all three profiles, native image-step latency for MiniT2I and Krea, and
qualified host/device memory observations for the Krea run.

Qwen is the smallest download and reuses the existing llama.cpp adapter, so it
is the first acceptance lane. MiniT2I gives image experimentation an
approachable mechanical surface without pretending that image state is a token
vocabulary. Krea exercises the same framework at a scale and component shape
useful to more advanced downstream engines.

## Architectural boundary

The existing token contracts remain token contracts. Diffusion state is not
represented as fake token IDs or logits.

The shared layer is an experiment lifecycle:

```text
load exact artifacts
        |
initialize exact inputs and seed
        |
step -> observe -> optionally intervene
        |                 |
        +---- checkpoint -+
        |
finish -> artifact + mechanical receipt
```

Text and image adapters have typed state:

- Text exposes candidate logits before native sampling and arbitrary token
  bytes only after causal admission.
- Diffusion exposes a documented tensor boundary at an exact scheduler step,
  with shape, dtype, device, scheduler state, and conditioning identity.
- Checkpoints bind every artifact, adapter build, allocation, schedule, seed,
  step index, and opaque state byte identity required for conservative replay.
- Interventions are ordered, bounded, identified operations. State changes
  commit only after all stages succeed.
- Receipts record mechanics and lineage. They do not interpret an output or
  establish that an intervention was useful.

The reviewed spike established a common image-only contract, now implemented
in `logit-loom-diffusion`. It does not define a cross-modality experiment trait
or freeze a token-shaped abstraction around diffusion. Native tensor/runtime
details remain in `logit-loom-diffusion-sdcpp`.

## Artifact and trust policy

The repository never contains model weights and never downloads a model from a
test, CI job, package build, or documentation build.

[`models/profiles.json`](models/profiles.json) is versioned under the identity
domain `logit-loom-model-catalog-v1`. It records exact repositories, immutable
40-character revisions, selected file paths, byte counts, and SHA-256 digests
for weights. A changed revision, file set, or interpretation requires an
explicit catalog update and review.

Acquisition uses `hf download` with exact file names and revisions:

```sh
cargo run --quiet -p logit-loom-xtask -- models check
cargo run --quiet -p logit-loom-xtask -- models list
cargo run --quiet -p logit-loom-xtask -- models fetch \
  minit2i-b16 --dir /path/to/model-store --dry-run
```

The tool delegates authentication to the Hugging Face CLI, verifies local
artifacts after a real fetch, and never places a token in process arguments.
Gated profiles require prior acceptance on the upstream page plus an explicit
local acknowledgement. That acknowledgement records operator intent; it does
not alter or accept upstream terms automatically.

Remote repository code is not an execution dependency. MiniT2I's upstream
Diffusers example asks callers to trust custom remote Python code. Logit Loom
instead implements the reviewed, pinned adapter mechanics in maintained local
code.

## Delivery plan

### Phase 0 — catalog and acquisition foundation

Status: **complete; all catalog artifacts have exact verification receipts**

- [x] Pin three default profiles and every required source revision.
- [x] List exact files, byte counts, weight hashes, and upstream licenses.
- [x] Add bounded schema validation to repository checks.
- [x] Add opt-in `list`, `fetch`, `verify`, and no-network `--dry-run`
  commands.
- [x] Require explicit acknowledgement before a gated Krea fetch.
- [x] Verify the exact Qwen and MiniT2I artifacts and preserve path-free
  receipts.
- [x] Record measured MiniT2I storage, available filesystem bytes, and `hf`
  CLI version in a bounded acquisition report.
- [x] Verify Krea's three runtime components from a caller-managed store.
- [x] After the maintainer completes the upstream browser agreement, fetch and
  verify Krea's exact gated official license artifact.

Phase 0 makes acquisition reproducible. It does not establish adapter support.

### Phase 1 — text profile acceptance

Status: **complete; retained Vulkan acceptance passed**

- [x] Add a Qwen-specific profile loader over the existing llama.cpp runtime
  without hiding model path, tokenization flags, chat formatting, placement,
  or session allocation.
- [x] Validate exact profile identity before and after native loading and
  require accelerator placement before inference.
- [x] Run the existing fork-and-jolt, token-byte microscope, and exact-byte
  tripwire examples against the pinned GGUF.
- [x] Record model digest, backend build identity, device report, allocation,
  exact input bytes, plans, receipts, and output bytes.
- [x] Add a short profile runbook whose success criteria concern checkpoint replay,
  transform invocation, and receipt consistency—not output quality.

Acceptance gate: the pinned GGUF completes the runbook on an explicitly
selected accelerator backend, with no CPU-only fallback, and the retained
report passes its mechanical consistency checks.

### Phase 2 — MiniT2I acceptance spike

Status: **complete; model-backed Vulkan execution passed**

The narrow implementation spike:

- [x] inventories the tensor operations used by both pinned image profiles and
  records a tensor-runtime decision based on supported accelerator backends,
  dtype and operator coverage, safetensors loading, memory behavior,
  deterministic random state, callback containment, and maintenance cost;
- [x] loads only the pinned B/16 transformer and FLAN-T5-Large files through an
  exact maintained stable-diffusion.cpp companion, with no remote code
  execution;
- [x] binds prompt-tokenization output, seed, shape, dtype, device, schedule, and
  artifact identities;
- [x] exposes one exact post-Euler tensor boundary;
- [x] serializes exact state for authenticated deterministic-prefix replay at a
  selected step;
- [x] contains callback errors and unwinding before tensor state is committed;
  and
- [x] rejects non-finite, wrong-shape, wrong-device, and out-of-bound
  interventions.

The spike answers two design questions with code: which state is necessary for
exact replay, and which observation/intervention boundary is stable enough to
support publicly. An unpublished, reviewed local companion may be useful for
cross-checking numerical behavior, but a profile is not first-class if it
depends on mutable remote Python code or an unversioned process boundary.

Acceptance gate: two branches restored from one checkpoint remain identical
without an intervention; one bounded deterministic intervention produces a
different artifact identity and an internally consistent receipt. This says
nothing about whether either image is better.

### Phase 3 — diffusion contracts and toy runbook

Status: **complete; model-free validation and live MiniT2I Vulkan run passed**

- [x] Write the reviewed image-step contract using evidence from the spike.
- [x] Add bounded, serializable schedule, conditioning, checkpoint, intervention,
  and receipt shapes in a backend-neutral crate only where they are genuinely
  backend-neutral.
- [x] Keep tensor-runtime types and unsafe/native details in the adapter.
- [x] Implement transactional ordered interventions and post-step observation.
- [x] Add model-free tests for bounds, order, shape, non-finite values, callback
  errors and panics, partial write-back, checkpoint mismatch, and receipt
  accounting.
- [x] Publish a MiniT2I “fork the developing image” runbook: start from one exact
  seed, checkpoint at a chosen step, replay unchanged, apply one small bounded
  intervention, and compare per-step and final artifact identities.

Acceptance gate: model-free contracts pass the complete repository gate and
the opt-in MiniT2I runbook passes on the documented accelerator configuration.

### Phase 4 — Krea 2 Turbo integration

Status: **complete; retained Vulkan acceptance passed**

- [x] Review the pinned model layout and license obligations before implementation.
- [x] Map Krea's scheduler, text encoder, conditioning, transformer, latent, and
  VAE boundaries into the established diffusion contracts without weakening
  them.
- [x] Map supported artifacts and fail explicitly when the
  requested device placement or allocation cannot be honored.
- [x] Add an advanced runbook that checkpoints one latent trajectory and
  applies one declared bounded latent operation at a precise step. Changing
  conditioning or schedule remains an explicit plan mismatch rather than a
  weakened checkpoint.
- [x] Emit exact state dtype/placement and native denoiser-plus-Euler latency
  for every completed step as deployment facts outside deterministic
  identities.
- [x] On the authorized live Krea run, record peak host/device memory and bind
  the exact component, dtype, placement, and step-timing evidence in the
  retained acceptance report.

Acceptance gate: the pinned gated artifact completes the advanced runbook with
all components identified and no unreported fallback. License acceptance and
weights remain outside the repository.

### Phase 5 — first-class status and release gate

Status: **complete; all three profiles are first-class after their live gates**

A profile may move from `catalogued` to first-class only when:

- its pinned artifacts pass local verification;
- its maintained adapter and runbook are present;
- model-free contract tests and rustdocs pass;
- an opt-in accelerator acceptance report schema is checked in without model
  output or private paths;
- the profile's current license and access mechanism have been reviewed;
- README, capability, compatibility, and runbook documentation agree; and
- `scripts/release-check.sh` passes from a clean checkout.

Qwen, MiniT2I, and Krea satisfy those conditions. Repository validation now
rejects a first-class profile without passed acceptance status and rejects a
passed status without a matching retained passed report.

The complete release check passed in an isolated task-only clean snapshot on
2026-07-25. The snapshot deliberately excluded unrelated caller-local files
and did not depend on a Cargo alias or host configuration.

Artifact revisions are not updated implicitly. A revision bump starts a fresh
acceptance lane and retains the old report as lineage.

### Phase 6 — downstream bulk-tokenizer return

Status: **open; identity/cache/chunk/batch primitives exist locally, but the
SIMD BPE kernel and dedicated pool are not delivered**

- [ ] Record the exact Gigatoken `0.9.0`
  `0d9765fa7312af7534535e6315a5c49d74807b2a` import manifest and MIT notice.
- [ ] Land the pinned Rust SIMD BPE path with exact token-ID, special-token,
  normalization, Unicode, offset, and reconstruction parity against supplied
  engine oracles.
- [ ] Provide vector-free exact/threshold count and same-pass chunk planning.
- [ ] Provide a bounded caller-sized pool with no global Rayon state, helper
  process, network surface, Python, retry, or model fallback.
- [ ] Retain bounded collision-checked pretoken cache and reusable scratch
  capacity under exact identity and byte ceilings.
- [ ] Exercise whole/partitioned execution, every qualified batch/task shape,
  cold/warm cache, cancellation, eviction, multilingual/pathological inputs,
  and hostile bounds.
- [ ] Return one immutable revision and a clean model-free validation receipt
  for downstream qualification.

Downstream applications keep method selection, coalescing, resource
admission, accelerator leases, profile activation, and execution receipts
private. No alternate method is active merely because these public mechanics
compile.

## Initial experiments

Each profile lane makes one interesting experiment short to configure even
though the machinery underneath is substantial:

1. **Qwen — fork and jolt.** Checkpoint after exact prompt admission, replay the
   baseline, then add one ordered rank bias for one generation step. Inspect
   exact token bytes and pipeline receipts.
2. **MiniT2I — fork the developing image.** Checkpoint one deterministic
   direct-RGB trajectory, replay it unchanged, then apply a small bounded
   channel-local operation at one declared step. Compare state and final image
   digests.
3. **Krea — latent transplant.** Restore one latent checkpoint into two
   branches and apply one explicit bounded latent operation in the second.
   Inspect component identities, exact conditioning/schedule equality, step
   lineage, and the final artifact digest.

These are mechanics demonstrations, not benchmark tasks. Visual or textual
differences are observations; unchanged or subjectively poor output does not
by itself prove a contract failure.

## Maintenance boundary

Do not move any pinned artifact revision, companion ABI, backend compatibility
identity, serialized contract, or digest domain implicitly. A reviewed change
to one of those inputs starts a fresh compatibility and acceptance lane. Keep
downloads and model execution opt-in, retain only path-free mechanical
evidence, and repeat the clean release gate before publishing a coordinated
crate release.

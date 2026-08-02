<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# MiniT2I runbook: fork the developing image

Run one exact `MiniT2I-B/16` trajectory three ways: capture a post-Euler
checkpoint, replay it unchanged, then replay it with one bounded channel bias
at the checkpoint boundary. The example writes three caller-local PPM images
and a path-free JSON mechanics report. It does not interpret the images.

Source:
[`minit2i_fork.rs`](../../crates/diffusion-sdcpp/examples/minit2i_fork.rs)

## 1. Acquire the exact profile

Choose a model store outside the repository:

```sh
MODEL_STORE=/path/to/model-store
cargo run --quiet -p logit-loom-xtask -- models fetch \
  minit2i-b16 --dir "$MODEL_STORE"
cargo run --quiet -p logit-loom-xtask -- models verify \
  minit2i-b16 --dir "$MODEL_STORE"

DIFFUSION="$MODEL_STORE/minit2i-b16/model/minit2i-b-16/transformer/diffusion_pytorch_model.safetensors"
ENCODER="$MODEL_STORE/minit2i-b16/text-encoder/model.safetensors"
```

The fetch requests only the catalogued files at immutable revisions and
verifies the selected weight sizes and SHA-256 digests. No model acquisition
runs in tests, package builds, documentation builds, or CI.

## 2. Build and probe the exact companion

Use source and build directories outside the repository:

```sh
scripts/prepare-sdcpp.sh \
  --source /path/to/stable-diffusion.cpp-logit-loom \
  --build /path/to/stable-diffusion.cpp-logit-loom-build \
  --backend vulkan

LIB=/path/to/stable-diffusion.cpp-logit-loom-build/bin/libstable-diffusion.so

cargo run --quiet -p logit-loom-diffusion-sdcpp \
  --example probe_companion -- "$LIB" \
  | tee companion.json
```

The probe performs no model inference. It checks the shared-library digest,
companion ABI `2`, exact upstream commit, required symbols, and native device
report. Select the exact non-CPU device name from that report:

```sh
BACKEND=$(jq -r '
  .devices[]
  | select(test("vulkan"; "i"))
  | split("\t")[0]
' companion.json | head -n 1)
test -n "$BACKEND"
```

Use `--backend hip`, `cuda`, or `metal` instead when that is the deployment's
explicit accelerator. The Rust adapter rejects a CPU-named backend and never
retries on CPU.

## 3. Run the experiment

Choose a fresh output directory and an explicit native host thread count:

```sh
OUT=/path/to/minit2i-fork-output

cargo run --quiet -p logit-loom-diffusion-sdcpp \
  --example minit2i_fork -- \
  "$LIB" "$DIFFUSION" "$ENCODER" \
  "$BACKEND" 8 "$OUT" \
  "a clockwork cat beside a small brass loom" \
  | tee minit2i-fork.json
```

The example uses an exact 512 by 512 request, seed `7`, twelve-step custom
Euler schedule, and checkpoint after completed step `5`. The branch adds
`0.10` to channel zero at that boundary under a declared maximum absolute
delta of `0.25`. Those values are mechanics for this run, not tuned quality
claims.

Files are created without overwriting:

- `baseline.ppm`
- `replay.ppm`
- `branch.ppm`

## 4. Inspect the mechanics

```sh
jq -e '
  .profile_id == "minit2i-b16"
  and .integration_status == "first-class"
  and .passed
  and ([.checks[]] | all(. == "passed"))
  and .baseline.image == .replay.image
  and .baseline.image != .branch.image
  and (.baseline_measurements.step_latency_milliseconds | length == 12)
  and (.replay_measurements.step_latency_milliseconds | length == 12)
  and (.branch_measurements.step_latency_milliseconds | length == 12)
  and .intervention.invocations == 1
  and .intervention.failed_stage == null
  and any(
    .branch.steps[];
    .step_index == 5 and .elements_changed > 0
  )
' minit2i-fork.json
```

The unchanged replay must authenticate the deterministic prefix and reproduce
every post-step and final image identity. The intervened replay must commit
one complete transaction and change the final pixel-byte identity. A failure
is retained as a mechanical result; it is not papered over with a CPU retry.
Each measurement array records native denoiser-plus-Euler-update time
immediately before its corresponding callback. Timing is an execution fact,
not a replay invariant.

## Failure diagnosis

- **Artifact mismatch:** rerun
  `cargo run --quiet -p logit-loom-xtask -- models verify`; do not substitute
  a sibling checkpoint under this profile ID.
- **ABI or commit mismatch:** rebuild with `scripts/prepare-sdcpp.sh` from the
  pinned source revision.
- **No accelerator device:** fix deployment visibility outside Logit Loom.
  A CPU-only companion probe is compilation evidence, not model acceptance.
- **Checkpoint mismatch:** compare baseline/replay plan, native runtime,
  condition, schedule, seed, and step identities. Replay rejects any mismatch.
- **Existing PPM file:** choose a fresh output directory; the example does not
  overwrite an earlier result.
- **Branch image unchanged:** retain the failed assertion and report. A state
  change that quantizes to identical pixels does not satisfy this runbook's
  final artifact criterion.

No output image is committed as acceptance evidence. A retained acceptance
projection uses
[`model-run.schema.json`](../acceptance/model-run.schema.json) and contains
only identities, accounting, assertions, and deployment facts. Project the
latencies from one identified run into its optional `measurements` object;
do not combine the three trajectories into one synthetic timing series.

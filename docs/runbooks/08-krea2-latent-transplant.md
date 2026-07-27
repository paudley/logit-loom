<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Krea 2 runbook: latent transplant

Run one exact Krea 2 Turbo trajectory three ways: capture an Euler latent
boundary, replay it unchanged, then restore the same boundary and apply one
bounded channel-local operation. The checkpoint remains bound to the exact
conditioning and schedule; attempting to substitute either is rejected rather
than silently weakening replay identity.

Source:
[`krea2_latent_transplant.rs`](../../crates/diffusion-sdcpp/examples/krea2_latent_transplant.rs)

## 1. Acquire the pinned execution profile

```sh
MODEL_STORE=/path/to/model-store
cargo run --quiet -p logit-loom-xtask -- models fetch krea-2-turbo \
  --dir "$MODEL_STORE"
cargo run --quiet -p logit-loom-xtask -- models verify \
  krea-2-turbo --dir "$MODEL_STORE"
```

Select the exact catalogued components:

```sh
DIFFUSION="$MODEL_STORE/krea-2-turbo/diffusion/TURBO/Krea-2-Turbo-Q6_K.gguf"
ENCODER="$MODEL_STORE/krea-2-turbo/text-encoder/Qwen3VL-4B-Instruct-Q4_K_M.gguf"
VAE="$MODEL_STORE/krea-2-turbo/vae/split_files/vae/wan_2.1_vae.safetensors"
```

The adapter verifies every selected execution component before and after native
context loading. Legal documents, if present upstream, are not runtime inputs.

At this revision Krea is first-class with passed mechanical acceptance. The
example resolves that status from the packaged catalog. A future artifact,
adapter, or acceptance-domain revision must repeat the applicable gate rather
than inheriting this result.

## 2. Build and probe the companion

Follow the exact native preparation in the
[MiniT2I runbook](07-minit2i-fork.md#2-build-and-probe-the-exact-companion).
For example:

```sh
scripts/prepare-sdcpp.sh \
  --source /path/to/stable-diffusion.cpp-logit-loom \
  --build /path/to/stable-diffusion.cpp-logit-loom-build \
  --backend vulkan

LIB=/path/to/stable-diffusion.cpp-logit-loom-build/bin/libstable-diffusion.so
cargo run --quiet -p logit-loom-diffusion-sdcpp \
  --example probe_companion -- "$LIB" \
  | tee companion.json

BACKEND=$(jq -r '
  .devices[]
  | select(test("vulkan"; "i"))
  | split("\t")[0]
' companion.json | head -n 1)
test -n "$BACKEND"
```

The probe is model-free. A passing probe with only a CPU device is not Krea
acceptance.

## 3. Run the experiment

Choose a fresh output directory:

```sh
OUT=/path/to/krea2-latent-transplant-output

cargo run --quiet -p logit-loom-diffusion-sdcpp \
  --example krea2_latent_transplant -- \
  "$LIB" "$DIFFUSION" "$ENCODER" "$VAE" \
  "$BACKEND" 8 "$OUT" \
  "a clockwork cat beside a small brass loom" \
  | tee krea2-latent-transplant.json
```

The example uses an exact 1024 by 1024 request, seed `11`, four-step custom
Euler schedule, and checkpoint after completed step `1`. The branch adds
`0.20` to latent channel zero under a declared maximum absolute delta of
`0.50`. It writes `baseline.ppm`, `replay.ppm`, and `branch.ppm` without
overwriting.

## 4. Inspect the mechanics

```sh
jq -e '
  .profile_id == "krea-2-turbo"
  and .integration_status == "first-class"
  and .passed
  and ([.checks[]] | all(. == "passed"))
  and .baseline.image == .replay.image
  and .baseline.image != .branch.image
  and .baseline.profile == .branch.profile
  and .baseline.native.identity == .branch.native.identity
  and .baseline.plan.conditioning == .branch.plan.conditioning
  and .baseline.plan.schedule == .branch.plan.schedule
  and (.baseline_measurements.step_latency_milliseconds | length == 4)
  and (.replay_measurements.step_latency_milliseconds | length == 4)
  and (.branch_measurements.step_latency_milliseconds | length == 4)
  and .intervention.invocations == 1
  and any(
    .branch.steps[];
    .step_index == 1 and .elements_changed > 0
  )
' krea2-latent-transplant.json
```

The important evidence is exact component/runtime identity, an authenticated
unchanged replay, one committed intervention at the declared step, and final
artifact identities. Visual interpretation is outside this mechanics gate.

The report carries the exact state dtype and placement plus native
denoiser-plus-Euler-update latency for each completed step in all three runs.
The retained
[`Krea acceptance report`](../acceptance/krea-2-turbo-vulkan-2026-07-25.json)
projects the baseline timings, process `VmHWM`, and a system-wide AMD UMA
VRAM-plus-GTT peak sampled at 100 ms into the optional `measurements` object.
That device allocator observation is not process-attributed. Do not infer
memory values from catalog byte counts or combine the three timing series.

## Failure diagnosis

- **Artifact mismatch:** rerun
  `cargo run --quiet -p logit-loom-xtask -- models verify` and keep the
  catalog revision fixed.
- **Allocation or device rejection:** retain the explicit backend and device
  report. The adapter never silently offloads model inference to CPU.
- **Condition or schedule mismatch:** this checkpoint intentionally rejects
  changed plans. A future conditioning-transfer contract requires its own
  compatibility domain and review.
- **Branch image unchanged:** retain the failed result. A committed latent
  difference alone does not meet this runbook's final artifact criterion.

No weights, prompt bytes, output images, or caller-local paths belong in the
retained acceptance projection.

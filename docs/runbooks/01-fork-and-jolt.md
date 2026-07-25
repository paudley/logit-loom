<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Runbook 1: fork and jolt

Replay one prompt checkpoint twice. The first branch uses deterministic greedy
sampling. The second starts from the same native state and adds `4.0` to the
current runner-up logit before each sampling decision.

The experiment asks whether checkpoint lineage and transform execution are
visible and internally consistent. It does not assume the jolt will change the
selected token, and any changed text is not evidence that one branch is better.

Source:
[`fork_and_jolt.rs`](../../crates/runtime/examples/fork_and_jolt.rs)

## Inputs

- one caller-supplied local GGUF model;
- a prompt appropriate for that model's raw text interface; and
- an explicit native backend feature.

No chat template is added. If the model expects one, pass the exact formatted
text as the prompt.

## Run

From the repository root:

```sh
cargo run --quiet -p logit-loom-runtime --example fork_and_jolt \
  --features vulkan -- \
  /path/to/model.gguf "The creature opened its eyes and" \
  | tee fork-and-jolt.json
```

The example admits the prompt, captures one checkpoint, generates 32 greedy
tokens, restores the checkpoint, then generates up to 32 tokens through a
single full-vocabulary `RankBias` stage.

## Inspect

Confirm that both branches began at the captured causal position and the
pipeline completed without a failed stage:

```sh
jq -e '
  .checkpoint.position == .baseline.receipt.initial_position
  and .checkpoint.position == .jolted.receipt.initial_position
  and .transform.begins == 1
  and .transform.failed_stage == null
' fork-and-jolt.json
```

Inspect how much work the transform performed:

```sh
jq '{
  outputs_differ,
  finish: .jolted.receipt.finish,
  admitted_tokens: .jolted.receipt.admitted_tokens,
  transform_invocations: .transform.invocations,
  candidates_copied: .transform.candidates_copied,
  candidates_committed: .transform.candidates_committed,
  stage: .transform.stages[0]
}' fork-and-jolt.json
```

To retain the arbitrary output bytes without sending them to a terminal:

```sh
jq -r '.baseline.bytes_hex' fork-and-jolt.json | xxd -r -p > baseline.bin
jq -r '.jolted.bytes_hex' fork-and-jolt.json | xxd -r -p > jolted.bin
```

## Mechanical success criteria

- The checkpoint, baseline, and jolted branch agree on their starting causal
  position.
- The transform receipt has one begin, no failed stage, and equal copied and
  committed candidate counts.
- The jolted generation receipt refers to the digest of the reported transform
  receipt.
- Stage `accepted_tokens` equals the causally admitted jolted token count.

`outputs_differ` may be either `true` or `false`. The rank-one candidate needs
enough added logit to overtake the selected candidate before any branch can
diverge.

## Variations

- Try a smaller or larger `JOLT` constant to find a mechanical divergence
  threshold for one prompt.
- Change `JOLTED_RANK` to target a lower-ranked candidate.
- Replace `FullVocabulary` with a bounded sparse mode and compare receipt
  candidate counts.
- Add a second ordered stage and inspect per-stage `logits_changed` counts.

Each variation changes the exact pipeline identity, which makes reports from
different mechanics distinguishable.

## Failure diagnosis

- A model load rejection usually means the GGUF, selected feature, or device
  placement is incompatible with the deployment.
- A context-bound error means the admitted prompt plus the 32-token budget does
  not fit the configured session.
- A restore compatibility error means model bytes, native build, or session
  allocation no longer match the checkpoint contract.
- A successful run with identical branch bytes is not a failure; inspect
  `transform.stages[0].invocations` and `logits_changed` to confirm what ran.

The example writes no files itself. Remove the JSON and optional `.bin` files
when they are no longer needed.

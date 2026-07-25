<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Runbook 5: LoRA transplant

Capture one admitted prompt, generate a deterministic baseline, restore the
checkpoint, apply a caller-supplied LoRA for one branch, explicitly clear it,
then restore and replay the unsteered baseline.

The experiment tests steering lifecycle and checkpoint isolation. It does not
assume the LoRA changes the selected bytes, and it makes no claim about the
relative quality or meaning of either branch.

Source:
[`lora_transplant.rs`](../../crates/runtime/examples/lora_transplant.rs)

## Inputs

- one caller-supplied local GGUF model;
- one caller-supplied LoRA adapter compatible with that model;
- exact prompt text; and
- an explicit native backend feature.

The repository supplies neither model nor adapter. Review the origin and
license of both artifacts before using or sharing results.

## Run

```sh
cargo run --quiet -p logit-loom-runtime --example lora_transplant \
  --features vulkan -- \
  /path/to/model.gguf /path/to/adapter.gguf \
  "The creature opened its eyes and" \
  | tee lora-transplant.json
```

The example uses greedy sampling for three bounded 32-token generations. The
LoRA scale is exactly `1.0`. Prompt admission occurs before the checkpoint, so
all branches begin with the same causal token lineage.

## Inspect

Verify the steering lifecycle and restored replay:

```sh
jq -e '
  .steering_applied.action == "applied"
  and .steering_cleared.action == "cleared"
  and .steering_applied.resource == .steering_cleared.resource
  and .session_healthy_after_clear
  and .baseline_replay_matches
' lora-transplant.json
```

Compare the terminal accounting without decoding arbitrary bytes:

```sh
jq '{
  baseline_finish: .baseline.receipt.finish,
  steered_finish: .steered.receipt.finish,
  replay_finish: .replay.receipt.finish,
  steered_bytes_differ: (.baseline.bytes_hex != .steered.bytes_hex),
  baseline_replay_matches,
  session_healthy_after_clear,
  applied: .steering_applied,
  cleared: .steering_cleared
}' lora-transplant.json
```

## Mechanical success criteria

- Applied and cleared receipts name the same steering resource and record the
  expected lifecycle actions.
- Explicit cleanup succeeds and leaves the session healthy.
- Baseline and replay `GenerationOutput` values match exactly after restoring
  the same checkpoint, including bytes, token IDs, and generation receipt.
- All three generation receipts start from the checkpoint's causal position.

The steered output may equal the baseline for a particular prompt and bound.
That does not invalidate the lifecycle evidence.

## Variations

- Change `LORA_SCALE` in the example and observe the resulting resource
  identity and branch bytes.
- Repeat the run with several prompts while keeping model, adapter, backend,
  and allocation fixed.
- Attach the token byte microscope's observer to the steered branch to record
  its exact post-admission pieces.
- In a downstream application, persist the checkpoint bytes and receipt
  together, then restore only under the same model, backend build, and session
  allocation contract.

Do not compare different adapters as an efficacy study without a separately
defined workload, corpus, baselines, and analysis.

## Failure diagnosis

- Adapter load or application errors usually mean the adapter is incompatible
  with the loaded model or native backend.
- If explicit cleanup fails, the call returns an error and the session is
  poisoned instead of silently continuing with uncertain native state.
- A checkpoint compatibility error means the model bytes, backend build, or
  allocation contract differs from the captured state.
- A false `baseline_replay_matches` result should be retained with the full
  identities and investigated as a deterministic replay failure.

The example writes no files itself. Remove the JSON report when it is no
longer needed.

<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Runbook 4: causal circuit breaker

Place two observers in declared order. The first counts post-admission tokens
and cancels a shared signal on token `N`. The second is the built-in
cancellation observer; it sees that signal during the same token event and
requests a cooperative stop.

The experiment exposes exactly which causal work is retained when cancellation
originates at a post-admission boundary.

Source:
[`causal_circuit_breaker.rs`](../../crates/runtime/examples/causal_circuit_breaker.rs)

## Inputs

- one caller-supplied local GGUF model;
- exact prompt text;
- a breaker count in `1..=1024`; and
- an explicit native backend feature.

## Run

Stop after the fifth causally admitted generated token:

```sh
cargo run --quiet -p logit-loom-runtime --example causal_circuit_breaker \
  --features vulkan -- \
  /path/to/model.gguf "Count slowly from one:" 5 \
  | tee causal-circuit-breaker.json
```

The generation budget is `N + 16`, so the circuit breaker has room to fire
while remaining explicitly bounded.

## Inspect

```sh
jq '{
  requested_break_after,
  callback_observations,
  cancellation_requested,
  finish: .generation.receipt.finish,
  admitted_tokens: .generation.receipt.admitted_tokens,
  observer_receipts: .observers
}' causal-circuit-breaker.json
```

Validate either the intended stop or an earlier model terminal selection:

```sh
jq -e '
  if .cancellation_requested then
    .callback_observations == .requested_break_after
    and .generation.receipt.admitted_tokens == .requested_break_after
    and .generation.receipt.finish.kind == "observer_stop"
    and .observers[0].stop_requested == false
    and .observers[1].stop_requested == true
  else
    .generation.receipt.finish.kind == "end_of_generation"
    and .callback_observations < .requested_break_after
  end
' causal-circuit-breaker.json
```

## Mechanical success criteria

- If the model admits at least `N` generated tokens, the callback count and
  admitted token count are exactly `N`.
- The first observer reports no stop request; it only flips the cancellation
  signal.
- The second observer reports the stop at the same final causal position.
- The generation finish is `observer_stop`, and the `N`th token remains in
  causal state and exact output.

An end-of-generation selection before token `N` is a valid earlier terminal
condition. It is not delivered to either observer.

## Variations

- Run counts `1`, `2`, and `8` to make the retained-boundary behavior easy to
  compare.
- Reverse the observer order in a downstream copy. Cancellation then becomes
  visible at the next pre-sampling poll rather than the same token callback,
  demonstrating why declaration order matters.
- Give a cloned `CancellationToken` to another thread or application owner.
  Keep the `LoomSession` itself on its single owner thread.
- Replace the counter with a byte count or application event while preserving
  the cooperative stop boundary.

## Failure diagnosis

- Zero and counts above 1,024 are rejected before model loading.
- A finish of `token_limit` would indicate the breaker did not operate as
  declared and should be investigated; the example's bound is always larger
  than `N`.
- Cancellation is cooperative. It is checked at documented polls and
  post-admission observer delivery, not by interrupting a native operation in
  the middle.
- If a callback errors or unwinds, Rust contains it and returns an error rather
  than emitting a successful report.

The example writes no files itself. Remove the JSON report when it is no
longer needed.

<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Architecture

Logit Loom separates stable token-stream contracts from callback execution and
from a fast-moving model backend.

```text
application
  ├─ plans, token IDs, receipts ─────────────── logit-loom-core
  ├─ transforms, observers, cancellation ───── logit-loom
  └─ model, session, native sampling/state ──── logit-loom-llamacpp
                                                   │
                                                   └─ llama-cpp-4 → llama.cpp
```

This split lets another backend consume `logit-loom-core` and `logit-loom`
without importing llama.cpp types. Native ownership and compatibility churn
remain in the adapter.

## Candidate and sampling sequence

For each generation step, the llama.cpp adapter performs these mechanics in
order:

1. Poll generated-token observers at the pre-sampling boundary.
2. Copy the current raw vocabulary logits from the causal context.
3. Select the full vocabulary or a deterministic sparse top-ranked view.
4. Run Rust transform stages in declared order on the copied view.
5. Commit transformed candidates only if every stage succeeds.
6. Apply native grammar, logit bias, repetition/DRY penalties, probability
   filters, temperature, and the terminal sampler.
7. Treat an end-of-generation token as terminal without decoding it.
8. Obtain the selected token's exact bytes, decode the token into causal state,
   then notify transforms and observers of the admission.
9. Check exact byte stop suffixes after admission.

Sparse selection orders finite raw logits from highest to lowest and breaks
ties by ascending token ID. It is an exposure optimization, not a semantic
selection policy. All stages in one pipeline use the same exposure mode.

Repetition and DRY sampler state is initialized with the exact causal token
history. Grammar state begins at the first generated token, so prompt tokens do
not accidentally consume the output grammar. Every admitted generated token is
then accepted by the complete native sampler chain.

## Transactional transforms

A `Pipeline` copies its candidate view before invoking user code. If a stage
returns an error, exceeds its step bound, produces a `NaN`, or panics, no
candidate changes from that invocation are written back to the backend view.
The pipeline enters a failed state until the caller begins a new call.

Successful invocations use consecutive zero-based step values. This makes the
declared invocation bound enforceable even for a custom adapter and prevents a
caller from reusing an earlier step number to bypass it. Backend-selected
candidate views must contain unique token identifiers. A nonterminal
successful invocation must be matched by one causally admitted token before
the next transform step begins.

`apply_to_vocabulary` performs Logit Loom's deterministic full/sparse exposure
selection. `apply_to_candidates` instead accepts a view already selected and
copied by a backend adapter. In that form, the adapter is responsible for
proving complete full-vocabulary exposure or its native sparse-selection rule;
Logit Loom still enforces array shape, declared sparse bounds, stage limits,
containment, accounting, and transactional write-back to the supplied scratch
slice.

Implementation-local state in stages that ran before a later failure cannot be
rolled back. This is why retries require `begin`, which resets every stage and
starts new accounting.

## Exact-byte observation

Tokenizer pieces are byte slices. Individual pieces are not required to be
valid UTF-8, and observers receive them without conversion. `GenerationOutput`
can return the complete byte buffer as text only when the complete output is
valid UTF-8.

Observers run synchronously and in declared order. Every observer is polled at
the same boundary even when an earlier observer requests a stop. Token
callbacks occur only after native decode succeeds, so each observed token is
already part of causal state.

An observer set requires at least one successful poll before each delivered
token and rejects delivery beyond the call's requested token count. A stop
request is terminal for that observer call. Exact byte stop sequences are
checked after transform admission callbacks and generated-token observers; an
observer stop therefore takes precedence when both conditions occur on the
same admitted token. Stop bytes remain in output and causal state, and the
lowest declared stop index wins when several suffixes match.

## Causal sessions

A llama.cpp `Session` owns one mutable context and exact admitted token history.
It is deliberately neither `Send` nor `Sync`. Applications that need
concurrency should place a session inside a single-owner worker and communicate
with it through their own bounded channel or actor interface.

Prefill submits complete bounded chunks. A cooperative stop or callback error
retains every chunk that native decode already accepted. With `clear_first`,
the prior context is cleared before the first observer poll; this mutation is
part of the requested replacement operation.

## Checkpoints

A `StateSnapshot` combines opaque llama.cpp state bytes with exact token
history and a receipt. Restore verifies:

- the model-file content identity;
- the adapter build compatibility identity;
- the context, batch, micro-batch, and thread allocation contract;
- the state-byte identity;
- the token-history identity and causal position.

Snapshots wrap backend-owned state and are not a portable interchange format.
Capture and restore are rejected while a steering scope is active so the
checkpoint cannot omit required active steering state.

Applications that persist a snapshot choose their own container format.
`StateSnapshot::from_parts` revalidates the opaque byte count and identity,
token-history identity, and causal position before a later restore checks the
model and backend identities.

## Steering scopes

`LoRA` adapters and control vectors are applied through scopes that exclusively
borrow the session. Only one steering resource may be active. Explicit
`clear()` returns a lifecycle receipt; dropping a scope also attempts cleanup.

The safe upstream control-vector binding cannot pass a null slice to the native
clear operation. Logit Loom therefore validates a complete model-sized vector
and neutralizes it with an explicit all-zero vector on cleanup. This restores a
zero steering contribution while staying inside the safe binding.

If automatic `LoRA` removal, vector neutralization, or complete checkpoint
restore fails, the session is poisoned. Later mutation returns
`Error::Poisoned` rather than silently running with uncertain native state.
Callers can inspect `Session::is_healthy` and `Session::poison_reason`.

## Identities and receipts

Digest domains are explicit and versioned. A plan digest binds exact serialized
mechanics; a receipt binds exact accounting and lineage. Changing a serialized
shape or its interpretation requires a new versioned digest domain.

Receipts are not cryptographic signatures and make no statement about semantic
quality, truth, or efficacy. Applications may persist or sign them when they
need provenance beyond content identity.

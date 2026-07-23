<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Repository guidance for agents

## Purpose

Logit Loom is a public Rust toolkit for low-level token-stream mechanics. It
provides serializable contracts, ordered logit transforms, token observers,
bounded generation controls, checkpoint accounting, and a llama.cpp adapter.

Keep the project functionality-oriented. Code and documentation may describe
what a mechanism does, what boundary it runs at, and how it is validated. Do
not claim that a transform, sampler, prompt strategy, or steering method
improves cognition, model quality, safety, truthfulness, or any other semantic
outcome without a separately reviewed body of evidence. This repository does
not contain that research layer.

The toolkit must remain permissive about model behavior. Mechanisms are
caller-selected tools, not a policy system and not a framework for restricting
what a model may think, generate, or explore.

## Workspace layout

- `crates/core/`: backend-neutral, serializable plans and receipts.
- `crates/loom/`: safe Rust transform and observation runtime.
- `crates/llamacpp/`: llama.cpp integration through `llama-cpp-4`.
- `docs/`: architecture, capabilities, compatibility, and release policy.
- `scripts/`: deterministic repository checks.
- `.github/`: continuous integration and contribution templates.

Downstream applications and research systems are consumers. Do not copy their
private prompts, ontologies, models, paths, fixtures, or architectural names
into this public repository. Add integration logic downstream unless it is a
generally useful, documented token-stream primitive.

## Architectural invariants

Preserve these unless a design change is explicitly reviewed:

1. Token pieces are arbitrary bytes. Never assume one token is valid UTF-8.
2. Observers see tokens only after native causal admission.
3. End-of-generation tokens are terminal selections, not causal admissions.
4. Transform stages execute in declared order. Candidate logit changes commit
   only after every stage succeeds.
5. Callback errors and unwinding are contained in Rust before any native
   boundary can observe them.
6. A `Session` has one owner and is deliberately neither `Send` nor `Sync`.
7. Steering is explicitly scoped. Cleanup failure poisons the session instead
   of silently continuing with unknown native state.
8. Checkpoints are bound to model bytes and a conservative backend build
   identity. Native state bytes are opaque, not a portable file format.
9. Plans and receipts describe mechanics and lineage. They are not evidence of
   semantic correctness or efficacy.
10. Public inputs are bounded and validated before allocation or native calls.

The foundational crates forbid unsafe Rust. Keep native unsafety inside the
upstream binding unless there is no safe alternative; any proposed local
unsafe boundary requires a written safety contract, focused tests, and review.

## Development commands

The repository pins its minimum supported Rust toolchain in
`rust-toolchain.toml`.

```sh
make check-core
make check
make doc
make package-list
```

`make check-core` avoids compiling llama.cpp. `make check` compiles the default
adapter but performs no model inference. Never download a model in a test or
CI job.

Live model tests must be opt-in, use a caller-supplied local artifact, record
the backend feature and device placement, and never fall back to CPU inference
in this development environment. Shared accelerator use is normal; do not
require exclusive access. Compilation and host orchestration are not model
inference.

## Rust style and API design

- Run `cargo fmt`; the workspace denies missing public documentation.
- Keep `cargo clippy --workspace --all-targets -- -D warnings` clean.
- Prefer typed errors and checked conversions over panics or lossy casts.
- Keep public contracts small, serializable, and backend-neutral when possible.
- Put native types only in adapter crates.
- Use content identities for artifacts and exact execution contracts.
- State ordering, inclusivity, causal timing, and failure behavior in docs.
- Avoid convenience APIs that hide state mutation or native fallback.
- Do not expose raw pointers from the public API.

When changing a serialized shape or digest domain, treat it as a compatibility
change. Add a new domain/version rather than reinterpreting existing identities.

## Testing expectations

Every behavioral change needs a focused test at the lowest viable layer.
Important cases include:

- invalid and extreme bounds;
- deterministic ordering and tie-breaking;
- arbitrary non-UTF-8 token bytes;
- callback errors and panics;
- no partial logit write-back on failure;
- cooperative stops at exact documented boundaries;
- causal position and receipt consistency;
- steering application/cleanup and poisoned-session behavior;
- checkpoint identity mismatch;
- adapter compilation against the pinned binding.

Do not infer efficacy from these tests. They establish contract behavior only.
Model-backed acceptance fixtures belong in opt-in tests and must report their
artifact identities without committing model weights or outputs.

## Documentation and release hygiene

Public claims must match implemented, validated functionality. Keep
`README.md`, crate READMEs, `docs/capabilities.md`, and
`docs/compatibility.md` synchronized with code.

Before a release:

- confirm crate names and versions;
- review `CHANGELOG.md`;
- run `scripts/release-check.sh` from a clean checkout;
- inspect every `cargo package --list` result;
- verify package license metadata is `MIT OR Apache-2.0`;
- audit for internal paths, private artifacts, credentials, model files, and
  research or efficacy claims;
- publish foundational crates before dependent crates.

Do not publish, tag, push, or create a release unless the maintainer explicitly
asks for that external action.

## Licensing and contributions

Repository source is offered under `MIT OR Apache-2.0`; a separate proprietary
license is available from the copyright holder. Preserve the existing license
files and SPDX headers. Contributions are submitted under the inbound terms in
`LICENSING.md` and `CONTRIBUTING.md`; substantial contributions may require the
project CLA.

Do not add code with incompatible licensing, vendored model assets, generated
model output corpora, or copied native sources without an explicit dependency
and license review.

## Change discipline

Keep edits scoped and preserve unrelated work in a dirty checkout. Prefer
imperative commit subjects, for example `Add sparse transform receipts`. A
review handoff should state the mechanical behavior changed, compatibility
impact, validation performed, and any native feature required.

<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# logit-loom-runtime

An explicit higher-level local runtime for Logit Loom's llama.cpp adapter.

The crate shortens model loading, exact text admission, bounded generation,
transform and observer construction, checkpoint branching, and steering scopes.
It does not download models, choose a chat template, silently fall back to CPU,
assume generated token pieces are UTF-8, or make claims about output quality.

Use this crate for a direct local llama.cpp workflow. Use `logit-loom` when
building backend-neutral mechanics, or `logit-loom-llamacpp` when an
application needs to own the native runtime and model objects separately.

## Select a backend

No native backend feature is enabled by default. Select the deployment feature
explicitly:

```toml
[dependencies]
logit-loom-runtime = { version = "=0.2.0", features = ["vulkan"] }
```

`LoomOptions::default` continues to require accelerator participation. Native
logs are preserved unless `NativeLogPolicy::Silence` is selected.

[`Loom::load`] owns one initialized process runtime and one model. Applications
that need several models under a separately managed runtime should use the
lower-level adapter.

`Loom::load_qwen3_small` adds exact byte-count and SHA-256 verification for the
catalogued Qwen3 0.6B `Q8_0` GGUF. It still takes a caller path and
`LoomOptions`; it does not choose tokenization, chat formatting, placement, or
session allocation. `Loom::profile_artifact` returns a path-free verification
receipt for profiled loads.

## One bounded completion

The caller supplies a local GGUF, exact text, tokenization flags, and a nonzero
token bound:

```no_run
use logit_loom_runtime::{
    GenerationRequest, Loom, LoomOptions, Tokenization,
};

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let loom = Loom::load("model.gguf", LoomOptions::default())?;
    let output = loom.complete(
        "The creature opened its eyes and",
        Tokenization { add_bos: true },
        GenerationRequest::new(64)?,
    )?;
    std::io::Write::write_all(&mut std::io::stdout(), output.bytes())?;
    Ok(())
}
```

`CompletionOutput::bytes` is authoritative. Use `CompletionOutput::text` only
when the complete output is expected to be valid UTF-8.

The one-shot call creates a fresh session, replaces its state with the supplied
text, and drops the session after generation. It never applies a hidden system
prompt or chat template.

## Controlled generation

Built-in pipeline stages receive versioned identities that bind their exact
configuration. Custom callbacks still require a caller-defined stable digest:

```no_run
use logit_loom_runtime::{
    CandidateMode, GenerationRequest, Loom, LoomOptions, PipelineBuilder,
    Tokenization,
};

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let loom = Loom::load("model.gguf", LoomOptions::default())?;
    let mut pipeline = PipelineBuilder::new(CandidateMode::FullVocabulary, 32)?
        .rank_bias(1, 4.0)?
        .build()?;
    let request = GenerationRequest::new(32)?.pipeline(&mut pipeline)?;
    let _output = loom.complete(
        "The creature opened its eyes and",
        Tokenization { add_bos: true },
        request,
    )?;
    Ok(())
}
```

Transforms still run before native grammar, bias, penalties, filtering,
temperature, and terminal sampling. Candidate write-back remains transactional.
`GenerationRequest::bias` configures native post-transform logit bias;
`PipelineBuilder::token_bias` is an ordered Rust transform.

## Observe and stop

Generated-token callbacks run synchronously after native causal admission. An
observer receives exact token bytes and may request a cooperative stop:

```no_run
use logit_loom_runtime::{
    ControlFlow, Digest, GenerationRequest, Loom, LoomOptions,
    ObserversBuilder, Tokenization,
};

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let loom = Loom::load("model.gguf", LoomOptions::default())?;
    let mut observers = ObserversBuilder::new()
        .on_token(
            Digest::of_bytes("my-observer", b"stop-after-eight-v1"),
            |token| Ok(if token.position >= 8 {
                ControlFlow::Stop
            } else {
                ControlFlow::Continue
            }),
        )?
        .build()?;
    let request = GenerationRequest::new(32)?.observers(&mut observers)?;
    let _output = loom.complete(
        "Exact caller text",
        Tokenization { add_bos: true },
        request,
    )?;
    Ok(())
}
```

The position is the native causal position, not a generation-local counter.
Use [`CancellationToken`] with `ObserversBuilder::cancellation` when another
owner should signal the same documented safe boundaries.

## Stateful sessions and steering

`LoomSession` borrows its `Loom` and remains single-owner and synchronous.
Text replacement and append are separate methods, and tokenization flags remain
explicit. Checkpoint bytes retain the adapter's model/build/allocation identity
requirements.

```no_run
use logit_loom_runtime::{
    GenerationRequest, Loom, LoomOptions, Tokenization,
};

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let loom = Loom::load("model.gguf", LoomOptions::default())?;
    let mut session = loom.session()?;
    session.replace_text("A branching point:", Tokenization { add_bos: true })?;
    let checkpoint = session.capture_state()?;

    let _first = session.generate(GenerationRequest::new(16)?)?;
    session.restore_state(&checkpoint)?;
    let _second = session.generate(GenerationRequest::new(16)?)?;
    Ok(())
}
```

`LoomSession::lora` and `LoomSession::control_vector` return typed high-level
scopes. Call `clear` to observe cleanup and its receipt. Dropped scopes retain
the adapter's automatic cleanup and poisoning behavior. Checkpoint helpers are
intentionally unavailable on active steering scopes.

```no_run
use logit_loom_runtime::{
    GenerationRequest, Loom, LoomOptions, Tokenization,
};

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let loom = Loom::load("model.gguf", LoomOptions::default())?;
    let mut adapter = loom.load_lora("adapter.gguf")?;
    let mut session = loom.session()?;
    let mut steered = session.lora(&mut adapter, 0.75)?;
    steered.replace_text("Exact caller text", Tokenization { add_bos: true })?;
    let _output = steered.generate(GenerationRequest::new(16)?)?;
    let _cleanup = steered.clear()?;
    assert!(session.is_healthy());
    Ok(())
}
```

## Activation and speculative generation

The runtime re-exports the adapter's topology-bound activation and
target-authoritative speculation types without hiding their native ownership.
For same-model MTP, pass [`Loom::raw_runtime`] and [`Loom::raw_model`] to
[`generate_speculative`]. EAGLE-3 requires two models loaded under one
explicitly owned native [`low_level::llamacpp::Runtime`], so applications
should construct the lower-level runtime and both `Model` values directly.

No high-level call chooses tensor sites, mirrors a target activation program
onto a draft, changes context allocation, or falls back to ordinary
generation. The current speculative operation supports one sequence and does
not expose persistent checkpoint restore; see the adapter guide and
compatibility policy for the exact boundary.

## Identities and escape hatches

[`Loom::model_identity`], [`Loom::backend_identity`], and
[`Loom::backend_compatibility`] expose the identities used for checkpoint
compatibility. First-party pipeline and cancellation helpers receive
configuration-bound, versioned identities automatically. A custom transform or
observer requires a caller-supplied [`Digest`], because the crate cannot infer
the identity of arbitrary Rust code.

The [`low_level`] module re-exports the foundational crates for mechanics that
should remain fully explicit.

See the
[runtime interface guide](https://github.com/paudley/logit-loom/blob/main/docs/runtime-interface.md),
[mechanical experiment runbooks](https://github.com/paudley/logit-loom/blob/main/docs/runbooks/README.md),
[architecture](https://github.com/paudley/logit-loom/blob/main/docs/architecture.md),
and
[compatibility policy](https://github.com/paudley/logit-loom/blob/main/docs/compatibility.md)
for the complete ownership and release boundaries.

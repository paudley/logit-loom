<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# logit-loom-llamacpp

The safe llama.cpp adapter for Logit Loom.

It maps llama.cpp candidate logits and admitted tokens into the transform and
observer boundaries supplied by `logit-loom`. It also exposes causal prefill,
compatibility-bound checkpoints, and scoped `LoRA` and control vectors.

## Select a backend

The crate enables no llama.cpp backend feature by default. Select the feature
for the deployment explicitly:

```toml
[dependencies]
logit-loom = "=0.1.1"
logit-loom-llamacpp = { version = "=0.1.1", features = ["vulkan"] }
```

`ModelOptions::default` requires accelerator participation and does not retry a
rejected load as CPU-only inference. Use `DevicePolicy::Any` only when that
fallback is intentional.

## Example

The adapter never downloads a model. Supply a local GGUF explicitly:

```no_run
use logit_loom::{GenerationPlan, SamplingPlan};
use logit_loom_llamacpp::{
    Model, ModelOptions, Runtime, SessionOptions, Tokenization,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::path::Path::new("model.gguf");
    let mut runtime = Runtime::initialize()?;
    runtime.silence_native_logs();
    let model = Model::load(&runtime, model_path, ModelOptions::default())?;
    let prompt = model.tokenize("Hello", Tokenization { add_bos: true })?;
    let mut session = model.session(&runtime, SessionOptions::default())?;
    session.prefill(&prompt, true)?;

    let output = session.generate(
        &GenerationPlan {
            sampling: SamplingPlan::default(),
            max_tokens: 16,
            biases: Vec::new(),
            grammar: None,
            stops: Vec::new(),
        },
        None,
        None,
    )?;
    let _exact_bytes = output.bytes;
    Ok(())
}
```

Generated pieces remain arbitrary bytes. Call `GenerationOutput::text` only
when the complete output is expected to be valid UTF-8.
`Model::tokenize` rejects NUL bytes and inputs larger than
`MAX_TOKENIZATION_BYTES` before calling the native binding.

Checkpoints bind the model bytes, adapter build identity, and exact session
allocation options; native state is opaque and is not a portable interchange
format. A failed automatic steering cleanup or partial checkpoint restore
poisons the session, and subsequent mutation returns `Error::Poisoned`.

Applications may persist `StateSnapshot::into_parts` in their own container
format and reconstruct it with `StateSnapshot::from_parts`. Reconstruction
validates internal byte and token-lineage identities; restore additionally
requires the original model and compatible backend build.

See the [compatibility policy](https://github.com/paudley/logit-loom/blob/main/docs/compatibility.md)
and [capability status](https://github.com/paudley/logit-loom/blob/main/docs/capabilities.md)
before selecting features or interpreting validation results.

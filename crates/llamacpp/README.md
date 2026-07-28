<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# logit-loom-llamacpp

The safe llama.cpp adapter for Logit Loom.

It maps llama.cpp candidate logits and admitted tokens into the transform and
observer boundaries supplied by `logit-loom`. It also exposes causal prefill,
compatibility-bound checkpoints, topology-bound activation capture and
transactions, target-authoritative MTP/EAGLE-3 generation, and scoped `LoRA`
and control vectors.

## Select a backend

The crate enables no llama.cpp backend feature by default. Select the feature
for the deployment explicitly:

```toml
[dependencies]
logit-loom = "=0.2.0"
logit-loom-llamacpp = { version = "=0.2.0", features = ["vulkan"] }
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

## Activation and speculation

`ActivationConfiguration` validates exact tensor sites, bounded capture plans,
ordered operators, sparse vector banks, and the loaded model topology before
allocating a context. Graph-node selectors are bound to the exact adapter,
binding, llama.cpp revision, and architecture profile. A selected tensor is
copied into owned storage; one complete finite result is written back only
after every declared operator succeeds.

`generate_speculative` drives either same-model MTP or a separately loaded
EAGLE-3 draft. The supplied `SpeculationPlanV1` must name
`speculation_implementation_identity()`, both exact model/topology identities,
one supported mechanism, and an explicit activation policy. Before allocating
contexts, the adapter also applies llama.cpp HEAD's vocabulary-compatibility
rules, checks the MTP hidden-row width, and requires an `eagle3` draft with
exactly three in-range target extraction layers:

```no_run
use logit_loom::{
    GenerationPlan, SamplingPlan, SpeculationActivationPolicyV1,
    SpeculationPlanV1, TextSpeculativeMechanismV1,
};
use logit_loom_llamacpp::{
    Model, ModelOptions, Runtime, SpeculativeRequest, Tokenization,
    generate_speculative, speculation_implementation_identity,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = Runtime::initialize()?;
    let model = Model::load(&runtime, "mtp-model.gguf", ModelOptions::default())?;
    let prompt = model.tokenize("A small mechanical experiment:", Tokenization {
        add_bos: true,
    })?;
    let generation = GenerationPlan {
        sampling: SamplingPlan::default(),
        max_tokens: 16,
        biases: Vec::new(),
        grammar: None,
        stops: Vec::new(),
    };
    let topology = model.topology().digest()?;
    let speculation = SpeculationPlanV1 {
        target_model: model.artifact_digest().clone(),
        target_topology: topology.clone(),
        draft_model: model.artifact_digest().clone(),
        draft_topology: topology,
        implementation: speculation_implementation_identity(),
        mechanism: TextSpeculativeMechanismV1::Mtp,
        sequences: 1,
        maximum_draft_tokens: 4,
        minimum_draft_tokens: 0,
        probability_floor_bits: 0.0_f32.to_bits(),
        activation: SpeculationActivationPolicyV1::None,
    };
    let output = generate_speculative(
        &runtime,
        &model,
        &model,
        SpeculativeRequest::new(&prompt, &generation, &speculation),
    )?;
    let _exact_bytes = output.generation.bytes;
    Ok(())
}
```

The target sampler verifies the longest proposal prefix. Rejected proposals
and end-of-generation selections never reach token observers. Verification
captures remain provisional until rollback and native acceptance establish
their admitted/rejected positions. The current high-level operation supports
one sequence and requires explicit context headroom; larger sequence plans are
rejected before allocation rather than silently lowered.

The runnable `speculative_mtp` example takes one compatible caller-supplied
GGUF and writes exact generated bytes:

```sh
cargo run -p logit-loom-llamacpp --example speculative_mtp \
  --features vulkan -- /path/to/mtp-model.gguf "Draft from here:"
```

`generate_speculative_checkpointed` captures exact target, draft, and
MTP/EAGLE-3 implementation state at a quiescent boundary together with an
opaque target-sampler clone, activation configuration, cross-operation stop
state, and causal lineage. `resume_speculative_checkpointed` validates the
complete parent before allocation, clones the sampler for an independent
branch, restores contexts and implementation state, and captures the next
boundary.

The checkpoint object is process-local and thread-affine because llama.cpp
provides no portable target-sampler encoding. Its
`SpeculativeCheckpointReceiptV1` is serializable mechanical evidence, but the
receipt alone cannot reconstruct the native sampler. No partial portable
checkpoint is advertised.

## Coordinator integration

`Runtime::native_backend` and `Model::native_model` provide read-only borrows
for an in-process coordinator that must compose other llama.cpp mechanics,
such as multimodal projection or embeddings, with the same backend and loaded
model. Logit Loom retains backend and model lifecycle ownership. Calls made
through those handles are outside Logit Loom's plan and receipt surface; a
coordinator must give those calls their own bounded contract and evidence.

Checkpoints bind the model bytes, exact safe-binding source, literal llama.cpp
revision, adapter build identity, and exact session allocation options; native
state is opaque and is not a portable interchange format. A failed automatic
steering cleanup or partial checkpoint restore poisons the session, and
subsequent mutation returns `Error::Poisoned`.

Applications may persist `StateSnapshot::into_parts` in their own container
format and reconstruct it with `StateSnapshot::from_parts`. Reconstruction
validates internal byte and token-lineage identities; restore additionally
requires the original model and compatible backend build. Because the pinned
llama.cpp state format omits next-token logits, restore re-decodes the final
recorded token at its exact position after restoring causal memory. A backend
that cannot remove that one position is rejected, and a failed refresh poisons
the session.

See the [compatibility policy](https://github.com/paudley/logit-loom/blob/main/docs/compatibility.md)
and [capability status](https://github.com/paudley/logit-loom/blob/main/docs/capabilities.md)
before selecting features or interpreting validation results.

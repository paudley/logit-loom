# ADR 0004: Krea model-block residual operators

- Status: accepted; source implementation complete; model-backed acceptance
  pending
- Date: 2026-07-29

## Context

The public image-program contract already names exact model-block sites, but
the stable-diffusion.cpp resident adapter previously rejected every
`TensorSelector::ModelBlock`. Upstream skip-layer guidance does not solve this
gap: at the pinned stable-diffusion.cpp revision it reaches SD3- and
Flux-family runners, while the Krea runner receives no skip-layer state.

The official Krea Turbo configuration distinguishes 28 main transformer
layers from 12 selected text-encoder layers. Those are different axes. The
adapter must discover the main-block count from the loaded weights and must
not encode either the number 28 or a purported semantic role for any block in
the public operator.

Sources:

- [official Krea Turbo transformer configuration](https://huggingface.co/krea/Krea-2-Turbo/blob/main/transformer/config.json);
- [official Krea 2 inference source](https://github.com/krea-ai/krea-2);
- [pinned stable-diffusion.cpp revision](https://github.com/leejet/stable-diffusion.cpp/tree/ea4e566ccffa10f853ecc3f29e74b1820bc91beb).

## Decision

`logit-loom-diffusion-sdcpp` installs
`ModelBlockResidualScaleControlV1` for this exact selector shape:

```text
ModelBlock {
    component: "krea2",
    block: <zero-based loaded block index>,
    site: "residual",
}
```

The fixed eight-byte control body contains the exact IEEE-754 residual scale
and its caller-declared positive absolute bound. Both values are finite, the
declared bound is at most 16, and the scale must fit inside it.

For one selected Krea block with input `x` and ordinary output `f(x)`, the
installed graph computes:

```text
x + scale * (f(x) - x)
```

A scale of zero bypasses the block without executing it. A scale of one
preserves the ordinary graph. Other admitted scales attenuate or extrapolate
the complete block residual. This is a mechanical graph operation, not a
semantic claim about what the block represents.

`StepSelector::All` applies the control to every denoising transition.
`StepSelector::Exact` applies it only while computing the transition whose
zero-based completed boundary has the selected index. Step lists remain
canonical, bounded, unique, and in range.

The native model-block extension is ABI version 4 layered over the existing
ABI-v3 resident arena. Rust lowers only installed Krea residual operators into
the v4 call. Scheduler-state operators continue through the transactional
post-Euler pipeline. The native boundary:

- accepts at most 64 model-block invocations;
- rejects unknown components, sites, selectors, non-finite or out-of-bound
  scales, and overlapping controls for the same block and step;
- dynamically queries `Krea2Config::layers`, which stable-diffusion.cpp
  detects from the loaded `blocks.<index>` tensor namespace;
- rejects every block index outside that loaded topology;
- applies the same selected graph to all conditioning passes in one denoising
  transition; and
- clears request-local controls on every returned success or failure path.

The implementation does not download a model, start a listener, add a content
policy, or claim that any block controls safety, refusal, style, identity, or
image quality.

## Validation

Default-built Rust tests cover exact control bytes and identities, scalar
bounds, and rejection of uninstalled sites. The complete companion patch
applies after ABI v3 and compiles as a shared library from the pinned upstream
revision without loading weights.

Model-backed acceptance remains opt-in. It must bind exact model artifacts,
backend build, device placement, seed, schedule, prompt identity, selected
blocks and steps, output identity, and cleanup result. A useful semantic
effect requires a controlled differential study; a compiled graph or valid
image proves only mechanics.

## Consequences

Public consumers can now express exact Krea block bypass, attenuation, and
amplification through the existing serializable operator contract. Private
applications remain responsible for choosing experimental blocks and judging
their effects. Future model families or tensor sites require separately
installed component/site identities and native topology validation.

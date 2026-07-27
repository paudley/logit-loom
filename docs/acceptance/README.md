<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Model-backed acceptance reports

[`model-run.schema.json`](model-run.schema.json) defines the retained,
path-free projection of one opt-in model run. It records exact artifact,
runtime, plan, receipt, output, assertion, and blocker identities. It cannot
record model weights, prompt or output bytes, caller-local paths, or an
interpretation of generated content.

Experiment examples may emit a richer transient report for local diagnosis.
Before a profile status changes, the maintainer projects that evidence into
this schema, validates it with a JSON Schema 2020-12 validator, reviews it for
private data, and retains it alongside the exact catalog revision. A `passed`
report requires an explicitly selected accelerator and `cpu_fallback: false`.
A blocked report names the external condition without turning partial
compilation or artifact verification into model acceptance.

Reports establish mechanics for one exact artifact/runtime combination. They
do not establish model quality, safety, truthfulness, or efficacy.

## Retained reports

- [`qwen3-0.6b-q8-0-vulkan-2026-07-25.json`](qwen3-0.6b-q8-0-vulkan-2026-07-25.json)
  records an exact checkpoint replay on the first-class Qwen text profile using
  the Vulkan backend. It contains identities and accounting only.
- [`minit2i-b16-vulkan-2026-07-25.json`](minit2i-b16-vulkan-2026-07-25.json)
  records exact unchanged replay plus one bounded state intervention on the
  first-class MiniT2I image profile using the Vulkan backend, including one
  identified run's native per-step deployment timings.
- [`krea-2-turbo-vulkan-2026-07-25.json`](krea-2-turbo-vulkan-2026-07-25.json)
  records exact unchanged replay plus one bounded latent intervention on the
  first-class Krea image profile using the Vulkan backend. It also retains one
  identified run's native per-step timings, process `VmHWM`, and a qualified
  system-wide AMD UMA VRAM-plus-GTT peak sampled at 100 ms; the latter is not
  process-attributed.

<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Worker-local image execution

Logit Loom separates a serializable image program from the process that owns a
resident native model. The public crates provide mechanics only: they do not
define a queue, transport, scheduler, model store, admission policy, or
semantic evaluation.

## Layers

- `logit-loom-executor` defines borrowed input buffers, caller-owned output
  allocations, cooperative cancellation, explicit lifecycle state, cleanup
  receipts, and failure dispositions.
- `logit-loom-diffusion` defines versioned whole-image plans and receipts over
  exact buffer identities. A plan can describe text-to-image, image-to-image,
  inpaint, outpaint, direct VAE encode/decode, ordered LoRA schedules,
  installed tensor operators, and batched observations.
- `logit-loom-diffusion-sdcpp` implements the currently available native
  subset over a pinned stable-diffusion.cpp companion. It remains a
  single-owner, synchronous, in-process adapter.

The generic plan is intentionally broader than one adapter. A lowerer must
reject any mechanic it cannot implement exactly; silently approximating a
schedule, target selector, tensor site, format, or cleanup boundary is a
contract failure.

## stable-diffusion.cpp image ABI v2

The image extension is layered over companion ABI v1 and is required by safe
adapter contract v3. It provides:

| Mechanic | Current adapter behavior |
| --- | --- |
| Text-to-image | Exact prompt, optional negative prompt, seed, RGB8 geometry, guidance, and custom Euler sigmas |
| Image-to-image | Exact tightly packed RGB8/RGBA8 source at output geometry plus a fixed request strength |
| Inpaint/outpaint | Exact source and Gray8 mask at output geometry; canvas expansion remains caller-owned |
| References | Up to 16 tightly packed RGB8/RGBA8 images in declared order |
| LoRA | Up to 32 caller-verified local artifacts with one fixed scale per complete request |
| VAE encode/decode | Direct finite host-tensor exchange for the Krea profile with rank and element bounds |
| Cancellation | Cooperative stop after an exact completed Euler transition |
| Output | One validated, tightly packed RGB8 image written to caller-owned storage or a synchronous sink |

Every requested LoRA must match and participate in at least one model tensor
before native success is reported. The companion clears the request-local
stack before a normal, stopped, callback-error, or unsupported return. A
native exception or unconfirmed cleanup poisons the Rust session, which must
then be replaced.

Image ABI v2 does not implement per-step LoRA schedules, model-specific LoRA
target selectors, arbitrary model-block operators, PNG encoding, or a
multi-operation image graph. The backend-neutral contracts can represent
those mechanics so another adapter or a future ABI can implement them under a
new reviewed contract version. The existing full-state `StepProgram` path
continues to support scheduler-state interventions and checkpoint experiments;
it is distinct from the lower-copy whole-image path.

## Integrating a local worker

A downstream worker should:

1. verify artifact bytes and bind them to an exact load identity;
2. create one `Sdcpp` owner on an explicitly selected accelerator;
3. validate the serialized plan and every buffer identity before reading
   storage or entering native code;
4. lower only the supported subset and keep all borrowed storage alive for the
   synchronous call;
5. retain exact output and mechanical receipt identities;
6. call `clear_session` at the end of the downstream request scope; and
7. discard the owner after any error classified as `Poisoned`.

`Rejected` means the request was not accepted and resident state remains
known. `Cancelled` means a documented cooperative boundary was reached.
`Poisoned` means native or cleanup state is uncertain. These classifications
describe reuse mechanics, not retry policy; retries and admission remain
application decisions.

No adapter call downloads a model, starts a server, opens a network service,
or falls back to CPU inference. Model-backed execution is opt-in and requires
caller-supplied artifacts plus an explicit non-CPU device.

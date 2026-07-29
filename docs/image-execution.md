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
- `logit-loom-diffusion` preserves version-one whole-image plans and receipts
  and adds `ImageExecutionPlanV3`/`ImageExecutionReceiptV3` under new digest
  domains. Version two binds checkpoint restore/capture, an ordered
  deterministic compositing graph, explicit output routes, and request-scope
  cleanup. The separately versioned `ImageProgramPlanV1` family describes
  multiple resident native/VAE stages over typed single-assignment values.
- `logit-loom-diffusion-sdcpp` implements a strict version-two subset over a
  pinned stable-diffusion.cpp companion. `ImagePlanExecutor` is a
  single-owner, synchronous, in-process `LocalExecutor`.

The generic plan is intentionally broader than one adapter. A lowerer must
reject any mechanic it cannot implement exactly; silently approximating a
schedule, target selector, tensor site, format, or cleanup boundary is a
contract failure.

## Resident program preflight

`ImageProgramPlanV1` is built in the default diffusion crate with no feature or
configuration switch. Before allocation or native execution it validates:

- canonically numbered values and stages with exactly one producer per value;
- external or earlier-stage-only references and complete value consumption;
- native operation roles, canvas-bound source/mask geometry, independently
  sized bounded RGB8/RGBA8 reference geometry, tensor representation,
  schedules, scheduled `LoRA` values, checkpoint state, and snapshot output
  types;
- deterministic RGB8 join inputs and checkpoint conversion compatibility;
- unique non-aliasing output allocations, one final program-receipt route, and
  non-routable mutable checkpoint state; and
- liveness-derived releases under the public 2 GiB arena ceiling.

`ImageProgramReceiptV1` binds the exact plan, backend, runtime epoch, completed
stage prefix, produced value identities, output prefixes, terminal boundary,
and cleanup disposition. `ImageProgramMeasurementsV1` separately reports
stage timing, placement, transfers, and observed peak arena bytes so those
deployment observations do not alter deterministic plan or receipt identity.

`SdcppResidentProgram` is the default-built stable-diffusion.cpp backend for
this contract. Program ABI v3 is mandatory at library load: there is no
runtime switch that removes the capability. Ordinary tests validate the graph,
lowering, arena driver, receipts, and failure paths model-free; the retained
companion build validates the exact C++ patch without loading a model. Live
execution remains a separate opt-in acceptance boundary.

Routed native values copy directly into their caller-owned output allocations.
The driver verifies those uncommitted prefixes, confirms cleanup, and only
then publishes their initialized lengths; it does not stage a second complete
image or tensor allocation.

## Whole-plan order

For an accepted `ImageExecutionPlanV3`, the stable-diffusion.cpp executor
performs these mechanics in order:

1. Validate the complete graph, resident profile/load/RNG/placement identities,
   every input identity and byte layout, every output allocation, and the
   adapter-supported subset.
2. Decode and authenticate an optional checkpoint envelope without mutating
   native state.
3. Enter one native diffusion call with exact prompt, optional negative
   conditioning, source/mask/reference images, fixed seed and schedule, and a
   fixed whole-request `LoRA` stack.
4. At each completed post-Euler boundary, restore or capture the declared
   checkpoint before applying installed scheduler-state channel bias.
5. Record scheduler-state digest/statistics observations, then poll
   cancellation. A stop therefore names an exact completed, observed
   transition.
6. Decode one RGB8 primary image, execute ordered integer `MaskBlend`
   compositing stages, and preflight every RGB8/checkpoint route.
7. Retain the known session or confirm `clear_session`, as the plan requests.
8. Only after successful cleanup accounting, initialize the caller-owned
   output prefixes and return one whole-plan receipt.

Checkpoint receipts conservatively bind the native diffusion plan, backend
identity, continuation position, and exact finite little-endian state bytes.
Restore replays the deterministic prefix and fails on a plan, backend,
position, shape, or state mismatch before checkpoint bytes are committed.
Checkpoint envelopes are adapter-local, versioned, and authenticated; they are
not a portable model format.

The deterministic compositor validates all input/output lengths before its
first write. All route payloads are preflighted before cleanup, and route
writes occur only after cleanup succeeds. Callback failures and unwinding are
contained by the existing transactional full-state path.

## Resident-program support matrix

Program ABI v3, model-block ABI v4, and safe adapter contract v6 provide this
exact resident surface:

| Mechanic | Resident adapter behavior |
| --- | --- |
| Native stages | Multiple ordered text-to-image, image-to-image, inpaint, outpaint, direct VAE encode, and direct VAE decode stages |
| Inputs | Exact UTF-8 conditioning, canvas-bound tight RGB8/RGBA8 sources and Gray8 masks, independently sized bounded tight RGB8/RGBA8 references, finite dimension-zero-fastest `f32` tensors, authenticated checkpoints, and caller-retained verified `LoRA` descriptor paths |
| LoRA | Ordered request-local fixed or scheduled scales at exact pre-denoiser boundaries; every adapter must participate in a native model tensor |
| Checkpoint | Authenticated restore/capture with exact runtime compatibility and post-Euler boundary identity |
| Operator | Installed scheduler-state channel bias plus Krea residual block scaling at all or exact selected transitions; uninstalled model components/sites and conditioning selectors fail during whole-program preflight |
| Observation | Scheduler-state digest, statistics, and exact selected-boundary snapshots |
| Compositing | Ordered deterministic tight-RGB8 mask blends |
| Output | Individually typed RGB8, RGBA8, bounded PNG (RGB or RGBA plus encoder identity), tensor, checkpoint, snapshot, and final program-receipt routes |
| Cancellation | Before start, between stages, or after one exact completed post-Euler boundary |
| Cleanup | Generation-checked value release, retained or confirmed-clear disposition, epoch advancement, and poisoning on uncertainty |
| Measurements | Per-stage wall/native time, arena peak, placement, and transfers outside deterministic identity |

The adapter rejects unsupported selector semantics before the arena begins.
Krea block indices are checked against the topology detected from the loaded
weights. It does not reinterpret opaque conditioning, infer semantic block
roles, invent additional hook sites, approximate a scheduled adapter, select a
seed for the coordinator, or fall back to another backend.

## Single-primary stable-diffusion.cpp support matrix

The image extension is layered over companion ABI v1, is required by safe
adapter contract v6, and provides the following exact legacy subset:

| Mechanic | Current adapter behavior |
| --- | --- |
| Text-to-image | Exact prompt, optional negative prompt, seed, RGB8 geometry, guidance, and custom Euler sigmas |
| Image-to-image | Exact tightly packed RGB8/RGBA8 source at output geometry plus a fixed request strength |
| Inpaint/outpaint | Exact source and Gray8 mask at output geometry; canvas expansion remains caller-owned |
| References | Up to 16 tightly packed RGB8/RGBA8 images in declared order |
| LoRA | Up to 32 caller-verified local artifacts with one fixed scale per complete request and one of two exact whole-model target identities |
| Checkpoint | One optional authenticated restore and one optional post-step capture, routed as a versioned opaque envelope with at most 16,777,216 `f32` state elements |
| Operator | Ordered installed scheduler-state channel bias at `All` or exact selected steps |
| Observation | Scheduler-state digest or numerical-statistics lineage; snapshots are rejected |
| Cancellation | Cooperative stop after the exact post-operator observation boundary of a completed Euler transition |
| Compositing | Up to 32 ordered tight-RGB8 integer mask blends within a 512 MiB aggregate internal scratch bound |
| Output | Explicit tight-RGB8 image routes and at most one final captured-checkpoint route into caller-owned storage; an unreached cancellation boundary leaves that final allocation uninitialized |
| Cleanup | Explicit retain or confirmed clear disposition before output initialization |
| Direct VAE | Bounded Krea encode/decode remains available through `Sdcpp`, outside `ImageExecutionPlanV3` |

Every requested `LoRA` must match and participate in at least one model tensor
before native success is reported. The companion clears the request-local
stack before a normal, stopped, callback-error, or unsupported return. A
native exception or unconfirmed cleanup poisons the Rust session, which must
then be replaced.

The older whole-plan lowerer rejects per-step `LoRA` schedules, model-block or
conditioning operators, snapshot observations, PNG/RGBA/tensor output routes,
multiple native inference operations, and version-two direct VAE execution.
Krea residual block controls are available only through the resident program
surface; unsupported model components, tensor sites, and control schemas are
rejected instead of approximated.

## Integrating a local worker

A downstream worker should:

1. verify artifact bytes and bind them to an exact load identity;
2. create one `Sdcpp` owner on an explicitly selected accelerator;
3. construct one version-two plan and exact ordered input/output bindings;
4. provide an `ArtifactPathResolver` only when fixed `LoRA` inputs need a
   confined synchronous descriptor path;
5. call `ImagePlanExecutor::execute` and retain the plan, receipt, output, and
   checkpoint identities;
6. honor `Rejected`, `Cancelled`, and `Poisoned` failure dispositions; and
7. discard the owner after any `Poisoned` result.

`Rejected` means the request was not accepted and resident state remains
known. `Cancelled` means a documented cooperative boundary was reached.
`Poisoned` means native or cleanup state is uncertain. These classifications
describe reuse mechanics, not retry policy; retries and admission remain
application decisions.

No adapter call downloads a model, starts a server, opens a network service,
or falls back to CPU inference. Ordinary tests validate the contracts and
lowering model-free. Model-backed execution requires caller-supplied artifacts,
an explicit non-CPU device, and a separate opt-in acceptance run.

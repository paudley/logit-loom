<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# logit-loom-executor

`logit-loom-executor` defines two synchronous, policy-neutral execution seams:

- a worker-local borrowed-buffer executor for hosts that already own
  transport, admission, artifact verification, and resource policy; and
- an owned whole-request backend for execution boundaries that may place those
  mechanics behind a transport or scheduler.

The crate provides:

- bounded, serializable buffer identities;
- borrowed input and writable output views suitable for mapped storage;
- cooperative cancellation probes at backend-declared safe boundaries;
- explicit executor lifecycle and failure dispositions; and
- cleanup receipts that distinguish confirmed cleanup from uncertainty.

The whole-request contract adds:

- exact input, plan, output, backend, and evidence identities;
- bounded owned request, output, and opaque evidence bytes;
- cooperative cancellation with classified deadline and protocol failures;
  and
- a two-phase terminal result that is acknowledged only after the caller
  supplies a durable-record identity.

It does not open listeners, schedule jobs, select devices, download artifacts,
or define model behavior. Backends remain single-owner unless their own
contract explicitly says otherwise.

An output view exposes the complete caller-owned allocation but separately
tracks its initialized prefix. On failure, callers must ignore storage beyond
the last explicitly recorded prefix; the trait does not promise transactional
rollback of arbitrary output memory. `Rejected`, `Cancelled`, and `Poisoned`
describe whether a resident executor may be reused. They do not authorize an
automatic retry.

Whole-request backends expose no token-step IPC, per-token transforms,
observers, or checkpoints. A backend that cannot represent selected generation
mechanics exactly must reject the request before execution. Dropping or
aborting a pending result must not acknowledge successful completion.

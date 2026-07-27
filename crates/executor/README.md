<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# logit-loom-executor

`logit-loom-executor` defines the synchronous, worker-local seam between an
inference backend and a host that already owns transport, admission, artifact
verification, and resource policy.

The crate provides:

- bounded, serializable buffer identities;
- borrowed input and writable output views suitable for mapped storage;
- cooperative cancellation probes at backend-declared safe boundaries;
- explicit executor lifecycle and failure dispositions; and
- cleanup receipts that distinguish confirmed cleanup from uncertainty.

It does not open listeners, schedule jobs, select devices, download artifacts,
or define model behavior. Backends remain single-owner unless their own
contract explicitly says otherwise.

An output view exposes the complete caller-owned allocation but separately
tracks its initialized prefix. On failure, callers must ignore storage beyond
the last explicitly recorded prefix; the trait does not promise transactional
rollback of arbitrary output memory. `Rejected`, `Cancelled`, and `Poisoned`
describe whether a resident executor may be reused. They do not authorize an
automatic retry.

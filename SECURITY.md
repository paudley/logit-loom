<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Security policy

## Supported versions

Logit Loom is pre-1.0. Security fixes are made on the latest published release
and the default branch. Older alpha releases may require upgrading rather than
receiving a backport.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository. Do
not open a public issue for an undisclosed vulnerability. If private reporting
is unavailable, contact a repository maintainer privately through GitHub before
sharing exploit details.

Include the affected crate and version, backend feature, operating system,
minimal reproduction, impact, and any suggested remediation. Do not include
model weights, private prompts, credentials, or unrelated generated content.

## Scope notes

The native adapter inherits part of its attack surface from llama.cpp and
`llama-cpp-4`. A malformed GGUF, grammar, adapter, state snapshot, or control
vector should be treated as untrusted input. Public Rust APIs validate their
own bounds, but they do not make arbitrary native artifacts safe.

Receipts are content identities and accounting records, not signatures. Callers
that need authenticity must sign or otherwise authenticate persisted plans,
receipts, and checkpoint bytes at the application layer.

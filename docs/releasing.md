<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Release process

Only a maintainer publishes crates or creates GitHub releases.

## Prepare

1. Choose one workspace version and update every exact internal dependency.
2. Move relevant entries from `Unreleased` to a dated changelog section.
3. Confirm crate names are still available and package metadata points to the
   public repository.
4. Review `LICENSING.md`, `LICENSE-MIT`, and `LICENSE-APACHE`; Cargo metadata
   must remain `MIT OR Apache-2.0`. The separate proprietary option is not part
   of the Cargo SPDX expression.
5. Upgrade the pinned native binding only as a separate reviewed change.
6. Confirm the public repository description, topics, issue intake, private
   vulnerability reporting, dependency alerts, and security updates are
   configured.
7. Review the resolved dependency licenses and current security advisories for
   the exact `Cargo.lock`.
8. Verify the intended package owners and trusted-publisher records before
   publication.

## Validate

From a clean checkout:

```sh
scripts/release-check.sh
make package-list
```

Inspect package lists for missing license/readme files, private paths, model
artifacts, generated corpora, credentials, or unrelated internal references.
The release check performs no model inference.

The local release script does not install a dependency-audit tool implicitly.
The hosted publishing workflow explicitly installs its pinned `cargo-audit`
version. Record the tool and advisory snapshot used for the release, and review
any ignored or withdrawn advisory explicitly.

Dependent package verification may require the exact foundational version to
exist on crates.io. Use `cargo publish --dry-run` again immediately before each
real publication; do not bypass verification for a release.

Before the first foundational version is indexed, `make package-list` is the
available package-content audit for dependent crates. `make package` performs
full Cargo verification in publication order and is expected to succeed only
once each exact dependency it needs is available from the registry.

## Trusted publishing

The first version of each crate must be published manually with a current,
least-privilege crates.io credential. Subsequent coordinated releases use
`.github/workflows/release-cargo.yaml` and
[crates.io trusted publishing](https://crates.io/docs/trusted-publishing).
Each of the three crate records must use this exact identity:

| Field | Value |
| --- | --- |
| Repository owner | `paudley` |
| Repository name | `logit-loom` |
| Workflow filename | `release-cargo.yaml` |
| Environment | none |

The workflow has read-only repository access plus `id-token: write` only in
the publishing job. The pinned crates.io authentication action exchanges the
GitHub OIDC identity for a short-lived token and revokes it when the job
finishes. Do not add a long-lived crates.io token to GitHub secrets.

## Publish order

Publish and wait for crates.io indexing in this order:

1. `logit-loom-core`
2. `logit-loom`
3. `logit-loom-llamacpp`

The workflow performs a dry run immediately before publishing each crate:

```sh
cargo publish -p CRATE_NAME --locked --dry-run
cargo publish -p CRATE_NAME --locked
```

Create one signed `vVERSION` tag from the exact release commit and push that
tag. The tag must match the shared workspace version and a dated changelog
heading; the workflow rejects mismatches, reruns the clean release gate and
RustSec audit, then publishes in the order above.

After all packages resolve from crates.io, create the GitHub release from the
matching changelog section. A failed partially published release requires
maintainer review; do not move or reuse its tag.

## Verify

- Query each crates.io package and docs.rs page.
- In a fresh temporary project, resolve exact published versions without path
  overrides.
- Compile the backend-neutral example.
- Compile the adapter example with at least one supported accelerator feature.
- Confirm the tagged commit's hosted CI and dependency-security checks are
  green.
- Record any platform not validated in the release notes.

Do not describe model-quality, performance, or research efficacy unless that
specific claim has a separate cited evaluation.

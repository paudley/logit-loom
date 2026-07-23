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
8. Use a current, least-privilege crates.io credential and verify the intended
   package owners before publication.

## Validate

From a clean checkout:

```sh
scripts/release-check.sh
make package-list
```

Inspect package lists for missing license/readme files, private paths, model
artifacts, generated corpora, credentials, or unrelated internal references.
The release check performs no model inference.

The repository does not install a dependency-audit tool implicitly. Record the
tool and advisory snapshot used for the release, and review any ignored or
withdrawn advisory explicitly.

Dependent package verification may require the exact foundational version to
exist on crates.io. Use `cargo publish --dry-run` again immediately before each
real publication; do not bypass verification for a release.

Before the first foundational version is indexed, `make package-list` is the
available package-content audit for dependent crates. `make package` performs
full Cargo verification in publication order and is expected to succeed only
once each exact dependency it needs is available from the registry.

## Publish order

Publish and wait for crates.io indexing in this order:

1. `logit-loom-core`
2. `logit-loom`
3. `logit-loom-llamacpp`

For each crate:

```sh
cargo publish -p CRATE_NAME --locked --dry-run
cargo publish -p CRATE_NAME --locked
```

After all packages resolve from crates.io, create one signed tag such as
`v0.1.0-alpha.1` from the exact published commit and create the GitHub release
from the matching changelog section.

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

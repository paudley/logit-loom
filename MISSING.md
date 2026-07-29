<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Downstream return required

## Resident reference images need independent geometry

Observed against Logit Loom commit
`9e8f8a1f9789a27332e33c2a052034da1538c624`.

A downstream resident image program supplied a valid tightly packed RGB8
reference image whose geometry intentionally differs from the output canvas.
The program failed before native execution with:

```text
image program native input is invalid:
role is incompatible with the typed value
```

The public contract currently combines `SourceImage` and `ReferenceImage` in
`validate_role_spec` and requires both value specifications to equal the
program canvas. That is correct for source images used by image-to-image and
inpaint operations, but unnecessarily narrows reference conditioning.
`ImagePixels` carries its own checked width and height, and
`AdvancedImageRequest::with_reference` already accepts independently sized
reference pixels.

Required public return:

- Keep `SourceImage` and `Mask` exactly canvas-bound.
- Admit bounded, validated RGB8 or RGBA8 `ReferenceImage` values using the
  dimensions in their own `ImageProgramValueSpecV1`.
- Preserve those dimensions in serialization and mechanical identities; do
  not normalize, crop, stretch, or otherwise alter caller bytes implicitly.
- Add focused backend-neutral tests that accept non-canvas reference geometry
  and continue to reject zero, oversized, malformed, wrong-role, and
  byte-length-inconsistent values.
- Add a resident stable-diffusion.cpp test proving the reference reaches
  `AdvancedImageRequest::with_reference` with its original dimensions.
- Return one immutable Logit Loom commit with `make check-core`, `make check`,
  and `make doc` passing.

This is a mechanical reference-conditioning contract. It makes no claim about
the semantic effect or quality of any reference image.

After the public revision lands, the private runtime authority must vendor
that exact revision, rebuild its resident worker, and pass its own deployment
and live receipt gates. Logit Loom should not add a daemon, queue, deployment
policy, private fixture, or downstream application semantics.

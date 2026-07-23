<!-- SPDX-FileCopyrightText: 2026 Blackcat Informatics® Inc. <paudley@blackcatinformatics.ca> -->
<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Logit Loom brand

## Positioning

Logit Loom is a low-level Rust toolkit for bounded token-stream mechanics. Brand
copy may describe its ordered logit transforms, exact-byte observers, generation
controls, checkpoints, mechanical receipts, and llama.cpp adapter.

Do not use the brand to imply that a transform, sampler, prompt strategy, or
steering mechanism improves cognition, model quality, safety, truthfulness, or
any other semantic outcome. The project supplies caller-selected mechanisms, not
model policy.

## Name and tagline

- Name: **Logit Loom**
- Short tagline: **Bounded mechanics at the token boundary**
- Repository description: **Bounded Rust primitives for token-stream
  transforms, observers, receipts, and llama.cpp integration**

## Black-cat family system

The logo extends the shared black-cat family template. It preserves the
`cat-head-core` SVG group verbatim and replaces only the held service object.

Logit Loom's `service-loom` combines:

- five dark vertical warp threads, representing a bounded candidate-logit view;
- four coloured weft passes, representing transform stages executing in
  declared top-to-bottom order; and
- a paper-and-ink loom frame, preserving the family's held-object geometry.

The visual describes mechanics only. It does not depict or imply a preferred
token, semantic objective, or model outcome.

## Colour tokens

- cat and ink: `#111214`
- paper: `#fffdf5`
- feature: `#ffffff`
- red stage: `#ea4335`
- blue stage: `#4285f4`
- yellow stage: `#fbbc05`
- green stage: `#34a853`

Keep the cat, paper, ink, and feature colours stable. Accent colours are isolated
to the transform-stage paths and the social-card accents.

## Assets

- `docs/logit-loom-logo.svg` — canonical transparent vector logo.
- `docs/social-preview.svg` — editable 1280×640 social-card source.
- `docs/social-preview.png` — rendered 1280×640 sharing image.

The SVGs contain accessible titles and descriptions. Preserve them when making
future edits.

Rebuild the PNG from its canonical SVG source:

```sh
rsvg-convert -w 1280 -h 640 docs/social-preview.svg \
  -o docs/social-preview.png
```

GitHub repository social previews are uploaded through **Settings → Social
preview**. Use `docs/social-preview.png`; GitHub does not currently provide a
repository API for this upload.

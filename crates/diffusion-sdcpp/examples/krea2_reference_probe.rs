// SPDX-License-Identifier: MIT OR Apache-2.0

//! Probes Krea 2 reference-image conditioning for latent-mosaic corruption.
//!
//! Runs one exact request twice — without and with one reference image — and
//! reports per-run saturation plus 8/16-pixel grid-edge statistics that
//! detect the per-latent-cell mosaic defect signature
//! (`DRAW-REFERENCE-BINDING-DEFECT-2026-08-04`). Against a companion carrying
//! `logit-loom-krea-reference-guard-v16.patch` and a text-only Qwen3-VL
//! artifact, the reference run instead fails with a typed native error, which
//! this probe records as the expected guarded outcome.

// The shared support module also carries fork-report helpers used by the
// fork examples; this probe only needs the argument and JSON helpers.
#[allow(dead_code)]
mod support;

use logit_loom_diffusion_sdcpp::{
    AdvancedImageRequest, ContinueControl, ImageOutputSink, ImagePixels, ImageRequest,
    ProfileArtifacts, Sdcpp, SdcppOptions,
};
use serde::Serialize;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};

const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;
const SEED: u64 = 11;
const CFG_SCALE: f32 = 1.0;
const STEPS: u32 = 4;
const REFERENCE_SIDE: u32 = 256;

#[derive(Serialize)]
struct RunStats {
    saturated_fraction: f64,
    grid8_edge_ratio_horizontal: f64,
    grid8_edge_ratio_vertical: f64,
    grid16_edge_ratio_horizontal: f64,
    grid16_edge_ratio_vertical: f64,
}

#[derive(Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
enum RunOutcome {
    Image { stats: RunStats },
    Rejected { error: String },
}

#[derive(Serialize)]
struct Report {
    clean: RunOutcome,
    reference: RunOutcome,
}

struct VecSink {
    expected: usize,
    bytes: Vec<u8>,
}

impl ImageOutputSink for VecSink {
    fn expected_len(&self) -> usize {
        self.expected
    }

    fn write_image(&mut self, bytes: &[u8]) -> Result<(), String> {
        self.bytes = bytes.to_vec();
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let library = next(&mut arguments)?;
    let diffusion_model = next(&mut arguments)?;
    let text_encoder = next(&mut arguments)?;
    let vae = next(&mut arguments)?;
    let backend = next_utf8(&mut arguments, "backend")?;
    let threads = support::parse_threads(next(&mut arguments)?)?;
    let output_directory = support::output_directory(next(&mut arguments)?)?;
    let prompt = next_utf8(&mut arguments, "prompt")?;
    if arguments.next().is_some() {
        return Err(usage().into());
    }

    let artifacts = ProfileArtifacts::krea2(diffusion_model, text_encoder, vae);
    let options = SdcppOptions::new(backend.clone(), backend, threads)?;
    let mut runtime = Sdcpp::load(library, &artifacts, options)?;
    let request = ImageRequest::linear_euler(&prompt, WIDTH, HEIGHT, SEED, CFG_SCALE, STEPS)?;

    let clean_request = AdvancedImageRequest::text_to_image(request.clone())?;
    let clean = run(&mut runtime, &clean_request, &output_directory, "clean")?;

    let reference_bytes = synthetic_reference(REFERENCE_SIDE);
    let reference = ImagePixels::rgb8(&reference_bytes, REFERENCE_SIDE, REFERENCE_SIDE)?;
    let reference_request =
        AdvancedImageRequest::text_to_image(request)?.with_reference(reference)?;
    let reference = run(
        &mut runtime,
        &reference_request,
        &output_directory,
        "reference",
    )?;

    support::write_json(&Report { clean, reference })?;
    runtime.close()?;
    Ok(())
}

fn run(
    runtime: &mut Sdcpp,
    request: &AdvancedImageRequest<'_>,
    output_directory: &std::path::Path,
    name: &str,
) -> Result<RunOutcome, Box<dyn std::error::Error>> {
    let expected = usize::try_from(WIDTH)? * usize::try_from(HEIGHT)? * 3;
    let mut sink = VecSink {
        expected,
        bytes: Vec::new(),
    };
    match runtime.generate_advanced_controlled_to(
        request,
        &mut ContinueControl::default(),
        &mut sink,
    ) {
        Ok(_) => {
            write_rgb_ppm(
                output_directory,
                &format!("{name}.ppm"),
                &sink.bytes,
                WIDTH,
                HEIGHT,
            )?;
            Ok(RunOutcome::Image {
                stats: stats(&sink.bytes, WIDTH, HEIGHT),
            })
        }
        Err(error) => Ok(RunOutcome::Rejected {
            error: error.to_string(),
        }),
    }
}

/// Deterministic smooth photographic-ish reference: radial gradient with two
/// soft discs.
fn synthetic_reference(side: u32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(usize::try_from(side * side * 3).unwrap_or(0));
    let center = f64::from(side) / 2.0;
    for y in 0..side {
        for x in 0..side {
            let dx = (f64::from(x) - center) / center;
            let dy = (f64::from(y) - center) / center;
            let radius = (dx * dx + dy * dy).sqrt().min(1.0);
            let disc_a = (((f64::from(x) - center * 0.6).powi(2)
                + (f64::from(y) - center * 0.7).powi(2))
            .sqrt()
                / (center * 0.4))
                .min(1.0);
            let disc_b = (((f64::from(x) - center * 1.4).powi(2)
                + (f64::from(y) - center * 1.2).powi(2))
            .sqrt()
                / (center * 0.5))
                .min(1.0);
            let red = 60.0 + 150.0 * (1.0 - radius) + 40.0 * (1.0 - disc_a);
            let green = 80.0 + 120.0 * (1.0 - radius) + 50.0 * (1.0 - disc_b);
            let blue = 120.0 + 100.0 * radius;
            bytes.push(clamp_u8(red));
            bytes.push(clamp_u8(green));
            bytes.push(clamp_u8(blue));
        }
    }
    bytes
}

fn clamp_u8(value: f64) -> u8 {
    if value <= 0.0 {
        0
    } else if value >= 255.0 {
        255
    } else {
        let clamped = value.round();
        // Bounded by the checks above.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            clamped as u8
        }
    }
}

#[allow(clippy::cast_precision_loss)]
fn stats(bytes: &[u8], width: u32, height: u32) -> RunStats {
    let width = usize::try_from(width).unwrap_or(0);
    let height = usize::try_from(height).unwrap_or(0);
    let saturated = bytes
        .iter()
        .filter(|value| **value == 0 || **value == 255)
        .count();
    let saturated_fraction = if bytes.is_empty() {
        0.0
    } else {
        saturated as f64 / bytes.len() as f64
    };

    let luma = |x: usize, y: usize| -> f64 {
        let base = (y * width + x) * 3;
        0.299 * f64::from(bytes[base])
            + 0.587 * f64::from(bytes[base + 1])
            + 0.114 * f64::from(bytes[base + 2])
    };
    let grid_ratio = |period: usize, horizontal: bool| -> f64 {
        let mut on = (0.0, 0_u64);
        let mut off = (0.0, 0_u64);
        for y in usize::from(!horizontal)..height {
            for x in usize::from(horizontal)..width {
                let delta = if horizontal {
                    (luma(x, y) - luma(x - 1, y)).abs()
                } else {
                    (luma(x, y) - luma(x, y - 1)).abs()
                };
                let coordinate = if horizontal { x } else { y };
                if coordinate % period == 0 {
                    on.0 += delta;
                    on.1 += 1;
                } else {
                    off.0 += delta;
                    off.1 += 1;
                }
            }
        }
        let on_mean = if on.1 == 0 { 0.0 } else { on.0 / on.1 as f64 };
        let off_mean = if off.1 == 0 {
            0.0
        } else {
            off.0 / off.1 as f64
        };
        if off_mean <= f64::EPSILON {
            if on_mean <= f64::EPSILON {
                1.0
            } else {
                f64::INFINITY
            }
        } else {
            on_mean / off_mean
        }
    };

    RunStats {
        saturated_fraction,
        grid8_edge_ratio_horizontal: grid_ratio(8, true),
        grid8_edge_ratio_vertical: grid_ratio(8, false),
        grid16_edge_ratio_horizontal: grid_ratio(16, true),
        grid16_edge_ratio_vertical: grid_ratio(16, false),
    }
}

fn write_rgb_ppm(
    directory: &std::path::Path,
    name: &str,
    bytes: &[u8],
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = directory.join(name);
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut output = BufWriter::new(file);
    writeln!(output, "P6\n{width} {height}\n255")?;
    output.write_all(bytes)?;
    output.flush()?;
    Ok(())
}

fn next(
    arguments: &mut impl Iterator<Item = std::ffi::OsString>,
) -> Result<std::ffi::OsString, Box<dyn std::error::Error>> {
    arguments.next().ok_or_else(|| usage().into())
}

fn next_utf8(
    arguments: &mut impl Iterator<Item = std::ffi::OsString>,
    label: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    next(arguments)?
        .into_string()
        .map_err(|_| format!("{label} must be valid UTF-8").into())
}

fn usage() -> &'static str {
    "usage: krea2_reference_probe COMPANION_LIBRARY DIFFUSION_MODEL \
     TEXT_ENCODER VAE BACKEND THREADS OUTPUT_DIRECTORY PROMPT"
}

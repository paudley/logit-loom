// SPDX-License-Identifier: MIT OR Apache-2.0

//! Restores one prompt checkpoint to produce two independently sampled branches.

use std::io::{self, Write as _};

use logit_loom_runtime::{
    GenerationRequest, Loom, LoomOptions, NativeLogPolicy, SamplingPlan, Tokenization,
};

fn sampled(seed: u32) -> Result<GenerationRequest<'static>, logit_loom_runtime::Error> {
    GenerationRequest::new(48)?.sampling(SamplingPlan {
        seed,
        temperature: 0.8,
        ..SamplingPlan::default()
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let model_path = arguments.next().ok_or("usage: branch MODEL.gguf PROMPT")?;
    let prompt = arguments
        .next()
        .ok_or("usage: branch MODEL.gguf PROMPT")?
        .into_string()
        .map_err(|_| "prompt must be valid UTF-8")?;

    let loom = Loom::load(
        model_path,
        LoomOptions {
            native_logs: NativeLogPolicy::Silence,
            ..LoomOptions::default()
        },
    )?;
    let mut session = loom.session()?;
    session.replace_text(&prompt, Tokenization { add_bos: true })?;
    let checkpoint = session.capture_state()?;

    let first = session.generate(sampled(7)?)?;
    session.restore_state(&checkpoint)?;
    let second = session.generate(sampled(8)?)?;

    let mut stdout = io::stdout().lock();
    stdout.write_all(b"branch 1:\n")?;
    stdout.write_all(&first.bytes)?;
    stdout.write_all(b"\n\nbranch 2:\n")?;
    stdout.write_all(&second.bytes)?;
    stdout.write_all(b"\n")?;
    Ok(())
}

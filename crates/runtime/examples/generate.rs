// SPDX-License-Identifier: MIT OR Apache-2.0

//! Generates exact output bytes from a caller-supplied local model.

use std::io::{self, Write as _};

use logit_loom_runtime::{GenerationRequest, Loom, LoomOptions, NativeLogPolicy, Tokenization};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let model_path = arguments
        .next()
        .ok_or("usage: generate MODEL.gguf PROMPT")?;
    let prompt = arguments
        .next()
        .ok_or("usage: generate MODEL.gguf PROMPT")?
        .into_string()
        .map_err(|_| "prompt must be valid UTF-8")?;

    let loom = Loom::load(
        model_path,
        LoomOptions {
            native_logs: NativeLogPolicy::Silence,
            ..LoomOptions::default()
        },
    )?;
    let output = loom.complete(
        &prompt,
        Tokenization { add_bos: true },
        GenerationRequest::new(64)?,
    )?;

    io::stdout().lock().write_all(output.bytes())?;
    Ok(())
}

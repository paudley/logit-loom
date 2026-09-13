// SPDX-License-Identifier: MIT OR Apache-2.0

//! Generates exact output bytes with a caller-supplied local model.

use std::io::{self, Write as _};

use logit_loom::{GenerationPlan, SamplingPlan};
use logit_loom_llamacpp::{Model, ModelOptions, Runtime, SessionOptions, Tokenization};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let model_path = arguments
        .next()
        .ok_or("usage: generate MODEL.gguf PROMPT")?;
    let prompt = arguments
        .next()
        .ok_or("usage: generate MODEL.gguf PROMPT [MAX_TOKENS]")?
        .into_string()
        .map_err(|_| "prompt must be valid UTF-8")?;
    let max_tokens = match arguments.next() {
        Some(v) => v
            .into_string()
            .map_err(|_| "MAX_TOKENS must be UTF-8")?
            .parse()?,
        None => 64,
    };

    let mut runtime = Runtime::initialize()?;
    runtime.silence_native_logs();
    let model = Model::load(&runtime, model_path, ModelOptions::default())?;
    let tokens = model.tokenize(&prompt, Tokenization { add_bos: true })?;
    let mut session = model.session(&runtime, SessionOptions::default())?;
    session.prefill(&tokens, true)?;

    let output = session.generate(
        &GenerationPlan {
            sampling: SamplingPlan::default(),
            max_tokens,
            biases: Vec::new(),
            grammar: None,
            stops: Vec::new(),
        },
        None,
        None,
    )?;
    io::stdout().lock().write_all(&output.bytes)?;
    Ok(())
}

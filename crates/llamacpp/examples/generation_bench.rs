// SPDX-License-Identifier: MIT OR Apache-2.0

//! Measures prefill and decode throughput for one model at one session
//! geometry, with every knob flowing through [`SessionOptions`].
//!
//! Emits one JSON line per repeat so a sweep can collect medians. Timing is
//! a deployment measurement, never part of any identity.
//!
//! ```text
//! generation_bench MODEL.gguf [--prompt-file FILE | --prompt TEXT]
//!     [--ctx N] [--batch N] [--micro N] [--threads N]
//!     [--kv f16|q8_0|q4_0] [--flash auto|on|off]
//!     [--max-tokens N] [--repeat N] [--native-logs on|off]
//! ```

use std::fs;
use std::num::NonZeroU32;
use std::time::Instant;

use logit_loom::{GenerationPlan, SamplingPlan};
use logit_loom_llamacpp::{
    ContextCompatibility, ContextKind, FlashAttention, KvCacheType, Model, ModelOptions, Runtime,
    SessionOptions, Tokenization,
};

struct Arguments {
    model: std::ffi::OsString,
    prompt: String,
    options: SessionOptions,
    max_tokens: u32,
    repeat: u32,
    native_logs: bool,
}

fn parse_arguments() -> Result<Arguments, Box<dyn std::error::Error>> {
    let usage = "usage: generation_bench MODEL.gguf [--prompt-file FILE | --prompt TEXT] \
                 [--ctx N] [--batch N] [--micro N] [--threads N] [--kv f16|q8_0|q4_0] \
                 [--flash auto|on|off] [--max-tokens N] [--repeat N] [--native-logs on|off]";
    let mut arguments = std::env::args_os().skip(1);
    let model = arguments.next().ok_or(usage)?;
    let mut prompt = None;
    let mut options = SessionOptions::default();
    let mut max_tokens = 128;
    let mut repeat = 1;
    let mut native_logs = false;
    while let Some(flag) = arguments.next() {
        let flag = flag.into_string().map_err(|_| "flags must be UTF-8")?;
        let value = arguments
            .next()
            .ok_or_else(|| format!("{flag} requires a value"))?
            .into_string()
            .map_err(|_| "flag values must be UTF-8")?;
        match flag.as_str() {
            "--prompt-file" => prompt = Some(fs::read_to_string(value)?),
            "--prompt" => prompt = Some(value),
            "--ctx" => {
                options.context_size =
                    NonZeroU32::new(value.parse()?).ok_or("--ctx must be nonzero")?;
            }
            "--batch" => options.batch_size = value.parse()?,
            "--micro" => options.micro_batch_size = value.parse()?,
            "--threads" => options.threads = value.parse()?,
            "--kv" => {
                options.kv_cache_type = match value.as_str() {
                    "f16" => KvCacheType::F16,
                    "q8_0" => KvCacheType::Q8_0,
                    "q4_0" => KvCacheType::Q4_0,
                    other => return Err(format!("unknown --kv {other}").into()),
                };
            }
            "--flash" => {
                options.flash_attention = match value.as_str() {
                    "auto" => FlashAttention::Auto,
                    "on" => FlashAttention::Enabled,
                    "off" => FlashAttention::Disabled,
                    other => return Err(format!("unknown --flash {other}").into()),
                };
            }
            "--max-tokens" => max_tokens = value.parse()?,
            "--repeat" => repeat = value.parse()?,
            "--native-logs" => {
                native_logs = match value.as_str() {
                    "on" => true,
                    "off" => false,
                    other => return Err(format!("unknown --native-logs {other}").into()),
                };
            }
            other => return Err(format!("unknown flag {other}\n{usage}").into()),
        }
    }
    Ok(Arguments {
        model,
        prompt: prompt.ok_or("--prompt-file or --prompt is required")?,
        options,
        max_tokens,
        repeat,
        native_logs,
    })
}

/// Token counts here are far below 2^52, so the cast is exact.
#[allow(clippy::cast_precision_loss)]
fn tokens_per_second(count: u64, millis: f64) -> f64 {
    count as f64 / millis * 1_000.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = parse_arguments()?;
    let mut runtime = Runtime::initialize()?;
    if !arguments.native_logs {
        runtime.silence_native_logs();
    }
    let load_started = Instant::now();
    let model = Model::load(&runtime, &arguments.model, ModelOptions::default())?;
    let load_millis = load_started.elapsed().as_millis();
    let tokens = model.tokenize(&arguments.prompt, Tokenization { add_bos: true })?;
    let compatibility = ContextCompatibility {
        runtime: runtime.identity().clone(),
        options: arguments.options,
        context: ContextKind::Ordinary,
        recurrent_state_slots: 0,
    }
    .digest();
    let plan = GenerationPlan {
        sampling: SamplingPlan::default(),
        max_tokens: arguments.max_tokens,
        biases: Vec::new(),
        grammar: None,
        stops: Vec::new(),
    };

    for repeat in 0..arguments.repeat {
        let mut session = model.session(&runtime, arguments.options)?;
        let prefill_started = Instant::now();
        let prefill = session.prefill(&tokens, true)?;
        let prefill_millis = prefill_started.elapsed().as_secs_f64() * 1_000.0;
        let decode_started = Instant::now();
        let output = session.generate(&plan, None, None)?;
        let decode_millis = decode_started.elapsed().as_secs_f64() * 1_000.0;
        let checkpoint_bytes = session.capture_state()?.bytes().len();
        let generated = output.tokens.len();
        let prefill_tokens_per_second = tokens_per_second(prefill.admitted_tokens, prefill_millis);
        let decode_tokens_per_second = tokens_per_second(generated as u64, decode_millis);
        println!(
            "{{\"repeat\":{repeat},\"load_millis\":{load_millis},\
             \"prompt_tokens\":{},\"prefill_millis\":{prefill_millis:.1},\
             \"prefill_tokens_per_second\":{prefill_tokens_per_second:.1},\
             \"generated_tokens\":{generated},\"decode_millis\":{decode_millis:.1},\
             \"decode_tokens_per_second\":{decode_tokens_per_second:.2},\
             \"checkpoint_bytes\":{checkpoint_bytes},\
             \"checkpoint_bytes_per_token\":{},\
             \"context_size\":{},\"batch_size\":{},\"micro_batch_size\":{},\
             \"threads\":{},\"kv_cache_type\":\"{:?}\",\"flash_attention\":\"{:?}\",\
             \"compatibility\":\"{}\"}}",
            prefill.admitted_tokens,
            checkpoint_bytes / (usize::try_from(prefill.position).unwrap_or(0) + generated).max(1),
            arguments.options.context_size,
            arguments.options.batch_size,
            arguments.options.micro_batch_size,
            arguments.options.threads,
            arguments.options.kv_cache_type,
            arguments.options.flash_attention,
            compatibility.as_str(),
        );
    }
    Ok(())
}

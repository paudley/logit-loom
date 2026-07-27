// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cancels generation at the same post-admission boundary as a counting observer.

mod support;

use std::cell::Cell;
use std::rc::Rc;

use logit_loom_runtime::{
    CancellationToken, ControlFlow, Digest, GenerationPlan, GenerationRequest, Loom, LoomOptions,
    NativeLogPolicy, ObserverError, ObserverReceipt, ObserversBuilder, Tokenization,
};
use serde::Serialize;
use support::{AdmissionRecord, GenerationRecord, RunMetadata};

const MAX_BREAKER_TOKENS: u32 = 1_024;
const TOKEN_LIMIT_HEADROOM: u32 = 16;

#[derive(Debug, Serialize)]
struct Report {
    scenario: &'static str,
    metadata: RunMetadata,
    admission: AdmissionRecord,
    generation_plan: GenerationPlan,
    requested_break_after: u32,
    callback_observations: u32,
    cancellation_requested: bool,
    generation: GenerationRecord,
    observers: Vec<ObserverReceipt>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let model_path = arguments
        .next()
        .ok_or("usage: causal_circuit_breaker MODEL.gguf PROMPT ADMITTED_TOKENS")?;
    let prompt = arguments
        .next()
        .ok_or("usage: causal_circuit_breaker MODEL.gguf PROMPT ADMITTED_TOKENS")?
        .into_string()
        .map_err(|_| "prompt must be valid UTF-8")?;
    let break_after = arguments
        .next()
        .ok_or("usage: causal_circuit_breaker MODEL.gguf PROMPT ADMITTED_TOKENS")?
        .into_string()
        .map_err(|_| "ADMITTED_TOKENS must be valid UTF-8")?
        .parse::<u32>()?;
    if arguments.next().is_some() {
        return Err("usage: causal_circuit_breaker MODEL.gguf PROMPT ADMITTED_TOKENS".into());
    }
    if !(1..=MAX_BREAKER_TOKENS).contains(&break_after) {
        return Err(format!("ADMITTED_TOKENS must be in 1..={MAX_BREAKER_TOKENS}").into());
    }
    let maximum_tokens = break_after
        .checked_add(TOKEN_LIMIT_HEADROOM)
        .ok_or("generation token bound overflowed")?;

    let loom = Loom::load(
        model_path,
        LoomOptions {
            native_logs: NativeLogPolicy::Silence,
            ..LoomOptions::default()
        },
    )?;
    let metadata = RunMetadata::capture(&loom);
    let mut session = loom.session()?;
    let admission = session.replace_text(&prompt, Tokenization { add_bos: true })?;

    let cancellation = CancellationToken::new();
    let callback_cancellation = cancellation.clone();
    let callback_observations = Rc::new(Cell::new(0_u32));
    let callback_counter = Rc::clone(&callback_observations);
    let mut observers = ObserversBuilder::new()
        .on_token(
            Digest::of_bytes("runbook-observer-v1", b"causal-circuit-breaker"),
            move |_token| {
                let next = callback_counter
                    .get()
                    .checked_add(1)
                    .ok_or_else(|| ObserverError::new("callback observation count overflowed"))?;
                callback_counter.set(next);
                if next == break_after {
                    callback_cancellation.cancel();
                }
                Ok(ControlFlow::Continue)
            },
        )?
        .cancellation(&cancellation)?
        .build()?;
    let generation_request = GenerationRequest::new(maximum_tokens)?;
    let generation_plan = generation_request.plan().clone();
    let generation = session.generate(generation_request.observers(&mut observers)?)?;

    support::write_json(&Report {
        scenario: "causal_circuit_breaker",
        metadata,
        admission: admission.into(),
        generation_plan,
        requested_break_after: break_after,
        callback_observations: callback_observations.get(),
        cancellation_requested: cancellation.is_cancelled(),
        generation: generation.try_into()?,
        observers: observers.receipts(),
    })
}

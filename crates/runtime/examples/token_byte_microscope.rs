// SPDX-License-Identifier: MIT OR Apache-2.0

//! Records every causally admitted generated token as an exact byte piece.

mod support;

use std::cell::RefCell;
use std::rc::Rc;

use logit_loom_runtime::{
    ControlFlow, Digest, GenerationPlan, GenerationRequest, Loom, LoomOptions, NativeLogPolicy,
    ObserverError, ObserverReceipt, ObserversBuilder, TokenId, Tokenization,
};
use serde::Serialize;
use support::{AdmissionRecord, GenerationRecord, RunMetadata};

const GENERATED_TOKENS: u32 = 16;

#[derive(Clone, Debug, Serialize)]
struct TokenEvent {
    generation_index: u32,
    causal_position: u64,
    token: TokenId,
    piece_hex: String,
}

#[derive(Debug, Serialize)]
struct Report {
    scenario: &'static str,
    metadata: RunMetadata,
    admission: AdmissionRecord,
    generation_plan: GenerationPlan,
    observed_tokens: Vec<TokenEvent>,
    generation: GenerationRecord,
    observers: Vec<ObserverReceipt>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let model_path = arguments
        .next()
        .ok_or("usage: token_byte_microscope MODEL.gguf PROMPT")?;
    let prompt = arguments
        .next()
        .ok_or("usage: token_byte_microscope MODEL.gguf PROMPT")?
        .into_string()
        .map_err(|_| "prompt must be valid UTF-8")?;
    if arguments.next().is_some() {
        return Err("usage: token_byte_microscope MODEL.gguf PROMPT".into());
    }

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

    let events = Rc::new(RefCell::new(Vec::new()));
    let event_sink = Rc::clone(&events);
    let mut observers = ObserversBuilder::new()
        .on_token(
            Digest::of_bytes("runbook-observer-v1", b"token-byte-microscope"),
            move |token| {
                let generation_index = u32::try_from(event_sink.borrow().len())
                    .map_err(|_| ObserverError::new("token event count exceeds u32"))?;
                event_sink.borrow_mut().push(TokenEvent {
                    generation_index,
                    causal_position: token.position,
                    token: token.token,
                    piece_hex: support::encode_hex(token.piece),
                });
                Ok(ControlFlow::Continue)
            },
        )?
        .build()?;
    let generation_request = GenerationRequest::new(GENERATED_TOKENS)?;
    let generation_plan = generation_request.plan().clone();
    let generation = session.generate(generation_request.observers(&mut observers)?)?;
    let observed_tokens = events.borrow().clone();

    support::write_json(&Report {
        scenario: "token_byte_microscope",
        metadata,
        admission: admission.into(),
        generation_plan,
        observed_tokens,
        generation: generation.into(),
        observers: observers.receipts(),
    })
}

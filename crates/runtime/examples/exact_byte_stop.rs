// SPDX-License-Identifier: MIT OR Apache-2.0

//! Stops bounded generation after an exact caller-supplied byte suffix.

mod support;

use logit_loom_runtime::{
    GenerationFinish, GenerationPlan, GenerationRequest, Loom, LoomOptions,
    MAX_STOP_SEQUENCE_BYTES, NativeLogPolicy, Tokenization,
};
use serde::Serialize;
use support::{AdmissionRecord, GenerationRecord, RunMetadata};

const GENERATED_TOKENS: u32 = 128;

#[derive(Debug, Serialize)]
struct Report {
    scenario: &'static str,
    metadata: RunMetadata,
    admission: AdmissionRecord,
    generation_plan: GenerationPlan,
    stop_bytes_hex: String,
    stop_selected: bool,
    output_ends_with_stop: bool,
    generation: GenerationRecord,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let model_path = arguments
        .next()
        .ok_or("usage: exact_byte_stop MODEL.gguf PROMPT STOP_HEX")?;
    let prompt = arguments
        .next()
        .ok_or("usage: exact_byte_stop MODEL.gguf PROMPT STOP_HEX")?
        .into_string()
        .map_err(|_| "prompt must be valid UTF-8")?;
    let stop_hex = arguments
        .next()
        .ok_or("usage: exact_byte_stop MODEL.gguf PROMPT STOP_HEX")?
        .into_string()
        .map_err(|_| "stop bytes must be lowercase or uppercase hexadecimal")?;
    if arguments.next().is_some() {
        return Err("usage: exact_byte_stop MODEL.gguf PROMPT STOP_HEX".into());
    }
    let stop = decode_hex(&stop_hex)?;

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
    let generation_request = GenerationRequest::new(GENERATED_TOKENS)?.stop_bytes(&stop)?;
    let generation_plan = generation_request.plan().clone();
    let generation = session.generate(generation_request)?;
    let stop_selected = matches!(
        generation.receipt.finish,
        GenerationFinish::StopSequence { index: 0 }
    );
    let output_ends_with_stop = generation.bytes.ends_with(&stop);

    support::write_json(&Report {
        scenario: "exact_byte_stop",
        metadata,
        admission: admission.into(),
        generation_plan,
        stop_bytes_hex: support::encode_hex(&stop),
        stop_selected,
        output_ends_with_stop,
        generation: generation.try_into()?,
    })
}

fn decode_hex(input: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let maximum_hex_bytes = MAX_STOP_SEQUENCE_BYTES
        .checked_mul(2)
        .ok_or("stop-sequence hexadecimal bound overflowed")?;
    if input.is_empty() || !input.len().is_multiple_of(2) || input.len() > maximum_hex_bytes {
        return Err(format!(
            "STOP_HEX must encode 1..={MAX_STOP_SEQUENCE_BYTES} bytes using complete byte pairs"
        )
        .into());
    }

    let mut decoded = Vec::with_capacity(input.len() / 2);
    for pair in input.as_bytes().chunks_exact(2) {
        let high = hex_nibble(pair[0]).ok_or("STOP_HEX contains a non-hexadecimal byte")?;
        let low = hex_nibble(pair[1]).ok_or("STOP_HEX contains a non-hexadecimal byte")?;
        decoded.push((high << 4) | low);
    }
    Ok(decoded)
}

const fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hexadecimal_stops_preserve_arbitrary_bytes() {
        assert_eq!(decode_hex("00ff4A").unwrap(), [0x00, 0xff, 0x4a]);
    }

    #[test]
    fn hexadecimal_stops_reject_malformed_or_oversized_inputs() {
        assert!(decode_hex("").is_err());
        assert!(decode_hex("0").is_err());
        assert!(decode_hex("xx").is_err());
        assert!(decode_hex(&"00".repeat(MAX_STOP_SEQUENCE_BYTES + 1)).is_err());
    }
}

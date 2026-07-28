// SPDX-License-Identifier: MIT OR Apache-2.0

//! Differentially qualifies the public ranked-BPE Qwen adapter against the
//! exact Rust tokenizer pipeline loaded from the same caller-supplied JSON.

use std::error::Error;
use std::path::PathBuf;
use std::str::FromStr;

use logit_loom_core::{Digest, TokenId};
use logit_loom_tokenizer::{
    BoundaryTokenPolicy, BpeBoundaryTokens, CancellationToken, OffsetPolicy, QwenRankedBpe,
    QwenTokenizerConfig, SourceSpecialTokenPolicy, TokenSpan, TokenizationOracleCase,
    TokenizationOracleReceipt, TokenizationPolicy, qualify_tokenizer_oracle,
};
use serde::Serialize;
use tokenizers::Tokenizer;

const CORPUS: &[&str] = &[
    "",
    "hello",
    "The quick brown fox can't jump 13.5 times.\n",
    "  leading\tand trailing  ",
    "\r\n\nline one\r\nline two",
    "café déjà vu",
    "cafe\u{301} de\u{301}ja\u{300} vu",
    "中文、かな、한국어",
    "العَرَبِيَّةُ",
    "हिन्दी और संस्कृत",
    "👩🏽‍💻🧑‍🚀🏳️‍🌈",
    "fn main() { println!(\"<tag>&value\"); }\n",
    "<|im_start|>user\nhello<|im_end|>",
    "<tool_call>{\"name\":\"x\"}</tool_call>",
    "\u{2003}\u{2009}\u{00a0}\t\n",
    "Z͑͗͛a̐͊l͌g̎o̅",
];

#[derive(Debug, Serialize)]
struct Qualification {
    schema: &'static str,
    tokenizer_json_bytes: u64,
    tokenizer_json_blake3: String,
    pretokenizer: logit_loom_tokenizer::QwenPretokenizer,
    ordinary_text: TokenizationOracleReceipt,
    configured_specials: TokenizationOracleReceipt,
    hostile_nul_rejected: bool,
    cancelled_before_work: bool,
}

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut arguments = std::env::args_os().skip(1);
    let tokenizer_path = PathBuf::from(
        arguments
            .next()
            .ok_or("usage: qualify_qwen TOKENIZER_JSON MODEL_BLAKE3")?,
    );
    let model = arguments
        .next()
        .ok_or("usage: qualify_qwen TOKENIZER_JSON MODEL_BLAKE3")?
        .into_string()
        .map_err(|_| "model digest must be UTF-8")?;
    if arguments.next().is_some() {
        return Err("usage: qualify_qwen TOKENIZER_JSON MODEL_BLAKE3".into());
    }
    let model = Digest::from_str(&model)?;
    let tokenizer_json = std::fs::read(tokenizer_path)?;
    let ranked = QwenRankedBpe::from_tokenizer_json(
        QwenTokenizerConfig {
            model,
            boundaries: BpeBoundaryTokens::default(),
        },
        &tokenizer_json,
    )?;
    let ordinary_text = qualify_policy(
        &ranked,
        &tokenizer_json,
        SourceSpecialTokenPolicy::OrdinaryText,
    )?;
    let configured_specials = qualify_policy(
        &ranked,
        &tokenizer_json,
        SourceSpecialTokenPolicy::RecognizeConfigured,
    )?;
    let hostile_policy = policy(SourceSpecialTokenPolicy::OrdinaryText);
    let hostile_nul_rejected = logit_loom_tokenizer::ExactTokenizer::count(
        &ranked,
        b"a\0b",
        &hostile_policy,
        &CancellationToken::default(),
    )
    .is_err();
    let cancellation = CancellationToken::default();
    cancellation.request();
    let cancelled_before_work = logit_loom_tokenizer::ExactTokenizer::count(
        &ranked,
        b"cancel",
        &hostile_policy,
        &cancellation,
    )
    .is_err();
    let receipt = Qualification {
        schema: "logit-loom-qwen-ranked-bpe-qualification-v1",
        tokenizer_json_bytes: u64::try_from(tokenizer_json.len())?,
        tokenizer_json_blake3: blake3::hash(&tokenizer_json).to_hex().to_string(),
        pretokenizer: ranked.pretokenizer(),
        ordinary_text,
        configured_specials,
        hostile_nul_rejected,
        cancelled_before_work,
    };
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}

fn qualify_policy(
    ranked: &QwenRankedBpe,
    tokenizer_json: &[u8],
    special_tokens: SourceSpecialTokenPolicy,
) -> Result<TokenizationOracleReceipt, Box<dyn Error + Send + Sync>> {
    let mut reference = Tokenizer::from_bytes(tokenizer_json)?;
    reference.set_encode_special_tokens(special_tokens == SourceSpecialTokenPolicy::OrdinaryText);
    let expected = CORPUS
        .iter()
        .map(|source| {
            let encoding = reference.encode(*source, false)?;
            encoding
                .get_ids()
                .iter()
                .zip(encoding.get_offsets())
                .map(|(&id, &(start, end))| {
                    Ok(TokenSpan {
                        token: TokenId::new(i32::try_from(id)?)?,
                        start: u32::try_from(start)?,
                        end: u32::try_from(end)?,
                    })
                })
                .collect::<Result<Vec<_>, Box<dyn Error + Send + Sync>>>()
        })
        .collect::<Result<Vec<_>, Box<dyn Error + Send + Sync>>>()?;
    let cases = CORPUS
        .iter()
        .zip(&expected)
        .map(|(source, expected)| TokenizationOracleCase {
            source: source.as_bytes(),
            expected,
        })
        .collect::<Vec<_>>();
    Ok(qualify_tokenizer_oracle(
        ranked,
        &policy(special_tokens),
        &cases,
        &CancellationToken::default(),
    )?)
}

const fn policy(special_tokens: SourceSpecialTokenPolicy) -> TokenizationPolicy {
    TokenizationPolicy {
        boundary_tokens: BoundaryTokenPolicy::None,
        source_special_tokens: special_tokens,
        offsets: OffsetPolicy::Include,
        count_through: None,
    }
}

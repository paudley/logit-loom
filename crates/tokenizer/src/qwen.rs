// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact Qwen byte-BPE adapter over a pinned tokenizer JSON artifact.
//!
//! The Hugging Face Rust pipeline owns NFC normalization, configured
//! added-token extraction, Qwen pretokenization, byte-level transformation,
//! and source-offset alignment. The public [`RankedBpe`] kernel owns every
//! BPE merge. No model, network client, Python runtime, or global worker pool
//! participates in this adapter.

use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;

use logit_loom_core::{Digest, TokenId};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokenizers::{AddedVocabulary, OffsetType, PreTokenizedString, PreTokenizer, Token, Tokenizer};

use crate::operator::{validate_source, validate_spans};
use crate::{
    BoundaryTokenPolicy, BpeBoundaryTokens, BpeMerge, BpeScratch, CancellationToken, CountResult,
    CountingSink, ExactTokenizer, MAX_ROW_BYTES, MAX_TOKENS_PER_ROW, OffsetPolicy, RankedBpe,
    SinkFlow, SourceSpecialTokenPolicy, TokenOutputSink, TokenSpan, TokenizationError,
    TokenizationIdentity, TokenizationIdentityV2, TokenizationPolicy, VecTokenSink,
};

const QWEN2_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const QWEN35_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const TOKENIZERS_REVISION: &str = "tokenizers-0.23.1";
const UNICODE_REVISION: &str = "unicode-normalization-alignments-0.1.12";
const ADAPTER_REVISION: &str = "qwen-ranked-bpe-v1";

/// Exact Qwen regex/pretokenizer family accepted by the adapter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum QwenPretokenizer {
    /// Qwen2/Qwen3 tokenizer using Unicode letter runs.
    Qwen2,
    /// Qwen3.5 tokenizer whose letter runs also admit combining marks.
    Qwen35,
}

/// Caller-owned immutable bindings not encoded by tokenizer JSON.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenTokenizerConfig {
    /// Exact model artifact whose token IDs are projected.
    pub model: Digest,
    /// Optional boundary token IDs exposed through explicit policy.
    pub boundaries: BpeBoundaryTokens,
}

/// Exact Qwen JSON tokenizer whose merges execute through [`RankedBpe`].
#[derive(Clone, Debug)]
pub struct QwenRankedBpe {
    identity: TokenizationIdentity,
    identity_v2: TokenizationIdentityV2,
    pretokenizer: QwenPretokenizer,
    pipeline: Tokenizer,
    ordinary_added: AddedVocabulary,
    vocabulary: HashMap<String, TokenId>,
    boundaries: BpeBoundaryTokens,
    kernel: RankedBpe,
}

impl QwenRankedBpe {
    /// Parses one complete tokenizer JSON artifact and builds its canonical
    /// ranked merge table.
    ///
    /// Only the exact Qwen2 and Qwen3.5 NFC + regex + byte-level BPE pipeline
    /// shapes are accepted. Truncation, padding, dropout, unknown-token
    /// substitution, byte fallback, ignored merges, and non-Qwen pipeline
    /// stages fail closed.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed JSON, an unsupported pipeline, invalid
    /// token IDs, non-canonical merges, or unavailable boundary tokens.
    pub fn from_tokenizer_json(
        config: QwenTokenizerConfig,
        tokenizer_json: &[u8],
    ) -> Result<Self, TokenizationError> {
        if tokenizer_json.is_empty() || tokenizer_json.len() > MAX_ROW_BYTES {
            return Err(TokenizationError::Bound {
                field: "tokenizer JSON bytes",
                limit: MAX_ROW_BYTES,
            });
        }
        let document: Value = serde_json::from_slice(tokenizer_json)
            .map_err(|error| TokenizationError::Invalid(format!("tokenizer JSON: {error}")))?;
        let pretokenizer = validate_qwen_document(&document)?;
        let pipeline = Tokenizer::from_bytes(tokenizer_json)
            .map_err(|error| TokenizationError::Operator(error.to_string()))?;
        if pipeline.get_encode_special_tokens() {
            return Err(TokenizationError::Invalid(
                "tokenizer JSON unexpectedly encodes configured specials as ordinary text"
                    .to_owned(),
            ));
        }
        validate_boundaries(&pipeline, config.boundaries)?;

        let vocabulary = parse_vocabulary(&document)?;
        let canonical_vocabulary = vocabulary
            .iter()
            .map(|(token, id)| (token.clone(), id.get()))
            .collect::<BTreeMap<_, _>>();
        let merges = parse_merges(&document, &vocabulary)?;
        let kernel = RankedBpe::new(merges)?;
        let tokenizer = Digest::of_bytes("qwen-tokenizer-json-artifact-v1", tokenizer_json);
        let vocabulary_identity =
            Digest::of_serializable("qwen-tokenizer-vocabulary-v1", &canonical_vocabulary)
                .map_err(|error| TokenizationError::Identity(error.to_string()))?;
        let normalizer =
            component_identity("qwen-tokenizer-normalizer-v1", document.get("normalizer"))?;
        let pretokenizer_identity = component_identity(
            "qwen-tokenizer-pretokenizer-v1",
            document.get("pre_tokenizer"),
        )?;
        let added_tokens = component_identity(
            "qwen-tokenizer-added-tokens-v1",
            document.get("added_tokens"),
        )?;
        let special_tokens = special_token_identity(&document)?;
        let unicode = Digest::of_serializable(
            "qwen-tokenizer-unicode-contract-v1",
            &(TOKENIZERS_REVISION, UNICODE_REVISION, pretokenizer),
        )
        .map_err(|error| TokenizationError::Identity(error.to_string()))?;
        let policy_schema = Digest::of_serializable(
            "qwen-tokenizer-policy-schema-v1",
            &(
                config.boundaries,
                BoundaryTokenPolicy::None,
                SourceSpecialTokenPolicy::OrdinaryText,
                OffsetPolicy::Include,
            ),
        )
        .map_err(|error| TokenizationError::Identity(error.to_string()))?;
        let implementation = Digest::of_serializable(
            "qwen-ranked-bpe-implementation-v1",
            &(
                ADAPTER_REVISION,
                TOKENIZERS_REVISION,
                UNICODE_REVISION,
                pretokenizer,
                kernel.implementation(),
            ),
        )
        .map_err(|error| TokenizationError::Identity(error.to_string()))?;
        let identity_v2 = TokenizationIdentityV2 {
            model: config.model,
            tokenizer,
            vocabulary: vocabulary_identity,
            merges: kernel.merges_identity().clone(),
            normalizer,
            pretokenizer: pretokenizer_identity,
            unicode,
            added_tokens,
            special_tokens,
            implementation,
            policy_schema,
        };
        let identity = identity_v2.legacy_v1()?;
        let mut ordinary_added = pipeline.get_added_vocabulary().clone();
        ordinary_added.set_encode_special_tokens(true);
        Ok(Self {
            identity,
            identity_v2,
            pretokenizer,
            pipeline,
            ordinary_added,
            vocabulary: vocabulary.into_iter().collect(),
            boundaries: config.boundaries,
            kernel,
        })
    }

    /// Returns the exact accepted Qwen pretokenizer family.
    #[must_use]
    pub const fn pretokenizer(&self) -> QwenPretokenizer {
        self.pretokenizer
    }

    fn encoding(
        &self,
        source: &[u8],
        special_policy: SourceSpecialTokenPolicy,
        cancellation: &CancellationToken,
    ) -> Result<tokenizers::Encoding, TokenizationError> {
        let text = validate_source(source)?;
        cancellation.check()?;
        let added = match special_policy {
            SourceSpecialTokenPolicy::RecognizeConfigured => self.pipeline.get_added_vocabulary(),
            SourceSpecialTokenPolicy::OrdinaryText => &self.ordinary_added,
        };
        let mut pretokenized: PreTokenizedString =
            added.extract_and_normalize(self.pipeline.get_normalizer(), text);
        if let Some(pretokenizer) = self.pipeline.get_pre_tokenizer() {
            pretokenizer
                .pre_tokenize(&mut pretokenized)
                .map_err(|error| TokenizationError::Operator(error.to_string()))?;
        }
        let scratch = Mutex::new((BpeScratch::new(), Vec::new(), Vec::new()));
        let tokenization = pretokenized.tokenize(|normalized| {
            cancellation
                .check()
                .map_err(|error| -> tokenizers::Error { Box::new(error) })?;
            let mut scratch = scratch
                .lock()
                .map_err(|_| -> tokenizers::Error { "Qwen BPE scratch poisoned".into() })?;
            let (bpe, initial, merged) = &mut *scratch;
            initial.clear();
            for (start, character) in normalized.get().char_indices() {
                let end = start + character.len_utf8();
                let token = self
                    .vocabulary
                    .get(character.encode_utf8(&mut [0_u8; 4]))
                    .copied()
                    .ok_or_else(|| -> tokenizers::Error {
                        format!("Qwen byte root {character:?} is absent").into()
                    })?;
                initial.push(TokenSpan {
                    token,
                    start: u32::try_from(start).map_err(|_| -> tokenizers::Error {
                        "Qwen split offset exceeds u32".into()
                    })?,
                    end: u32::try_from(end).map_err(|_| -> tokenizers::Error {
                        "Qwen split offset exceeds u32".into()
                    })?,
                });
            }
            if initial.is_empty() {
                return Ok(Vec::new());
            }
            let mut sink = VecTokenSink::new(merged, initial.len())
                .map_err(|error| -> tokenizers::Error { Box::new(error) })?;
            self.kernel
                .merge_to_sink(initial, OffsetPolicy::Include, bpe, &mut sink, cancellation)
                .map_err(|error| -> tokenizers::Error { Box::new(error) })?;
            sink.output()
                .iter()
                .map(|span| {
                    let id = u32::try_from(span.token.get())
                        .map_err(|_| -> tokenizers::Error { "negative Qwen token ID".into() })?;
                    let value =
                        self.pipeline
                            .id_to_token(id)
                            .ok_or_else(|| -> tokenizers::Error {
                                format!("Qwen token ID {id} is absent").into()
                            })?;
                    Ok(Token::new(
                        id,
                        value,
                        (
                            usize::try_from(span.start).map_err(|_| -> tokenizers::Error {
                                "Qwen token start exceeds usize".into()
                            })?,
                            usize::try_from(span.end).map_err(|_| -> tokenizers::Error {
                                "Qwen token end exceeds usize".into()
                            })?,
                        ),
                    ))
                })
                .collect()
        });
        if let Err(error) = tokenization {
            if cancellation.is_requested() {
                return Err(TokenizationError::Cancelled);
            }
            return Err(TokenizationError::Operator(error.to_string()));
        }
        cancellation.check()?;
        pretokenized
            .into_encoding(None, 0, OffsetType::Byte)
            .map_err(|error| TokenizationError::Operator(error.to_string()))
    }

    fn stream(
        &self,
        source: &[u8],
        policy: &TokenizationPolicy,
        sink: &mut dyn TokenOutputSink,
        cancellation: &CancellationToken,
    ) -> Result<bool, TokenizationError> {
        let encoding = self.encoding(source, policy.source_special_tokens, cancellation)?;
        if encoding.get_ids().len() > MAX_TOKENS_PER_ROW {
            return Err(TokenizationError::Bound {
                field: "Qwen output tokens",
                limit: MAX_TOKENS_PER_ROW,
            });
        }
        let source_end = u32::try_from(source.len()).map_err(|_| TokenizationError::Bound {
            field: "source bytes",
            limit: MAX_ROW_BYTES,
        })?;
        let beginning = match policy.boundary_tokens {
            BoundaryTokenPolicy::Beginning | BoundaryTokenPolicy::Both => {
                Some(self.boundaries.beginning.ok_or_else(|| {
                    TokenizationError::Invalid("beginning token is not configured".to_owned())
                })?)
            }
            BoundaryTokenPolicy::None | BoundaryTokenPolicy::Ending => None,
        };
        let ending = match policy.boundary_tokens {
            BoundaryTokenPolicy::Ending | BoundaryTokenPolicy::Both => {
                Some(self.boundaries.ending.ok_or_else(|| {
                    TokenizationError::Invalid("ending token is not configured".to_owned())
                })?)
            }
            BoundaryTokenPolicy::None | BoundaryTokenPolicy::Beginning => None,
        };
        sink.begin()?;
        if let Some(token) = beginning
            && sink.push(policy_span(token, 0, 0, policy.offsets))? == SinkFlow::Stop
        {
            return Ok(false);
        }
        for (&id, &(start, end)) in encoding.get_ids().iter().zip(encoding.get_offsets()) {
            cancellation.check()?;
            let token =
                TokenId::new(i32::try_from(id).map_err(|_| {
                    TokenizationError::Invalid("Qwen token ID exceeds i32".to_owned())
                })?)
                .map_err(|error| TokenizationError::Invalid(error.to_string()))?;
            let start = u32::try_from(start).map_err(|_| TokenizationError::Bound {
                field: "Qwen source offset",
                limit: MAX_ROW_BYTES,
            })?;
            let end = u32::try_from(end).map_err(|_| TokenizationError::Bound {
                field: "Qwen source offset",
                limit: MAX_ROW_BYTES,
            })?;
            if sink.push(policy_span(token, start, end, policy.offsets))? == SinkFlow::Stop {
                return Ok(false);
            }
        }
        if let Some(token) = ending
            && sink.push(policy_span(token, source_end, source_end, policy.offsets))?
                == SinkFlow::Stop
        {
            return Ok(false);
        }
        Ok(true)
    }
}

impl ExactTokenizer for QwenRankedBpe {
    fn identity(&self) -> &TokenizationIdentity {
        &self.identity
    }

    fn identity_v2(&self) -> Option<&TokenizationIdentityV2> {
        Some(&self.identity_v2)
    }

    fn tokenize_into(
        &self,
        source: &[u8],
        policy: &TokenizationPolicy,
        output: &mut Vec<TokenSpan>,
        cancellation: &CancellationToken,
    ) -> Result<(), TokenizationError> {
        let maximum = MAX_TOKENS_PER_ROW
            .min(source.len().saturating_mul(4).saturating_add(2))
            .max(1);
        let mut sink = VecTokenSink::new(output, maximum)?;
        if !self.stream(source, policy, &mut sink, cancellation)? {
            return Err(TokenizationError::Invalid(
                "vector output unexpectedly requested early stop".to_owned(),
            ));
        }
        validate_spans(source.len(), output)
    }

    fn tokenize_to_sink(
        &self,
        source: &[u8],
        policy: &TokenizationPolicy,
        sink: &mut dyn TokenOutputSink,
        cancellation: &CancellationToken,
    ) -> Result<bool, TokenizationError> {
        self.stream(source, policy, sink, cancellation)
    }

    fn count(
        &self,
        source: &[u8],
        policy: &TokenizationPolicy,
        cancellation: &CancellationToken,
    ) -> Result<CountResult, TokenizationError> {
        let mut sink = CountingSink::new(policy.count_through);
        let complete = self.stream(source, policy, &mut sink, cancellation)?;
        if complete {
            sink.finish();
        }
        Ok(sink.result())
    }
}

fn policy_span(token: TokenId, start: u32, end: u32, offsets: OffsetPolicy) -> TokenSpan {
    if offsets == OffsetPolicy::Include {
        TokenSpan { token, start, end }
    } else {
        TokenSpan {
            token,
            start: 0,
            end: 0,
        }
    }
}

fn validate_qwen_document(document: &Value) -> Result<QwenPretokenizer, TokenizationError> {
    if document.get("version").and_then(Value::as_str) != Some("1.0")
        || document
            .get("truncation")
            .is_some_and(|value| !value.is_null())
        || document
            .get("padding")
            .is_some_and(|value| !value.is_null())
    {
        return Err(TokenizationError::Invalid(
            "Qwen tokenizer must use version 1.0 without truncation or padding".to_owned(),
        ));
    }
    let model = document
        .get("model")
        .and_then(Value::as_object)
        .ok_or_else(|| TokenizationError::Invalid("Qwen BPE model is absent".to_owned()))?;
    let expected_model = [
        ("type", Value::String("BPE".to_owned())),
        ("dropout", Value::Null),
        ("unk_token", Value::Null),
        ("continuing_subword_prefix", Value::String(String::new())),
        ("end_of_word_suffix", Value::String(String::new())),
        ("fuse_unk", Value::Bool(false)),
        ("byte_fallback", Value::Bool(false)),
        ("ignore_merges", Value::Bool(false)),
    ];
    if expected_model
        .iter()
        .any(|(field, expected)| model.get(*field) != Some(expected))
    {
        return Err(TokenizationError::Invalid(
            "Qwen BPE model options are unsupported".to_owned(),
        ));
    }
    if document.get("normalizer") != Some(&serde_json::json!({"type": "NFC"}))
        || document.get("decoder")
            != Some(&serde_json::json!({
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": false,
                "use_regex": false
            }))
    {
        return Err(TokenizationError::Invalid(
            "Qwen NFC normalizer or byte-level decoder differs".to_owned(),
        ));
    }
    let pretokenizers = document
        .pointer("/pre_tokenizer/pretokenizers")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            TokenizationError::Invalid("Qwen pretokenizer sequence is absent".to_owned())
        })?;
    if document
        .pointer("/pre_tokenizer/type")
        .and_then(Value::as_str)
        != Some("Sequence")
        || pretokenizers.len() != 2
        || pretokenizers[1]
            != serde_json::json!({
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": false,
                "use_regex": false
            })
        || pretokenizers[0].get("type").and_then(Value::as_str) != Some("Split")
        || pretokenizers[0].get("behavior").and_then(Value::as_str) != Some("Isolated")
        || pretokenizers[0].get("invert").and_then(Value::as_bool) != Some(false)
    {
        return Err(TokenizationError::Invalid(
            "Qwen split + byte-level pretokenizer differs".to_owned(),
        ));
    }
    match pretokenizers[0]
        .pointer("/pattern/Regex")
        .and_then(Value::as_str)
    {
        Some(QWEN2_PATTERN) => Ok(QwenPretokenizer::Qwen2),
        Some(QWEN35_PATTERN) => Ok(QwenPretokenizer::Qwen35),
        _ => Err(TokenizationError::Invalid(
            "Qwen pretokenizer regex is unsupported".to_owned(),
        )),
    }
}

fn validate_boundaries(
    tokenizer: &Tokenizer,
    boundaries: BpeBoundaryTokens,
) -> Result<(), TokenizationError> {
    for boundary in [boundaries.beginning, boundaries.ending]
        .into_iter()
        .flatten()
    {
        let id = u32::try_from(boundary.get())
            .map_err(|_| TokenizationError::Invalid("negative boundary token".to_owned()))?;
        if tokenizer.id_to_token(id).is_none() {
            return Err(TokenizationError::Invalid(
                "configured boundary token is absent from the tokenizer".to_owned(),
            ));
        }
    }
    Ok(())
}

fn parse_vocabulary(document: &Value) -> Result<BTreeMap<String, TokenId>, TokenizationError> {
    let source = document
        .pointer("/model/vocab")
        .and_then(Value::as_object)
        .ok_or_else(|| TokenizationError::Invalid("Qwen vocabulary is absent".to_owned()))?;
    if source.is_empty() || source.len() > MAX_TOKENS_PER_ROW {
        return Err(TokenizationError::Bound {
            field: "Qwen vocabulary",
            limit: MAX_TOKENS_PER_ROW,
        });
    }
    let mut vocabulary = BTreeMap::new();
    let mut seen = std::collections::HashSet::with_capacity(source.len());
    for (token, id) in source {
        let id = id
            .as_u64()
            .and_then(|id| i32::try_from(id).ok())
            .ok_or_else(|| {
                TokenizationError::Invalid("Qwen vocabulary token ID exceeds i32".to_owned())
            })?;
        if !seen.insert(id) {
            return Err(TokenizationError::Invalid(
                "Qwen vocabulary repeats a token ID".to_owned(),
            ));
        }
        vocabulary.insert(
            token.clone(),
            TokenId::new(id).map_err(|error| TokenizationError::Invalid(error.to_string()))?,
        );
    }
    Ok(vocabulary)
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum MergeEntry {
    Text(String),
    Pair([String; 2]),
}

fn parse_merges(
    document: &Value,
    vocabulary: &BTreeMap<String, TokenId>,
) -> Result<Vec<BpeMerge>, TokenizationError> {
    let entries: Vec<MergeEntry> = serde_json::from_value(
        document
            .pointer("/model/merges")
            .cloned()
            .ok_or_else(|| TokenizationError::Invalid("Qwen merges are absent".to_owned()))?,
    )
    .map_err(|error| TokenizationError::Invalid(format!("Qwen merges: {error}")))?;
    if entries.is_empty() || entries.len() > MAX_TOKENS_PER_ROW {
        return Err(TokenizationError::Bound {
            field: "Qwen merges",
            limit: MAX_TOKENS_PER_ROW,
        });
    }
    entries
        .into_iter()
        .enumerate()
        .map(|(rank, entry)| {
            let (left, right) = match entry {
                MergeEntry::Pair([left, right]) => (left, right),
                MergeEntry::Text(pair) => pair.split_once(' ').map_or_else(
                    || {
                        Err(TokenizationError::Invalid(
                            "Qwen text merge has no separator".to_owned(),
                        ))
                    },
                    |(left, right)| Ok((left.to_owned(), right.to_owned())),
                )?,
            };
            let merged = format!("{left}{right}");
            Ok(BpeMerge {
                left: *vocabulary.get(&left).ok_or_else(|| {
                    TokenizationError::Invalid("Qwen merge left token is absent".to_owned())
                })?,
                right: *vocabulary.get(&right).ok_or_else(|| {
                    TokenizationError::Invalid("Qwen merge right token is absent".to_owned())
                })?,
                merged: *vocabulary.get(&merged).ok_or_else(|| {
                    TokenizationError::Invalid("Qwen merged token is absent".to_owned())
                })?,
                rank: u32::try_from(rank).map_err(|_| TokenizationError::Bound {
                    field: "Qwen merge rank",
                    limit: MAX_TOKENS_PER_ROW,
                })?,
            })
        })
        .collect()
}

fn component_identity(
    domain: &'static str,
    value: Option<&Value>,
) -> Result<Digest, TokenizationError> {
    Digest::of_serializable(
        domain,
        value.ok_or_else(|| TokenizationError::Invalid(format!("{domain} component is absent")))?,
    )
    .map_err(|error| TokenizationError::Identity(error.to_string()))
}

fn special_token_identity(document: &Value) -> Result<Digest, TokenizationError> {
    let added = document
        .get("added_tokens")
        .and_then(Value::as_array)
        .ok_or_else(|| TokenizationError::Invalid("Qwen added tokens are absent".to_owned()))?;
    let special = added
        .iter()
        .filter(|token| token.get("special").and_then(Value::as_bool) == Some(true))
        .collect::<Vec<_>>();
    Digest::of_serializable("qwen-tokenizer-special-tokens-v1", &special)
        .map_err(|error| TokenizationError::Identity(error.to_string()))
}

#[cfg(test)]
mod tests {
    use tokenizers::AddedToken;
    use tokenizers::models::bpe::{BPE, Vocab};
    use tokenizers::normalizers::unicode::NFC;
    use tokenizers::pre_tokenizers::{
        PreTokenizerWrapper, byte_level::ByteLevel, sequence::Sequence, split::Split,
    };
    use tokenizers::{SplitDelimiterBehavior, Tokenizer};

    use super::*;

    fn fixture() -> (Vec<u8>, u32) {
        let mut alphabet = ByteLevel::alphabet().into_iter().collect::<Vec<_>>();
        alphabet.sort_unstable();
        let mut vocabulary = alphabet
            .into_iter()
            .enumerate()
            .map(|(id, character)| {
                (
                    character.to_string(),
                    u32::try_from(id).expect("byte alphabet fits u32"),
                )
            })
            .collect::<Vocab>();
        let merged = u32::try_from(vocabulary.len()).unwrap();
        vocabulary.insert("ab".to_owned(), merged);
        let bpe = BPE::builder()
            .vocab_and_merges(vocabulary, vec![("a".to_owned(), "b".to_owned())])
            .continuing_subword_prefix(String::new())
            .end_of_word_suffix(String::new())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(bpe);
        tokenizer
            .with_normalizer(Some(NFC))
            .unwrap()
            .with_pre_tokenizer(Some(Sequence::new(vec![
                PreTokenizerWrapper::Split(
                    Split::new(
                        tokenizers::pre_tokenizers::split::SplitPattern::Regex(
                            QWEN2_PATTERN.to_owned(),
                        ),
                        SplitDelimiterBehavior::Isolated,
                        false,
                    )
                    .unwrap(),
                ),
                PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)),
            ])))
            .with_post_processor(Some(ByteLevel::new(false, false, false)))
            .with_decoder(Some(ByteLevel::new(false, false, false)));
        tokenizer
            .add_special_tokens([AddedToken::from("<|im_start|>", true).normalized(false)])
            .unwrap();
        let special = tokenizer.token_to_id("<|im_start|>").unwrap();
        (serde_json::to_vec(&tokenizer).unwrap(), special)
    }

    fn policy(
        boundaries: BoundaryTokenPolicy,
        specials: SourceSpecialTokenPolicy,
        offsets: OffsetPolicy,
        count_through: Option<u64>,
    ) -> TokenizationPolicy {
        TokenizationPolicy {
            boundary_tokens: boundaries,
            source_special_tokens: specials,
            offsets,
            count_through,
        }
    }

    fn reference(bytes: &[u8], source: &str, specials: SourceSpecialTokenPolicy) -> Vec<TokenSpan> {
        let mut tokenizer = Tokenizer::from_bytes(bytes).unwrap();
        tokenizer.set_encode_special_tokens(specials == SourceSpecialTokenPolicy::OrdinaryText);
        let encoding = tokenizer.encode(source, false).unwrap();
        encoding
            .get_ids()
            .iter()
            .zip(encoding.get_offsets())
            .map(|(&id, &(start, end))| TokenSpan {
                token: TokenId::new(i32::try_from(id).unwrap()).unwrap(),
                start: u32::try_from(start).unwrap(),
                end: u32::try_from(end).unwrap(),
            })
            .collect()
    }

    #[test]
    fn qwen_ranked_path_matches_reference_ids_offsets_and_special_policy() {
        let (bytes, _) = fixture();
        let adapter = QwenRankedBpe::from_tokenizer_json(
            QwenTokenizerConfig {
                model: Digest::of_bytes("model", b"fixture"),
                boundaries: BpeBoundaryTokens::default(),
            },
            &bytes,
        )
        .unwrap();
        assert_eq!(adapter.pretokenizer(), QwenPretokenizer::Qwen2);
        for source in [
            "ab ab!",
            "café cafe\u{301}",
            "中文 👩🏽‍💻",
            "<|im_start|>ab",
            "\r\n \t",
        ] {
            for specials in [
                SourceSpecialTokenPolicy::OrdinaryText,
                SourceSpecialTokenPolicy::RecognizeConfigured,
            ] {
                let mut actual = Vec::new();
                adapter
                    .tokenize_into(
                        source.as_bytes(),
                        &policy(
                            BoundaryTokenPolicy::None,
                            specials,
                            OffsetPolicy::Include,
                            None,
                        ),
                        &mut actual,
                        &CancellationToken::default(),
                    )
                    .unwrap();
                assert_eq!(actual, reference(&bytes, source, specials));
            }
        }
    }

    #[test]
    fn qwen_boundaries_count_cancellation_and_hostile_input_fail_closed() {
        let (bytes, special) = fixture();
        let boundary = TokenId::new(i32::try_from(special).unwrap()).unwrap();
        let adapter = QwenRankedBpe::from_tokenizer_json(
            QwenTokenizerConfig {
                model: Digest::of_bytes("model", b"fixture"),
                boundaries: BpeBoundaryTokens {
                    beginning: Some(boundary),
                    ending: Some(boundary),
                },
            },
            &bytes,
        )
        .unwrap();
        let both = policy(
            BoundaryTokenPolicy::Both,
            SourceSpecialTokenPolicy::RecognizeConfigured,
            OffsetPolicy::Include,
            None,
        );
        let mut tokens = Vec::new();
        adapter
            .tokenize_into(b"ab", &both, &mut tokens, &CancellationToken::default())
            .unwrap();
        assert_eq!(tokens.first().unwrap().start, 0);
        assert_eq!(tokens.first().unwrap().end, 0);
        assert_eq!(tokens.last().unwrap().start, 2);
        assert_eq!(tokens.last().unwrap().end, 2);
        let count = adapter
            .count(
                b"abcdef",
                &policy(
                    BoundaryTokenPolicy::None,
                    SourceSpecialTokenPolicy::OrdinaryText,
                    OffsetPolicy::Omit,
                    Some(2),
                ),
                &CancellationToken::default(),
            )
            .unwrap();
        assert_eq!(count.count, 3);
        assert!(!count.complete);
        assert!(
            adapter
                .count(b"a\0b", &both, &CancellationToken::default(),)
                .is_err()
        );
        let cancellation = CancellationToken::default();
        cancellation.request();
        assert_eq!(
            adapter.count(b"ab", &both, &cancellation),
            Err(TokenizationError::Cancelled)
        );
    }

    #[test]
    fn qwen_adapter_rejects_changed_regex_and_unknown_boundaries() {
        let (bytes, _) = fixture();
        let mut changed: Value = serde_json::from_slice(&bytes).unwrap();
        changed["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] =
            Value::String(".*".to_owned());
        assert!(
            QwenRankedBpe::from_tokenizer_json(
                QwenTokenizerConfig {
                    model: Digest::of_bytes("model", b"fixture"),
                    boundaries: BpeBoundaryTokens::default(),
                },
                &serde_json::to_vec(&changed).unwrap(),
            )
            .is_err()
        );
        assert!(
            QwenRankedBpe::from_tokenizer_json(
                QwenTokenizerConfig {
                    model: Digest::of_bytes("model", b"fixture"),
                    boundaries: BpeBoundaryTokens {
                        beginning: Some(TokenId::new(i32::MAX).unwrap()),
                        ending: None,
                    },
                },
                &bytes,
            )
            .is_err()
        );
    }
}

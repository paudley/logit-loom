// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation-reusing and direct token output sinks.

use logit_loom_core::TokenId;

use crate::{
    CountResult, MAX_TOKENS_PER_ROW, OffsetPolicy, TokenSpan, TokenizationError, TokenizationPolicy,
};

/// Whether an output sink needs another token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SinkFlow {
    /// Continue producing tokens.
    Continue,
    /// The sink has enough information and requests an inclusive early stop.
    Stop,
}

/// Caller-owned destination for one token stream.
pub trait TokenOutputSink {
    /// Clears prior per-row state before the first token.
    ///
    /// # Errors
    ///
    /// Returns an error when the destination cannot begin another row.
    fn begin(&mut self) -> Result<(), TokenizationError>;

    /// Retains one exact token in source order.
    ///
    /// # Errors
    ///
    /// Returns an error when the destination bound is exhausted.
    fn push(&mut self, token: TokenSpan) -> Result<SinkFlow, TokenizationError>;
}

/// Reusable bounded sink backed by a caller-owned vector.
#[derive(Debug)]
pub struct VecTokenSink<'a> {
    output: &'a mut Vec<TokenSpan>,
    maximum_tokens: usize,
}

impl<'a> VecTokenSink<'a> {
    /// Binds reusable output capacity to an explicit token bound.
    ///
    /// # Errors
    ///
    /// Returns an error for a zero or excessive bound.
    pub fn new(
        output: &'a mut Vec<TokenSpan>,
        maximum_tokens: usize,
    ) -> Result<Self, TokenizationError> {
        if maximum_tokens == 0 || maximum_tokens > MAX_TOKENS_PER_ROW {
            return Err(TokenizationError::Bound {
                field: "sink tokens",
                limit: MAX_TOKENS_PER_ROW,
            });
        }
        Ok(Self {
            output,
            maximum_tokens,
        })
    }

    /// Returns the retained exact token spans.
    pub fn output(&self) -> &[TokenSpan] {
        self.output
    }
}

impl TokenOutputSink for VecTokenSink<'_> {
    fn begin(&mut self) -> Result<(), TokenizationError> {
        self.output.clear();
        Ok(())
    }

    fn push(&mut self, token: TokenSpan) -> Result<SinkFlow, TokenizationError> {
        if self.output.len() >= self.maximum_tokens {
            return Err(TokenizationError::Bound {
                field: "sink tokens",
                limit: self.maximum_tokens,
            });
        }
        self.output.push(token);
        Ok(SinkFlow::Continue)
    }
}

/// Direct caller-sized token-ID destination.
#[derive(Debug)]
pub struct TokenIdSliceSink<'a> {
    tokens: &'a mut [TokenId],
    offsets: Option<&'a mut [(u32, u32)]>,
    written: usize,
}

impl<'a> TokenIdSliceSink<'a> {
    /// Binds caller-owned token and optional offset storage.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty token slice or an offset slice of another
    /// length.
    pub fn new(
        tokens: &'a mut [TokenId],
        offsets: Option<&'a mut [(u32, u32)]>,
    ) -> Result<Self, TokenizationError> {
        if tokens.is_empty()
            || offsets
                .as_ref()
                .is_some_and(|value| value.len() != tokens.len())
        {
            return Err(TokenizationError::Invalid(
                "token-ID sink storage is empty or mismatched".to_owned(),
            ));
        }
        Ok(Self {
            tokens,
            offsets,
            written: 0,
        })
    }

    /// Returns the initialized token-ID prefix.
    pub fn tokens(&self) -> &[TokenId] {
        &self.tokens[..self.written]
    }

    /// Returns the initialized offset prefix when requested.
    pub fn offsets(&self) -> Option<&[(u32, u32)]> {
        self.offsets
            .as_deref()
            .map(|offsets| &offsets[..self.written])
    }
}

impl TokenOutputSink for TokenIdSliceSink<'_> {
    fn begin(&mut self) -> Result<(), TokenizationError> {
        self.written = 0;
        Ok(())
    }

    fn push(&mut self, token: TokenSpan) -> Result<SinkFlow, TokenizationError> {
        let limit = self.tokens.len();
        let destination = self
            .tokens
            .get_mut(self.written)
            .ok_or(TokenizationError::Bound {
                field: "token-ID sink tokens",
                limit,
            })?;
        *destination = token.token;
        if let Some(offsets) = &mut self.offsets {
            offsets[self.written] = (token.start, token.end);
        }
        self.written += 1;
        Ok(SinkFlow::Continue)
    }
}

/// Count-only sink with an optional inclusive early-stop threshold.
#[derive(Clone, Copy, Debug)]
pub struct CountingSink {
    count_through: Option<u64>,
    count: u64,
    complete: bool,
}

impl CountingSink {
    /// Creates an exact or inclusive-threshold count destination.
    pub const fn new(count_through: Option<u64>) -> Self {
        Self {
            count_through,
            count: 0,
            complete: false,
        }
    }

    /// Returns current count accounting.
    pub const fn result(self) -> CountResult {
        CountResult {
            count: self.count,
            complete: self.complete,
        }
    }

    /// Marks the complete source as consumed.
    pub const fn finish(&mut self) {
        self.complete = true;
    }
}

impl TokenOutputSink for CountingSink {
    fn begin(&mut self) -> Result<(), TokenizationError> {
        self.count = 0;
        self.complete = false;
        Ok(())
    }

    fn push(&mut self, _token: TokenSpan) -> Result<SinkFlow, TokenizationError> {
        self.count = self.count.checked_add(1).ok_or(TokenizationError::Bound {
            field: "token count",
            limit: MAX_TOKENS_PER_ROW,
        })?;
        Ok(
            if self.count_through.is_some_and(|limit| self.count > limit) {
                SinkFlow::Stop
            } else {
                SinkFlow::Continue
            },
        )
    }
}

/// Streams an exact tokenizer result through a reusable scratch vector into a
/// caller-owned destination.
///
/// Backends with a direct/vector-free path should override
/// [`crate::ExactTokenizer::tokenize_to_sink`]. This helper is the allocation-
/// reusing compatibility path for implementations that only provide
/// `tokenize_into`.
///
/// # Errors
///
/// Returns a tokenizer, offset, cancellation, or sink-bound failure.
pub fn tokenize_via_scratch(
    tokenizer: &dyn crate::ExactTokenizer,
    source: &[u8],
    policy: &TokenizationPolicy,
    scratch: &mut Vec<TokenSpan>,
    sink: &mut dyn TokenOutputSink,
    cancellation: &crate::CancellationToken,
) -> Result<bool, TokenizationError> {
    scratch.clear();
    tokenizer.tokenize_into(source, policy, scratch, cancellation)?;
    crate::operator::validate_spans(source.len(), scratch)?;
    sink.begin()?;
    for token in scratch.iter().copied() {
        if sink.push(token)? == SinkFlow::Stop {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Validates whether a sink's retained offsets match the requested policy.
///
/// # Errors
///
/// Returns an error when omitted offsets contain nonzero source ranges.
pub fn validate_sink_policy(
    policy: &TokenizationPolicy,
    tokens: &[TokenSpan],
) -> Result<(), TokenizationError> {
    if policy.offsets == OffsetPolicy::Omit
        && tokens
            .iter()
            .any(|token| token.start != 0 || token.end != 0)
    {
        return Err(TokenizationError::Invalid(
            "offset-omitting output contains source ranges".to_owned(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counting_sink_stops_at_first_count_above_inclusive_threshold() {
        let mut sink = CountingSink::new(Some(2));
        sink.begin().unwrap();
        let token = TokenSpan {
            token: TokenId::new(1).unwrap(),
            start: 0,
            end: 1,
        };
        assert_eq!(sink.push(token).unwrap(), SinkFlow::Continue);
        assert_eq!(sink.push(token).unwrap(), SinkFlow::Continue);
        assert_eq!(sink.push(token).unwrap(), SinkFlow::Stop);
        assert_eq!(
            sink.result(),
            CountResult {
                count: 3,
                complete: false,
            }
        );
    }

    #[test]
    fn slice_sink_never_writes_past_caller_capacity() {
        let zero = TokenId::new(0).unwrap();
        let mut tokens = [zero; 1];
        let mut offsets = [(0, 0); 1];
        let mut sink = TokenIdSliceSink::new(&mut tokens, Some(&mut offsets)).unwrap();
        sink.begin().unwrap();
        let span = TokenSpan {
            token: TokenId::new(7).unwrap(),
            start: 1,
            end: 2,
        };
        sink.push(span).unwrap();
        assert!(sink.push(span).is_err());
        assert_eq!(sink.tokens(), &[TokenId::new(7).unwrap()]);
        assert_eq!(sink.offsets(), Some(&[(1, 2)][..]));
    }
}

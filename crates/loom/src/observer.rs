// SPDX-License-Identifier: MIT OR Apache-2.0

//! Generated-token and prefill observation with cooperative cancellation.

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use logit_loom_core::{
    CallbackFailure, CallbackPhase, ControlFlow, Digest, ObserverReceipt, PrefillFinish,
    PrefillProgress, PrefillReceipt, TokenId,
};

use crate::{ObserverError, PrefillObserverError};

/// Maximum observers sharing one ordered fan-out.
pub const MAX_OBSERVERS: usize = 16;

/// One token already admitted to backend causal state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObservedToken<'a> {
    /// Tokenizer identifier.
    pub token: TokenId,
    /// Exact, potentially non-UTF-8 token-piece bytes.
    pub piece: &'a [u8],
    /// Causal position after token admission.
    pub position: u64,
}

/// Safe synchronous observer for generated-token boundaries.
pub trait Observer {
    /// Resets local state for a new call.
    ///
    /// # Errors
    ///
    /// Returns an implementation-defined reset failure.
    fn reset(&mut self, _initial_position: u64) -> Result<(), ObserverError> {
        Ok(())
    }

    /// Polls immediately before one sampling decision.
    ///
    /// # Errors
    ///
    /// Returns an implementation-defined polling failure.
    fn poll(&mut self) -> Result<ControlFlow, ObserverError> {
        Ok(ControlFlow::Continue)
    }

    /// Observes one token after causal admission.
    ///
    /// # Errors
    ///
    /// Returns an implementation-defined observation failure.
    fn on_token(&mut self, token: ObservedToken<'_>) -> Result<ControlFlow, ObserverError>;
}

struct ObserverEntry {
    implementation: Digest,
    observer: Box<dyn Observer>,
    receipt: ObserverReceipt,
}

/// Ordered observer fan-out behind one backend callback point.
pub struct ObserverSet {
    entries: Vec<ObserverEntry>,
    begun: bool,
    failed: bool,
}

impl ObserverSet {
    /// Creates a non-empty bounded observer set.
    ///
    /// # Errors
    ///
    /// Returns an error unless one to 16 observers are supplied and identities
    /// are unique.
    pub fn new(
        observers: impl IntoIterator<Item = (Digest, Box<dyn Observer>)>,
    ) -> Result<Self, ObserverError> {
        let entries = observers
            .into_iter()
            .take(MAX_OBSERVERS + 1)
            .map(|(implementation, observer)| ObserverEntry {
                receipt: empty_observer_receipt(implementation.clone()),
                implementation,
                observer,
            })
            .collect::<Vec<_>>();
        if !(1..=MAX_OBSERVERS).contains(&entries.len()) {
            return Err(ObserverError::new(format!(
                "observer set requires 1..={MAX_OBSERVERS} entries"
            )));
        }
        let mut identities = entries
            .iter()
            .map(|entry| entry.implementation.clone())
            .collect::<Vec<_>>();
        identities.sort();
        identities.dedup();
        if identities.len() != entries.len() {
            return Err(ObserverError::new(
                "observer implementation identities must be unique",
            ));
        }
        Ok(Self {
            entries,
            begun: false,
            failed: false,
        })
    }

    /// Convenience constructor for one observer.
    pub fn single(implementation: Digest, observer: impl Observer + 'static) -> Self {
        Self {
            entries: vec![ObserverEntry {
                receipt: empty_observer_receipt(implementation.clone()),
                implementation,
                observer: Box::new(observer),
            }],
            begun: false,
            failed: false,
        }
    }

    /// Resets every observer for a new generation call.
    ///
    /// # Errors
    ///
    /// Returns a bounded error or panic from one observer.
    pub fn begin(
        &mut self,
        initial_position: u64,
        requested_tokens: u32,
    ) -> Result<(), ObserverError> {
        if requested_tokens == 0 {
            return Err(ObserverError::new(
                "requested token bound must be greater than zero",
            ));
        }
        if initial_position
            .checked_add(u64::from(requested_tokens))
            .is_none()
        {
            return Err(ObserverError::new(
                "requested token range overflows causal position",
            ));
        }
        self.begun = true;
        self.failed = false;
        for entry in &mut self.entries {
            entry.receipt = ObserverReceipt {
                implementation: entry.implementation.clone(),
                initial_position,
                requested_tokens,
                polls: 0,
                observed_tokens: 0,
                observed_bytes: 0,
                final_position: initial_position,
                stop_requested: false,
                failure: None,
            };
            let result = catch_unwind(AssertUnwindSafe(|| entry.observer.reset(initial_position)));
            if let Some(failure) = observer_result(CallbackPhase::Reset, result) {
                entry.receipt.failure = Some(failure.clone());
                self.failed = true;
                return Err(ObserverError::new(failure.message));
            }
        }
        Ok(())
    }

    /// Polls every observer in order before one sampling decision.
    ///
    /// All observers run even after an earlier observer requests a stop. This
    /// gives every sink the same pre-sampling boundary. Poll count cannot
    /// exceed the token bound supplied to [`Self::begin`].
    ///
    /// # Errors
    ///
    /// Returns a lifecycle or contained callback failure.
    pub fn poll(&mut self) -> Result<ControlFlow, ObserverError> {
        self.require_ready()?;
        if self
            .entries
            .iter()
            .any(|entry| entry.receipt.polls >= entry.receipt.requested_tokens)
        {
            self.failed = true;
            return Err(ObserverError::new(
                "observer polls exceed the requested token bound",
            ));
        }
        let mut control = ControlFlow::Continue;
        for entry in &mut self.entries {
            entry.receipt.polls += 1;
            let result = catch_unwind(AssertUnwindSafe(|| entry.observer.poll()));
            match observer_control_result(CallbackPhase::Poll, result) {
                Ok(ControlFlow::Continue) => {}
                Ok(ControlFlow::Stop) => {
                    entry.receipt.stop_requested = true;
                    control = ControlFlow::Stop;
                }
                Err(failure) => {
                    entry.receipt.failure = Some(failure.clone());
                    self.failed = true;
                    return Err(ObserverError::new(failure.message));
                }
            }
        }
        Ok(control)
    }

    /// Delivers one causally admitted token to every observer in order.
    ///
    /// A successful poll must precede delivery, and delivery cannot exceed the
    /// token bound supplied to [`Self::begin`].
    ///
    /// # Errors
    ///
    /// Returns a lifecycle or contained callback failure.
    pub fn observe(&mut self, token: ObservedToken<'_>) -> Result<ControlFlow, ObserverError> {
        self.require_ready()?;
        let Some(first) = self.entries.first() else {
            self.failed = true;
            return Err(ObserverError::new("observer set is empty"));
        };
        if first.receipt.polls <= first.receipt.observed_tokens {
            self.failed = true;
            return Err(ObserverError::new(
                "poll every observer before delivering the next token",
            ));
        }
        if first.receipt.observed_tokens >= first.receipt.requested_tokens {
            self.failed = true;
            return Err(ObserverError::new(
                "observed token count exceeds the requested bound",
            ));
        }
        let next_observed_tokens = first
            .receipt
            .observed_tokens
            .checked_add(1)
            .ok_or_else(|| ObserverError::new("observed token count overflowed"))?;
        let expected_position = first
            .receipt
            .initial_position
            .checked_add(u64::from(next_observed_tokens))
            .ok_or_else(|| ObserverError::new("observed causal position overflowed"))?;
        if token.position != expected_position {
            self.failed = true;
            return Err(ObserverError::new(format!(
                "observed position {} does not equal expected {expected_position}",
                token.position
            )));
        }
        let piece_bytes = u64::try_from(token.piece.len())
            .map_err(|_| ObserverError::new("observed byte count overflowed"))?;
        let next_observed_bytes = first
            .receipt
            .observed_bytes
            .checked_add(piece_bytes)
            .ok_or_else(|| ObserverError::new("observed byte count overflowed"))?;
        let mut control = ControlFlow::Continue;
        for entry in &mut self.entries {
            entry.receipt.observed_tokens = next_observed_tokens;
            entry.receipt.observed_bytes = next_observed_bytes;
            entry.receipt.final_position = token.position;
            let result = catch_unwind(AssertUnwindSafe(|| entry.observer.on_token(token)));
            match observer_control_result(CallbackPhase::Observe, result) {
                Ok(ControlFlow::Continue) => {}
                Ok(ControlFlow::Stop) => {
                    entry.receipt.stop_requested = true;
                    control = ControlFlow::Stop;
                }
                Err(failure) => {
                    entry.receipt.failure = Some(failure.clone());
                    self.failed = true;
                    return Err(ObserverError::new(failure.message));
                }
            }
        }
        Ok(control)
    }

    /// Returns current per-observer accounting in dispatch order.
    pub fn receipts(&self) -> Vec<ObserverReceipt> {
        self.entries
            .iter()
            .map(|entry| entry.receipt.clone())
            .collect()
    }

    /// Returns one observer receipt by dispatch index.
    pub fn receipt(&self, index: usize) -> Option<&ObserverReceipt> {
        self.entries.get(index).map(|entry| &entry.receipt)
    }

    fn require_ready(&self) -> Result<(), ObserverError> {
        if !self.begun {
            return Err(ObserverError::new("observer set has not begun"));
        }
        if self.failed {
            return Err(ObserverError::new(
                "observer set is failed; call begin before reuse",
            ));
        }
        if self
            .entries
            .iter()
            .any(|entry| entry.receipt.stop_requested)
        {
            return Err(ObserverError::new(
                "observer set requested a stop; call begin before reuse",
            ));
        }
        Ok(())
    }
}

/// Safe synchronous observer for complete prefill chunks.
pub trait PrefillObserver {
    /// Resets implementation-local state for a new prefill call.
    ///
    /// # Errors
    ///
    /// Returns an implementation-defined reset failure.
    fn reset(&mut self, _progress: PrefillProgress) -> Result<(), PrefillObserverError> {
        Ok(())
    }

    /// Polls before one native chunk is submitted.
    ///
    /// # Errors
    ///
    /// Returns an implementation-defined polling failure.
    fn poll(&mut self, _progress: PrefillProgress) -> Result<ControlFlow, PrefillObserverError> {
        Ok(ControlFlow::Continue)
    }

    /// Observes one complete chunk after causal admission.
    ///
    /// # Errors
    ///
    /// Returns an implementation-defined observation failure.
    fn on_chunk(&mut self, progress: PrefillProgress) -> Result<ControlFlow, PrefillObserverError>;
}

/// Panic-containing wrapper around one controlled-prefill observer.
pub struct PrefillMonitor {
    implementation: Digest,
    observer: Box<dyn PrefillObserver>,
    receipt: Option<PrefillReceipt>,
    failed: bool,
    stop_requested: bool,
    finished: bool,
}

impl PrefillMonitor {
    /// Creates one identified prefill observer.
    pub fn new(implementation: Digest, observer: impl PrefillObserver + 'static) -> Self {
        Self {
            implementation,
            observer: Box::new(observer),
            receipt: None,
            failed: false,
            stop_requested: false,
            finished: false,
        }
    }

    /// Returns current accounting, including a contained failure when present.
    pub const fn receipt(&self) -> Option<&PrefillReceipt> {
        self.receipt.as_ref()
    }

    /// Begins a prefill operation at zero admitted tokens.
    ///
    /// # Errors
    ///
    /// Returns a validation or contained reset failure.
    pub fn begin(&mut self, progress: PrefillProgress) -> Result<(), PrefillObserverError> {
        progress
            .validate()
            .map_err(|error| PrefillObserverError::new(error.to_string()))?;
        if progress.requested_tokens == 0
            || progress.admitted_tokens != 0
            || progress.admitted_chunks != 0
        {
            return Err(PrefillObserverError::new(
                "initial prefill progress requires requested tokens and no admitted work",
            ));
        }
        self.failed = false;
        self.stop_requested = false;
        self.finished = false;
        self.receipt = Some(PrefillReceipt {
            implementation: self.implementation.clone(),
            progress,
            finish: PrefillFinish::InProgress,
            polls: 0,
            chunks: 0,
            failure: None,
        });
        let result = catch_unwind(AssertUnwindSafe(|| self.observer.reset(progress)));
        if let Some(failure) = prefill_result(CallbackPhase::Reset, result) {
            self.fail(failure.clone());
            return Err(PrefillObserverError::new(failure.message));
        }
        Ok(())
    }

    /// Polls before a chunk.
    ///
    /// # Errors
    ///
    /// Returns a lifecycle or contained callback failure.
    pub fn poll(&mut self, progress: PrefillProgress) -> Result<ControlFlow, PrefillObserverError> {
        self.require_continuable()?;
        progress
            .validate()
            .map_err(|error| PrefillObserverError::new(error.to_string()))?;
        let Some(receipt) = &self.receipt else {
            return Err(PrefillObserverError::new("prefill monitor has not begun"));
        };
        if progress != receipt.progress {
            return Err(PrefillObserverError::new(
                "polled prefill progress does not match admitted work",
            ));
        }
        if receipt.polls != receipt.chunks {
            return Err(PrefillObserverError::new(
                "each prefill chunk requires exactly one preceding poll",
            ));
        }
        let next_poll = receipt
            .polls
            .checked_add(1)
            .ok_or_else(|| PrefillObserverError::new("prefill poll count overflowed"))?;
        if let Some(receipt) = &mut self.receipt {
            receipt.polls = next_poll;
        }
        let result = catch_unwind(AssertUnwindSafe(|| self.observer.poll(progress)));
        match prefill_control_result(CallbackPhase::Poll, result) {
            Ok(control) => {
                if control == ControlFlow::Stop {
                    self.stop_requested = true;
                }
                Ok(control)
            }
            Err(failure) => {
                self.fail(failure.clone());
                Err(PrefillObserverError::new(failure.message))
            }
        }
    }

    /// Observes a complete admitted chunk.
    ///
    /// # Errors
    ///
    /// Returns a lifecycle or contained callback failure.
    pub fn observe_chunk(
        &mut self,
        progress: PrefillProgress,
    ) -> Result<ControlFlow, PrefillObserverError> {
        self.require_continuable()?;
        progress
            .validate()
            .map_err(|error| PrefillObserverError::new(error.to_string()))?;
        let Some(receipt) = &mut self.receipt else {
            return Err(PrefillObserverError::new("prefill monitor has not begun"));
        };
        let previous = receipt.progress;
        let expected_chunks = previous
            .admitted_chunks
            .checked_add(1)
            .ok_or_else(|| PrefillObserverError::new("prefill chunk count overflowed"))?;
        let expected_polls = receipt
            .chunks
            .checked_add(1)
            .ok_or_else(|| PrefillObserverError::new("prefill poll count overflowed"))?;
        if progress.initial_position != previous.initial_position
            || progress.requested_tokens != previous.requested_tokens
            || progress.admitted_tokens <= previous.admitted_tokens
            || progress.admitted_chunks != expected_chunks
            || receipt.polls != expected_polls
        {
            return Err(PrefillObserverError::new(
                "admitted prefill chunks must follow polled progress monotonically",
            ));
        }
        receipt.progress = progress;
        receipt.chunks = expected_chunks;
        let result = catch_unwind(AssertUnwindSafe(|| self.observer.on_chunk(progress)));
        match prefill_control_result(CallbackPhase::Prefill, result) {
            Ok(control) => {
                if control == ControlFlow::Stop {
                    self.stop_requested = true;
                }
                Ok(control)
            }
            Err(failure) => {
                self.fail(failure.clone());
                Err(PrefillObserverError::new(failure.message))
            }
        }
    }

    /// Marks successful complete or stopped prefill and returns its receipt.
    ///
    /// `Stopped` requires a stop returned by the wrapped observer. A successful
    /// finish is terminal until [`Self::begin`] starts another call.
    ///
    /// # Errors
    ///
    /// Returns an error when the monitor failed or has not begun.
    pub fn finish(
        &mut self,
        finish: PrefillFinish,
    ) -> Result<PrefillReceipt, PrefillObserverError> {
        self.require_ready()?;
        if finish == PrefillFinish::InProgress {
            return Err(PrefillObserverError::new(
                "in-progress prefill is not a terminal finish",
            ));
        }
        if finish == PrefillFinish::Stopped && !self.stop_requested {
            return Err(PrefillObserverError::new(
                "stopped prefill requires an observer stop request",
            ));
        }
        let Some(mut receipt) = self.receipt.clone() else {
            return Err(PrefillObserverError::new("prefill monitor has not begun"));
        };
        receipt.finish = finish;
        receipt
            .digest()
            .map_err(|error| PrefillObserverError::new(error.to_string()))?;
        self.receipt = Some(receipt.clone());
        self.finished = true;
        Ok(receipt)
    }

    fn require_ready(&self) -> Result<(), PrefillObserverError> {
        if self.receipt.is_none() {
            return Err(PrefillObserverError::new("prefill monitor has not begun"));
        }
        if self.failed {
            return Err(PrefillObserverError::new(
                "prefill monitor is failed; call begin before reuse",
            ));
        }
        if self.finished {
            return Err(PrefillObserverError::new(
                "prefill monitor has finished; call begin before reuse",
            ));
        }
        Ok(())
    }

    fn require_continuable(&self) -> Result<(), PrefillObserverError> {
        self.require_ready()?;
        if self.stop_requested {
            return Err(PrefillObserverError::new(
                "prefill observer requested a stop; finish before reuse",
            ));
        }
        Ok(())
    }

    fn fail(&mut self, failure: CallbackFailure) {
        self.failed = true;
        if let Some(receipt) = &mut self.receipt {
            receipt.failure = Some(failure);
        }
    }
}

/// Cloneable one-shot cooperative cancellation signal.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken {
    stopped: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Creates a signal in the running state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Requests cancellation and reports whether this call changed the state.
    pub fn cancel(&self) -> bool {
        !self.stopped.swap(true, Ordering::AcqRel)
    }

    /// Returns whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.stopped.load(Ordering::Acquire)
    }

    fn control(&self) -> ControlFlow {
        if self.is_cancelled() {
            ControlFlow::Stop
        } else {
            ControlFlow::Continue
        }
    }
}

impl Observer for CancellationToken {
    fn poll(&mut self) -> Result<ControlFlow, ObserverError> {
        Ok(self.control())
    }

    fn on_token(&mut self, _token: ObservedToken<'_>) -> Result<ControlFlow, ObserverError> {
        Ok(self.control())
    }
}

impl PrefillObserver for CancellationToken {
    fn poll(&mut self, _progress: PrefillProgress) -> Result<ControlFlow, PrefillObserverError> {
        Ok(self.control())
    }

    fn on_chunk(
        &mut self,
        _progress: PrefillProgress,
    ) -> Result<ControlFlow, PrefillObserverError> {
        Ok(self.control())
    }
}

fn empty_observer_receipt(implementation: Digest) -> ObserverReceipt {
    ObserverReceipt {
        implementation,
        initial_position: 0,
        requested_tokens: 1,
        polls: 0,
        observed_tokens: 0,
        observed_bytes: 0,
        final_position: 0,
        stop_requested: false,
        failure: None,
    }
}

fn observer_result(
    phase: CallbackPhase,
    result: Result<Result<(), ObserverError>, Box<dyn Any + Send>>,
) -> Option<CallbackFailure> {
    match result {
        Ok(Ok(())) => None,
        Ok(Err(error)) => Some(CallbackFailure::new(phase, false, error.to_string())),
        Err(payload) => Some(CallbackFailure::new(phase, true, panic_message(&*payload))),
    }
}

fn observer_control_result(
    phase: CallbackPhase,
    result: Result<Result<ControlFlow, ObserverError>, Box<dyn Any + Send>>,
) -> Result<ControlFlow, CallbackFailure> {
    match result {
        Ok(Ok(control)) => Ok(control),
        Ok(Err(error)) => Err(CallbackFailure::new(phase, false, error.to_string())),
        Err(payload) => Err(CallbackFailure::new(phase, true, panic_message(&*payload))),
    }
}

fn prefill_result(
    phase: CallbackPhase,
    result: Result<Result<(), PrefillObserverError>, Box<dyn Any + Send>>,
) -> Option<CallbackFailure> {
    match result {
        Ok(Ok(())) => None,
        Ok(Err(error)) => Some(CallbackFailure::new(phase, false, error.to_string())),
        Err(payload) => Some(CallbackFailure::new(phase, true, panic_message(&*payload))),
    }
}

fn prefill_control_result(
    phase: CallbackPhase,
    result: Result<Result<ControlFlow, PrefillObserverError>, Box<dyn Any + Send>>,
) -> Result<ControlFlow, CallbackFailure> {
    match result {
        Ok(Ok(control)) => Ok(control),
        Ok(Err(error)) => Err(CallbackFailure::new(phase, false, error.to_string())),
        Err(payload) => Err(CallbackFailure::new(phase, true, panic_message(&*payload))),
    }
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    payload.downcast_ref::<&str>().map_or_else(
        || {
            payload
                .downcast_ref::<String>()
                .cloned()
                .unwrap_or_else(|| "non-string panic payload".to_owned())
        },
        |message| (*message).to_owned(),
    )
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;

    struct Recorder(Vec<Vec<u8>>);

    impl Observer for Recorder {
        fn on_token(&mut self, token: ObservedToken<'_>) -> Result<ControlFlow, ObserverError> {
            self.0.push(token.piece.to_vec());
            Ok(ControlFlow::Continue)
        }
    }

    struct PollStopper;

    impl Observer for PollStopper {
        fn poll(&mut self) -> Result<ControlFlow, ObserverError> {
            Ok(ControlFlow::Stop)
        }

        fn on_token(&mut self, _token: ObservedToken<'_>) -> Result<ControlFlow, ObserverError> {
            Ok(ControlFlow::Continue)
        }
    }

    struct PollPanics;

    impl Observer for PollPanics {
        fn poll(&mut self) -> Result<ControlFlow, ObserverError> {
            panic!("observer panic is contained")
        }

        fn on_token(&mut self, _token: ObservedToken<'_>) -> Result<ControlFlow, ObserverError> {
            Ok(ControlFlow::Continue)
        }
    }

    struct TokenPanics;

    impl Observer for TokenPanics {
        fn on_token(&mut self, _token: ObservedToken<'_>) -> Result<ControlFlow, ObserverError> {
            panic!("token observer panic is contained")
        }
    }

    struct ChunkRecorder;

    impl PrefillObserver for ChunkRecorder {
        fn on_chunk(
            &mut self,
            _progress: PrefillProgress,
        ) -> Result<ControlFlow, PrefillObserverError> {
            Ok(ControlFlow::Continue)
        }
    }

    struct PrefillStopper;

    impl PrefillObserver for PrefillStopper {
        fn poll(
            &mut self,
            _progress: PrefillProgress,
        ) -> Result<ControlFlow, PrefillObserverError> {
            Ok(ControlFlow::Stop)
        }

        fn on_chunk(
            &mut self,
            _progress: PrefillProgress,
        ) -> Result<ControlFlow, PrefillObserverError> {
            Ok(ControlFlow::Continue)
        }
    }

    struct PrefillPanics;

    impl PrefillObserver for PrefillPanics {
        fn on_chunk(
            &mut self,
            _progress: PrefillProgress,
        ) -> Result<ControlFlow, PrefillObserverError> {
            panic!("prefill observer panic is contained")
        }
    }

    #[test]
    fn observer_accepts_non_utf8_piece_bytes() {
        let id = Digest::of_bytes("test-observer", b"bytes");
        let mut set = ObserverSet::single(id, Recorder(Vec::new()));
        set.begin(7, 2).unwrap();
        set.poll().unwrap();
        let token = TokenId::new(4).unwrap();
        assert_eq!(
            set.observe(ObservedToken {
                token,
                piece: &[0xff, 0x00],
                position: 8,
            })
            .unwrap(),
            ControlFlow::Continue
        );
        assert_eq!(set.receipts()[0].observed_bytes, 2);
    }

    #[test]
    fn cancellation_propagates_between_threads() {
        let token = CancellationToken::new();
        let remote = token.clone();
        assert!(thread::spawn(move || remote.cancel()).join().unwrap());
        assert!(token.is_cancelled());
        assert!(!token.cancel());
    }

    #[test]
    fn polling_fans_out_after_a_stop_request() {
        let first = Digest::of_bytes("test-observer", b"stop");
        let second = Digest::of_bytes("test-observer", b"continue");
        let mut set = ObserverSet::new([
            (first, Box::new(PollStopper) as Box<dyn Observer>),
            (second, Box::new(Recorder(Vec::new())) as Box<dyn Observer>),
        ])
        .unwrap();
        set.begin(0, 1).unwrap();
        assert_eq!(set.poll().unwrap(), ControlFlow::Stop);
        let receipts = set.receipts();
        assert_eq!(receipts[0].polls, 1);
        assert_eq!(receipts[1].polls, 1);
        assert!(receipts[0].stop_requested);
        assert!(!receipts[1].stop_requested);
    }

    #[test]
    fn observer_panic_is_retained_and_fails_the_set() {
        let id = Digest::of_bytes("test-observer", b"panic");
        let mut set = ObserverSet::single(id, PollPanics);
        set.begin(0, 1).unwrap();
        assert!(set.poll().is_err());
        let failure = set.receipts()[0].failure.clone().unwrap();
        assert!(failure.panicked);
        assert_eq!(failure.phase, CallbackPhase::Poll);
        assert!(set.poll().is_err());
    }

    #[test]
    fn admitted_token_panic_is_retained_after_accounting() {
        let id = Digest::of_bytes("test-observer", b"token-panic");
        let mut set = ObserverSet::single(id, TokenPanics);
        set.begin(3, 1).unwrap();
        set.poll().unwrap();
        assert!(
            set.observe(ObservedToken {
                token: TokenId::new(1).unwrap(),
                piece: b"x",
                position: 4,
            })
            .is_err()
        );
        let receipt = set.receipt(0).unwrap();
        assert_eq!(receipt.observed_tokens, 1);
        assert_eq!(receipt.final_position, 4);
        let failure = receipt.failure.as_ref().unwrap();
        assert!(failure.panicked);
        assert_eq!(failure.phase, CallbackPhase::Observe);
    }

    #[test]
    fn prefill_cancellation_finishes_at_zero_admissions() {
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let id = Digest::of_bytes("test-prefill-observer", b"cancel");
        let mut monitor = PrefillMonitor::new(id, cancellation);
        let progress = PrefillProgress {
            initial_position: 0,
            requested_tokens: 4,
            admitted_tokens: 0,
            admitted_chunks: 0,
            position: 0,
        };
        monitor.begin(progress).unwrap();
        assert_eq!(monitor.poll(progress).unwrap(), ControlFlow::Stop);
        let receipt = monitor.finish(PrefillFinish::Stopped).unwrap();
        assert_eq!(receipt.progress.admitted_tokens, 0);
        assert!(receipt.digest().is_ok());
    }

    #[test]
    fn observer_set_bounds_iterator_consumption() {
        let observers = (0_u32..).map(|index| {
            (
                Digest::of_bytes("test-observer", &index.to_le_bytes()),
                Box::new(Recorder(Vec::new())) as Box<dyn Observer>,
            )
        });
        assert!(ObserverSet::new(observers).is_err());
    }

    #[test]
    fn observer_requires_polling_and_enforces_token_budget() {
        let id = Digest::of_bytes("test-observer", b"lifecycle");
        let token = TokenId::new(4).unwrap();
        let mut set = ObserverSet::single(id, Recorder(Vec::new()));
        set.begin(0, 1).unwrap();
        assert!(
            set.observe(ObservedToken {
                token,
                piece: b"x",
                position: 1,
            })
            .is_err()
        );

        set.begin(0, 1).unwrap();
        set.poll().unwrap();
        set.observe(ObservedToken {
            token,
            piece: b"x",
            position: 1,
        })
        .unwrap();
        assert!(set.poll().is_err());
    }

    #[test]
    fn observer_begin_rejects_position_overflow() {
        let id = Digest::of_bytes("test-observer", b"overflow");
        let mut set = ObserverSet::single(id, Recorder(Vec::new()));
        assert!(set.begin(u64::MAX, 1).is_err());
    }

    #[test]
    fn prefill_monitor_rejects_unpolled_or_nonmonotonic_chunks() {
        let id = Digest::of_bytes("test-prefill-observer", b"sequence");
        let mut monitor = PrefillMonitor::new(id, ChunkRecorder);
        let initial = PrefillProgress {
            initial_position: 3,
            requested_tokens: 4,
            admitted_tokens: 0,
            admitted_chunks: 0,
            position: 3,
        };
        let first = PrefillProgress {
            admitted_tokens: 2,
            admitted_chunks: 1,
            position: 5,
            ..initial
        };
        monitor.begin(initial).unwrap();
        assert!(monitor.observe_chunk(first).is_err());

        monitor.poll(initial).unwrap();
        monitor.observe_chunk(first).unwrap();
        assert!(monitor.poll(initial).is_err());

        let skipped = PrefillProgress {
            admitted_tokens: 4,
            admitted_chunks: 3,
            position: 7,
            ..initial
        };
        monitor.poll(first).unwrap();
        assert!(monitor.observe_chunk(skipped).is_err());
    }

    #[test]
    fn prefill_stop_is_explicit_and_terminal() {
        let initial = PrefillProgress {
            initial_position: 0,
            requested_tokens: 1,
            admitted_tokens: 0,
            admitted_chunks: 0,
            position: 0,
        };
        let id = Digest::of_bytes("test-prefill-observer", b"finish");
        let mut monitor = PrefillMonitor::new(id, ChunkRecorder);
        monitor.begin(initial).unwrap();
        assert_eq!(monitor.receipt().unwrap().finish, PrefillFinish::InProgress);
        assert!(monitor.finish(PrefillFinish::InProgress).is_err());
        assert!(monitor.finish(PrefillFinish::Stopped).is_err());

        let id = Digest::of_bytes("test-prefill-observer", b"stop");
        let mut monitor = PrefillMonitor::new(id, PrefillStopper);
        monitor.begin(initial).unwrap();
        assert_eq!(monitor.poll(initial).unwrap(), ControlFlow::Stop);
        assert!(monitor.poll(initial).is_err());
        monitor.finish(PrefillFinish::Stopped).unwrap();
        assert!(monitor.finish(PrefillFinish::Stopped).is_err());
    }

    #[test]
    fn prefill_panic_is_retained_after_chunk_accounting() {
        let initial = PrefillProgress {
            initial_position: 0,
            requested_tokens: 2,
            admitted_tokens: 0,
            admitted_chunks: 0,
            position: 0,
        };
        let admitted = PrefillProgress {
            admitted_tokens: 1,
            admitted_chunks: 1,
            position: 1,
            ..initial
        };
        let id = Digest::of_bytes("test-prefill-observer", b"panic");
        let mut monitor = PrefillMonitor::new(id, PrefillPanics);
        monitor.begin(initial).unwrap();
        monitor.poll(initial).unwrap();
        assert!(monitor.observe_chunk(admitted).is_err());
        let receipt = monitor.receipt().unwrap();
        assert_eq!(receipt.progress, admitted);
        let failure = receipt.failure.as_ref().unwrap();
        assert!(failure.panicked);
        assert_eq!(failure.phase, CallbackPhase::Prefill);
    }
}

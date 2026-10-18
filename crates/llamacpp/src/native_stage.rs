// SPDX-License-Identifier: MIT OR Apache-2.0

//! Read-only native sampler stage that measures raw next-token distributions.
//!
//! A backend that samples through llama.cpp's own chain (`llama_sampler_sample`
//! at the negative row index its batches require) never hands the row to
//! Rust. [`DistributionProbe::stage`] is a native `llama_sampler` placed first
//! in that chain: `apply` copies the untouched candidate logits, `accept`
//! measures them against the token the chain chose, and nothing is written
//! back, so selection is bit-identical with or without it.
//!
//! The same stage can also record, per emitted token, the chosen token's
//! log-probability and the top-N alternatives of the raw row (see
//! [`DistributionProbe::with_logprobs`]); one leading stage and one row copy
//! serve both.
//!
//! This module holds the adapter crate's only unsafe code; its contract is in
//! `SAFETY.md`.
#![allow(unsafe_code)]

use std::ffi::{CStr, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex};

use llama_cpp_4::sampling::LlamaSampler;
use llama_cpp_sys_4 as sys;
use logit_loom::{DistributionSummary, TokenDistribution, TokenId};

use crate::logprobs::{TokenLogprobs, record_row};

/// State one probe's stages share.
#[derive(Debug, Default)]
struct ProbeState {
    /// Raw logits of the row being sampled, indexed by token id.
    row: Vec<f32>,
    /// Whether `row` holds a complete, identity-ordered row awaiting accept.
    pending: bool,
    summary: DistributionSummary,
    /// Set if a callback ever panicked; the record is then incomplete.
    poisoned: bool,
    /// `Some(n)` records per-token log-probabilities with `n` alternatives.
    logprob_top_n: Option<usize>,
    /// Records not yet drained, one per `accept`, in call order.
    records: Vec<TokenLogprobs>,
    /// Reusable index storage for the top-N selection.
    scratch: Vec<u32>,
}

/// Measures every token a native chain samples, from outside the chain.
///
/// Clones share one record. Build a stage with [`Self::stage`] for each chain
/// the probe should lead, and read the record with [`Self::summary`].
#[derive(Clone, Debug, Default)]
pub struct DistributionProbe {
    state: Arc<Mutex<ProbeState>>,
}

impl DistributionProbe {
    /// A probe with an empty record.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A probe that also records log-probabilities of the raw row.
    ///
    /// For every token the chain accepts, one [`TokenLogprobs`] is appended:
    /// the emitted token's log-probability and the `top_n` most probable
    /// tokens of the same untouched row (`0` records the emitted token only).
    /// `top_n` is limited only by the vocabulary: larger values return the
    /// whole row. Records grow by `top_n` entries per token until
    /// [`Self::drain_logprobs`] takes them, so a long generation should drain
    /// periodically.
    ///
    /// # Guarantees
    ///
    /// `accept` is the unit of record: one record per `accept` call, in call
    /// order, so records line up with the tokens the chain admitted. A record
    /// is measured only when the most recent `apply` since the previous
    /// `accept` was the full identity-ordered row; otherwise it is an
    /// unmeasured placeholder ([`TokenLogprobs::measured`]). A sampler loop
    /// that calls `apply` for several positions before the matching `accept`s
    /// therefore measures at most the last one and marks the rest unmeasured,
    /// never attributing a row to the wrong token. Only the sequential
    /// `apply` then `accept` pairing, as in `llama_sampler_sample`, is covered
    /// by tests; no speculative or multi-token decoder has been exercised.
    #[must_use]
    pub fn with_logprobs(top_n: usize) -> Self {
        let probe = Self::default();
        if let Ok(mut state) = probe.state.lock() {
            state.logprob_top_n = Some(top_n);
        }
        probe
    }

    /// The configured alternative count, or `None` if records are off.
    #[must_use]
    pub fn top_logprobs(&self) -> Option<usize> {
        self.state.lock().ok()?.logprob_top_n
    }

    /// A copy of the undrained records, or `None` if a callback panicked and
    /// the record can no longer be trusted.
    #[must_use]
    pub fn logprobs(&self) -> Option<Vec<TokenLogprobs>> {
        let state = self.state.lock().ok()?;
        (!state.poisoned).then(|| state.records.clone())
    }

    /// Takes the undrained records in token order, leaving the probe empty, or
    /// `None` (taking nothing) if a callback panicked.
    ///
    /// Drained batches concatenate to the full emitted sequence.
    #[must_use]
    pub fn drain_logprobs(&self) -> Option<Vec<TokenLogprobs>> {
        let mut state = self.state.lock().ok()?;
        (!state.poisoned).then(|| std::mem::take(&mut state.records))
    }

    /// Discards undrained records. The distribution summary and the
    /// poisoned flag are unchanged.
    pub fn clear_logprobs(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.records.clear();
        }
    }

    /// Whether a callback ever panicked, leaving the summary and log-probability
    /// records incomplete.
    #[must_use]
    pub fn is_poisoned(&self) -> bool {
        !self.state.lock().is_ok_and(|state| !state.poisoned)
    }

    /// A native stage that records into this probe.
    ///
    /// Put it **first** in a chain: only the first stage sees the row before
    /// grammar, penalties, or truncation change it. Rows it cannot measure (a
    /// backend-sampled subset, or candidates already sorted) are counted as
    /// unmeasured, never guessed.
    #[must_use]
    pub fn stage(&self) -> LlamaSampler {
        new_stage(Arc::clone(&self.state))
    }

    /// The record so far, or `None` if a callback panicked and the record can
    /// no longer be trusted.
    #[must_use]
    pub fn summary(&self) -> Option<DistributionSummary> {
        let state = self.state.lock().ok()?;
        (!state.poisoned).then(|| state.summary.clone())
    }
}

fn new_stage(state: Arc<Mutex<ProbeState>>) -> LlamaSampler {
    let context = Box::into_raw(Box::new(state)).cast::<c_void>();
    // SAFETY: `PROBE_INTERFACE` is a `'static` table llama.cpp only reads,
    // and `context` is a leaked `Box<Arc<Mutex<ProbeState>>>` that `free`
    // reclaims exactly once when llama.cpp releases the sampler.
    unsafe { LlamaSampler::from_raw(std::ptr::addr_of!(PROBE_INTERFACE).cast_mut(), context) }
}

static PROBE_INTERFACE: sys::llama_sampler_i = sys::llama_sampler_i {
    name: Some(probe_name),
    accept: Some(probe_accept),
    apply: Some(probe_apply),
    reset: None,
    clone: Some(probe_clone),
    free: Some(probe_free),
    backend_init: None,
    backend_accept: None,
    backend_apply: None,
    backend_set_input: None,
    backend_reset: None,
    copy_state: None,
};

/// The shared state behind a probe sampler.
///
/// # Safety
///
/// `sampler` must be a live sampler created by [`new_stage`] or
/// [`probe_clone`], so its context is a live `Box<Arc<Mutex<ProbeState>>>`.
unsafe fn state_of<'a>(sampler: *const sys::llama_sampler) -> &'a Arc<Mutex<ProbeState>> {
    // SAFETY: guaranteed by the caller.
    unsafe { &*(*sampler).ctx.cast::<Arc<Mutex<ProbeState>>>() }
}

/// Runs `body` on the locked state, containing any panic: unwinding across
/// the C boundary would abort the process, so a panic marks the record
/// poisoned instead.
fn with_state(state: &Mutex<ProbeState>, body: impl FnOnce(&mut ProbeState)) {
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        if let Ok(mut guard) = state.lock() {
            body(&mut guard);
        }
    }));
    if outcome.is_err()
        && let Ok(mut guard) = state.lock()
    {
        guard.poisoned = true;
    }
}

unsafe extern "C" fn probe_name(_sampler: *const sys::llama_sampler) -> *const c_char {
    const NAME: &CStr = c"logit-loom-distribution-probe";
    NAME.as_ptr()
}

unsafe extern "C" fn probe_apply(
    sampler: *mut sys::llama_sampler,
    candidates: *mut sys::llama_token_data_array,
) {
    // SAFETY: llama.cpp passes a live probe sampler and a candidate array
    // valid for this call; the array is read, never written.
    let (state, candidates) = unsafe { (state_of(sampler), &*candidates) };
    let data: &[sys::llama_token_data] = if candidates.data.is_null() || candidates.size == 0 {
        &[]
    } else {
        // SAFETY: `data` points at `size` initialized entries for this call.
        unsafe { std::slice::from_raw_parts(candidates.data, candidates.size) }
    };
    let sorted = candidates.sorted;
    with_state(state, |state| {
        // Only the chain's untouched first view is a full row in id order:
        // entry `i` holds token `i`. Anything else is not measured.
        let identity = !sorted
            && data.first().is_some_and(|first| first.id == 0)
            && data
                .last()
                .is_some_and(|last| usize::try_from(last.id).ok() == Some(data.len() - 1));
        state.pending = identity;
        if identity {
            state.row.clear();
            state.row.extend(data.iter().map(|entry| entry.logit));
        }
    });
}

unsafe extern "C" fn probe_accept(sampler: *mut sys::llama_sampler, token: sys::llama_token) {
    // SAFETY: llama.cpp passes a live probe sampler.
    let state = unsafe { state_of(sampler) };
    with_state(state, |state| {
        let pending = std::mem::take(&mut state.pending);
        if let Some(top_n) = state.logprob_top_n {
            let state = &mut *state;
            let row: &[f32] = if pending { &state.row } else { &[] };
            let record = record_row(row, token, top_n, &mut state.scratch);
            state.records.push(record);
        }
        let measured = pending
            .then(|| TokenId::new(token).ok())
            .flatten()
            .and_then(|token| TokenDistribution::measure(&state.row, token).ok());
        match measured {
            Some(distribution) => state.summary.push(distribution),
            None => state.summary.push_unmeasured(),
        }
    });
}

unsafe extern "C" fn probe_clone(sampler: *const sys::llama_sampler) -> *mut sys::llama_sampler {
    // SAFETY: llama.cpp passes a live probe sampler.
    let state = Arc::clone(unsafe { state_of(sampler) });
    let context = Box::into_raw(Box::new(state)).cast::<c_void>();
    // SAFETY: as in `new_stage`; ownership of the returned sampler passes to
    // llama.cpp, which frees it through `probe_free`.
    unsafe { sys::llama_sampler_init(std::ptr::addr_of!(PROBE_INTERFACE).cast_mut(), context) }
}

unsafe extern "C" fn probe_free(sampler: *mut sys::llama_sampler) {
    // SAFETY: llama.cpp calls `free` exactly once per sampler, and the
    // context is the box `new_stage` or `probe_clone` leaked.
    unsafe {
        let context = (*sampler).ctx.cast::<Arc<Mutex<ProbeState>>>();
        if !context.is_null() {
            drop(Box::from_raw(context));
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
mod tests {
    use super::*;

    fn data(logits: &[f32]) -> Vec<sys::llama_token_data> {
        logits
            .iter()
            .zip(0..)
            .map(|(&logit, id)| sys::llama_token_data { id, logit, p: 0.0 })
            .collect()
    }

    /// A raw probe sampler, owned by the test and released with
    /// `llama_sampler_free`, exactly as llama.cpp would.
    fn raw_stage(probe: &DistributionProbe) -> *mut sys::llama_sampler {
        let context = Box::into_raw(Box::new(Arc::clone(&probe.state))).cast::<c_void>();
        // SAFETY: as in `new_stage`.
        unsafe { sys::llama_sampler_init(std::ptr::addr_of!(PROBE_INTERFACE).cast_mut(), context) }
    }

    /// Drives the callbacks exactly as `llama_sampler_sample` does.
    fn sample(
        stage: *mut sys::llama_sampler,
        row: &mut [sys::llama_token_data],
        sorted: bool,
        token: i32,
    ) {
        let mut array = sys::llama_token_data_array {
            data: row.as_mut_ptr(),
            size: row.len(),
            selected: -1,
            sorted,
        };
        // SAFETY: `stage` is a live probe sampler and `array` is valid here.
        unsafe {
            sys::llama_sampler_apply(stage, &raw mut array);
            sys::llama_sampler_accept(stage, token);
        }
    }

    fn free(stage: *mut sys::llama_sampler) {
        // SAFETY: each test sampler is freed exactly once.
        unsafe { sys::llama_sampler_free(stage) };
    }

    #[test]
    fn the_stage_measures_identity_rows_and_counts_the_rest() {
        let probe = DistributionProbe::new();
        let stage = raw_stage(&probe);
        let mut row = data(&[2.0, 0.0, -1.0, 0.5]);
        let before: Vec<f32> = row.iter().map(|entry| entry.logit).collect();
        sample(stage, &mut row, false, 0);
        assert_eq!(
            row.iter().map(|entry| entry.logit).collect::<Vec<_>>(),
            before,
            "the probe never writes the candidates"
        );
        sample(stage, &mut row, false, 2);
        // Sorted or reordered views are not raw rows.
        sample(stage, &mut row, true, 0);
        row.reverse();
        sample(stage, &mut row, false, 0);
        free(stage);
        let report = probe
            .summary()
            .expect("healthy")
            .report(8, 0.1)
            .expect("measured");
        assert_eq!(report.measured, 2);
        assert_eq!(report.unmeasured, 2);
        assert_eq!(report.steered, 1);
    }

    #[test]
    fn clones_and_safe_stages_share_one_record_and_free_cleanly() {
        let probe = DistributionProbe::new();
        let first = raw_stage(&probe);
        // SAFETY: `first` is a live probe sampler.
        let second = unsafe { sys::llama_sampler_clone(first) };
        let mut row = data(&[1.0, 1.0]);
        sample(first, &mut row, false, 0);
        sample(second, &mut row, false, 1);
        free(first);
        free(second);
        // The safe constructor's sampler releases its context on drop.
        drop(probe.stage());
        assert_eq!(
            Arc::strong_count(&probe.state),
            1,
            "every stage released its share"
        );
        assert_eq!(
            probe
                .summary()
                .expect("healthy")
                .report(4, 0.1)
                .expect("measured")
                .measured,
            2
        );
    }

    // ---- log-probability records ----

    use llama_cpp_4::token::LlamaToken;
    use llama_cpp_4::token::data::LlamaTokenData;
    use llama_cpp_4::token::data_array::LlamaTokenDataArray;

    /// Deterministic pseudo-random logits so no test needs a model or an RNG
    /// crate.
    fn synthetic_row(vocabulary: usize, step: u64, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ step.wrapping_mul(0xD1B5_4A32);
        (0..vocabulary)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 40) as f32 / 16_777_216.0 - 0.5) * 12.0
            })
            .collect()
    }

    fn array(logits: &[f32]) -> LlamaTokenDataArray {
        LlamaTokenDataArray::new(
            logits
                .iter()
                .zip(0..)
                .map(|(&logit, id)| LlamaTokenData::new(LlamaToken(id), logit, 0.0))
                .collect(),
            false,
        )
    }

    #[derive(Clone, Copy)]
    enum Kind {
        Greedy,
        Temperature,
        TopK,
        TopP,
    }

    fn tail(kind: Kind, seed: u32) -> Vec<LlamaSampler> {
        match kind {
            Kind::Greedy => vec![LlamaSampler::greedy()],
            Kind::Temperature => vec![LlamaSampler::temp(0.7), LlamaSampler::dist(seed)],
            Kind::TopK => vec![
                LlamaSampler::top_k(5),
                LlamaSampler::temp(1.3),
                LlamaSampler::dist(seed),
            ],
            Kind::TopP => vec![
                LlamaSampler::top_p(0.8, 1),
                LlamaSampler::temp(0.9),
                LlamaSampler::dist(seed),
            ],
        }
    }

    /// Runs `steps` sample-then-accept rounds through a real native chain.
    fn run_chain(
        probe: Option<&DistributionProbe>,
        kind: Kind,
        seed: u32,
        vocabulary: usize,
        steps: u64,
    ) -> Vec<i32> {
        let mut stages = Vec::new();
        stages.extend(probe.map(DistributionProbe::stage));
        stages.extend(tail(kind, seed));
        let mut chain = LlamaSampler::chain_simple(stages);
        (0..steps)
            .map(|step| {
                let mut candidates = array(&synthetic_row(vocabulary, step, u64::from(seed)));
                chain.apply(&mut candidates);
                let token = candidates.selected_token().expect("selected");
                chain.accept(token);
                token.0
            })
            .collect()
    }

    #[test]
    fn the_probe_never_changes_selection_and_records_every_token_in_order() {
        for kind in [Kind::Greedy, Kind::Temperature, Kind::TopK, Kind::TopP] {
            for seed in 0..40_u32 {
                let bare = run_chain(None, kind, seed, 50, 24);
                let probe = DistributionProbe::with_logprobs(7);
                let probed = run_chain(Some(&probe), kind, seed, 50, 24);
                assert_eq!(bare, probed, "selection changed (seed {seed})");

                let records = probe.drain_logprobs().expect("healthy");
                assert_eq!(
                    records.iter().map(|r| r.token).collect::<Vec<_>>(),
                    probed,
                    "one record per emitted token, in order"
                );
                for (step, record) in records.iter().enumerate() {
                    let row = synthetic_row(50, step as u64, u64::from(seed));
                    let sum: f64 = row.iter().map(|&x| f64::from(x).exp()).sum();
                    let expected = f64::from(row[record.token as usize]).exp() / sum;
                    let got = record.logprob.expect("measured");
                    assert!(
                        (got - expected.ln()).abs() < 1e-9,
                        "raw logprob, not the sampled distribution"
                    );
                    assert_eq!(record.top.len(), 7);
                    assert!(record.top.windows(2).all(|w| w[0].logprob >= w[1].logprob));
                }
            }
        }
    }

    #[test]
    fn records_drain_in_batches_and_clear_on_request() {
        let probe = DistributionProbe::with_logprobs(2);
        assert_eq!(probe.top_logprobs(), Some(2));
        assert_eq!(DistributionProbe::new().top_logprobs(), None);
        let stage = raw_stage(&probe);
        let mut row = data(&[0.0, 1.0, 2.0]);
        for token in [2, 1] {
            sample(stage, &mut row, false, token);
        }
        let first = probe.drain_logprobs().expect("healthy");
        assert_eq!(first.iter().map(|r| r.token).collect::<Vec<_>>(), [2, 1]);
        assert_eq!(probe.logprobs().expect("healthy"), Vec::new());
        sample(stage, &mut row, false, 0);
        assert_eq!(
            probe.logprobs().expect("healthy").len(),
            1,
            "snapshot keeps"
        );
        assert_eq!(probe.logprobs().expect("healthy").len(), 1);
        probe.clear_logprobs();
        assert_eq!(probe.drain_logprobs().expect("healthy"), Vec::new());
        free(stage);
        // A probe without logprobs records nothing.
        let off = DistributionProbe::new();
        let stage = raw_stage(&off);
        sample(stage, &mut row, false, 0);
        free(stage);
        assert_eq!(off.logprobs().expect("healthy"), Vec::new());
    }

    #[test]
    fn several_applies_before_the_accepts_never_misattribute_a_row() {
        let probe = DistributionProbe::with_logprobs(1);
        let stage = raw_stage(&probe);
        let mut first = data(&[5.0, 0.0]);
        let mut second = data(&[0.0, 5.0]);
        for row in [&mut first, &mut second] {
            let mut array = sys::llama_token_data_array {
                data: row.as_mut_ptr(),
                size: row.len(),
                selected: -1,
                sorted: false,
            };
            // SAFETY: `stage` is live and `array` is valid for the call.
            unsafe { sys::llama_sampler_apply(stage, &raw mut array) };
        }
        // SAFETY: `stage` is a live probe sampler.
        unsafe {
            sys::llama_sampler_accept(stage, 0);
            sys::llama_sampler_accept(stage, 1);
        }
        free(stage);
        let records = probe.drain_logprobs().expect("healthy");
        assert_eq!(records.len(), 2, "one record per accept");
        assert!(records[0].measured(), "paired with the latest row");
        assert!(!records[1].measured(), "no row left for the second accept");
    }

    #[test]
    fn sorted_views_and_bare_accepts_leave_unmeasured_placeholders() {
        let probe = DistributionProbe::with_logprobs(3);
        let stage = raw_stage(&probe);
        let mut row = data(&[1.0, 2.0]);
        sample(stage, &mut row, true, 1);
        free(stage);
        let records = probe.drain_logprobs().expect("healthy");
        assert_eq!(records.len(), 1);
        assert!(!records[0].measured() && records[0].top.is_empty());
    }

    #[test]
    fn a_panicking_callback_marks_every_record_incomplete() {
        let probe = DistributionProbe::with_logprobs(1);
        let stage = raw_stage(&probe);
        let mut row = data(&[1.0, 2.0]);
        sample(stage, &mut row, false, 1);
        assert!(!probe.is_poisoned());
        with_state(&probe.state, |_| panic!("injected callback failure"));
        assert!(probe.is_poisoned());
        assert!(probe.summary().is_none());
        assert!(probe.logprobs().is_none());
        assert!(probe.drain_logprobs().is_none(), "nothing is handed out");
        free(stage);
    }
}

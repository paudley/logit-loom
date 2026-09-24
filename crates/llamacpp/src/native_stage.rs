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
//! This module holds the adapter crate's only unsafe code; its contract is in
//! `SAFETY.md`.
#![allow(unsafe_code)]

use std::ffi::{CStr, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex};

use llama_cpp_4::sampling::LlamaSampler;
use llama_cpp_sys_4 as sys;
use logit_loom::{DistributionSummary, TokenDistribution, TokenId};

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
        let measured = std::mem::take(&mut state.pending)
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
}

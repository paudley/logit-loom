<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Safety contract

Unsafe Rust in this adapter is confined to the private `src/native_stage.rs`
module, which implements one read-only native `llama_sampler`
(`DistributionProbe`). Every other module stays free of unsafe code; the crate
root denies it. The module relies on these conditions:

1. `PROBE_INTERFACE` is a `'static` `llama_sampler_i` table. llama.cpp stores a
   pointer to it and only reads it, so casting its address to `*mut` for
   `llama_sampler_init` never permits a write. Its field list is the bindgen
   struct of the pinned `llama-cpp-sys-4`; a binding that adds or changes a
   field fails to compile rather than misreading the table.
2. Each probe sampler's context is a `Box<Arc<Mutex<ProbeState>>>` leaked by
   `new_stage` or `probe_clone`. It stays live until llama.cpp calls `free`,
   exactly once per sampler, which reclaims it. `reset` and `copy_state` are
   absent (llama.cpp accepts null for both), and no backend hook is provided,
   so a chain containing the probe samples on the CPU exactly as it would
   without it.
3. `apply` receives a candidate array valid for the synchronous call. The probe
   forms a shared slice of exactly `size` entries, only when `data` is
   non-null and `size` is non-zero, and never writes through it. Selection is
   therefore unchanged by the probe.
4. No callback unwinds into C. Every callback body runs under `catch_unwind`;
   a panic marks the record poisoned and `DistributionProbe::summary` then
   returns `None` instead of a partial record.
5. The probe is used on the thread that samples its chain; the shared state is
   behind a `Mutex`, so a clone used from another thread would still be sound,
   only serialized.

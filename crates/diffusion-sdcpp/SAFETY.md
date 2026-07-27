<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Safety contract

Unsafe Rust is confined to the private dynamic-ABI module in `src/ffi.rs` and
the narrowly scoped descriptor/slice/callback operations in `src/runtime.rs`.
The safe public adapter relies on all of the following conditions:

1. The loaded library exports companion ABI version 1 and reports exact
   upstream commit `ea4e566ccffa10f853ecc3f29e74b1820bc91beb`.
2. Resolved symbols retain the C signatures declared in
   `native/stable-diffusion.cpp/logit-loom-step-v1.patch` and
   `native/stable-diffusion.cpp/logit-loom-image-v2.patch` for the lifetime of
   the loaded library. Every image parameter or tensor descriptor carries
   extension version 2, which the companion checks before reading the rest of
   that descriptor.
3. A non-null native context returned by `sd_loom_new_ctx_v1` belongs to the
   adapter and is released exactly once with `free_sd_ctx` before unloading
   the library.
4. Native callback descriptors and strings remain alive only for the
   synchronous callback. Rank, byte count, element count, shape, dtype,
   pointer/null relationships, finite nonnegative elapsed time, and configured
   bounds are validated before a Rust slice is formed.
5. The step-state pointer is valid, aligned, writable, contiguous `f32` for
   `state_len` elements during that callback. Rust copies it before invoking
   user code and writes back only a complete, finite, equal-length result.
6. Condition tensor pointers are aligned, readable, and valid for the reported
   byte count during their callback. They are hashed synchronously and never
   retained.
7. A successful native image points to one allocation owned by the native
   library. Width, height, channel count, multiplication, nullness, and bounds
   are checked before copying. `free_sd_images` releases it exactly once on
   every success and error path after ownership transfers.
8. Image-v2 prompts, artifact paths, pixel views, reference arrays, `LoRA`
   arrays, schedules, and callback state outlive the synchronous native call.
   Rust validates their public lengths, geometry, pointer/count relationships,
   finite controls, and C-string constraints before passing their borrowed
   pointers.
9. A successful VAE encode returns one native-owned tensor allocation. Rust
   validates its ABI, pointer, rank, shape, element count, finite values, and
   public tensor bound before copying. `sd_loom_free_tensor_v2` releases it
   exactly once. VAE decode borrows a validated contiguous finite `f32` slice
   only for the synchronous call and returns a native image covered by rule 7.
10. Rust callback state outlives the synchronous native generation call.
   Errors and panics are caught and stored before returning a nonzero callback
   result. No unwind crosses C.
11. Session clear and close occur only while the single owner is resident and
    no operation is running. A native exception, callback uncertainty, or
    unconfirmed request-local cleanup poisons the owner before it can be
    reused.
12. The context contains `PhantomData<Rc<()>>`, making it neither `Send` nor
   `Sync`. Reentrant generation on one context is not exposed.

The exact ABI/commit checks turn an accidental ordinary stable-diffusion.cpp
library or another companion revision into a load error before any model
context call. They are compatibility checks, not a sandbox for a malicious
shared library.

Model-free tests exercise malformed ranks and byte counts, callback error and
panic containment, no write-back after failed mutation, complete successful
write-back, invalid native timing, exact profile shapes, checkpoint mismatch,
advanced-image geometry, bounded VAE tensors, image copying, failure
dispositions, and compile-fail `Send`/`Sync` assertions. The public
`probe_companion` path checks both symbol sets, companion ABI, commit, library
bytes, and device-report handling against a caller-built companion without
loading a model.

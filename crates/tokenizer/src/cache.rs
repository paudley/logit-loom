// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded frequency-admitted exact span cache.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::mem::size_of;

use logit_loom_core::{Digest, TokenId};
use serde::{Deserialize, Serialize};

use crate::{TokenSpan, TokenizationError};

const SKETCH_DEPTH: usize = 4;
const MIN_SKETCH_WIDTH: usize = 64;

/// Strict entry, byte, and admission bounds for one worker-local cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheConfig {
    /// Maximum resident entries; zero disables the cache.
    pub maximum_entries: u32,
    /// Maximum bytes charged for retained source and projections.
    pub maximum_bytes: u64,
    /// Longest source span eligible for admission.
    pub maximum_span_bytes: u32,
    /// Minimum estimated frequency required for admission.
    pub admission_frequency: u8,
    /// Width of each four-row saturating frequency sketch.
    pub sketch_width: u32,
}

impl CacheConfig {
    /// Validates strict bounds.
    ///
    /// # Errors
    ///
    /// Returns an error for an enabled cache with zero/inconsistent bounds.
    pub fn validate(self) -> Result<(), TokenizationError> {
        if self.maximum_entries == 0 {
            if self.maximum_bytes != 0 || self.maximum_span_bytes != 0 {
                return Err(TokenizationError::Invalid(
                    "disabled cache must have zero byte and span bounds".to_owned(),
                ));
            }
            return Ok(());
        }
        let width = usize::try_from(self.sketch_width)
            .map_err(|_| TokenizationError::Invalid("cache sketch width overflowed".to_owned()))?;
        if self.maximum_bytes == 0
            || self.maximum_span_bytes == 0
            || self.admission_frequency == 0
            || width < MIN_SKETCH_WIDTH
            || !width.is_power_of_two()
        {
            return Err(TokenizationError::Invalid(
                "enabled cache bounds are inconsistent".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Compact exact projection retained by the cache.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CachedProjection {
    /// Exact token identifiers.
    pub tokens: Vec<TokenId>,
    /// Optional exact source offsets, parallel to `tokens`.
    pub offsets: Option<Vec<(u32, u32)>>,
}

impl CachedProjection {
    fn validate(&self) -> Result<(), TokenizationError> {
        if self
            .offsets
            .as_ref()
            .is_some_and(|offsets| offsets.len() != self.tokens.len())
        {
            return Err(TokenizationError::Invalid(
                "cached offsets do not match token count".to_owned(),
            ));
        }
        Ok(())
    }

    fn bytes(&self) -> Option<u64> {
        let tokens = self.tokens.len().checked_mul(size_of::<i32>())?;
        let offsets = self
            .offsets
            .as_ref()
            .map_or(0, |value| value.len().saturating_mul(8));
        u64::try_from(tokens.checked_add(offsets)?).ok()
    }

    /// Converts token spans into a compact projection.
    #[must_use]
    pub fn from_spans(spans: &[TokenSpan], retain_offsets: bool) -> Self {
        Self {
            tokens: spans.iter().map(|span| span.token).collect(),
            offsets: retain_offsets
                .then(|| spans.iter().map(|span| (span.start, span.end)).collect()),
        }
    }
}

/// Result of one cache admission attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheDisposition {
    /// Cache is disabled.
    Disabled,
    /// Exact entry was already resident.
    AlreadyResident,
    /// Candidate was admitted.
    Admitted,
    /// Candidate was seen but not frequent enough.
    FrequencyBypass,
    /// Candidate exceeded a span or byte bound.
    OversizeBypass,
    /// Candidate lost frequency comparison with the resident victim.
    VictimBypass,
}

/// Content-free cache accounting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheStats {
    /// Exact hits.
    pub hits: u64,
    /// Misses including collision-safe byte mismatches.
    pub misses: u64,
    /// Successful admissions.
    pub admissions: u64,
    /// Evicted entries.
    pub evictions: u64,
    /// Frequency, oversize, or victim bypasses.
    pub bypasses: u64,
    /// Source plus projection bytes avoided on hits.
    pub bytes_avoided: u64,
    /// Current charged bytes.
    pub resident_bytes: u64,
    /// Current resident entries.
    pub resident_entries: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct CacheKey {
    identity: [u8; 32],
    policy: [u8; 32],
    source_digest: [u8; 32],
    source_length: u32,
    has_offsets: bool,
}

struct CacheEntry {
    source: Vec<u8>,
    projection: CachedProjection,
    charge: u64,
    frequency: u8,
    last_access: u64,
}

/// Worker-local cache with W-TinyLFU-style frequency admission and recency
/// victim selection.
///
/// The caller supplies an epoch-local secret key. Keys bind exact execution
/// and policy identities, byte length, and a keyed source digest. Every hit
/// additionally compares the retained source bytes.
pub struct SpanCache {
    config: CacheConfig,
    secret: [u8; 32],
    entries: HashMap<CacheKey, CacheEntry>,
    sketch: Vec<u8>,
    clock: u64,
    resident_bytes: u64,
    stats: CacheStats,
}

impl SpanCache {
    /// Creates one empty cache.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid bounds.
    pub fn new(config: CacheConfig, secret: [u8; 32]) -> Result<Self, TokenizationError> {
        config.validate()?;
        let sketch_len = usize::try_from(config.sketch_width)
            .unwrap_or_default()
            .saturating_mul(SKETCH_DEPTH);
        Ok(Self {
            config,
            secret,
            entries: HashMap::with_capacity(
                usize::try_from(config.maximum_entries).unwrap_or_default(),
            ),
            sketch: vec![0; sketch_len],
            clock: 0,
            resident_bytes: 0,
            stats: CacheStats::default(),
        })
    }

    /// Returns content-free cumulative and current accounting.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        let mut stats = self.stats;
        stats.resident_bytes = self.resident_bytes;
        stats.resident_entries = u64::try_from(self.entries.len()).unwrap_or(u64::MAX);
        stats
    }

    /// Looks up and copies an exact projection into caller-owned scratch.
    ///
    /// # Errors
    ///
    /// Returns an error only if the supplied identities cannot be represented
    /// by the internal exact key.
    pub fn lookup_into(
        &mut self,
        identity: &Digest,
        policy: &Digest,
        source: &[u8],
        offsets: bool,
        tokens: &mut Vec<TokenId>,
        token_offsets: &mut Vec<(u32, u32)>,
    ) -> Result<bool, TokenizationError> {
        if self.config.maximum_entries == 0 {
            return Ok(false);
        }
        let key = self.key(identity, policy, source, offsets)?;
        self.observe(key.source_digest);
        self.clock = self.clock.wrapping_add(1);
        let Some(entry) = self.entries.get_mut(&key) else {
            self.stats.misses = self.stats.misses.saturating_add(1);
            return Ok(false);
        };
        if entry.source.as_slice() != source {
            self.stats.misses = self.stats.misses.saturating_add(1);
            return Ok(false);
        }
        entry.frequency = entry.frequency.saturating_add(1);
        entry.last_access = self.clock;
        tokens.clear();
        tokens.extend_from_slice(&entry.projection.tokens);
        token_offsets.clear();
        if let Some(offsets) = &entry.projection.offsets {
            token_offsets.extend_from_slice(offsets);
        }
        self.stats.hits = self.stats.hits.saturating_add(1);
        self.stats.bytes_avoided = self.stats.bytes_avoided.saturating_add(entry.charge);
        Ok(true)
    }

    /// Admits an exact projection under strict entry and byte ceilings.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid projection shape or identity encoding.
    pub fn admit(
        &mut self,
        identity: &Digest,
        policy: &Digest,
        source: &[u8],
        projection: CachedProjection,
    ) -> Result<CacheDisposition, TokenizationError> {
        projection.validate()?;
        if self.config.maximum_entries == 0 {
            return Ok(CacheDisposition::Disabled);
        }
        let offsets = projection.offsets.is_some();
        let key = self.key(identity, policy, source, offsets)?;
        self.observe(key.source_digest);
        if self.entries.contains_key(&key) {
            return Ok(CacheDisposition::AlreadyResident);
        }
        let span_limit = usize::try_from(self.config.maximum_span_bytes)
            .map_err(|_| TokenizationError::Invalid("cache span bound overflowed".to_owned()))?;
        let projection_bytes = projection.bytes().ok_or_else(|| {
            TokenizationError::Invalid("cache projection byte size overflowed".to_owned())
        })?;
        let charge = u64::try_from(source.len())
            .ok()
            .and_then(|bytes| bytes.checked_add(projection_bytes))
            .ok_or_else(|| TokenizationError::Invalid("cache charge overflowed".to_owned()))?;
        if source.len() > span_limit || charge > self.config.maximum_bytes {
            self.stats.bypasses = self.stats.bypasses.saturating_add(1);
            return Ok(CacheDisposition::OversizeBypass);
        }
        let frequency = self.estimate(key.source_digest);
        if frequency < self.config.admission_frequency {
            self.stats.bypasses = self.stats.bypasses.saturating_add(1);
            return Ok(CacheDisposition::FrequencyBypass);
        }
        let maximum_entries = usize::try_from(self.config.maximum_entries)
            .map_err(|_| TokenizationError::Invalid("cache entry bound overflowed".to_owned()))?;
        while self.entries.len() >= maximum_entries
            || self
                .resident_bytes
                .checked_add(charge)
                .is_none_or(|total| total > self.config.maximum_bytes)
        {
            let Some(victim) = self.victim() else {
                self.stats.bypasses = self.stats.bypasses.saturating_add(1);
                return Ok(CacheDisposition::OversizeBypass);
            };
            let victim_frequency = self.entries.get(&victim).map_or(0, |entry| entry.frequency);
            if victim_frequency > frequency {
                self.stats.bypasses = self.stats.bypasses.saturating_add(1);
                return Ok(CacheDisposition::VictimBypass);
            }
            self.remove(victim);
        }
        self.clock = self.clock.wrapping_add(1);
        self.entries.insert(
            key,
            CacheEntry {
                source: source.to_vec(),
                projection,
                charge,
                frequency,
                last_access: self.clock,
            },
        );
        self.resident_bytes = self.resident_bytes.saturating_add(charge);
        self.stats.admissions = self.stats.admissions.saturating_add(1);
        Ok(CacheDisposition::Admitted)
    }

    /// Clears content-bearing entries and frequency history.
    pub fn clear(&mut self) {
        for entry in self.entries.values_mut() {
            clear_entry(entry);
        }
        self.entries.clear();
        self.sketch.fill(0);
        self.resident_bytes = 0;
        self.clock = 0;
    }

    fn key(
        &self,
        identity: &Digest,
        policy: &Digest,
        source: &[u8],
        offsets: bool,
    ) -> Result<CacheKey, TokenizationError> {
        let source_length = u32::try_from(source.len()).map_err(|_| TokenizationError::Bound {
            field: "cache source bytes",
            limit: u32::MAX as usize,
        })?;
        Ok(CacheKey {
            identity: *blake3::hash(identity.as_str().as_bytes()).as_bytes(),
            policy: *blake3::hash(policy.as_str().as_bytes()).as_bytes(),
            source_digest: *blake3::keyed_hash(&self.secret, source).as_bytes(),
            source_length,
            has_offsets: offsets,
        })
    }

    fn observe(&mut self, digest: [u8; 32]) {
        if self.sketch.is_empty() {
            return;
        }
        let width = usize::try_from(self.config.sketch_width).unwrap_or_default();
        for depth in 0..SKETCH_DEPTH {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            depth.hash(&mut hasher);
            digest.hash(&mut hasher);
            let index = usize::try_from(hasher.finish()).unwrap_or_default() & (width - 1);
            let slot = depth * width + index;
            self.sketch[slot] = self.sketch[slot].saturating_add(1);
        }
    }

    fn estimate(&self, digest: [u8; 32]) -> u8 {
        if self.sketch.is_empty() {
            return 0;
        }
        let width = usize::try_from(self.config.sketch_width).unwrap_or_default();
        (0..SKETCH_DEPTH)
            .map(|depth| {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                depth.hash(&mut hasher);
                digest.hash(&mut hasher);
                let index = usize::try_from(hasher.finish()).unwrap_or_default() & (width - 1);
                self.sketch[depth * width + index]
            })
            .min()
            .unwrap_or(0)
    }

    fn victim(&self) -> Option<CacheKey> {
        self.entries
            .iter()
            .min_by_key(|(_, entry)| (entry.frequency, entry.last_access))
            .map(|(key, _)| *key)
    }

    fn remove(&mut self, key: CacheKey) {
        if let Some(mut entry) = self.entries.remove(&key) {
            self.resident_bytes = self.resident_bytes.saturating_sub(entry.charge);
            clear_entry(&mut entry);
            self.stats.evictions = self.stats.evictions.saturating_add(1);
        }
    }
}

impl Drop for SpanCache {
    fn drop(&mut self) {
        self.clear();
        self.secret.fill(0);
    }
}

fn clear_entry(entry: &mut CacheEntry) {
    entry.source.fill(0);
    entry
        .projection
        .tokens
        .fill(TokenId::new(0).expect("zero token is valid"));
    if let Some(offsets) = &mut entry.projection.offsets {
        offsets.fill((0, 0));
    }
}

#[cfg(test)]
mod tests {
    use logit_loom_core::{Digest, TokenId};

    use super::*;

    fn config(entries: u32, bytes: u64) -> CacheConfig {
        CacheConfig {
            maximum_entries: entries,
            maximum_bytes: bytes,
            maximum_span_bytes: 64,
            admission_frequency: 2,
            sketch_width: 64,
        }
    }

    fn projection(token: i32) -> CachedProjection {
        CachedProjection {
            tokens: vec![TokenId::new(token).unwrap()],
            offsets: Some(vec![(0, 1)]),
        }
    }

    #[test]
    fn admission_is_frequency_gated_and_hits_verify_bytes() {
        let identity = Digest::of_bytes("test", b"identity");
        let policy = Digest::of_bytes("test", b"policy");
        let mut cache = SpanCache::new(config(2, 128), [7; 32]).unwrap();
        assert_eq!(
            cache
                .admit(&identity, &policy, b"a", projection(1))
                .unwrap(),
            CacheDisposition::FrequencyBypass
        );
        assert_eq!(
            cache
                .admit(&identity, &policy, b"a", projection(1))
                .unwrap(),
            CacheDisposition::Admitted
        );
        let mut tokens = Vec::new();
        let mut offsets = Vec::new();
        assert!(
            cache
                .lookup_into(&identity, &policy, b"a", true, &mut tokens, &mut offsets)
                .unwrap()
        );
        assert_eq!(tokens, vec![TokenId::new(1).unwrap()]);
        assert!(
            !cache
                .lookup_into(&identity, &policy, b"b", true, &mut tokens, &mut offsets)
                .unwrap()
        );
    }

    #[test]
    fn byte_and_entry_ceilings_evict_without_growth() {
        let identity = Digest::of_bytes("test", b"identity");
        let policy = Digest::of_bytes("test", b"policy");
        let mut cache = SpanCache::new(config(1, 64), [3; 32]).unwrap();
        cache
            .admit(&identity, &policy, b"a", projection(1))
            .unwrap();
        cache
            .admit(&identity, &policy, b"a", projection(1))
            .unwrap();
        cache
            .admit(&identity, &policy, b"b", projection(2))
            .unwrap();
        cache
            .admit(&identity, &policy, b"b", projection(2))
            .unwrap();
        let stats = cache.stats();
        assert_eq!(stats.resident_entries, 1);
        assert!(stats.resident_bytes <= 64);
        assert_eq!(stats.evictions, 1);
    }

    #[test]
    fn disabled_cache_has_zero_state() {
        let cache = SpanCache::new(
            CacheConfig {
                maximum_entries: 0,
                maximum_bytes: 0,
                maximum_span_bytes: 0,
                admission_frequency: 1,
                sketch_width: 0,
            },
            [0; 32],
        )
        .unwrap();
        assert_eq!(cache.stats(), CacheStats::default());
    }
}

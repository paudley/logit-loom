// SPDX-License-Identifier: MIT OR Apache-2.0

//! Domain-separated content identities.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::CoreError;

const DIGEST_HEX_BYTES: usize = 64;

/// A domain-separated BLAKE3 digest encoded as 64 lowercase hexadecimal bytes.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Digest(String);

impl Digest {
    /// Hashes bytes under an explicit domain separator.
    pub fn of_bytes(domain: &str, bytes: &[u8]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"logit-loom\0");
        hasher.update(
            &u64::try_from(domain.len())
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        hasher.update(domain.as_bytes());
        hasher.update(&u64::try_from(bytes.len()).unwrap_or(u64::MAX).to_le_bytes());
        hasher.update(bytes);
        Self(hasher.finalize().to_hex().to_string())
    }

    /// Hashes a serializable contract using deterministic JSON bytes.
    ///
    /// Contract maps should use ordered collections. Changing a serialized
    /// shape is an identity and compatibility change.
    ///
    /// # Errors
    ///
    /// Returns an error when the value cannot be serialized.
    pub fn of_serializable<T: Serialize + ?Sized>(
        domain: &str,
        value: &T,
    ) -> Result<Self, CoreError> {
        Ok(Self::of_bytes(domain, &serde_json::to_vec(value)?))
    }

    /// Returns the lowercase hexadecimal representation.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for Digest {
    type Err = CoreError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.len() != DIGEST_HEX_BYTES
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(CoreError::invalid(
                "digest",
                "expected 64 lowercase hexadecimal bytes",
            ));
        }
        Ok(Self(value.to_owned()))
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl fmt::Debug for Digest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("Digest").field(&self.0).finish()
    }
}

impl Serialize for Digest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for Digest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        value.parse().map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_and_length_are_bound_into_identity() {
        let a = Digest::of_bytes("a", b"bc");
        let b = Digest::of_bytes("ab", b"c");
        let c = Digest::of_bytes("a", b"bc\0");
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.as_str().len(), DIGEST_HEX_BYTES);
    }

    #[test]
    fn deserialize_rejects_noncanonical_text() {
        let uppercase = format!("\"{}\"", "A".repeat(DIGEST_HEX_BYTES));
        assert!(serde_json::from_str::<Digest>(&uppercase).is_err());
    }
}

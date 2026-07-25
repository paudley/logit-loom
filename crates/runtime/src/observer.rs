// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded construction of generated-token observer fan-out.

use logit_loom::{
    CancellationToken, ControlFlow, CoreError, Digest, MAX_OBSERVERS, ObservedToken, Observer,
    ObserverError, ObserverSet,
};

use crate::{Error, Result};

const CANCELLATION_IDENTITY_DOMAIN: &str = "runtime-cancellation-observer-v1";

struct TokenCallback<F> {
    callback: F,
}

impl<F> Observer for TokenCallback<F>
where
    F: for<'token> FnMut(ObservedToken<'token>) -> std::result::Result<ControlFlow, ObserverError>,
{
    fn on_token(
        &mut self,
        token: ObservedToken<'_>,
    ) -> std::result::Result<ControlFlow, ObserverError> {
        (self.callback)(token)
    }
}

/// Constructs an ordered, bounded [`ObserverSet`].
pub struct ObserversBuilder {
    observers: Vec<(Digest, Box<dyn Observer>)>,
}

impl ObserversBuilder {
    /// Creates an empty observer builder.
    pub const fn new() -> Self {
        Self {
            observers: Vec::new(),
        }
    }

    /// Returns the number of observers currently declared.
    pub fn observer_count(&self) -> usize {
        self.observers.len()
    }

    /// Appends a cooperative cancellation observer.
    ///
    /// The built-in identity describes the cancellation implementation. The
    /// supplied token is cloned so its remote handle remains available.
    ///
    /// # Errors
    ///
    /// Returns an error when the observer bound is already reached.
    pub fn cancellation(mut self, token: &CancellationToken) -> Result<Self> {
        self.ensure_observer_capacity()?;
        let identity = Digest::of_bytes(CANCELLATION_IDENTITY_DOMAIN, b"cooperative-cancellation");
        self.push(identity, Box::new(token.clone()))?;
        Ok(self)
    }

    /// Appends a token callback with a caller-defined stable identity.
    ///
    /// The callback runs synchronously after causal admission. Its default
    /// pre-sampling poll always continues.
    ///
    /// # Errors
    ///
    /// Returns an error when the observer bound is already reached.
    pub fn on_token<F>(mut self, implementation: Digest, callback: F) -> Result<Self>
    where
        F: for<'token> FnMut(
                ObservedToken<'token>,
            ) -> std::result::Result<ControlFlow, ObserverError>
            + 'static,
    {
        self.ensure_observer_capacity()?;
        self.push(implementation, Box::new(TokenCallback { callback }))?;
        Ok(self)
    }

    /// Appends a custom observer with a caller-defined stable identity.
    ///
    /// # Errors
    ///
    /// Returns an error when the observer bound is already reached.
    pub fn observer(
        mut self,
        implementation: Digest,
        observer: impl Observer + 'static,
    ) -> Result<Self> {
        self.ensure_observer_capacity()?;
        self.push(implementation, Box::new(observer))?;
        Ok(self)
    }

    /// Builds the ordered observer set.
    ///
    /// # Errors
    ///
    /// Returns an error unless at least one observer exists and every identity
    /// is unique.
    pub fn build(self) -> Result<ObserverSet> {
        ObserverSet::new(self.observers).map_err(Error::from)
    }

    fn push(&mut self, implementation: Digest, observer: Box<dyn Observer>) -> Result<()> {
        self.ensure_observer_capacity()?;
        self.observers.push((implementation, observer));
        Ok(())
    }

    fn ensure_observer_capacity(&self) -> Result<()> {
        if self.observers.len() >= MAX_OBSERVERS {
            return Err(CoreError::invalid(
                "observer set",
                format!("requires at most {MAX_OBSERVERS} observers"),
            )
            .into());
        }
        Ok(())
    }
}

impl Default for ObserversBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use logit_loom::TokenId;

    use super::*;

    #[test]
    fn callback_observes_exact_non_utf8_bytes() {
        let pieces = Rc::new(RefCell::new(Vec::new()));
        let sink = Rc::clone(&pieces);
        let mut observers = ObserversBuilder::new()
            .on_token(
                Digest::of_bytes("test-observer", b"exact-bytes"),
                move |token| {
                    sink.borrow_mut().push(token.piece.to_vec());
                    Ok(ControlFlow::Continue)
                },
            )
            .unwrap()
            .build()
            .unwrap();
        observers.begin(0, 1).unwrap();
        observers.poll().unwrap();
        observers
            .observe(ObservedToken {
                token: TokenId::new(1).unwrap(),
                piece: &[0xff],
                position: 1,
            })
            .unwrap();
        assert_eq!(*pieces.borrow(), vec![vec![0xff]]);
    }

    #[test]
    fn cancellation_identity_is_stable_and_duplicate_registration_fails() {
        let token = CancellationToken::new();
        let first = ObserversBuilder::new()
            .cancellation(&token)
            .unwrap()
            .build()
            .unwrap();
        let second = ObserversBuilder::new()
            .cancellation(&token)
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(
            first.receipt(0).unwrap().implementation,
            second.receipt(0).unwrap().implementation
        );
        assert!(
            ObserversBuilder::new()
                .cancellation(&token)
                .unwrap()
                .cancellation(&token)
                .unwrap()
                .build()
                .is_err()
        );
    }

    #[test]
    fn empty_observer_builder_is_rejected() {
        assert!(ObserversBuilder::new().build().is_err());
    }

    #[test]
    fn observer_bound_is_enforced_by_the_builder() {
        let mut builder = ObserversBuilder::new();
        for index in 0..MAX_OBSERVERS {
            builder = builder
                .on_token(
                    Digest::of_bytes("test-observer-bound", &index.to_le_bytes()),
                    |_| Ok(ControlFlow::Continue),
                )
                .unwrap();
        }
        assert_eq!(builder.observer_count(), MAX_OBSERVERS);
        assert!(
            builder
                .on_token(
                    Digest::of_bytes("test-observer-bound", b"one-too-many"),
                    |_| Ok(ControlFlow::Continue),
                )
                .is_err()
        );
    }
}

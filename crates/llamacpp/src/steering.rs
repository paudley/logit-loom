// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reversible `LoRA` and control-vector session scopes.

use std::path::Path;

use llama_cpp_4::model::LlamaLoraAdapter;
use logit_loom::{
    ControlVectorSpec, Digest, LoraSpec, MAX_TEXT_MECHANICS_LORAS, SteeringAction, SteeringKind,
    SteeringReceipt,
};

use crate::{
    Error, Model, Session,
    error::native,
    model::{LORA_ARTIFACT_DOMAIN, digest_file},
};

/// Loaded model-compatible `LoRA` adapter.
pub struct LoraAdapter {
    native: LlamaLoraAdapter,
    artifact: Digest,
}

impl std::fmt::Debug for LoraAdapter {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LoraAdapter")
            .field("artifact", &self.artifact)
            .finish_non_exhaustive()
    }
}

impl LoraAdapter {
    /// Returns the adapter-file content identity.
    pub const fn artifact_digest(&self) -> &Digest {
        &self.artifact
    }
}

/// One borrowed `LoRA` and its exact application scale.
pub struct LoraApplication<'adapter> {
    adapter: &'adapter mut LoraAdapter,
    scale: f32,
}

impl<'adapter> LoraApplication<'adapter> {
    /// Selects one loaded adapter for an aggregate steering scope.
    #[must_use]
    pub const fn new(adapter: &'adapter mut LoraAdapter, scale: f32) -> Self {
        Self { adapter, scale }
    }
}

impl Model {
    /// Loads a `LoRA` adapter without applying it to a session.
    ///
    /// # Errors
    ///
    /// Returns an I/O or native compatibility error.
    pub fn load_lora(&self, path: impl AsRef<Path>) -> Result<LoraAdapter, Error> {
        let path = path.as_ref();
        let artifact_before_load = digest_file(path, LORA_ARTIFACT_DOMAIN)?;
        let native = self.native.lora_adapter_init(path).map_err(native)?;
        let artifact = digest_file(path, LORA_ARTIFACT_DOMAIN)?;
        if artifact != artifact_before_load {
            return Err(Error::Incompatible(
                "LoRA file changed while llama.cpp was loading it".to_owned(),
            ));
        }
        Ok(LoraAdapter { native, artifact })
    }
}

/// In-memory native control-vector data and layer contract.
#[derive(Clone, Debug)]
pub struct ControlVector {
    values: Vec<f32>,
    neutral: Vec<f32>,
    specification: ControlVectorSpec,
}

impl ControlVector {
    /// Creates a content-bound control vector.
    ///
    /// # Errors
    ///
    /// Returns an error for non-finite data, invalid dimensions, or a malformed
    /// layer range.
    pub fn new(
        values: Vec<f32>,
        embedding_width: u32,
        layer_start: i32,
        layer_end: i32,
    ) -> Result<Self, Error> {
        if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
            return Err(Error::Invalid(
                "control-vector values must be non-empty and finite".to_owned(),
            ));
        }
        let width = usize::try_from(embedding_width)
            .map_err(|_| Error::Invalid("control-vector width exceeds usize".to_owned()))?;
        if width == 0 || !values.len().is_multiple_of(width) {
            return Err(Error::Invalid(
                "control-vector length must be divisible by embedding width".to_owned(),
            ));
        }
        let byte_capacity = values
            .len()
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| Error::Invalid("control-vector byte length overflowed".to_owned()))?;
        let mut bytes = Vec::with_capacity(byte_capacity);
        for value in &values {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        let specification = ControlVectorSpec {
            data: Digest::of_bytes("control-vector-f32-le-v1", &bytes),
            embedding_width,
            layer_start,
            layer_end,
        };
        specification.validate()?;
        Ok(Self {
            neutral: vec![0.0; values.len()],
            values,
            specification,
        })
    }

    /// Returns the content-bound application contract.
    pub const fn specification(&self) -> &ControlVectorSpec {
        &self.specification
    }
}

impl<'model> Session<'model> {
    /// Applies ordered `LoRA` adapters and an optional control vector as one
    /// reversible aggregate scope.
    ///
    /// Every descriptor is validated before native mutation. `LoRA`s are
    /// applied in caller order, followed by the control vector. Explicit
    /// cleanup removes the control vector first and then removes `LoRA`s in
    /// reverse order. A partial application is rolled back before this method
    /// returns; cleanup uncertainty poisons the session.
    ///
    /// # Errors
    ///
    /// Returns a contract, compatibility, native application, or rollback
    /// error. At least one steering resource must be selected.
    pub fn steering_scope<'scope>(
        &'scope mut self,
        loras: Vec<LoraApplication<'scope>>,
        vector: Option<&'scope ControlVector>,
    ) -> Result<SteeringScope<'scope, 'model>, Error> {
        self.ensure_steering_available()?;
        if loras.is_empty() && vector.is_none() {
            return Err(Error::Invalid(
                "aggregate steering requires at least one resource".to_owned(),
            ));
        }
        if loras.len() > MAX_TEXT_MECHANICS_LORAS {
            return Err(Error::Invalid(format!(
                "aggregate steering exceeds {MAX_TEXT_MECHANICS_LORAS} LoRA applications"
            )));
        }

        let position = self.position();
        let mut active_loras = Vec::with_capacity(loras.len());
        for selection in loras {
            let specification = LoraSpec {
                artifact: selection.adapter.artifact.clone(),
                scale: selection.scale,
            };
            let resource = specification.digest()?;
            let applied = SteeringReceipt {
                kind: SteeringKind::Lora,
                resource: resource.clone(),
                action: SteeringAction::Applied,
                position,
            };
            applied.digest()?;
            active_loras.push(ActiveLora {
                adapter: selection.adapter,
                scale: selection.scale,
                resource,
                applied,
            });
        }
        let active_vector = vector
            .map(|vector| {
                vector.validate_for_model(self.model)?;
                let resource = vector.specification().digest()?;
                let applied = SteeringReceipt {
                    kind: SteeringKind::ControlVector,
                    resource: resource.clone(),
                    action: SteeringAction::Applied,
                    position,
                };
                applied.digest()?;
                Ok::<ActiveControlVector<'scope>, Error>(ActiveControlVector {
                    vector,
                    resource,
                    applied,
                })
            })
            .transpose()?;

        let mut application_failure = None;
        for (index, active) in active_loras.iter_mut().enumerate() {
            if let Err(error) = self
                .context
                .lora_adapter_set(&mut active.adapter.native, active.scale)
                .map_err(native)
            {
                application_failure = Some((index, error));
                break;
            }
        }
        if let Some((applied_loras, error)) = application_failure {
            return Err(rollback_partial_loras(
                self,
                &mut active_loras[..applied_loras],
                error,
            ));
        }
        if let Some(active) = &active_vector
            && let Err(error) = apply_control_vector(self, active.vector)
        {
            return Err(rollback_partial_steering(
                self,
                &mut active_loras,
                Some(active.vector),
                error,
            ));
        }

        let mut kinds = vec![SteeringKind::Lora; active_loras.len()];
        if active_vector.is_some() {
            kinds.push(SteeringKind::ControlVector);
        }
        self.mark_steering_stack_active(kinds);
        Ok(SteeringScope {
            session: self,
            loras: active_loras,
            vector: active_vector,
            active: true,
        })
    }

    /// Applies one `LoRA` adapter until the returned scope is explicitly cleared
    /// or dropped.
    ///
    /// # Errors
    ///
    /// Returns a contract or native application error.
    pub fn lora_scope<'scope>(
        &'scope mut self,
        adapter: &'scope mut LoraAdapter,
        scale: f32,
    ) -> Result<LoraScope<'scope, 'model>, Error> {
        self.ensure_steering_available()?;
        let specification = LoraSpec {
            artifact: adapter.artifact.clone(),
            scale,
        };
        let resource = specification.digest()?;
        let applied = SteeringReceipt {
            kind: SteeringKind::Lora,
            resource: resource.clone(),
            action: SteeringAction::Applied,
            position: self.position(),
        };
        applied.digest()?;
        self.context
            .lora_adapter_set(&mut adapter.native, scale)
            .map_err(native)?;
        self.mark_steering_active(SteeringKind::Lora);
        Ok(LoraScope {
            session: self,
            adapter,
            resource,
            applied,
            active: true,
        })
    }

    /// Applies one control vector until the returned scope is explicitly
    /// cleared or dropped.
    ///
    /// # Errors
    ///
    /// Returns a contract or native application error.
    pub fn control_vector_scope<'scope>(
        &'scope mut self,
        vector: &'scope ControlVector,
    ) -> Result<ControlVectorScope<'scope, 'model>, Error> {
        self.ensure_steering_available()?;
        let specification = vector.specification();
        vector.validate_for_model(self.model)?;
        let resource = specification.digest()?;
        let applied = SteeringReceipt {
            kind: SteeringKind::ControlVector,
            resource: resource.clone(),
            action: SteeringAction::Applied,
            position: self.position(),
        };
        applied.digest()?;
        self.context
            .set_adapter_cvec(
                &vector.values,
                i32::try_from(specification.embedding_width).map_err(|_| {
                    Error::Invalid("control-vector embedding width exceeds i32".to_owned())
                })?,
                specification.layer_start,
                specification.layer_end,
            )
            .map_err(|code| Error::Native(format!("control-vector apply returned {code}")))?;
        self.mark_steering_active(SteeringKind::ControlVector);
        Ok(ControlVectorScope {
            session: self,
            vector,
            resource,
            applied,
            active: true,
        })
    }
}

struct ActiveLora<'adapter> {
    adapter: &'adapter mut LoraAdapter,
    scale: f32,
    resource: Digest,
    applied: SteeringReceipt,
}

struct ActiveControlVector<'vector> {
    vector: &'vector ControlVector,
    resource: Digest,
    applied: SteeringReceipt,
}

/// Active aggregate steering application.
#[must_use = "retain the scope while steering should remain active and call clear to observe cleanup"]
pub struct SteeringScope<'scope, 'model> {
    session: &'scope mut Session<'model>,
    loras: Vec<ActiveLora<'scope>>,
    vector: Option<ActiveControlVector<'scope>>,
    active: bool,
}

impl<'model> SteeringScope<'_, 'model> {
    /// Returns successful application receipts in exact native apply order.
    pub fn applied_receipts(&self) -> impl Iterator<Item = &SteeringReceipt> {
        self.loras
            .iter()
            .map(|active| &active.applied)
            .chain(self.vector.iter().map(|active| &active.applied))
    }

    /// Borrows the causally mutable session while every selected resource
    /// remains active.
    pub fn session_mut(&mut self) -> &mut Session<'model> {
        self.session
    }

    /// Explicitly clears every resource and returns receipts in exact native
    /// cleanup order.
    ///
    /// # Errors
    ///
    /// Returns a native cleanup or receipt error and poisons the session when
    /// any cleanup becomes uncertain.
    pub fn clear(mut self) -> Result<Vec<SteeringReceipt>, Error> {
        let result = self.clear_inner();
        self.active = false;
        if let Err(error) = &result {
            self.session.record_cleanup_failure(error);
        }
        result
    }

    fn clear_inner(&mut self) -> Result<Vec<SteeringReceipt>, Error> {
        let mut cleared = Vec::with_capacity(self.loras.len() + usize::from(self.vector.is_some()));
        let mut failures = Vec::new();
        if let Some(active) = &self.vector {
            match clear_control_vector(self.session, active.vector) {
                Ok(()) => {
                    let receipt = SteeringReceipt {
                        kind: SteeringKind::ControlVector,
                        resource: active.resource.clone(),
                        action: SteeringAction::Cleared,
                        position: self.session.position(),
                    };
                    receipt.digest()?;
                    cleared.push(receipt);
                }
                Err(error) => failures.push(error.to_string()),
            }
        }
        for active in self.loras.iter_mut().rev() {
            match self
                .session
                .context
                .lora_adapter_remove(&mut active.adapter.native)
                .map_err(native)
            {
                Ok(()) => {
                    let receipt = SteeringReceipt {
                        kind: SteeringKind::Lora,
                        resource: active.resource.clone(),
                        action: SteeringAction::Cleared,
                        position: self.session.position(),
                    };
                    receipt.digest()?;
                    cleared.push(receipt);
                }
                Err(error) => failures.push(error.to_string()),
            }
        }
        if !failures.is_empty() {
            return Err(Error::Poisoned(format!(
                "aggregate steering cleanup failed: {}",
                failures.join("; ")
            )));
        }
        self.session.mark_steering_cleared();
        Ok(cleared)
    }
}

impl Drop for SteeringScope<'_, '_> {
    fn drop(&mut self) {
        if self.active
            && let Err(error) = self.clear_inner()
        {
            self.session.record_cleanup_failure(&error);
        }
    }
}

/// Active reversible `LoRA` application.
#[must_use = "retain the scope while LoRA should remain active and call clear to observe cleanup"]
pub struct LoraScope<'scope, 'model> {
    session: &'scope mut Session<'model>,
    adapter: &'scope mut LoraAdapter,
    resource: Digest,
    applied: SteeringReceipt,
    active: bool,
}

impl<'model> LoraScope<'_, 'model> {
    /// Returns the successful application receipt.
    pub const fn applied_receipt(&self) -> &SteeringReceipt {
        &self.applied
    }

    /// Borrows the causally mutable session while `LoRA` remains active.
    pub fn session_mut(&mut self) -> &mut Session<'model> {
        self.session
    }

    /// Explicitly clears `LoRA` and returns observable lifecycle accounting.
    ///
    /// # Errors
    ///
    /// Returns a native cleanup or receipt error.
    pub fn clear(mut self) -> Result<SteeringReceipt, Error> {
        let result = self.clear_inner();
        self.active = false;
        if let Err(error) = &result {
            self.session.record_cleanup_failure(error);
        }
        result
    }

    fn clear_inner(&mut self) -> Result<SteeringReceipt, Error> {
        let receipt = SteeringReceipt {
            kind: SteeringKind::Lora,
            resource: self.resource.clone(),
            action: SteeringAction::Cleared,
            position: self.session.position(),
        };
        receipt.digest()?;
        self.session
            .context
            .lora_adapter_remove(&mut self.adapter.native)
            .map_err(native)?;
        self.session.mark_steering_cleared();
        Ok(receipt)
    }
}

impl Drop for LoraScope<'_, '_> {
    fn drop(&mut self) {
        if self.active
            && let Err(error) = self.clear_inner()
        {
            self.session.record_cleanup_failure(&error);
        }
    }
}

/// Active reversible control-vector application.
#[must_use = "retain the scope while the vector should remain active and call clear to observe cleanup"]
pub struct ControlVectorScope<'scope, 'model> {
    session: &'scope mut Session<'model>,
    vector: &'scope ControlVector,
    resource: Digest,
    applied: SteeringReceipt,
    active: bool,
}

impl<'model> ControlVectorScope<'_, 'model> {
    /// Returns the successful application receipt.
    pub const fn applied_receipt(&self) -> &SteeringReceipt {
        &self.applied
    }

    /// Borrows the causally mutable session while the vector remains active.
    pub fn session_mut(&mut self) -> &mut Session<'model> {
        self.session
    }

    /// Explicitly clears the vector and returns observable lifecycle accounting.
    ///
    /// # Errors
    ///
    /// Returns a native cleanup or receipt error.
    pub fn clear(mut self) -> Result<SteeringReceipt, Error> {
        let result = self.clear_inner();
        self.active = false;
        if let Err(error) = &result {
            self.session.record_cleanup_failure(error);
        }
        result
    }

    fn clear_inner(&mut self) -> Result<SteeringReceipt, Error> {
        let specification = self.vector.specification();
        let receipt = SteeringReceipt {
            kind: SteeringKind::ControlVector,
            resource: self.resource.clone(),
            action: SteeringAction::Cleared,
            position: self.session.position(),
        };
        receipt.digest()?;
        self.session
            .context
            .set_adapter_cvec(
                &self.vector.neutral,
                i32::try_from(specification.embedding_width).map_err(|_| {
                    Error::Invalid("control-vector embedding width exceeds i32".to_owned())
                })?,
                specification.layer_start,
                specification.layer_end,
            )
            .map_err(|code| Error::Native(format!("control-vector clear returned {code}")))?;
        self.session.mark_steering_cleared();
        Ok(receipt)
    }
}

impl Drop for ControlVectorScope<'_, '_> {
    fn drop(&mut self) {
        if self.active
            && let Err(error) = self.clear_inner()
        {
            self.session.record_cleanup_failure(&error);
        }
    }
}

impl ControlVector {
    fn validate_for_model(&self, model: &Model) -> Result<(), Error> {
        let native_width = u32::try_from(model.native.n_embd())
            .map_err(|_| Error::Native("model returned an invalid embedding width".to_owned()))?;
        let native_layers = u32::try_from(model.native.n_layer())
            .map_err(|_| Error::Native("model returned an invalid layer count".to_owned()))?;
        if native_layers < 2 || self.specification.embedding_width != native_width {
            return Err(Error::Incompatible(
                "control-vector embedding width does not match the model".to_owned(),
            ));
        }
        let last_layer = i32::try_from(native_layers - 1)
            .map_err(|_| Error::Native("model layer count exceeds i32".to_owned()))?;
        if self.specification.layer_end > last_layer {
            return Err(Error::Incompatible(format!(
                "control-vector layer end exceeds model layer {last_layer}"
            )));
        }
        let width = usize::try_from(native_width)
            .map_err(|_| Error::Native("model embedding width exceeds usize".to_owned()))?;
        let rows = usize::try_from(native_layers - 1)
            .map_err(|_| Error::Native("model layer count exceeds usize".to_owned()))?;
        let expected = width
            .checked_mul(rows)
            .ok_or_else(|| Error::Invalid("control-vector dimensions overflowed".to_owned()))?;
        if self.values.len() != expected {
            return Err(Error::Incompatible(format!(
                "control vector contains {} values; model requires {expected}",
                self.values.len()
            )));
        }
        Ok(())
    }
}

fn apply_control_vector(session: &mut Session<'_>, vector: &ControlVector) -> Result<(), Error> {
    let specification = vector.specification();
    session
        .context
        .set_adapter_cvec(
            &vector.values,
            i32::try_from(specification.embedding_width).map_err(|_| {
                Error::Invalid("control-vector embedding width exceeds i32".to_owned())
            })?,
            specification.layer_start,
            specification.layer_end,
        )
        .map_err(|code| Error::Native(format!("control-vector apply returned {code}")))
}

fn clear_control_vector(session: &mut Session<'_>, vector: &ControlVector) -> Result<(), Error> {
    let specification = vector.specification();
    session
        .context
        .set_adapter_cvec(
            &vector.neutral,
            i32::try_from(specification.embedding_width).map_err(|_| {
                Error::Invalid("control-vector embedding width exceeds i32".to_owned())
            })?,
            specification.layer_start,
            specification.layer_end,
        )
        .map_err(|code| Error::Native(format!("control-vector clear returned {code}")))
}

fn rollback_partial_loras(
    session: &mut Session<'_>,
    loras: &mut [ActiveLora<'_>],
    application_error: Error,
) -> Error {
    let mut failures = Vec::new();
    for active in loras.iter_mut().rev() {
        if let Err(cleanup_error) = session
            .context
            .lora_adapter_remove(&mut active.adapter.native)
            .map_err(native)
        {
            failures.push(cleanup_error.to_string());
        }
    }
    if failures.is_empty() {
        application_error
    } else {
        let error = Error::Poisoned(format!(
            "steering application failed ({application_error}); partial LoRA rollback failed: {}",
            failures.join("; ")
        ));
        session.record_cleanup_failure(&error);
        error
    }
}

fn rollback_partial_steering(
    session: &mut Session<'_>,
    loras: &mut [ActiveLora<'_>],
    vector: Option<&ControlVector>,
    application_error: Error,
) -> Error {
    let mut failures = Vec::new();
    if let Some(vector) = vector
        && let Err(cleanup_error) = clear_control_vector(session, vector)
    {
        failures.push(format!("control vector: {cleanup_error}"));
    }
    for active in loras.iter_mut().rev() {
        if let Err(cleanup_error) = session
            .context
            .lora_adapter_remove(&mut active.adapter.native)
            .map_err(native)
        {
            failures.push(format!("LoRA: {cleanup_error}"));
        }
    }
    if failures.is_empty() {
        application_error
    } else {
        let error = Error::Poisoned(format!(
            "steering application failed ({application_error}); aggregate rollback failed: {}",
            failures.join("; ")
        ));
        session.record_cleanup_failure(&error);
        error
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_vector_builds_a_matching_neutral_buffer() {
        let vector = ControlVector::new(vec![1.0, -1.0, 0.5, 0.0], 2, 1, 1).unwrap();
        assert_eq!(vector.values.len(), vector.neutral.len());
        assert!(vector.neutral.iter().all(|value| *value == 0.0));
        assert_eq!(vector.specification.layer_end, 1);
    }

    #[test]
    fn control_vector_rejects_malformed_values_and_ranges() {
        assert!(ControlVector::new(vec![f32::NAN, 0.0], 2, 1, 1).is_err());
        assert!(ControlVector::new(vec![0.0, 1.0, 2.0], 2, 1, 1).is_err());
        assert!(ControlVector::new(vec![0.0, 1.0], 2, 0, 1).is_err());
    }
}

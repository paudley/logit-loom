// SPDX-License-Identifier: MIT OR Apache-2.0

//! Topology-bound activation capture and transactional tensor programs.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, MutexGuard};

use llama_cpp_4::context::tensor_transaction::MAX_TENSOR_ROWS;
use llama_cpp_4::context::{
    CapturedTensorData, TensorAccess, TensorDataMut, TensorElementType, TensorRowMapping,
    TensorSelector, TensorTransaction, TensorTransactionError, TensorTransactionHandler,
    TensorTransactions, TensorWriteback, TransactionalTensorCapture,
};
use logit_loom::{
    ActivationCapturePlanV1, ActivationCapturePositionsV1, ActivationCaptureReceiptV1,
    ActivationCaptureRecordV1, ActivationCapturedDataV1, ActivationInvocationReceiptV1,
    ActivationInvocationRowV1, ActivationOperatorV1, ActivationPhaseV1, ActivationProgramReceiptV1,
    ActivationProgramV1, ActivationTelemetryDispositionV1, ActivationVectorBankV1,
    ActivationVectorNormalizationV1, Digest, MAX_ACTIVATION_INVOCATION_RECEIPTS,
    SpeculationTelemetryResolutionV1, TextTensorElementTypeV1, TextTensorSiteV1,
    activation_f32_row_identity,
};

use crate::{Error, Model};

/// Explicit llama.cpp graph-site compatibility profile.
///
/// Residual layer outputs and exact named sites require no additional layer
/// declaration. `MoE` router and selected-expert sites are accepted only for
/// the caller-declared layers in this profile; the profile identity remains
/// bound to the model's exact selector implementation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LlamaCppTensorProfile {
    selector_implementation: Digest,
    moe_layers: Vec<u32>,
    identity: Digest,
}

impl LlamaCppTensorProfile {
    /// Constructs a profile with no architecture-specific `MoE` sites.
    ///
    /// Layer-output and exact named-site mechanics remain available.
    ///
    /// # Errors
    ///
    /// Returns an identity-encoding error.
    pub fn layer_outputs(model: &Model) -> Result<Self, Error> {
        Self::new(model, Vec::new())
    }

    /// Constructs a profile with canonically ordered, explicitly qualified
    /// `MoE` layers.
    ///
    /// # Errors
    ///
    /// Returns an error for duplicate, unordered, or out-of-range layers, or
    /// identity encoding failure.
    pub fn new(model: &Model, moe_layers: Vec<u32>) -> Result<Self, Error> {
        if moe_layers.windows(2).any(|window| window[0] >= window[1])
            || moe_layers
                .iter()
                .any(|layer| *layer >= model.topology().layers)
            || (!moe_layers.is_empty() && model.topology().experts.is_none())
        {
            return Err(Error::Invalid(
                "MoE profile layers must be unique, ordered, in range, and expert-backed"
                    .to_owned(),
            ));
        }
        let selector_implementation = model.tensor_selector_implementation().clone();
        let identity = Digest::of_serializable(
            "llamacpp-tensor-profile-v1",
            &(
                model.topology().digest()?,
                &selector_implementation,
                &moe_layers,
            ),
        )?;
        Ok(Self {
            selector_implementation,
            moe_layers,
            identity,
        })
    }

    /// Returns the exact compatibility-profile identity.
    pub const fn identity(&self) -> &Digest {
        &self.identity
    }

    fn supports(&self, site: &TextTensorSiteV1) -> bool {
        match site {
            TextTensorSiteV1::LayerOutput { .. } => true,
            TextTensorSiteV1::RouterLogits { layer, selector }
            | TextTensorSiteV1::RouterProbabilities { layer, selector }
            | TextTensorSiteV1::SelectedExperts { layer, selector } => {
                selector == &self.selector_implementation
                    && self.moe_layers.binary_search(layer).is_ok()
            }
            TextTensorSiteV1::Named { selector, .. } => selector == &self.selector_implementation,
        }
    }
}

/// Validated activation program, vector banks, captures, and graph profile.
#[derive(Clone, Debug, PartialEq)]
pub struct ActivationConfiguration {
    topology: logit_loom::TextModelTopologyV1,
    program: ActivationProgramV1,
    program_identity: Digest,
    captures: Vec<(Digest, ActivationCapturePlanV1)>,
    banks: BTreeMap<Digest, ActivationVectorBankV1>,
    profile: LlamaCppTensorProfile,
}

impl ActivationConfiguration {
    /// Validates one complete activation runtime before context allocation.
    ///
    /// Every vector reference must resolve exactly, every capture identity must
    /// match the program's canonical observation list, and every selected
    /// graph site must be present in the supplied compatibility profile.
    ///
    /// # Errors
    ///
    /// Returns an error for any topology, profile, vector, operation, capture,
    /// ordering, or identity mismatch.
    pub fn new(
        model: &Model,
        profile: LlamaCppTensorProfile,
        program: ActivationProgramV1,
        capture_plans: Vec<ActivationCapturePlanV1>,
        vector_banks: Vec<ActivationVectorBankV1>,
    ) -> Result<Self, Error> {
        if profile.selector_implementation != *model.tensor_selector_implementation() {
            return Err(Error::Incompatible(
                "tensor profile belongs to another selector implementation".to_owned(),
            ));
        }
        let topology = model.topology().clone();
        let program_identity = program.digest_for(&topology)?;

        let mut captures = Vec::with_capacity(capture_plans.len());
        for plan in capture_plans {
            let identity = plan.digest_for(&topology)?;
            if let Some((prior, _)) = captures.last()
                && prior >= &identity
            {
                return Err(Error::Invalid(
                    "capture plans must have unique canonical identity order".to_owned(),
                ));
            }
            captures.push((identity, plan));
        }
        let capture_identities = captures
            .iter()
            .map(|(identity, _)| identity.clone())
            .collect::<Vec<_>>();
        if capture_identities != program.observations {
            return Err(Error::Incompatible(
                "activation capture plans differ from program observations".to_owned(),
            ));
        }

        let mut banks = BTreeMap::new();
        for bank in vector_banks {
            let identity = bank.digest_for(&topology)?;
            if banks.insert(identity, bank).is_some() {
                return Err(Error::Invalid(
                    "activation vector-bank identities must be unique".to_owned(),
                ));
            }
        }
        let referenced = program
            .operations
            .iter()
            .map(|operation| operation.vector_bank.clone())
            .collect::<BTreeSet<_>>();
        if referenced.len() != banks.len()
            || referenced
                .iter()
                .any(|identity| !banks.contains_key(identity))
        {
            return Err(Error::Incompatible(
                "activation vector banks must exactly resolve program references".to_owned(),
            ));
        }
        for operation in &program.operations {
            let bank = banks.get(&operation.vector_bank).ok_or_else(|| {
                Error::Incompatible("activation vector bank is unavailable".to_owned())
            })?;
            if operation.operator == ActivationOperatorV1::ScaledProjectionRemoval
                && bank.normalization != ActivationVectorNormalizationV1::UnitL2
            {
                return Err(Error::Invalid(
                    "projection removal requires a unit-L2 vector bank".to_owned(),
                ));
            }
            for site in &operation.sites {
                validate_profile_site(&profile, site)?;
                if bank.site_family != site.family()
                    || bank.row_elements != site.row_elements(&topology)?
                    || site.layer().and_then(|layer| bank.row(layer)).is_none()
                {
                    return Err(Error::Incompatible(
                        "activation vector bank does not cover an operation site".to_owned(),
                    ));
                }
            }
        }
        for (_, plan) in &captures {
            for site in &plan.sites {
                validate_profile_site(&profile, site)?;
                if plan.retention == logit_loom::ActivationCaptureRetentionV1::Statistics
                    && site.element_type() != TextTensorElementTypeV1::F32
                {
                    return Err(Error::Invalid(
                        "statistics capture requires f32 tensor sites".to_owned(),
                    ));
                }
            }
        }
        Ok(Self {
            topology,
            program,
            program_identity,
            captures,
            banks,
            profile,
        })
    }

    /// Returns the exact activation-program identity.
    pub const fn program_identity(&self) -> &Digest {
        &self.program_identity
    }

    /// Returns the exact graph compatibility-profile identity.
    pub const fn profile_identity(&self) -> &Digest {
        self.profile.identity()
    }

    pub(crate) fn lower(
        self,
        maximum_rows: u32,
    ) -> Result<(ActivationController, TensorTransactions), Error> {
        let maximum_rows = usize::try_from(maximum_rows)
            .map_err(|_| Error::Invalid("activation batch rows exceed usize".to_owned()))?;
        if maximum_rows == 0 || maximum_rows > MAX_TENSOR_ROWS {
            return Err(Error::Invalid(format!(
                "activation batch rows must be between 1 and {MAX_TENSOR_ROWS}"
            )));
        }

        let program = Arc::new(self.program);
        let banks = Arc::new(self.banks);
        let captures = Arc::new(self.captures);
        let mut selected_sites = BTreeSet::new();
        for operation in &program.operations {
            selected_sites.extend(operation.sites.iter().cloned());
        }
        for (_, plan) in captures.iter() {
            selected_sites.extend(plan.sites.iter().cloned());
        }

        let mutable_sites = program
            .operations
            .iter()
            .flat_map(|operation| operation.sites.iter().cloned())
            .collect::<BTreeSet<_>>();
        let retained_sites = captures
            .iter()
            .flat_map(|(_, plan)| plan.sites.iter().cloned())
            .collect::<BTreeSet<_>>();

        let mut site_bindings = BTreeMap::new();
        for site in selected_sites {
            validate_profile_site(&self.profile, &site)?;
            let name = native_tensor_name(&site);
            let binding = SiteBinding {
                site: site.clone(),
                name: name.clone(),
            };
            if let Some(prior) = site_bindings.insert(name.clone(), binding)
                && prior.site != site
            {
                return Err(Error::Incompatible(format!(
                    "multiple tensor sites resolve to native graph node {name}"
                )));
            }
        }
        let site_bindings = Arc::new(site_bindings);
        let mut selectors = Vec::with_capacity(site_bindings.len());
        for binding in site_bindings.values() {
            let site = &binding.site;
            let element_type = match site.element_type() {
                TextTensorElementTypeV1::F32 => TensorElementType::F32,
                TextTensorElementTypeV1::I32 => TensorElementType::I32,
            };
            let access = if mutable_sites.contains(site) {
                TensorAccess::ReadWriteF32
            } else {
                TensorAccess::ReadOnly
            };
            selectors.push(
                TensorSelector::new(
                    binding.name.clone(),
                    element_type,
                    usize::try_from(site.row_elements(&self.topology)?)
                        .map_err(|_| Error::Invalid("tensor row width exceeds usize".to_owned()))?,
                    maximum_rows,
                    access,
                    TensorRowMapping::BatchTokens,
                    retained_sites.contains(site),
                )
                .map_err(|error| Error::Invalid(error.to_string()))?,
            );
        }

        let state = Arc::new(Mutex::new(ActivationState::default()));
        let controller = ActivationController {
            topology: self.topology,
            program: Arc::clone(&program),
            program_identity: self.program_identity,
            captures: Arc::clone(&captures),
            site_bindings: Arc::clone(&site_bindings),
            state: Arc::clone(&state),
            captured_records: Vec::new(),
            capture_aggregates: BTreeMap::new(),
            provisional_captures: Vec::new(),
            provisional_capture_aggregates: BTreeMap::new(),
        };
        let transactions = if mutable_sites.is_empty() {
            TensorTransactions::capture(selectors)
        } else {
            TensorTransactions::new(
                selectors,
                ActivationHandler {
                    program,
                    banks,
                    sites: site_bindings,
                    state,
                },
            )
        }
        .map_err(|error| Error::Invalid(error.to_string()))?;
        Ok((controller, transactions))
    }
}

/// Captured records and per-plan aggregate receipts drained from a session.
#[derive(Clone, Debug, PartialEq)]
pub struct ActivationCaptureOutput {
    /// Captured records in native execution order, then capture-plan order.
    pub records: Vec<ActivationCaptureRecordV1>,
    /// Aggregate receipts in canonical capture-plan identity order.
    pub receipts: Vec<ActivationCaptureReceiptV1>,
}

/// Transaction records and aggregate program receipt drained from a session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActivationProgramOutput {
    /// Successful native write-back records in execution order.
    pub invocations: Vec<ActivationInvocationReceiptV1>,
    /// Aggregate activation-program receipt.
    pub receipt: ActivationProgramReceiptV1,
}

#[derive(Clone, Debug)]
struct SiteBinding {
    site: TextTensorSiteV1,
    name: String,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ActiveDecode {
    phase: ActivationPhaseV1,
    last_prefill: bool,
    disposition: ActivationTelemetryDispositionV1,
}

#[derive(Debug, Default)]
struct ActivationState {
    active: Option<ActiveDecode>,
    invocations: Vec<ActivationInvocationReceiptV1>,
    invocation_identities: Vec<Digest>,
    provisional_invocations: Vec<ProvisionalInvocation>,
    elements_copied: u64,
    write_backs: u64,
}

#[derive(Debug, Default)]
struct CaptureAggregate {
    records: Vec<Digest>,
    elements: u64,
    retained_bytes: u64,
}

#[derive(Clone, Debug)]
struct ProvisionalCapture {
    record: ActivationCaptureRecordV1,
    identity: Digest,
}

#[derive(Clone, Debug)]
struct ProvisionalInvocation {
    invocation: ActivationInvocationReceiptV1,
}

#[derive(Debug)]
pub(crate) struct ActivationController {
    topology: logit_loom::TextModelTopologyV1,
    program: Arc<ActivationProgramV1>,
    program_identity: Digest,
    captures: Arc<Vec<(Digest, ActivationCapturePlanV1)>>,
    site_bindings: Arc<BTreeMap<String, SiteBinding>>,
    state: Arc<Mutex<ActivationState>>,
    captured_records: Vec<ActivationCaptureRecordV1>,
    capture_aggregates: BTreeMap<Digest, CaptureAggregate>,
    provisional_captures: Vec<ProvisionalCapture>,
    provisional_capture_aggregates: BTreeMap<Digest, CaptureAggregate>,
}

impl ActivationController {
    pub(crate) fn has_last_prefill_capture(&self) -> bool {
        self.captures
            .iter()
            .any(|(_, plan)| plan.positions == ActivationCapturePositionsV1::LastPrefillToken)
    }

    pub(crate) fn begin_decode(
        &self,
        phase: ActivationPhaseV1,
        last_prefill: bool,
        disposition: ActivationTelemetryDispositionV1,
    ) -> Result<(), Error> {
        if !self.provisional_captures.is_empty() {
            return Err(Error::Poisoned(
                "provisional activation captures must be resolved before another decode".to_owned(),
            ));
        }
        let mut state = lock_state(&self.state)?;
        if state.active.is_some() {
            return Err(Error::Poisoned(
                "activation decode phase was already active".to_owned(),
            ));
        }
        if !state.provisional_invocations.is_empty() {
            return Err(Error::Poisoned(
                "provisional activation invocations must be resolved before another decode"
                    .to_owned(),
            ));
        }
        state.active = Some(ActiveDecode {
            phase,
            last_prefill,
            disposition,
        });
        Ok(())
    }

    pub(crate) fn end_decode(&self) -> Result<ActiveDecode, Error> {
        lock_state(&self.state)?
            .active
            .take()
            .ok_or_else(|| Error::Poisoned("activation decode phase was not active".to_owned()))
    }

    pub(crate) fn consume_captures(
        &mut self,
        native: Vec<TransactionalTensorCapture>,
        decode: ActiveDecode,
    ) -> Result<(), Error> {
        let capture_plans = Arc::clone(&self.captures);
        for capture in native {
            let site = self
                .site_bindings
                .get(&capture.name)
                .ok_or_else(|| {
                    Error::Incompatible(format!(
                        "native capture {} has no site binding",
                        capture.name
                    ))
                })?
                .site
                .clone();
            if capture.rows.len() != capture.shape.rows
                || capture.shape.elements
                    != capture
                        .shape
                        .row_elements
                        .saturating_mul(capture.shape.rows)
            {
                return Err(Error::Incompatible(
                    "native capture row accounting is inconsistent".to_owned(),
                ));
            }
            for (row_index, row) in capture.rows.iter().enumerate() {
                let position = u64::try_from(row.position).map_err(|_| {
                    Error::Incompatible("native capture position is negative".to_owned())
                })?;
                let start = row_index
                    .checked_mul(capture.shape.row_elements)
                    .ok_or_else(|| Error::Invalid("capture row offset overflowed".to_owned()))?;
                let end = start
                    .checked_add(capture.shape.row_elements)
                    .ok_or_else(|| Error::Invalid("capture row end overflowed".to_owned()))?;
                for (plan_identity, plan) in capture_plans.iter() {
                    if !plan.sites.contains(&site)
                        || !capture_position_selected(plan, position, decode, capture.shape.rows)
                    {
                        continue;
                    }
                    let retained = match &capture.data {
                        CapturedTensorData::F32(values) => {
                            ActivationCapturedDataV1::from_f32(&values[start..end], plan.retention)?
                        }
                        CapturedTensorData::I32(values) => {
                            ActivationCapturedDataV1::from_i32(&values[start..end], plan.retention)?
                        }
                    };
                    let record = ActivationCaptureRecordV1 {
                        plan: plan_identity.clone(),
                        site: site.clone(),
                        position,
                        disposition: decode.disposition,
                        retained,
                    };
                    let record_identity = record.digest_for(plan, &self.topology)?;
                    if decode.disposition == ActivationTelemetryDispositionV1::Provisional {
                        self.push_provisional_capture(
                            plan_identity,
                            plan,
                            record,
                            record_identity,
                        )?;
                    } else {
                        self.commit_capture(plan_identity, plan, record, record_identity)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn push_provisional_capture(
        &mut self,
        plan_identity: &Digest,
        plan: &ActivationCapturePlanV1,
        record: ActivationCaptureRecordV1,
        identity: Digest,
    ) -> Result<(), Error> {
        let elements = u64::from(record.retained.elements());
        let bytes = retained_bytes(&record.retained);
        let committed = self.capture_aggregates.get(plan_identity);
        let pending = self.provisional_capture_aggregates.get(plan_identity);
        let pending_elements = pending
            .map_or(0, |aggregate| aggregate.elements)
            .checked_add(elements)
            .ok_or_else(|| Error::Invalid("capture element accounting overflowed".to_owned()))?;
        let pending_bytes = pending
            .map_or(0, |aggregate| aggregate.retained_bytes)
            .checked_add(bytes)
            .ok_or_else(|| {
                Error::Invalid("capture retained-byte accounting overflowed".to_owned())
            })?;
        let total_elements = committed
            .map_or(0, |aggregate| aggregate.elements)
            .checked_add(pending_elements)
            .ok_or_else(|| Error::Invalid("capture element accounting overflowed".to_owned()))?;
        let total_bytes = committed
            .map_or(0, |aggregate| aggregate.retained_bytes)
            .checked_add(pending_bytes)
            .ok_or_else(|| {
                Error::Invalid("capture retained-byte accounting overflowed".to_owned())
            })?;
        if total_elements > plan.maximum_elements || total_bytes > plan.maximum_retained_bytes {
            return Err(Error::Invalid(
                "activation capture exceeded its declared aggregate bound".to_owned(),
            ));
        }
        let aggregate = self
            .provisional_capture_aggregates
            .entry(plan_identity.clone())
            .or_default();
        aggregate.elements = pending_elements;
        aggregate.retained_bytes = pending_bytes;
        aggregate.records.push(identity.clone());
        self.provisional_captures
            .push(ProvisionalCapture { record, identity });
        Ok(())
    }

    fn commit_capture(
        &mut self,
        plan_identity: &Digest,
        plan: &ActivationCapturePlanV1,
        record: ActivationCaptureRecordV1,
        identity: Digest,
    ) -> Result<(), Error> {
        let aggregate = self
            .capture_aggregates
            .entry(plan_identity.clone())
            .or_default();
        let elements = u64::from(record.retained.elements());
        let bytes = retained_bytes(&record.retained);
        let next_elements = aggregate
            .elements
            .checked_add(elements)
            .ok_or_else(|| Error::Invalid("capture element accounting overflowed".to_owned()))?;
        let next_bytes = aggregate.retained_bytes.checked_add(bytes).ok_or_else(|| {
            Error::Invalid("capture retained-byte accounting overflowed".to_owned())
        })?;
        if next_elements > plan.maximum_elements || next_bytes > plan.maximum_retained_bytes {
            return Err(Error::Invalid(
                "activation capture exceeded its declared aggregate bound".to_owned(),
            ));
        }
        aggregate.elements = next_elements;
        aggregate.retained_bytes = next_bytes;
        aggregate.records.push(identity);
        self.captured_records.push(record);
        Ok(())
    }

    pub(crate) fn resolve_provisional(
        &mut self,
        admitted_position_exclusive: u64,
    ) -> Result<Vec<SpeculationTelemetryResolutionV1>, Error> {
        let capture_plans = Arc::clone(&self.captures);
        let mut resolved_captures = Vec::with_capacity(self.provisional_captures.len());
        for pending in &self.provisional_captures {
            let (_, plan) = capture_plans
                .iter()
                .find(|(identity, _)| identity == &pending.record.plan)
                .ok_or_else(|| Error::Incompatible("provisional capture has no plan".to_owned()))?;
            let mut record = pending.record.clone();
            record.disposition = if record.position < admitted_position_exclusive {
                ActivationTelemetryDispositionV1::Admitted
            } else {
                ActivationTelemetryDispositionV1::Rejected
            };
            let identity = record.digest_for(plan, &self.topology)?;
            let disposition = record.disposition;
            resolved_captures.push((
                record,
                identity.clone(),
                SpeculationTelemetryResolutionV1 {
                    provisional: pending.identity.clone(),
                    disposition,
                    resolved: identity,
                },
            ));
        }

        let mut state = lock_state(&self.state)?;
        if state.active.is_some() {
            return Err(Error::Poisoned(
                "cannot resolve activation telemetry during a decode".to_owned(),
            ));
        }
        let mut resolved_invocations = Vec::with_capacity(state.provisional_invocations.len());
        for pending in &state.provisional_invocations {
            let mut invocation = pending.invocation.clone();
            for row in &mut invocation.rows {
                row.disposition = if row.position < admitted_position_exclusive {
                    ActivationTelemetryDispositionV1::Admitted
                } else {
                    ActivationTelemetryDispositionV1::Rejected
                };
            }
            let identity = invocation.digest()?;
            resolved_invocations.push((invocation, identity));
        }
        let final_invocation_count = state
            .invocation_identities
            .len()
            .checked_add(resolved_invocations.len())
            .ok_or_else(|| Error::Invalid("activation invocation count overflowed".to_owned()))?;
        if final_invocation_count > MAX_ACTIVATION_INVOCATION_RECEIPTS {
            return Err(Error::Invalid(
                "activation invocation count exceeded its supported bound".to_owned(),
            ));
        }
        let added_elements =
            resolved_invocations
                .iter()
                .try_fold(0_u64, |total, (invocation, _)| {
                    total
                        .checked_add(u64::from(invocation.elements))
                        .ok_or_else(|| {
                            Error::Invalid("activation element accounting overflowed".to_owned())
                        })
                })?;
        let final_elements = state
            .elements_copied
            .checked_add(added_elements)
            .ok_or_else(|| Error::Invalid("activation element accounting overflowed".to_owned()))?;
        let added_write_backs = u64::try_from(resolved_invocations.len())
            .map_err(|_| Error::Invalid("activation write-back count exceeds u64".to_owned()))?;
        let final_write_backs = state
            .write_backs
            .checked_add(added_write_backs)
            .ok_or_else(|| {
                Error::Invalid("activation write-back accounting overflowed".to_owned())
            })?;

        let mut resolutions = Vec::with_capacity(resolved_captures.len());
        for (invocation, identity) in resolved_invocations {
            state.invocations.push(invocation);
            state.invocation_identities.push(identity);
        }
        state.elements_copied = final_elements;
        state.write_backs = final_write_backs;
        state.provisional_invocations.clear();
        drop(state);

        for (record, identity, resolution) in resolved_captures {
            let (_, plan) = capture_plans
                .iter()
                .find(|(plan_identity, _)| plan_identity == &record.plan)
                .ok_or_else(|| Error::Incompatible("resolved capture has no plan".to_owned()))?;
            self.commit_capture(&record.plan.clone(), plan, record, identity)?;
            resolutions.push(resolution);
        }
        self.provisional_captures.clear();
        self.provisional_capture_aggregates.clear();
        Ok(resolutions)
    }

    pub(crate) fn take_capture_output(&mut self) -> Result<ActivationCaptureOutput, Error> {
        if !self.provisional_captures.is_empty() {
            return Err(Error::Invalid(
                "cannot drain activation captures before provisional resolution".to_owned(),
            ));
        }
        let mut receipts = Vec::with_capacity(self.capture_aggregates.len());
        for (plan_identity, aggregate) in std::mem::take(&mut self.capture_aggregates) {
            let (_, plan) = self
                .captures
                .iter()
                .find(|(identity, _)| identity == &plan_identity)
                .ok_or_else(|| Error::Incompatible("capture aggregate has no plan".to_owned()))?;
            let receipt = ActivationCaptureReceiptV1 {
                plan: plan_identity,
                records: aggregate.records,
                elements: aggregate.elements,
                retained_bytes: aggregate.retained_bytes,
            };
            receipt.digest_for(plan)?;
            receipts.push(receipt);
        }
        Ok(ActivationCaptureOutput {
            records: std::mem::take(&mut self.captured_records),
            receipts,
        })
    }

    pub(crate) fn take_program_output(&self) -> Result<ActivationProgramOutput, Error> {
        let mut state = lock_state(&self.state)?;
        if state.active.is_some() {
            return Err(Error::Invalid(
                "cannot drain activation receipts during a decode".to_owned(),
            ));
        }
        if !state.provisional_invocations.is_empty() {
            return Err(Error::Invalid(
                "cannot drain activation receipts before provisional resolution".to_owned(),
            ));
        }
        let receipt = ActivationProgramReceiptV1 {
            program: self.program_identity.clone(),
            topology: self.topology.digest()?,
            invocations: std::mem::take(&mut state.invocation_identities),
            elements_copied: std::mem::take(&mut state.elements_copied),
            write_backs: std::mem::take(&mut state.write_backs),
        };
        receipt.digest_for(&self.program)?;
        Ok(ActivationProgramOutput {
            invocations: std::mem::take(&mut state.invocations),
            receipt,
        })
    }

    pub(crate) fn reset(&mut self) -> Result<(), Error> {
        if !self.provisional_captures.is_empty() {
            return Err(Error::Poisoned(
                "cannot reset unresolved provisional activation captures".to_owned(),
            ));
        }
        self.captured_records.clear();
        self.capture_aggregates.clear();
        self.provisional_capture_aggregates.clear();
        let mut state = lock_state(&self.state)?;
        if state.active.is_some() {
            return Err(Error::Poisoned(
                "cannot reset activation evidence during a decode".to_owned(),
            ));
        }
        if !state.provisional_invocations.is_empty() {
            return Err(Error::Poisoned(
                "cannot reset unresolved provisional activation invocations".to_owned(),
            ));
        }
        state.invocations.clear();
        state.invocation_identities.clear();
        state.elements_copied = 0;
        state.write_backs = 0;
        Ok(())
    }
}

struct ActivationHandler {
    program: Arc<ActivationProgramV1>,
    banks: Arc<BTreeMap<Digest, ActivationVectorBankV1>>,
    sites: Arc<BTreeMap<String, SiteBinding>>,
    state: Arc<Mutex<ActivationState>>,
}

impl TensorTransactionHandler for ActivationHandler {
    #[allow(
        clippy::too_many_lines,
        reason = "the callback keeps validation, ordered row transforms, and receipt commit in one transactional scope"
    )]
    fn apply(
        &mut self,
        mut transaction: TensorTransaction<'_>,
    ) -> Result<TensorWriteback, TensorTransactionError> {
        let binding = self
            .sites
            .get(transaction.name)
            .ok_or_else(|| TensorTransactionError::new("selected tensor has no site binding"))?;
        let TensorDataMut::F32(values) = &mut transaction.data else {
            return Err(TensorTransactionError::new(
                "mutable activation transaction is not f32",
            ));
        };
        let mut state = self
            .state
            .lock()
            .map_err(|_| TensorTransactionError::new("activation state mutex was poisoned"))?;
        let active = state.active.ok_or_else(|| {
            TensorTransactionError::new("activation callback ran outside an active decode phase")
        })?;
        let input = activation_f32_row_identity(values)
            .map_err(|error| TensorTransactionError::new(error.to_string()))?;
        let mut changed_rows = Vec::new();
        for (row_index, row) in transaction.rows.iter().enumerate() {
            let position = u64::try_from(row.position)
                .map_err(|_| TensorTransactionError::new("native causal position is negative"))?;
            let start = row_index
                .checked_mul(transaction.shape.row_elements)
                .ok_or_else(|| TensorTransactionError::new("activation row offset overflowed"))?;
            let end = start
                .checked_add(transaction.shape.row_elements)
                .ok_or_else(|| TensorTransactionError::new("activation row end overflowed"))?;
            let row_values = &mut values[start..end];
            let mut operations = Vec::new();
            for (operation_index, operation) in self.program.operations.iter().enumerate() {
                if operation.phase != active.phase
                    || !operation.positions.contains(position)
                    || !operation.sites.contains(&binding.site)
                {
                    continue;
                }
                let bank = self.banks.get(&operation.vector_bank).ok_or_else(|| {
                    TensorTransactionError::new("activation vector bank is unavailable")
                })?;
                let layer = binding.site.layer().ok_or_else(|| {
                    TensorTransactionError::new("mutable activation site has no layer")
                })?;
                let vector = bank.row(layer).ok_or_else(|| {
                    TensorTransactionError::new("activation vector row is unavailable")
                })?;
                apply_operator(
                    row_values,
                    vector,
                    operation.operator,
                    operation
                        .scale()
                        .map_err(|error| TensorTransactionError::new(error.to_string()))?,
                )?;
                operations.push(u32::try_from(operation_index).map_err(|_| {
                    TensorTransactionError::new("activation operation index exceeds u32")
                })?);
            }
            if !operations.is_empty() {
                let sequence_ids = row
                    .sequence_ids
                    .iter()
                    .map(|sequence| {
                        u32::try_from(*sequence).map_err(|_| {
                            TensorTransactionError::new("native activation sequence id is negative")
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if sequence_ids.windows(2).any(|window| window[0] >= window[1]) {
                    return Err(TensorTransactionError::new(
                        "native activation sequence ids are not canonical",
                    ));
                }
                changed_rows.push(ActivationInvocationRowV1 {
                    batch_index: row.batch_index,
                    operations,
                    sequence_ids,
                    position,
                    disposition: active.disposition,
                });
            }
        }
        let output = activation_f32_row_identity(values)
            .map_err(|error| TensorTransactionError::new(error.to_string()))?;
        if changed_rows.is_empty() || input == output {
            return Ok(TensorWriteback::Unchanged);
        }
        let elements = u32::try_from(transaction.shape.elements)
            .map_err(|_| TensorTransactionError::new("tensor element count exceeds u32"))?;
        let invocation = ActivationInvocationReceiptV1 {
            site: binding.site.clone(),
            phase: active.phase,
            rows: changed_rows,
            elements,
            input,
            output,
        };
        let identity = invocation
            .digest()
            .map_err(|error| TensorTransactionError::new(error.to_string()))?;
        let invocation_count = state
            .invocation_identities
            .len()
            .checked_add(state.provisional_invocations.len())
            .ok_or_else(|| TensorTransactionError::new("activation invocation count overflowed"))?;
        if invocation_count >= MAX_ACTIVATION_INVOCATION_RECEIPTS {
            return Err(TensorTransactionError::new(
                "activation invocation count exceeded its supported bound",
            ));
        }
        if active.disposition == ActivationTelemetryDispositionV1::Provisional {
            state
                .provisional_invocations
                .push(ProvisionalInvocation { invocation });
            return Ok(TensorWriteback::Commit);
        }
        state.elements_copied = state
            .elements_copied
            .checked_add(u64::from(elements))
            .ok_or_else(|| {
                TensorTransactionError::new("activation element accounting overflowed")
            })?;
        state.write_backs = state.write_backs.checked_add(1).ok_or_else(|| {
            TensorTransactionError::new("activation write-back accounting overflowed")
        })?;
        state.invocation_identities.push(identity);
        state.invocations.push(invocation);
        Ok(TensorWriteback::Commit)
    }
}

fn validate_profile_site(
    profile: &LlamaCppTensorProfile,
    site: &TextTensorSiteV1,
) -> Result<(), Error> {
    if profile.supports(site) {
        Ok(())
    } else {
        Err(Error::Incompatible(
            "tensor site is absent from the exact llama.cpp compatibility profile".to_owned(),
        ))
    }
}

fn native_tensor_name(site: &TextTensorSiteV1) -> String {
    match site {
        TextTensorSiteV1::LayerOutput { layer } => format!("l_out-{layer}"),
        TextTensorSiteV1::RouterLogits { layer, .. } => {
            format!("ffn_moe_logits-{layer}")
        }
        TextTensorSiteV1::RouterProbabilities { layer, .. } => {
            format!("ffn_moe_probs-{layer}")
        }
        TextTensorSiteV1::SelectedExperts { layer, .. } => {
            format!("ffn_moe_topk-{layer}")
        }
        TextTensorSiteV1::Named { name, .. } => name.clone(),
    }
}

fn capture_position_selected(
    plan: &ActivationCapturePlanV1,
    position: u64,
    decode: ActiveDecode,
    native_rows: usize,
) -> bool {
    match &plan.positions {
        ActivationCapturePositionsV1::LastPrefillToken => {
            decode.phase == ActivationPhaseV1::Prefill && decode.last_prefill && native_rows == 1
        }
        ActivationCapturePositionsV1::InclusiveRanges(_) => {
            plan.positions.contains_explicit(position)
        }
    }
}

fn retained_bytes(data: &ActivationCapturedDataV1) -> u64 {
    match data {
        ActivationCapturedDataV1::Digest { .. } => 0,
        ActivationCapturedDataV1::Statistics { .. } => 4 * 4,
        ActivationCapturedDataV1::F32Snapshot { values, .. } => u64::try_from(values.len())
            .unwrap_or(u64::MAX)
            .saturating_mul(4),
        ActivationCapturedDataV1::I32Snapshot { values, .. } => u64::try_from(values.len())
            .unwrap_or(u64::MAX)
            .saturating_mul(4),
    }
}

fn lock_state(
    state: &Arc<Mutex<ActivationState>>,
) -> Result<MutexGuard<'_, ActivationState>, Error> {
    state
        .lock()
        .map_err(|_| Error::Poisoned("activation callback state was poisoned".to_owned()))
}

fn apply_operator(
    values: &mut [f32],
    vector: &[f32],
    operator: ActivationOperatorV1,
    scale: f32,
) -> Result<(), TensorTransactionError> {
    if values.len() != vector.len() {
        return Err(TensorTransactionError::new(
            "activation vector width differs from tensor row",
        ));
    }
    match operator {
        ActivationOperatorV1::ScaledAdd => {
            for (value, direction) in values.iter_mut().zip(vector) {
                *value = finite_f32(f64::from(*value) + f64::from(scale) * f64::from(*direction))?;
            }
        }
        ActivationOperatorV1::ScaledProjectionRemoval => {
            let dot =
                values
                    .iter()
                    .zip(vector)
                    .try_fold(0.0_f64, |total, (value, direction)| {
                        let next = total + f64::from(*value) * f64::from(*direction);
                        next.is_finite().then_some(next).ok_or_else(|| {
                            TensorTransactionError::new(
                                "activation projection dot product overflowed",
                            )
                        })
                    })?;
            let factor = f64::from(scale) * dot;
            if !factor.is_finite() {
                return Err(TensorTransactionError::new(
                    "activation projection factor overflowed",
                ));
            }
            for (value, direction) in values.iter_mut().zip(vector) {
                *value = finite_f32(f64::from(*value) - factor * f64::from(*direction))?;
            }
        }
    }
    Ok(())
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "transaction results intentionally commit to the native f32 tensor contract"
)]
fn finite_f32(value: f64) -> Result<f32, TensorTransactionError> {
    let value = value as f32;
    value
        .is_finite()
        .then_some(value)
        .ok_or_else(|| TensorTransactionError::new("activation operator produced non-finite f32"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use logit_loom::{
        ActivationCaptureRetentionV1, TextModelTopologyV1, TextSpeculativeMechanismV1,
    };

    #[test]
    fn scaled_add_is_ordered_and_finite() {
        let mut values = [1.0_f32, 2.0];
        apply_operator(
            &mut values,
            &[2.0, -1.0],
            ActivationOperatorV1::ScaledAdd,
            0.5,
        )
        .unwrap();
        assert_eq!(
            values.map(f32::to_bits),
            [2.0_f32.to_bits(), 1.5_f32.to_bits()]
        );
    }

    #[test]
    fn projection_removal_uses_the_declared_scale() {
        let mut values = [2.0_f32, 3.0];
        apply_operator(
            &mut values,
            &[1.0, 0.0],
            ActivationOperatorV1::ScaledProjectionRemoval,
            0.5,
        )
        .unwrap();
        assert_eq!(
            values.map(f32::to_bits),
            [1.0_f32.to_bits(), 3.0_f32.to_bits()]
        );
    }

    #[test]
    fn operator_failure_does_not_accept_nonfinite_output() {
        let mut values = [f32::MAX];
        assert!(
            apply_operator(
                &mut values,
                &[f32::MAX],
                ActivationOperatorV1::ScaledAdd,
                f32::MAX,
            )
            .is_err()
        );
    }

    #[test]
    fn provisional_capture_rows_resolve_against_the_final_causal_prefix() {
        let topology = TextModelTopologyV1 {
            model: Digest::of_bytes("test-model", b"one"),
            backend: Digest::of_bytes("test-backend", b"one"),
            architecture_implementation: Digest::of_bytes("test-architecture", b"one"),
            layers: 2,
            embedding_width: 2,
            experts: None,
            experts_used: None,
            nextn_layers: 1,
            supported_speculation: vec![
                TextSpeculativeMechanismV1::Mtp,
                TextSpeculativeMechanismV1::Eagle3,
            ],
        };
        let site = TextTensorSiteV1::LayerOutput { layer: 0 };
        let plan = ActivationCapturePlanV1 {
            topology: topology.digest().unwrap(),
            sites: vec![site.clone()],
            positions: ActivationCapturePositionsV1::InclusiveRanges(vec![
                logit_loom::CausalPositionRangeV1 { start: 4, end: 5 },
            ]),
            retention: ActivationCaptureRetentionV1::Digest,
            maximum_elements: 4,
            maximum_retained_bytes: 0,
        };
        let plan_identity = plan.digest_for(&topology).unwrap();
        let program = ActivationProgramV1 {
            topology: topology.digest().unwrap(),
            operations: Vec::new(),
            observations: vec![plan_identity.clone()],
        };
        let program_identity = program.digest_for(&topology).unwrap();
        let captures = Arc::new(vec![(plan_identity.clone(), plan.clone())]);
        let pending = [4_u64, 5]
            .into_iter()
            .map(|position| {
                let record = ActivationCaptureRecordV1 {
                    plan: plan_identity.clone(),
                    site: site.clone(),
                    position,
                    disposition: ActivationTelemetryDispositionV1::Provisional,
                    retained: ActivationCapturedDataV1::from_f32(
                        &[1.0, 2.0],
                        ActivationCaptureRetentionV1::Digest,
                    )
                    .unwrap(),
                };
                let identity = record.digest_for(&plan, &topology).unwrap();
                ProvisionalCapture { record, identity }
            })
            .collect::<Vec<_>>();
        let mut provisional_aggregates = BTreeMap::new();
        provisional_aggregates.insert(
            plan_identity,
            CaptureAggregate {
                records: pending
                    .iter()
                    .map(|capture| capture.identity.clone())
                    .collect(),
                elements: 4,
                retained_bytes: 0,
            },
        );
        let mut controller = ActivationController {
            topology,
            program: Arc::new(program),
            program_identity,
            captures,
            site_bindings: Arc::new(BTreeMap::new()),
            state: Arc::new(Mutex::new(ActivationState::default())),
            captured_records: Vec::new(),
            capture_aggregates: BTreeMap::new(),
            provisional_captures: pending,
            provisional_capture_aggregates: provisional_aggregates,
        };

        let resolutions = controller.resolve_provisional(5).unwrap();
        assert_eq!(resolutions.len(), 2);
        let output = controller.take_capture_output().unwrap();
        assert_eq!(
            output
                .records
                .iter()
                .map(|record| record.disposition)
                .collect::<Vec<_>>(),
            vec![
                ActivationTelemetryDispositionV1::Admitted,
                ActivationTelemetryDispositionV1::Rejected,
            ]
        );
        assert_eq!(output.receipts[0].records.len(), 2);
    }
}

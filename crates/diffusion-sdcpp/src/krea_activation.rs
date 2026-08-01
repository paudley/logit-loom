// SPDX-License-Identifier: MIT OR Apache-2.0

//! Safe lowering and evidence collection for the native Krea activation ABI.

use std::{collections::HashMap, ffi::c_void, str::FromStr};

use logit_loom_diffusion::{
    ActivationStatisticsV1, Digest, KreaActivationApplicationV1, KreaActivationBoundaryKindV1,
    KreaActivationBoundaryV1, KreaActivationCaptureReceiptV1, KreaActivationCaptureRetentionV1,
    KreaActivationCleanupDispositionV1, KreaActivationElementTypeV1, KreaActivationInputKindV1,
    KreaActivationInputMeasurementV1, KreaActivationInputSourceV1, KreaActivationLayoutV1,
    KreaActivationMeasurementsV1, KreaActivationOperatorV1, KreaActivationPlanV1,
    KreaActivationReceiptV1, KreaActivationSiteKindV1, KreaActivationSiteV1,
    KreaActivationTerminalV1, KreaActivationTopologyV1, KreaCfgBranchV1, KreaTokenDomainKindV1,
    KreaTokenDomainV1, KreaTokenSelectionV1, KreaVectorRepresentationV1,
    MAX_KREA_ACTIVATION_ELEMENTS, MAX_KREA_ACTIVATION_SITES, StepSelector,
};
use logit_loom_executor::ExecutorState;

use crate::{
    ADAPTER_CONTRACT_VERSION, Error, KREA_ACTIVATION_ABI_VERSION, Profile, Result, Sdcpp,
    UPSTREAM_COMMIT,
    ffi::{
        self, KreaApplicationResultV6, KreaBoundarySelectionV6, KreaCaptureResultV6, KreaCaptureV6,
        KreaInputDescriptionV6, KreaInputHandleV6, KreaInputV6, KreaOperationV6, KreaSiteV6,
        KreaTokenRangeV6, KreaTokenSelectionV6, KreaTopologyV6,
    },
};

const KREA_PATCH: &[u8] =
    include_bytes!("../../../native/stable-diffusion.cpp/logit-loom-krea-activation-v6.patch");

/// One caller-verified finite input mapped from a sealed descriptor.
#[derive(Clone, Copy, Debug)]
pub struct KreaActivationInputBuffer<'a> {
    input: u16,
    bytes: &'a [u8],
}

impl<'a> KreaActivationInputBuffer<'a> {
    /// Binds bytes to one sealed-input index in a Krea activation plan.
    pub const fn new(input: u16, bytes: &'a [u8]) -> Self {
        Self { input, bytes }
    }

    /// Returns the declared input index.
    pub const fn input(self) -> u16 {
        self.input
    }

    /// Returns the borrowed exact bytes.
    pub const fn bytes(self) -> &'a [u8] {
        self.bytes
    }
}

/// Exact deterministic and placement evidence for one native activation job.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KreaActivationExecutionV1 {
    /// Deterministic topology, capture, application, terminal, and cleanup evidence.
    pub receipt: KreaActivationReceiptV1,
    /// Resident placement and transfer accounting.
    pub measurements: KreaActivationMeasurementsV1,
}

impl Sdcpp {
    /// Discovers the exact activation topology of the resident Krea model.
    ///
    /// The caller supplies the already verified model artifact identity. The
    /// returned backend and implementation identities bind the exact loaded
    /// companion and the complete v6 patch bytes.
    ///
    /// # Errors
    ///
    /// Returns an error unless the runtime is resident, Krea-backed, and
    /// reports one bounded internally consistent topology.
    pub fn krea_activation_topology(&self, model: Digest) -> Result<KreaActivationTopologyV1> {
        if self.state != ExecutorState::Resident {
            return Err(Error::Invalid(
                "Krea topology requires a resident runtime".to_owned(),
            ));
        }
        if self.profile != Profile::Krea2Turbo {
            return Err(Error::Incompatible(
                "Krea activation mechanics require the Krea profile".to_owned(),
            ));
        }
        let mut native = KreaTopologyV6::default();
        // SAFETY: The context is live for this immutable synchronous query.
        let status = unsafe {
            self.api
                .krea_topology_v6(self.context.as_ptr(), &mut native, &mut [])
        };
        require_status(status, "Krea topology count")?;
        if native.abi_version != KREA_ACTIVATION_ABI_VERSION
            || native.site_count == 0
            || native.site_count > MAX_KREA_ACTIVATION_SITES
        {
            return Err(Error::Incompatible(
                "native Krea topology count or ABI differs".to_owned(),
            ));
        }
        let mut sites = vec![KreaSiteV6::default(); native.site_count];
        // SAFETY: `sites` has exactly the capacity returned by the first call.
        let status = unsafe {
            self.api
                .krea_topology_v6(self.context.as_ptr(), &mut native, &mut sites)
        };
        require_status(status, "Krea topology")?;
        if native.abi_version != KREA_ACTIVATION_ABI_VERSION || native.site_count != sites.len() {
            return Err(Error::Incompatible(
                "native Krea topology changed between count and copy".to_owned(),
            ));
        }
        let backend = Digest::of_serializable(
            "sdcpp-krea-activation-backend-v1",
            &(
                KREA_ACTIVATION_ABI_VERSION,
                ADAPTER_CONTRACT_VERSION,
                UPSTREAM_COMMIT,
                self.execution_bindings()?,
                self.native_receipt(),
            ),
        )
        .map_err(logit_loom_diffusion::Error::from)?;
        let implementation = Digest::of_bytes("sdcpp-krea-activation-patch-v6", KREA_PATCH);
        let topology = KreaActivationTopologyV1 {
            model,
            backend,
            implementation,
            conditioner_layers: native.conditioner_layers,
            transformer_blocks: native.transformer_blocks,
            sites: sites
                .into_iter()
                .map(site_from_native)
                .collect::<Result<_>>()?,
        };
        topology
            .digest()
            .map_err(logit_loom_diffusion::Error::from)?;
        Ok(topology)
    }
}

fn site_from_native(site: KreaSiteV6) -> Result<KreaActivationSiteV1> {
    if site.width == 0 {
        return Err(Error::Incompatible(
            "native Krea site width is zero".to_owned(),
        ));
    }
    let width = site.width;
    let kind = match site.kind {
        ffi::KREA_CONDITIONER_LAYER_V6 => {
            KreaActivationSiteKindV1::ConditionerLayerOutput { layer: site.index }
        }
        ffi::KREA_POST_FUSION_V6 => KreaActivationSiteKindV1::ConditioningPostFusion,
        ffi::KREA_POST_PROJECTION_V6 => KreaActivationSiteKindV1::ConditioningPostProjection,
        ffi::KREA_TEXT_RESIDUAL_V6 => KreaActivationSiteKindV1::TextResidual { block: site.index },
        ffi::KREA_TRANSFORMER_RESIDUAL_V6 => {
            KreaActivationSiteKindV1::TransformerResidual { block: site.index }
        }
        other => {
            return Err(Error::Incompatible(format!(
                "native Krea site kind {other} is unknown"
            )));
        }
    };
    let boundaries = mask_values(
        site.boundary_mask,
        &[
            (
                ffi::KREA_PRE_DENOISER_V6,
                KreaActivationBoundaryKindV1::PreDenoiser,
            ),
            (
                ffi::KREA_TRANSITION_V6,
                KreaActivationBoundaryKindV1::Transition,
            ),
        ],
        "boundary",
    )?;
    let maximum_tokens = u32::try_from(MAX_KREA_ACTIVATION_ELEMENTS / u64::from(site.width.max(1)))
        .unwrap_or(u32::MAX)
        .max(1);
    let token_domains = mask_values(
        site.domain_mask,
        &[
            (ffi::KREA_TEXT_V6, KreaTokenDomainKindV1::Text),
            (ffi::KREA_IMAGE_V6, KreaTokenDomainKindV1::Image),
            (ffi::KREA_REFERENCE_V6, KreaTokenDomainKindV1::Reference),
        ],
        "token domain",
    )?
    .into_iter()
    .map(|kind| KreaTokenDomainV1 {
        kind,
        maximum_tokens,
    })
    .collect();
    let branches = mask_values(
        site.branch_mask,
        &[
            (ffi::KREA_CONDITIONAL_V6, KreaCfgBranchV1::Conditional),
            (ffi::KREA_UNCONDITIONAL_V6, KreaCfgBranchV1::Unconditional),
        ],
        "CFG branch",
    )?;
    let site = u16::try_from(site.site)
        .map_err(|_| Error::Incompatible("native Krea site exceeds u16".to_owned()))?;
    Ok(KreaActivationSiteV1 {
        site,
        kind,
        width,
        element_type: KreaActivationElementTypeV1::F32,
        layout: KreaActivationLayoutV1::FeatureTokenBatch,
        boundaries,
        token_domains,
        branches,
    })
}

fn mask_values<T: Copy>(mask: u32, values: &[(i32, T)], label: &str) -> Result<Vec<T>> {
    let known = values
        .iter()
        .fold(0_u32, |known, (value, _)| known | (1_u32 << (value - 1)));
    if mask == 0 || mask & !known != 0 {
        return Err(Error::Incompatible(format!(
            "native Krea {label} mask is empty or unknown"
        )));
    }
    Ok(values
        .iter()
        .filter(|(value, _)| mask & (1_u32 << (value - 1)) != 0)
        .map(|(_, value)| *value)
        .collect())
}

pub(crate) struct InstalledKreaActivation {
    pub(crate) topology: KreaActivationTopologyV1,
    pub(crate) plan: KreaActivationPlanV1,
    plan_digest: Digest,
    resident: HashMap<u16, ResidentInput>,
    device: String,
    jobs: u64,
}

struct ResidentInput {
    handle: KreaInputHandleV6,
    description: KreaInputDescriptionV6,
}

impl InstalledKreaActivation {
    #[allow(clippy::too_many_lines)]
    pub(crate) fn install(
        runtime: &mut Sdcpp,
        topology: KreaActivationTopologyV1,
        plan: KreaActivationPlanV1,
        buffers: &[KreaActivationInputBuffer<'_>],
    ) -> Result<Self> {
        if !cfg!(target_endian = "little") {
            return Err(Error::Incompatible(
                "Krea activation ABI requires little-endian f32 bytes".to_owned(),
            ));
        }
        if runtime.state != ExecutorState::Resident {
            return Err(Error::Invalid(
                "Krea inputs require a resident runtime".to_owned(),
            ));
        }
        let discovered = runtime.krea_activation_topology(topology.model.clone())?;
        if discovered != topology {
            return Err(Error::Incompatible(
                "supplied Krea activation topology differs from the resident runtime".to_owned(),
            ));
        }
        let plan_digest = plan
            .digest_for(&topology)
            .map_err(logit_loom_diffusion::Error::from)?;
        let mut supplied = HashMap::new();
        for buffer in buffers {
            if supplied.insert(buffer.input, buffer.bytes).is_some() {
                return Err(Error::Invalid(format!(
                    "Krea input {} was supplied twice",
                    buffer.input
                )));
            }
        }
        let expected = plan
            .inputs
            .iter()
            .filter(|input| matches!(input.source, KreaActivationInputSourceV1::Sealed { .. }))
            .count();
        if supplied.len() != expected {
            return Err(Error::Invalid(
                "sealed Krea input count differs from the plan".to_owned(),
            ));
        }

        let mut resident = HashMap::new();
        for input in &plan.inputs {
            let KreaActivationInputSourceV1::Sealed { .. } = input.source else {
                continue;
            };
            let bytes = supplied.remove(&input.input).ok_or_else(|| {
                Error::Invalid(format!("sealed Krea input {} is absent", input.input))
            })?;
            input
                .validate_bytes(bytes)
                .map_err(logit_loom_diffusion::Error::from)?;
            let values = decode_f32(bytes)?;
            let (site, rows, representation) = match input.kind {
                KreaActivationInputKindV1::Donor { site, tokens, .. } => {
                    (site, tokens, ffi::KREA_DONOR_F32_ROWS_V6)
                }
                KreaActivationInputKindV1::VectorBank {
                    site,
                    rank,
                    representation: KreaVectorRepresentationV1::F32Rows,
                    ..
                } => (site, u32::from(rank), ffi::KREA_VECTOR_F32_ROWS_V6),
                KreaActivationInputKindV1::VectorBank {
                    site,
                    rank,
                    representation: KreaVectorRepresentationV1::OrthonormalF32Rows,
                    ..
                } => (site, u32::from(rank), ffi::KREA_ORTHONORMAL_F32_ROWS_V6),
            };
            let native = KreaInputV6 {
                abi_version: KREA_ACTIVATION_ABI_VERSION,
                site: u32::from(site),
                rows,
                representation,
                values: values.as_ptr(),
                element_count: values.len(),
            };
            let mut description = KreaInputDescriptionV6::default();
            // SAFETY: The decoded finite values stay live for the synchronous import.
            let status = unsafe {
                runtime.api.krea_import_input_v6(
                    runtime.context.as_ptr(),
                    &native,
                    &mut description,
                )
            };
            if let Err(error) = require_status(status, "Krea resident input import") {
                release_imported(runtime, &resident)?;
                return Err(error);
            }
            if description.abi_version != KREA_ACTIVATION_ABI_VERSION
                || description.handle == KreaInputHandleV6::EMPTY
                || description.site != u32::from(site)
                || description.rows != rows
                || description.representation != representation
                || description.bytes != input.bytes().map_err(logit_loom_diffusion::Error::from)?
                || description.host_to_device_transfers != 1
                || description.host_to_device_bytes != description.bytes
            {
                // The new handle is included so an incompatible receipt cannot leak it.
                resident.insert(
                    input.input,
                    ResidentInput {
                        handle: description.handle,
                        description,
                    },
                );
                release_imported(runtime, &resident)?;
                runtime.state = ExecutorState::Poisoned;
                return Err(Error::Poisoned(
                    "native Krea input placement attestation differs".to_owned(),
                ));
            }
            resident.insert(
                input.input,
                ResidentInput {
                    handle: description.handle,
                    description,
                },
            );
        }
        let device = runtime.native_receipt.backend.clone();
        Ok(Self {
            topology,
            plan,
            plan_digest,
            resident,
            device,
            jobs: 0,
        })
    }

    pub(crate) fn verify_resident(&self, runtime: &mut Sdcpp) -> Result<()> {
        for input in self.resident.values() {
            let mut description = KreaInputDescriptionV6::default();
            // SAFETY: The handle belongs to this exclusively owned live context.
            let status = unsafe {
                runtime.api.krea_describe_input_v6(
                    runtime.context.as_ptr(),
                    input.handle,
                    &mut description,
                )
            };
            if let Err(error) = require_status(status, "Krea resident input verification") {
                runtime.state = ExecutorState::Poisoned;
                return Err(Error::Poisoned(format!(
                    "native Krea resident input became stale: {error}"
                )));
            }
            if !same_description(&input.description, &description) {
                runtime.state = ExecutorState::Poisoned;
                return Err(Error::Poisoned(
                    "native Krea resident input changed before reuse".to_owned(),
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn lower(&self) -> Result<LoweredKreaActivation> {
        LoweredKreaActivation::new(&self.plan, &self.resident)
    }

    pub(crate) fn callback_state(&self) -> KreaCallbackState {
        KreaCallbackState::new(&self.plan)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn finish_job(
        &mut self,
        runtime_epoch: u64,
        terminal: KreaActivationTerminalV1,
        captures: &[KreaCaptureResultV6],
        applications: &[KreaApplicationResultV6],
        peak_host_bytes: u64,
        peak_device_bytes: u64,
        callback: KreaCallbackState,
    ) -> Result<KreaActivationExecutionV1> {
        self.jobs = self.jobs.checked_add(1).ok_or_else(|| {
            Error::Poisoned("Krea resident-input job count overflowed".to_owned())
        })?;
        let receipt = callback.receipt(
            &self.plan,
            &self.topology,
            self.plan_digest.clone(),
            runtime_epoch,
            terminal,
            captures,
            applications,
        )?;
        let measurements = KreaActivationMeasurementsV1 {
            plan: self.plan_digest.clone(),
            runtime_epoch,
            peak_host_bytes,
            peak_device_bytes,
            inputs: self
                .plan
                .inputs
                .iter()
                .map(|input| {
                    let bytes = input.bytes().map_err(|error| {
                        Error::Poisoned(format!(
                            "validated Krea input shape changed after execution: {error}"
                        ))
                    })?;
                    let resident = self.resident.get(&input.input);
                    Ok(KreaActivationInputMeasurementV1 {
                        input: input.input,
                        device: self.device.clone(),
                        resident_bytes: bytes,
                        host_to_device_transfers: resident
                            .map_or(0, |input| input.description.host_to_device_transfers),
                        host_to_device_bytes: resident
                            .map_or(0, |input| input.description.host_to_device_bytes),
                        jobs: self.jobs,
                    })
                })
                .collect::<Result<_>>()?,
        };
        measurements
            .validate_for(&self.plan, &self.topology, &receipt)
            .map_err(|error| {
                Error::Poisoned(format!(
                    "native Krea resource evidence differs from the plan: {error}"
                ))
            })?;
        Ok(KreaActivationExecutionV1 {
            receipt,
            measurements,
        })
    }

    pub(crate) fn release(self, runtime: &mut Sdcpp) -> Result<()> {
        release_imported(runtime, &self.resident)
    }
}

fn release_imported(runtime: &mut Sdcpp, resident: &HashMap<u16, ResidentInput>) -> Result<()> {
    let mut failed = false;
    for input in resident.values() {
        // SAFETY: Each handle belongs to this exclusively owned context.
        let status = unsafe {
            runtime
                .api
                .krea_release_input_v6(runtime.context.as_ptr(), input.handle)
        };
        failed |= status != ffi::STATUS_OK;
    }
    // SAFETY: The context is exclusively owned. Clearing is idempotent and
    // advances the native generation so no released handle can be reused.
    let clear_status = unsafe { runtime.api.krea_clear_inputs_v6(runtime.context.as_ptr()) };
    failed |= clear_status != ffi::STATUS_OK;
    if failed {
        runtime.state = ExecutorState::Poisoned;
        return Err(Error::Poisoned(
            "native Krea resident-input release was uncertain".to_owned(),
        ));
    }
    Ok(())
}

fn same_description(left: &KreaInputDescriptionV6, right: &KreaInputDescriptionV6) -> bool {
    left.abi_version == right.abi_version
        && left.handle == right.handle
        && left.site == right.site
        && left.width == right.width
        && left.rows == right.rows
        && left.representation == right.representation
        && left.bytes == right.bytes
        && left.host_to_device_transfers == right.host_to_device_transfers
        && left.host_to_device_bytes == right.host_to_device_bytes
}

pub(crate) struct LoweredKreaActivation {
    _capture_ranges: Vec<Vec<KreaTokenRangeV6>>,
    _capture_steps: Vec<Vec<u32>>,
    _operation_ranges: Vec<Vec<KreaTokenRangeV6>>,
    _operation_steps: Vec<Vec<u32>>,
    pub(crate) captures: Vec<KreaCaptureV6>,
    pub(crate) operations: Vec<KreaOperationV6>,
}

impl LoweredKreaActivation {
    fn new(plan: &KreaActivationPlanV1, resident: &HashMap<u16, ResidentInput>) -> Result<Self> {
        let capture_ranges = plan
            .captures
            .iter()
            .map(|capture| ranges(&capture.tokens))
            .collect::<Vec<_>>();
        let capture_steps = plan
            .captures
            .iter()
            .map(|capture| steps(&capture.boundary))
            .collect::<Vec<_>>();
        let captures = plan
            .captures
            .iter()
            .zip(&capture_ranges)
            .zip(&capture_steps)
            .map(|((capture, ranges), steps)| {
                Ok(KreaCaptureV6 {
                    capture_index: u32::from(capture.capture),
                    site: u32::from(capture.site),
                    tokens: token_selection(&capture.tokens, ranges),
                    boundary: boundary_selection(&capture.boundary, steps),
                    branch: branch(capture.branch),
                    retention: retention(capture.retention),
                    maximum_elements: capture.maximum_elements,
                    maximum_host_bytes: capture.maximum_host_bytes,
                    maximum_device_bytes: capture.maximum_device_bytes,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let operation_ranges = plan
            .operations
            .iter()
            .map(|operation| ranges(&operation.tokens))
            .collect::<Vec<_>>();
        let operation_steps = plan
            .operations
            .iter()
            .map(|operation| steps(&operation.boundary))
            .collect::<Vec<_>>();
        let inputs = plan
            .inputs
            .iter()
            .map(|input| (input.input, input))
            .collect::<HashMap<_, _>>();
        let operations = plan
            .operations
            .iter()
            .zip(&operation_ranges)
            .zip(&operation_steps)
            .map(|((operation, ranges), steps)| {
                let input = inputs
                    .get(&operation_input(&operation.operator))
                    .ok_or_else(|| {
                        Error::Invalid(
                            "Krea operation input disappeared after validation".to_owned(),
                        )
                    })?;
                let (input_source, resident_input, capture_input) = match input.source {
                    KreaActivationInputSourceV1::Sealed { .. } => (
                        ffi::KREA_RESIDENT_INPUT_V6,
                        resident
                            .get(&input.input)
                            .ok_or_else(|| {
                                Error::Invalid("sealed Krea input was not imported".to_owned())
                            })?
                            .handle,
                        u32::MAX,
                    ),
                    KreaActivationInputSourceV1::Capture { capture } => (
                        ffi::KREA_CAPTURE_INPUT_V6,
                        KreaInputHandleV6::EMPTY,
                        u32::from(capture),
                    ),
                };
                let (operator, vector) = operator(&operation.operator);
                Ok(KreaOperationV6 {
                    operation_index: u32::from(operation.operation),
                    site: u32::from(operation.site),
                    tokens: token_selection(&operation.tokens, ranges),
                    boundary: boundary_selection(&operation.boundary, steps),
                    branch: branch(operation.branch),
                    operation: operator,
                    input_source,
                    resident_input,
                    capture_input,
                    vector,
                    strength: operation
                        .strength()
                        .map_err(logit_loom_diffusion::Error::from)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            _capture_ranges: capture_ranges,
            _capture_steps: capture_steps,
            _operation_ranges: operation_ranges,
            _operation_steps: operation_steps,
            captures,
            operations,
        })
    }
}

fn ranges(selection: &KreaTokenSelectionV1) -> Vec<KreaTokenRangeV6> {
    match selection {
        KreaTokenSelectionV1::All { .. } => Vec::new(),
        KreaTokenSelectionV1::Ranges { ranges, .. } => ranges
            .iter()
            .map(|range| KreaTokenRangeV6 {
                start: range.start,
                end: range.end,
            })
            .collect(),
    }
}

fn steps(boundary: &KreaActivationBoundaryV1) -> Vec<u32> {
    match boundary {
        KreaActivationBoundaryV1::PreDenoiser
        | KreaActivationBoundaryV1::Transitions {
            steps: StepSelector::All,
        } => Vec::new(),
        KreaActivationBoundaryV1::Transitions {
            steps: StepSelector::Exact { steps },
        } => steps.clone(),
    }
}

fn token_selection(
    selection: &KreaTokenSelectionV1,
    ranges: &[KreaTokenRangeV6],
) -> KreaTokenSelectionV6 {
    KreaTokenSelectionV6 {
        domain: domain(selection.domain()),
        selection: if matches!(selection, KreaTokenSelectionV1::All { .. }) {
            ffi::KREA_ALL_TOKENS_V6
        } else {
            ffi::KREA_TOKEN_RANGES_V6
        },
        ranges: pointer_or_null(ranges),
        range_count: ranges.len(),
    }
}

fn boundary_selection(
    boundary: &KreaActivationBoundaryV1,
    steps: &[u32],
) -> KreaBoundarySelectionV6 {
    KreaBoundarySelectionV6 {
        boundary: match boundary {
            KreaActivationBoundaryV1::PreDenoiser => ffi::KREA_PRE_DENOISER_V6,
            KreaActivationBoundaryV1::Transitions { .. } => ffi::KREA_TRANSITION_V6,
        },
        step_selection: if steps.is_empty() {
            ffi::STEP_ALL_V5
        } else {
            ffi::STEP_EXACT_V5
        },
        steps: pointer_or_null(steps),
        step_count: steps.len(),
    }
}

fn branch(branch: KreaCfgBranchV1) -> i32 {
    match branch {
        KreaCfgBranchV1::Conditional => ffi::KREA_CONDITIONAL_V6,
        KreaCfgBranchV1::Unconditional => ffi::KREA_UNCONDITIONAL_V6,
    }
}

fn domain(domain: KreaTokenDomainKindV1) -> i32 {
    match domain {
        KreaTokenDomainKindV1::Text => ffi::KREA_TEXT_V6,
        KreaTokenDomainKindV1::Image => ffi::KREA_IMAGE_V6,
        KreaTokenDomainKindV1::Reference => ffi::KREA_REFERENCE_V6,
    }
}

fn retention(retention: KreaActivationCaptureRetentionV1) -> i32 {
    match retention {
        KreaActivationCaptureRetentionV1::Digest => ffi::KREA_CAPTURE_DIGEST_V6,
        KreaActivationCaptureRetentionV1::Statistics => ffi::KREA_CAPTURE_STATISTICS_V6,
        KreaActivationCaptureRetentionV1::DeviceSnapshot => ffi::KREA_CAPTURE_DEVICE_SNAPSHOT_V6,
    }
}

fn operation_input(operator: &KreaActivationOperatorV1) -> u16 {
    match operator {
        KreaActivationOperatorV1::DonorTransplant { input }
        | KreaActivationOperatorV1::ScaledVectorAdd { input, .. }
        | KreaActivationOperatorV1::ScaledVectorSubtract { input, .. }
        | KreaActivationOperatorV1::OrthogonalProjectionRemoval { input }
        | KreaActivationOperatorV1::OneSidedProjectionRemoval { input } => *input,
    }
}

fn operator(operator: &KreaActivationOperatorV1) -> (i32, u32) {
    match operator {
        KreaActivationOperatorV1::DonorTransplant { .. } => (ffi::KREA_DONOR_TRANSPLANT_V6, 0),
        KreaActivationOperatorV1::ScaledVectorAdd { vector, .. } => {
            (ffi::KREA_SCALED_VECTOR_ADD_V6, u32::from(*vector))
        }
        KreaActivationOperatorV1::ScaledVectorSubtract { vector, .. } => {
            (ffi::KREA_SCALED_VECTOR_SUBTRACT_V6, u32::from(*vector))
        }
        KreaActivationOperatorV1::OrthogonalProjectionRemoval { .. } => {
            (ffi::KREA_PROJECTION_REMOVAL_V6, 0)
        }
        KreaActivationOperatorV1::OneSidedProjectionRemoval { .. } => {
            (ffi::KREA_ONE_SIDED_REMOVAL_V6, 0)
        }
    }
}

pub(crate) struct KreaCallbackState {
    captures: HashMap<u16, CaptureAccumulator>,
    applications: HashMap<u16, ApplicationAccumulator>,
    error: Option<String>,
}

struct CaptureAccumulator {
    retention: KreaActivationCaptureRetentionV1,
    reached: u64,
    elements: u64,
    content: StreamDigest,
    snapshot: StreamDigest,
    statistics: StatisticsAccumulator,
}

struct ApplicationAccumulator {
    reached_before: u64,
    reached_after: u64,
    input: StreamDigest,
    output: StreamDigest,
}

impl KreaCallbackState {
    fn new(plan: &KreaActivationPlanV1) -> Self {
        Self {
            captures: plan
                .captures
                .iter()
                .map(|capture| {
                    (
                        capture.capture,
                        CaptureAccumulator {
                            retention: capture.retention,
                            reached: 0,
                            elements: 0,
                            content: StreamDigest::new("krea-activation-capture-f32-stream-v1"),
                            snapshot: StreamDigest::new(
                                "krea-activation-device-snapshot-stream-v1",
                            ),
                            statistics: StatisticsAccumulator::default(),
                        },
                    )
                })
                .collect(),
            applications: plan
                .operations
                .iter()
                .map(|operation| {
                    (
                        operation.operation,
                        ApplicationAccumulator {
                            reached_before: 0,
                            reached_after: 0,
                            input: StreamDigest::new("krea-activation-application-f32-stream-v1"),
                            output: StreamDigest::new("krea-activation-application-f32-stream-v1"),
                        },
                    )
                })
                .collect(),
            error: None,
        }
    }

    fn event(&mut self, kind: i32, index: u32, reach: u64, values: &[f32]) -> Result<()> {
        if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
            return Err(Error::Callback(
                "native Krea event was empty or non-finite".to_owned(),
            ));
        }
        let index = u16::try_from(index)
            .map_err(|_| Error::Callback("native Krea event index exceeds u16".to_owned()))?;
        match kind {
            ffi::KREA_CAPTURE_EVENT_V6 => {
                let capture = self.captures.get_mut(&index).ok_or_else(|| {
                    Error::Callback("native Krea capture index is undeclared".to_owned())
                })?;
                if reach != capture.reached + 1 {
                    return Err(Error::Callback(
                        "native Krea capture reach is non-canonical".to_owned(),
                    ));
                }
                capture.reached = reach;
                capture.elements = capture
                    .elements
                    .checked_add(u64::try_from(values.len()).map_err(|_| {
                        Error::Callback("native Krea capture length exceeds u64".to_owned())
                    })?)
                    .ok_or_else(|| {
                        Error::Callback("native Krea capture element count overflowed".to_owned())
                    })?;
                capture.content.update(index, reach, values);
                if capture.retention == KreaActivationCaptureRetentionV1::DeviceSnapshot {
                    capture.snapshot.update(index, reach, values);
                }
                if capture.retention == KreaActivationCaptureRetentionV1::Statistics {
                    capture.statistics.update(values)?;
                }
            }
            ffi::KREA_APPLICATION_BEFORE_EVENT_V6 => {
                let application = self.applications.get_mut(&index).ok_or_else(|| {
                    Error::Callback("native Krea operation index is undeclared".to_owned())
                })?;
                if reach != application.reached_before + 1 {
                    return Err(Error::Callback(
                        "native Krea input reach is non-canonical".to_owned(),
                    ));
                }
                application.reached_before = reach;
                application.input.update(index, reach, values);
            }
            ffi::KREA_APPLICATION_AFTER_EVENT_V6 => {
                let application = self.applications.get_mut(&index).ok_or_else(|| {
                    Error::Callback("native Krea operation index is undeclared".to_owned())
                })?;
                if reach != application.reached_after + 1 || reach > application.reached_before {
                    return Err(Error::Callback(
                        "native Krea output reach is non-canonical".to_owned(),
                    ));
                }
                application.reached_after = reach;
                application.output.update(index, reach, values);
            }
            other => {
                return Err(Error::Callback(format!(
                    "native Krea event kind {other} is unknown"
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn take_error(&mut self) -> Option<String> {
        self.error.take()
    }

    #[allow(clippy::too_many_arguments)]
    fn receipt(
        self,
        plan: &KreaActivationPlanV1,
        topology: &KreaActivationTopologyV1,
        plan_digest: Digest,
        runtime_epoch: u64,
        terminal: KreaActivationTerminalV1,
        native_captures: &[KreaCaptureResultV6],
        native_applications: &[KreaApplicationResultV6],
    ) -> Result<KreaActivationReceiptV1> {
        if let Some(error) = self.error {
            return Err(Error::Callback(error));
        }
        if native_captures.len() != plan.captures.len()
            || native_applications.len() != plan.operations.len()
        {
            return Err(Error::Poisoned(
                "native Krea result coverage differs".to_owned(),
            ));
        }
        let captures = plan
            .captures
            .iter()
            .zip(native_captures)
            .map(|(expected, native)| {
                let accumulator = self.captures.get(&expected.capture).ok_or_else(|| {
                    Error::Poisoned("Krea capture accumulator disappeared".to_owned())
                })?;
                if native.capture_index != u32::from(expected.capture)
                    || native.reached != accumulator.reached
                    || native.elements != accumulator.elements
                {
                    return Err(Error::Poisoned(
                        "native Krea capture counts differ from callbacks".to_owned(),
                    ));
                }
                let reached = native.reached > 0;
                Ok(KreaActivationCaptureReceiptV1 {
                    capture: expected.capture,
                    reached: native.reached,
                    elements: native.elements,
                    content: reached.then(|| accumulator.content.digest()),
                    statistics: if reached
                        && expected.retention == KreaActivationCaptureRetentionV1::Statistics
                    {
                        Some(accumulator.statistics.finish()?)
                    } else {
                        None
                    },
                    snapshot: if reached
                        && expected.retention == KreaActivationCaptureRetentionV1::DeviceSnapshot
                    {
                        Some(accumulator.snapshot.digest())
                    } else {
                        None
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let applications = plan
            .operations
            .iter()
            .zip(native_applications)
            .map(|(expected, native)| {
                let accumulator = self.applications.get(&expected.operation).ok_or_else(|| {
                    Error::Poisoned("Krea operation accumulator disappeared".to_owned())
                })?;
                if native.operation_index != u32::from(expected.operation)
                    || native.reached != accumulator.reached_before
                    || native.reached != accumulator.reached_after
                {
                    return Err(Error::Poisoned(
                        "native Krea application counts differ from callbacks".to_owned(),
                    ));
                }
                let reached = native.reached > 0;
                Ok(KreaActivationApplicationV1 {
                    operation: expected.operation,
                    reached: native.reached,
                    applied: native.applied,
                    unchanged: native.unchanged,
                    input: reached.then(|| accumulator.input.digest()),
                    output: reached.then(|| accumulator.output.digest()),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let receipt = KreaActivationReceiptV1 {
            plan: plan_digest,
            topology: topology.digest().map_err(|error| {
                Error::Poisoned(format!(
                    "validated Krea topology changed after execution: {error}"
                ))
            })?,
            backend: topology.backend.clone(),
            runtime_epoch,
            terminal,
            captures,
            applications,
            cleanup: KreaActivationCleanupDispositionV1::Confirmed,
        };
        receipt.digest_for(plan, topology).map_err(|error| {
            Error::Poisoned(format!(
                "native Krea execution evidence differs from the plan: {error}"
            ))
        })?;
        Ok(receipt)
    }
}

/// Receives one native event without unwinding across the C ABI.
pub(crate) unsafe extern "C" fn krea_event_callback(
    kind: i32,
    index: u32,
    reach: u64,
    values: *const f32,
    elements: usize,
    data: *mut c_void,
) -> i32 {
    if data.is_null() || values.is_null() || elements == 0 {
        return ffi::CALLBACK_ERROR;
    }
    if u64::try_from(elements).unwrap_or(u64::MAX) > MAX_KREA_ACTIVATION_ELEMENTS {
        // SAFETY: `data` was validated non-null and names the callback state.
        let state = unsafe { &mut *data.cast::<KreaCallbackState>() };
        state.error = Some("native Krea event exceeded the public element bound".to_owned());
        return ffi::CALLBACK_ERROR;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: Native v6 promises one synchronous finite array for this call.
        let values = unsafe { std::slice::from_raw_parts(values, elements) };
        // SAFETY: The caller passes the sole live callback state.
        let state = unsafe { &mut *data.cast::<KreaCallbackState>() };
        state.event(kind, index, reach, values)
    }));
    match result {
        Ok(Ok(())) => ffi::CALLBACK_CONTINUE,
        Ok(Err(error)) => {
            // SAFETY: Validated non-null above.
            let state = unsafe { &mut *data.cast::<KreaCallbackState>() };
            state.error = Some(error.to_string());
            ffi::CALLBACK_ERROR
        }
        Err(_) => {
            // SAFETY: Validated non-null above.
            let state = unsafe { &mut *data.cast::<KreaCallbackState>() };
            state.error = Some("Krea activation callback panicked".to_owned());
            ffi::CALLBACK_ERROR
        }
    }
}

struct StreamDigest {
    hasher: blake3::Hasher,
}

impl StreamDigest {
    fn new(domain: &str) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"logit-loom-krea-stream\0");
        hasher.update(
            &u64::try_from(domain.len())
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        hasher.update(domain.as_bytes());
        Self { hasher }
    }

    fn update(&mut self, index: u16, reach: u64, values: &[f32]) {
        self.hasher.update(&index.to_le_bytes());
        self.hasher.update(&reach.to_le_bytes());
        self.hasher.update(
            &u64::try_from(values.len())
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        for value in values {
            self.hasher.update(&value.to_bits().to_le_bytes());
        }
    }

    fn digest(&self) -> Digest {
        Digest::from_str(self.hasher.clone().finalize().to_hex().as_str())
            .expect("BLAKE3 emits one canonical lowercase digest")
    }
}

#[derive(Default)]
struct StatisticsAccumulator {
    elements: u64,
    minimum: f32,
    maximum: f32,
    sum: f64,
    squared: f64,
}

impl StatisticsAccumulator {
    fn update(&mut self, values: &[f32]) -> Result<()> {
        for value in values {
            if !value.is_finite() {
                return Err(Error::Callback(
                    "Krea statistics received a non-finite value".to_owned(),
                ));
            }
            if self.elements == 0 {
                self.minimum = *value;
                self.maximum = *value;
            } else {
                self.minimum = self.minimum.min(*value);
                self.maximum = self.maximum.max(*value);
            }
            let wide = f64::from(*value);
            self.sum += wide;
            self.squared += wide * wide;
            self.elements = self.elements.checked_add(1).ok_or_else(|| {
                Error::Callback("Krea statistics element count overflowed".to_owned())
            })?;
        }
        Ok(())
    }

    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn finish(&self) -> Result<ActivationStatisticsV1> {
        if self.elements == 0 {
            return Err(Error::Poisoned(
                "reached Krea statistics capture is empty".to_owned(),
            ));
        }
        let mean = (self.sum / self.elements as f64) as f32;
        let norm = self.squared.sqrt() as f32;
        if !mean.is_finite() || !norm.is_finite() {
            return Err(Error::Poisoned(
                "Krea activation statistics overflowed".to_owned(),
            ));
        }
        Ok(ActivationStatisticsV1 {
            minimum_bits: self.minimum.to_bits(),
            maximum_bits: self.maximum.to_bits(),
            mean_bits: mean.to_bits(),
            l2_norm_bits: norm.to_bits(),
        })
    }
}

fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return Err(Error::Invalid(
            "Krea input bytes are not complete f32 elements".to_owned(),
        ));
    }
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let value =
                f32::from_bits(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            value
                .is_finite()
                .then_some(value)
                .ok_or_else(|| Error::Invalid("Krea input contains non-finite f32".to_owned()))
        })
        .collect()
}

fn pointer_or_null<T>(values: &[T]) -> *const T {
    if values.is_empty() {
        std::ptr::null()
    } else {
        values.as_ptr()
    }
}

fn require_status(status: i32, operation: &str) -> Result<()> {
    match status {
        ffi::STATUS_OK => Ok(()),
        ffi::STATUS_INVALID_ARGUMENT => Err(Error::Invalid(format!(
            "{operation} arguments or handle differ"
        ))),
        ffi::STATUS_UNSUPPORTED => Err(Error::Incompatible(format!(
            "{operation} is unavailable in the loaded native profile"
        ))),
        ffi::STATUS_CALLBACK_ERROR => Err(Error::Callback(format!("{operation} callback failed"))),
        other => Err(Error::Native(format!(
            "{operation} failed with native status {other}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use logit_loom_diffusion::{
        KreaActivationInputV1, KreaTokenRangeV1, krea_activation_input_content_v1,
    };

    fn identity(label: &str) -> Digest {
        Digest::of_bytes("test-krea-activation-identity", label.as_bytes())
    }

    fn topology() -> KreaActivationTopologyV1 {
        KreaActivationTopologyV1 {
            model: identity("model"),
            backend: identity("backend"),
            implementation: identity("implementation"),
            conditioner_layers: 1,
            transformer_blocks: 1,
            sites: vec![KreaActivationSiteV1 {
                site: 0,
                kind: KreaActivationSiteKindV1::TransformerResidual { block: 0 },
                width: 2,
                element_type: KreaActivationElementTypeV1::F32,
                layout: KreaActivationLayoutV1::FeatureTokenBatch,
                boundaries: vec![KreaActivationBoundaryKindV1::Transition],
                token_domains: vec![KreaTokenDomainV1 {
                    kind: KreaTokenDomainKindV1::Text,
                    maximum_tokens: 1,
                }],
                branches: vec![KreaCfgBranchV1::Conditional],
            }],
        }
    }

    fn installed() -> InstalledKreaActivation {
        let topology = topology();
        let bytes = [1_f32.to_le_bytes(), 0_f32.to_le_bytes()].concat();
        let input = KreaActivationInputV1 {
            input: 0,
            topology: topology.digest().unwrap(),
            kind: KreaActivationInputKindV1::VectorBank {
                site: 0,
                width: 2,
                rank: 1,
                representation: KreaVectorRepresentationV1::F32Rows,
            },
            source: KreaActivationInputSourceV1::Sealed {
                content: krea_activation_input_content_v1(&bytes),
                bytes: 8,
            },
        };
        let plan = KreaActivationPlanV1 {
            topology: topology.digest().unwrap(),
            step_count: 1,
            inputs: vec![input],
            captures: Vec::new(),
            operations: vec![logit_loom_diffusion::KreaActivationOperationV1 {
                operation: 0,
                site: 0,
                tokens: KreaTokenSelectionV1::Ranges {
                    domain: KreaTokenDomainKindV1::Text,
                    ranges: vec![KreaTokenRangeV1 { start: 0, end: 1 }],
                },
                boundary: KreaActivationBoundaryV1::Transitions {
                    steps: StepSelector::Exact { steps: vec![0] },
                },
                branch: KreaCfgBranchV1::Conditional,
                operator: KreaActivationOperatorV1::ScaledVectorAdd {
                    input: 0,
                    vector: 0,
                },
                strength_bits: 0_f32.to_bits(),
            }],
            maximum_host_bytes: 16,
            maximum_device_bytes: 8,
            maximum_applications: 1,
        };
        let plan_digest = plan.digest_for(&topology).unwrap();
        InstalledKreaActivation {
            topology,
            plan,
            plan_digest,
            resident: HashMap::from([(
                0,
                ResidentInput {
                    handle: KreaInputHandleV6 {
                        generation: 1,
                        slot: 0,
                        reserved: 0,
                    },
                    description: KreaInputDescriptionV6 {
                        abi_version: KREA_ACTIVATION_ABI_VERSION,
                        handle: KreaInputHandleV6 {
                            generation: 1,
                            slot: 0,
                            reserved: 0,
                        },
                        site: 0,
                        width: 2,
                        rows: 1,
                        representation: ffi::KREA_VECTOR_F32_ROWS_V6,
                        bytes: 8,
                        host_to_device_transfers: 1,
                        host_to_device_bytes: 8,
                    },
                },
            )]),
            device: "vulkan:test".to_owned(),
            jobs: 0,
        }
    }

    #[test]
    fn callback_receipt_binds_native_counts_and_resource_peaks() {
        let mut installed = installed();
        let mut callback = installed.callback_state();
        let values = [3.0_f32, 4.0];
        let callback_pointer = (&raw mut callback).cast::<c_void>();
        // SAFETY: Both the callback state and finite value slice are live for
        // each synchronous invocation.
        unsafe {
            assert_eq!(
                krea_event_callback(
                    ffi::KREA_APPLICATION_BEFORE_EVENT_V6,
                    0,
                    1,
                    values.as_ptr(),
                    values.len(),
                    callback_pointer,
                ),
                ffi::CALLBACK_CONTINUE
            );
            assert_eq!(
                krea_event_callback(
                    ffi::KREA_APPLICATION_AFTER_EVENT_V6,
                    0,
                    1,
                    values.as_ptr(),
                    values.len(),
                    callback_pointer,
                ),
                ffi::CALLBACK_CONTINUE
            );
        }
        let execution = installed
            .finish_job(
                7,
                KreaActivationTerminalV1::Completed,
                &[],
                &[KreaApplicationResultV6 {
                    operation_index: 0,
                    reached: 1,
                    applied: 1,
                    unchanged: 1,
                }],
                16,
                8,
                callback,
            )
            .unwrap();
        assert_eq!(
            execution.receipt.applications[0].input,
            execution.receipt.applications[0].output
        );
        assert_eq!(execution.measurements.peak_host_bytes, 16);
        assert_eq!(execution.measurements.peak_device_bytes, 8);
        assert_eq!(execution.measurements.inputs[0].jobs, 1);
    }

    #[test]
    fn native_topology_rejects_zero_width_before_publication() {
        let mut native = KreaSiteV6 {
            site: 0,
            kind: ffi::KREA_TRANSFORMER_RESIDUAL_V6,
            index: 0,
            width: 0,
            boundary_mask: 1 << (ffi::KREA_TRANSITION_V6 - 1),
            domain_mask: 1 << (ffi::KREA_TEXT_V6 - 1),
            branch_mask: 1 << (ffi::KREA_CONDITIONAL_V6 - 1),
        };
        assert!(site_from_native(native).is_err());
        native.width = 2;
        assert_eq!(site_from_native(native).unwrap().width, 2);
    }

    #[test]
    fn callback_rejects_excessive_length_before_forming_a_slice() {
        let installed = installed();
        let mut callback = installed.callback_state();
        let callback_pointer = (&raw mut callback).cast::<c_void>();
        let dangling = std::ptr::NonNull::<f32>::dangling().as_ptr();
        let elements = usize::try_from(MAX_KREA_ACTIVATION_ELEMENTS).unwrap() + 1;
        // SAFETY: The excessive length is rejected before the deliberately
        // dangling value pointer can be used to form or read a slice.
        let status = unsafe {
            krea_event_callback(
                ffi::KREA_APPLICATION_BEFORE_EVENT_V6,
                0,
                1,
                dangling,
                elements,
                callback_pointer,
            )
        };
        assert_eq!(status, ffi::CALLBACK_ERROR);
        assert!(callback.take_error().is_some());
    }
}

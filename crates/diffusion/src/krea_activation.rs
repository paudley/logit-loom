// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded, topology-bound Krea activation mechanics.

use std::collections::{BTreeSet, HashMap};

use logit_loom_core::{ActivationStatisticsV1, CoreError, Digest};
use serde::{Deserialize, Serialize};

use crate::StepSelector;

/// Maximum Krea activation sites reported by one loaded topology.
pub const MAX_KREA_ACTIVATION_SITES: usize = 256;
/// Maximum token-domain ranges in one selection.
pub const MAX_KREA_TOKEN_RANGES: usize = 256;
/// Maximum captures in one resident request.
pub const MAX_KREA_ACTIVATION_CAPTURES: usize = 128;
/// Maximum activation inputs retained by one resident session.
pub const MAX_KREA_ACTIVATION_INPUTS: usize = 128;
/// Maximum ordered Krea activation operations in one request.
pub const MAX_KREA_ACTIVATION_OPERATIONS: usize = 256;
/// Maximum vector rank accepted by one activation input.
pub const MAX_KREA_VECTOR_RANK: u16 = 256;
/// Maximum finite `f32` elements in one Krea activation value.
pub const MAX_KREA_ACTIVATION_ELEMENTS: u64 = 16 * 1024 * 1024;
/// Maximum bytes retained for one activation input or capture.
pub const MAX_KREA_ACTIVATION_VALUE_BYTES: u64 = MAX_KREA_ACTIVATION_ELEMENTS * 4;
/// Maximum aggregate applications retained in one receipt.
pub const MAX_KREA_ACTIVATION_APPLICATIONS: u64 = 1_000_000;

/// Scalar representation at a Krea activation site.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KreaActivationElementTypeV1 {
    /// Finite IEEE-754 single-precision values.
    F32,
}

/// Canonical logical layout of a Krea activation tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KreaActivationLayoutV1 {
    /// Feature-major rows addressed by token and batch/branch.
    FeatureTokenBatch,
}

/// Evaluation families available at one runtime-derived site.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KreaActivationBoundaryKindV1 {
    /// One logical boundary after conditioning and before denoising.
    PreDenoiser,
    /// The site is evaluated inside selected denoising transitions.
    Transition,
}

/// Runtime-derived Krea site family.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum KreaActivationSiteKindV1 {
    /// Output of one selected conditioner transformer layer.
    ConditionerLayerOutput {
        /// Zero-based conditioner layer.
        layer: u32,
    },
    /// Conditioner output after visual/text fusion.
    ConditioningPostFusion,
    /// Conditioner output after projection into the denoiser width.
    ConditioningPostProjection,
    /// Text-token rows after one Krea transformer block.
    TextResidual {
        /// Zero-based Krea transformer block.
        block: u32,
    },
    /// Complete token rows after one Krea transformer block.
    TransformerResidual {
        /// Zero-based Krea transformer block.
        block: u32,
    },
}

/// Logical token family addressable at a site.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KreaTokenDomainKindV1 {
    /// Prompt-conditioning tokens.
    Text,
    /// Primary image latent tokens.
    Image,
    /// Reference-image latent tokens.
    Reference,
}

/// Runtime bound for one token family at a site.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KreaTokenDomainV1 {
    /// Logical token family.
    pub kind: KreaTokenDomainKindV1,
    /// Inclusive maximum token count accepted by the loaded topology.
    pub maximum_tokens: u32,
}

/// Classifier-free-guidance branch selected by a mechanic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KreaCfgBranchV1 {
    /// Conditional/prompt-bearing branch.
    Conditional,
    /// Unconditional/negative-conditioning branch.
    Unconditional,
}

/// One runtime-derived, topology-bound activation site.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KreaActivationSiteV1 {
    /// Stable site identifier within this exact topology.
    pub site: u16,
    /// Mechanical boundary represented by the site.
    pub kind: KreaActivationSiteKindV1,
    /// Elements in every selected token row.
    pub width: u32,
    /// Scalar representation.
    pub element_type: KreaActivationElementTypeV1,
    /// Logical tensor layout.
    pub layout: KreaActivationLayoutV1,
    /// Canonically ordered evaluation families available at this site.
    pub boundaries: Vec<KreaActivationBoundaryKindV1>,
    /// Canonically ordered token families accepted at this site.
    pub token_domains: Vec<KreaTokenDomainV1>,
    /// Canonically ordered CFG branches accepted at this site.
    pub branches: Vec<KreaCfgBranchV1>,
}

/// Exact topology discovered from one loaded Krea runtime.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KreaActivationTopologyV1 {
    /// Exact model artifact identity.
    pub model: Digest,
    /// Exact backend build identity.
    pub backend: Digest,
    /// Exact native topology-discovery implementation.
    pub implementation: Digest,
    /// Runtime-reported conditioner-layer count.
    pub conditioner_layers: u32,
    /// Runtime-reported Krea transformer-block count.
    pub transformer_blocks: u32,
    /// Canonically site-ordered supported boundaries.
    pub sites: Vec<KreaActivationSiteV1>,
}

impl KreaActivationTopologyV1 {
    /// Validates runtime dimensions and returns the exact topology identity.
    ///
    /// # Errors
    ///
    /// Returns an error for empty, excessive, duplicate, or internally
    /// inconsistent topology data.
    pub fn digest(&self) -> Result<Digest, CoreError> {
        if self.conditioner_layers == 0
            || self.transformer_blocks == 0
            || self.sites.is_empty()
            || self.sites.len() > MAX_KREA_ACTIVATION_SITES
        {
            return Err(CoreError::invalid(
                "Krea activation topology",
                "layer, block, or site count is outside the supported bound",
            ));
        }
        let mut prior = None;
        let mut kinds = BTreeSet::new();
        for site in &self.sites {
            if prior.is_some_and(|prior| prior >= site.site)
                || site.width == 0
                || u64::from(site.width) > MAX_KREA_ACTIVATION_ELEMENTS
                || site.boundaries.is_empty()
                || site.token_domains.is_empty()
                || site.branches.is_empty()
                || !strictly_ordered(&site.boundaries)
                || !strictly_ordered(&site.token_domains)
                || !strictly_ordered(&site.branches)
                || site
                    .token_domains
                    .iter()
                    .any(|domain| domain.maximum_tokens == 0)
                || !kinds.insert(site.kind.clone())
            {
                return Err(CoreError::invalid(
                    "Krea activation topology",
                    "sites, domains, branches, and widths must be canonical and bounded",
                ));
            }
            match site.kind {
                KreaActivationSiteKindV1::ConditionerLayerOutput { layer }
                    if layer >= self.conditioner_layers =>
                {
                    return Err(CoreError::invalid(
                        "Krea activation topology",
                        "conditioner site exceeds the loaded conditioner",
                    ));
                }
                KreaActivationSiteKindV1::TextResidual { block }
                | KreaActivationSiteKindV1::TransformerResidual { block }
                    if block >= self.transformer_blocks =>
                {
                    return Err(CoreError::invalid(
                        "Krea activation topology",
                        "residual site exceeds the loaded transformer",
                    ));
                }
                _ => {}
            }
            prior = Some(site.site);
        }
        Digest::of_serializable("krea-activation-topology-v1", self)
    }

    /// Resolves one stable site identifier.
    pub fn site(&self, site: u16) -> Option<&KreaActivationSiteV1> {
        self.sites
            .binary_search_by_key(&site, |candidate| candidate.site)
            .ok()
            .map(|index| &self.sites[index])
    }
}

/// Half-open token interval.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KreaTokenRangeV1 {
    /// Inclusive first token.
    pub start: u32,
    /// Exclusive final token.
    pub end: u32,
}

impl KreaTokenRangeV1 {
    fn len(self) -> Option<u32> {
        self.end
            .checked_sub(self.start)
            .filter(|length| *length > 0)
    }
}

/// Exact token rows selected at one site.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum KreaTokenSelectionV1 {
    /// Every runtime-present token in one declared family.
    All {
        /// Logical token family.
        domain: KreaTokenDomainKindV1,
    },
    /// Canonically ordered, disjoint half-open ranges.
    Ranges {
        /// Logical token family.
        domain: KreaTokenDomainKindV1,
        /// Selected ranges.
        ranges: Vec<KreaTokenRangeV1>,
    },
}

impl KreaTokenSelectionV1 {
    /// Returns the selected token family.
    pub const fn domain(&self) -> KreaTokenDomainKindV1 {
        match self {
            Self::All { domain } | Self::Ranges { domain, .. } => *domain,
        }
    }

    fn validate_for(&self, site: &KreaActivationSiteV1) -> Result<u32, CoreError> {
        let maximum = site
            .token_domains
            .iter()
            .find(|candidate| candidate.kind == self.domain())
            .map(|candidate| candidate.maximum_tokens)
            .ok_or_else(|| {
                CoreError::invalid(
                    "Krea token selection",
                    "token family is unavailable at the selected site",
                )
            })?;
        match self {
            Self::All { .. } => Ok(maximum),
            Self::Ranges { ranges, .. } => {
                if ranges.is_empty()
                    || ranges.len() > MAX_KREA_TOKEN_RANGES
                    || ranges
                        .iter()
                        .any(|range| range.len().is_none() || range.end > maximum)
                    || ranges.windows(2).any(|pair| pair[0].end > pair[1].start)
                {
                    return Err(CoreError::invalid(
                        "Krea token selection",
                        "ranges must be nonempty, disjoint, increasing, and in range",
                    ));
                }
                ranges.iter().try_fold(0_u32, |total, range| {
                    total
                        .checked_add(range.len().ok_or_else(|| {
                            CoreError::invalid("Krea token selection", "range is empty")
                        })?)
                        .ok_or_else(|| {
                            CoreError::invalid("Krea token selection", "token count overflowed")
                        })
                })
            }
        }
    }
}

/// Execution boundary selected by a capture or operator.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum KreaActivationBoundaryV1 {
    /// Apply at one logical boundary after conditioning and before denoising.
    PreDenoiser,
    /// Run at selected denoising transitions.
    Transitions {
        /// Exact transition selection.
        steps: StepSelector,
    },
}

impl KreaActivationBoundaryV1 {
    /// Returns the site capability required by this boundary.
    pub const fn kind(&self) -> KreaActivationBoundaryKindV1 {
        match self {
            Self::PreDenoiser => KreaActivationBoundaryKindV1::PreDenoiser,
            Self::Transitions { .. } => KreaActivationBoundaryKindV1::Transition,
        }
    }

    fn validate_for(
        &self,
        site: &KreaActivationSiteV1,
        step_count: usize,
    ) -> Result<(), CoreError> {
        if !site.boundaries.contains(&self.kind()) {
            return Err(CoreError::invalid(
                "Krea activation boundary",
                "selected boundary is unavailable at the site",
            ));
        }
        match self {
            Self::PreDenoiser => Ok(()),
            Self::Transitions { steps } => steps.validate_for(step_count),
        }
    }
}

/// Data retained for one activation capture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KreaActivationCaptureRetentionV1 {
    /// Retain only an exact content identity and element count.
    Digest,
    /// Retain identity plus deterministic scalar statistics.
    Statistics,
    /// Retain a bounded device-resident snapshot value.
    DeviceSnapshot,
}

/// One bounded capture request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KreaActivationCaptureV1 {
    /// Canonical capture index.
    pub capture: u16,
    /// Selected topology site.
    pub site: u16,
    /// Selected token rows.
    pub tokens: KreaTokenSelectionV1,
    /// Selected evaluation boundary.
    pub boundary: KreaActivationBoundaryV1,
    /// Selected CFG branch.
    pub branch: KreaCfgBranchV1,
    /// Retention behavior.
    pub retention: KreaActivationCaptureRetentionV1,
    /// Inclusive element bound.
    pub maximum_elements: u64,
    /// Inclusive returned-host-byte bound.
    pub maximum_host_bytes: u64,
    /// Inclusive retained-device-byte bound.
    pub maximum_device_bytes: u64,
}

impl KreaActivationCaptureV1 {
    fn validate_for(
        &self,
        topology: &KreaActivationTopologyV1,
        step_count: usize,
    ) -> Result<(), CoreError> {
        let site = topology.site(self.site).ok_or_else(|| {
            CoreError::invalid("Krea activation capture", "selected site is absent")
        })?;
        let tokens = u64::from(self.tokens.validate_for(site)?);
        self.boundary.validate_for(site, step_count)?;
        if !site.branches.contains(&self.branch) {
            return Err(CoreError::invalid(
                "Krea activation capture",
                "CFG branch is unavailable at the selected site",
            ));
        }
        let required = tokens.checked_mul(u64::from(site.width)).ok_or_else(|| {
            CoreError::invalid("Krea activation capture", "element count overflowed")
        })?;
        if self.maximum_elements == 0
            || self.maximum_elements < required
            || self.maximum_elements > MAX_KREA_ACTIVATION_ELEMENTS
            || self.maximum_host_bytes > MAX_KREA_ACTIVATION_VALUE_BYTES
            || self.maximum_device_bytes > MAX_KREA_ACTIVATION_VALUE_BYTES
        {
            return Err(CoreError::invalid(
                "Krea activation capture",
                "capture bounds are zero, excessive, or smaller than the selection",
            ));
        }
        let required_bytes = required.checked_mul(4).ok_or_else(|| {
            CoreError::invalid("Krea activation capture", "byte count overflowed")
        })?;
        match self.retention {
            KreaActivationCaptureRetentionV1::Digest
                if self.maximum_host_bytes != 0 || self.maximum_device_bytes != 0 =>
            {
                Err(CoreError::invalid(
                    "Krea activation capture",
                    "digest retention cannot retain host or device bytes",
                ))
            }
            KreaActivationCaptureRetentionV1::Statistics
                if self.maximum_host_bytes < 16 || self.maximum_device_bytes != 0 =>
            {
                Err(CoreError::invalid(
                    "Krea activation capture",
                    "statistics require bounded host bytes and no device snapshot",
                ))
            }
            KreaActivationCaptureRetentionV1::DeviceSnapshot
                if self.maximum_host_bytes != 0 || self.maximum_device_bytes < required_bytes =>
            {
                Err(CoreError::invalid(
                    "Krea activation capture",
                    "device snapshots require sufficient device bytes and no host return",
                ))
            }
            _ => Ok(()),
        }
    }
}

/// Exact representation of an imported Krea vector bank.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KreaVectorRepresentationV1 {
    /// Canonical rank-major little-endian finite `f32` rows.
    F32Rows,
    /// Canonical rank-major orthonormal little-endian finite `f32` rows.
    OrthonormalF32Rows,
}

/// Mechanical role of one imported activation value.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum KreaActivationInputKindV1 {
    /// Captured donor rows with the same row width as the selected site.
    Donor {
        /// Donor site.
        site: u16,
        /// Donor token family.
        domain: KreaTokenDomainKindV1,
        /// Number of donor rows.
        tokens: u32,
        /// Donor CFG branch.
        branch: KreaCfgBranchV1,
        /// Elements in each row.
        width: u32,
    },
    /// Low-rank vector rows.
    VectorBank {
        /// Compatible site.
        site: u16,
        /// Elements in each vector.
        width: u32,
        /// Number of vectors.
        rank: u16,
        /// Exact scalar/ordering contract.
        representation: KreaVectorRepresentationV1,
    },
}

/// Origin of one typed activation input.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum KreaActivationInputSourceV1 {
    /// Caller-supplied sealed finite `f32` bytes imported exactly once.
    Sealed {
        /// Canonical little-endian finite `f32` byte identity.
        content: Digest,
        /// Exact imported byte count.
        bytes: u64,
    },
    /// A single device snapshot produced earlier at the same boundary.
    Capture {
        /// Capture whose device-resident output supplies this input.
        capture: u16,
    },
}

/// One topology-bound activation input retained by a resident session.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KreaActivationInputV1 {
    /// Canonical input index.
    pub input: u16,
    /// Required topology identity.
    pub topology: Digest,
    /// Mechanical input role.
    pub kind: KreaActivationInputKindV1,
    /// Sealed import or same-run SSA capture source.
    pub source: KreaActivationInputSourceV1,
}

impl KreaActivationInputV1 {
    fn validate_for(
        &self,
        topology: &KreaActivationTopologyV1,
        captures: &HashMap<u16, &KreaActivationCaptureV1>,
        step_count: usize,
    ) -> Result<(), CoreError> {
        if self.topology != topology.digest()? {
            return Err(CoreError::invalid(
                "Krea activation input",
                "topology identity does not match",
            ));
        }
        let (site_id, width, rows) = match self.kind {
            KreaActivationInputKindV1::Donor {
                site,
                domain,
                tokens,
                branch,
                width,
            } => {
                let selected = topology.site(site).ok_or_else(|| {
                    CoreError::invalid("Krea activation input", "donor site is absent")
                })?;
                let maximum = selected
                    .token_domains
                    .iter()
                    .find(|candidate| candidate.kind == domain)
                    .map(|candidate| candidate.maximum_tokens)
                    .ok_or_else(|| {
                        CoreError::invalid(
                            "Krea activation input",
                            "donor token family is unavailable",
                        )
                    })?;
                if tokens == 0
                    || tokens > maximum
                    || width != selected.width
                    || !selected.branches.contains(&branch)
                {
                    return Err(CoreError::invalid(
                        "Krea activation input",
                        "donor shape or branch differs from the selected site",
                    ));
                }
                (site, width, u64::from(tokens))
            }
            KreaActivationInputKindV1::VectorBank {
                site, width, rank, ..
            } => {
                let selected = topology.site(site).ok_or_else(|| {
                    CoreError::invalid("Krea activation input", "vector site is absent")
                })?;
                if rank == 0 || rank > MAX_KREA_VECTOR_RANK || width != selected.width {
                    return Err(CoreError::invalid(
                        "Krea activation input",
                        "vector rank or width differs from the selected site",
                    ));
                }
                (site, width, u64::from(rank))
            }
        };
        let expected = rows
            .checked_mul(u64::from(width))
            .and_then(|elements| elements.checked_mul(4))
            .ok_or_else(|| CoreError::invalid("Krea activation input", "byte count overflowed"))?;
        if expected == 0 || expected > MAX_KREA_ACTIVATION_VALUE_BYTES {
            return Err(CoreError::invalid(
                "Krea activation input",
                "declared shape exceeds the input byte bound",
            ));
        }
        self.validate_source_for(topology, captures, step_count, site_id, expected)
    }

    fn validate_source_for(
        &self,
        topology: &KreaActivationTopologyV1,
        captures: &HashMap<u16, &KreaActivationCaptureV1>,
        step_count: usize,
        site_id: u16,
        expected: u64,
    ) -> Result<(), CoreError> {
        match &self.source {
            KreaActivationInputSourceV1::Sealed { bytes, .. } if *bytes == expected => {}
            KreaActivationInputSourceV1::Sealed { .. } => {
                return Err(CoreError::invalid(
                    "Krea activation input",
                    "sealed byte count differs from the declared shape",
                ));
            }
            KreaActivationInputSourceV1::Capture { capture } => {
                let KreaActivationInputKindV1::Donor {
                    domain,
                    tokens,
                    branch,
                    ..
                } = self.kind
                else {
                    return Err(CoreError::invalid(
                        "Krea activation input",
                        "only donor values may originate from a capture",
                    ));
                };
                let capture = captures.get(capture).ok_or_else(|| {
                    CoreError::invalid("Krea activation input", "source capture is absent")
                })?;
                let selected_tokens = match &capture.tokens {
                    KreaTokenSelectionV1::Ranges {
                        domain: selected_domain,
                        ranges,
                    } if *selected_domain == domain => {
                        ranges.iter().try_fold(0_u32, |total, range| {
                            total.checked_add(range.end - range.start).ok_or_else(|| {
                                CoreError::invalid(
                                    "Krea activation input",
                                    "capture token count overflowed",
                                )
                            })
                        })?
                    }
                    _ => {
                        return Err(CoreError::invalid(
                            "Krea activation input",
                            "SSA donors require explicit captured token ranges",
                        ));
                    }
                };
                let one_reach = match &capture.boundary {
                    KreaActivationBoundaryV1::PreDenoiser => true,
                    KreaActivationBoundaryV1::Transitions {
                        steps: StepSelector::Exact { steps },
                    } => steps.len() == 1,
                    KreaActivationBoundaryV1::Transitions { .. } => false,
                };
                if capture.site != site_id
                    || capture.branch != branch
                    || capture.retention != KreaActivationCaptureRetentionV1::DeviceSnapshot
                    || selected_tokens != tokens
                    || capture.maximum_device_bytes < expected
                    || !one_reach
                {
                    return Err(CoreError::invalid(
                        "Krea activation input",
                        "SSA donor capture shape, branch, retention, or reach count differs",
                    ));
                }
                capture.validate_for(topology, step_count)?;
            }
        }
        Ok(())
    }

    /// Returns the exact logical bytes represented by this input.
    ///
    /// # Errors
    ///
    /// Returns an error if the declared shape overflows.
    pub fn bytes(&self) -> Result<u64, CoreError> {
        let (width, rows) = match self.kind {
            KreaActivationInputKindV1::Donor { width, tokens, .. } => (width, u64::from(tokens)),
            KreaActivationInputKindV1::VectorBank { width, rank, .. } => (width, u64::from(rank)),
        };
        rows.checked_mul(u64::from(width))
            .and_then(|elements| elements.checked_mul(4))
            .ok_or_else(|| CoreError::invalid("Krea activation input", "byte count overflowed"))
    }

    /// Validates exact finite canonical input bytes.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed length, non-finite values, content
    /// substitution, a non-orthonormal declared basis, or an SSA-only input.
    pub fn validate_bytes(&self, bytes: &[u8]) -> Result<(), CoreError> {
        let KreaActivationInputSourceV1::Sealed {
            content,
            bytes: expected,
        } = &self.source
        else {
            return Err(CoreError::invalid(
                "Krea activation input bytes",
                "SSA capture inputs have no caller-supplied bytes",
            ));
        };
        if u64::try_from(bytes.len()).unwrap_or(u64::MAX) != *expected
            || !bytes.len().is_multiple_of(4)
            || krea_activation_input_content_v1(bytes) != *content
        {
            return Err(CoreError::invalid(
                "Krea activation input bytes",
                "length or content identity differs from the declaration",
            ));
        }
        let values = decode_f32(bytes)?;
        if let KreaActivationInputKindV1::VectorBank {
            width,
            rank,
            representation: KreaVectorRepresentationV1::OrthonormalF32Rows,
            ..
        } = self.kind
        {
            validate_orthonormal(&values, width, rank)?;
        }
        Ok(())
    }
}

/// Returns the canonical content identity for imported finite `f32` bytes.
pub fn krea_activation_input_content_v1(bytes: &[u8]) -> Digest {
    Digest::of_bytes("krea-activation-input-f32-le-v1", bytes)
}

/// Built-in Krea activation operation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum KreaActivationOperatorV1 {
    /// Replace or linearly blend selected rows with donor rows.
    DonorTransplant {
        /// Imported donor input.
        input: u16,
    },
    /// `x' = x + alpha * v` for one vector row.
    ScaledVectorAdd {
        /// Imported vector-bank input.
        input: u16,
        /// Zero-based vector row.
        vector: u16,
    },
    /// `x' = x - alpha * v` for one vector row.
    ScaledVectorSubtract {
        /// Imported vector-bank input.
        input: u16,
        /// Zero-based vector row.
        vector: u16,
    },
    /// `x' = x - alpha * U(U^T x)` for an orthonormal bank.
    OrthogonalProjectionRemoval {
        /// Imported orthonormal vector bank.
        input: u16,
    },
    /// Remove only positive projection coefficients.
    OneSidedProjectionRemoval {
        /// Imported orthonormal vector bank.
        input: u16,
    },
}

impl KreaActivationOperatorV1 {
    fn input(&self) -> u16 {
        match self {
            Self::DonorTransplant { input }
            | Self::ScaledVectorAdd { input, .. }
            | Self::ScaledVectorSubtract { input, .. }
            | Self::OrthogonalProjectionRemoval { input }
            | Self::OneSidedProjectionRemoval { input } => *input,
        }
    }
}

/// One ordered, site/token/transition/branch-scoped operation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KreaActivationOperationV1 {
    /// Canonical operation index.
    pub operation: u16,
    /// Selected topology site.
    pub site: u16,
    /// Selected token rows.
    pub tokens: KreaTokenSelectionV1,
    /// Selected evaluation boundary.
    pub boundary: KreaActivationBoundaryV1,
    /// Selected CFG branch.
    pub branch: KreaCfgBranchV1,
    /// Built-in operation and imported input.
    pub operator: KreaActivationOperatorV1,
    /// Exact IEEE-754 strength bits.
    pub strength_bits: u32,
}

impl KreaActivationOperationV1 {
    /// Returns the exact finite operation strength.
    ///
    /// # Errors
    ///
    /// Returns an error for NaN, infinity, or an operation-specific range
    /// violation. Zero is a valid exact no-op control.
    pub fn strength(&self) -> Result<f32, CoreError> {
        let strength = f32::from_bits(self.strength_bits);
        let valid = match self.operator {
            KreaActivationOperatorV1::ScaledVectorAdd { .. }
            | KreaActivationOperatorV1::ScaledVectorSubtract { .. } => {
                strength.is_finite() && (-64.0..=64.0).contains(&strength)
            }
            _ => strength.is_finite() && (0.0..=1.0).contains(&strength),
        };
        if !valid {
            return Err(CoreError::invalid(
                "Krea activation operation",
                "strength is non-finite or outside its mechanical bound",
            ));
        }
        Ok(strength)
    }

    fn validate_for(
        &self,
        topology: &KreaActivationTopologyV1,
        step_count: usize,
        inputs: &HashMap<u16, &KreaActivationInputV1>,
        captures: &HashMap<u16, &KreaActivationCaptureV1>,
    ) -> Result<(), CoreError> {
        let site = topology.site(self.site).ok_or_else(|| {
            CoreError::invalid("Krea activation operation", "selected site is absent")
        })?;
        let selected_tokens = self.tokens.validate_for(site)?;
        self.boundary.validate_for(site, step_count)?;
        self.strength()?;
        if !site.branches.contains(&self.branch) {
            return Err(CoreError::invalid(
                "Krea activation operation",
                "CFG branch is unavailable at the selected site",
            ));
        }
        let input = inputs
            .get(&self.operator.input())
            .ok_or_else(|| CoreError::invalid("Krea activation operation", "input is absent"))?;
        let input_site = match input.kind {
            KreaActivationInputKindV1::Donor {
                site,
                domain,
                tokens,
                ..
            } => {
                if !matches!(
                    self.operator,
                    KreaActivationOperatorV1::DonorTransplant { .. }
                ) || domain != self.tokens.domain()
                    || tokens != selected_tokens
                {
                    return Err(CoreError::invalid(
                        "Krea activation operation",
                        "donor input role or token family is incompatible",
                    ));
                }
                site
            }
            KreaActivationInputKindV1::VectorBank {
                site,
                rank,
                representation,
                ..
            } => {
                match self.operator {
                    KreaActivationOperatorV1::ScaledVectorAdd { vector, .. }
                    | KreaActivationOperatorV1::ScaledVectorSubtract { vector, .. }
                        if vector < rank => {}
                    KreaActivationOperatorV1::OrthogonalProjectionRemoval { .. }
                    | KreaActivationOperatorV1::OneSidedProjectionRemoval { .. }
                        if representation == KreaVectorRepresentationV1::OrthonormalF32Rows => {}
                    _ => {
                        return Err(CoreError::invalid(
                            "Krea activation operation",
                            "vector operator, rank, or representation is incompatible",
                        ));
                    }
                }
                site
            }
        };
        if input_site != self.site {
            return Err(CoreError::invalid(
                "Krea activation operation",
                "input and selected site differ",
            ));
        }
        if let KreaActivationInputSourceV1::Capture { capture } = input.source {
            let capture = captures.get(&capture).ok_or_else(|| {
                CoreError::invalid("Krea activation operation", "source capture is absent")
            })?;
            if capture.boundary != self.boundary || capture.branch != self.branch {
                return Err(CoreError::invalid(
                    "Krea activation operation",
                    "SSA donor capture must precede the operation at the same boundary and branch",
                ));
            }
        }
        Ok(())
    }
}

/// Complete always-on Krea activation request for one resident image stage.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KreaActivationPlanV1 {
    /// Required loaded-topology identity.
    pub topology: Digest,
    /// Exact denoising transition count.
    pub step_count: u32,
    /// Canonically input-ordered resident values.
    pub inputs: Vec<KreaActivationInputV1>,
    /// Canonically capture-ordered observations.
    pub captures: Vec<KreaActivationCaptureV1>,
    /// Ordered installed operations.
    pub operations: Vec<KreaActivationOperationV1>,
    /// Inclusive host residency bound for all activation values.
    pub maximum_host_bytes: u64,
    /// Inclusive device residency bound for all activation values.
    pub maximum_device_bytes: u64,
    /// Inclusive aggregate native application bound.
    pub maximum_applications: u64,
}

impl KreaActivationPlanV1 {
    /// Validates the exact request and returns its stable identity.
    ///
    /// The contract has no policy gate, classifier, semantic selector, or
    /// optional enablement field. Supplying the plan installs it for the whole
    /// request.
    ///
    /// # Errors
    ///
    /// Returns the first topology, collection, reference, selection, or
    /// resource-bound defect.
    pub fn digest_for(&self, topology: &KreaActivationTopologyV1) -> Result<Digest, CoreError> {
        if self.topology != topology.digest()?
            || self.step_count == 0
            || self.inputs.len() > MAX_KREA_ACTIVATION_INPUTS
            || self.captures.len() > MAX_KREA_ACTIVATION_CAPTURES
            || self.operations.len() > MAX_KREA_ACTIVATION_OPERATIONS
            || (self.captures.is_empty() && self.operations.is_empty())
            || self.maximum_applications == 0
            || self.maximum_applications > MAX_KREA_ACTIVATION_APPLICATIONS
            || self.maximum_host_bytes > MAX_KREA_ACTIVATION_VALUE_BYTES
            || self.maximum_device_bytes > MAX_KREA_ACTIVATION_VALUE_BYTES
        {
            return Err(CoreError::invalid(
                "Krea activation plan",
                "topology, collection, transition, or resource bounds are invalid",
            ));
        }
        let step_count = usize::try_from(self.step_count)
            .map_err(|_| CoreError::invalid("Krea activation plan", "step count exceeds usize"))?;
        let mut captures = HashMap::new();
        let mut prior_capture = None;
        let mut retained_device = 0_u64;
        let mut retained_host = 0_u64;
        for capture in &self.captures {
            if prior_capture.is_some_and(|prior| prior >= capture.capture)
                || captures.insert(capture.capture, capture).is_some()
            {
                return Err(CoreError::invalid(
                    "Krea activation plan",
                    "captures must be uniquely and canonically ordered",
                ));
            }
            capture.validate_for(topology, step_count)?;
            retained_device = retained_device
                .checked_add(capture.maximum_device_bytes)
                .ok_or_else(|| {
                    CoreError::invalid("Krea activation plan", "device bytes overflowed")
                })?;
            retained_host = retained_host
                .checked_add(capture.maximum_host_bytes)
                .ok_or_else(|| {
                    CoreError::invalid("Krea activation plan", "host bytes overflowed")
                })?;
            prior_capture = Some(capture.capture);
        }
        let mut inputs = HashMap::new();
        let mut prior_input = None;
        for input in &self.inputs {
            input.validate_for(topology, &captures, step_count)?;
            if prior_input.is_some_and(|prior| prior >= input.input)
                || inputs.insert(input.input, input).is_some()
            {
                return Err(CoreError::invalid(
                    "Krea activation plan",
                    "inputs must be uniquely and canonically ordered",
                ));
            }
            prior_input = Some(input.input);
            if matches!(input.source, KreaActivationInputSourceV1::Sealed { .. }) {
                retained_device = retained_device.checked_add(input.bytes()?).ok_or_else(|| {
                    CoreError::invalid("Krea activation plan", "input bytes overflowed")
                })?;
            }
        }
        let mut prior_operation = None;
        let mut consumed_inputs = BTreeSet::new();
        for operation in &self.operations {
            if prior_operation.is_some_and(|prior| prior >= operation.operation) {
                return Err(CoreError::invalid(
                    "Krea activation plan",
                    "operations must be uniquely and canonically ordered",
                ));
            }
            operation.validate_for(topology, step_count, &inputs, &captures)?;
            consumed_inputs.insert(operation.operator.input());
            prior_operation = Some(operation.operation);
        }
        if consumed_inputs.len() != inputs.len() {
            return Err(CoreError::invalid(
                "Krea activation plan",
                "every retained input must have a declared operation consumer",
            ));
        }
        if retained_device > self.maximum_device_bytes || retained_host > self.maximum_host_bytes {
            return Err(CoreError::invalid(
                "Krea activation plan",
                "aggregate retained bytes exceed the declared resource bound",
            ));
        }
        Digest::of_serializable("krea-activation-plan-v1", self)
    }
}

/// Deterministic capture evidence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KreaActivationCaptureReceiptV1 {
    /// Capture index.
    pub capture: u16,
    /// Exact number of reached boundaries.
    pub reached: u64,
    /// Exact selected element count across reached boundaries.
    pub elements: u64,
    /// Canonical activation content identity when the boundary was reached.
    pub content: Option<Digest>,
    /// Deterministic statistics when requested.
    pub statistics: Option<ActivationStatisticsV1>,
    /// Device-resident snapshot identity when requested.
    pub snapshot: Option<Digest>,
}

/// Deterministic application evidence for one installed operation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KreaActivationApplicationV1 {
    /// Operation index.
    pub operation: u16,
    /// Number of selected native boundaries reached.
    pub reached: u64,
    /// Number of complete transactional writes.
    pub applied: u64,
    /// Writes whose result was byte-identical to the input.
    pub unchanged: u64,
    /// Canonical input tensor identity before the first reached write.
    pub input: Option<Digest>,
    /// Canonical output tensor identity after the final reached write.
    pub output: Option<Digest>,
}

/// Terminal boundary reached by an activation request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum KreaActivationTerminalV1 {
    /// Every selected diffusion boundary completed.
    Completed,
    /// Cooperative cancellation with the last completed transition, if any.
    Cancelled {
        /// Zero-based final completed transition; `None` means pre-transition.
        after_transition: Option<u32>,
    },
}

/// Request-local cleanup disposition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KreaActivationCleanupDispositionV1 {
    /// Every hook and request-local reference was removed.
    Confirmed,
    /// Cleanup was uncertain and the executor was poisoned.
    Poisoned,
}

/// Complete deterministic Krea activation execution evidence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KreaActivationReceiptV1 {
    /// Exact activation-plan identity.
    pub plan: Digest,
    /// Exact loaded-topology identity.
    pub topology: Digest,
    /// Exact backend build identity.
    pub backend: Digest,
    /// Native runtime epoch.
    pub runtime_epoch: u64,
    /// Exact successful or cooperatively cancelled terminal boundary.
    pub terminal: KreaActivationTerminalV1,
    /// Canonical capture evidence.
    pub captures: Vec<KreaActivationCaptureReceiptV1>,
    /// Canonical operation evidence.
    pub applications: Vec<KreaActivationApplicationV1>,
    /// Request-local cleanup disposition.
    pub cleanup: KreaActivationCleanupDispositionV1,
}

impl KreaActivationReceiptV1 {
    /// Validates complete lineage/accounting and returns the receipt identity.
    ///
    /// # Errors
    ///
    /// Returns an error for missing evidence, over-bound accounting, invalid
    /// no-op behavior, or unconfirmed cleanup.
    pub fn digest_for(
        &self,
        plan: &KreaActivationPlanV1,
        topology: &KreaActivationTopologyV1,
    ) -> Result<Digest, CoreError> {
        if self.plan != plan.digest_for(topology)?
            || self.topology != topology.digest()?
            || self.backend != topology.backend
            || self.runtime_epoch == 0
            || self.cleanup != KreaActivationCleanupDispositionV1::Confirmed
            || self.captures.len() != plan.captures.len()
            || self.applications.len() != plan.operations.len()
        {
            return Err(CoreError::invalid(
                "Krea activation receipt",
                "lineage, evidence coverage, epoch, or cleanup is invalid",
            ));
        }
        if let KreaActivationTerminalV1::Cancelled {
            after_transition: Some(step),
        } = self.terminal
            && step >= plan.step_count
        {
            return Err(CoreError::invalid(
                "Krea activation receipt",
                "cancellation transition exceeds the plan",
            ));
        }
        let completed = self.terminal == KreaActivationTerminalV1::Completed;
        for (receipt, expected) in self.captures.iter().zip(&plan.captures) {
            let statistics = matches!(
                expected.retention,
                KreaActivationCaptureRetentionV1::Statistics
            );
            let snapshot = matches!(
                expected.retention,
                KreaActivationCaptureRetentionV1::DeviceSnapshot
            );
            let reached = receipt.reached > 0;
            if receipt.capture != expected.capture
                || (completed && !reached)
                || (reached != receipt.content.is_some())
                || (!reached && receipt.elements != 0)
                || (reached && receipt.elements == 0)
                || receipt.elements > expected.maximum_elements.saturating_mul(receipt.reached)
                || receipt.statistics.is_some() != (statistics && reached)
                || receipt.snapshot.is_some() != (snapshot && reached)
            {
                return Err(CoreError::invalid(
                    "Krea activation receipt",
                    "capture evidence differs from the selected retention or bounds",
                ));
            }
        }
        let mut applications = 0_u64;
        for (receipt, expected) in self.applications.iter().zip(&plan.operations) {
            applications = applications.checked_add(receipt.applied).ok_or_else(|| {
                CoreError::invalid("Krea activation receipt", "application count overflowed")
            })?;
            let zero = expected.strength()? == 0.0;
            let reached = receipt.reached > 0;
            if receipt.operation != expected.operation
                || (completed && !reached)
                || receipt.applied > receipt.reached
                || receipt.unchanged > receipt.applied
                || (completed && receipt.applied != receipt.reached)
                || (reached != receipt.input.is_some())
                || (reached != receipt.output.is_some())
                || (!reached && (receipt.applied != 0 || receipt.unchanged != 0))
                || (zero
                    && reached
                    && (receipt.applied != receipt.reached
                        || receipt.unchanged != receipt.applied
                        || receipt.input != receipt.output))
            {
                return Err(CoreError::invalid(
                    "Krea activation receipt",
                    "application evidence or zero-strength behavior is invalid",
                ));
            }
        }
        if applications > plan.maximum_applications {
            return Err(CoreError::invalid(
                "Krea activation receipt",
                "application count exceeds the declared bound",
            ));
        }
        Digest::of_serializable("krea-activation-receipt-v1", self)
    }
}

/// Observed placement and transfer accounting for one imported input.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KreaActivationInputMeasurementV1 {
    /// Input index.
    pub input: u16,
    /// Device identity retaining the input.
    pub device: String,
    /// Actual retained bytes.
    pub resident_bytes: u64,
    /// Host-to-device copy count across the session.
    pub host_to_device_transfers: u64,
    /// Host-to-device bytes across the session.
    pub host_to_device_bytes: u64,
    /// Jobs that reused this resident value.
    pub jobs: u64,
}

/// Non-deterministic placement/transfer measurements excluded from semantic
/// identities.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KreaActivationMeasurementsV1 {
    /// Exact plan identity.
    pub plan: Digest,
    /// Exact runtime epoch.
    pub runtime_epoch: u64,
    /// Peak request-local host bytes.
    pub peak_host_bytes: u64,
    /// Peak request-local device bytes.
    pub peak_device_bytes: u64,
    /// Canonically input-ordered placement records.
    pub inputs: Vec<KreaActivationInputMeasurementV1>,
}

impl KreaActivationMeasurementsV1 {
    /// Validates placement and copy accounting against a completed request.
    ///
    /// # Errors
    ///
    /// Returns an error for lineage, resource, input, device-label, or copy
    /// inconsistencies.
    pub fn validate_for(
        &self,
        plan: &KreaActivationPlanV1,
        topology: &KreaActivationTopologyV1,
        receipt: &KreaActivationReceiptV1,
    ) -> Result<(), CoreError> {
        receipt.digest_for(plan, topology)?;
        if self.plan != receipt.plan
            || self.runtime_epoch != receipt.runtime_epoch
            || self.peak_host_bytes > plan.maximum_host_bytes
            || self.peak_device_bytes > plan.maximum_device_bytes
            || self.inputs.len() != plan.inputs.len()
        {
            return Err(CoreError::invalid(
                "Krea activation measurements",
                "lineage, peak resources, or input coverage differs",
            ));
        }
        for (measurement, input) in self.inputs.iter().zip(&plan.inputs) {
            let bytes = input.bytes()?;
            let (expected_transfers, expected_transfer_bytes) = match input.source {
                KreaActivationInputSourceV1::Sealed { .. } => (1, bytes),
                KreaActivationInputSourceV1::Capture { .. } => (0, 0),
            };
            if measurement.input != input.input
                || measurement.device.is_empty()
                || measurement.device.len() > 256
                || measurement.device.contains('\0')
                || measurement.resident_bytes != bytes
                || measurement.host_to_device_transfers != expected_transfers
                || measurement.host_to_device_bytes != expected_transfer_bytes
                || measurement.jobs == 0
            {
                return Err(CoreError::invalid(
                    "Krea activation measurements",
                    "resident input placement, source transfer, or job reuse differs",
                ));
            }
        }
        Ok(())
    }
}

/// Applies one built-in operation to a single finite activation row.
///
/// This content-free reference implementation establishes arithmetic and
/// no-op behavior. Native adapters must emit their own execution evidence.
///
/// # Errors
///
/// Returns an error for a malformed row/input, incompatible shape or role,
/// non-orthonormal projection basis, or non-finite result.
pub fn apply_krea_activation_operation_v1(
    row: &mut [f32],
    operation: &KreaActivationOperationV1,
    input: &KreaActivationInputV1,
    input_bytes: &[u8],
) -> Result<(), CoreError> {
    input.validate_bytes(input_bytes)?;
    let values = decode_f32(input_bytes)?;
    let width = match input.kind {
        KreaActivationInputKindV1::Donor { width, .. }
        | KreaActivationInputKindV1::VectorBank { width, .. } => width,
    };
    let width = usize::try_from(width)
        .map_err(|_| CoreError::invalid("Krea activation arithmetic", "width exceeds usize"))?;
    if row.len() != width || row.iter().any(|value| !value.is_finite()) {
        return Err(CoreError::invalid(
            "Krea activation arithmetic",
            "target row is non-finite or width-mismatched",
        ));
    }
    let strength = operation.strength()?;
    if strength == 0.0 {
        return Ok(());
    }
    let original = row.to_vec();
    match operation.operator {
        KreaActivationOperatorV1::DonorTransplant { input: selected } => {
            require_input(selected, input.input)?;
            let donor = values.get(..width).ok_or_else(|| {
                CoreError::invalid("Krea activation arithmetic", "donor row is absent")
            })?;
            for (target, donor) in row.iter_mut().zip(donor) {
                *target = (*target).mul_add(1.0 - strength, *donor * strength);
            }
        }
        KreaActivationOperatorV1::ScaledVectorAdd {
            input: selected,
            vector,
        }
        | KreaActivationOperatorV1::ScaledVectorSubtract {
            input: selected,
            vector,
        } => {
            require_input(selected, input.input)?;
            let start = usize::from(vector).checked_mul(width).ok_or_else(|| {
                CoreError::invalid("Krea activation arithmetic", "vector offset overflowed")
            })?;
            let vector_values =
                values
                    .get(start..start.saturating_add(width))
                    .ok_or_else(|| {
                        CoreError::invalid("Krea activation arithmetic", "vector row is absent")
                    })?;
            let direction = if matches!(
                operation.operator,
                KreaActivationOperatorV1::ScaledVectorSubtract { .. }
            ) {
                -strength
            } else {
                strength
            };
            for (target, vector) in row.iter_mut().zip(vector_values) {
                *target = direction.mul_add(*vector, *target);
            }
        }
        KreaActivationOperatorV1::OrthogonalProjectionRemoval { input: selected }
        | KreaActivationOperatorV1::OneSidedProjectionRemoval { input: selected } => {
            require_input(selected, input.input)?;
            let rank = match input.kind {
                KreaActivationInputKindV1::VectorBank { rank, .. } => usize::from(rank),
                KreaActivationInputKindV1::Donor { .. } => {
                    return Err(CoreError::invalid(
                        "Krea activation arithmetic",
                        "projection requires a vector bank",
                    ));
                }
            };
            for vector in values.chunks_exact(width).take(rank) {
                let coefficient = original
                    .iter()
                    .zip(vector)
                    .fold(0.0_f32, |sum, (value, basis)| value.mul_add(*basis, sum));
                let selected = if matches!(
                    operation.operator,
                    KreaActivationOperatorV1::OneSidedProjectionRemoval { .. }
                ) {
                    coefficient.max(0.0)
                } else {
                    coefficient
                };
                for (target, basis) in row.iter_mut().zip(vector) {
                    *target = (-strength * selected).mul_add(*basis, *target);
                }
            }
        }
    }
    if row.iter().any(|value| !value.is_finite()) {
        row.copy_from_slice(&original);
        return Err(CoreError::invalid(
            "Krea activation arithmetic",
            "operation produced a non-finite row",
        ));
    }
    Ok(())
}

fn require_input(selected: u16, actual: u16) -> Result<(), CoreError> {
    if selected != actual {
        return Err(CoreError::invalid(
            "Krea activation arithmetic",
            "operation references a different imported input",
        ));
    }
    Ok(())
}

fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>, CoreError> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return Err(CoreError::invalid(
            "Krea activation f32 bytes",
            "bytes must contain complete nonempty f32 elements",
        ));
    }
    let mut values = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let value = f32::from_bits(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        if !value.is_finite() {
            return Err(CoreError::invalid(
                "Krea activation f32 bytes",
                "every element must be finite",
            ));
        }
        values.push(value);
    }
    Ok(values)
}

fn validate_orthonormal(values: &[f32], width: u32, rank: u16) -> Result<(), CoreError> {
    const TOLERANCE: f64 = 1.0e-5;

    let width = usize::try_from(width)
        .map_err(|_| CoreError::invalid("Krea vector bank", "width exceeds usize"))?;
    let rank = usize::from(rank);
    if values.len() != width.saturating_mul(rank) {
        return Err(CoreError::invalid(
            "Krea vector bank",
            "orthonormal basis shape differs from its declaration",
        ));
    }
    for left in 0..rank {
        for right in left..rank {
            let dot = values[left * width..(left + 1) * width]
                .iter()
                .zip(&values[right * width..(right + 1) * width])
                .fold(0.0_f64, |sum, (left, right)| {
                    sum + f64::from(*left) * f64::from(*right)
                });
            let expected = if left == right { 1.0 } else { 0.0 };
            if !dot.is_finite() || (dot - expected).abs() > TOLERANCE {
                return Err(CoreError::invalid(
                    "Krea vector bank",
                    "declared orthonormal rows are not orthonormal",
                ));
            }
        }
    }
    Ok(())
}

fn strictly_ordered<T: Ord>(values: &[T]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn topology() -> KreaActivationTopologyV1 {
        KreaActivationTopologyV1 {
            model: Digest::of_bytes("test-krea-model", b"one"),
            backend: Digest::of_bytes("test-krea-backend", b"v6"),
            implementation: Digest::of_bytes("test-krea-topology", b"v1"),
            conditioner_layers: 12,
            transformer_blocks: 28,
            sites: vec![KreaActivationSiteV1 {
                site: 0,
                kind: KreaActivationSiteKindV1::TextResidual { block: 3 },
                width: 2,
                element_type: KreaActivationElementTypeV1::F32,
                layout: KreaActivationLayoutV1::FeatureTokenBatch,
                boundaries: vec![
                    KreaActivationBoundaryKindV1::PreDenoiser,
                    KreaActivationBoundaryKindV1::Transition,
                ],
                token_domains: vec![KreaTokenDomainV1 {
                    kind: KreaTokenDomainKindV1::Text,
                    maximum_tokens: 8,
                }],
                branches: vec![KreaCfgBranchV1::Conditional, KreaCfgBranchV1::Unconditional],
            }],
        }
    }

    fn bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect()
    }

    fn bank(values: &[f32], representation: KreaVectorRepresentationV1) -> KreaActivationInputV1 {
        let raw = bytes(values);
        KreaActivationInputV1 {
            input: 0,
            topology: topology().digest().unwrap(),
            kind: KreaActivationInputKindV1::VectorBank {
                site: 0,
                width: 2,
                rank: u16::try_from(values.len() / 2).unwrap(),
                representation,
            },
            source: KreaActivationInputSourceV1::Sealed {
                content: krea_activation_input_content_v1(&raw),
                bytes: u64::try_from(raw.len()).unwrap(),
            },
        }
    }

    fn operation(operator: KreaActivationOperatorV1, strength: f32) -> KreaActivationOperationV1 {
        KreaActivationOperationV1 {
            operation: 0,
            site: 0,
            tokens: KreaTokenSelectionV1::All {
                domain: KreaTokenDomainKindV1::Text,
            },
            boundary: KreaActivationBoundaryV1::Transitions {
                steps: StepSelector::All,
            },
            branch: KreaCfgBranchV1::Conditional,
            operator,
            strength_bits: strength.to_bits(),
        }
    }

    #[test]
    fn topology_rejects_out_of_range_and_duplicate_sites() {
        let mut invalid = topology();
        invalid.sites[0].kind = KreaActivationSiteKindV1::TextResidual { block: 28 };
        assert!(invalid.digest().is_err());
        let mut duplicate = topology();
        duplicate.sites.push(duplicate.sites[0].clone());
        assert!(duplicate.digest().is_err());
    }

    #[test]
    fn zero_strength_preserves_exact_bits() {
        let input = bank(&[1.0, 0.0], KreaVectorRepresentationV1::F32Rows);
        let raw = bytes(&[1.0, 0.0]);
        let mut row = [f32::from_bits(0x8000_0000), 3.5];
        let before = row.map(f32::to_bits);
        apply_krea_activation_operation_v1(
            &mut row,
            &operation(
                KreaActivationOperatorV1::ScaledVectorAdd {
                    input: 0,
                    vector: 0,
                },
                0.0,
            ),
            &input,
            &raw,
        )
        .unwrap();
        assert_eq!(row.map(f32::to_bits), before);
    }

    #[test]
    fn donor_and_projection_arithmetic_are_deterministic() {
        let donor_raw = bytes(&[4.0, 2.0]);
        let donor = KreaActivationInputV1 {
            input: 0,
            topology: topology().digest().unwrap(),
            kind: KreaActivationInputKindV1::Donor {
                site: 0,
                domain: KreaTokenDomainKindV1::Text,
                tokens: 1,
                branch: KreaCfgBranchV1::Conditional,
                width: 2,
            },
            source: KreaActivationInputSourceV1::Sealed {
                content: krea_activation_input_content_v1(&donor_raw),
                bytes: 8,
            },
        };
        let mut row = [0.0, 0.0];
        apply_krea_activation_operation_v1(
            &mut row,
            &operation(KreaActivationOperatorV1::DonorTransplant { input: 0 }, 0.5),
            &donor,
            &donor_raw,
        )
        .unwrap();
        assert_eq!(
            row.map(f32::to_bits),
            [2.0_f32.to_bits(), 1.0_f32.to_bits()]
        );

        let basis_raw = bytes(&[1.0, 0.0]);
        let basis = bank(&[1.0, 0.0], KreaVectorRepresentationV1::OrthonormalF32Rows);
        apply_krea_activation_operation_v1(
            &mut row,
            &operation(
                KreaActivationOperatorV1::OrthogonalProjectionRemoval { input: 0 },
                1.0,
            ),
            &basis,
            &basis_raw,
        )
        .unwrap();
        assert_eq!(
            row.map(f32::to_bits),
            [0.0_f32.to_bits(), 1.0_f32.to_bits()]
        );
    }

    #[test]
    fn plan_requires_bounded_complete_default_enabled_mechanics() {
        let topology = topology();
        let raw = bytes(&[1.0, 0.0]);
        let input = bank(&[1.0, 0.0], KreaVectorRepresentationV1::F32Rows);
        input.validate_bytes(&raw).unwrap();
        let plan = KreaActivationPlanV1 {
            topology: topology.digest().unwrap(),
            step_count: 2,
            inputs: vec![input],
            captures: vec![KreaActivationCaptureV1 {
                capture: 0,
                site: 0,
                tokens: KreaTokenSelectionV1::Ranges {
                    domain: KreaTokenDomainKindV1::Text,
                    ranges: vec![KreaTokenRangeV1 { start: 0, end: 1 }],
                },
                boundary: KreaActivationBoundaryV1::PreDenoiser,
                branch: KreaCfgBranchV1::Conditional,
                retention: KreaActivationCaptureRetentionV1::Digest,
                maximum_elements: 2,
                maximum_host_bytes: 0,
                maximum_device_bytes: 0,
            }],
            operations: vec![operation(
                KreaActivationOperatorV1::ScaledVectorAdd {
                    input: 0,
                    vector: 0,
                },
                0.0,
            )],
            maximum_host_bytes: 0,
            maximum_device_bytes: 8,
            maximum_applications: 2,
        };
        assert!(plan.digest_for(&topology).is_ok());
        let mut empty = plan;
        empty.captures.clear();
        empty.operations.clear();
        assert!(empty.digest_for(&topology).is_err());
    }

    #[test]
    fn resident_measurement_requires_one_copy_and_job_reuse() {
        let topology = topology();
        let input = bank(&[1.0, 0.0], KreaVectorRepresentationV1::F32Rows);
        let plan = KreaActivationPlanV1 {
            topology: topology.digest().unwrap(),
            step_count: 1,
            inputs: vec![input],
            captures: Vec::new(),
            operations: vec![operation(
                KreaActivationOperatorV1::ScaledVectorAdd {
                    input: 0,
                    vector: 0,
                },
                0.0,
            )],
            maximum_host_bytes: 0,
            maximum_device_bytes: 8,
            maximum_applications: 1,
        };
        let identity = Digest::of_bytes("test-activation-row", b"same");
        let receipt = KreaActivationReceiptV1 {
            plan: plan.digest_for(&topology).unwrap(),
            topology: topology.digest().unwrap(),
            backend: topology.backend.clone(),
            runtime_epoch: 1,
            terminal: KreaActivationTerminalV1::Completed,
            captures: Vec::new(),
            applications: vec![KreaActivationApplicationV1 {
                operation: 0,
                reached: 1,
                applied: 1,
                unchanged: 1,
                input: Some(identity.clone()),
                output: Some(identity),
            }],
            cleanup: KreaActivationCleanupDispositionV1::Confirmed,
        };
        receipt.digest_for(&plan, &topology).unwrap();
        let measurements = KreaActivationMeasurementsV1 {
            plan: receipt.plan.clone(),
            runtime_epoch: 1,
            peak_host_bytes: 0,
            peak_device_bytes: 8,
            inputs: vec![KreaActivationInputMeasurementV1 {
                input: 0,
                device: "vulkan0".to_owned(),
                resident_bytes: 8,
                host_to_device_transfers: 1,
                host_to_device_bytes: 8,
                jobs: 3,
            }],
        };
        measurements
            .validate_for(&plan, &topology, &receipt)
            .unwrap();
    }

    #[test]
    fn same_boundary_snapshot_is_a_typed_zero_copy_ssa_donor() {
        let topology = topology();
        let boundary = KreaActivationBoundaryV1::Transitions {
            steps: StepSelector::Exact { steps: vec![0] },
        };
        let capture = KreaActivationCaptureV1 {
            capture: 0,
            site: 0,
            tokens: KreaTokenSelectionV1::Ranges {
                domain: KreaTokenDomainKindV1::Text,
                ranges: vec![KreaTokenRangeV1 { start: 0, end: 1 }],
            },
            boundary: boundary.clone(),
            branch: KreaCfgBranchV1::Conditional,
            retention: KreaActivationCaptureRetentionV1::DeviceSnapshot,
            maximum_elements: 2,
            maximum_host_bytes: 0,
            maximum_device_bytes: 8,
        };
        let input = KreaActivationInputV1 {
            input: 0,
            topology: topology.digest().unwrap(),
            kind: KreaActivationInputKindV1::Donor {
                site: 0,
                domain: KreaTokenDomainKindV1::Text,
                tokens: 1,
                branch: KreaCfgBranchV1::Conditional,
                width: 2,
            },
            source: KreaActivationInputSourceV1::Capture { capture: 0 },
        };
        let plan = KreaActivationPlanV1 {
            topology: topology.digest().unwrap(),
            step_count: 2,
            inputs: vec![input],
            captures: vec![capture],
            operations: vec![KreaActivationOperationV1 {
                operation: 0,
                site: 0,
                tokens: KreaTokenSelectionV1::Ranges {
                    domain: KreaTokenDomainKindV1::Text,
                    ranges: vec![KreaTokenRangeV1 { start: 0, end: 1 }],
                },
                boundary,
                branch: KreaCfgBranchV1::Conditional,
                operator: KreaActivationOperatorV1::DonorTransplant { input: 0 },
                strength_bits: 1.0_f32.to_bits(),
            }],
            maximum_host_bytes: 0,
            maximum_device_bytes: 8,
            maximum_applications: 1,
        };
        assert!(plan.digest_for(&topology).is_ok());

        let mut wrong_boundary = plan;
        wrong_boundary.operations[0].boundary = KreaActivationBoundaryV1::Transitions {
            steps: StepSelector::Exact { steps: vec![1] },
        };
        assert!(wrong_boundary.digest_for(&topology).is_err());
    }

    #[test]
    fn clean_pretransition_cancellation_has_complete_terminal_evidence() {
        let topology = topology();
        let input = bank(&[1.0, 0.0], KreaVectorRepresentationV1::F32Rows);
        let plan = KreaActivationPlanV1 {
            topology: topology.digest().unwrap(),
            step_count: 1,
            inputs: vec![input],
            captures: Vec::new(),
            operations: vec![operation(
                KreaActivationOperatorV1::ScaledVectorAdd {
                    input: 0,
                    vector: 0,
                },
                1.0,
            )],
            maximum_host_bytes: 0,
            maximum_device_bytes: 8,
            maximum_applications: 1,
        };
        let receipt = KreaActivationReceiptV1 {
            plan: plan.digest_for(&topology).unwrap(),
            topology: topology.digest().unwrap(),
            backend: topology.backend.clone(),
            runtime_epoch: 1,
            terminal: KreaActivationTerminalV1::Cancelled {
                after_transition: None,
            },
            captures: Vec::new(),
            applications: vec![KreaActivationApplicationV1 {
                operation: 0,
                reached: 0,
                applied: 0,
                unchanged: 0,
                input: None,
                output: None,
            }],
            cleanup: KreaActivationCleanupDispositionV1::Confirmed,
        };
        receipt.digest_for(&plan, &topology).unwrap();
    }
}

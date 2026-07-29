// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact native evidence for resident model-block applications.

use logit_loom_diffusion::{
    Digest, ImageProgramPlanV1, ImageProgramReceiptV1, ImageProgramStageOperationV1, StepSelector,
    TensorSelector,
};
use serde::{Deserialize, Serialize};

use crate::{Error, Result, execution::InstalledModelBlockResidualScale};

/// Native graph-application accounting for one installed model-block operator.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelBlockApplicationV1 {
    /// Zero-based resident-program stage.
    pub stage: u16,
    /// Zero-based operator within the native stage's complete ordered stack.
    pub operator: u16,
    /// Main model-block count reported by the exact loaded Krea runner.
    pub loaded_model_blocks: u32,
    /// Zero-based model block selected by the operator.
    pub block: u32,
    /// Exact IEEE-754 residual-scale bits observed at the native boundary.
    pub residual_scale_bits: u32,
    /// Little-endian transition-selection bitmap, one bit per denoising
    /// transition and no trailing words.
    pub selected_transitions: Vec<u64>,
    /// Number of Krea graphs in which the selected block branch was reached.
    pub graph_applications: u32,
    /// Reached graphs that retained the ordinary block output.
    pub ordinary_graphs: u32,
    /// Reached graphs that bypassed the block before its forward call.
    pub bypassed_graphs: u32,
    /// Reached graphs that installed residual-scaling arithmetic.
    pub scaled_residual_graphs: u32,
}

/// Exact application evidence emitted beside one resident image-program
/// receipt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelBlockApplicationReceiptV1 {
    /// Exact program-plan identity.
    pub plan: Digest,
    /// Exact deterministic resident-program receipt identity.
    pub program_receipt: Digest,
    /// Exact backend build/runtime identity.
    pub backend: Digest,
    /// Runtime epoch used by native handles.
    pub runtime_epoch: u64,
    /// Number of completely published stages.
    pub completed_stages: u16,
    /// Canonical stage/operator-ordered native application records.
    pub applications: Vec<ModelBlockApplicationV1>,
}

impl ModelBlockApplicationReceiptV1 {
    /// Validates lineage and exact native application accounting.
    ///
    /// Every installed model-block operator in the completed stage prefix must
    /// have exactly one record. Transition bitmaps must equal the requested
    /// selector, and every reached graph must report exactly the graph branch
    /// implied by the installed residual scale.
    ///
    /// # Errors
    ///
    /// Returns the first lineage, topology, selector, or graph-accounting
    /// inconsistency.
    pub fn validate_for(
        &self,
        plan: &ImageProgramPlanV1,
        receipt: &ImageProgramReceiptV1,
    ) -> Result<()> {
        receipt
            .validate_for(plan)
            .map_err(logit_loom_diffusion::Error::from)?;
        let receipt_identity = receipt
            .digest_for(plan)
            .map_err(logit_loom_diffusion::Error::from)?;
        if self.plan != plan.digest().map_err(logit_loom_diffusion::Error::from)?
            || self.program_receipt != receipt_identity
            || self.backend != receipt.backend
            || self.runtime_epoch != receipt.runtime_epoch
            || self.completed_stages != receipt.completed_stages
        {
            return Err(Error::Incompatible(
                "model-block application lineage differs from the resident program".to_owned(),
            ));
        }

        let expected = expected_applications(plan, receipt.completed_stages)?;
        if self.applications.len() != expected.len() {
            return Err(Error::Incompatible(
                "model-block application count differs from the completed program prefix"
                    .to_owned(),
            ));
        }

        for (application, expected) in self.applications.iter().zip(expected) {
            validate_application(application, &expected)?;
        }
        Ok(())
    }

    /// Returns the identity of exact validated native application evidence.
    ///
    /// # Errors
    ///
    /// Returns an error when validation or deterministic serialization fails.
    pub fn digest_for(
        &self,
        plan: &ImageProgramPlanV1,
        receipt: &ImageProgramReceiptV1,
    ) -> Result<Digest> {
        self.validate_for(plan, receipt)?;
        Digest::of_serializable("sdcpp-resident-model-block-application-receipt-v1", self)
            .map_err(logit_loom_diffusion::Error::from)
            .map_err(Into::into)
    }
}

struct ExpectedApplication {
    stage: u16,
    operator: u16,
    installed: InstalledModelBlockResidualScale,
    step_count: usize,
}

fn expected_applications(
    plan: &ImageProgramPlanV1,
    completed_stages: u16,
) -> Result<Vec<ExpectedApplication>> {
    let mut expected = Vec::new();
    for stage in plan.stages.iter().take(usize::from(completed_stages)) {
        let ImageProgramStageOperationV1::Native { plan: native } = &stage.operation else {
            continue;
        };
        let step_count = native
            .schedule
            .as_ref()
            .map_or(0, logit_loom_diffusion::DiffusionSchedule::steps);
        for (operator, invocation) in native.operators.iter().enumerate() {
            if !matches!(invocation.selector, TensorSelector::ModelBlock { .. }) {
                continue;
            }
            expected.push(ExpectedApplication {
                stage: stage.stage,
                operator: u16::try_from(operator).map_err(|_| {
                    Error::Incompatible(
                        "model-block operator index exceeds receipt bounds".to_owned(),
                    )
                })?,
                installed: InstalledModelBlockResidualScale::from_invocation(invocation)?,
                step_count,
            });
        }
    }
    Ok(expected)
}

fn validate_application(
    application: &ModelBlockApplicationV1,
    expected: &ExpectedApplication,
) -> Result<()> {
    let expected_transitions = transition_bitmap(&expected.installed.steps, expected.step_count)?;
    let selected = expected_transitions.iter().try_fold(0_u32, |count, word| {
        count.checked_add(word.count_ones()).ok_or_else(|| {
            Error::Incompatible("model-block selected-transition count overflowed".to_owned())
        })
    })?;
    let action_total = application
        .ordinary_graphs
        .checked_add(application.bypassed_graphs)
        .and_then(|total| total.checked_add(application.scaled_residual_graphs))
        .ok_or_else(|| Error::Incompatible("model-block graph accounting overflowed".to_owned()))?;
    let scale_bits = expected.installed.scale.to_bits();
    let zero_scale = matches!(scale_bits, 0 | 0x8000_0000);
    let action_matches = if zero_scale {
        application.bypassed_graphs == application.graph_applications
            && application.ordinary_graphs == 0
            && application.scaled_residual_graphs == 0
    } else if scale_bits == 1.0_f32.to_bits() {
        application.ordinary_graphs == application.graph_applications
            && application.bypassed_graphs == 0
            && application.scaled_residual_graphs == 0
    } else {
        application.scaled_residual_graphs == application.graph_applications
            && application.ordinary_graphs == 0
            && application.bypassed_graphs == 0
    };
    if application.stage != expected.stage
        || application.operator != expected.operator
        || application.loaded_model_blocks == 0
        || application.block >= application.loaded_model_blocks
        || application.block != expected.installed.block
        || application.residual_scale_bits != scale_bits
        || application.selected_transitions != expected_transitions
        || selected == 0
        || application.graph_applications < selected
        || action_total != application.graph_applications
        || !action_matches
    {
        return Err(Error::Incompatible(
            "model-block native application evidence differs from its installed operator"
                .to_owned(),
        ));
    }
    Ok(())
}

fn transition_bitmap(steps: &StepSelector, step_count: usize) -> Result<Vec<u64>> {
    let word_count = step_count
        .checked_add(63)
        .ok_or_else(|| Error::Incompatible("model-block transition count overflowed".to_owned()))?
        / 64;
    let mut words = vec![0_u64; word_count];
    match steps {
        StepSelector::All => {
            for step in 0..step_count {
                words[step / 64] |= 1_u64 << (step % 64);
            }
        }
        StepSelector::Exact { steps } => {
            for step in steps {
                let step = usize::try_from(*step).map_err(|_| {
                    Error::Incompatible("model-block transition exceeds usize".to_owned())
                })?;
                if step >= step_count {
                    return Err(Error::Incompatible(
                        "model-block transition exceeds the native schedule".to_owned(),
                    ));
                }
                words[step / 64] |= 1_u64 << (step % 64);
            }
        }
    }
    Ok(words)
}

#[cfg(test)]
mod tests {
    use logit_loom_diffusion::{
        DiffusionSchedule, ImageBufferRole, ImageCleanupPolicy, ImageOperation, ImageOutputFormat,
        ImageProgramCleanupDispositionV1, ImageProgramInputBindingV1, ImageProgramInputV1,
        ImageProgramNativeOutputRoleV1, ImageProgramNativeOutputV1, ImageProgramNativeStageV1,
        ImageProgramOutputReceiptV1, ImageProgramOutputRouteV1, ImageProgramOutputSourceV1,
        ImageProgramStageReceiptV1, ImageProgramStageV1, ImageProgramTerminalV1,
        ImageProgramValueReceiptV1, ImageProgramValueSpecV1, ImageProgramValueV1,
        OperatorInvocation, SeedSelection,
    };
    use logit_loom_executor::BufferSpec;

    use super::*;
    use crate::{ModelBlockResidualScaleControlV1, model_block_residual_scale_schema_v1};

    fn buffer(label: &str, bytes: u64, media_type: &str) -> BufferSpec {
        BufferSpec::new(
            Digest::of_bytes("application-test-buffer", label.as_bytes()),
            bytes,
            media_type,
        )
        .unwrap()
    }

    #[allow(clippy::too_many_lines)]
    fn fixture(
        scale: f32,
        steps: StepSelector,
    ) -> (
        ImageProgramPlanV1,
        ImageProgramReceiptV1,
        ModelBlockApplicationReceiptV1,
    ) {
        let selector = TensorSelector::ModelBlock {
            component: "krea2".to_owned(),
            block: 9,
            site: "residual".to_owned(),
        };
        let control = ModelBlockResidualScaleControlV1::new(scale, 1.0).unwrap();
        let operator = OperatorInvocation {
            schema: model_block_residual_scale_schema_v1(),
            implementation: control.implementation_for(&selector, &steps).unwrap(),
            selector,
            steps,
            controls: control.to_control_bytes(),
        };
        let stage = ImageProgramStageV1 {
            stage: 0,
            operation: ImageProgramStageOperationV1::Native {
                plan: Box::new(ImageProgramNativeStageV1 {
                    profile: Digest::of_bytes("application-test-profile", b"krea"),
                    load: Digest::of_bytes("application-test-load", b"krea"),
                    operation: ImageOperation::TextToImage,
                    width: 1,
                    height: 1,
                    output_format: ImageOutputFormat::Rgb8,
                    seed: SeedSelection::Fixed { seed: 7 },
                    rng: Digest::of_bytes("application-test-rng", b"v1"),
                    placement: Digest::of_bytes("application-test-placement", b"device"),
                    schedule: Some(
                        DiffusionSchedule::new(
                            Digest::of_bytes("application-test-schedule", b"v1"),
                            vec![1.0, 0.7, 0.3, 0.0],
                        )
                        .unwrap(),
                    ),
                    guidance_scale_bits: 1.0_f32.to_bits(),
                    strength_bits: 1.0_f32.to_bits(),
                    inputs: vec![ImageProgramInputBindingV1 {
                        role: ImageBufferRole::PositiveConditioning,
                        value: 0,
                    }],
                    loras: Vec::new(),
                    operators: vec![operator],
                    observations: Vec::new(),
                    checkpoint_restore_at_step: None,
                    checkpoint_after_step: None,
                    outputs: vec![ImageProgramNativeOutputV1 {
                        role: ImageProgramNativeOutputRoleV1::Primary,
                        value: 1,
                    }],
                }),
            },
        };
        let plan = ImageProgramPlanV1 {
            values: vec![
                ImageProgramValueV1 {
                    value: 0,
                    spec: ImageProgramValueSpecV1::Utf8 { maximum_bytes: 1 },
                },
                ImageProgramValueV1 {
                    value: 1,
                    spec: ImageProgramValueSpecV1::Rgb8 {
                        width: 1,
                        height: 1,
                    },
                },
            ],
            inputs: vec![ImageProgramInputV1 {
                value: 0,
                buffer: buffer("prompt", 1, "text/plain"),
            }],
            stages: vec![stage],
            outputs: vec![
                ImageProgramOutputRouteV1 {
                    route: 0,
                    source: ImageProgramOutputSourceV1::Value { value: 1 },
                    buffer: buffer("image", 3, "image/rgb"),
                },
                ImageProgramOutputRouteV1 {
                    route: 1,
                    source: ImageProgramOutputSourceV1::ProgramReceipt,
                    buffer: buffer("receipt", 16_384, "application/json"),
                },
            ],
            cleanup: ImageCleanupPolicy::ClearSession,
        };
        let content = Digest::of_bytes("application-test-output", b"rgb");
        let backend = Digest::of_bytes("application-test-backend", b"v5");
        let receipt = ImageProgramReceiptV1 {
            plan: plan.digest().unwrap(),
            backend: backend.clone(),
            runtime_epoch: 4,
            completed_stages: 1,
            stages: vec![ImageProgramStageReceiptV1 {
                stage: 0,
                operation: plan.stages[0].operation.digest().unwrap(),
                outputs: vec![ImageProgramValueReceiptV1 {
                    value: 1,
                    content: content.clone(),
                    bytes: 3,
                }],
                observations: Vec::new(),
            }],
            outputs: vec![
                ImageProgramOutputReceiptV1 {
                    route: 0,
                    allocation: plan.outputs[0].buffer.identity.clone(),
                    content: Some(content),
                    bytes_written: 3,
                },
                ImageProgramOutputReceiptV1 {
                    route: 1,
                    allocation: plan.outputs[1].buffer.identity.clone(),
                    content: None,
                    bytes_written: 256,
                },
            ],
            terminal: ImageProgramTerminalV1::Completed,
            cleanup: ImageProgramCleanupDispositionV1::Confirmed { cleared_epoch: 4 },
        };
        receipt.validate_for(&plan).unwrap();
        let selected_transitions = transition_bitmap(
            &match &plan.stages[0].operation {
                ImageProgramStageOperationV1::Native { plan } => plan.operators[0].steps.clone(),
                _ => unreachable!(),
            },
            3,
        )
        .unwrap();
        let graph_applications = selected_transitions
            .iter()
            .map(|word| word.count_ones())
            .sum::<u32>();
        let scale_bits = scale.to_bits();
        let (ordinary_graphs, bypassed_graphs, scaled_residual_graphs) =
            if matches!(scale_bits, 0 | 0x8000_0000) {
                (0, graph_applications, 0)
            } else if scale_bits == 1.0_f32.to_bits() {
                (graph_applications, 0, 0)
            } else {
                (0, 0, graph_applications)
            };
        let applications = ModelBlockApplicationReceiptV1 {
            plan: receipt.plan.clone(),
            program_receipt: receipt.digest_for(&plan).unwrap(),
            backend,
            runtime_epoch: 4,
            completed_stages: 1,
            applications: vec![ModelBlockApplicationV1 {
                stage: 0,
                operator: 0,
                loaded_model_blocks: 28,
                block: 9,
                residual_scale_bits: scale.to_bits(),
                selected_transitions,
                graph_applications,
                ordinary_graphs,
                bypassed_graphs,
                scaled_residual_graphs,
            }],
        };
        (plan, receipt, applications)
    }

    #[test]
    fn exact_transition_bypass_is_bound_to_the_program_receipt() {
        let (plan, receipt, applications) = fixture(0.0, StepSelector::Exact { steps: vec![1] });
        applications.validate_for(&plan, &receipt).unwrap();
        assert_eq!(applications.applications[0].selected_transitions, vec![2]);
        assert_eq!(applications.applications[0].bypassed_graphs, 1);
        assert!(applications.digest_for(&plan, &receipt).is_ok());
    }

    #[test]
    fn sham_and_scaled_actions_require_the_matching_native_branch() {
        let (plan, receipt, sham) = fixture(1.0, StepSelector::All);
        sham.validate_for(&plan, &receipt).unwrap();
        assert_eq!(sham.applications[0].selected_transitions, vec![7]);
        assert_eq!(sham.applications[0].ordinary_graphs, 3);

        let (plan, receipt, scaled) = fixture(0.5, StepSelector::Exact { steps: vec![0, 2] });
        scaled.validate_for(&plan, &receipt).unwrap();
        assert_eq!(scaled.applications[0].selected_transitions, vec![5]);
        assert_eq!(scaled.applications[0].scaled_residual_graphs, 2);
    }

    #[test]
    fn wrong_transition_or_graph_action_is_rejected() {
        let (plan, receipt, mut applications) =
            fixture(0.0, StepSelector::Exact { steps: vec![1] });
        applications.applications[0].selected_transitions[0] = 1;
        assert!(applications.validate_for(&plan, &receipt).is_err());

        let (_, _, mut applications) = fixture(0.0, StepSelector::Exact { steps: vec![1] });
        applications.applications[0].bypassed_graphs = 0;
        applications.applications[0].ordinary_graphs = 1;
        assert!(applications.validate_for(&plan, &receipt).is_err());
    }
}

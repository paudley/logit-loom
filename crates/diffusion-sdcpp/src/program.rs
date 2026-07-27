// SPDX-License-Identifier: MIT OR Apache-2.0

use logit_loom_diffusion::{
    ControlFlow, DiffusionPlan, Digest, ObserverReceipt, ObserverSet, Pipeline, PipelineReceipt,
    StepContext,
};

use crate::{DiffusionCheckpoint, Error, Result, StepProgram};

/// Deferred construction of a pipeline bound to the native execution plan.
pub type PipelineFactory<'a> =
    Box<dyn FnOnce(&DiffusionPlan) -> std::result::Result<Pipeline, String> + 'a>;
/// Deferred construction of observers bound to the native execution plan.
pub type ObserverFactory<'a> =
    Box<dyn FnOnce(&DiffusionPlan) -> std::result::Result<ObserverSet, String> + 'a>;

/// Adapter glue for backend-neutral transactional pipelines and observers.
pub struct PipelineProgram<'a> {
    implementation: Digest,
    pipeline_factory: Option<PipelineFactory<'a>>,
    observer_factory: Option<ObserverFactory<'a>>,
    pipeline: Option<Pipeline>,
    observers: Option<ObserverSet>,
    intervention_step: Option<u32>,
}

impl std::fmt::Debug for PipelineProgram<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PipelineProgram")
            .field("implementation", &self.implementation)
            .field("pipeline", &self.pipeline)
            .field("observers", &self.observers)
            .field("intervention_step", &self.intervention_step)
            .finish_non_exhaustive()
    }
}

impl<'a> PipelineProgram<'a> {
    /// Creates deferred backend-neutral hooks.
    ///
    /// Factories run once after native conditioning establishes the exact
    /// [`DiffusionPlan`] and before the first state is exposed.
    ///
    /// # Errors
    ///
    /// Returns an error unless at least one factory is supplied.
    pub fn new(
        implementation: &Digest,
        pipeline_factory: Option<PipelineFactory<'a>>,
        observer_factory: Option<ObserverFactory<'a>>,
    ) -> Result<Self> {
        if pipeline_factory.is_none() && observer_factory.is_none() {
            return Err(Error::Invalid(
                "pipeline program requires a pipeline or observer factory".to_owned(),
            ));
        }
        let wrapper_identity =
            Digest::of_serializable("sdcpp-pipeline-program-v1", &(implementation, "every-step"))
                .map_err(logit_loom_diffusion::Error::from)?;
        Ok(Self {
            implementation: wrapper_identity,
            pipeline_factory,
            observer_factory,
            pipeline: None,
            observers: None,
            intervention_step: None,
        })
    }

    /// Creates a pipeline that intervenes at exactly one post-step boundary.
    ///
    /// The observer factory, when present, still sees every committed step.
    ///
    /// # Errors
    ///
    /// Returns an identity serialization error. The selected step is checked
    /// against the exact native schedule during initialization.
    pub fn at_step(
        implementation: &Digest,
        step_index: u32,
        pipeline_factory: PipelineFactory<'a>,
        observer_factory: Option<ObserverFactory<'a>>,
    ) -> Result<Self> {
        let wrapper_identity = Digest::of_serializable(
            "sdcpp-pipeline-program-v1",
            &(implementation, "selected-step", step_index),
        )
        .map_err(logit_loom_diffusion::Error::from)?;
        Ok(Self {
            implementation: wrapper_identity,
            pipeline_factory: Some(pipeline_factory),
            observer_factory,
            pipeline: None,
            observers: None,
            intervention_step: Some(step_index),
        })
    }

    /// Returns current pipeline accounting after initialization.
    pub fn pipeline_receipt(&self) -> Option<&PipelineReceipt> {
        self.pipeline.as_ref().map(Pipeline::receipt)
    }

    /// Returns current observer accounting after initialization.
    pub fn observer_receipts(&self) -> Option<&[ObserverReceipt]> {
        self.observers.as_ref().map(ObserverSet::receipts)
    }
}

impl StepProgram for PipelineProgram<'_> {
    fn implementation(&self) -> &Digest {
        &self.implementation
    }

    fn begin(&mut self, plan: &DiffusionPlan) -> std::result::Result<(), String> {
        if self.intervention_step.is_some_and(|step| {
            usize::try_from(step).map_or(true, |step| step >= plan.schedule.steps())
        }) {
            return Err("selected intervention step is outside the schedule".to_owned());
        }
        if let Some(factory) = self.pipeline_factory.take() {
            let mut pipeline = factory(plan)?;
            pipeline.begin().map_err(|error| error.to_string())?;
            self.pipeline = Some(pipeline);
        }
        if let Some(factory) = self.observer_factory.take() {
            self.observers = Some(factory(plan)?);
        }
        Ok(())
    }

    fn intervene(
        &mut self,
        context: &StepContext,
        state: &mut [f32],
    ) -> std::result::Result<(), String> {
        if self
            .intervention_step
            .is_none_or(|step| step == context.step_index)
            && let Some(pipeline) = &mut self.pipeline
        {
            pipeline
                .apply(context, state)
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    fn observe(
        &mut self,
        context: &StepContext,
        state: &[f32],
    ) -> std::result::Result<ControlFlow, String> {
        self.observers
            .as_mut()
            .map_or(Ok(ControlFlow::Continue), |observers| {
                observers
                    .observe(context, state)
                    .map_err(|error| error.to_string())
            })
    }
}

enum ForkMode {
    Capture { step_index: u32 },
    Replay { checkpoint: DiffusionCheckpoint },
}

/// Captures or restores one exact deterministic-prefix branch boundary.
///
/// Capture/restore occurs before the delegated program's intervention at the
/// selected post-Euler step. Replay first requires the recomputed prefix state
/// identity to equal the checkpoint, then copies its exact bytes and continues.
pub struct ForkProgram<P> {
    implementation: Digest,
    backend: Digest,
    mode: ForkMode,
    delegate: P,
    plan: Option<DiffusionPlan>,
    captured: Option<DiffusionCheckpoint>,
    applied: bool,
}

impl<P> std::fmt::Debug for ForkProgram<P> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ForkProgram")
            .field("implementation", &self.implementation)
            .field("backend", &self.backend)
            .field("applied", &self.applied)
            .finish_non_exhaustive()
    }
}

impl<P: StepProgram> ForkProgram<P> {
    /// Creates a program that captures one post-step checkpoint.
    ///
    /// # Errors
    ///
    /// Returns an identity serialization error.
    pub fn capture(step_index: u32, backend: Digest, delegate: P) -> Result<Self> {
        let implementation = Digest::of_serializable(
            "sdcpp-fork-program-v1",
            &("capture", step_index, &backend, delegate.implementation()),
        )
        .map_err(logit_loom_diffusion::Error::from)?;
        Ok(Self {
            implementation,
            backend,
            mode: ForkMode::Capture { step_index },
            delegate,
            plan: None,
            captured: None,
            applied: false,
        })
    }

    /// Creates a program that verifies and restores one checkpoint boundary.
    ///
    /// # Errors
    ///
    /// Returns an identity serialization error.
    pub fn replay(checkpoint: DiffusionCheckpoint, backend: Digest, delegate: P) -> Result<Self> {
        let implementation = Digest::of_serializable(
            "sdcpp-fork-program-v1",
            &(
                "replay",
                checkpoint.receipt(),
                &backend,
                delegate.implementation(),
            ),
        )
        .map_err(logit_loom_diffusion::Error::from)?;
        Ok(Self {
            implementation,
            backend,
            mode: ForkMode::Replay { checkpoint },
            delegate,
            plan: None,
            captured: None,
            applied: false,
        })
    }

    /// Returns whether the selected checkpoint step was reached.
    pub const fn applied(&self) -> bool {
        self.applied
    }

    /// Removes the checkpoint produced by capture mode.
    pub fn take_checkpoint(&mut self) -> Option<DiffusionCheckpoint> {
        self.captured.take()
    }

    /// Returns the delegated program.
    pub const fn delegate(&self) -> &P {
        &self.delegate
    }
}

impl<P: StepProgram> StepProgram for ForkProgram<P> {
    fn implementation(&self) -> &Digest {
        &self.implementation
    }

    fn begin(&mut self, plan: &DiffusionPlan) -> std::result::Result<(), String> {
        if let ForkMode::Replay { checkpoint } = &self.mode {
            checkpoint
                .receipt()
                .validate_for(plan)
                .map_err(|error| error.to_string())?;
            if checkpoint.receipt().backend != self.backend {
                return Err("checkpoint native backend identity differs".to_owned());
            }
        }
        self.delegate.begin(plan)?;
        self.plan = Some(plan.clone());
        Ok(())
    }

    fn intervene(
        &mut self,
        context: &StepContext,
        state: &mut [f32],
    ) -> std::result::Result<(), String> {
        let plan = self
            .plan
            .as_ref()
            .ok_or_else(|| "fork program was not initialized".to_owned())?;
        match &self.mode {
            ForkMode::Capture { step_index } if *step_index == context.step_index => {
                if self.applied {
                    return Err("checkpoint capture step repeated".to_owned());
                }
                self.captured = Some(
                    DiffusionCheckpoint::capture(plan, &self.backend, context, state)
                        .map_err(|error| error.to_string())?,
                );
                self.applied = true;
            }
            ForkMode::Replay { checkpoint }
                if checkpoint.receipt().next_step == context.step_index.saturating_add(1) =>
            {
                if self.applied {
                    return Err("checkpoint replay step repeated".to_owned());
                }
                let current = checkpoint_state_digest(state);
                if current != checkpoint.receipt().state {
                    return Err("deterministic prefix state differs from the checkpoint".to_owned());
                }
                checkpoint
                    .restore(plan, &self.backend, context, state)
                    .map_err(|error| error.to_string())?;
                self.applied = true;
            }
            _ => {}
        }
        self.delegate.intervene(context, state)
    }

    fn observe(
        &mut self,
        context: &StepContext,
        state: &[f32],
    ) -> std::result::Result<ControlFlow, String> {
        self.delegate.observe(context, state)
    }
}

fn checkpoint_state_digest(state: &[f32]) -> Digest {
    let mut bytes = Vec::with_capacity(state.len().saturating_mul(4));
    for value in state {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    Digest::of_bytes("sdcpp-checkpoint-f32-le-v1", &bytes)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use logit_loom_diffusion::{
        ChannelBias, DiffusionSchedule, TensorDType, TensorLayout, TensorSpec,
    };

    use crate::NoopProgram;

    use super::*;

    fn fixture() -> (DiffusionPlan, StepContext) {
        let tensor = TensorSpec::new(
            vec![2, 2],
            TensorDType::F32,
            TensorLayout::DimensionZeroFastest,
            "host-f32:test",
        )
        .expect("valid tensor");
        let schedule =
            DiffusionSchedule::new(Digest::of_bytes("schedule", b"one"), vec![1.0, 0.5, 0.0])
                .expect("valid schedule");
        let mut components = BTreeMap::new();
        components.insert("model".to_owned(), Digest::of_bytes("model", b"one"));
        let plan = DiffusionPlan::new(
            components,
            Digest::of_bytes("condition", b"one"),
            Digest::of_bytes("rng", b"one"),
            7,
            tensor,
            schedule,
        )
        .expect("valid plan");
        let context = StepContext::for_plan(&plan, 0).expect("valid step");
        (plan, context)
    }

    #[test]
    fn capture_and_replay_require_exact_prefix() {
        let (plan, context) = fixture();
        let backend = Digest::of_bytes("backend", b"one");
        let mut capture = ForkProgram::capture(0, backend.clone(), NoopProgram::default())
            .expect("capture program");
        capture.begin(&plan).expect("begin capture");
        let mut state = vec![1.0, 2.0, 3.0, 4.0];
        capture
            .intervene(&context, &mut state)
            .expect("capture state");
        let checkpoint = capture.take_checkpoint().expect("checkpoint");

        let mut replay = ForkProgram::replay(checkpoint, backend, NoopProgram::default())
            .expect("replay program");
        replay.begin(&plan).expect("begin replay");
        let mut same = state.clone();
        replay
            .intervene(&context, &mut same)
            .expect("matching prefix");
        assert!(replay.applied());
    }

    #[test]
    fn replay_rejects_different_prefix_without_writeback() {
        let (plan, context) = fixture();
        let backend = Digest::of_bytes("backend", b"one");
        let mut capture = ForkProgram::capture(0, backend.clone(), NoopProgram::default())
            .expect("capture program");
        capture.begin(&plan).expect("begin capture");
        let mut original = vec![1.0, 2.0, 3.0, 4.0];
        capture
            .intervene(&context, &mut original)
            .expect("capture state");
        let checkpoint = capture.take_checkpoint().expect("checkpoint");

        let mut replay = ForkProgram::replay(checkpoint, backend, NoopProgram::default())
            .expect("replay program");
        replay.begin(&plan).expect("begin replay");
        let mut changed = vec![9.0, 2.0, 3.0, 4.0];
        let before = changed.clone();
        assert!(replay.intervene(&context, &mut changed).is_err());
        assert_eq!(changed, before);
    }

    #[test]
    fn selected_pipeline_runs_only_at_its_declared_step() {
        let (plan, first_context) = fixture();
        let second_context = StepContext::for_plan(&plan, 1).expect("second step");
        let identity = Digest::of_bytes("test-program", b"selected-bias");
        let mut program = PipelineProgram::at_step(
            &identity,
            1,
            Box::new(|plan| {
                let bias = ChannelBias::new(&plan.tensor, 0, 0, 0.25, 0.5)
                    .map_err(|error| error.to_string())?;
                Pipeline::new(
                    plan.digest().map_err(|error| error.to_string())?,
                    plan.tensor.clone(),
                    vec![Box::new(bias)],
                )
                .map_err(|error| error.to_string())
            }),
            None,
        )
        .expect("selected program");
        program.begin(&plan).expect("begin");

        let mut state = vec![0.0; 4];
        program
            .intervene(&first_context, &mut state)
            .expect("first step");
        assert_eq!(state, vec![0.0; 4]);

        program
            .intervene(&second_context, &mut state)
            .expect("selected step");
        assert_eq!(state, vec![0.25, 0.0, 0.25, 0.0]);
        assert_eq!(
            program
                .pipeline_receipt()
                .expect("pipeline receipt")
                .invocations,
            1
        );
    }
}

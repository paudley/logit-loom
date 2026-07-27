// SPDX-License-Identifier: MIT OR Apache-2.0

//! Ordered transactional intervention execution.

use std::{
    any::Any,
    panic::{AssertUnwindSafe, catch_unwind},
};

use logit_loom_core::{CoreError, Digest};
use serde::Serialize;

use crate::{
    Error, InterventionFailure, InterventionSpec, PipelineReceipt, PipelineSpec, Result,
    StageReceipt, StepContext, TensorSpec,
};

/// One synchronous state intervention.
///
/// Implementations receive a private transactional copy. Returning an error,
/// unwinding, or producing a non-finite value prevents every change from being
/// committed to the adapter-owned state.
pub trait Intervention {
    /// Returns the exact implementation and configuration contract.
    fn specification(&self) -> &InterventionSpec;

    /// Resets per-run implementation state.
    ///
    /// # Errors
    ///
    /// Returns an error when the intervention cannot reset its internal state.
    fn reset(&mut self) -> std::result::Result<(), String> {
        Ok(())
    }

    /// Applies one post-step intervention to contiguous `f32` state.
    ///
    /// # Errors
    ///
    /// Returns an error when the intervention rejects the step or cannot apply
    /// its mutation. The enclosing pipeline discards all staged mutations when
    /// any intervention returns an error.
    fn apply(
        &mut self,
        context: &StepContext,
        state: &mut [f32],
    ) -> std::result::Result<(), String>;
}

/// An ordered transactional intervention pipeline.
pub struct Pipeline {
    expected_plan: Digest,
    tensor: TensorSpec,
    stages: Vec<Box<dyn Intervention>>,
    receipt: PipelineReceipt,
}

impl std::fmt::Debug for Pipeline {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Pipeline")
            .field("expected_plan", &self.expected_plan)
            .field("tensor", &self.tensor)
            .field("receipt", &self.receipt)
            .finish_non_exhaustive()
    }
}

impl Pipeline {
    /// Creates a pipeline bound to one plan and tensor identity.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty, oversized, or incompatible stage set.
    pub fn new(
        expected_plan: Digest,
        tensor: TensorSpec,
        stages: Vec<Box<dyn Intervention>>,
    ) -> Result<Self> {
        tensor.validate()?;
        let specifications = stages
            .iter()
            .map(|stage| stage.specification().clone())
            .collect::<Vec<_>>();
        let specification = PipelineSpec {
            stages: specifications,
        };
        specification.validate()?;
        let tensor_digest = tensor.digest()?;
        if specification
            .stages
            .iter()
            .any(|stage| stage.tensor != tensor_digest)
        {
            return Err(CoreError::invalid(
                "diffusion pipeline",
                "stage tensor identity differs from the pipeline tensor",
            )
            .into());
        }
        let pipeline = specification.digest()?;
        let receipts = specification
            .stages
            .iter()
            .cloned()
            .map(|stage| StageReceipt {
                specification: stage,
                resets: 0,
                invocations: 0,
                elements_seen: 0,
                elements_changed: 0,
                failure: None,
            })
            .collect();
        Ok(Self {
            expected_plan,
            tensor,
            stages,
            receipt: PipelineReceipt {
                specification,
                pipeline,
                begins: 0,
                invocations: 0,
                elements_copied: 0,
                elements_committed: 0,
                failed_stage: None,
                stages: receipts,
            },
        })
    }

    /// Resets every stage in order before the first state boundary.
    ///
    /// # Errors
    ///
    /// Returns a contained callback error or panic. A pipeline begins once.
    pub fn begin(&mut self) -> Result<()> {
        if self.receipt.begins != 0 {
            return Err(Error::Incompatible(
                "an intervention pipeline may begin only once".to_owned(),
            ));
        }
        self.receipt.begins = 1;
        for (index, stage) in self.stages.iter_mut().enumerate() {
            let outcome = catch_unwind(AssertUnwindSafe(|| stage.reset()));
            self.receipt.stages[index].resets = 1;
            match callback_outcome(outcome) {
                Ok(()) => {}
                Err(failure) => {
                    self.record_failure(index, failure.clone())?;
                    return Err(Error::Intervention {
                        stage: index,
                        message: failure.message,
                    });
                }
            }
        }
        Ok(())
    }

    /// Applies every stage in order and commits only after all succeed.
    ///
    /// # Errors
    ///
    /// Returns a compatibility, bound, callback, panic, or finite-value error.
    /// The caller's slice is unchanged on every error path.
    pub fn apply(&mut self, context: &StepContext, state: &mut [f32]) -> Result<()> {
        self.validate_call(context, state)?;
        let elements = u64::try_from(state.len())
            .map_err(|_| Error::Incompatible("state length exceeds u64".to_owned()))?;
        self.receipt.elements_copied = self
            .receipt
            .elements_copied
            .checked_add(elements)
            .ok_or_else(|| {
                Error::Incompatible("copied-element accounting overflowed".to_owned())
            })?;
        let mut working = state.to_vec();

        for index in 0..self.stages.len() {
            let before = working
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>();
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                self.stages[index].apply(context, &mut working)
            }));
            let stage_receipt = &mut self.receipt.stages[index];
            stage_receipt.invocations = stage_receipt
                .invocations
                .checked_add(1)
                .ok_or_else(|| Error::Incompatible("stage invocation overflowed".to_owned()))?;
            stage_receipt.elements_seen = stage_receipt
                .elements_seen
                .checked_add(elements)
                .ok_or_else(|| {
                    Error::Incompatible("stage element accounting overflowed".to_owned())
                })?;

            if let Err(failure) = callback_outcome(outcome) {
                self.record_failure(index, failure.clone())?;
                return Err(Error::Intervention {
                    stage: index,
                    message: failure.message,
                });
            }
            if working.iter().any(|value| !value.is_finite()) {
                let failure =
                    InterventionFailure::new(false, "stage produced a non-finite state value");
                self.record_failure(index, failure.clone())?;
                return Err(Error::Intervention {
                    stage: index,
                    message: failure.message,
                });
            }
            let changed = before
                .iter()
                .zip(&working)
                .filter(|(old, new)| **old != new.to_bits())
                .count();
            stage_receipt.elements_changed = stage_receipt
                .elements_changed
                .checked_add(u64::try_from(changed).map_err(|_| {
                    Error::Incompatible("changed-element count exceeds u64".to_owned())
                })?)
                .ok_or_else(|| {
                    Error::Incompatible("changed-element accounting overflowed".to_owned())
                })?;
        }

        state.copy_from_slice(&working);
        self.receipt.invocations = self
            .receipt
            .invocations
            .checked_add(1)
            .ok_or_else(|| Error::Incompatible("pipeline invocation overflowed".to_owned()))?;
        self.receipt.elements_committed = self
            .receipt
            .elements_committed
            .checked_add(elements)
            .ok_or_else(|| {
                Error::Incompatible("committed-element accounting overflowed".to_owned())
            })?;
        Ok(())
    }

    /// Returns current serializable accounting.
    pub const fn receipt(&self) -> &PipelineReceipt {
        &self.receipt
    }

    fn validate_call(&self, context: &StepContext, state: &[f32]) -> Result<()> {
        if self.receipt.begins != 1 {
            return Err(Error::Incompatible(
                "intervention pipeline has not begun".to_owned(),
            ));
        }
        if self.receipt.failed_stage.is_some() {
            return Err(Error::Incompatible(
                "intervention pipeline is failed and cannot continue".to_owned(),
            ));
        }
        if context.plan != self.expected_plan || context.tensor != self.tensor {
            return Err(Error::Incompatible(
                "step plan, shape, dtype, layout, or device differs".to_owned(),
            ));
        }
        let expected = usize::try_from(self.tensor.elements()?)
            .map_err(|_| Error::Incompatible("tensor element count exceeds usize".to_owned()))?;
        if state.len() != expected {
            return Err(Error::Incompatible(format!(
                "state contains {} elements; expected {expected}",
                state.len()
            )));
        }
        if state.iter().any(|value| !value.is_finite()) {
            return Err(Error::Incompatible(
                "input state contains a non-finite value".to_owned(),
            ));
        }
        for (stage, receipt) in self.stages.iter().zip(&self.receipt.stages) {
            if receipt.invocations >= stage.specification().max_invocations {
                return Err(Error::Incompatible(
                    "intervention stage invocation bound is exhausted".to_owned(),
                ));
            }
        }
        Ok(())
    }

    fn record_failure(&mut self, index: usize, failure: InterventionFailure) -> Result<()> {
        self.receipt.failed_stage = Some(
            u32::try_from(index)
                .map_err(|_| Error::Incompatible("stage index exceeds u32".to_owned()))?,
        );
        self.receipt.stages[index].failure = Some(failure);
        Ok(())
    }
}

#[derive(Serialize)]
struct ChannelBiasIdentity<'a> {
    tensor: &'a Digest,
    axis: usize,
    channel: u64,
    delta_bits: u32,
    maximum_delta_bits: u32,
}

/// A bounded additive operation over one channel of a contiguous tensor.
#[derive(Clone, Debug)]
pub struct ChannelBias {
    specification: InterventionSpec,
    tensor: TensorSpec,
    axis: usize,
    channel: u64,
    delta: f32,
}

impl ChannelBias {
    /// Creates a channel-local additive stage.
    ///
    /// `maximum_absolute_delta` is part of the stage identity and must be
    /// positive, finite, and at most `16.0`. The absolute delta must not exceed
    /// that declared bound.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid tensor, axis, channel, or delta.
    pub fn new(
        tensor: &TensorSpec,
        axis: usize,
        channel: u64,
        delta: f32,
        maximum_absolute_delta: f32,
    ) -> Result<Self> {
        tensor.validate()?;
        if axis >= tensor.shape.len() {
            return Err(
                CoreError::invalid("channel bias axis", "is outside the tensor rank").into(),
            );
        }
        if channel >= tensor.shape[axis] {
            return Err(CoreError::invalid(
                "channel bias channel",
                "is outside the selected dimension",
            )
            .into());
        }
        if !delta.is_finite()
            || !maximum_absolute_delta.is_finite()
            || maximum_absolute_delta <= 0.0
            || maximum_absolute_delta > 16.0
            || delta.abs() > maximum_absolute_delta
        {
            return Err(CoreError::invalid(
                "channel bias delta",
                "must be finite and within a positive declared bound no greater than 16",
            )
            .into());
        }
        let tensor_digest = tensor.digest()?;
        let implementation = Digest::of_serializable(
            "diffusion-channel-bias-v1",
            &ChannelBiasIdentity {
                tensor: &tensor_digest,
                axis,
                channel,
                delta_bits: delta.to_bits(),
                maximum_delta_bits: maximum_absolute_delta.to_bits(),
            },
        )?;
        Ok(Self {
            specification: InterventionSpec {
                implementation,
                tensor: tensor_digest,
                max_invocations: 1,
            },
            tensor: tensor.clone(),
            axis,
            channel,
            delta,
        })
    }
}

impl Intervention for ChannelBias {
    fn specification(&self) -> &InterventionSpec {
        &self.specification
    }

    fn apply(
        &mut self,
        _context: &StepContext,
        state: &mut [f32],
    ) -> std::result::Result<(), String> {
        let stride = self.tensor.shape[..self.axis]
            .iter()
            .try_fold(1_u64, |product, dimension| product.checked_mul(*dimension))
            .ok_or_else(|| "channel stride overflowed".to_owned())?;
        let dimension = self.tensor.shape[self.axis];
        for (flat, value) in state.iter_mut().enumerate() {
            let flat = u64::try_from(flat).map_err(|_| "flat index exceeds u64".to_owned())?;
            if (flat / stride) % dimension == self.channel {
                *value += self.delta;
            }
        }
        Ok(())
    }
}

fn callback_outcome(
    outcome: std::thread::Result<std::result::Result<(), String>>,
) -> std::result::Result<(), InterventionFailure> {
    match outcome {
        Ok(Ok(())) => Ok(()),
        Ok(Err(message)) => Err(InterventionFailure::new(false, message)),
        Err(payload) => Err(InterventionFailure::new(true, panic_message(&payload))),
    }
}

fn panic_message(payload: &Box<dyn Any + Send>) -> String {
    payload.downcast_ref::<&str>().map_or_else(
        || {
            payload
                .downcast_ref::<String>()
                .map_or_else(|| "callback panicked".to_owned(), Clone::clone)
        },
        |message| (*message).to_owned(),
    )
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use super::*;
    use crate::{DiffusionSchedule, TensorDType, TensorLayout};
    use std::collections::BTreeMap;

    fn fixture() -> (crate::DiffusionPlan, StepContext, TensorSpec) {
        let tensor = TensorSpec::new(
            vec![2, 2, 2, 1],
            TensorDType::F32,
            TensorLayout::DimensionZeroFastest,
            "test0",
        )
        .unwrap();
        let schedule =
            DiffusionSchedule::new(Digest::of_bytes("scheduler", b"v1"), vec![1.0, 0.0]).unwrap();
        let mut components = BTreeMap::new();
        components.insert("model".to_owned(), Digest::of_bytes("artifact", b"model"));
        let plan = crate::DiffusionPlan::new(
            components,
            Digest::of_bytes("conditioning", b"input"),
            Digest::of_bytes("rng", b"v1"),
            7,
            tensor.clone(),
            schedule,
        )
        .unwrap();
        let context = StepContext::for_plan(&plan, 0).unwrap();
        (plan, context, tensor)
    }

    struct CallbackStage {
        specification: InterventionSpec,
        name: &'static str,
        order: Rc<RefCell<Vec<&'static str>>>,
        action: fn(&mut [f32]) -> std::result::Result<(), String>,
    }

    impl Intervention for CallbackStage {
        fn specification(&self) -> &InterventionSpec {
            &self.specification
        }

        fn apply(
            &mut self,
            _context: &StepContext,
            state: &mut [f32],
        ) -> std::result::Result<(), String> {
            self.order.borrow_mut().push(self.name);
            (self.action)(state)
        }
    }

    fn callback_stage(
        tensor: &TensorSpec,
        name: &'static str,
        order: Rc<RefCell<Vec<&'static str>>>,
        action: fn(&mut [f32]) -> std::result::Result<(), String>,
    ) -> CallbackStage {
        CallbackStage {
            specification: InterventionSpec {
                implementation: Digest::of_bytes("test-stage", name.as_bytes()),
                tensor: tensor.digest().unwrap(),
                max_invocations: 1,
            },
            name,
            order,
            action,
        }
    }

    #[test]
    fn stages_run_in_order_and_commit_once() {
        let (plan, context, tensor) = fixture();
        let order = Rc::new(RefCell::new(Vec::new()));
        let add_one = |state: &mut [f32]| {
            for value in state {
                *value += 1.0;
            }
            Ok(())
        };
        let multiply_two = |state: &mut [f32]| {
            for value in state {
                *value *= 2.0;
            }
            Ok(())
        };
        let mut pipeline = Pipeline::new(
            plan.digest().unwrap(),
            tensor.clone(),
            vec![
                Box::new(callback_stage(&tensor, "first", order.clone(), add_one)),
                Box::new(callback_stage(
                    &tensor,
                    "second",
                    order.clone(),
                    multiply_two,
                )),
            ],
        )
        .unwrap();
        pipeline.begin().unwrap();
        let mut state = vec![1.0; 8];
        pipeline.apply(&context, &mut state).unwrap();
        assert_eq!(&*order.borrow(), &["first", "second"]);
        assert_eq!(state, vec![4.0; 8]);
        assert_eq!(pipeline.receipt().elements_copied, 8);
        assert_eq!(pipeline.receipt().elements_committed, 8);
        pipeline.receipt().validate().unwrap();
    }

    #[test]
    fn callback_error_rolls_back_every_stage() {
        let (plan, context, tensor) = fixture();
        let order = Rc::new(RefCell::new(Vec::new()));
        let change = |state: &mut [f32]| {
            state[0] = 9.0;
            Ok(())
        };
        let fail = |_state: &mut [f32]| Err("no commit".to_owned());
        let mut pipeline = Pipeline::new(
            plan.digest().unwrap(),
            tensor.clone(),
            vec![
                Box::new(callback_stage(&tensor, "change", order.clone(), change)),
                Box::new(callback_stage(&tensor, "fail", order, fail)),
            ],
        )
        .unwrap();
        pipeline.begin().unwrap();
        let mut state = vec![1.0; 8];
        assert!(pipeline.apply(&context, &mut state).is_err());
        assert_eq!(state, vec![1.0; 8]);
        assert_eq!(pipeline.receipt().elements_committed, 0);
        assert_eq!(pipeline.receipt().failed_stage, Some(1));
        pipeline.receipt().validate().unwrap();
    }

    #[test]
    fn panic_and_nonfinite_output_are_contained_without_writeback() {
        let (plan, context, tensor) = fixture();
        let order = Rc::new(RefCell::new(Vec::new()));
        let panic_stage =
            |_state: &mut [f32]| -> std::result::Result<(), String> { panic!("contained") };
        let mut pipeline = Pipeline::new(
            plan.digest().unwrap(),
            tensor.clone(),
            vec![Box::new(callback_stage(
                &tensor,
                "panic",
                order,
                panic_stage,
            ))],
        )
        .unwrap();
        pipeline.begin().unwrap();
        let mut state = vec![1.0; 8];
        assert!(pipeline.apply(&context, &mut state).is_err());
        assert_eq!(state, vec![1.0; 8]);
        assert!(
            pipeline.receipt().stages[0]
                .failure
                .as_ref()
                .is_some_and(|failure| failure.panicked)
        );

        let order = Rc::new(RefCell::new(Vec::new()));
        let nan_stage = |state: &mut [f32]| {
            state[0] = f32::NAN;
            Ok(())
        };
        let mut pipeline = Pipeline::new(
            plan.digest().unwrap(),
            tensor.clone(),
            vec![Box::new(callback_stage(&tensor, "nan", order, nan_stage))],
        )
        .unwrap();
        pipeline.begin().unwrap();
        assert!(pipeline.apply(&context, &mut state).is_err());
        assert_eq!(state, vec![1.0; 8]);
    }

    #[test]
    fn shape_and_device_mismatch_fail_before_callbacks() {
        let (plan, mut context, tensor) = fixture();
        let order = Rc::new(RefCell::new(Vec::new()));
        let no_op = |_state: &mut [f32]| Ok(());
        let mut pipeline = Pipeline::new(
            plan.digest().unwrap(),
            tensor.clone(),
            vec![Box::new(callback_stage(
                &tensor,
                "noop",
                order.clone(),
                no_op,
            ))],
        )
        .unwrap();
        pipeline.begin().unwrap();
        context.tensor.device = "other0".to_owned();
        let mut state = vec![1.0; 8];
        assert!(pipeline.apply(&context, &mut state).is_err());
        assert!(order.borrow().is_empty());

        let context = StepContext::for_plan(&plan, 0).unwrap();
        assert!(pipeline.apply(&context, &mut state[..7]).is_err());
        assert!(order.borrow().is_empty());
    }

    #[test]
    fn channel_bias_targets_dimension_zero_fastest_channel() {
        let (plan, context, tensor) = fixture();
        let bias = ChannelBias::new(&tensor, 2, 1, 0.25, 0.5).unwrap();
        let mut pipeline =
            Pipeline::new(plan.digest().unwrap(), tensor, vec![Box::new(bias)]).unwrap();
        pipeline.begin().unwrap();
        let mut state = vec![0.0; 8];
        pipeline.apply(&context, &mut state).unwrap();
        assert_eq!(state, [0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25]);
    }
}

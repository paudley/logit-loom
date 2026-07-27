// SPDX-License-Identifier: MIT OR Apache-2.0

//! Post-commit diffusion-state observation.

use std::{
    any::Any,
    panic::{AssertUnwindSafe, catch_unwind},
};

use logit_loom_core::{ControlFlow, CoreError, Digest};
use serde::{Deserialize, Serialize};

use crate::{Error, InterventionFailure, Result, StepContext, TensorSpec};

const MAX_STEP_OBSERVERS: usize = 32;

/// One synchronous observer at an exact post-step committed-state boundary.
pub trait StepObserver {
    /// Returns the caller-defined implementation identity.
    fn implementation(&self) -> &Digest;

    /// Observes immutable committed state and may request a cooperative stop.
    ///
    /// # Errors
    ///
    /// Returns an error when the observer cannot record the step. The error is
    /// contained before it can cross the native callback boundary.
    fn observe(
        &mut self,
        context: &StepContext,
        state: &[f32],
    ) -> std::result::Result<ControlFlow, String>;
}

/// Mechanical accounting for one post-step observer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObserverReceipt {
    /// Caller-defined implementation identity.
    pub implementation: Digest,
    /// Complete post-step states delivered.
    pub observed_steps: u32,
    /// Finite state elements delivered across observations.
    pub elements_seen: u64,
    /// Whether this observer requested a stop.
    pub stop_requested: bool,
    /// First contained error or panic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<InterventionFailure>,
}

/// Ordered post-step observer fan-out.
pub struct ObserverSet {
    expected_plan: Digest,
    tensor: TensorSpec,
    max_steps: u32,
    observers: Vec<Box<dyn StepObserver>>,
    receipts: Vec<ObserverReceipt>,
}

impl std::fmt::Debug for ObserverSet {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ObserverSet")
            .field("expected_plan", &self.expected_plan)
            .field("tensor", &self.tensor)
            .field("max_steps", &self.max_steps)
            .field("receipts", &self.receipts)
            .finish_non_exhaustive()
    }
}

impl ObserverSet {
    /// Creates an ordered observer set bound to one plan and tensor.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty or oversized set or zero step bound.
    pub fn new(
        expected_plan: Digest,
        tensor: TensorSpec,
        max_steps: u32,
        observers: Vec<Box<dyn StepObserver>>,
    ) -> Result<Self> {
        tensor.validate()?;
        if max_steps == 0 {
            return Err(
                CoreError::invalid("diffusion observer step bound", "must be positive").into(),
            );
        }
        if !(1..=MAX_STEP_OBSERVERS).contains(&observers.len()) {
            return Err(CoreError::invalid(
                "diffusion observers",
                format!("requires 1..={MAX_STEP_OBSERVERS} entries"),
            )
            .into());
        }
        let receipts = observers
            .iter()
            .map(|observer| ObserverReceipt {
                implementation: observer.implementation().clone(),
                observed_steps: 0,
                elements_seen: 0,
                stop_requested: false,
                failure: None,
            })
            .collect();
        Ok(Self {
            expected_plan,
            tensor,
            max_steps,
            observers,
            receipts,
        })
    }

    /// Delivers one immutable committed post-step state in declared order.
    ///
    /// All observers receive the boundary even when an earlier observer asks
    /// to stop. A callback error or panic is contained and terminates fan-out.
    ///
    /// # Errors
    ///
    /// Returns a compatibility, bound, callback, panic, or finite-value error.
    pub fn observe(&mut self, context: &StepContext, state: &[f32]) -> Result<ControlFlow> {
        self.validate_call(context, state)?;
        let elements = u64::try_from(state.len())
            .map_err(|_| Error::Incompatible("state length exceeds u64".to_owned()))?;
        let mut control = ControlFlow::Continue;
        for index in 0..self.observers.len() {
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                self.observers[index].observe(context, state)
            }));
            self.receipts[index].observed_steps = self.receipts[index]
                .observed_steps
                .checked_add(1)
                .ok_or_else(|| Error::Incompatible("observer step count overflowed".to_owned()))?;
            self.receipts[index].elements_seen = self.receipts[index]
                .elements_seen
                .checked_add(elements)
                .ok_or_else(|| {
                    Error::Incompatible("observer element accounting overflowed".to_owned())
                })?;
            match observer_outcome(outcome) {
                Ok(ControlFlow::Continue) => {}
                Ok(ControlFlow::Stop) => {
                    self.receipts[index].stop_requested = true;
                    control = ControlFlow::Stop;
                }
                Err(failure) => {
                    self.receipts[index].failure = Some(failure.clone());
                    return Err(Error::Observer {
                        observer: index,
                        message: failure.message,
                    });
                }
            }
        }
        Ok(control)
    }

    /// Returns current accounting in observer order.
    pub fn receipts(&self) -> &[ObserverReceipt] {
        &self.receipts
    }

    fn validate_call(&self, context: &StepContext, state: &[f32]) -> Result<()> {
        if context.plan != self.expected_plan || context.tensor != self.tensor {
            return Err(Error::Incompatible(
                "observer step plan, shape, dtype, layout, or device differs".to_owned(),
            ));
        }
        let expected = usize::try_from(self.tensor.elements()?)
            .map_err(|_| Error::Incompatible("tensor element count exceeds usize".to_owned()))?;
        if state.len() != expected {
            return Err(Error::Incompatible(format!(
                "observer state contains {} elements; expected {expected}",
                state.len()
            )));
        }
        if state.iter().any(|value| !value.is_finite()) {
            return Err(Error::Incompatible(
                "observer state contains a non-finite value".to_owned(),
            ));
        }
        if self
            .receipts
            .iter()
            .any(|receipt| receipt.failure.is_some())
        {
            return Err(Error::Incompatible(
                "observer set is failed and cannot continue".to_owned(),
            ));
        }
        if self
            .receipts
            .iter()
            .any(|receipt| receipt.observed_steps >= self.max_steps)
        {
            return Err(Error::Incompatible(
                "observer step bound is exhausted".to_owned(),
            ));
        }
        Ok(())
    }
}

fn observer_outcome(
    outcome: std::thread::Result<std::result::Result<ControlFlow, String>>,
) -> std::result::Result<ControlFlow, InterventionFailure> {
    match outcome {
        Ok(Ok(control)) => Ok(control),
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
    use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

    use crate::{DiffusionPlan, DiffusionSchedule, TensorDType, TensorLayout};

    use super::*;

    struct CallbackObserver {
        implementation: Digest,
        order: Rc<RefCell<Vec<&'static str>>>,
        name: &'static str,
        outcome: fn(&[f32]) -> std::result::Result<ControlFlow, String>,
    }

    impl StepObserver for CallbackObserver {
        fn implementation(&self) -> &Digest {
            &self.implementation
        }

        fn observe(
            &mut self,
            _context: &StepContext,
            state: &[f32],
        ) -> std::result::Result<ControlFlow, String> {
            self.order.borrow_mut().push(self.name);
            (self.outcome)(state)
        }
    }

    fn fixture() -> (DiffusionPlan, StepContext) {
        let tensor = TensorSpec::new(
            vec![2, 2, 1],
            TensorDType::F32,
            TensorLayout::DimensionZeroFastest,
            "test0",
        )
        .unwrap();
        let schedule =
            DiffusionSchedule::new(Digest::of_bytes("scheduler", b"v1"), vec![1.0, 0.0]).unwrap();
        let mut components = BTreeMap::new();
        components.insert("model".to_owned(), Digest::of_bytes("artifact", b"model"));
        let plan = DiffusionPlan::new(
            components,
            Digest::of_bytes("conditioning", b"input"),
            Digest::of_bytes("rng", b"v1"),
            7,
            tensor,
            schedule,
        )
        .unwrap();
        let context = StepContext::for_plan(&plan, 0).unwrap();
        (plan, context)
    }

    fn observer(
        order: Rc<RefCell<Vec<&'static str>>>,
        name: &'static str,
        outcome: fn(&[f32]) -> std::result::Result<ControlFlow, String>,
    ) -> CallbackObserver {
        CallbackObserver {
            implementation: Digest::of_bytes("test-observer", name.as_bytes()),
            order,
            name,
            outcome,
        }
    }

    #[test]
    fn stop_still_delivers_the_same_boundary_to_later_observers() {
        let (plan, context) = fixture();
        let order = Rc::new(RefCell::new(Vec::new()));
        let stop = |_state: &[f32]| Ok(ControlFlow::Stop);
        let continue_ = |_state: &[f32]| Ok(ControlFlow::Continue);
        let mut observers = ObserverSet::new(
            plan.digest().unwrap(),
            plan.tensor.clone(),
            1,
            vec![
                Box::new(observer(order.clone(), "stop", stop)),
                Box::new(observer(order.clone(), "later", continue_)),
            ],
        )
        .unwrap();
        assert_eq!(
            observers.observe(&context, &[0.0; 4]).unwrap(),
            ControlFlow::Stop
        );
        assert_eq!(&*order.borrow(), &["stop", "later"]);
        assert_eq!(observers.receipts()[1].observed_steps, 1);
    }

    #[test]
    fn observer_error_and_panic_are_contained() {
        let (plan, context) = fixture();
        let order = Rc::new(RefCell::new(Vec::new()));
        let fail = |_state: &[f32]| Err("failed".to_owned());
        let mut observers = ObserverSet::new(
            plan.digest().unwrap(),
            plan.tensor.clone(),
            1,
            vec![Box::new(observer(order, "fail", fail))],
        )
        .unwrap();
        assert!(observers.observe(&context, &[0.0; 4]).is_err());
        assert!(
            observers.receipts()[0]
                .failure
                .as_ref()
                .is_some_and(|failure| !failure.panicked)
        );

        let order = Rc::new(RefCell::new(Vec::new()));
        let panic_observer =
            |_state: &[f32]| -> std::result::Result<ControlFlow, String> { panic!("contained") };
        let mut observers = ObserverSet::new(
            plan.digest().unwrap(),
            plan.tensor.clone(),
            1,
            vec![Box::new(observer(order, "panic", panic_observer))],
        )
        .unwrap();
        assert!(observers.observe(&context, &[0.0; 4]).is_err());
        assert!(
            observers.receipts()[0]
                .failure
                .as_ref()
                .is_some_and(|failure| failure.panicked)
        );
    }

    #[test]
    fn observer_rejects_wrong_shape_device_and_nonfinite_state() {
        let (plan, mut context) = fixture();
        let order = Rc::new(RefCell::new(Vec::new()));
        let continue_ = |_state: &[f32]| Ok(ControlFlow::Continue);
        let mut observers = ObserverSet::new(
            plan.digest().unwrap(),
            plan.tensor.clone(),
            1,
            vec![Box::new(observer(order, "observe", continue_))],
        )
        .unwrap();
        assert!(observers.observe(&context, &[0.0; 3]).is_err());
        assert!(
            observers
                .observe(&context, &[0.0, 0.0, f32::INFINITY, 0.0])
                .is_err()
        );
        context.tensor.device = "other0".to_owned();
        assert!(observers.observe(&context, &[0.0; 4]).is_err());
    }
}

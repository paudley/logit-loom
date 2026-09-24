// SPDX-License-Identifier: MIT OR Apache-2.0

#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod distribution;
mod error;
mod first_party;
mod observer;
mod reasoning_budget;
mod transform;

pub use distribution::{DistributionReport, DistributionSummary, TokenDistribution};
pub use error::{
    DistributionError, ObserverError, PipelineError, PrefillObserverError, TransformError,
};
pub use first_party::{RankBias, TokenBias};
pub use logit_loom_core::*;
pub use observer::{
    CancellationToken, MAX_OBSERVERS, ObservedToken, Observer, ObserverSet, PrefillMonitor,
    PrefillObserver,
};
pub use reasoning_budget::{ReasoningBudget, ReasoningBudgetReport, ReasoningStart};
pub use transform::{LogitTransform, Pipeline, Stage, TransformContext};

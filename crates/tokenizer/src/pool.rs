// SPDX-License-Identifier: MIT OR Apache-2.0

//! Caller-sized dedicated execution pool with explicit backpressure.

use std::{
    any::Any,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver, SyncSender, TrySendError},
    },
    thread::{self, JoinHandle},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Maximum dedicated worker threads in one pool.
pub const MAX_POOL_WORKERS: u32 = 256;
/// Maximum queued tasks in one pool.
pub const MAX_POOL_QUEUE: u32 = 65_536;

/// Exact caller-selected dedicated-pool bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoolConfig {
    /// Dedicated worker-thread count.
    pub workers: u32,
    /// Maximum tasks waiting for a worker.
    pub queue_capacity: u32,
}

impl PoolConfig {
    /// Validates nonzero public bounds.
    ///
    /// # Errors
    ///
    /// Returns an error for a zero or excessive worker/queue count.
    pub fn validate(self) -> Result<(), PoolError> {
        if !(1..=MAX_POOL_WORKERS).contains(&self.workers)
            || !(1..=MAX_POOL_QUEUE).contains(&self.queue_capacity)
        {
            return Err(PoolError::InvalidConfig);
        }
        Ok(())
    }
}

/// Content-free execution-pool failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum PoolError {
    /// Worker or queue bounds were invalid.
    #[error("dedicated pool configuration is outside public bounds")]
    InvalidConfig,
    /// The bounded queue had no free slot.
    #[error("dedicated pool queue is full")]
    QueueFull,
    /// The pool was already closed.
    #[error("dedicated pool is closed")]
    Closed,
    /// A worker thread could not be created.
    #[error("dedicated pool worker creation failed")]
    ThreadSpawn,
    /// A submitted task unwound; its panic payload is not returned by this
    /// pool.
    ///
    /// Rust invokes the process panic hook before `catch_unwind`; hook policy
    /// remains caller-owned and this pool does not mutate it globally.
    #[error("dedicated pool task panicked")]
    TaskPanicked,
    /// A worker ended unexpectedly before delivering its result.
    #[error("dedicated pool worker ended before result delivery")]
    WorkerLost,
}

/// Content-free accounting returned by explicit pool shutdown.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoolReceipt {
    /// Exact configured worker count.
    pub workers: u32,
    /// Exact configured waiting-task capacity.
    pub queue_capacity: u32,
    /// Tasks accepted by the bounded queue.
    pub submitted: u64,
    /// Tasks that returned or unwound inside a worker.
    pub completed: u64,
    /// Accepted tasks whose closures unwound.
    pub panicked: u64,
    /// Whether every worker joined.
    pub joined: bool,
}

/// Result handle for one accepted task.
#[derive(Debug)]
pub struct PoolJob<T> {
    receiver: Receiver<Result<T, PoolError>>,
}

impl<T> PoolJob<T> {
    /// Waits for exactly this task's result.
    ///
    /// # Errors
    ///
    /// Returns a content-free panic or lost-worker failure.
    pub fn wait(self) -> Result<T, PoolError> {
        self.receiver.recv().map_err(|_| PoolError::WorkerLost)?
    }
}

trait ErasedJob: Send {
    fn run(self: Box<Self>);
}

struct Job<F, T> {
    task: Option<F>,
    result: mpsc::Sender<Result<T, PoolError>>,
    completed: Arc<AtomicU64>,
    panicked: Arc<AtomicU64>,
}

impl<F, T> ErasedJob for Job<F, T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    fn run(mut self: Box<Self>) {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.task.take().expect("pool job is consumed exactly once")()
        }))
        .map_err(|_payload: Box<dyn Any + Send>| PoolError::TaskPanicked);
        if outcome.is_err() {
            self.panicked.fetch_add(1, Ordering::AcqRel);
        }
        let _ = self.result.send(outcome);
        self.completed.fetch_add(1, Ordering::AcqRel);
    }
}

/// Persistent bounded pool whose thread and queue counts are supplied by the
/// caller.
///
/// The pool creates no ambient Rayon state, process-global mutation, helper
/// process, socket, retry, or fallback. Submitted closures must own or
/// reference-count their inputs because workers persist beyond a submit call.
pub struct DedicatedPool {
    config: PoolConfig,
    sender: Option<SyncSender<Box<dyn ErasedJob>>>,
    workers: Vec<JoinHandle<()>>,
    submitted: Arc<AtomicU64>,
    completed: Arc<AtomicU64>,
    panicked: Arc<AtomicU64>,
}

impl std::fmt::Debug for DedicatedPool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DedicatedPool")
            .field("config", &self.config)
            .field("submitted", &self.submitted.load(Ordering::Acquire))
            .field("completed", &self.completed.load(Ordering::Acquire))
            .field("panicked", &self.panicked.load(Ordering::Acquire))
            .finish_non_exhaustive()
    }
}

impl DedicatedPool {
    /// Starts the exact caller-sized dedicated worker set.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid bounds or worker creation failure.
    pub fn new(config: PoolConfig) -> Result<Self, PoolError> {
        config.validate()?;
        let queue_capacity =
            usize::try_from(config.queue_capacity).map_err(|_| PoolError::InvalidConfig)?;
        let (sender, receiver) = mpsc::sync_channel::<Box<dyn ErasedJob>>(queue_capacity);
        let receiver = Arc::new(Mutex::new(receiver));
        let submitted = Arc::new(AtomicU64::new(0));
        let completed = Arc::new(AtomicU64::new(0));
        let panicked = Arc::new(AtomicU64::new(0));
        let mut workers = Vec::with_capacity(
            usize::try_from(config.workers).map_err(|_| PoolError::InvalidConfig)?,
        );
        for index in 0..config.workers {
            let receiver = Arc::clone(&receiver);
            let worker = thread::Builder::new()
                .name(format!("logit-loom-tokenizer-{index}"))
                .spawn(move || worker_loop(&receiver));
            if let Ok(worker) = worker {
                workers.push(worker);
            } else {
                drop(sender);
                for worker in workers {
                    let _ = worker.join();
                }
                return Err(PoolError::ThreadSpawn);
            }
        }
        Ok(Self {
            config,
            sender: Some(sender),
            workers,
            submitted,
            completed,
            panicked,
        })
    }

    /// Attempts to enqueue one owned task without waiting for queue space.
    ///
    /// # Errors
    ///
    /// Returns [`PoolError::QueueFull`] for backpressure or
    /// [`PoolError::Closed`] after shutdown.
    pub fn try_submit<F, T>(&self, task: F) -> Result<PoolJob<T>, PoolError>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let sender = self.sender.as_ref().ok_or(PoolError::Closed)?;
        let (result, receiver) = mpsc::channel();
        let job = Box::new(Job {
            task: Some(task),
            result,
            completed: Arc::clone(&self.completed),
            panicked: Arc::clone(&self.panicked),
        });
        match sender.try_send(job) {
            Ok(()) => {
                self.submitted.fetch_add(1, Ordering::AcqRel);
                Ok(PoolJob { receiver })
            }
            Err(TrySendError::Full(_)) => Err(PoolError::QueueFull),
            Err(TrySendError::Disconnected(_)) => Err(PoolError::Closed),
        }
    }

    /// Stops accepting work, drains accepted tasks, and joins every worker.
    pub fn close(mut self) -> PoolReceipt {
        self.sender.take();
        let joined = self.workers.drain(..).all(|worker| worker.join().is_ok());
        PoolReceipt {
            workers: self.config.workers,
            queue_capacity: self.config.queue_capacity,
            submitted: self.submitted.load(Ordering::Acquire),
            completed: self.completed.load(Ordering::Acquire),
            panicked: self.panicked.load(Ordering::Acquire),
            joined,
        }
    }
}

impl Drop for DedicatedPool {
    fn drop(&mut self) {
        self.sender.take();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn worker_loop(receiver: &Mutex<Receiver<Box<dyn ErasedJob>>>) {
    loop {
        let next = receiver
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .recv();
        match next {
            Ok(job) => job.run(),
            Err(_) => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caller_sized_pool_executes_and_joins_exactly() {
        let pool = DedicatedPool::new(PoolConfig {
            workers: 2,
            queue_capacity: 4,
        })
        .unwrap();
        let first = pool.try_submit(|| 2 + 2).unwrap();
        let second = pool.try_submit(|| 6 * 7).unwrap();
        assert_eq!(first.wait().unwrap(), 4);
        assert_eq!(second.wait().unwrap(), 42);
        let receipt = pool.close();
        assert_eq!(receipt.workers, 2);
        assert_eq!(receipt.queue_capacity, 4);
        assert_eq!(receipt.submitted, 2);
        assert_eq!(receipt.completed, 2);
        assert_eq!(receipt.panicked, 0);
        assert!(receipt.joined);
    }

    #[test]
    fn task_panics_are_contained_without_returning_payloads() {
        let pool = DedicatedPool::new(PoolConfig {
            workers: 1,
            queue_capacity: 1,
        })
        .unwrap();
        let failed = pool
            .try_submit(|| -> usize { panic!("private task content") })
            .unwrap();
        assert_eq!(failed.wait(), Err(PoolError::TaskPanicked));
        let healthy = pool.try_submit(|| 7).unwrap();
        assert_eq!(healthy.wait().unwrap(), 7);
        let receipt = pool.close();
        assert_eq!(receipt.panicked, 1);
        assert!(receipt.joined);
    }

    #[test]
    fn bounded_queue_reports_backpressure_without_blocking() {
        let pool = DedicatedPool::new(PoolConfig {
            workers: 1,
            queue_capacity: 1,
        })
        .unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let running = pool
            .try_submit(move || {
                started_tx.send(()).unwrap();
                release_rx.recv().unwrap();
            })
            .unwrap();
        started_rx.recv().unwrap();
        let queued = pool.try_submit(|| 2).unwrap();
        assert_eq!(pool.try_submit(|| 3).unwrap_err(), PoolError::QueueFull);
        release_tx.send(()).unwrap();
        running.wait().unwrap();
        assert_eq!(queued.wait().unwrap(), 2);
        let receipt = pool.close();
        assert_eq!(receipt.submitted, 2);
        assert_eq!(receipt.completed, 2);
    }

    #[test]
    fn public_pool_bounds_fail_closed() {
        assert_eq!(
            DedicatedPool::new(PoolConfig {
                workers: 0,
                queue_capacity: 1,
            })
            .unwrap_err(),
            PoolError::InvalidConfig
        );
        assert_eq!(
            DedicatedPool::new(PoolConfig {
                workers: 1,
                queue_capacity: MAX_POOL_QUEUE + 1,
            })
            .unwrap_err(),
            PoolError::InvalidConfig
        );
    }
}

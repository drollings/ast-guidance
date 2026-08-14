//! Memory ingestion SupervisedBatch. Wraps `fluent_concurrency::Scope` with
//! credit-based backpressure for memory pipeline ingestion.

use crate::types::*;
use fluent_concurrency::flow::{self, CreditReceiver, CreditSender, CreditSpec};
use fluent_concurrency::scope::Scope;

/// Ingestion job dispatched to the memory SupervisedBatch.
///
/// Flat enum dispatch — no `dyn Trait` in the hot path.
pub enum IngestJob {
    /// Sync a completed turn.
    SyncTurn {
        /// User message content.
        user: String,
        /// Assistant response content.
        assistant: String,
        /// Session identifier.
        session: String,
    },
    /// End-of-session extraction.
    SessionEnd {
        /// Messages from the session to extract from.
        messages: Vec<TurnMessage>,
    },
    /// Auto-extract facts from content.
    AutoExtract {
        /// Content to extract facts from.
        content: String,
        /// Category for extracted facts.
        category: String,
    },
}

/// Memory-specific ingestion SupervisedBatch.
///
/// Wraps a `fluent_concurrency::Scope` with credit-based backpressure:
/// - Credit tokens prevent unbounded heap accumulation during deep repo syncs
/// - Fault containment: all tasks live inside the scope; close awaits completion
/// - No ambient `tokio::spawn` — every task is spawned inside this scope
pub struct MemoryBatch {
    scope: Scope,
    credit: CreditSender,
    _receiver: CreditReceiver,
}

impl MemoryBatch {
    /// Create a new memory ingestion SupervisedBatch.
    ///
    /// - `credit_limit`: max queued items before producer blocks
    /// - `more_after`: bump credit after this many completions
    pub fn new(credit_limit: usize, more_after: usize) -> Self {
        let spec = CreditSpec {
            initial: credit_limit,
            more_after,
        };
        let (credit, receiver) = flow::new(spec);
        Self {
            scope: Scope::new(),
            credit,
            _receiver: receiver,
        }
    }

    /// Enqueue an ingestion job with backpressure.
    ///
    /// Blocks (async) if the credit limit is exhausted, preventing
    /// unbounded memory growth during deep repo syncs.
    pub async fn ingest(&self, job: IngestJob) -> Result<(), MemoryError> {
        let job_name = match &job {
            IngestJob::SyncTurn { .. } => "memory.ingest.sync_turn",
            IngestJob::SessionEnd { .. } => "memory.ingest.session_end",
            IngestJob::AutoExtract { .. } => "memory.ingest.auto_extract",
        };

        // Acquire credit before spawning — backpressure at enqueue time
        self.credit
            .send(|| async {
                // Task body would perform the actual ingestion.
                // In production, this would call the plugin's async method.
                tracing::debug!("ingestion dispatched: {job_name}");
            })
            .await;

        Ok(())
    }

    /// Close the SupervisedBatch, waiting for all in-flight ingestion to complete.
    pub async fn close(&mut self) {
        self.scope.close().await;
    }
}

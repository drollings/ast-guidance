use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::batch::SupervisedBatch;
use crate::runtime::test::TestRuntime;
use crate::runtime::tokio::TokioRuntime;
use fluent_wvr::prelude::*;
use fluent_wvr::Runtime;
use fluent_wvr_testutil::StubComponent;
use internment::ArcIntern;

/// A fresh `SupervisedBatch` on the production `TokioRuntime` with an empty
/// `CapabilitySet` — the shared setup for the m2 supervision suites.
pub fn make_batch() -> SupervisedBatch {
    SupervisedBatch::new(crate::tokio_runtime(), CapabilitySet::new())
}

/// A `SupervisedBatch` with a custom config (see `make_batch` for the
/// runtime/caps defaults).
pub fn make_batch_with_config(config: crate::batch::SupervisedBatchConfig) -> SupervisedBatch {
    SupervisedBatch::new_with_config(crate::tokio_runtime(), CapabilitySet::new(), config)
}

struct TestCapA;
impl Capability for TestCapA {
    fn name(&self) -> &'static str {
        "cap_a"
    }
}

struct TestCapB;
impl Capability for TestCapB {
    fn name(&self) -> &'static str {
        "cap_b"
    }
}

mod e2e;
mod m1;
mod m2;
mod m3;
mod m4;
// m5 exercises the capability-gated I/O engines (fs/net/db), which are
// compiled only when the `db` feature is on.
#[cfg(feature = "db")]
mod m5;

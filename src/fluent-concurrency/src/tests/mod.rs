use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::runtime::test::TestRuntime;
use crate::runtime::tokio::TokioRuntime;
use fluent_wvr::prelude::*;
use fluent_wvr::Runtime;
use fluent_wvr_testutil::{impl_component_for_test, StubComponent};
use internment::ArcIntern;

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

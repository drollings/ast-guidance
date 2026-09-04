#![allow(unused_imports)]
#[allow(unused_imports)]
use fluent_wvr::capability::*;
#[allow(unused_imports)]
use fluent_wvr::prelude::*;
#[allow(unused_imports)]
use fluent_wvr::{FieldAccess, WorkUnit, Component, Describable, FieldError};


struct NetCapability;
impl Capability for NetCapability {
    fn name(&self) -> &'static str {
        "net"
    }
}

#[test]
fn capability_set_remove_returns_capability() {
    let mut caps = CapabilitySet::new().with(NetCapability);
    let removed = caps.remove::<NetCapability>();
    assert!(removed.is_some());
    assert!(caps.get::<NetCapability>().is_none());
}

#[test]
fn capability_set_remove_returns_none_when_absent() {
    let mut caps = CapabilitySet::new();
    assert!(caps.remove::<NetCapability>().is_none());
}

#[test]
fn capability_set_remove_as_returns_concrete() {
    let mut caps = CapabilitySet::new().with(NetCapability);
    let removed: Option<NetCapability> = caps.remove_as::<NetCapability>();
    assert!(removed.is_some(), "should return concrete NetCapability");
    assert!(
        caps.get::<NetCapability>().is_none(),
        "should be removed from set"
    );
}

#[test]
fn capability_set_remove_as_returns_none_when_absent() {
    let mut caps = CapabilitySet::new();
    assert!(caps.remove_as::<NetCapability>().is_none());
}

#[test]
fn capability_set_contains() {
    let caps = CapabilitySet::new().with(NetCapability);
    assert!(caps.contains::<NetCapability>());
    assert!(!caps.contains::<FsCapability>());    }

#[test]
fn capability_set_iter_yields_correct_count() {
    let caps = CapabilitySet::new().with(NetCapability).with(FsCapability::new());
    assert_eq!(caps.iter().count(), 2);
}

#[test]
fn capability_set_len_and_is_empty() {
    let mut caps = CapabilitySet::new();
    assert!(caps.is_empty());
    assert_eq!(caps.len(), 0);
    caps = caps.with(NetCapability);
    assert!(!caps.is_empty());
    assert_eq!(caps.len(), 1);
}

#[test]
fn capability_error_missing_display_and_io_kind() {
    let err = CapabilityError::Missing { name: "db" };
    assert_eq!(err.to_string(), "missing capability: db");
    let io_err: std::io::Error = err.into();
    assert_eq!(io_err.kind(), std::io::ErrorKind::PermissionDenied);
    assert!(io_err.to_string().contains("missing capability: db"));
}

#[tokio::test]
async fn check_capability_gates_on_task_local() {
    let cap = NetCapability;
    // Outside a CURRENT_CAPS scope: denied.
    assert!(check_capability(&cap).is_err());
    // Inside a scope that contains the capability: allowed.
    let result = CURRENT_CAPS
        .scope(CapabilitySet::new().with(NetCapability), async {
            check_capability(&cap)
        })
        .await;
    assert!(result.is_ok());
    // Inside a scope that lacks it: denied with PermissionDenied.
    let result = CURRENT_CAPS
        .scope(CapabilitySet::new(), async { check_capability(&cap) })
        .await
        .unwrap_err();
    assert_eq!(result.kind(), std::io::ErrorKind::PermissionDenied);
    assert!(result.to_string().contains("missing"));
}

#[test]
fn capability_aware_fs_denied_without_token() {
    use fluent_wvr::capability::capability_aware_fs;
    let dir = std::env::temp_dir().join(format!(
        "fluent-wvr-fs-gate-{}",
        common_core::hash::uuid_v4()
    ));
    // No FsCapability in the current scope: the gate refuses.
    assert!(capability_aware_fs::create_dir_all(&dir).is_err());
    assert_eq!(
        capability_aware_fs::create_dir_all(&dir).unwrap_err().kind(),
        std::io::ErrorKind::PermissionDenied
    );
}

#[test]
fn capability_aware_fs_allowed_with_token() {
    use fluent_wvr::capability::capability_aware_fs;
    let dir = std::env::temp_dir().join(format!(
        "fluent-wvr-fs-gate-{}",
        common_core::hash::uuid_v4()
    ));
    let result = CURRENT_CAPS.sync_scope(
        CapabilitySet::new().with(FsCapability::new()),
        || {
            let r = capability_aware_fs::create_dir_all(&dir);
            assert!(r.is_ok(), "gated create_dir_all must succeed with the token");
            r
        },
    );
    assert!(result.is_ok());
    let _ = std::fs::remove_dir_all(&dir);
}

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;

// The `CapabilitySet` installed for the current async task.
//
// This is the canonical home of the capability-gating task-local. It lives
// here (rather than in a consumer crate) so that both `fluent-concurrency`
// (which propagates capabilities through `Scope`/`Zone` spawn boundaries)
// and `fluent-db` (whose `DbCapability` must gate its operations) can read
// the *same* variable without a cyclic dependency between the two crates.
//
// - `fluent-concurrency::scope::CURRENT_CAPS` is a re-export of this static.
// - `fluent-concurrency::io::check_capability` re-exports the check below.
//
// `tokio::task_local!` expands to a `const`-like static; see `CapabilitySet`
// for the type-map backing it.
tokio::task_local! {
    pub static CURRENT_CAPS: CapabilitySet;
}

/// Why a capability request was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityError {
    /// The capability was not present in the current task-local `CapabilitySet`.
    Missing { name: &'static str },
    /// The capability is present, but the underlying resource is exhausted.
    Exhausted { name: &'static str, detail: String },
}

impl std::fmt::Display for CapabilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing { name } => write!(f, "missing capability: {name}"),
            Self::Exhausted { name, detail } => {
                write!(f, "capability exhausted: {name} — {detail}")
            }
        }
    }
}

impl std::error::Error for CapabilityError {}

impl From<CapabilityError> for std::io::Error {
    fn from(err: CapabilityError) -> Self {
        std::io::Error::new(std::io::ErrorKind::PermissionDenied, err)
    }
}

/// Validates that the current task-local `CapabilitySet` contains the requested
/// capability. Returns `Err(PermissionDenied)` if the capability is absent.
pub fn check_capability<C: Capability>(cap: &C) -> Result<(), std::io::Error> {
    let present = CURRENT_CAPS
        .try_with(|caps| caps.get::<C>().is_some())
        .unwrap_or(false);
    if present {
        Ok(())
    } else {
        Err(CapabilityError::Missing { name: cap.name() }.into())
    }
}

/// A capability token that can be placed in a `CapabilitySet` to gate access
/// to resources (network, filesystem, database).
///
/// # Examples
///
/// ```
/// use fluent_wvr::Capability;
///
/// struct NetCapability;
/// impl Capability for NetCapability {
///     fn name(&self) -> &'static str { "net" }
/// }
/// ```
pub trait Capability: Send + Sync + 'static {
    fn name(&self) -> &'static str;
}

/// A type-map of capability tokens, used to gate access to resources.
///
/// # Examples
///
/// ```
/// use fluent_wvr::{Capability, CapabilitySet};
///
/// struct FsCapability;
/// impl Capability for FsCapability {
///     fn name(&self) -> &'static str { "fs" }
/// }
///
/// let caps = CapabilitySet::new().with(FsCapability);
/// assert!(caps.get::<FsCapability>().is_some());
/// ```
#[derive(Default, Debug)]
pub struct CapabilitySet {
    caps: HashMap<TypeId, Arc<dyn Any + Send + Sync>>,
}

impl Clone for CapabilitySet {
    fn clone(&self) -> Self {
        Self {
            caps: self.caps.clone(),
        }
    }
}

impl CapabilitySet {
    pub fn new() -> Self {
        Self {
            caps: HashMap::new(),
        }
    }

    #[must_use]
    pub fn with<C: Capability>(mut self, cap: C) -> Self {
        self.caps.insert(TypeId::of::<C>(), Arc::new(cap));
        self
    }

    pub fn get<C: Capability>(&self) -> Option<&C> {
        self.caps
            .get(&TypeId::of::<C>())
            .and_then(|arc| (&**arc as &dyn Any).downcast_ref::<C>())
    }

    /// Remove a capability by type. Returns the removed capability if present.
    pub fn remove<C: Capability>(&mut self) -> Option<Arc<dyn Any + Send + Sync>> {
        self.caps.remove(&TypeId::of::<C>())
    }

    /// Remove a capability by type and downcast to the concrete type.
    ///
    /// The capability must be uniquely owned. If the capability is held by
    /// multiple `Arc`s (via clone of the inner), this returns `None`.
    /// Use `remove::<C>()` to get the `Arc` directly without the ownership
    /// requirement.
    pub fn remove_as<C: Capability>(&mut self) -> Option<C> {
        self.caps
            .remove(&TypeId::of::<C>())
            .and_then(|arc| Arc::downcast::<C>(arc).ok())
            .and_then(|arc| Arc::try_unwrap(arc).ok())
    }

    /// Check whether a capability of type `C` is present.
    pub fn contains<C: Capability>(&self) -> bool {
        self.caps.contains_key(&TypeId::of::<C>())
    }

    /// Iterate over all capabilities in the set (order is unspecified).
    pub fn iter(&self) -> impl Iterator<Item = &Arc<dyn Any + Send + Sync>> {
        self.caps.values()
    }

    /// Number of capabilities in the set.
    pub fn len(&self) -> usize {
        self.caps.len()
    }

    /// Returns true if no capabilities are present.
    pub fn is_empty(&self) -> bool {
        self.caps.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NetCapability;
    impl Capability for NetCapability {
        fn name(&self) -> &'static str {
            "net"
        }
    }

    struct FsCapability;
    impl Capability for FsCapability {
        fn name(&self) -> &'static str {
            "fs"
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
        assert!(!caps.contains::<FsCapability>());
    }

    #[test]
    fn capability_set_iter_yields_correct_count() {
        let caps = CapabilitySet::new().with(NetCapability).with(FsCapability);
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
}

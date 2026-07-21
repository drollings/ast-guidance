use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;

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

pub struct Reserve {
    counter: Arc<std::sync::atomic::AtomicUsize>,
    committed: bool,
}

impl Reserve {
    /// Attempt to acquire a permit from the counter.
    ///
    /// Returns `None` if the counter is already at zero (no permits available).
    /// Does NOT underflow — this is the safe alternative to `new()`.
    pub fn try_acquire(counter: Arc<std::sync::atomic::AtomicUsize>) -> Option<Self> {
        let prev = counter.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        if prev == 0 {
            // Underflow would occur — restore counter and return None
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            None
        } else {
            Some(Self {
                counter,
                committed: false,
            })
        }
    }

    pub fn commit(mut self) {
        self.committed = true;
    }
}

impl Drop for Reserve {
    fn drop(&mut self) {
        if !self.committed {
            self.counter
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
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
}

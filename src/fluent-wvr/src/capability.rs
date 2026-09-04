use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Capability-gated **synchronous** filesystem I/O for serving-path code.
///
/// Every function first checks that the caller holds the `FsCapability` token
/// in the current `CURRENT_CAPS` task-local, then performs the standard-library
/// operation. This is the shared gate the router's boot/serving fs calls use so
/// each site does not re-implement `check_capability(...)?`. The async
/// (tokio-based) counterpart lives on [`FsCapability`] itself.
pub mod capability_aware_fs {
    use std::io;
    use std::path::Path;

    use super::{check_capability, FsCapability};

    fn granted() -> io::Result<()> {
        check_capability(&FsCapability::new())
    }

    /// `create_dir_all` gated on the `FsCapability` token.
    pub fn create_dir_all(path: impl AsRef<Path>) -> io::Result<()> {
        granted()?;
        std::fs::create_dir_all(path)
    }

    /// `metadata` gated on the `FsCapability` token.
    pub fn metadata(path: impl AsRef<Path>) -> io::Result<std::fs::Metadata> {
        granted()?;
        std::fs::metadata(path)
    }

    /// `read_dir` gated on the `FsCapability` token.
    pub fn read_dir(path: impl AsRef<Path>) -> io::Result<std::fs::ReadDir> {
        granted()?;
        std::fs::read_dir(path)
    }

    /// `read_to_string` gated on the `FsCapability` token.
    pub fn read_to_string(path: impl AsRef<Path>) -> io::Result<String> {
        granted()?;
        std::fs::read_to_string(path)
    }
}

// The `CapabilitySet` installed for the current async task.
//
// This is the canonical home of the capability-gating task-local. It lives
// here (rather than in a consumer crate) so that both `fluent-concurrency`
// (which propagates capabilities through `Scope`/`SupervisedBatch` spawn boundaries)
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

/// Capability-gated filesystem operations.
///
/// This is the canonical `FsCapability` token (the capability model lives in
/// `fluent-wvr`). It is re-exported unchanged from
/// `fluent-concurrency::io::fs`, which is where the async tokio-backed
/// operations were originally introduced.
///
/// Cannot be constructed directly outside this crate; use `FsCapability::new()`.
pub struct FsCapability {
    _priv: (),
}

impl FsCapability {
    pub fn new() -> Self {
        Self { _priv: () }
    }

    pub async fn read(&self, path: impl AsRef<Path>) -> Result<Vec<u8>, common_core::error::IoError> {
        check_capability(self)?;
        Ok(tokio::fs::read(path).await?)
    }

    pub async fn write(
        &self,
        path: impl AsRef<Path>,
        contents: impl AsRef<[u8]>,
    ) -> Result<(), common_core::error::IoError> {
        check_capability(self)?;
        Ok(tokio::fs::write(path, contents).await?)
    }

    pub async fn metadata(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<std::fs::Metadata, common_core::error::IoError> {
        check_capability(self)?;
        Ok(tokio::fs::metadata(path).await?)
    }
}

impl Default for FsCapability {
    fn default() -> Self {
        Self::new()
    }
}

impl Capability for FsCapability {
    fn name(&self) -> &'static str {
        "fs"
    }
}

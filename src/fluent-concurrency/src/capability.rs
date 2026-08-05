//! Concrete capability tokens for filesystem, network, and database access.
//! Used with `CapabilitySet` to gate I/O operations.

use fluent_wvr::CapabilitySet;

use crate::io::db::DbCapability;
use crate::io::fs::FsCapability;
use crate::io::net::NetCapability;

/// Returns a `CapabilitySet` pre-populated with Fs and Net capabilities.
/// DbCapability requires a path to open, so it's not included by default.
pub fn default_capability_set() -> CapabilitySet {
    CapabilitySet::new()
        .with(FsCapability::new())
        .with(NetCapability::new())
}

/// Returns a `CapabilitySet` with Fs, Net, and Db capabilities.
///
/// `DbCapability` is provided by `fluent-db`; the error taxonomy is
/// `fluent_db::error::DbError`.
pub fn capability_set_with_db(path: &str) -> Result<CapabilitySet, fluent_db::error::DbError> {
    let db = DbCapability::open(path)?;
    Ok(CapabilitySet::new()
        .with(FsCapability::new())
        .with(NetCapability::new())
        .with(db))
}

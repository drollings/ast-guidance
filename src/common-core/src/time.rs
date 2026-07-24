//! Time utilities: epoch-second helpers.

use std::time::{SystemTime, UNIX_EPOCH};

/// Current system time as seconds since the Unix epoch.
pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}
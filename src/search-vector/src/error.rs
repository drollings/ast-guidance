//! Database error taxonomy — re-export of `fluent-db`'s `DbError` (D3).
//!
//! The hand-written `From<rusqlite::Error>` impl and the four local variants
//! (`Sqlite`/`NotFound`/`DuplicateEntry`/`InvalidSchemaVersion`) were
//! consolidated into `fluent_db::error::DbError` in M4, which centralizes the
//! `is_unique_violation` → `DuplicateEntry` and `SQLITE_BUSY` → `Busy`
//! mappings. This module keeps the `search_vector::error::DbError` path alive
//! for existing consumers (`coral`), with no behavior change.

pub use fluent_db::error::DbError;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn db_error_not_found() {
        let err = DbError::NotFound("test_node".into());
        assert!(format!("{err}").contains("test_node"));
    }
}

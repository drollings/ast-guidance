use super::*;

#[test]
fn registry_error_display() {
    let err = RegistryError::DuplicateTarget {
        name: "build".into(),
    };
    assert!(format!("{err}").contains("build"));
    assert!(format!("{err}").contains("already exists"));
}

#[test]
fn target_not_found_display() {
    let err = RegistryError::TargetNotFound("missing".into());
    assert!(format!("{err}").contains("missing"));
    assert!(format!("{err}").contains("not found"));
}

#[test]
fn invalid_capability_display() {
    let err = RegistryError::InvalidCapability("bogus:cap".into());
    assert!(format!("{err}").contains("bogus:cap"));
    assert!(format!("{err}").contains("invalid capability"));
}

#[test]
fn bit_index_out_of_range_display() {
    let err = RegistryError::BitIndexOutOfRange(42);
    assert!(format!("{err}").contains("42"));
    assert!(format!("{err}").contains("out of range"));
}

#[test]
fn database_from_sqlite_error() {
    let sqlite = common_core::error::SqliteError::from(rusqlite::Error::InvalidQuery);
    let err = RegistryError::Database(sqlite);
    assert!(format!("{err}").contains("database error"));
}

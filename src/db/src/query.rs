//! Typed statement helpers shared by `SqliteStore` and `SqlitePool`.
//!
//! These free functions are the single source of truth for the statement
//! shapes every consumer re-implemented inline: `query_row` maps
//! `QueryReturnedNoRows` to `None`, `query_rows` to `vec![]`, and
//! `query_rows_from_iter` generalizes the `in_clause` + `params_from_iter`
//! combo that coral and search-vector hand-rolled (see
//! `coral/src/db/nodes.rs:45-66` and `search-vector/src/db.rs:361-389`).

use rusqlite::Connection;

use crate::error::DbError;

/// Query at most one row.
///
/// A statement that matches no rows returns `Ok(None)` (not an error).
/// `params` is any `rusqlite::Params` — a `rusqlite::params![...]` slice, an
/// owned value array, or a `ParamsFromIter` for dynamic arity.
pub fn query_row<T, P, M>(
    conn: &Connection,
    sql: &str,
    params: P,
    map: M,
) -> Result<Option<T>, DbError>
where
    P: rusqlite::Params,
    M: FnOnce(&rusqlite::Row<'_>) -> rusqlite::Result<T>,
{
    let mut stmt = conn.prepare(sql).map_err(DbError::from)?;
    let mut rows = stmt.query(params).map_err(DbError::from)?;
    match rows.next().map_err(DbError::from)? {
        Some(row) => map(row).map(Some).map_err(DbError::from),
        None => Ok(None),
    }
}

/// Query all rows.
///
/// A statement that matches no rows returns `Ok(vec![])`.
pub fn query_rows<T, P, M>(
    conn: &Connection,
    sql: &str,
    params: P,
    mut map: M,
) -> Result<Vec<T>, DbError>
where
    P: rusqlite::Params,
    M: FnMut(&rusqlite::Row<'_>) -> rusqlite::Result<T>,
{
    let mut stmt = conn.prepare(sql).map_err(DbError::from)?;
    let mapped = stmt.query_map(params, &mut map).map_err(DbError::from)?;
    mapped
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(DbError::from)
}

/// Execute a DML statement and return the number of rows affected.
pub fn execute<P: rusqlite::Params>(
    conn: &Connection,
    sql: &str,
    params: P,
) -> Result<usize, DbError> {
    conn.execute(sql, params).map_err(DbError::from)
}

/// Execute a multi-statement SQL batch (e.g. DDL).
pub fn execute_batch(conn: &Connection, sql: &str) -> Result<(), DbError> {
    conn.execute_batch(sql).map_err(DbError::from)
}

/// Query all rows with dynamic-arity params from an iterator.
///
/// This is the generalized `in_clause` + `params_from_iter` combo: build a
/// `WHERE x IN (?,?,…)` placeholder list with `common_core::sqlite::in_clause`
/// and pass the actual values through `I` (e.g. a `Vec<&str>` of node names).
pub fn query_rows_from_iter<T, I, M>(
    conn: &Connection,
    sql: &str,
    params: I,
    mut map: M,
) -> Result<Vec<T>, DbError>
where
    I: IntoIterator,
    I::Item: rusqlite::ToSql,
    M: FnMut(&rusqlite::Row<'_>) -> rusqlite::Result<T>,
{
    let mut stmt = conn.prepare(sql).map_err(DbError::from)?;
    let mapped = stmt
        .query_map(rusqlite::params_from_iter(params), &mut map)
        .map_err(DbError::from)?;
    mapped
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(DbError::from)
}

/// The `rowid` of the most recent successful `INSERT` on `conn`.
pub fn last_insert_rowid(conn: &Connection) -> i64 {
    conn.last_insert_rowid()
}

/// Run a closure inside a transaction.
///
/// Commits on `Ok`, rolls back on `Err` (the transaction is dropped while
/// uncommitted, which rolls it back).
pub fn transaction<T>(
    conn: &mut Connection,
    f: impl FnOnce(&mut rusqlite::Transaction<'_>) -> Result<T, DbError>,
) -> Result<T, DbError> {
    let mut tx = conn.transaction().map_err(DbError::from)?;
    let result = f(&mut tx)?;
    tx.commit().map_err(DbError::from)?;
    Ok(result)
}

#[cfg(all(test, feature = "sqlite"))]
#[path = "../tests/query.rs"]
mod tests;

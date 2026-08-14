//! SQLite-backed fact store with entity resolution and trust scoring.
//!
//! Ported from `hermes-agent/plugins/memory/holographic/store.py`.
//! Uses `rusqlite` via `fluent_db::store::SqliteStore` (WAL mode, FTS5, and
//! the same schema as Hermes).
//!
//! Thread safety: the connection is a `std::sync::Mutex` inside `SqliteStore`
//! (the historical `tokio::sync::Mutex` around a single connection was pure
//! ceremony — every operation here is synchronous rusqlite work with no await
//! points between acquisition and release). The async entry points offload
//! that synchronous SQLite work through `tokio::task::spawn_blocking`, so a
//! tokio worker thread is never blocked while the connection mutex is held.

use crate::plugins::holographic::hrr;
use crate::types::MemoryError;
use fluent_db::error::DbError;
use fluent_db::store::SqliteStore;
use rusqlite::Connection;
use std::path::PathBuf;
use std::sync::Arc;

/// Trust adjustment constants.
const HELPFUL_DELTA: f64 = 0.05;
const UNHELPFUL_DELTA: f64 = -0.10;
const TRUST_MIN: f64 = 0.0;
const TRUST_MAX: f64 = 1.0;

/// Schema DDL — identical to Hermes, compiled into the binary.
const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS facts (
    fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL UNIQUE,
    category        TEXT DEFAULT 'general',
    tags            TEXT DEFAULT '',
    trust_score     REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    helpful_count   INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hrr_vector      BLOB
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    entity_type TEXT DEFAULT 'unknown',
    aliases     TEXT DEFAULT '',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fact_entities (
    fact_id   INTEGER REFERENCES facts(fact_id),
    entity_id INTEGER REFERENCES entities(entity_id),
    PRIMARY KEY (fact_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_facts_trust    ON facts(trust_score DESC);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
CREATE INDEX IF NOT EXISTS idx_entities_name  ON entities(name);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
    USING fts5(content, tags, content=facts, content_rowid=fact_id);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TABLE IF NOT EXISTS memory_banks (
    bank_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    bank_name  TEXT NOT NULL UNIQUE,
    vector     BLOB NOT NULL,
    dim        INTEGER NOT NULL,
    fact_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
";

/// Configuration for the holographic store.
#[derive(Debug, Clone)]
pub struct StoreConfig {
    /// Path to the SQLite database file.
    pub db_path: PathBuf,
    /// Default trust score for new facts.
    pub default_trust: f64,
    /// HRR vector dimensions.
    pub hrr_dim: usize,
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            db_path: PathBuf::from("memory_store.db"),
            default_trust: 0.5,
            hrr_dim: 1024,
        }
    }
}

/// A fact stored in the database.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Fact {
    /// Unique fact identifier.
    pub fact_id: i64,
    /// Fact content text.
    pub content: String,
    /// Category label (e.g. `"user_pref"`, `"project"`, `"general"`).
    pub category: String,
    /// Comma-separated tags.
    pub tags: String,
    /// Trust score in the range [0.0, 1.0].
    pub trust_score: f64,
    /// Number of times this fact has been retrieved.
    pub retrieval_count: i64,
    /// Number of positive feedback ratings.
    pub helpful_count: i64,
    /// Creation timestamp (SQLite `CURRENT_TIMESTAMP`).
    pub created_at: String,
    /// Last update timestamp.
    pub updated_at: String,
}

/// SQLite-backed fact store with entity resolution and trust scoring.
///
/// Thread safety: the underlying `SqliteStore` serializes all access through
/// a single connection. SQLite WAL mode allows concurrent reads while the
/// connection mutex serializes writes. Async methods offload the synchronous
/// rusqlite work via `tokio::task::spawn_blocking`.
pub struct HolographicStore {
    config: StoreConfig,
    store: Arc<SqliteStore>,
}

impl HolographicStore {
    /// Open or create the store. Enables WAL mode and creates schema.
    pub fn open(config: StoreConfig) -> Result<Self, MemoryError> {
        // Ensure parent directory exists
        if let Some(parent) = config.db_path.parent() {
            common_core::io::ensure_dir(parent).map_err(|e| {
                MemoryError::InitFailed(format!(
                    "failed to create db directory {}: {e}",
                    parent.display()
                ))
            })?;
        }

        let store = Arc::new(
            SqliteStore::open(&config.db_path)
                .map_err(|e| MemoryError::InitFailed(format!("failed to open db: {e}")))?,
        );

        // Create schema
        store
            .init_schema(SCHEMA)
            .map_err(|e| MemoryError::InitFailed(format!("failed to init schema: {e}")))?;

        // Migrate: add hrr_vector column if missing (idempotent)
        store
            .with_conn(|conn| {
                fluent_db::migrate::ensure_column(conn, "facts", "hrr_vector", "hrr_vector BLOB")
            })
            .map_err(|e| MemoryError::InitFailed(format!("failed to migrate schema: {e}")))?;

        Ok(Self { config, store })
    }

    /// Insert a fact. Deduplicates by content (UNIQUE constraint).
    /// Returns the fact_id.
    pub async fn add_fact(
        &self,
        content: &str,
        category: &str,
        tags: &str,
    ) -> Result<i64, MemoryError> {
        let content = content.trim().to_string();
        if content.is_empty() {
            return Err(MemoryError::IngestionFailed(
                "content must not be empty".into(),
            ));
        }

        let store = Arc::clone(&self.store);
        let default_trust = self.config.default_trust;
        let hrr_dim = self.config.hrr_dim;
        let category = category.to_string();
        let tags = tags.to_string();

        tokio::task::spawn_blocking(move || {
            store.with_conn(|conn| {
                // Try insert; on duplicate, return existing id
                let result = conn.execute(
                    "INSERT INTO facts (content, category, tags, trust_score) VALUES (?1, ?2, ?3, ?4)",
                    rusqlite::params![content, category, tags, default_trust],
                );

                let fact_id = match result {
                    Ok(_) => conn.last_insert_rowid(),
                    Err(e) => {
                        // Unique-violation classification is centralized in
                        // `DbError::DuplicateEntry`.
                        let db_err = DbError::from(e);
                        match db_err {
                            DbError::DuplicateEntry(_) => conn
                                .query_row(
                                    "SELECT fact_id FROM facts WHERE content = ?1",
                                    rusqlite::params![content],
                                    |row| row.get::<_, i64>(0),
                                )
                                .map_err(DbError::from)?,
                            other => return Err(other),
                        }
                    }
                };

                // Extract and link entities
                let entities = Self::extract_entities(&content);
                for entity_name in &entities {
                    let entity_id = Self::resolve_entity(conn, entity_name)?;
                    Self::link_fact_entity(conn, fact_id, entity_id)?;
                }

                // Compute HRR vector
                Self::compute_hrr_vector(conn, fact_id, &content, &entities, hrr_dim)?;

                Ok(fact_id)
            })
        })
        .await
        .map_err(|e| MemoryError::IngestionFailed(format!("add_fact blocking task failed: {e}")))?
        .map_err(MemoryError::from)
    }

    /// Full-text search over facts using FTS5.
    pub async fn search_facts(
        &self,
        query: &str,
        category: Option<&str>,
        min_trust: f64,
        limit: usize,
    ) -> Result<Vec<Fact>, MemoryError> {
        let query = query.trim().to_string();
        if query.is_empty() {
            return Ok(vec![]);
        }

        let store = Arc::clone(&self.store);
        let category = category.map(str::to_string);

        tokio::task::spawn_blocking(move || {
            store.with_conn(|conn| {
                let sql = if category.is_some() {
                    "SELECT f.fact_id, f.content, f.category, f.tags,
                                f.trust_score, f.retrieval_count, f.helpful_count,
                                f.created_at, f.updated_at
                         FROM facts f
                         JOIN facts_fts fts ON fts.rowid = f.fact_id
                         WHERE facts_fts MATCH ?1
                           AND f.trust_score >= ?2
                           AND f.category = ?3
                         ORDER BY fts.rank, f.trust_score DESC
                         LIMIT ?4"
                        .to_string()
                } else {
                    "SELECT f.fact_id, f.content, f.category, f.tags,
                                f.trust_score, f.retrieval_count, f.helpful_count,
                                f.created_at, f.updated_at
                         FROM facts f
                         JOIN facts_fts fts ON fts.rowid = f.fact_id
                         WHERE facts_fts MATCH ?1
                           AND f.trust_score >= ?2
                         ORDER BY fts.rank, f.trust_score DESC
                         LIMIT ?3"
                        .to_string()
                };

                let results = if let Some(cat) = category.as_deref() {
                    let mut stmt = conn.prepare(&sql).map_err(DbError::from)?;
                    let rows = stmt
                        .query_map(
                            rusqlite::params![query, min_trust, cat, limit as i64],
                            |row| {
                                Ok(Fact {
                                    fact_id: row.get(0)?,
                                    content: row.get(1)?,
                                    category: row.get(2)?,
                                    tags: row.get(3)?,
                                    trust_score: row.get(4)?,
                                    retrieval_count: row.get(5)?,
                                    helpful_count: row.get(6)?,
                                    created_at: row.get(7)?,
                                    updated_at: row.get(8)?,
                                })
                            },
                        )
                        .map_err(DbError::from)?;
                    rows.filter_map(|r| r.ok()).collect::<Vec<_>>()
                } else {
                    let mut stmt = conn.prepare(&sql).map_err(DbError::from)?;
                    let rows = stmt
                        .query_map(rusqlite::params![query, min_trust, limit as i64], |row| {
                            Ok(Fact {
                                fact_id: row.get(0)?,
                                content: row.get(1)?,
                                category: row.get(2)?,
                                tags: row.get(3)?,
                                trust_score: row.get(4)?,
                                retrieval_count: row.get(5)?,
                                helpful_count: row.get(6)?,
                                created_at: row.get(7)?,
                                updated_at: row.get(8)?,
                            })
                        })
                        .map_err(DbError::from)?;
                    rows.filter_map(|r| r.ok()).collect::<Vec<_>>()
                };

                // Increment retrieval counts
                if !results.is_empty() {
                    let ids: Vec<i64> = results.iter().map(|f| f.fact_id).collect();
                    let placeholders = common_core::sqlite::in_clause(ids.len());
                    let sql = format!(
                        "UPDATE facts SET retrieval_count = retrieval_count + 1 WHERE fact_id IN ({placeholders})"
                    );
                    conn.execute(&sql, rusqlite::params_from_iter(ids.iter().copied()))
                        .map_err(DbError::from)?;
                }

                Ok(results)
            })
        })
        .await
        .map_err(|e| MemoryError::QueryFailed(format!("search_facts blocking task failed: {e}")))?
        .map_err(MemoryError::from)
    }

    /// Record user feedback and adjust trust asymmetrically.
    pub async fn record_feedback(
        &self,
        fact_id: i64,
        helpful: bool,
    ) -> Result<serde_json::Value, MemoryError> {
        let store = Arc::clone(&self.store);

        tokio::task::spawn_blocking(move || {
            store.with_conn(|conn| {
                let (old_trust, old_helpful) = conn
                    .query_row(
                        "SELECT trust_score, helpful_count FROM facts WHERE fact_id = ?1",
                        rusqlite::params![fact_id],
                        |row| Ok((row.get::<_, f64>(0)?, row.get::<_, i64>(1)?)),
                    )
                    .map_err(DbError::from)?;

                let delta = if helpful {
                    HELPFUL_DELTA
                } else {
                    UNHELPFUL_DELTA
                };
                let new_trust = clamp_trust(old_trust + delta);
                let helpful_inc = if helpful { 1 } else { 0 };

                conn.execute(
                    "UPDATE facts SET trust_score = ?1, helpful_count = helpful_count + ?2,
                     updated_at = CURRENT_TIMESTAMP WHERE fact_id = ?3",
                    rusqlite::params![new_trust, helpful_inc, fact_id],
                )
                .map_err(DbError::from)?;

                Ok(serde_json::json!({
                    "fact_id": fact_id,
                    "old_trust": old_trust,
                    "new_trust": new_trust,
                    "helpful_count": old_helpful + helpful_inc,
                }))
            })
        })
        .await
        .map_err(|e| {
            MemoryError::QueryFailed(format!("record_feedback blocking task failed: {e}"))
        })?
        .map_err(MemoryError::from)
    }

    // ── Entity helpers ──────────────────────────────────────────

    /// Extract entity candidates from text using regex rules.
    fn extract_entities(text: &str) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        let mut candidates = Vec::new();

        // Capitalized multi-word phrases
        let re_cap = regex::Regex::new(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b").unwrap();
        for m in re_cap.find_iter(text) {
            let name = m.as_str().trim().to_string();
            let lower = name.to_lowercase();
            if !lower.is_empty() && seen.insert(lower) {
                candidates.push(name);
            }
        }

        // Double-quoted terms
        let re_dq = regex::Regex::new(r#""([^"]+)""#).unwrap();
        for m in re_dq.captures_iter(text) {
            if let Some(name) = m.get(1) {
                let name = name.as_str().trim().to_string();
                let lower = name.to_lowercase();
                if !lower.is_empty() && seen.insert(lower) {
                    candidates.push(name);
                }
            }
        }

        candidates
    }

    /// Find an existing entity by name or create one.
    fn resolve_entity(conn: &Connection, name: &str) -> Result<i64, DbError> {
        // Exact name match
        let result = conn.query_row(
            "SELECT entity_id FROM entities WHERE name = ?1",
            rusqlite::params![name],
            |row| row.get::<_, i64>(0),
        );

        match result {
            Ok(id) => Ok(id),
            Err(_) => {
                // Create new entity
                conn.execute(
                    "INSERT INTO entities (name) VALUES (?1)",
                    rusqlite::params![name],
                )
                .map_err(DbError::from)?;
                Ok(conn.last_insert_rowid())
            }
        }
    }

    /// Link a fact to an entity (ignore duplicate).
    fn link_fact_entity(conn: &Connection, fact_id: i64, entity_id: i64) -> Result<(), DbError> {
        conn.execute(
            "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?1, ?2)",
            rusqlite::params![fact_id, entity_id],
        )
        .map_err(DbError::from)?;
        Ok(())
    }

    /// Compute and store HRR vector for a fact.
    fn compute_hrr_vector(
        conn: &Connection,
        fact_id: i64,
        content: &str,
        entities: &[String],
        hrr_dim: usize,
    ) -> Result<(), DbError> {
        let vector = hrr::encode_fact(content, entities, hrr_dim);
        let bytes = hrr::phases_to_bytes(&vector);
        conn.execute(
            "UPDATE facts SET hrr_vector = ?1 WHERE fact_id = ?2",
            rusqlite::params![bytes, fact_id],
        )
        .map_err(DbError::from)?;
        Ok(())
    }
}

fn clamp_trust(value: f64) -> f64 {
    value.clamp(TRUST_MIN, TRUST_MAX)
}

//! SQLite-backed fact store with entity resolution and trust scoring.
//!
//! Ported from `hermes-agent/plugins/memory/holographic/store.py`.
//! Uses `rusqlite` with WAL mode, FTS5, and the same schema.

use crate::plugins::holographic::hrr;
use crate::types::MemoryError;
use rusqlite::{params, Connection};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

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
/// Thread safety: The `Connection` is wrapped in `tokio::sync::Mutex`.
/// SQLite WAL mode allows concurrent reads while the mutex serializes writes.
pub struct HolographicStore {
    config: StoreConfig,
    conn: Arc<Mutex<Connection>>,
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

        let conn = common_core::sqlite::open_wal(&config.db_path)
            .map_err(|e| MemoryError::InitFailed(format!("failed to open db: {e}")))?;

        // Create schema
        conn.execute_batch(SCHEMA)?;

        // Migrate: add hrr_vector column if missing
        let columns: Vec<String> = conn
            .prepare("PRAGMA table_info(facts)")?
            .query_map([], |row| row.get::<_, String>(1))?
            .filter_map(|r| r.ok())
            .collect();
        if !columns.contains(&"hrr_vector".to_string()) {
            conn.execute_batch("ALTER TABLE facts ADD COLUMN hrr_vector BLOB")?;
        }

        Ok(Self {
            config,
            conn: Arc::new(Mutex::new(conn)),
        })
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

        let conn = self.conn.lock().await;

        // Try insert; on duplicate, return existing id
        let result = conn.execute(
            "INSERT INTO facts (content, category, tags, trust_score) VALUES (?1, ?2, ?3, ?4)",
            params![content, category, tags, self.config.default_trust],
        );

        let fact_id = match result {
            Ok(_) => conn.last_insert_rowid(),
            Err(rusqlite::Error::SqliteFailure(err, _))
                if err.code == rusqlite::ErrorCode::ConstraintViolation =>
            {
                // Duplicate content — return existing id
                conn.query_row(
                    "SELECT fact_id FROM facts WHERE content = ?1",
                    params![content],
                    |row| row.get::<_, i64>(0),
                )?
            }
            Err(e) => return Err(e.into()),
        };

        // Extract and link entities
        let entities = Self::extract_entities(&content);
        for entity_name in &entities {
            let entity_id = Self::resolve_entity(&conn, entity_name)?;
            Self::link_fact_entity(&conn, fact_id, entity_id)?;
        }

        // Compute HRR vector
        self.compute_hrr_vector(&conn, fact_id, &content, &entities)?;

        Ok(fact_id)
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

        let conn = self.conn.lock().await;

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

        let results = if let Some(cat) = category {
            let mut stmt = conn.prepare(&sql)?;
            let rows = stmt.query_map(params![query, min_trust, cat, limit as i64], |row| {
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
            })?;
            rows.filter_map(|r| r.ok()).collect::<Vec<_>>()
        } else {
            let mut stmt = conn.prepare(&sql)?;
            let rows = stmt.query_map(params![query, min_trust, limit as i64], |row| {
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
            })?;
            rows.filter_map(|r| r.ok()).collect::<Vec<_>>()
        };

        // Increment retrieval counts
        if !results.is_empty() {
            let ids: Vec<i64> = results.iter().map(|f| f.fact_id).collect();
            let placeholders: Vec<&str> = ids.iter().map(|_| "?").collect();
            let sql = format!(
                "UPDATE facts SET retrieval_count = retrieval_count + 1 WHERE fact_id IN ({})",
                placeholders.join(",")
            );
            let params: Vec<Box<dyn rusqlite::types::ToSql>> = ids
                .iter()
                .map(|id| Box::new(*id) as Box<dyn rusqlite::types::ToSql>)
                .collect();
            let param_refs: Vec<&dyn rusqlite::types::ToSql> =
                params.iter().map(|p| p.as_ref()).collect();
            conn.execute(&sql, param_refs.as_slice())?;
        }

        Ok(results)
    }

    /// Record user feedback and adjust trust asymmetrically.
    pub async fn record_feedback(
        &self,
        fact_id: i64,
        helpful: bool,
    ) -> Result<serde_json::Value, MemoryError> {
        let conn = self.conn.lock().await;

        let row = conn.query_row(
            "SELECT fact_id, trust_score, helpful_count FROM facts WHERE fact_id = ?1",
            params![fact_id],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, f64>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            },
        )?;

        let (_id, old_trust, old_helpful) = row;
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
            params![new_trust, helpful_inc, fact_id],
        )?;

        Ok(serde_json::json!({
            "fact_id": fact_id,
            "old_trust": old_trust,
            "new_trust": new_trust,
            "helpful_count": old_helpful + helpful_inc,
        }))
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
    fn resolve_entity(conn: &Connection, name: &str) -> Result<i64, MemoryError> {
        // Exact name match
        let result = conn.query_row(
            "SELECT entity_id FROM entities WHERE name = ?1",
            params![name],
            |row| row.get::<_, i64>(0),
        );

        match result {
            Ok(id) => Ok(id),
            Err(_) => {
                // Create new entity
                conn.execute("INSERT INTO entities (name) VALUES (?1)", params![name])?;
                Ok(conn.last_insert_rowid())
            }
        }
    }

    /// Link a fact to an entity (ignore duplicate).
    fn link_fact_entity(
        conn: &Connection,
        fact_id: i64,
        entity_id: i64,
    ) -> Result<(), MemoryError> {
        conn.execute(
            "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?1, ?2)",
            params![fact_id, entity_id],
        )?;
        Ok(())
    }

    /// Compute and store HRR vector for a fact.
    fn compute_hrr_vector(
        &self,
        conn: &Connection,
        fact_id: i64,
        content: &str,
        entities: &[String],
    ) -> Result<(), MemoryError> {
        let vector = hrr::encode_fact(content, entities, self.config.hrr_dim);
        let bytes = hrr::phases_to_bytes(&vector);
        conn.execute(
            "UPDATE facts SET hrr_vector = ?1 WHERE fact_id = ?2",
            params![bytes, fact_id],
        )?;
        Ok(())
    }
}

fn clamp_trust(value: f64) -> f64 {
    value.clamp(TRUST_MIN, TRUST_MAX)
}

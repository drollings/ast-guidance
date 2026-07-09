use std::path::PathBuf;
use std::sync::Mutex;

use common_core::hash::blake3_hex;
use rusqlite::{params, Connection};

use crate::schema::ValueSource;

/// Trust adjustment constants.
const HELPFUL_DELTA: f64 = 0.05;
const UNHELPFUL_DELTA: f64 = -0.10;
const TRUST_MIN: f64 = 0.0;
const TRUST_MAX: f64 = 1.0;

/// Schema DDL for the form fills store.
const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS form_fills (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    field_label     TEXT NOT NULL,
    value           TEXT NOT NULL,
    value_hash      TEXT NOT NULL,
    confidence      REAL NOT NULL,
    form_url        TEXT,
    source          TEXT NOT NULL,
    trust_score     REAL NOT NULL DEFAULT 0.5,
    retrieval_count INTEGER NOT NULL DEFAULT 0,
    helpful_count   INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_form_fills_label ON form_fills(field_label);
CREATE INDEX IF NOT EXISTS idx_form_fills_trust ON form_fills(trust_score DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS form_fills_fts
    USING fts5(field_label, value, content='form_fills', content_rowid='id');

CREATE TRIGGER IF NOT EXISTS form_fills_ai AFTER INSERT ON form_fills BEGIN
    INSERT INTO form_fills_fts(rowid, field_label, value)
        VALUES (new.id, new.field_label, new.value);
END;

CREATE TRIGGER IF NOT EXISTS form_fills_ad AFTER DELETE ON form_fills BEGIN
    INSERT INTO form_fills_fts(form_fills_fts, rowid, field_label, value)
        VALUES ('delete', old.id, old.field_label, old.value);
END;

CREATE TRIGGER IF NOT EXISTS form_fills_au AFTER UPDATE ON form_fills BEGIN
    INSERT INTO form_fills_fts(form_fills_fts, rowid, field_label, value)
        VALUES ('delete', old.id, old.field_label, old.value);
    INSERT INTO form_fills_fts(rowid, field_label, value)
        VALUES (new.id, new.field_label, new.value);
END;
";

/// Configuration for the form fill store.
#[derive(Debug, Clone)]
pub struct FormFillConfig {
    /// Path to the SQLite database file.
    pub db_path: PathBuf,
    /// Maximum number of entries to retain.
    pub max_entries: usize,
    /// Default trust score for new entries.
    pub default_trust: f64,
}

impl Default for FormFillConfig {
    fn default() -> Self {
        Self {
            db_path: PathBuf::from("form-fills.db"),
            max_entries: 10_000,
            default_trust: 0.5,
        }
    }
}

/// A single form fill memory entry.
#[derive(Debug, Clone)]
pub struct FormFillEntry {
    pub id: i64,
    pub field_label: String,
    pub value: String,
    pub value_hash: String,
    pub confidence: f32,
    pub form_url: Option<String>,
    pub source: String,
    pub trust_score: f64,
    pub retrieval_count: i64,
    pub helpful_count: i64,
}

/// SQLite-backed form fill memory store with FTS5 search.
pub struct FormFillStore {
    conn: Mutex<Connection>,
    config: FormFillConfig,
}

impl FormFillStore {
    /// Open or create the store at the configured path.
    pub fn open(config: FormFillConfig) -> Result<Self, crate::error::CopilotError> {
        let conn = Connection::open(&config.db_path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self {
            conn: Mutex::new(conn),
            config,
        })
    }

    /// Open an in-memory store (no persistence). Used as fallback when
    /// the file-backed store fails to open.
    pub fn open_memory() -> Result<Self, crate::error::CopilotError> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self {
            conn: Mutex::new(conn),
            config: FormFillConfig::default(),
        })
    }

    /// Record a form fill entry.
    pub fn record(
        &self,
        field_label: &str,
        value: &str,
        confidence: f32,
        form_url: Option<&str>,
        source: &ValueSource,
    ) -> Result<(), crate::error::CopilotError> {
        let value_hash = blake3_hex(value.as_bytes());
        let source_str = serde_json::to_string(source).unwrap_or_default();
        let source_str = source_str.trim_matches('"');
        let conn = self
            .conn
            .lock()
            .map_err(|e| crate::error::CopilotError::Internal(format!("lock poisoned: {e}")))?;
        conn.execute(
            "INSERT INTO form_fills (field_label, value, value_hash, confidence, form_url, source, trust_score)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                field_label,
                value,
                value_hash,
                f64::from(confidence),
                form_url,
                source_str,
                self.config.default_trust,
            ],
        )?;
        Ok(())
    }

    /// Search for matching form fills by field label using FTS5.
    pub fn search(
        &self,
        field_label: &str,
        limit: usize,
    ) -> Result<Vec<FormFillEntry>, crate::error::CopilotError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| crate::error::CopilotError::Internal(format!("lock poisoned: {e}")))?;
        let mut stmt = conn.prepare(
            "SELECT f.id, f.field_label, f.value, f.value_hash, f.confidence,
                    f.form_url, f.source, f.trust_score, f.retrieval_count, f.helpful_count
             FROM form_fills f
             WHERE f.field_label = ?1
             ORDER BY f.trust_score DESC, f.created_at DESC
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![field_label, limit as i64], |row| {
            Ok(FormFillEntry {
                id: row.get(0)?,
                field_label: row.get(1)?,
                value: row.get(2)?,
                value_hash: row.get(3)?,
                confidence: row.get::<_, f64>(4)? as f32,
                form_url: row.get(5)?,
                source: row.get(6)?,
                trust_score: row.get(7)?,
                retrieval_count: row.get(8)?,
                helpful_count: row.get(9)?,
            })
        })?;
        let mut entries = Vec::new();
        for row in rows {
            entries.push(row?);
        }
        Ok(entries)
    }

    /// Record user feedback for a form fill, adjusting trust score.
    pub fn record_feedback(
        &self,
        field_label: &str,
        value_hash: &str,
        helpful: bool,
    ) -> Result<(), crate::error::CopilotError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| crate::error::CopilotError::Internal(format!("lock poisoned: {e}")))?;
        let delta = if helpful {
            HELPFUL_DELTA
        } else {
            UNHELPFUL_DELTA
        };
        conn.execute(
            "UPDATE form_fills
             SET trust_score = MAX(?1, MIN(?2, trust_score + ?3)),
                 helpful_count = helpful_count + ?4,
                 retrieval_count = retrieval_count + 1,
                 updated_at = datetime('now')
             WHERE field_label = ?5 AND value_hash = ?6",
            params![
                TRUST_MIN,
                TRUST_MAX,
                delta,
                i32::from(helpful),
                field_label,
                value_hash,
            ],
        )?;
        Ok(())
    }

    /// Extract user preference patterns from session messages (no-op for now).
    pub fn on_session_end(
        &self,
        _messages: &[TurnMessage],
    ) -> Result<(), crate::error::CopilotError> {
        // Future: extract "I prefer X" patterns via regex.
        Ok(())
    }

    /// Get the total number of entries.
    pub fn count(&self) -> Result<i64, crate::error::CopilotError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| crate::error::CopilotError::Internal(format!("lock poisoned: {e}")))?;
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM form_fills", [], |row| row.get(0))?;
        Ok(count)
    }
}

/// Re-export for convenience.
pub use memory_plugin::types::TurnMessage;

#[cfg(test)]
mod tests {
    use super::*;

    fn test_store() -> FormFillStore {
        FormFillStore::open_memory().unwrap()
    }

    #[test]
    fn record_and_search() {
        let store = test_store();
        store
            .record("First Name", "Ada", 0.95, None, &ValueSource::Resume)
            .unwrap();
        store
            .record(
                "First Name",
                "Ada Lovelace",
                0.8,
                None,
                &ValueSource::LlmGenerated,
            )
            .unwrap();

        let results = store.search("First Name", 10).unwrap();
        assert_eq!(results.len(), 2);
        // Higher trust first.
        assert!(results[0].trust_score >= results[1].trust_score);
    }

    #[test]
    fn search_returns_empty_for_unknown_label() {
        let store = test_store();
        let results = store.search("Nonexistent", 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn record_feedback_adjusts_trust() {
        let store = test_store();
        store
            .record("Email", "ada@example.com", 1.0, None, &ValueSource::Resume)
            .unwrap();

        let entries = store.search("Email", 1).unwrap();
        let original_trust = entries[0].trust_score;

        // Helpful feedback.
        store
            .record_feedback("Email", &entries[0].value_hash, true)
            .unwrap();
        let entries = store.search("Email", 1).unwrap();
        assert!(entries[0].trust_score > original_trust);
    }

    #[test]
    fn record_feedback_decreases_on_unhelpful() {
        let store = test_store();
        store
            .record("Phone", "555-1234", 1.0, None, &ValueSource::Resume)
            .unwrap();

        let entries = store.search("Phone", 1).unwrap();
        let original_trust = entries[0].trust_score;

        store
            .record_feedback("Phone", &entries[0].value_hash, false)
            .unwrap();
        let entries = store.search("Phone", 1).unwrap();
        assert!(entries[0].trust_score < original_trust);
    }

    #[test]
    fn count_reflects_entries() {
        let store = test_store();
        assert_eq!(store.count().unwrap(), 0);
        store
            .record("Name", "Test", 1.0, None, &ValueSource::Resume)
            .unwrap();
        assert_eq!(store.count().unwrap(), 1);
    }

    #[test]
    fn on_session_end_is_noop() {
        let store = test_store();
        store.on_session_end(&[]).unwrap();
    }
}

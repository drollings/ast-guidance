use std::path::PathBuf;
use std::sync::Mutex;

use fluent_types::NodeId;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LedgerError {
    #[error("database error: {0}")]
    Db(String),
    #[error("node not found: {0:?}")]
    NotFound(NodeId),
}

/// Full-detail ledger entry recorded before any filter runs (§5).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerEntry {
    pub node_id: NodeId,
    pub session_id: String,
    pub request_id: String,
    pub role: String,
    pub content: String,
    pub turn_index: u64,
    pub accepted: bool,
    pub acceptance_score: Option<f64>,
    pub active_lod: u8,
    pub parent_id: Option<NodeId>,
    pub step_id: Option<String>,
    pub metadata: serde_json::Map<String, serde_json::Value>,
    pub created_at: u64,
}

pub struct ContentNodeLedger {
    db: Mutex<rusqlite::Connection>,
    next_id: Mutex<i64>,
}

impl ContentNodeLedger {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, LedgerError> {
        let db = rusqlite::Connection::open(path.into())
            .map_err(|e| LedgerError::Db(e.to_string()))?;
        db.execute_batch(
            "CREATE TABLE IF NOT EXISTS ledger (
                node_id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                turn_index INTEGER NOT NULL DEFAULT 0,
                accepted INTEGER NOT NULL DEFAULT 1,
                acceptance_score REAL,
                active_lod INTEGER NOT NULL DEFAULT 0,
                parent_id INTEGER,
                step_id TEXT,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at INTEGER NOT NULL DEFAULT (unixepoch())
            );
            CREATE INDEX IF NOT EXISTS idx_ledger_session ON ledger(session_id, turn_index);",
        )
        .map_err(|e| LedgerError::Db(e.to_string()))?;
        Ok(Self {
            db: Mutex::new(db),
            next_id: Mutex::new(1),
        })
    }

    pub fn record_request(
        &self,
        session_id: &str,
        request_id: &str,
        content: &str,
    ) -> Result<NodeId, LedgerError> {
        let mut next = self.next_id.lock().unwrap();
        let id = NodeId::from_int(*next);
        *next += 1;

        let db = self.db.lock().unwrap();
        db.execute(
            "INSERT INTO ledger (node_id, session_id, request_id, role, content)
             VALUES (?1, ?2, ?3, 'user', ?4)",
            rusqlite::params![id.as_int(), session_id, request_id, content],
        )
        .map_err(|e| LedgerError::Db(e.to_string()))?;

        Ok(id)
    }

    pub fn record_result(
        &self,
        node_id: NodeId,
        accepted: bool,
        score: Option<f64>,
        content: &str,
    ) -> Result<(), LedgerError> {
        let db = self.db.lock().unwrap();
        db.execute(
            "UPDATE ledger SET accepted = ?1, acceptance_score = ?2, content = ?3
             WHERE node_id = ?4",
            rusqlite::params![accepted, score, content, node_id.as_int()],
        )
        .map_err(|e| LedgerError::Db(e.to_string()))?;
        Ok(())
    }

    pub fn collapse_node(
        &self,
        node_id: NodeId,
        summary: &str,
        lod: u8,
    ) -> Result<(), LedgerError> {
        let db = self.db.lock().unwrap();
        db.execute(
            "UPDATE ledger SET content = ?1, active_lod = ?2 WHERE node_id = ?3",
            rusqlite::params![summary, lod, node_id.as_int()],
        )
        .map_err(|e| LedgerError::Db(e.to_string()))?;
        Ok(())
    }

    pub fn get_session_entries(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<LedgerEntry>, LedgerError> {
        let db = self.db.lock().unwrap();
        let mut stmt = db
            .prepare(
                "SELECT node_id, session_id, request_id, role, content,
                        turn_index, accepted, acceptance_score, active_lod,
                        parent_id, step_id, metadata, created_at
                 FROM ledger WHERE session_id = ?1
                 ORDER BY turn_index DESC LIMIT ?2",
            )
            .map_err(|e| LedgerError::Db(e.to_string()))?;

        let entries = stmt
            .query_map(rusqlite::params![session_id, limit as i64], |row| {
                Ok(LedgerEntry {
                    node_id: NodeId::from_int(row.get(0)?),
                    session_id: row.get(1)?,
                    request_id: row.get(2)?,
                    role: row.get(3)?,
                    content: row.get(4)?,
                    turn_index: row.get(5)?,
                    accepted: row.get(6)?,
                    acceptance_score: row.get(7)?,
                    active_lod: row.get(8)?,
                    parent_id: row.get::<_, Option<i64>>(9)?.map(NodeId::from_int),
                    step_id: row.get(10)?,
                    metadata: serde_json::from_str(row.get::<_, String>(11)?.as_str())
                        .unwrap_or_default(),
                    created_at: row.get(12)?,
                })
            })
            .map_err(|e| LedgerError::Db(e.to_string()))?;

        let mut result = Vec::new();
        for entry in entries {
            result.push(entry.map_err(|e| LedgerError::Db(e.to_string()))?);
        }
        Ok(result)
    }
}

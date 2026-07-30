use super::LibraryError;

impl super::Library {
    pub fn init_router_schema(&self) -> Result<(), LibraryError> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(
            "ALTER TABLE context_nodes ADD COLUMN session_id TEXT;
             ALTER TABLE context_nodes ADD COLUMN role TEXT DEFAULT 'user';
             ALTER TABLE context_nodes ADD COLUMN turn_index INTEGER DEFAULT 0;
             ALTER TABLE context_nodes ADD COLUMN accepted INTEGER DEFAULT 1;
             ALTER TABLE context_nodes ADD COLUMN acceptance_score REAL;
             ALTER TABLE context_nodes ADD COLUMN parent_id INTEGER REFERENCES context_nodes(id);
             ALTER TABLE context_nodes ADD COLUMN step_id TEXT;
             ALTER TABLE context_nodes ADD COLUMN step_status TEXT;

             CREATE TABLE IF NOT EXISTS kv_cache_snapshots (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 model_name TEXT NOT NULL,
                 adapter_name TEXT,
                 session_id TEXT NOT NULL,
                 file_path TEXT NOT NULL,
                 created_at TEXT DEFAULT (datetime('now')),
                 last_used_at TEXT DEFAULT (datetime('now')),
                 token_count INTEGER NOT NULL,
                 llama_cpp_version TEXT,
                 model_quant TEXT,
                 base_model_hash TEXT,
                 UNIQUE(model_name, adapter_name, session_id)
             );

             CREATE TABLE IF NOT EXISTS session_checkpoints (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 session_id TEXT NOT NULL,
                 checkpoint_name TEXT NOT NULL,
                 node_id INTEGER NOT NULL REFERENCES context_nodes(id),
                 created_at TEXT DEFAULT (datetime('now')),
                 UNIQUE(session_id, checkpoint_name)
             );

             CREATE TABLE IF NOT EXISTS model_catalog (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 name TEXT NOT NULL UNIQUE,
                 role TEXT NOT NULL,
                 path TEXT,
                 model_name TEXT,
                 provider TEXT,
                 api_base TEXT,
                 api_key_env TEXT,
                 credentials_ref TEXT,
                 context_size INTEGER,
                 quant TEXT,
                 base_model TEXT,
                 created_at TEXT DEFAULT (datetime('now'))
             );",
        )?;
        Ok(())
    }
}

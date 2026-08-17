use fluent_db::error::DbError;

use super::LibraryError;

impl super::Library {
    pub fn init_schema(&self) -> Result<(), LibraryError> {
        self.store.init_schema(
            "CREATE TABLE IF NOT EXISTS context_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                source TEXT NOT NULL DEFAULT '',
                lod TEXT NOT NULL DEFAULT '[]',
                embedding BLOB,
                capabilities BLOB,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_node_id INTEGER NOT NULL,
                target_node_id INTEGER NOT NULL,
                edge_type TEXT NOT NULL DEFAULT 'depends',
                weight REAL NOT NULL DEFAULT 1.0,
                FOREIGN KEY (source_node_id) REFERENCES context_nodes(id),
                FOREIGN KEY (target_node_id) REFERENCES context_nodes(id)
            );

            CREATE TABLE IF NOT EXISTS wasm_tools (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                path TEXT NOT NULL,
                capabilities TEXT NOT NULL DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS targets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                bit_index INTEGER NOT NULL,
                depends BLOB,
                provides BLOB,
                essential INTEGER NOT NULL DEFAULT 0,
                command TEXT NOT NULL DEFAULT ''
            );",
        )?;
        self.store.with_conn(|conn| {
            common_core::sqlite::init_embedding_cache(conn).map_err(DbError::from)?;
            common_core::sqlite::run_batch(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_nodes_name_source
                    ON context_nodes(name, source);

                CREATE TABLE IF NOT EXISTS entity_types (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id INTEGER NOT NULL,
                    type_iri TEXT NOT NULL,
                    FOREIGN KEY (node_id) REFERENCES context_nodes(id)
                );

                CREATE TABLE IF NOT EXISTS entity_hierarchy (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subclass_iri TEXT NOT NULL,
                    superclass_iri TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_node_id);
                CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_node_id);
                CREATE INDEX IF NOT EXISTS idx_nodes_name ON context_nodes(name);
                CREATE INDEX IF NOT EXISTS idx_entity_types_node ON entity_types(node_id);
                CREATE INDEX IF NOT EXISTS idx_entity_types_iri ON entity_types(type_iri);",
            )
            .map_err(DbError::from)?;
            Ok(())
        })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn init_schema_is_idempotent() {
        let lib = crate::db::Library::open_in_memory().expect("in-memory db");
        lib.init_schema().expect("first init");
        lib.init_schema().expect("second init (IF NOT EXISTS)");
        // Core tables are queryable after init.
        lib.insert_node(&crate::tests::common::make_node("a", "s")).expect("insert");
        assert_eq!(lib.node_count().expect("count"), 1);
    }

    #[test]
    fn open_in_memory_initializes_schema() {
        let lib = crate::db::Library::open_in_memory().expect("in-memory db");
        // The embedding-cache table created by init_schema is usable.
        lib.cache_embedding("k", "q", &[0.1]).expect("cache");
        assert!(lib.get_cached_embedding("k").expect("get").is_some());
    }
}

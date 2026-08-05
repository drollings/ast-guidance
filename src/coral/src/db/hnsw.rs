use std::collections::HashMap;

use fluent_db::vector::try_bytes_to_vec;
use fluent_types::{KnnHit, NodeId};

use super::LibraryError;

impl super::Library {
    pub(crate) fn hnsw_insert(&self, node_id: i64, embedding: &[f32]) {
        self.hnsw.insert(node_id, embedding);
    }

    pub fn rebuild_hnsw(&self) -> Result<usize, LibraryError> {
        let rows = self.store.query_rows(
            "SELECT id, embedding FROM context_nodes WHERE embedding IS NOT NULL",
            &[],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?)),
        )?;
        let count = rows.len();
        self.hnsw.rebuild_from(rows.into_iter(), try_bytes_to_vec)?;
        Ok(count)
    }

    pub(crate) fn hnsw_search(&self, query_vec: &[f32], k: usize) -> Option<Vec<KnnHit>> {
        let neighbours = self.hnsw.search(query_vec, k);
        if neighbours.is_empty() {
            return Some(Vec::new());
        }

        // Lock ordering is preserved: `HnswIndex::search` releases the
        // `hnsw`/`id_map` guards before returning, and only then do we touch
        // the connection (hnsw → id_map → conn, never inverted — R9).
        let id_map = self.hnsw.id_map_snapshot();

        let mut node_ids: Vec<i64> = Vec::with_capacity(neighbours.len());
        for (d_id, _distance) in &neighbours {
            if *d_id < id_map.len() {
                node_ids.push(id_map[*d_id]);
            }
        }
        if node_ids.is_empty() {
            return Some(Vec::new());
        }

        // M8.2: resolve every neighbour's name in a single parameterized
        // `WHERE id IN (...)` query instead of one query per neighbour.
        let name_by_id: HashMap<i64, String> = {
            let placeholders = common_core::sqlite::in_clause(node_ids.len());
            let sql = format!("SELECT id, name FROM context_nodes WHERE id IN ({placeholders})");
            let rows = self
                .store
                .with_conn(|conn| {
                    fluent_db::query::query_rows_from_iter(
                        conn,
                        &sql,
                        node_ids.iter().copied(),
                        |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
                    )
                })
                .ok()?;
            rows.into_iter().collect()
        };

        let mut results = Vec::with_capacity(neighbours.len());
        for (d_id, distance) in neighbours {
            if d_id >= id_map.len() {
                continue;
            }
            let node_id = id_map[d_id];
            let Some(name) = name_by_id.get(&node_id) else {
                continue;
            };
            results.push(KnnHit {
                node_id: NodeId::from_int(node_id),
                distance,
                name: name.as_str().into(),
            });
        }
        Some(results)
    }

    pub fn has_hnsw(&self) -> bool {
        self.hnsw.is_built()
    }

    pub fn hnsw_len(&self) -> usize {
        self.hnsw.len()
    }
}

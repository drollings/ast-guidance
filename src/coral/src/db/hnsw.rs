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

        // Resolve every neighbour's name in a single parameterized
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::common::make_node;
    use fluent_types::ContentNode;

    fn lib() -> crate::db::Library {
        crate::db::Library::open_in_memory().expect("in-memory db")
    }

    fn insert_embedded(lib: &crate::db::Library, name: &str, emb: Vec<f32>) -> NodeId {
        lib.insert_node(&ContentNode {
            embedding: Some(emb),
            ..make_node(name, "s")
        })
        .expect("insert embedded")
    }

    #[test]
    fn index_is_not_built_until_rebuilt() {
        let lib = lib();
        assert!(!lib.has_hnsw());
        assert_eq!(lib.hnsw_len(), 0);
    }

    #[test]
    fn rebuild_hnsw_indexes_embedded_nodes() {
        let lib = lib();
        insert_embedded(&lib, "a", vec![1.0, 0.0]);
        insert_embedded(&lib, "b", vec![0.0, 1.0]);
        let count = lib.rebuild_hnsw().expect("rebuild");
        assert_eq!(count, 2);
        assert!(lib.has_hnsw());
        assert_eq!(lib.hnsw_len(), 2);
    }

    #[test]
    fn hnsw_search_routes_to_nearest_embedding() {
        let lib = lib();
        insert_embedded(&lib, "right", vec![1.0, 0.0]);
        insert_embedded(&lib, "up", vec![0.0, 1.0]);
        lib.rebuild_hnsw().expect("rebuild");

        let hits = lib.hnsw_search(&[0.9, 0.0], 1).expect("hnsw search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].name.as_str(), "right");
        assert!(hits[0].distance < 0.1, "nearest distance is small");

        // k larger than the index returns all neighbours.
        let hits = lib.hnsw_search(&[0.9, 0.0], 10).expect("hnsw search all");
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn rebuild_overwrites_previous_index() {
        let lib = lib();
        insert_embedded(&lib, "a", vec![1.0]);
        lib.rebuild_hnsw().expect("rebuild");
        assert_eq!(lib.hnsw_len(), 1);
        // A second build on the same rows keeps the index coherent.
        lib.rebuild_hnsw().expect("rebuild again");
        assert_eq!(lib.hnsw_len(), 1);
    }
}

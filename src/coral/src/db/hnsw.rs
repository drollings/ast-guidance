use common_core::constants::HnswParams;
use fluent_types::{KnnHit, NodeId};
use rusqlite::params;
use search_vector::math::try_bytes_to_vec;

use super::LibraryError;

impl super::Library {
    pub(crate) fn hnsw_insert(&self, node_id: i64, embedding: &[f32]) {
        let mut guard = self.hnsw.write().unwrap();
        let hnsw = guard.get_or_insert_with(|| {
            let p = HnswParams::default();
            common_core::sqlite::make_hnsw(&p, p.initial_capacity)
        });

        let external_id = {
            let mut id_map = self.hnsw_id_map.lock().unwrap();
            let idx = id_map.len();
            id_map.push(node_id);
            idx
        };

        hnsw.insert((embedding, external_id));
    }

    pub fn rebuild_hnsw(&self) -> Result<usize, LibraryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt =
            conn.prepare("SELECT id, embedding FROM context_nodes WHERE embedding IS NOT NULL")?;

        let rows: Vec<(i64, Vec<u8>)> = stmt
            .query_map([], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
            })?
            .filter_map(Result::ok)
            .collect();
        drop(stmt);
        drop(conn);

        let count = rows.len();
        let p = HnswParams::default();
        let hnsw = common_core::sqlite::make_hnsw(&p, count.max(p.initial_capacity));

        let mut id_map = Vec::with_capacity(count);
        for (node_id, blob) in rows {
            if let Some(embedding) = try_bytes_to_vec(&blob) {
                if !embedding.is_empty() {
                    hnsw.insert((&embedding, id_map.len()));
                    id_map.push(node_id);
                }
            }
        }

        *self.hnsw.write().unwrap() = Some(hnsw);
        *self.hnsw_id_map.lock().unwrap() = id_map;

        Ok(count)
    }

    pub(crate) fn hnsw_search(&self, query_vec: &[f32], k: usize) -> Option<Vec<KnnHit>> {
        let guard = self.hnsw.read().ok()?;
        let hnsw = guard.as_ref()?;
        let id_map = self.hnsw_id_map.lock().ok()?;

        let neighbours = hnsw.search(query_vec, k, k);
        let conn = self.conn.lock().ok()?;

        let mut results = Vec::with_capacity(neighbours.len());
        for n in &neighbours {
            if n.d_id >= id_map.len() {
                continue;
            }
            let node_id = id_map[n.d_id];
            if let Ok(name) = conn.query_row(
                "SELECT name FROM context_nodes WHERE id = ?1",
                params![node_id],
                |row| row.get::<_, String>(0),
            ) {
                results.push(KnnHit {
                    node_id: NodeId::from_int(node_id),
                    distance: n.distance,
                    name: name.as_str().into(),
                });
            }
        }
        Some(results)
    }

    pub fn has_hnsw(&self) -> bool {
        self.hnsw.read().is_ok_and(|g| g.is_some())
    }

    pub fn hnsw_len(&self) -> usize {
        self.hnsw
            .read()
            .ok()
            .and_then(|g| g.as_ref().map(hnsw_rs::hnsw::Hnsw::get_nb_point))
            .unwrap_or(0)
    }
}

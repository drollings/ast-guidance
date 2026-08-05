use std::collections::HashMap;

use fluent_types::{ContentNode, KnnHit, NodeId, WasmTool};
use rusqlite::params;
use search_vector::math::{knn_brute_force, try_bytes_to_vec, vec_to_bytes};

use super::{blob_to_bitvec, LibraryError, MAX_KNN_CANDIDATES};

impl super::Library {
    pub fn insert_node(&self, node: &ContentNode) -> Result<NodeId, LibraryError> {
        let conn = self.conn.lock().unwrap();
        let lod_json = serde_json::to_string(&node.lod).unwrap_or_default();
        let embedding_blob = node.embedding.as_ref().map(|v| vec_to_bytes(v));
        let capabilities_blob = node.capabilities.as_deref();

        conn.execute(
            "INSERT INTO context_nodes (name, source, lod, embedding, capabilities) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![node.name.as_str(), node.source, lod_json, embedding_blob, capabilities_blob],
        )?;

        let node_id = NodeId::from_int(conn.last_insert_rowid());
        drop(conn);

        if let Some(ref emb) = node.embedding {
            self.hnsw_insert(node_id.as_int(), emb);
        }

        Ok(node_id)
    }

    pub fn find_node_by_name(&self, name: &str) -> Result<Option<NodeId>, LibraryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT id FROM context_nodes WHERE name = ?1")?;
        let result = stmt
            .query_row(params![name], |row| {
                let id: i64 = row.get(0)?;
                Ok(NodeId::from_int(id))
            })
            .ok();
        Ok(result)
    }

    /// Resolve many names to their node ids in a single parameterized query
    /// (avoids the per-name N+1 round-trip through the shared `conn` Mutex).
    pub fn find_node_ids_by_names(
        &self,
        names: &[&str],
    ) -> Result<HashMap<String, NodeId>, LibraryError> {
        if names.is_empty() {
            return Ok(HashMap::new());
        }
        let conn = self.conn.lock().unwrap();
        let placeholders = common_core::sqlite::in_clause(names.len());
        let sql = format!("SELECT id, name FROM context_nodes WHERE name IN ({placeholders})");
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(names.iter().copied()), |row| {
            let id: i64 = row.get(0)?;
            let name: String = row.get(1)?;
            Ok((name, NodeId::from_int(id)))
        })?;
        let mut map = HashMap::with_capacity(names.len());
        for (name, id) in rows.flatten() {
            map.insert(name, id);
        }
        Ok(map)
    }

    pub fn get_node(&self, node_id: NodeId) -> Result<Option<ContentNode>, LibraryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt =
            conn.prepare("SELECT id, name, source, lod, embedding, capabilities FROM context_nodes WHERE id = ?1")?;
        let result = stmt
            .query_row(params![node_id.as_int()], |row| {
                let id: i64 = row.get(0)?;
                let name: String = row.get(1)?;
                let source: String = row.get(2)?;
                let lod_json: String = row.get(3)?;
                let embedding_blob: Option<Vec<u8>> = row.get(4)?;
                let capabilities_blob: Option<Vec<u8>> = row.get(5)?;
                let lod: Vec<String> = serde_json::from_str(&lod_json).unwrap_or_default();
                let embedding = embedding_blob.and_then(|b| try_bytes_to_vec(&b));
                Ok(ContentNode {
                    id: Some(NodeId::from_int(id)),
                    name: name.as_str().into(),
                    source,
                    lod,
                    embedding,
                    capabilities: capabilities_blob,
                    ..Default::default()
                })
            })
            .ok();
        Ok(result)
    }

    pub fn insert_wasm_tool(&self, tool: &WasmTool) -> Result<(), LibraryError> {
        let conn = self.conn.lock().unwrap();
        let caps_json = serde_json::to_string(&tool.capabilities).unwrap_or_default();
        conn.execute(
            "INSERT INTO wasm_tools (name, path, capabilities) VALUES (?1, ?2, ?3)",
            params![tool.name.as_str(), tool.path, caps_json],
        )?;
        Ok(())
    }

    pub fn find_wasm_tools_by_capability(
        &self,
        capability: &str,
    ) -> Result<Vec<WasmTool>, LibraryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT name, path, capabilities FROM wasm_tools")?;

        let results = stmt
            .query_map([], |row| {
                let name: String = row.get(0)?;
                let path: String = row.get(1)?;
                let caps_json: String = row.get(2)?;
                let capabilities: Vec<String> =
                    serde_json::from_str(&caps_json).unwrap_or_default();
                Ok((name, path, capabilities))
            })?
            .filter_map(|r| {
                r.ok().and_then(|(name, path, caps)| {
                    if caps.iter().any(|c| c == capability) {
                        Some(WasmTool {
                            name: name.as_str().into(),
                            path,
                            capabilities: caps.into_iter().map(|c| c.as_str().into()).collect(),
                        })
                    } else {
                        None
                    }
                })
            })
            .collect();

        Ok(results)
    }

    pub fn insert_nodes_batch(&self, nodes: &[ContentNode]) -> Result<(), LibraryError> {
        let conn = self.conn.lock().unwrap();
        let tx = conn.unchecked_transaction()?;
        for node in nodes {
            let lod_json = serde_json::to_string(&node.lod).unwrap_or_default();
            let embedding_blob = node.embedding.as_ref().map(|v| vec_to_bytes(v));
            let capabilities_blob = node.capabilities.as_deref();
            tx.execute(
                "INSERT INTO context_nodes (name, source, lod, embedding, capabilities) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![node.name.as_str(), node.source, lod_json, embedding_blob, capabilities_blob],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn node_count(&self) -> Result<i64, LibraryError> {
        let conn = self.conn.lock().unwrap();
        let count: i64 =
            conn.query_row("SELECT COUNT(*) FROM context_nodes", [], |row| row.get(0))?;
        Ok(count)
    }

    pub fn get_all_node_ids(&self) -> Result<Vec<NodeId>, LibraryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT id FROM context_nodes ORDER BY id")?;
        let ids = stmt
            .query_map([], |row| {
                let id: i64 = row.get(0)?;
                Ok(NodeId::from_int(id))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ids)
    }

    pub fn keyword_search(&self, query: &str) -> Result<Vec<KnnHit>, LibraryError> {
        let conn = self.conn.lock().unwrap();
        let pattern = format!("%{query}%");
        let mut stmt = conn.prepare(
            "SELECT id, name FROM context_nodes WHERE name LIKE ?1 OR source LIKE ?1 LIMIT 10",
        )?;
        let results = stmt
            .query_map(params![pattern], |row| {
                let id: i64 = row.get(0)?;
                let name: String = row.get(1)?;
                Ok(KnnHit {
                    node_id: NodeId::from_int(id),
                    distance: 0.0,
                    name: name.as_str().into(),
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(results)
    }

    pub fn embedded_node_count(&self) -> Result<i64, LibraryError> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM context_nodes WHERE embedding IS NOT NULL",
            [],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    pub fn knn_search(
        &self,
        query_vec: &[f32],
        k: usize,
        capability_filter: Option<&bitvec::vec::BitVec>,
    ) -> Result<Vec<KnnHit>, LibraryError> {
        if query_vec.is_empty() {
            return Ok(Vec::new());
        }
        // M8.1: the row scan is hard-capped at MAX_KNN_CANDIDATES so a large
        // table can never be fully scanned. When the embedded-node count
        // exceeds the cap, the approximate HNSW index is the primary query
        // path; only fall back to the capped brute-force scan when no HNSW
        // index is available. Capability-filtered searches bypass the HNSW
        // route because the index cannot apply the filter.
        if capability_filter.is_none() && self.embedded_node_count()? > MAX_KNN_CANDIDATES as i64 {
            if let Some(hits) = self.hnsw_search(query_vec, k) {
                return Ok(hits);
            }
        }
        let conn = self.conn.lock().unwrap();
        let capabilities_col = if capability_filter.is_some() {
            ", capabilities"
        } else {
            ""
        };
        let sql = format!(
            "SELECT id, name, embedding{capabilities_col} FROM context_nodes WHERE embedding IS NOT NULL LIMIT {MAX_KNN_CANDIDATES}"
        );
        let mut stmt = conn.prepare(&sql)?;

        let rows = stmt.query_map([], move |row| {
            let id: i64 = row.get(0)?;
            let name: String = row.get(1)?;
            let blob: Vec<u8> = row.get(2)?;
            let caps_blob: Option<Vec<u8>> = if capability_filter.is_some() {
                row.get(3).ok().flatten()
            } else {
                None
            };
            Ok((id, name, blob, caps_blob))
        })?;

        let mut row_meta: Vec<(NodeId, String)> = Vec::new();
        let mut candidates: Vec<(usize, Vec<f32>)> = Vec::new();

        for row_result in rows {
            let (id, name, blob, caps_blob) = row_result?;
            if let Some(filter) = capability_filter {
                let node_bv = caps_blob.as_deref().map(blob_to_bitvec).unwrap_or_default();
                let overlap = node_bv.iter().zip(filter.iter()).any(|(a, b)| *a && *b);
                if !overlap {
                    continue;
                }
            }
            if let Some(emb) = try_bytes_to_vec(&blob) {
                let idx = row_meta.len();
                row_meta.push((NodeId::from_int(id), name));
                candidates.push((idx, emb));
            }
        }

        let top_k = knn_brute_force(
            query_vec,
            candidates.iter().map(|(idx, emb)| (*idx, emb.as_slice())),
            k,
        );
        let results = top_k
            .into_iter()
            .map(|(idx, distance)| KnnHit {
                node_id: row_meta[idx].0,
                distance,
                name: row_meta[idx].1.as_str().into(),
            })
            .collect();
        Ok(results)
    }

    pub fn hybrid_search(
        &self,
        query: &str,
        query_vec: Option<&[f32]>,
        k: usize,
    ) -> Result<Vec<KnnHit>, LibraryError> {
        let keyword_results = self.keyword_search(query)?;

        let vector_results = if let Some(vec) = query_vec {
            // M8.3: merge (do not discard) partial HNSW hits with the
            // brute-force hits, dedup by node id keeping the best score,
            // then fill to `k`. The HNSW hits are never thrown away when the
            // index recall is below `k`.
            let mut merged: HashMap<i64, KnnHit> = HashMap::new();
            if let Some(hnsw_hits) = self.hnsw_search(vec, k) {
                for hit in hnsw_hits {
                    merged.insert(hit.node_id.as_int(), hit);
                }
            }
            for hit in self.knn_search(vec, k, None)? {
                let id = hit.node_id.as_int();
                match merged.get(&id) {
                    Some(existing) if existing.distance <= hit.distance => {}
                    _ => {
                        merged.insert(id, hit);
                    }
                }
            }
            let mut hits: Vec<KnnHit> = merged.into_values().collect();
            hits.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            hits.truncate(k);
            hits
        } else {
            Vec::new()
        };

        let mut rrf_scores: HashMap<i64, (f64, KnnHit)> = HashMap::new();
        let k_constant = 60.0_f64;

        for (rank, result) in keyword_results.into_iter().enumerate() {
            let id = result.node_id.as_int();
            rrf_scores.insert(id, (1.0 / (k_constant + rank as f64), result));
        }

        for (rank, result) in vector_results.into_iter().enumerate() {
            let id = result.node_id.as_int();
            let score = 1.0 / (k_constant + rank as f64);
            let entry = rrf_scores
                .entry(id)
                .or_insert_with(|| (0.0, result.clone()));
            entry.0 += score;
        }

        let mut merged: Vec<(f64, KnnHit)> = rrf_scores.into_values().collect();
        merged.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        Ok(merged.into_iter().take(k).map(|(_, hit)| hit).collect())
    }
}

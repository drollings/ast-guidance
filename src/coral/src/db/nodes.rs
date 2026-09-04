use std::collections::HashMap;

use fluent_db::vector::{knn_brute_force, try_bytes_to_vec, vec_to_bytes};
use fluent_types::{ContentNode, KnnHit, NodeId, WasmTool};
use rusqlite::params;

use super::{blob_to_bitvec, LibraryError, MAX_KNN_CANDIDATES};

impl super::Library {
    pub fn insert_node(&self, node: &ContentNode) -> Result<NodeId, LibraryError> {
        let lod_json = serde_json::to_string(&node.lod).unwrap_or_default();
        let embedding_blob = node.embedding.as_ref().map(|v| vec_to_bytes(v));
        let capabilities_blob = node.capabilities.as_deref();

        // INSERT + (last_insert_rowid) run under one connection lock so the
        // returned id is the row this call wrote. Content-addressed callers
        // supply `id` (the `hash_iri` value IS the primary key); `INSERT OR
        // IGNORE` makes re-ingestion of the same content idempotent.
        let node_id = self.store.with_conn(|conn| {
            if let Some(nid) = node.id {
                fluent_db::query::execute(
                    conn,
                    "INSERT OR IGNORE INTO context_nodes \
                     (id, name, source, lod, embedding, capabilities) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![
                        nid.as_int(),
                        node.name.as_str(),
                        node.source,
                        lod_json,
                        embedding_blob,
                        capabilities_blob
                    ],
                )?;
                Ok(nid)
            } else {
                // Legacy path (tests, non-content-addressed callers): autoincrement.
                fluent_db::query::execute(
                    conn,
                    "INSERT INTO context_nodes (name, source, lod, embedding, capabilities) VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![node.name.as_str(), node.source, lod_json, embedding_blob, capabilities_blob],
                )?;
                Ok(NodeId::from_int(fluent_db::query::last_insert_rowid(conn)))
            }
        })?;

        // The connection lock is released before the index lock so the
        // `hnsw → id_map → conn` ordering is never inverted (R9).
        if let Some(ref emb) = node.embedding {
            self.hnsw_insert(node_id.as_int(), emb);
        }

        Ok(node_id)
    }

    pub fn find_node_by_name(&self, name: &str) -> Result<Option<NodeId>, LibraryError> {
        Ok(self.store.query_row(
            "SELECT id FROM context_nodes WHERE name = ?1",
            params![name],
            |row| {
                let id: i64 = row.get(0)?;
                Ok(NodeId::from_int(id))
            },
        )?)
    }

    /// Resolve many names to their node ids in a single parameterized query
    /// (avoids the per-name N+1 round-trip through the shared connection).
    pub fn find_node_ids_by_names(
        &self,
        names: &[&str],
    ) -> Result<HashMap<String, NodeId>, LibraryError> {
        if names.is_empty() {
            return Ok(HashMap::new());
        }
        let placeholders = common_core::sqlite::in_clause(names.len());
        let sql = format!("SELECT id, name FROM context_nodes WHERE name IN ({placeholders})");
        let rows = self.store.with_conn(|conn| {
            fluent_db::query::query_rows_from_iter(conn, &sql, names.iter().copied(), |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            })
        })?;
        let mut map = HashMap::with_capacity(names.len());
        for (id, name) in rows {
            map.insert(name, NodeId::from_int(id));
        }
        Ok(map)
    }

    pub fn get_node(&self, node_id: NodeId) -> Result<Option<ContentNode>, LibraryError> {
        Ok(self.store.query_row(
            "SELECT id, name, source, lod, embedding, capabilities FROM context_nodes WHERE id = ?1",
            params![node_id.as_int()],
            |row| {
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
            },
        )?)
    }

    pub fn insert_wasm_tool(&self, tool: &WasmTool) -> Result<(), LibraryError> {
        let caps_json = serde_json::to_string(&tool.capabilities).unwrap_or_default();
        self.store.execute(
            "INSERT INTO wasm_tools (name, path, capabilities) VALUES (?1, ?2, ?3)",
            params![tool.name.as_str(), tool.path, caps_json],
        )?;
        Ok(())
    }

    pub fn find_wasm_tools_by_capability(
        &self,
        capability: &str,
    ) -> Result<Vec<WasmTool>, LibraryError> {
        let rows = self.store.query_rows(
            "SELECT name, path, capabilities FROM wasm_tools",
            &[],
            |row| {
                let name: String = row.get(0)?;
                let path: String = row.get(1)?;
                let caps_json: String = row.get(2)?;
                Ok((name, path, caps_json))
            },
        )?;

        let results = rows
            .into_iter()
            .filter_map(|(name, path, caps_json)| {
                let capabilities: Vec<String> =
                    serde_json::from_str(&caps_json).unwrap_or_default();
                if capabilities.iter().any(|c| c == capability) {
                    Some(WasmTool {
                        name: name.as_str().into(),
                        path,
                        capabilities: capabilities
                            .into_iter()
                            .map(|c| c.as_str().into())
                            .collect(),
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(results)
    }

    pub fn insert_nodes_batch(&self, nodes: &[ContentNode]) -> Result<(), LibraryError> {
        Ok(self.store.transaction(|tx| {
            for node in nodes {
                let lod_json = serde_json::to_string(&node.lod).unwrap_or_default();
                let embedding_blob = node.embedding.as_ref().map(|v| vec_to_bytes(v));
                let capabilities_blob = node.capabilities.as_deref();
                if let Some(nid) = node.id {
                    // Content-addressed: hash_iri IS the primary key;
                    // INSERT OR IGNORE keeps re-ingestion idempotent.
                    tx.execute(
                        "INSERT OR IGNORE INTO context_nodes \
                         (id, name, source, lod, embedding, capabilities) \
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                        params![
                            nid.as_int(),
                            node.name.as_str(),
                            node.source,
                            lod_json,
                            embedding_blob,
                            capabilities_blob
                        ],
                    )?;
                } else {
                    tx.execute(
                        "INSERT INTO context_nodes (name, source, lod, embedding, capabilities) VALUES (?1, ?2, ?3, ?4, ?5)",
                        params![node.name.as_str(), node.source, lod_json, embedding_blob, capabilities_blob],
                    )?;
                }
            }
            Ok(())
        })?)
    }

    pub fn node_count(&self) -> Result<i64, LibraryError> {
        let count = self
            .store
            .query_row("SELECT COUNT(*) FROM context_nodes", &[], |row| {
                row.get::<_, i64>(0)
            })?;
        Ok(count.unwrap_or(0))
    }

    /// Count nodes whose `source` starts with `prefix` — the scoped-count
    /// primitive for boot reconciliation (L6): a shared coral DB may hold
    /// unrelated content, so YaGO-content cross-checks must never use the total
    /// `node_count()`.
    pub fn count_nodes_by_source_prefix(&self, prefix: &str) -> Result<i64, LibraryError> {
        let pattern = format!("{prefix}%");
        let count = self.store.query_row(
            "SELECT COUNT(*) FROM context_nodes WHERE source LIKE ?1",
            params![pattern],
            |row| row.get::<_, i64>(0),
        )?;
        Ok(count.unwrap_or(0))
    }

    pub fn get_all_node_ids(&self) -> Result<Vec<NodeId>, LibraryError> {
        Ok(self
            .store
            .query_rows("SELECT id FROM context_nodes ORDER BY id", &[], |row| {
                let id: i64 = row.get(0)?;
                Ok(NodeId::from_int(id))
            })?)
    }

    pub fn keyword_search(&self, query: &str) -> Result<Vec<KnnHit>, LibraryError> {
        let pattern = format!("%{query}%");
        Ok(self.store.query_rows(
            "SELECT id, name FROM context_nodes WHERE name LIKE ?1 OR source LIKE ?1 LIMIT 10",
            params![pattern],
            |row| {
                let id: i64 = row.get(0)?;
                let name: String = row.get(1)?;
                Ok(KnnHit {
                    node_id: NodeId::from_int(id),
                    distance: 0.0,
                    name: name.as_str().into(),
                })
            },
        )?)
    }

    pub fn embedded_node_count(&self) -> Result<i64, LibraryError> {
        let count = self.store.query_row(
            "SELECT COUNT(*) FROM context_nodes WHERE embedding IS NOT NULL",
            &[],
            |row| row.get::<_, i64>(0),
        )?;
        Ok(count.unwrap_or(0))
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
        // The row scan is hard-capped at MAX_KNN_CANDIDATES so a large
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
        let capabilities_col = if capability_filter.is_some() {
            ", capabilities"
        } else {
            ""
        };
        let sql = format!(
            "SELECT id, name, embedding{capabilities_col} FROM context_nodes WHERE embedding IS NOT NULL LIMIT {MAX_KNN_CANDIDATES}"
        );
        let rows = self.store.query_rows(&sql, &[], move |row| {
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

        for (id, name, blob, caps_blob) in rows {
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
            // Merge (do not discard) partial HNSW hits with the
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
            let hits: Vec<KnnHit> = merged.into_values().collect();
            // Shared top-K tail (P2): ascending distance + truncate(k). The
            // HNSW∪brute-force merge + `rrf_merge` staging above stays
            // (domain logic, not eligible).
            fluent_db::vector::top_k_by_score(hits, k, |h| h.distance, false)
        } else {
            Vec::new()
        };

        // Generic ranked fusion over `(id, item)` pairs — replaces the
        // inline RRF; dedup + truncation semantics unchanged.
        let fused: Vec<(f64, KnnHit)> = fluent_db::vector::rrf_merge(
            keyword_results
                .into_iter()
                .map(|r| (r.node_id.as_int(), r))
                .collect(),
            vector_results
                .into_iter()
                .map(|r| (r.node_id.as_int(), r))
                .collect(),
            60.0,
        );

        Ok(fused.into_iter().take(k).map(|(_, hit)| hit).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::common::make_node;

    fn lib() -> crate::db::Library {
        crate::db::Library::open_in_memory().expect("in-memory db")
    }

    #[test]
    fn insert_and_get_round_trip() {
        let lib = lib();
        let node = ContentNode {
            embedding: Some(vec![0.1, 0.2, 0.3]),
            ..make_node("n1", "s1")
        };
        let id = lib.insert_node(&node).expect("insert");
        assert!(id.as_int() > 0);
        let back = lib.get_node(id).expect("get").expect("exists");
        assert_eq!(back.id, Some(id));
        assert_eq!(back.name.as_str(), "n1");
        assert_eq!(back.source, "s1");
        assert_eq!(back.embedding, Some(vec![0.1, 0.2, 0.3]));
        assert_eq!(lib.node_count().expect("count"), 1);
        // Missing node -> None.
        assert!(lib.get_node(NodeId::from_int(999)).expect("missing").is_none());
    }

    #[test]
    fn insert_round_trips_lod_and_capabilities() {
        let lib = lib();
        let node = ContentNode {
            lod: vec!["a".into(), "b".into()],
            capabilities: Some(vec![0b0000_0011]),
            ..make_node("n2", "s2")
        };
        let id = lib.insert_node(&node).expect("insert");
        let back = lib.get_node(id).expect("get").expect("exists");
        assert_eq!(back.lod, vec!["a", "b"]);
        assert_eq!(back.capabilities, Some(vec![0b0000_0011]));
    }

    #[test]
    fn find_by_name_and_batch_resolution() {
        let lib = lib();
        let a = lib.insert_node(&make_node("alpha", "src-a")).expect("insert a");
        let b = lib.insert_node(&make_node("beta", "src-b")).expect("insert b");
        assert_eq!(lib.find_node_by_name("alpha").expect("find"), Some(a));
        assert!(lib.find_node_by_name("nope").expect("find").is_none());

        let map = lib
            .find_node_ids_by_names(&["alpha", "beta", "missing"])
            .expect("batch");
        assert_eq!(map.get("alpha").copied(), Some(a));
        assert_eq!(map.get("beta").copied(), Some(b));
        assert!(!map.contains_key("missing"));
        // Empty lookup is a no-op.
        assert!(lib.find_node_ids_by_names(&[]).expect("empty").is_empty());
    }

    #[test]
    fn batch_insert_commits_all_rows() {
        let lib = lib();
        let nodes: Vec<ContentNode> = (0..5)
            .map(|i| make_node(&format!("n{i}"), "src"))
            .collect();
        lib.insert_nodes_batch(&nodes).expect("batch insert");
        assert_eq!(lib.node_count().expect("count"), 5);
        assert_eq!(lib.get_all_node_ids().expect("ids").len(), 5);
    }

    #[test]
    fn keyword_search_matches_name_and_source() {
        let lib = lib();
        lib.insert_node(&make_node("bug-triage", "jira ticket 42")).expect("insert");
        lib.insert_node(&make_node("review", "unrelated")).expect("insert");
        let hits = lib.keyword_search("jira").expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].name.as_str(), "bug-triage");
        let hits = lib.keyword_search("bug").expect("search name");
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn wasm_tools_insert_and_find_by_capability() {
        let lib = lib();
        lib.insert_wasm_tool(&WasmTool {
            name: "search".into(),
            path: "/opt/tools/search.wasm".into(),
            capabilities: vec!["search".into(), "web".into()],
        })
        .expect("insert tool");
        lib.insert_wasm_tool(&WasmTool {
            name: "scorer".into(),
            path: "/opt/tools/scorer.wasm".into(),
            capabilities: vec!["score".into()],
        })
        .expect("insert tool");
        let hits = lib.find_wasm_tools_by_capability("search").expect("find");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].name.as_str(), "search");
        assert!(lib.find_wasm_tools_by_capability("nope").expect("find").is_empty());
    }

    #[test]
    fn embedded_node_count_only_counts_embedded() {
        let lib = lib();
        lib.insert_node(&make_node("plain", "s")).expect("insert plain");
        lib.insert_node(&ContentNode {
            embedding: Some(vec![0.5]),
            ..make_node("embedded", "s")
        })
        .expect("insert embedded");
        assert_eq!(lib.embedded_node_count().expect("count"), 1);
    }

    #[test]
    fn insert_with_explicit_id_uses_it_as_primary_key() {
        let lib = lib();
        let nid = NodeId::from_int(0x1234_5678_9abc_def0_i64 as i64);
        let node = ContentNode {
            id: Some(nid),
            ..make_node("content-addressed", "s")
        };
        let back = lib.insert_node(&node).expect("insert");
        assert_eq!(back, nid);
        let fetched = lib.get_node(nid).expect("get").expect("exists");
        assert_eq!(fetched.id, Some(nid));
        assert_eq!(fetched.name.as_str(), "content-addressed");
        assert_eq!(lib.node_count().expect("count"), 1);
    }

    #[test]
    fn reinsert_same_content_id_is_idempotent() {
        let lib = lib();
        let nid = NodeId::from_int(hash_iri_fixture("http://example.org/dupe"));
        let node = ContentNode {
            id: Some(nid),
            ..make_node("dupe", "s")
        };
        lib.insert_node(&node).expect("first insert");
        let again = lib.insert_node(&node).expect("second insert");
        assert_eq!(again, nid, "explicit id returned whether or not the row was inserted");
        assert_eq!(lib.node_count().expect("count"), 1, "re-ingestion adds no row");
    }

    #[test]
    fn batch_preserves_explicit_ids_and_is_idempotent() {
        let lib = lib();
        let nodes: Vec<ContentNode> = (0..3)
            .map(|i| ContentNode {
                id: Some(NodeId::from_int(hash_iri_fixture(&format!("http://example.org/b{i}")))),
                ..make_node(&format!("b{i}"), "src")
            })
            .collect();
        lib.insert_nodes_batch(&nodes).expect("first batch");
        lib.insert_nodes_batch(&nodes).expect("second batch");
        assert_eq!(lib.node_count().expect("count"), 3);
        for n in &nodes {
            let id = n.id.expect("id set");
            assert_eq!(lib.get_node(id).expect("get").expect("exists").id, Some(id));
        }
    }

    #[test]
    fn explicit_and_autoincrement_ids_do_not_collide() {
        let lib = lib();
        let explicit = lib
            .insert_node(&ContentNode {
                id: Some(NodeId::from_int(hash_iri_fixture("http://example.org/x"))),
                ..make_node("explicit", "s")
            })
            .expect("explicit");
        let auto = lib.insert_node(&make_node("auto", "s")).expect("auto");
        assert_eq!(explicit.as_int(), hash_iri_fixture("http://example.org/x"));
        assert_ne!(explicit, auto);
        assert_eq!(lib.node_count().expect("count"), 2);
    }

    fn hash_iri_fixture(s: &str) -> i64 {
        guidance_rdf::normalize::hash_iri(s)
    }

    #[test]
    fn knn_search_returns_nearest_embeddings() {
        let lib = lib();
        lib.insert_node(&ContentNode {
            embedding: Some(vec![1.0, 0.0]),
            ..make_node("right", "s")
        })
        .expect("insert");
        lib.insert_node(&ContentNode {
            embedding: Some(vec![0.0, 1.0]),
            ..make_node("up", "s")
        })
        .expect("insert");
        let hits = lib.knn_search(&[0.9, 0.0], 1, None).expect("knn");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].name.as_str(), "right");
        // Empty query vec -> no results.
        assert!(lib.knn_search(&[], 5, None).expect("knn empty").is_empty());
    }
}

use std::path::Path;
use std::sync::{Mutex, RwLock};

use crate::error::DbError;
use common_core::constants::HnswParams;
use rusqlite::params;
use thiserror::Error;

use crate::math;

use anndists::dist::DistCosine;
use hnsw_rs::hnsw::Hnsw;

#[derive(Error, Debug)]
pub enum VectorDbError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] common_core::error::SqliteError),
    #[error("database error: {0}")]
    Db(#[from] DbError),
    #[error("embedding dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
}

impl From<rusqlite::Error> for VectorDbError {
    fn from(e: rusqlite::Error) -> Self {
        VectorDbError::Sqlite(common_core::error::SqliteError(e))
    }
}

/// Build a cosine-distance HNSW index from `HnswParams` with a pluggable
/// `initial_capacity`. Centralizing the positional-argument unpacking here
/// means the (max_nb_connection, initial_capacity, max_layer,
/// ef_construction, DistCosine) ordering is decided in exactly one place per
/// crate — see `ROADMAP_20260625_CONSOLIDATE.md` Priority 6. The rebuild
/// path passes `count.max(p.initial_capacity)` so the index grows with the
/// loaded row count while the insert path uses the default capacity.
fn make_hnsw(p: &HnswParams, initial_capacity: usize) -> Hnsw<'static, f32, DistCosine> {
    Hnsw::<f32, DistCosine>::new(
        p.max_nb_connection,
        initial_capacity,
        p.max_layer,
        p.ef_construction,
        DistCosine,
    )
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: i64,
    pub name: String,
    pub source: String,
    pub signature: Option<String>,
    pub similarity: f32,
}

pub struct GuidanceDb {
    conn: Mutex<rusqlite::Connection>,
    hnsw: RwLock<Option<Hnsw<'static, f32, DistCosine>>>,
    hnsw_id_map: Mutex<Vec<i64>>,
}

impl GuidanceDb {
    pub fn open(path: &Path) -> Result<Self, VectorDbError> {
        let conn = common_core::sqlite::open_wal(path)?;
        let db = Self {
            conn: Mutex::new(conn),
            hnsw: RwLock::new(None),
            hnsw_id_map: Mutex::new(Vec::new()),
        };
        db.init_schema()?;
        Ok(db)
    }

    pub fn open_in_memory() -> Result<Self, VectorDbError> {
        let conn = common_core::sqlite::open_in_memory()?;
        let db = Self {
            conn: Mutex::new(conn),
            hnsw: RwLock::new(None),
            hnsw_id_map: Mutex::new(Vec::new()),
        };
        db.init_schema()?;
        Ok(db)
    }

    fn init_schema(&self) -> Result<(), VectorDbError> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS guidance_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                source TEXT NOT NULL,
                signature TEXT,
                comment TEXT,
                module TEXT NOT NULL,
                language TEXT NOT NULL DEFAULT 'zig',
                embedding BLOB,
                created_at TEXT DEFAULT (datetime('now'))
            );",
        )?;
        common_core::sqlite::init_embedding_cache(&conn)?;
        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_nodes_name ON guidance_nodes(name);
             CREATE INDEX IF NOT EXISTS idx_nodes_source ON guidance_nodes(source);
             CREATE INDEX IF NOT EXISTS idx_nodes_name_source ON guidance_nodes(name, source);
             CREATE INDEX IF NOT EXISTS idx_cache_query_hash ON embedding_cache(query_hash);",
        )?;
        Ok(())
    }

    pub fn insert_node(
        &self,
        name: &str,
        source: &str,
        signature: Option<&str>,
        comment: Option<&str>,
        module: &str,
        language: &str,
        embedding: Option<&[f32]>,
    ) -> Result<i64, VectorDbError> {
        let conn = self.conn.lock().unwrap();
        let embedding_blob = embedding.map(math::vec_to_bytes);

        conn.execute(
            "INSERT INTO guidance_nodes (name, source, signature, comment, module, language, embedding)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                name,
                source,
                signature,
                comment,
                module,
                language,
                embedding_blob,
            ],
        )?;

        let node_id = conn.last_insert_rowid();

        // Insert into HNSW index if embedding is provided
        if let Some(emb) = embedding {
            self.hnsw_insert(node_id, emb);
        }

        Ok(node_id)
    }

    /// Insert a vector into the HNSW index.
    fn hnsw_insert(&self, node_id: i64, embedding: &[f32]) {
        let mut guard = self.hnsw.write().unwrap();
        let hnsw = guard.get_or_insert_with(|| {
            let p = HnswParams::default();
            make_hnsw(&p, p.initial_capacity)
        });

        let external_id = {
            let mut id_map = self.hnsw_id_map.lock().unwrap();
            let idx = id_map.len();
            id_map.push(node_id);
            idx
        };

        hnsw.insert((embedding, external_id));
    }

    /// Rebuild the HNSW index from all embedded nodes in the database.
    pub fn rebuild_hnsw(&self) -> Result<usize, VectorDbError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt =
            conn.prepare("SELECT id, embedding FROM guidance_nodes WHERE embedding IS NOT NULL")?;

        let rows: Vec<(i64, Vec<u8>)> = stmt
            .query_map([], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
            })?
            .filter_map(Result::ok)
            .collect();

        let count = rows.len();

        // Build new HNSW index
        let p = HnswParams::default();
        let hnsw = make_hnsw(&p, count.max(p.initial_capacity));

        let mut id_map = Vec::with_capacity(count);
        for (i, (node_id, blob)) in rows.into_iter().enumerate() {
            let embedding = math::bytes_to_vec(&blob);
            if !embedding.is_empty() {
                hnsw.insert((&embedding, i));
                id_map.push(node_id);
            }
        }

        *self.hnsw.write().unwrap() = Some(hnsw);
        *self.hnsw_id_map.lock().unwrap() = id_map;

        Ok(count)
    }

    /// Vector similarity search. Uses HNSW index when available, falls back
    /// to brute-force O(n × d) scan otherwise.
    ///
    /// ## Performance
    /// - With HNSW: O(log n) approximate nearest neighbor search
    /// - Without HNSW, n < 10_000:  sub-millisecond on modern CPU
    /// - Without HNSW, n < 100_000: ~10 ms
    pub fn vector_search(
        &self,
        query_vec: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>, VectorDbError> {
        // Try HNSW first, but fall back to brute-force if it returns fewer
        // results than requested (can happen with very small indices).
        if let Some(results) = self.hnsw_search(query_vec, k) {
            if results.len() >= k {
                return Ok(results);
            }
        }

        // Fall back to brute-force
        self.bruteforce_vector_search(query_vec, k)
    }

    /// HNSW approximate nearest neighbor search.
    fn hnsw_search(&self, query_vec: &[f32], k: usize) -> Option<Vec<SearchResult>> {
        let guard = self.hnsw.read().ok()?;
        let hnsw = guard.as_ref()?;
        let id_map = self.hnsw_id_map.lock().ok()?;

        let neighbours = hnsw.search(query_vec, k, k);

        let conn = self.conn.lock().ok()?;

        let mut results = Vec::with_capacity(neighbours.len());
        for n in &neighbours {
            let idx = n.d_id;
            if idx >= id_map.len() {
                continue;
            }
            let node_id = id_map[idx];

            // Convert cosine distance to similarity: dist = 1 - cos_sim
            let similarity = 1.0 - n.distance;

            if let Ok(row) = conn.query_row(
                "SELECT name, source, signature FROM guidance_nodes WHERE id = ?1",
                params![node_id],
                |row| {
                    Ok(SearchResult {
                        id: node_id,
                        name: row.get(0)?,
                        source: row.get(1)?,
                        signature: row.get(2)?,
                        similarity,
                    })
                },
            ) {
                results.push(row);
            }
        }

        Some(results)
    }

    /// Brute-force O(n × d) vector similarity search.
    fn bruteforce_vector_search(
        &self,
        query_vec: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>, VectorDbError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, name, source, signature, embedding FROM guidance_nodes WHERE embedding IS NOT NULL",
        )?;

        let mut results: Vec<SearchResult> = Vec::new();

        let rows = stmt.query_map([], |row| {
            let id: i64 = row.get(0)?;
            let name: String = row.get(1)?;
            let source: String = row.get(2)?;
            let signature: Option<String> = row.get(3)?;
            let embedding_blob: Option<Vec<u8>> = row.get(4)?;
            Ok((id, name, source, signature, embedding_blob))
        })?;

        for row_result in rows {
            let (id, name, source, signature, embedding_blob) = row_result?;
            if let Some(blob) = embedding_blob {
                let embedding = math::bytes_to_vec(&blob);
                if embedding.len() != query_vec.len() {
                    continue;
                }
                let similarity = math::cosine_similarity(query_vec, &embedding);
                results.push(SearchResult {
                    id,
                    name,
                    source,
                    signature,
                    similarity,
                });
            }
        }

        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);

        Ok(results)
    }

    pub fn keyword_search(&self, query: &str) -> Result<Vec<SearchResult>, VectorDbError> {
        let conn = self.conn.lock().unwrap();

        // 1. Try exact full-query substring match first (fast, precise).
        let pattern = format!("%{query}%");
        let mut stmt = conn.prepare(
            "SELECT id, name, source, signature FROM guidance_nodes
             WHERE name LIKE ?1 OR signature LIKE ?1 OR comment LIKE ?1
             LIMIT 50",
        )?;
        let exact: Vec<SearchResult> = stmt
            .query_map(params![pattern], |row| {
                Ok(SearchResult {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    source: row.get(2)?,
                    signature: row.get(3)?,
                    similarity: 1.0,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        if !exact.is_empty() {
            return Ok(exact);
        }

        // 2. Token-based fallback for natural-language queries. Split
        //    into tokens, require at least `min_matches` tokens to match
        //    name/signature/comment. Short queries (1-2 tokens) need 1
        //    match; longer queries need 2+ to filter noise.
        let tokens: Vec<&str> = query
            .split_whitespace()
            .filter(|t| t.len() >= 3)
            .collect();
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        // Require a proportional number of token matches to filter noise.
        // Short queries (1-2 tokens) need 1 match; longer queries need
        // ~30% of significant tokens to match, preventing false positives
        // from incidental keyword overlap (e.g. "coral" + "module" matching
        // unrelated code when the query is about quantum entanglement).
        let min_matches: i32 = if tokens.len() <= 2 {
            1
        } else {
            ((tokens.len() as f32 * 0.3).ceil() as i32).max(2)
        };

        // Build positional LIKE conditions for each token.
        let mut wheres = Vec::new();
        let mut param_idx = 1;
        let mut token_patterns: Vec<String> = Vec::new();
        for &tok in &tokens {
            let pattern = format!("%{tok}%");
            token_patterns.push(pattern);
            let t = param_idx;
            wheres.push(format!(
                "(name LIKE ?{t} OR signature LIKE ?{t} OR comment LIKE ?{t})"
            ));
            param_idx += 1;
        }
        let hits_expr = wheres
            .iter()
            .map(|w| format!("CASE WHEN {w} THEN 1 ELSE 0 END"))
            .collect::<Vec<_>>()
            .join(" + ");
        let min_hits_param = param_idx;

        let sql = format!(
            "SELECT id, name, source, signature, ({hits_expr}) AS hits
             FROM guidance_nodes
             WHERE hits >= ?{min_hits_param}
             ORDER BY hits DESC
             LIMIT 50"
        );

        let mut stmt = conn.prepare(&sql)?;
        let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = token_patterns
            .iter()
            .map(|p| Box::new(p.clone()) as Box<dyn rusqlite::types::ToSql>)
            .collect();
        params.push(Box::new(min_matches));

        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            params.iter().map(Box::as_ref).collect();
        let results = stmt
            .query_map(param_refs.as_slice(), |row| {
                Ok(SearchResult {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    source: row.get(2)?,
                    signature: row.get(3)?,
                    similarity: row.get::<_, i32>(4)? as f32,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(results)
    }

    pub fn hybrid_search(
        &self,
        query: &str,
        query_vec: Option<&[f32]>,
        k: usize,
    ) -> Result<Vec<SearchResult>, VectorDbError> {
        let keyword_results = self.keyword_search(query)?;

        let vector_results = if let Some(vec) = query_vec {
            self.vector_search(vec, k)?
        } else {
            Vec::new()
        };

        let mut fused = rrf_merge(keyword_results, vector_results, 60.0);
        fused.truncate(k);

        Ok(fused)
    }

    pub fn get_node_count(&self) -> Result<i64, VectorDbError> {
        let conn = self.conn.lock().unwrap();
        let count: i64 =
            conn.query_row("SELECT COUNT(*) FROM guidance_nodes", [], |row| row.get(0))?;
        Ok(count)
    }

    pub fn get_embedding_count(&self) -> Result<i64, VectorDbError> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM guidance_nodes WHERE embedding IS NOT NULL",
            [],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    /// Sync all JSON files from a directory into the `guidance_nodes` table.
    /// Walks JSON files, parses GuidanceDoc, upserts into database.
    /// Rebuilds HNSW index after sync.
    pub fn sync_from_dir(&self, json_dir: &std::path::Path) -> Result<usize, VectorDbError> {
        if !json_dir.is_dir() {
            return Ok(0);
        }

        let synced = {
            let mut synced = 0;
            let conn = self.conn.lock().unwrap();

            // Clear existing nodes before re-sync to avoid stale duplicates.
            conn.execute("DELETE FROM guidance_nodes", [])?;

            let mut json_files = Vec::new();
            common_core::walk::walk_files(json_dir, &["json"], |path| {
                json_files.push(path.to_path_buf());
            });

            for path in &json_files {
                let content = common_core::io::read_to_string_err(path).map_err(|e| {
                    VectorDbError::Sqlite(common_core::error::SqliteError(
                        rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                    ))
                })?;
                if content.trim().is_empty() {
                    continue;
                }

                let doc: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
                    VectorDbError::Sqlite(common_core::error::SqliteError(
                        rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
                    ))
                })?;

                let source = doc["meta"]["source"].as_str().unwrap_or("");
                let module = doc["meta"]["module"].as_str().unwrap_or("");
                let language = doc["meta"]["language"].as_str().unwrap_or("zig");
                let comment = doc["comment"].as_str();

                // Upsert node — skip duplicates where (name, signature)
                // already exists.  This handles multiple JSON files for
                // the same source (e.g. `guidance/src/...` vs `src/guidance/src/...`).
                if let Some(members) = doc["members"].as_array() {
                    for member in members {
                        let name = member["name"].as_str().unwrap_or("");
                        let signature = member["signature"].as_str();
                        let member_comment = member["comment"].as_str();
                        let _is_anchor = member["is_anchor"].as_bool().unwrap_or(false);

                        // Check for existing row with same (name, signature).
                        let exists: bool = conn
                            .query_row(
                                "SELECT 1 FROM guidance_nodes WHERE name = ?1 AND signature IS ?2 LIMIT 1",
                                rusqlite::params![name, signature],
                                |_| Ok(true),
                            )
                            .unwrap_or(false);
                        if exists {
                            continue;
                        }

                        let _ = conn.execute(
                            "INSERT INTO guidance_nodes (name, source, signature, comment, module, language)
                             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                            rusqlite::params![
                                name,
                                source,
                                signature,
                                member_comment.or(comment),
                                module,
                                language
                            ],
                        );
                        synced += 1;
                    }
                }
            }

            synced
        };

        // Rebuild HNSW index after releasing the conn lock
        if synced > 0 {
            let _ = self.rebuild_hnsw();
        }

        Ok(synced)
    }

    /// Check if the HNSW index is built.
    pub fn has_hnsw(&self) -> bool {
        self.hnsw.read().is_ok_and(|g| g.is_some())
    }

    /// Get the number of points in the HNSW index.
    pub fn hnsw_len(&self) -> usize {
        self.hnsw
            .read()
            .ok()
            .and_then(|g| g.as_ref().map(Hnsw::get_nb_point))
            .unwrap_or(0)
    }
}

/// Reciprocal Rank Fusion: merges two ranked result lists using RRF scoring.
/// RRF score = sum(1 / (k + rank(engine))) for each result appearing in either list.
/// Results not present in a list get rank = infinity (contribute 0).
/// `k` is the RRF constant (typically 60).
pub fn rrf_merge(
    keyword_results: Vec<SearchResult>,
    vector_results: Vec<SearchResult>,
    k_constant: f64,
) -> Vec<SearchResult> {
    use std::collections::HashMap;

    let mut rrf_scores: HashMap<i64, (f64, SearchResult)> = HashMap::new();

    for (rank, result) in keyword_results.into_iter().enumerate() {
        rrf_scores.insert(result.id, (1.0 / (k_constant + rank as f64), result));
    }

    for (rank, result) in vector_results.into_iter().enumerate() {
        let score = 1.0 / (k_constant + rank as f64);
        let entry = rrf_scores.entry(result.id).or_insert_with(|| (0.0, result));
        entry.0 += score;
    }

    let mut merged: Vec<SearchResult> = rrf_scores
        .into_values()
        .map(|(score, mut r)| {
            r.similarity = score as f32;
            r
        })
        .collect();

    merged.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_db() -> GuidanceDb {
        GuidanceDb::open_in_memory().expect("in-memory db")
    }

    #[test]
    fn test_insert_and_count() {
        let db = make_db();
        let id = db
            .insert_node(
                "hello",
                "src/test.zig",
                Some("fn hello() void"),
                Some("Says hello"),
                "test",
                "zig",
                None,
            )
            .expect("insert");
        assert!(id > 0);
        assert_eq!(db.get_node_count().expect("count"), 1);
    }

    #[test]
    fn test_keyword_search() {
        let db = make_db();
        db.insert_node(
            "greet",
            "src/test.zig",
            Some("fn greet() void"),
            Some("Greets the user"),
            "test",
            "zig",
            None,
        )
        .expect("insert");
        db.insert_node(
            "add",
            "src/math.zig",
            Some("fn add() i32"),
            Some("Adds numbers"),
            "math",
            "zig",
            None,
        )
        .expect("insert");

        let results = db.keyword_search("greet").expect("search");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "greet");
    }

    #[test]
    fn test_vector_search() {
        let db = make_db();
        let emb1: Vec<f32> = (0..4).map(|i| i as f32).collect();
        let emb2: Vec<f32> = (0..4).map(|i| (i + 10) as f32).collect();

        db.insert_node("a", "src/a.zig", None, None, "test", "zig", Some(&emb1))
            .expect("insert");
        db.insert_node("b", "src/b.zig", None, None, "test", "zig", Some(&emb2))
            .expect("insert");

        let query = vec![0.5, 1.5, 2.5, 3.5];
        let results = db.vector_search(&query, 2).expect("search");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].name, "a");
    }

    #[test]
    fn test_empty_search() {
        let db = make_db();
        let results = db.keyword_search("nonexistent").expect("search");
        assert!(results.is_empty());
    }

    #[test]
    fn test_hybrid_search() {
        let db = make_db();
        let emb = vec![0.1, 0.2, 0.3, 0.4];

        db.insert_node(
            "hello_fn",
            "src/test.zig",
            Some("fn hello() void"),
            Some("Says hello"),
            "test",
            "zig",
            Some(&emb),
        )
        .expect("insert");

        let results = db
            .hybrid_search("hello", Some(&emb), 5)
            .expect("hybrid search");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_rrf_merge_single_result() {
        let kw = vec![SearchResult {
            id: 1,
            name: "foo".into(),
            source: "src/foo.zig".into(),
            signature: None,
            similarity: 0.0,
        }];
        let vec_results = vec![];
        let merged = rrf_merge(kw, vec_results, 60.0);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].name, "foo");
    }

    #[test]
    fn test_rrf_merge_boosts_shared_results() {
        let kw = vec![
            SearchResult {
                id: 1,
                name: "shared".into(),
                source: "a.zig".into(),
                signature: None,
                similarity: 0.0,
            },
            SearchResult {
                id: 2,
                name: "kw_only".into(),
                source: "a.zig".into(),
                signature: None,
                similarity: 0.0,
            },
        ];
        let vec_results = vec![
            SearchResult {
                id: 1,
                name: "shared".into(),
                source: "a.zig".into(),
                signature: None,
                similarity: 0.0,
            },
            SearchResult {
                id: 3,
                name: "vec_only".into(),
                source: "a.zig".into(),
                signature: None,
                similarity: 0.0,
            },
        ];
        let merged = rrf_merge(kw, vec_results, 60.0);
        // shared (id=1) should be ranked first since it appears in both lists
        assert!(merged.len() >= 2);
        assert_eq!(merged[0].name, "shared");
        assert!(merged[0].similarity > merged[1].similarity);
    }

    #[test]
    fn test_rrf_merge_deduplicates() {
        let kw = vec![SearchResult {
            id: 1,
            name: "dup".into(),
            source: "x.zig".into(),
            signature: None,
            similarity: 0.0,
        }];
        let vec_results = vec![SearchResult {
            id: 1,
            name: "dup".into(),
            source: "x.zig".into(),
            signature: None,
            similarity: 0.0,
        }];
        let merged = rrf_merge(kw.clone(), vec_results, 60.0);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].name, "dup");
    }
}

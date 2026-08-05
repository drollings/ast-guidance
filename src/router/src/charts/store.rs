//! ChartStore — loads and holds a directory of chart files at boot, in the
//! `yamake_loader` tradition.
//!
//! The store is the single owner of the workflow_library index path; the
//! `index` handle is reserved for M7 retrieval. Minimal store — selection/
//! retrieval comes in M7.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::RwLock;

use common_core::sync::{lock_read, lock_write};
use fluent_db::error::DbError;
use fluent_db::hnsw::HnswIndex;
use fluent_db::store::SqliteStore;
use fluent_llm::EmbeddingProvider;
use search_vector::math::{knn_brute_force, try_bytes_to_vec, vec_to_bytes};

use super::{ChartDef, ChartError};

/// Cosine-similarity threshold for the M10 idempotent-upsert near-neighbor
/// check: a chart whose embedding is at or above this similarity to an
/// existing chart *subsumes* it (VISION: update/subsumes, never duplicate).
/// Router-only constant — promote to `common-core::constants` when a second
/// consumer appears (Consolidation Contract).
pub(crate) const CHART_SUBSUME_THRESHOLD: f32 = 0.9;

/// Per-chart runtime health — the M10 staleness/demotion policy.
///
/// Kept *out* of the persisted `ChartDef` content model: a chart's health is
/// a runtime property of the store, not a declarative fact about the DAG.
#[derive(Debug, Clone, Default)]
pub struct ChartHealth {
    /// Auto-extracted (M10) and not yet rubric-validated: excluded from
    /// selection until a rubric-validated run clears the flag.
    pub draft: bool,
    /// Consecutive rubric-gate failures (M9 gate). Reset on any pass.
    pub stale_failures: usize,
    /// Demoted after `CHART_STALE_FAILS` consecutive rubric failures:
    /// excluded from selection, flagged in the audit log. Sticky until the
    /// chart is re-extracted (upsert replaces it).
    pub demoted: bool,
}

/// Outcome of an idempotent chart upsert (M10).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub enum UpsertOutcome {
    /// No near neighbor — the chart was stored under its own name (as a
    /// draft; selectable only after a rubric-validated run).
    Inserted,
    /// A near neighbor existed; the new chart subsumed it (kept its name).
    Subsumed { by: String },
}

/// In-memory chart-retrieval index: a cosine HNSW graph over one vector per
/// chart, backed by a flat list for the `knn_brute_force` fallback.
///
/// The graph is rebuilt from the router-side SQLite file at each boot (or on
/// demand via [`ChartStore::build_index`]); the SQLite file persists the
/// per-chart embeddings so repeated boots skip re-embedding.
struct ChartIndex {
    /// Embedding provider used for both build and query. Same provider ⇒ same
    /// vector space ⇒ dims always agree.
    embedder: Arc<dyn EmbeddingProvider>,
    /// Chart name by HNSW external id (`index == ids.len()` positions).
    ids: Vec<String>,
    /// Flat `(chart, embedding)` list for the brute-force fallback.
    flat: Vec<(String, Vec<f32>)>,
    /// Cosine HNSW graph (owned by `fluent_db::hnsw::HnswIndex`). Written
    /// exactly once (in `build_index`) and read-only thereafter;
    /// `invalidate_index` drops the whole `ChartIndex` rather than mutating
    /// the graph, so a `RwLock` here would be pure ceremony.
    hnsw: HnswIndex,
}

/// In-memory registry of validated charts, backed by JSON files on disk.
pub struct ChartStore {
    /// Chart name → validated chart. Read-guarded: the store is shared
    /// (`Arc<ChartStore>`) across the selector and the M10 extraction path,
    /// so runtime upserts go through the lock rather than `&mut self`.
    charts: RwLock<HashMap<String, Arc<ChartDef>>>,
    /// Chart name → runtime health (M10 draft/staleness/demotion state).
    health: RwLock<HashMap<String, ChartHealth>>,
    /// workflow_library HNSW index path handle (M7 retrieval).
    index: Option<crate::hnsw::HnswIndexHandle>,
    /// Built index (HNSW graph + embedder), if `build_index` succeeded.
    built: RwLock<Option<Arc<ChartIndex>>>,
}

impl ChartStore {
    pub fn new(index: Option<crate::hnsw::HnswIndexHandle>) -> Self {
        Self {
            charts: RwLock::new(HashMap::new()),
            health: RwLock::new(HashMap::new()),
            index,
            built: RwLock::new(None),
        }
    }

    /// Load all `*.json` files in `dir`.
    ///
    /// - A missing directory yields an empty store (with a `warn!`) — a
    ///   directory is allowed to be absent at boot.
    /// - A present-but-invalid file is a **hard error** — a corrupted
    ///   library must not half-load (fail fast, decision D3).
    pub fn load_dir(&self, dir: &Path) -> Result<(), ChartError> {
        if !dir.is_dir() {
            tracing::warn!(
                target: "router.charts.store",
                path = %dir.display(),
                "charts directory missing — starting with an empty chart store"
            );
            return Ok(());
        }

        let mut entries: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(std::result::Result::ok)
            .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
            .collect();
        entries.sort_by_key(std::fs::DirEntry::file_name);

        for entry in entries {
            let path = entry.path();
            let chart = Self::load_chart_file(&path)?;
            self.upsert(chart)?;
        }

        Ok(())
    }

    /// Parse and validate a single chart file.
    fn load_chart_file(path: &Path) -> Result<ChartDef, ChartError> {
        let content = std::fs::read_to_string(path)?;
        let chart: ChartDef = serde_json::from_str(&content).map_err(|e| ChartError::Parse {
            source: e,
            path: path.display().to_string(),
        })?;
        chart.validate().map_err(|e| ChartError::Invalid {
            reason: format!("chart '{}' failed validation: {e}", chart.name),
        })?;
        Ok(chart)
    }

    /// Look up a chart by name. Returns an `Arc` (the store is shared), so a
    /// callers holding the returned handle stays valid across later upserts.
    pub fn get(&self, name: &str) -> Option<Arc<ChartDef>> {
        lock_read(&self.charts).get(name).cloned()
    }

    /// All chart names, in insertion order (owned — the store is lock-backed).
    pub fn list(&self) -> Vec<String> {
        lock_read(&self.charts).keys().cloned().collect()
    }

    /// Number of loaded charts.
    pub fn len(&self) -> usize {
        lock_read(&self.charts).len()
    }

    /// `true` if the store holds no charts.
    pub fn is_empty(&self) -> bool {
        lock_read(&self.charts).is_empty()
    }

    /// Insert or replace a chart. Used by boot loading and by M10
    /// auto-extraction upsert. Rejects invalid charts (fail fast).
    ///
    /// A plain upsert resets the chart's health: a fresh chart is
    /// selectable by default (`draft = false`, cleared demotion). The M10
    /// extraction path uses [`Self::upsert_idempotent`], which writes a
    /// draft and deduplicates against near neighbors.
    pub fn upsert(&self, chart: ChartDef) -> Result<(), ChartError> {
        chart.validate().map_err(|e| ChartError::Invalid {
            reason: format!("chart '{}' failed validation: {e}", chart.name),
        })?;
        self.reset_health(&chart.name);
        lock_write(&self.charts).insert(chart.name.clone(), Arc::new(chart));
        self.invalidate_index();
        Ok(())
    }

    /// Idempotent upsert for the M10 learning loop.
    ///
    /// Before writing, the `workflow_library` index is checked for a
    /// near-neighbor chart (VISION: "update/subsumes it rather than
    /// duplicating"):
    ///
    /// - A near neighbor at or above `near_threshold` → the new draft
    ///   **subsumes** it (keeps the existing chart's name, takes the new
    ///   content). This is how a newer, stronger model re-authors an older
    ///   chart without duplicating it.
    /// - Otherwise → stored under its own name.
    ///
    /// Either way the written chart is a **draft**: excluded from selection
    /// until a rubric-validated run (`record_rubric_result(.., true)`)
    /// promotes it. Fail-fast: an invalid chart is never written.
    pub fn upsert_idempotent(
        &self,
        chart: ChartDef,
        near_threshold: f32,
    ) -> Result<UpsertOutcome, ChartError> {
        chart.validate().map_err(|e| ChartError::Invalid {
            reason: format!("chart '{}' failed validation: {e}", chart.name),
        })?;

        let doc = chart_doc_text(&chart);
        let near = self
            .search(&doc, 1)?
            .into_iter()
            .next()
            .filter(|(name, sim)| name != &chart.name && *sim >= near_threshold);

        let outcome = if let Some((existing_name, _)) = near {
            let mut subsumed = chart;
            subsumed.name.clone_from(&existing_name);
            subsumed.validate().map_err(|e| ChartError::Invalid {
                reason: format!("subsumed chart '{}' failed validation: {e}", subsumed.name),
            })?;
            self.mark_draft(&subsumed.name);
            lock_write(&self.charts).insert(existing_name.clone(), Arc::new(subsumed));
            UpsertOutcome::Subsumed { by: existing_name }
        } else {
            self.mark_draft(&chart.name);
            lock_write(&self.charts).insert(chart.name.clone(), Arc::new(chart));
            UpsertOutcome::Inserted
        };
        self.invalidate_index();
        tracing::info!(
            target: "router.charts.store",
            outcome = ?outcome,
            "idempotent chart upsert",
        );
        Ok(outcome)
    }

    /// Record one rubric-gate result (M9 gate) for `chart` (M10 staleness).
    ///
    /// - A **pass** resets the consecutive-failure streak and promotes a
    ///   draft to selectable (the chart "became validated by a rubric run").
    /// - A **fail** increments the streak; when it reaches `CHART_STALE_FAILS`
    ///   the chart is **demoted** (excluded from selection).
    ///
    /// Returns the demoted chart name when this call crosses the demotion
    /// threshold — the caller should flag it in the audit log. `None`
    /// otherwise.
    pub fn record_rubric_result(&self, chart: &str, passed: bool) -> Option<String> {
        let mut health = lock_write(&self.health);
        let entry = health.entry(chart.to_string()).or_default();
        if passed {
            entry.stale_failures = 0;
            entry.draft = false;
            return None;
        }
        entry.stale_failures += 1;
        if entry.stale_failures >= crate::charts::CHART_STALE_FAILS {
            entry.demoted = true;
            crate::audit::emit(
                "chart_store",
                serde_json::json!({
                    "chart": chart,
                    "stale_failures": entry.stale_failures,
                    "demoted": true,
                }),
            );
            return Some(chart.to_string());
        }
        None
    }

    /// Whether `chart` is demoted (excluded from selection).
    pub fn is_demoted(&self, name: &str) -> bool {
        lock_read(&self.health).get(name).is_some_and(|h| h.demoted)
    }

    /// Whether `chart` is an auto-extracted draft awaiting rubric validation.
    pub fn is_draft(&self, name: &str) -> bool {
        lock_read(&self.health).get(name).is_some_and(|h| h.draft)
    }

    /// Snapshot of `chart`'s health, if present.
    pub fn health(&self, name: &str) -> Option<ChartHealth> {
        lock_read(&self.health).get(name).cloned()
    }

    /// Names of every demoted chart (audit/operator visibility).
    pub fn demoted_charts(&self) -> Vec<String> {
        lock_read(&self.health)
            .iter()
            .filter(|(_, h)| h.demoted)
            .map(|(n, _)| n.clone())
            .collect()
    }

    /// Clear a chart's health back to the default (selectable, no streak).
    fn reset_health(&self, name: &str) {
        lock_write(&self.health).insert(name.to_string(), ChartHealth::default());
    }

    /// Mark a chart as an auto-extracted draft (not yet rubric-validated).
    fn mark_draft(&self, name: &str) {
        let mut health = lock_write(&self.health);
        let entry = health.entry(name.to_string()).or_default();
        entry.draft = true;
        entry.stale_failures = 0;
        entry.demoted = false;
    }

    /// Drop the built retrieval index — the `charts` map changed, so the
    /// HNSW graph and SQLite embeddings are stale until the next build.
    fn invalidate_index(&self) {
        *lock_write(&self.built) = None;
    }

    /// Whether `name` is excluded from selection: demoted, or a draft that
    /// has not yet been rubric-validated (M10).
    fn is_excluded_from_selection(&self, name: &str) -> bool {
        lock_read(&self.health)
            .get(name)
            .is_some_and(|h| h.demoted || h.draft)
    }

    /// Borrow the workflow_library index handle, if configured.
    pub fn index_handle(&self) -> Option<&crate::hnsw::HnswIndexHandle> {
        self.index.as_ref()
    }

    /// `true` if the retrieval index has been built (M7 step 2 can run).
    pub fn is_index_built(&self) -> bool {
        self.built.read().is_ok_and(|g| g.is_some())
    }

    /// All *selectable* charts in a stable (name-sorted) order. Selection
    /// code must not depend on `HashMap` iteration order. Excludes charts
    /// demoted or still-draft (M10 staleness/draft gate) — demoted and
    /// unvalidated drafts are "no longer selected".
    pub fn charts_sorted(&self) -> Vec<Arc<ChartDef>> {
        let guard = lock_read(&self.charts);
        let mut names: Vec<&str> = guard.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
            .into_iter()
            .filter(|n| !self.is_excluded_from_selection(n))
            .filter_map(|n| guard.get(n).cloned())
            .collect()
    }

    /// Build the workflow_library retrieval index (M7 step 2).
    ///
    /// Lazy: with no `index_path` configured this is a no-op. When configured,
    /// one vector per chart is computed over the chart's *document text*
    /// (description + target provides + name n-grams), the cosine HNSW graph
    /// is rebuilt, and embeddings are upserted into the router-side SQLite
    /// file so repeated boots reuse them (idempotent per chart). Fail-fast:
    /// any embedding error aborts the whole build — a half-built index must
    /// not serve (decision D3).
    pub fn build_index(&self, embedder: Arc<dyn EmbeddingProvider>) -> Result<(), ChartError> {
        let Some(handle) = self.index.as_ref() else {
            tracing::warn!(
                target: "router.charts.store",
                "no index_path configured — workflow_library index build skipped"
            );
            return Ok(());
        };

        let path = Path::new(&handle.path);
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        // The workflow_library persistence file is a `fluent_db::SqliteStore`
        // (connection lifecycle, schema init); the in-memory retrieval graph
        // is a `fluent_db::hnsw::HnswIndex`.
        let store = SqliteStore::open(path).map_err(|e| ChartError::Index {
            reason: format!("open workflow_library db {}: {e}", path.display()),
        })?;
        store
            .init_schema(
                "CREATE TABLE IF NOT EXISTS workflow_library (\
                 chart TEXT PRIMARY KEY, doc_text TEXT NOT NULL, embedding BLOB NOT NULL);",
            )
            .map_err(|e| ChartError::Index {
                reason: format!("init workflow_library schema: {e}"),
            })?;

        // All-or-nothing embed pass: compute every chart's vector first; only
        // when all succeed do we build the graph and persist.
        let mut entries: Vec<(String, String, Vec<f32>)> = Vec::with_capacity(self.len());
        for chart in self.charts_sorted() {
            let doc = chart_doc_text(&chart);
            let cached = store.with_conn(|conn| {
                load_cached_embedding(conn, &chart.name, &doc)
                    .map_err(|e| DbError::Other(e.to_string()))
            });
            let embedding = match cached {
                Ok(Some(v)) => v,
                Ok(None) => {
                    let v = embedder.embed(&doc).map_err(|e| ChartError::Index {
                        reason: format!("embed chart '{}': {e}", chart.name),
                    })?;
                    if v.is_empty() {
                        return Err(ChartError::Index {
                            reason: format!(
                                "embedder returned an empty vector for '{}'",
                                chart.name
                            ),
                        });
                    }
                    v
                }
                Err(e) => {
                    return Err(ChartError::Index {
                        reason: match e {
                            DbError::Other(s) => s,
                            other => other.to_string(),
                        },
                    })
                }
            };
            entries.push((chart.name.clone(), doc, embedding));
        }

        let hnsw = HnswIndex::new();
        let mut ids: Vec<String> = Vec::with_capacity(entries.len());
        let mut flat: Vec<(String, Vec<f32>)> = Vec::with_capacity(entries.len());
        for (name, _doc, emb) in &entries {
            // External index (`d_id`) aligns with the `ids` position; the
            // node-id argument is not used for resolution here.
            hnsw.insert(ids.len() as i64, emb);
            ids.push(name.clone());
            flat.push((name.clone(), emb.clone()));
        }

        // Persist (upsert) so a later boot reuses these vectors.
        for (name, doc, emb) in &entries {
            store
                .execute(
                    "INSERT INTO workflow_library (chart, doc_text, embedding) VALUES (?1, ?2, ?3) \
                     ON CONFLICT(chart) DO UPDATE \
                     SET doc_text = excluded.doc_text, embedding = excluded.embedding",
                    rusqlite::params![name, doc, vec_to_bytes(emb)],
                )
                .map_err(|e| ChartError::Index {
                    reason: format!("persist embedding for '{name}': {e}"),
                })?;
        }

        *lock_write(&self.built) = Some(Arc::new(ChartIndex {
            embedder,
            ids,
            flat,
            hnsw,
        }));
        tracing::info!(
            target: "router.charts.store",
            chart_count = entries.len(),
            index_path = %path.display(),
            "workflow_library index built",
        );
        Ok(())
    }

    /// Retrieve the top `k` charts by cosine similarity to the embedded raw
    /// request (M7 step 2). Returns `(chart name, similarity in [0,1])`
    /// sorted most-similar first. An unbuilt index yields no candidates.
    pub fn search(&self, request: &str, k: usize) -> Result<Vec<(String, f32)>, ChartError> {
        let guard = lock_read(&self.built);
        let Some(index) = guard.as_ref() else {
            return Ok(Vec::new());
        };
        let query = index
            .embedder
            .embed(request)
            .map_err(|e| ChartError::Index {
                reason: format!("embed request: {e}"),
            })?;
        if query.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        // Cosine HNSW graph; brute force is the canonical fallback when the
        // graph is empty or returns nothing for this query.
        let mut hits: Vec<(String, f32)> = index
            .hnsw
            .search(&query, k)
            .into_iter()
            .filter_map(|(d_id, distance)| {
                index
                    .ids
                    .get(d_id)
                    .map(|name| (name.clone(), (1.0 - distance).max(0.0)))
            })
            .collect();
        if hits.is_empty() {
            // Borrow the flat list — no full-candidate clone per query; only
            // the top-K names are materialized into `String`s below.
            hits = knn_brute_force(
                &query,
                index.flat.iter().map(|(n, e)| (n.as_str(), e.as_slice())),
                k,
            )
            .into_iter()
            .map(|(name, d)| (name.to_string(), (1.0 - d).max(0.0)))
            .collect();
        }
        hits.sort_by(|a, b| b.1.total_cmp(&a.1));
        hits.truncate(k);
        // Never surface demoted or unvalidated-draft charts (M10).
        hits.retain(|(name, _)| !self.is_excluded_from_selection(name));
        Ok(hits)
    }
}

/// Load a cached embedding for `chart` whose document text still matches.
fn load_cached_embedding(
    conn: &rusqlite::Connection,
    chart: &str,
    doc: &str,
) -> Result<Option<Vec<f32>>, ChartError> {
    let row = fluent_db::query::query_row(
        conn,
        "SELECT doc_text, embedding FROM workflow_library WHERE chart = ?1",
        rusqlite::params![chart],
        |row| Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?)),
    )
    .map_err(|e| ChartError::Index {
        reason: format!("query cache for '{chart}': {e}"),
    })?;
    let Some((cached_doc, blob)) = row else {
        return Ok(None);
    };
    if cached_doc != doc {
        return Ok(None);
    }
    Ok(try_bytes_to_vec(&blob).filter(|v| !v.is_empty()))
}

/// Build the indexed document text for a chart: description + every target's
/// provides assets + name n-grams. One vector per chart (M7 spec).
fn chart_doc_text(chart: &ChartDef) -> String {
    let mut parts: Vec<String> = Vec::new();
    parts.push(chart.description.clone());
    parts.push(chart.name.clone());
    for target in &chart.targets {
        parts.extend(target.provides.iter().cloned());
    }
    for token in name_ngrams(&chart.name) {
        parts.push(token);
    }
    parts.join(" ")
}

/// N-grams of a chart name split on non-alphanumerics, e.g. `bug_triage` →
/// `["bug", "triage"]`. Lets a raw request that names only a fragment of the
/// chart (e.g. "triage this bug") still retrieve it.
fn name_ngrams(name: &str) -> Vec<String> {
    name.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .collect()
}

/// Deserialize a chart directly from JSON (used by tests and the M10
/// extraction path without going through the filesystem).
pub fn chart_from_str(json: &str) -> Result<ChartDef, ChartError> {
    let chart: ChartDef = serde_json::from_str(json).map_err(|e| ChartError::Parse {
        source: e,
        path: "<inline>".into(),
    })?;
    chart.validate().map_err(|e| ChartError::Invalid {
        reason: format!("chart '{}' failed validation: {e}", chart.name),
    })?;
    Ok(chart)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn seed_chart(name: &str, provides: &[&str]) -> String {
        let provides: Vec<String> = provides.iter().map(|p| format!("\"{p}\"")).collect();
        format!(
            r#"{{
                "name": "{name}",
                "description": "seed chart {name}",
                "schema_version": 1,
                "author_model": "human",
                "targets": [
                    {{
                        "name": "target_a",
                        "provides": [{}],
                        "template": "do {{{{ request }}}}",
                        "essential": true
                    }}
                ]
            }}"#,
            provides.join(", ")
        )
    }

    #[test]
    fn load_dir_with_seeded_tempdir() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("alpha.json"),
            seed_chart("alpha", &["a_out"]),
        )
        .unwrap();
        std::fs::write(dir.path().join("beta.json"), seed_chart("beta", &["b_out"])).unwrap();

        let store = ChartStore::new(None);
        store.load_dir(dir.path()).unwrap();
        assert_eq!(store.len(), 2);
        assert!(store.get("alpha").is_some());
        assert!(store.get("beta").is_some());
    }

    #[test]
    fn empty_dir_yields_empty_store() {
        let dir = TempDir::new().unwrap();
        let store = ChartStore::new(None);
        store.load_dir(dir.path()).unwrap();
        assert!(store.is_empty());
        assert!(store.list().is_empty());
    }

    #[test]
    fn missing_dir_yields_empty_store() {
        let missing = std::path::Path::new("/nonexistent/charts/dir");
        let store = ChartStore::new(None);
        store.load_dir(missing).unwrap();
        assert!(store.is_empty());
    }

    #[test]
    fn invalid_file_is_hard_error() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("broken.json"), "not json at all").unwrap();
        let store = ChartStore::new(None);
        let err = store.load_dir(dir.path()).unwrap_err();
        assert!(matches!(err, ChartError::Parse { .. }));
    }

    #[test]
    fn invalid_chart_fails_validation_at_load() {
        let dir = TempDir::new().unwrap();
        // Missing schema_version + a target with no template.
        std::fs::write(
            dir.path().join("bad.json"),
            r#"{
                "name": "bad",
                "description": "bad",
                "schema_version": 1,
                "author_model": "human",
                "targets": [
                    { "name": "t", "provides": ["x"], "template": "" }
                ]
            }"#,
        )
        .unwrap();
        let store = ChartStore::new(None);
        let err = store.load_dir(dir.path()).unwrap_err();
        assert!(matches!(err, ChartError::Invalid { .. }));
    }

    #[test]
    fn upsert_inserts_and_replaces() {
        let store = ChartStore::new(None);
        let chart: ChartDef = chart_from_str(&seed_chart("alpha", &["a_out"])).unwrap();
        store.upsert(chart.clone()).unwrap();
        assert_eq!(store.len(), 1);

        let mut replaced = chart.clone();
        replaced.description = "updated".into();
        store.upsert(replaced.clone()).unwrap();
        assert_eq!(store.len(), 1);
        assert_eq!(store.get("alpha").unwrap().description, "updated");
    }

    #[test]
    fn non_json_files_are_ignored() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("chart.json"),
            seed_chart("alpha", &["a_out"]),
        )
        .unwrap();
        std::fs::write(dir.path().join("README.md"), "not a chart").unwrap();
        let store = ChartStore::new(None);
        store.load_dir(dir.path()).unwrap();
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn golden_loads_real_seed_dir() {
        // Load the real env/workflows/charts seed directory (Appendix A).
        let seed_dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../env/workflows/charts");
        let store = ChartStore::new(None);
        store.load_dir(&seed_dir).expect("seed dir loads");
        assert_eq!(store.len(), 2, "expected exactly 2 seed charts");
        let mut names = store.list();
        names.sort_unstable();
        assert_eq!(names, vec!["bug_triage", "draft_doc"]);
    }

    // ── M10: idempotent upsert + draft gate ─────────────────────────────

    fn indexed_store() -> (ChartStore, tempfile::TempDir) {
        let tmp = tempfile::TempDir::new().unwrap();
        let handle = crate::hnsw::HnswIndexHandle {
            name: "workflow_library".into(),
            path: tmp
                .path()
                .join("workflow_library.sqlite")
                .display()
                .to_string(),
        };
        let store = ChartStore::new(Some(handle));
        let chart = chart_from_str(&seed_chart("alpha", &["a_out"])).unwrap();
        store.upsert(chart).unwrap();
        store
            .build_index(Arc::new(crate::test_stubs::HashEmbedder::new(256)))
            .expect("index builds");
        (store, tmp)
    }

    #[test]
    fn upsert_idempotent_inserts_unrelated_chart_as_draft() {
        let (store, _tmp) = indexed_store();
        let new_chart = chart_from_str(&seed_chart("omega", &["o_out"])).unwrap();
        let outcome = store
            .upsert_idempotent(new_chart, CHART_SUBSUME_THRESHOLD)
            .unwrap();
        assert_eq!(outcome, UpsertOutcome::Inserted);
        // The auto-extracted chart is a draft: present but not selectable.
        assert!(store.is_draft("omega"));
        assert!(
            !store.charts_sorted().iter().any(|c| c.name == "omega"),
            "drafts must not be selectable until rubric-validated"
        );
    }

    #[test]
    fn upsert_idempotent_subsumes_near_neighbor() {
        let (store, _tmp) = indexed_store();
        // A near-duplicate of `alpha` (same description/target assets) must be
        // folded into it, not stored twice. Threshold is deliberately below
        // the crude HashEmbedder's measured near-duplicate cosine (~0.61) —
        // the production `CHART_SUBSUME_THRESHOLD` targets real embeddings.
        let mut dup = chart_from_str(&seed_chart("alpha_copy", &["a_out"])).unwrap();
        dup.description = "seed chart alpha".into(); // identical doc text
        let outcome = store.upsert_idempotent(dup, 0.5).unwrap();
        match outcome {
            UpsertOutcome::Subsumed { by } => assert_eq!(by, "alpha"),
            other @ UpsertOutcome::Inserted => panic!("expected subsume, got {other:?}"),
        }
        assert_eq!(store.len(), 1, "near-neighbor dedup must not duplicate");
        assert!(store.get("alpha").is_some());
        assert!(store.is_draft("alpha"), "subsumed chart is a draft");
    }

    #[test]
    fn upsert_idempotent_subsumed_name_keeps_library_selectable() {
        let (store, _tmp) = indexed_store();
        let mut dup = chart_from_str(&seed_chart("alpha_copy", &["a_out"])).unwrap();
        dup.description = "seed chart alpha".into();
        store.upsert_idempotent(dup, 0.5).unwrap();
        // Even as a draft, the original human chart vanished (replaced).
        // After a rubric-validated run it becomes selectable again.
        store.record_rubric_result("alpha", true);
        assert!(store.charts_sorted().iter().any(|c| c.name == "alpha"));
    }

    // ── M10: staleness / demotion policy ────────────────────────────────

    #[test]
    fn record_rubric_result_demotes_after_stale_fails() {
        let (store, _tmp) = indexed_store();
        for i in 0..crate::charts::CHART_STALE_FAILS {
            let demoted = store.record_rubric_result("alpha", false);
            if i + 1 < crate::charts::CHART_STALE_FAILS {
                assert!(demoted.is_none(), "not yet demoted");
                assert!(!store.is_demoted("alpha"));
            } else {
                assert_eq!(
                    demoted.as_deref(),
                    Some("alpha"),
                    "crossing the threshold demotes the chart"
                );
            }
        }
        assert!(store.is_demoted("alpha"));
        assert_eq!(store.demoted_charts(), vec!["alpha".to_string()]);
        assert!(
            !store.charts_sorted().iter().any(|c| c.name == "alpha"),
            "demoted charts are no longer selected"
        );
    }

    #[test]
    fn record_rubric_result_resets_streak_on_success() {
        let (store, _tmp) = indexed_store();
        store.record_rubric_result("alpha", false);
        store.record_rubric_result("alpha", false);
        // A passing run resets the streak before it crosses the threshold.
        store.record_rubric_result("alpha", true);
        assert!(!store.is_demoted("alpha"));
        assert_eq!(store.health("alpha").unwrap().stale_failures, 0);
    }

    #[test]
    fn record_rubric_result_promotes_draft_on_pass() {
        let (store, _tmp) = indexed_store();
        let new_chart = chart_from_str(&seed_chart("omega", &["o_out"])).unwrap();
        store
            .upsert_idempotent(new_chart, CHART_SUBSUME_THRESHOLD)
            .unwrap();
        assert!(store.is_draft("omega"));
        // One rubric-validated run promotes the draft to selectable.
        store.record_rubric_result("omega", true);
        assert!(!store.is_draft("omega"));
        assert!(store.charts_sorted().iter().any(|c| c.name == "omega"));
    }

    #[test]
    fn demoted_chart_is_also_absent_from_hnsw_search() {
        let (store, _tmp) = indexed_store();
        store.record_rubric_result("alpha", false);
        store.record_rubric_result("alpha", false);
        store.record_rubric_result("alpha", false);
        assert!(store.is_demoted("alpha"));
        let hits = store.search("alpha", 5).unwrap();
        assert!(
            hits.iter().all(|(n, _)| n != "alpha"),
            "demoted chart must not surface via HNSW retrieval"
        );
    }
}

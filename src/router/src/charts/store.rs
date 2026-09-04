//! ChartStore — loads and holds a directory of chart files at boot, in the
//! `yamake_loader` tradition.
//!
//! The store is the single owner of the workflow_library index path; the
//! `index` handle is reserved for retrieval.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::RwLock;

use common_core::registry::ConcurrentRegistry;
use common_core::sync::{lock_read, lock_write};
use fluent_db::error::DbError;
use fluent_db::hnsw::{HnswIndex, HnswIndexHandle};
use fluent_db::store::SqliteStore;
// Canonical vector-math import (M4: the search-vector math shim is deleted;
// the owner `fluent_db::vector` is consumed directly).
use fluent_db::vector::{
    distance_to_similarity_clamped, knn_brute_force, scored_hits, try_bytes_to_vec, vec_to_bytes,
};
use fluent_llm::EmbeddingProvider;

use super::{ChartDef, ChartError};

/// Cosine-similarity threshold for the idempotent-upsert near-neighbor
/// check: a chart whose embedding is at or above this similarity to an
/// existing chart *subsumes* it (VISION: update/subsumes, never duplicate).
/// Router-only constant — promote to `common-core::constants` when a second
/// consumer appears (Consolidation Contract).
pub(crate) const CHART_SUBSUME_THRESHOLD: f32 = 0.9;

/// Per-chart runtime health — the staleness/demotion policy.
///
/// Kept *out* of the persisted `ChartDef` content model: a chart's health is
/// a runtime property of the store, not a declarative fact about the DAG.
#[derive(Debug, Clone, Default)]
pub struct ChartHealth {
    /// Auto-extracted and not yet rubric-validated: excluded from
    /// selection until a rubric-validated run clears the flag.
    pub draft: bool,
    /// Consecutive rubric-gate failures. Reset on any pass.
    pub stale_failures: usize,
    /// Demoted after `CHART_STALE_FAILS` consecutive rubric failures:
    /// excluded from selection, flagged in the audit log. Sticky until the
    /// chart is re-extracted (upsert replaces it).
    pub demoted: bool,
}

/// Outcome of an idempotent chart upsert.
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
    /// (`Arc<ChartStore>`) across the selector and the extraction path,
    /// so runtime upserts go through the lock rather than `&mut self`.
    charts: ConcurrentRegistry<String, ChartDef>,
    /// Chart name → runtime health (draft/staleness/demotion state).
    health: RwLock<HashMap<String, ChartHealth>>,
    /// workflow_library HNSW index path handle.
    index: Option<HnswIndexHandle>,
    /// Built index (HNSW graph + embedder), if `build_index` succeeded.
    built: RwLock<Option<Arc<ChartIndex>>>,
}

impl ChartStore {
    pub fn new(index: Option<HnswIndexHandle>) -> Self {
        Self {
            charts: ConcurrentRegistry::new(),
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
    ///   library must not half-load (fail fast).
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
        self.charts.get(&name.to_string())
    }

    /// All chart names, in insertion order (owned — the store is lock-backed).
    pub fn list(&self) -> Vec<String> {
        self.charts.keys()
    }

    /// Number of loaded charts.
    pub fn len(&self) -> usize {
        self.charts.len()
    }

    /// `true` if the store holds no charts.
    pub fn is_empty(&self) -> bool {
        self.charts.is_empty()
    }

    /// Insert or replace a chart. Used by boot loading and by
    /// auto-extraction upsert. Rejects invalid charts (fail fast).
    ///
    /// A plain upsert resets the chart's health: a fresh chart is
    /// selectable by default (`draft = false`, cleared demotion). The
    /// extraction path uses [`Self::upsert_idempotent`], which writes a
    /// draft and deduplicates against near neighbors.
    pub fn upsert(&self, chart: ChartDef) -> Result<(), ChartError> {
        chart.validate().map_err(|e| ChartError::Invalid {
            reason: format!("chart '{}' failed validation: {e}", chart.name),
        })?;
        self.reset_health(&chart.name);
        self.charts.insert(chart.name.clone(), chart);
        self.invalidate_index();
        Ok(())
    }

    /// Idempotent upsert for the learning loop.
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
            self.charts.insert(existing_name.clone(), subsumed);
            UpsertOutcome::Subsumed { by: existing_name }
        } else {
            self.mark_draft(&chart.name);
            self.charts.insert(chart.name.clone(), chart);
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

    /// Record one rubric-gate result for `chart` (staleness).
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
    /// has not yet been rubric-validated.
    fn is_excluded_from_selection(&self, name: &str) -> bool {
        lock_read(&self.health)
            .get(name)
            .is_some_and(|h| h.demoted || h.draft)
    }

    /// Borrow the workflow_library index handle, if configured.
    pub fn index_handle(&self) -> Option<&HnswIndexHandle> {
        self.index.as_ref()
    }

    /// `true` if the retrieval index has been built.
    pub fn is_index_built(&self) -> bool {
        self.built.read().is_ok_and(|g| g.is_some())
    }

    /// All *selectable* charts in a stable (name-sorted) order. Selection
    /// code must not depend on `HashMap` iteration order. Excludes charts
    /// demoted or still-draft (staleness/draft gate) — demoted and
    /// unvalidated drafts are "no longer selected".
    pub fn charts_sorted(&self) -> Vec<Arc<ChartDef>> {
        let mut names = self.charts.keys();
        names.sort_unstable();
        names
            .into_iter()
            .filter(|n| !self.is_excluded_from_selection(n))
            .filter_map(|n| self.charts.get(&n))
            .collect()
    }

    /// Build the workflow_library retrieval index.
    ///
    /// Lazy: with no `index_path` configured this is a no-op. When configured,
    /// one vector per chart is computed over the chart's *document text*
    /// (description + target provides + name n-grams), the cosine HNSW graph
    /// is rebuilt, and embeddings are upserted into the router-side SQLite
    /// file so repeated boots reuse them (idempotent per chart). Fail-fast:
    /// any embedding error aborts the whole build — a half-built index must
    /// not serve.
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
        //
        // NOTE (M9 decision): single-connection `SqliteStore`, not
        // `SqlitePool` — this is a boot-time index build (sync, single
        // writer, no concurrent serving traffic), and the cached-embedding
        // lookup composes the canonical `with_conn` + `db::query` helpers.
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
    /// request.  Returns `(chart name, similarity in [0,1])` sorted
    /// most-similar first.  An unbuilt index yields no candidates.
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
        // M6: deliberately NOT routed through `AdaptiveHnsw` — this corpus is
        // small and always-indexed; adding a brute-force dispatch gate here
        // would be a behavior change (an extra path), so the store keeps
        // always-probe. The dispatch policy measures [B] cost/recall at
        // `ContentNodeStore` scale only.
        // M5: the HNSW probe + id resolution is the shared
        // `fluent_db::hnsw::hnsw_lookup` (`None` = fall back). The inserted
        // key is the `ids` position, so the resolved key indexes `ids`
        // exactly as the external id did. Similarity mapping, sort/truncate,
        // and the demotion filter below stay call-site code.
        let mut hits: Vec<(String, f32)> = fluent_db::hnsw::hnsw_lookup(&index.hnsw, &query, k)
            .map(|resolved| {
                scored_hits(resolved, distance_to_similarity_clamped)
                    .into_iter()
                    .filter_map(|(key, similarity)| {
                        index.ids.get(key as usize).map(|name| (name.clone(), similarity))
                    })
                    .collect()
            })
            .unwrap_or_default();
        if hits.is_empty() {
            // Borrow the flat list — no full-candidate clone per query; only
            // the top-K names are materialized into `String`s below.
            hits = scored_hits(
                knn_brute_force(
                    &query,
                    index.flat.iter().map(|(n, e)| (n.as_str(), e.as_slice())),
                    k,
                ),
                distance_to_similarity_clamped,
            )
            .into_iter()
            .map(|(name, similarity)| (name.to_string(), similarity))
            .collect();
        }
        hits.sort_by(|a, b| b.1.total_cmp(&a.1));
        hits.truncate(k);
        // Never surface demoted or unvalidated-draft charts.
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
/// provides assets + name n-grams. One vector per chart.
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

/// Deserialize a chart directly from JSON (used by tests and the
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
#[path = "../../tests/charts_store.rs"]
mod tests;

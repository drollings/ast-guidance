//! Reference-only view layer over the shared `ContentNodeStore` (M2).
//!
//! Views are the VISION's read surface for the ledger: they hold node-id
//! lists + per-node fidelity policy + composition, and **never own text**
//! (D4). Text leaves the store through exactly one exit — [`LedgerView::render`]
//! → [`ContentNodeStore::lod_text`] — so there is a single place where a consumer can
//! observe/transform what is rendered.
//!
//! Two concrete views ship:
//!
//! - [`ParallelLedger`] — one store, N views sharing the same `Arc`; each view
//!   picks its own `Lod` (with per-node `with_override`). A lazily-derived tier
//!   is computed **at most once** on the shared node and visible to every view.
//! - [`FilteredLedger<V>`] — composes any inner view with an exclusion set and
//!   an optional render transform. It is the single mechanism for both the PII
//!   frontier view (`views::pii_redacted`, transform = M1's `scrub_for_ledger`)
//!   and rigor's red-team view (exclusion of dead ends, no transform) — one
//!   composition, two consumers (D4).
//!
//! Rendering degrades to LOD0 when a lazy tier is missing or un-derivable (no
//! summarizer) rather than erroring the whole render.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use fluent_types::NodeId;

use crate::ledger_guard;
use crate::node_store::ContentNodeStore;

/// Level of detail, 0..=5 (VISION table). Router-local: `ContentNode` keeps
/// `u8` fields for wire compat (D3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Lod(u8);

impl Lod {
    /// Full text.
    pub const LOD0: Lod = Lod(0);
    /// Compressed but complete.
    pub const LOD1: Lod = Lod(1);
    /// Short summary ≤ 1000 ch.
    pub const LOD2: Lod = Lod(2);
    /// Compact summary ≤ 280 ch.
    pub const LOD3: Lod = Lod(3);
    /// Single line ≤ 80 ch.
    pub const LOD4: Lod = Lod(4);
    /// Name / label.
    pub const LOD5: Lod = Lod(5);

    /// The tier as a `u8` (for `ContentNodeStore::lod_text`).
    pub fn as_u8(self) -> u8 {
        self.0
    }
}

impl TryFrom<u8> for Lod {
    type Error = u8;

    fn try_from(level: u8) -> Result<Self, Self::Error> {
        match level {
            0..=5 => Ok(Lod(level)),
            other => Err(other),
        }
    }
}

impl std::fmt::Display for Lod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LOD{}", self.0)
    }
}

/// A reference-only view over the shared store. Views hold node-id lists and
/// configuration, never owned text (D4). `render()` is provided and is the
/// **only** way text leaves a view.
pub trait LedgerView: Send + Sync {
    /// The shared store this view reads from.
    fn store(&self) -> &ContentNodeStore;

    /// The node ids this view covers, in render order (a snapshot).
    fn node_ids(&self) -> Vec<NodeId>;

    /// Per-view fidelity policy: the `Lod` a given node renders at.
    fn lod_for(&self, id: NodeId) -> Lod;

    /// Filtering hook: `true` excludes the node from `render` without touching
    /// the store (no LOD computation).
    fn exclude(&self, _id: NodeId) -> bool {
        false
    }

    /// Render hook: post-process the text just before it exits the view.
    fn transform(&self, _id: NodeId, text: String) -> String {
        text
    }

    /// Render every non-excluded node's text at its fidelity tier, joined by
    /// newlines. The single text-exit from the store.
    fn render(&self) -> String {
        let store = self.store();
        self.node_ids()
            .into_iter()
            .filter(|id| !self.exclude(*id))
            .filter_map(|id| {
                store
                    .get_node(id)
                    .map(|arc| common_core::sync::lock_read(&arc).id.unwrap_or(id))
            })
            .map(|id| {
                let lod = self.lod_for(id);
                let text = store
                    .lod_text(id, lod.as_u8())
                    .or_else(|_| store.lod_text(id, Lod::LOD0.as_u8())) // degrade to LOD0
                    .unwrap_or_default();
                self.transform(id, text)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// A parallel view over one store: every `ParallelLedger` shares the same
/// `Arc<ContentNodeStore>`, so a lazily-derived tier is computed once and visible to
/// all views. `default_lod` is LOD1; per-node `with_override` wins.
pub struct ParallelLedger {
    store: Arc<ContentNodeStore>,
    node_ids: Vec<NodeId>,
    default_lod: Lod,
    /// Per-node fidelity from day one.
    overrides: HashMap<NodeId, Lod>,
}

impl ParallelLedger {
    /// A view over `store` with no node ids yet. Use `for_session` (or the
    /// builder methods) to populate.
    pub fn new(store: Arc<ContentNodeStore>) -> Self {
        Self {
            store,
            node_ids: Vec::new(),
            default_lod: Lod::LOD1,
            overrides: HashMap::new(),
        }
    }

    /// Set the default fidelity tier for every node without an override.
    #[must_use]
    pub fn with_default_lod(mut self, lod: Lod) -> Self {
        self.default_lod = lod;
        self
    }

    /// Override a single node's fidelity tier (wins over `default_lod`).
    #[must_use]
    pub fn with_override(mut self, id: NodeId, lod: Lod) -> Self {
        self.overrides.insert(id, lod);
        self
    }

    /// A view over a whole session, in insertion order.
    pub fn for_session(store: Arc<ContentNodeStore>, session_id: &str) -> Self {
        let node_ids = store.session_node_ids(session_id);
        Self::new(store).with_node_ids(node_ids)
    }

    fn with_node_ids(mut self, node_ids: Vec<NodeId>) -> Self {
        self.node_ids = node_ids;
        self
    }
}

impl LedgerView for ParallelLedger {
    fn store(&self) -> &ContentNodeStore {
        &self.store
    }

    fn node_ids(&self) -> Vec<NodeId> {
        self.node_ids.clone()
    }

    fn lod_for(&self, id: NodeId) -> Lod {
        self.overrides.get(&id).copied().unwrap_or(self.default_lod)
    }
}

/// Composes any inner view with an exclusion set and an optional render
/// transform. Filtering is free: excluded nodes never trigger LOD derivation.
pub struct FilteredLedger<V> {
    inner: V,
    excluded: HashSet<NodeId>,
    transform: Option<Box<dyn Fn(NodeId, String) -> String + Send + Sync>>,
}

impl<V: LedgerView> FilteredLedger<V> {
    pub fn new<S: std::hash::BuildHasher>(inner: V, excluded: HashSet<NodeId, S>) -> Self {
        Self {
            inner,
            excluded: excluded.into_iter().collect(),
            transform: None,
        }
    }

    #[must_use]
    pub fn with_transform<F>(mut self, f: F) -> Self
    where
        F: Fn(NodeId, String) -> String + Send + Sync + 'static,
    {
        self.transform = Some(Box::new(f));
        self
    }
}

impl<V: LedgerView> LedgerView for FilteredLedger<V> {
    fn store(&self) -> &ContentNodeStore {
        self.inner.store()
    }

    fn node_ids(&self) -> Vec<NodeId> {
        self.inner.node_ids()
    }

    fn lod_for(&self, id: NodeId) -> Lod {
        self.inner.lod_for(id)
    }

    fn exclude(&self, id: NodeId) -> bool {
        self.excluded.contains(&id)
    }

    fn transform(&self, id: NodeId, text: String) -> String {
        self.transform
            .as_ref()
            .map_or(text.clone(), |f| f(id, text))
    }
}

/// The PII frontier view (the VISION-named consumer of M1): a `FilteredLedger`
/// whose transform scrubs every rendered line through the write-path guard.
/// One implementation (`scrub_for_ledger`), two callers (D4/DRY).
pub fn pii_redacted<V: LedgerView, S: std::hash::BuildHasher>(
    inner: V,
    excluded: HashSet<NodeId, S>,
) -> FilteredLedger<V> {
    FilteredLedger::new(inner, excluded)
        .with_transform(|_id, text| ledger_guard::scrub_for_ledger(&text).text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node_store::ContentNodeStore;
    use crate::summarization::Summarizer;
    use crate::test_stubs::CountingBackend;

    fn temp_store() -> Arc<ContentNodeStore> {
        let dir = std::env::temp_dir().join(format!(
            "coral-router-views-{}",
            common_core::hash::uuid_v4()
        ));
        let store = Arc::new(ContentNodeStore::open(&dir).unwrap());
        let _ = std::fs::remove_file(&dir);
        store
    }

    fn counting_store() -> (Arc<ContentNodeStore>, Arc<CountingBackend>) {
        let backend = Arc::new(CountingBackend::new("lazy tier text"));
        let summarizer = Summarizer::new(backend.clone(), 20);
        let store = temp_store();
        store.set_summarizer(summarizer);
        (store, backend)
    }

    #[test]
    fn lod_bounds_and_display() {
        assert_eq!(Lod::try_from(0).unwrap().as_u8(), 0);
        assert_eq!(Lod::try_from(5).unwrap().as_u8(), 5);
        assert!(Lod::try_from(6).is_err());
        assert!(Lod::try_from(255).is_err());
        assert_eq!(Lod::LOD2.to_string(), "LOD2");
        assert_eq!(Lod::LOD0, Lod::try_from(0).unwrap());
    }

    #[test]
    fn parallel_views_share_arc_and_derive_lazy_tier_once() {
        let (store, backend) = counting_store();
        let id = store
            .record_request("sess", "r1", "Shared full text for parallel views.")
            .unwrap();

        let v1 =
            ParallelLedger::for_session(Arc::clone(&store), "sess").with_default_lod(Lod::LOD2);
        let v2 =
            ParallelLedger::for_session(Arc::clone(&store), "sess").with_default_lod(Lod::LOD2);

        let r1 = v1.render();
        let r2 = v2.render();
        assert_eq!(r1, "lazy tier text");
        assert_eq!(r2, "lazy tier text");
        assert_eq!(
            backend.calls(),
            1,
            "two views over one store derive the shared lazy tier exactly once"
        );
        let _ = id;
    }

    #[test]
    fn with_override_wins_over_default_lod() {
        let store = temp_store();
        let a = store
            .record_request("sess", "r1", "short answer A")
            .unwrap();
        let _b = store
            .record_request("sess", "r2", "full text answer B")
            .unwrap();

        let view = ParallelLedger::for_session(Arc::clone(&store), "sess")
            .with_default_lod(Lod::LOD5)
            .with_override(a, Lod::LOD0);

        // a renders LOD0 (override), b renders LOD5 (label == first sentence).
        assert_eq!(view.lod_for(a), Lod::LOD0);
        assert_eq!(view.lod_for(_b), Lod::LOD5);
        let rendered = view.render();
        assert!(rendered.contains("short answer A"));
        assert!(rendered.contains("full text answer B"));
    }

    #[test]
    fn filtered_ledger_exclusion_is_free_no_derivation() {
        let (store, backend) = counting_store();
        let dead = store
            .record_request("sess", "r1", "Dead end rejected text.")
            .unwrap();
        store
            .record_request("sess", "r2", "Live accepted text.")
            .unwrap();

        let base =
            ParallelLedger::for_session(Arc::clone(&store), "sess").with_default_lod(Lod::LOD2);
        let excluded: HashSet<NodeId> = [dead].into_iter().collect();
        let view = FilteredLedger::new(base, excluded);

        let rendered = view.render();
        assert!(!rendered.contains("Dead end"), "excluded node removed");
        assert!(rendered.contains("lazy tier text"), "live node rendered");
        assert_eq!(
            backend.calls(),
            1,
            "only the live node triggers derivation — exclusion is free"
        );
    }

    #[test]
    fn pii_redacted_view_scrubs_email_keeps_clean() {
        let store = temp_store();
        store
            .record_request("sess", "r1", "Contact user@example.com now")
            .unwrap();
        store
            .record_request("sess", "r2", "plain clean text")
            .unwrap();

        let base =
            ParallelLedger::for_session(Arc::clone(&store), "sess").with_default_lod(Lod::LOD0);
        let view = pii_redacted(base, HashSet::new());

        let rendered = view.render();
        assert!(
            rendered.contains("[REDACTED:email]"),
            "email-bearing node scrubbed, got: {rendered}"
        );
        assert!(!rendered.contains("user@example.com"));
        assert!(rendered.contains("plain clean text"));
    }

    #[test]
    fn render_degrades_to_lod0_without_summarizer() {
        let store = temp_store();
        store
            .record_request("sess", "r1", "Full text, no summarizer present.")
            .unwrap();

        // LOD2 is un-derivable (no summarizer) → degrade to LOD0, never error.
        let view =
            ParallelLedger::for_session(Arc::clone(&store), "sess").with_default_lod(Lod::LOD2);
        assert_eq!(view.render(), "Full text, no summarizer present.");
    }

    #[test]
    fn render_of_empty_session_is_empty_string() {
        let store = temp_store();
        let view = ParallelLedger::for_session(store, "absent");
        assert_eq!(view.render(), "");
    }
}

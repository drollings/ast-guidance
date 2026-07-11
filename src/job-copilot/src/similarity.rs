use std::path::{Path, PathBuf};

use search_vector::aliases::SemanticAliases;
use search_vector::math::knn_brute_force;
use serde::{Deserialize, Serialize};

/// A stored field entry with its label, resolved value, optional embedding,
/// and the source that produced it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FieldEntry {
    pub label: String,
    pub value: String,
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
    pub source: String,
}

/// In-memory store of previously-resolved form field values with semantic
/// alias expansion and optional embedding-based similarity search.
pub struct FieldSimilarityStore {
    entries: Vec<FieldEntry>,
    aliases: SemanticAliases,
    path: Option<PathBuf>,
}

impl FieldSimilarityStore {
    /// Create an empty store with default aliases.
    pub fn new() -> Self {
        let mut aliases = SemanticAliases::new();
        aliases.insert(
            "phone",
            vec![
                "telephone".into(),
                "mobile".into(),
                "cell".into(),
                "ph".into(),
            ],
        );
        aliases.insert(
            "email",
            vec!["e-mail".into(), "mail".into(), "electronic-mail".into()],
        );
        aliases.insert(
            "address",
            vec!["street".into(), "addr".into(), "location".into()],
        );
        aliases.insert(
            "name",
            vec![
                "first".into(),
                "last".into(),
                "full".into(),
                "fname".into(),
                "lname".into(),
            ],
        );
        aliases.insert("city", vec!["town".into(), "locality".into()]);
        aliases.insert("state", vec!["province".into(), "region".into()]);
        aliases.insert(
            "zip",
            vec!["postal".into(), "zipcode".into(), "postcode".into()],
        );
        aliases.insert("country", vec!["nation".into()]);
        aliases.insert(
            "company",
            vec!["employer".into(), "org".into(), "organization".into()],
        );
        aliases.insert(
            "url",
            vec!["website".into(), "link".into(), "portfolio".into()],
        );
        aliases.insert("linkedin", vec!["linked-in".into(), "profile".into()]);

        Self {
            entries: Vec::new(),
            aliases,
            path: None,
        }
    }

    /// Set the persistence path for this store.
    #[must_use]
    pub fn with_path(mut self, path: PathBuf) -> Self {
        self.path = Some(path);
        self
    }

    /// Record a new field entry.
    pub fn record(
        &mut self,
        label: String,
        value: String,
        embedding: Option<Vec<f32>>,
        source: String,
    ) {
        self.entries.push(FieldEntry {
            label,
            value,
            embedding,
            source,
        });
    }

    /// Find entries by label using alias expansion and substring matching.
    ///
    /// Returns up to `k` matches, ordered by relevance (exact > alias > substring).
    pub fn find_by_label(&self, label: &str, k: usize) -> Vec<&FieldEntry> {
        let label_lower = label.to_lowercase();
        let expansions = self.aliases.expand(&label_lower);

        let mut exact_matches: Vec<&FieldEntry> = Vec::new();
        let mut alias_matches: Vec<&FieldEntry> = Vec::new();
        let mut substring_matches: Vec<&FieldEntry> = Vec::new();

        for entry in &self.entries {
            let entry_label_lower = entry.label.to_lowercase();
            if entry_label_lower == label_lower {
                exact_matches.push(entry);
            } else if expansions.iter().any(|e| e == &entry_label_lower) {
                alias_matches.push(entry);
            } else if entry_label_lower.contains(&label_lower)
                || label_lower.contains(&entry_label_lower)
            {
                substring_matches.push(entry);
            }
        }

        let mut results = Vec::new();
        results.append(&mut exact_matches);
        results.append(&mut alias_matches);
        results.append(&mut substring_matches);
        results.truncate(k);
        results
    }

    /// Find similar entries using cosine distance on embeddings.
    ///
    /// Returns up to `k` entries sorted by ascending distance (most similar first).
    /// Entries without embeddings are silently skipped.
    pub fn find_similar(
        &self,
        _label: &str,
        embedding: &[f32],
        k: usize,
    ) -> Vec<(&FieldEntry, f32)> {
        let candidates = self
            .entries
            .iter()
            .filter_map(|e| e.embedding.as_ref().map(|emb| (e, emb.clone())));

        knn_brute_force(embedding, candidates, k)
    }

    /// Number of stored entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get all entries (for iteration).
    pub fn entries(&self) -> &[FieldEntry] {
        &self.entries
    }

    /// Get the aliases reference.
    pub fn aliases(&self) -> &SemanticAliases {
        &self.aliases
    }

    /// Save the store to its configured path as JSONL.
    pub fn save(&self) -> Result<(), String> {
        let path = self.path.as_ref().ok_or("no path configured")?;
        let mut lines = Vec::new();
        for entry in &self.entries {
            let line = serde_json::to_string(entry).map_err(|e| format!("serialize entry: {e}"))?;
            lines.push(line);
        }
        let content = lines.join("\n");
        if let Some(parent) = path.parent() {
            common_core::io::ensure_dir(parent).map_err(|e| format!("create dir: {e}"))?;
        }
        common_core::io::write_atomic(path, content.as_bytes())
            .map_err(|e| format!("write store: {e}"))?;
        Ok(())
    }

    /// Load the store from a JSONL file.
    pub fn load(path: &Path) -> Result<Self, String> {
        let content =
            common_core::io::read_to_string_err(path).map_err(|e| format!("read store: {e}"))?;
        let mut store = Self::new();
        store.path = Some(path.to_path_buf());
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let entry: FieldEntry =
                serde_json::from_str(line).map_err(|e| format!("parse entry: {e}"))?;
            store.entries.push(entry);
        }
        Ok(store)
    }
}

impl Default for FieldSimilarityStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entries() -> FieldSimilarityStore {
        let mut store = FieldSimilarityStore::new();
        store.record("First Name".into(), "Ada".into(), None, "profile".into());
        store.record(
            "Email".into(),
            "ada@example.com".into(),
            None,
            "profile".into(),
        );
        store.record("Phone".into(), "555-1234".into(), None, "profile".into());
        store.record("City".into(), "London".into(), None, "profile".into());
        store
    }

    #[test]
    fn empty_store_is_empty() {
        let store = FieldSimilarityStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn record_adds_entry() {
        let mut store = FieldSimilarityStore::new();
        store.record("test".into(), "value".into(), None, "src".into());
        assert_eq!(store.len(), 1);
        assert_eq!(store.entries()[0].label, "test");
    }

    #[test]
    fn find_by_label_exact_match() {
        let store = sample_entries();
        let results = store.find_by_label("First Name", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].value, "Ada");
    }

    #[test]
    fn find_by_label_alias_expansion() {
        let store = sample_entries();
        // "telephone" should match "Phone" via alias
        let results = store.find_by_label("telephone", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].value, "555-1234");
    }

    #[test]
    fn find_by_label_substring_match() {
        let mut store = FieldSimilarityStore::new();
        store.record(
            "Work Email".into(),
            "work@co.com".into(),
            None,
            "profile".into(),
        );
        // "email" should substring-match "Work Email"
        let results = store.find_by_label("email", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].value, "work@co.com");
    }

    #[test]
    fn find_by_label_respects_k_limit() {
        let store = sample_entries();
        let results = store.find_by_label("Name", 1);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn find_by_label_no_match() {
        let store = sample_entries();
        let results = store.find_by_label("favorite_color", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn find_similar_with_embeddings() {
        let mut store = FieldSimilarityStore::new();
        store.record(
            "email".into(),
            "a@b.com".into(),
            Some(vec![1.0, 0.0, 0.0]),
            "test".into(),
        );
        store.record(
            "phone".into(),
            "555".into(),
            Some(vec![0.0, 1.0, 0.0]),
            "test".into(),
        );
        store.record(
            "city".into(),
            "NYC".into(),
            Some(vec![0.0, 0.0, 1.0]),
            "test".into(),
        );

        let query = vec![0.9, 0.1, 0.0]; // close to email
        let results = store.find_similar("email", &query, 2);
        assert_eq!(results.len(), 2);
        // Most similar should be email
        assert_eq!(results[0].0.label, "email");
        assert!(results[0].1 < results[1].1); // distance ascending
    }

    #[test]
    fn find_similar_skips_entries_without_embeddings() {
        let mut store = FieldSimilarityStore::new();
        store.record("email".into(), "a@b.com".into(), None, "test".into());
        store.record(
            "phone".into(),
            "555".into(),
            Some(vec![1.0, 0.0]),
            "test".into(),
        );

        let results = store.find_similar("email", &[0.9, 0.1], 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.label, "phone");
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("store.jsonl");

        let mut store = FieldSimilarityStore::new().with_path(path.clone());
        store.record(
            "email".into(),
            "a@b.com".into(),
            Some(vec![1.0, 0.0]),
            "test".into(),
        );
        store.record("phone".into(), "555".into(), None, "profile".into());
        store.save().unwrap();

        let loaded = FieldSimilarityStore::load(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.entries()[0].label, "email");
        assert_eq!(loaded.entries()[0].embedding, Some(vec![1.0, 0.0]));
        assert_eq!(loaded.entries()[1].label, "phone");
    }

    #[test]
    fn load_empty_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("empty.jsonl");
        std::fs::write(&path, "").unwrap();
        let store = FieldSimilarityStore::load(&path).unwrap();
        assert!(store.is_empty());
    }

    #[test]
    fn default_aliases_expand_correctly() {
        let store = FieldSimilarityStore::new();
        let expansions = store.aliases().expand("phone");
        assert!(expansions.contains(&"phone".to_string()));
        assert!(expansions.contains(&"telephone".to_string()));
        assert!(expansions.contains(&"mobile".to_string()));
    }

    #[test]
    fn find_by_label_case_insensitive() {
        let store = sample_entries();
        let results = store.find_by_label("first name", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].value, "Ada");
    }
}

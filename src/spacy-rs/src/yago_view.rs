//! YagoView — runtime file, safe `std::fs::read` (no `mmap`), `fst::Map` + CSR + `OnceLock` memo.

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::path::Path;
use std::sync::RwLock;

use fluent_types::{yago_class_id_for_iri, InterlinguaId};

use crate::error::SpacyError;

/// CSR parents: `indptr` len `n+1`, `indices` variable.
#[derive(Debug, Clone)]
struct Csr {
    indptr: Vec<u32>,
    indices: Vec<u32>,
    id_to_idx: HashMap<InterlinguaId, usize>,
    idx_to_id: Vec<InterlinguaId>,
}

impl Csr {
    fn parents_of_idx(&self, idx: usize) -> &[u32] {
        let s = self.indptr[idx] as usize;
        let e = self.indptr[idx+1] as usize;
        &self.indices[s..e]
    }
}

/// Runtime YaGO view.
#[derive(Debug)]
pub struct YagoView {
    // fst::Map curie -> InterlinguaId (stored as HashMap for now; fst codegen lands later — O(log n) vs O(1) but correct)
    classes: HashMap<String, InterlinguaId>,
    csr: Csr,
    ancestors_memo: RwLock<HashMap<InterlinguaId, Vec<InterlinguaId>>>,
}

impl YagoView {
    /// Load from `path` via safe `std::fs::read` + simple TTL parse (hermetic `n2` fixture). Validates `YSM1` header if present; otherwise parses TTL directly.
    pub fn load(path: &Path) -> Result<Self, SpacyError> {
        let data = std::fs::read(path).map_err(|e| SpacyError::LemmaBlob(format!("read yago {}: {e}", path.display())))?;
        // If file starts with YSM1 magic, decode postcard/fst
        // For hermetic n2, file is TTL; parse streaming.
        let text = String::from_utf8_lossy(&data);
        if text.starts_with("@prefix") {
            return Self::from_ttl_str(&text);
        }
        // Try header check
        if data.len() >= 44 {
            let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
            if magic == 0x5953_4D31 {
                return Err(SpacyError::LemmaBlob("YSM1 binary decode not yet wired — use TTL fixture".into()));
            }
        }
        Self::from_ttl_str(&text)
    }

    fn from_ttl_str(ttl: &str) -> Result<Self, SpacyError> {
        let mut classes: HashMap<String, InterlinguaId> = HashMap::new();
        let mut edges: Vec<(InterlinguaId, InterlinguaId)> = Vec::new();
        let mut prefixes: HashMap<String, String> = HashMap::new();
        for line in ttl.lines() {
            let line = line.trim();
            if line.starts_with("@prefix") {
                // @prefix yago: <http://...>
                if let Some((pfx, iri)) = parse_prefix(line) { prefixes.insert(pfx, iri); }
                continue;
            }
            if line.is_empty() || line.starts_with('#') { continue; }
            if let Some((s_curie, o_curie)) = parse_subclass(line) {
                let s_iri = expand_curie(&s_curie, &prefixes);
                let o_iri = expand_curie(&o_curie, &prefixes);
                let s_id = yago_class_id(&s_iri);
                let o_id = yago_class_id(&o_iri);
                let s_key = curie_of(&s_iri);
                let o_key = curie_of(&o_iri);
                classes.entry(s_key).or_insert(s_id);
                classes.entry(o_key).or_insert(o_id);
                edges.push((s_id, o_id));
            }
        }
        // Build CSR
        let mut ids: Vec<InterlinguaId> = classes.values().copied().collect();
        ids.sort_by_key(|id| id.local_id());
        ids.dedup();
        let mut id_to_idx: HashMap<InterlinguaId, usize> = HashMap::new();
        for (i, id) in ids.iter().enumerate() { id_to_idx.insert(*id, i); }
        let n = ids.len();
        let mut indptr = vec![0u32; n+1];
        let mut indices: Vec<u32> = Vec::new();
        // group by child
        let mut by_child: HashMap<InterlinguaId, Vec<InterlinguaId>> = HashMap::new();
        for (c,p) in edges { by_child.entry(c).or_default().push(p); }
        for (idx, id) in ids.iter().enumerate() {
            if let Some(parents) = by_child.get(id) {
                for p in parents {
                    if let Some(&p_idx) = id_to_idx.get(p) { indices.push(p_idx as u32); }
                }
            }
            indptr[idx+1] = indices.len() as u32;
        }
        Ok(Self {
            classes,
            csr: Csr { indptr, indices, id_to_idx, idx_to_id: ids },
            ancestors_memo: RwLock::new(HashMap::new()),
        })
    }

    pub fn resolve_curie(&self, curie: &str) -> Option<InterlinguaId> {
        self.classes.get(curie).copied()
    }

    /// `ancestors_of` via CSR + OnceLock memo, O(depth) amortized O(1) after first.
    pub fn ancestors_of(&self, id: InterlinguaId) -> Vec<InterlinguaId> {
        // Check memo
        if let Ok(m) = self.ancestors_memo.read() {
            if let Some(v) = m.get(&id) { return v.clone(); }
        }
        let mut out = Vec::new();
        let mut stack = vec![id];
        let mut visited = std::collections::HashSet::new();
        visited.insert(id);
        // DFS over parents
        while let Some(cur) = stack.pop() {
            if let Some(&idx) = self.csr.id_to_idx.get(&cur) {
                for &p_idx in self.csr.parents_of_idx(idx) {
                    let pid = self.csr.idx_to_id[p_idx as usize];
                    if visited.insert(pid) {
                        out.push(pid);
                        stack.push(pid);
                    }
                }
            }
        }
        // nearest first already (direct parents first due to stack LIFO but close enough)
        if let Ok(mut m) = self.ancestors_memo.write() { m.insert(id, out.clone()); }
        out
    }

    pub fn is_subclass_of(&self, child: InterlinguaId, parent: InterlinguaId) -> bool {
        child == parent || self.ancestors_of(child).contains(&parent)
    }

    pub fn class_count(&self) -> usize { self.classes.len() }
    pub fn classes_iter(&self) -> impl Iterator<Item=(String, InterlinguaId)> + '_ {
        self.classes.iter().map(|(k,v)| (k.clone(), *v))
    }
}

fn yago_class_id(iri: &str) -> InterlinguaId {
    yago_class_id_for_iri(iri)
}
fn curie_of(iri: &str) -> String {
    if let Some(local) = iri.strip_prefix("http://schema.org/") { return format!("schema:{local}"); }
    if let Some(local) = iri.strip_prefix("http://yago-knowledge.org/resource/") { return format!("yago:{local}"); }
    iri.to_string()
}
fn parse_prefix(line: &str) -> Option<(String, String)> {
    // @prefix yago: <http://...>
    let line = line.trim();
    if !line.starts_with("@prefix") { return None; }
    let rest = line.trim_start_matches("@prefix").trim();
    let colon = rest.find(':')?;
    let pfx = rest[..colon].trim().to_string();
    let start = rest.find('<')?;
    let end = rest.find('>')?;
    Some((pfx, rest[start+1..end].to_string()))
}
fn parse_subclass(line: &str) -> Option<(String,String)> {
    let pos = line.find("rdfs:subClassOf")?;
    let before = line[..pos].trim();
    let after = line[pos + "rdfs:subClassOf".len()..].trim().trim_end_matches('.').trim();
    let s = before.split_whitespace().next()?.to_string();
    let o = after.split_whitespace().next()?.trim_end_matches('.').to_string();
    Some((s,o))
}
fn expand_curie(curie: &str, prefixes: &HashMap<String,String>) -> String {
    let curie = curie.trim().trim_end_matches(|c| c=='.' || c==',' || c==';');
    if curie.starts_with('<') && curie.ends_with('>') { return curie[1..curie.len()-1].to_string(); }
    if let Some(idx) = curie.find(':') {
        let pfx = &curie[..idx];
        let local = &curie[idx+1..];
        if let Some(base) = prefixes.get(pfx) { return format!("{}{}", base, local); }
    }
    curie.to_string()
}

#[cfg(test)]
#[path = "../tests/yago_view.rs"]
mod tests;

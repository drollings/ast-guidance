//! Compiles `../../env/en_lemmatizer.json` into SLM2 blob.
use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use serde_json::Value;
const MAGIC: u32 = 0x534C_4D32;
const HEADER_VERSION: u16 = 1;
const SECTION_VERSION: u16 = 2;
fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let json_path = manifest_dir.join("../../env/en_lemmatizer.json");
    println!("cargo:rerun-if-changed={}", json_path.display());
    let text = fs::read_to_string(&json_path).expect("read json");
    let root: Value = serde_json::from_str(&text).expect("parse");
    let rules = root["lemma_rules"].as_object().expect("lemma_rules");
    let exc = root["lemma_exc"].as_object().expect("lemma_exc");
    let index = root["lemma_index"].as_object().expect("lemma_index");
    let mut keys: Vec<&str> = rules.keys().map(String::as_str).collect();
    keys.sort_unstable();
    // Build directory and sections first (payload)
    let mut dir_len = 0usize;
    for k in &keys { dir_len += 1 + k.len() + 6*4; }
    let mut payload = Vec::new();
    // placeholder for dir; we build dir bytes separately
    // Actually we need to compute offsets relative to file start (header 44 + dir)
    let header_len = 44usize;
    let mut off = header_len + dir_len;
    let mut dir: Vec<u8> = Vec::with_capacity(dir_len);
    let mut sections: Vec<u8> = Vec::new();
    let _dir_entries: Vec<Vec<u8>> = Vec::new();
    for k in &keys {
        let rules_bytes = encode_rules(&rules[*k]);
        let index_bytes = index.get(*k).map(encode_index).unwrap_or_default();
        let exc_bytes = exc.get(*k).map(encode_exc).unwrap_or_default();
        let rules_off = off as u32;
        let rules_len = rules_bytes.len() as u32;
        let index_off = (off + rules_bytes.len()) as u32;
        let index_len = index_bytes.len() as u32;
        let exc_off = (off + rules_bytes.len() + index_bytes.len()) as u32;
        let exc_len = exc_bytes.len() as u32;
        off += rules_bytes.len() + index_bytes.len() + exc_bytes.len();
        // dir entry
        dir.push(k.len() as u8);
        dir.extend_from_slice(k.as_bytes());
        dir.extend_from_slice(&rules_off.to_le_bytes());
        dir.extend_from_slice(&rules_len.to_le_bytes());
        dir.extend_from_slice(&index_off.to_le_bytes());
        dir.extend_from_slice(&index_len.to_le_bytes());
        dir.extend_from_slice(&exc_off.to_le_bytes());
        dir.extend_from_slice(&exc_len.to_le_bytes());
        sections.extend_from_slice(&rules_bytes);
        sections.extend_from_slice(&index_bytes);
        sections.extend_from_slice(&exc_bytes);
    }
    payload.extend_from_slice(&dir);
    payload.extend_from_slice(&sections);
    // Compute hashes
    let section_hash = {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        text.hash(&mut h);
        h.finish()
    };
    let crc = {
        let mut h = crc32fast::Hasher::new();
        h.update(&payload);
        h.finalize()
    };
    let sha = {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&payload);
        let out = hasher.finalize();
        let mut a = [0u8;16];
        a.copy_from_slice(&out[..16]);
        a
    };
    let mut blob = Vec::with_capacity(header_len + payload.len() + 4);
    blob.extend_from_slice(&MAGIC.to_le_bytes());
    blob.extend_from_slice(&HEADER_VERSION.to_le_bytes());
    blob.extend_from_slice(&SECTION_VERSION.to_le_bytes());
    blob.extend_from_slice(&section_hash.to_le_bytes());
    blob.extend_from_slice(&(keys.len() as u32).to_le_bytes());
    blob.extend_from_slice(&(header_len as u32).to_le_bytes());
    blob.extend_from_slice(&crc.to_le_bytes());
    blob.extend_from_slice(&sha);
    // foot crc (crc of header+payload)
    blob.extend_from_slice(&payload);
    let foot = {
        let mut h = crc32fast::Hasher::new();
        h.update(&blob);
        h.finalize()
    };
    blob.extend_from_slice(&foot.to_le_bytes());
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR");
    let out_path = Path::new(&out_dir).join("en_lemmas.bin");
    fs::write(&out_path, &blob).expect("write");
    println!("cargo:rustc-env=LEMMA_BLOB={}", out_path.display());
    eprintln!("lemmas blob SLM2: {} bytes -> {}", blob.len(), out_path.display());
}
fn encode_rules(v: &Value) -> Vec<u8> {
    let rules = v.as_array().expect("rules array");
    let mut out = Vec::new();
    out.extend_from_slice(&(rules.len() as u32).to_le_bytes());
    for pair in rules {
        let pair = pair.as_array().expect("rule pair");
        let a = pair[0].as_str().expect("suffix").as_bytes();
        let b = pair[1].as_str().expect("replacement").as_bytes();
        out.push(a.len() as u8);
        out.extend_from_slice(a);
        out.push(b.len() as u8);
        out.extend_from_slice(b);
    }
    out
}
fn encode_index(v: &Value) -> Vec<u8> {
    let words: Vec<&str> = v.as_array().expect("index array").iter().map(|w| w.as_str().expect("word")).collect();
    let mut sorted: Vec<&str> = words;
    sorted.sort_unstable();
    sorted.dedup();
    let mut body = Vec::new();
    for w in &sorted { if w.is_empty(){continue;} body.extend_from_slice(w.as_bytes()); body.push(0); }
    let mut out = Vec::new();
    out.extend_from_slice(&(sorted.len() as u32).to_le_bytes());
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    out
}
fn encode_exc(v: &Value) -> Vec<u8> {
    let table = v.as_object().expect("exc object");
    let mut entries: Vec<(&str, Vec<&str>)> = table.iter().map(|(s,l)| (s.as_str(), l.as_array().expect("lemmas").iter().map(|l| l.as_str().expect("lemma")).collect())).collect();
    entries.sort_unstable_by(|a,b| a.0.cmp(b.0));
    let mut surfaces = Vec::new();
    let mut offsets = Vec::new();
    let mut lemmas = Vec::new();
    for (surface, ls) in &entries { surfaces.extend_from_slice(surface.as_bytes()); surfaces.push(0); offsets.push(lemmas.len() as u32); for l in ls { if l.is_empty(){continue;} lemmas.extend_from_slice(l.as_bytes()); lemmas.push(0);} }
    let mut out = Vec::new();
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    out.extend_from_slice(&(surfaces.len() as u32).to_le_bytes());
    out.extend_from_slice(&surfaces);
    out.extend_from_slice(&((offsets.len()*4) as u32).to_le_bytes());
    for o in &offsets { out.extend_from_slice(&o.to_le_bytes()); }
    out.extend_from_slice(&(lemmas.len() as u32).to_le_bytes());
    out.extend_from_slice(&lemmas);
    out
}

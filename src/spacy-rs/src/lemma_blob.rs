//! The versioned binary lemma blob: per-POS `(rules, index, exc)` tables
//! embedded via `include_bytes!` and exposed with zero-alloc lookups.
//!
//! `build.rs` compiles `../../env/en_lemmatizer.json` into this blob; the layout
//! is documented there. Every lookup is a binary search over sorted,
//! NUL-joined UTF-8 word lists — no `HashMap`/`HashSet` is built at runtime,
//! so loading a language costs one parse of its ~1 MB blob plus a handful of
//! small slices. The blob is self-describing (magic + version), so a
//! mismatched artifact fails loudly at load instead of mis-parsing.

use std::cmp::Ordering;

use crate::error::SpacyError;
use crate::labels::Upos;

/// Magic bytes `"SLM1"` legacy.
pub const BLOB_MAGIC: u32 = 0x534C_4D31;
/// Magic `SLM2` — decoupled lemma artifact.
pub const BLOB_MAGIC_SLM2: u32 = 0x534C_4D32;
/// Current blob layout version (bumped by `build.rs` on format changes).
pub const BLOB_VERSION: u16 = 1;
/// SLM2 section version.
pub const BLOB_VERSION_SLM2: u16 = 2;

#[derive(Debug, Clone)]
struct PosEntry {
    key: &'static str,
    rules: Vec<(&'static str, &'static str)>,
    index: &'static [u8],
    exc: ExcSection,
}

#[derive(Debug, Clone)]
struct ExcSection {
    n: usize,
    surfaces: &'static [u8],
    offsets: &'static [u8],
    lemmas: &'static [u8],
}

/// A parsed lemma blob.
#[derive(Debug, Clone)]
pub struct LemmaBlob {
    pos: Vec<PosEntry>,
}

impl LemmaBlob {
    /// Parse a blob produced by `build.rs`. Supports `SLM1` (legacy 8-byte header) and `SLM2` (44-byte header + foot).
    /// Rejects wrong magic/version, truncated or out-of-range sections, and invalid UTF-8.
    pub fn from_bytes(data: &'static [u8]) -> Result<Self, SpacyError> {
        let err = |m: &str| SpacyError::LemmaBlob(m.to_string());
        let magic = rd_u32(data, 0).ok_or_else(|| err("truncated magic"))?;
        let (n, mut o) = if magic == BLOB_MAGIC_SLM2 {
            // SLM2: header 44 bytes, count at offset 16 (u32), dir starts at section_off (44)
            let header_version = rd_u16(data, 4).ok_or_else(|| err("truncated header version"))?;
            if header_version != 1 {
                return Err(err(&format!("unsupported SLM2 header version {header_version}")));
            }
            let section_version = rd_u16(data, 6).ok_or_else(|| err("truncated section version"))?;
            if section_version != BLOB_VERSION_SLM2 {
                return Err(err(&format!("unsupported SLM2 section version {section_version}")));
            }
            let count = rd_u32(data, 16).ok_or_else(|| err("truncated pos count"))? as usize;
            let section_off = rd_u32(data, 20).ok_or_else(|| err("truncated section off"))? as usize;
            // optional crc/sha validation (skip foot)
            (count, section_off)
        } else if magic == BLOB_MAGIC {
            let version = rd_u16(data, 4).ok_or_else(|| err("truncated version"))?;
            if version != BLOB_VERSION {
                return Err(err(&format!("unsupported blob version {version}")));
            }
            let count = usize::from(rd_u16(data, 6).ok_or_else(|| err("truncated pos count"))?);
            (count, 8usize)
        } else {
            return Err(err("bad magic (expected \"SLM1\" or \"SLM2\")"));
        };
        let mut pos = Vec::with_capacity(n);
        for _ in 0..n {
            let key = rd_str(data, &mut o).ok_or_else(|| err("truncated pos key"))?;
            let rules_off = rd_usize(data, &mut o).ok_or_else(|| err("truncated rules offset"))?;
            let rules_len = rd_usize(data, &mut o).ok_or_else(|| err("truncated rules length"))?;
            let index_off = rd_usize(data, &mut o).ok_or_else(|| err("truncated index offset"))?;
            let index_len = rd_usize(data, &mut o).ok_or_else(|| err("truncated index length"))?;
            let exc_off = rd_usize(data, &mut o).ok_or_else(|| err("truncated exc offset"))?;
            let exc_len = rd_usize(data, &mut o).ok_or_else(|| err("truncated exc length"))?;
            let rules_raw = slice(data, rules_off, rules_len)
                .ok_or_else(|| err("rules section out of range"))?;
            let index_raw = slice(data, index_off, index_len)
                .ok_or_else(|| err("index section out of range"))?;
            let exc_raw =
                slice(data, exc_off, exc_len).ok_or_else(|| err("exc section out of range"))?;
            let rules = parse_rules(rules_raw)?;
            let index = parse_index(index_raw)?;
            let exc = parse_exc(exc_raw)?;
            pos.push(PosEntry { key, rules, index, exc });
        }
        Ok(Self { pos })
    }

    /// Number of POS keys with tables.
    #[must_use]
    pub fn pos_count(&self) -> usize {
        self.pos.len()
    }

    /// The POS keys present (e.g. `"noun"`, `"verb"`, `"punct"`).
    pub fn pos_keys(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.pos.iter().map(|p| p.key)
    }

    /// The suffix rules for `key`, or an empty slice if absent.
    #[must_use]
    pub fn rules(&self, key: &str) -> &[(&'static str, &'static str)] {
        self.pos
            .iter()
            .find(|p| p.key == key)
            .map_or(&[], |p| p.rules.as_slice())
    }

    /// Whether any rule table covers `pos` (the enum-keyed twin of the
    /// `&str` probe — the lemmatizer matches on [`Upos`], never on a
    /// string literal).
    #[must_use]
    pub fn has_pos(&self, pos: Upos) -> bool {
        let key = pos.lemma_key();
        self.pos.iter().any(|p| p.key == key)
    }

    /// The suffix rules for `pos`, or an empty slice if absent.
    #[must_use]
    pub fn rules_for(&self, pos: Upos) -> &[(&'static str, &'static str)] {
        self.rules(pos.lemma_key())
    }

    /// Whether `word` is a known lemma-index word for `pos` (binary search).
    #[must_use]
    pub fn index_contains_pos(&self, pos: Upos, word: &str) -> bool {
        self.index_contains(pos.lemma_key(), word)
    }

    /// The NUL-joined lemma list for an exception surface under `pos`, if it
    /// exists (binary search). Split on `\0` for the individual lemmas.
    #[must_use]
    pub fn exc_for_pos(&self, pos: Upos, surface: &str) -> Option<&'static [u8]> {
        self.exc_for(pos.lemma_key(), surface)
    }

    /// Whether `word` is a known lemma-index word for `key` (binary search).
    #[must_use]
    pub fn index_contains(&self, key: &str, word: &str) -> bool {
        self.pos
            .iter()
            .find(|p| p.key == key)
            .is_some_and(|p| bsearch_words(p.index, word.as_bytes()).is_some())
    }

    /// The NUL-joined lemma list for an exception surface under `key`, if it
    /// exists (binary search). Split on `\0` for the individual lemmas.
    #[must_use]
    pub fn exc_for(&self, key: &str, surface: &str) -> Option<&'static [u8]> {
        let p = self.pos.iter().find(|p| p.key == key)?;
        let i = bsearch_words(p.exc.surfaces, surface.as_bytes())?;
        let start = rd_u32(p.exc.offsets, i * 4)? as usize;
        let end = if i + 1 < p.exc.n {
            rd_u32(p.exc.offsets, (i + 1) * 4)? as usize
        } else {
            p.exc.lemmas.len()
        };
        p.exc.lemmas.get(start..end)
    }
}

/// Binary search a sorted, NUL-joined word list for `target`, returning the
/// index of the matching word. Requires no empty tokens (guaranteed by
/// `build.rs`). The word index is derived by counting NUL separators, which is
/// exactly how the entry index maps into the exc offset table.
#[allow(clippy::naive_bytecount)]
fn bsearch_words(words: &[u8], target: &[u8]) -> Option<usize> {
    let mut lo = 0usize;
    let mut hi = words.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let start = words[..mid]
            .iter()
            .rposition(|&b| b == 0)
            .map_or(0, |sep| sep + 1);
        let end = words[mid..]
            .iter()
            .position(|&b| b == 0)
            .map_or(words.len(), |rel| mid + rel);
        if end <= start {
            // Empty token (double NUL); cannot occur in built data.
            lo = end + 1;
            continue;
        }
        match words[start..end].cmp(target) {
            Ordering::Equal => {
                let idx = words[..start].iter().filter(|&&b| b == 0).count();
                return Some(idx);
            }
            Ordering::Less => {
                lo = if end + 1 < words.len() {
                    end + 1
                } else {
                    words.len()
                };
            }
            Ordering::Greater => hi = start,
        }
    }
    None
}

fn parse_rules(raw: &'static [u8]) -> Result<Vec<(&'static str, &'static str)>, SpacyError> {
    let err = |m: &str| SpacyError::LemmaBlob(m.to_string());
    if raw.is_empty() {
        return Ok(Vec::new());
    }
    let n = rd_u32(raw, 0).ok_or_else(|| err("truncated rule count"))? as usize;
    let mut o = 4usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let old = rd_str(raw, &mut o).ok_or_else(|| err("truncated rule suffix"))?;
        let new = rd_str(raw, &mut o).ok_or_else(|| err("truncated rule replacement"))?;
        out.push((old, new));
    }
    Ok(out)
}

fn parse_index(raw: &'static [u8]) -> Result<&'static [u8], SpacyError> {
    let err = |m: &str| SpacyError::LemmaBlob(m.to_string());
    if raw.is_empty() {
        return Ok(&[]);
    }
    let _n_words = rd_u32(raw, 0).ok_or_else(|| err("truncated index count"))?;
    let words_len = rd_u32(raw, 4).ok_or_else(|| err("truncated index words len"))? as usize;
    slice(raw, 8, words_len).ok_or_else(|| err("index words out of range"))
}

fn parse_exc(raw: &'static [u8]) -> Result<ExcSection, SpacyError> {
    let err = |m: &str| SpacyError::LemmaBlob(m.to_string());
    if raw.is_empty() {
        return Ok(ExcSection {
            n: 0,
            surfaces: &[],
            offsets: &[],
            lemmas: &[],
        });
    }
    let n = rd_u32(raw, 0).ok_or_else(|| err("truncated exc count"))? as usize;
    let surfaces_len = rd_u32(raw, 4).ok_or_else(|| err("truncated exc surfaces len"))? as usize;
    let surfaces =
        slice(raw, 8, surfaces_len).ok_or_else(|| err("exc surfaces out of range"))?;
    let o = 8 + surfaces_len;
    let offsets_len = rd_u32(raw, o).ok_or_else(|| err("truncated exc offsets len"))? as usize;
    if offsets_len != n * 4 {
        return Err(err("exc offset table size mismatch"));
    }
    let offsets = slice(raw, o + 4, offsets_len).ok_or_else(|| err("exc offsets out of range"))?;
    let lemmas_len = rd_u32(raw, o + 4 + offsets_len)
        .ok_or_else(|| err("truncated exc lemmas len"))? as usize;
    let lemmas =
        slice(raw, o + 8 + offsets_len, lemmas_len).ok_or_else(|| err("exc lemmas out of range"))?;
    Ok(ExcSection { n, surfaces, offsets, lemmas })
}

fn slice(b: &[u8], off: usize, len: usize) -> Option<&[u8]> {
    b.get(off..off.checked_add(len)?)
}

fn rd_u16(b: &[u8], o: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(o..o + 2)?.try_into().ok()?))
}

fn rd_u32(b: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?))
}

fn rd_usize(b: &[u8], o: &mut usize) -> Option<usize> {
    let v = rd_u32(b, *o)? as usize;
    *o += 4;
    Some(v)
}

fn rd_str<'a>(b: &'a [u8], o: &mut usize) -> Option<&'a str> {
    let n = usize::from(*b.get(*o)?);
    *o += 1;
    let s = b.get(*o..*o + n)?;
    *o += n;
    std::str::from_utf8(s).ok()
}

#[cfg(test)]
#[path = "../tests/lemma_blob.rs"]
mod tests;

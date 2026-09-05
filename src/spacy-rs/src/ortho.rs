//! Tagger orthography: the string fragments and suffixes the ArcEager
//! heuristic parser evaluates, loaded from a versioned binary blob.
//!
//! Every literal the parser matches (morpheme suffixes, clause punctuation,
//! sentence boundaries) lives in `env/en_orthography.json`, compiled by
//! `build.rs` into the `SOR1` blob embedded as [`crate::lang::en::ORTHO_BLOB`].
//! A new language ships a new blob ([`TaggerOrtho::from_bytes`]), never a
//! parser edit. Load-time parsing allocates the small lookup slices once;
//! every predicate below is allocation-free on the hot path.
//!
//! Blob layout (little-endian; see `build.rs`):
//! ```text
//! u32 magic "SOR1" (0x534F5231), u16 version (=1),
//! 3 × lists [u32 count][u32 bytelen][NUL-joined UTF-8]:
//!   trailing punctuation, sentence boundaries, bare-demonstrative follows,
//! 5 × scalars [u32 bytelen][UTF-8 bytes]:
//!   plural-s, manner-ly, past-ed, participial-ing, clause comma.
//! ```
//! Deliberately no CRC/SHA (unlike the ~1 MB lemma blob): at ~10² bytes the
//! magic + version + exhaustive bounds checks fail just as loudly on drift.
//! Empty scalars and NUL-containing entries are rejected — an empty suffix
//! would match every word, so that data bug must fail at load, not parse.

use crate::error::SpacyError;

/// Magic bytes `"SOR1"` — tagger orthography artifact.
pub const BLOB_MAGIC_SOR1: u32 = 0x534F_5231;
/// Current blob layout version (bumped by `build.rs` on format changes).
pub const BLOB_VERSION_SOR1: u16 = 1;

/// The parser-facing language orthography: morpheme spellings plus the
/// punctuation sets the tagger frames evaluate. Borrowed from the blob
/// (English: the embedded [`crate::lang::en::ORTHO_BLOB`]); swapping
/// languages swaps the blob, never the parser.
#[derive(Debug, Clone)]
pub struct TaggerOrtho<'a> {
    trailing: Vec<&'a str>,
    boundary: Vec<&'a str>,
    bare_follow: Vec<&'a str>,
    plural_s: &'a str,
    manner_ly: &'a str,
    past_ed: &'a str,
    part_ing: &'a str,
    comma: &'a str,
}

fn rd_u32(data: &[u8], o: usize) -> Option<u32> {
    data.get(o..o + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn rd_u16(data: &[u8], o: usize) -> Option<u16> {
    data.get(o..o + 2).map(|b| u16::from_le_bytes([b[0], b[1]]))
}

fn split_list<'a>(body: &'a [u8], count: usize, what: &str) -> Result<Vec<&'a str>, SpacyError> {
    let err = |m: String| SpacyError::OrthoBlob(m);
    let text = std::str::from_utf8(body).map_err(|_| err(format!("{what} is not UTF-8")))?;
    // NUL-joined with a trailing NUL: exactly count + 1 splits.
    let parts: Vec<&str> = text.split('\0').collect();
    if parts.len() != count + 1 || parts[count] != "" {
        return Err(err(format!("{what} framing drift")));
    }
    let items: Vec<&str> = parts[..count].to_vec();
    if items.iter().any(|w| w.is_empty()) {
        return Err(err(format!("{what} holds an empty entry")));
    }
    Ok(items)
}

impl<'a> TaggerOrtho<'a> {
    /// Parse a blob produced by `build.rs`. Rejects wrong magic/version,
    /// truncation, non-UTF-8 sections, framing drift, and empty entries or
    /// scalars — every failure is loud, never a silent misparse.
    pub fn from_bytes(data: &'a [u8]) -> Result<Self, SpacyError> {
        let err = |m: String| SpacyError::OrthoBlob(m);
        let magic = rd_u32(data, 0).ok_or_else(|| err("truncated magic".to_string()))?;
        if magic != BLOB_MAGIC_SOR1 {
            return Err(err(format!("bad magic {magic:#x}")));
        }
        let version = rd_u16(data, 4).ok_or_else(|| err("truncated version".to_string()))?;
        if version != BLOB_VERSION_SOR1 {
            return Err(err(format!("unsupported version {version}")));
        }
        let mut o = 6usize;
        let mut lists = Vec::with_capacity(3);
        for what in ["trailing", "boundary", "bare_follow"] {
            let count =
                rd_u32(data, o).ok_or_else(|| err(format!("{what} truncated count")))? as usize;
            let len =
                rd_u32(data, o + 4).ok_or_else(|| err(format!("{what} truncated length")))? as usize;
            o += 8;
            if count == 0 {
                return Err(err(format!("{what} must not be empty")));
            }
            let body = data.get(o..o + len).ok_or_else(|| err(format!("{what} truncated body")))?;
            o += len;
            lists.push(split_list(body, count, what)?);
        }
        let mut scalars = Vec::with_capacity(5);
        for what in ["plural_s", "manner_ly", "past_ed", "part_ing", "comma"] {
            let len =
                rd_u32(data, o).ok_or_else(|| err(format!("{what} truncated length")))? as usize;
            o += 4;
            let body = data.get(o..o + len).ok_or_else(|| err(format!("{what} truncated body")))?;
            o += len;
            let s = std::str::from_utf8(body).map_err(|_| err(format!("{what} is not UTF-8")))?;
            if s.is_empty() || s.contains('\0') {
                return Err(err(format!("{what} must be non-empty NUL-free")));
            }
            scalars.push(s);
        }
        if o != data.len() {
            return Err(err("trailing bytes after orthography blob".to_string()));
        }
        let mut lists = lists.into_iter();
        let mut scalars = scalars.into_iter();
        Ok(Self {
            trailing: lists.next().expect("trailing"),
            boundary: lists.next().expect("boundary"),
            bare_follow: lists.next().expect("bare_follow"),
            plural_s: scalars.next().expect("plural_s"),
            manner_ly: scalars.next().expect("manner_ly"),
            past_ed: scalars.next().expect("past_ed"),
            part_ing: scalars.next().expect("part_ing"),
            comma: scalars.next().expect("comma"),
        })
    }

    /// The embedded English orthography. Fails loudly at startup if the
    /// build-time artifact drifts (same contract as the lemma blob).
    pub fn english() -> TaggerOrtho<'static> {
        TaggerOrtho::from_bytes(crate::lang::en::ORTHO_BLOB)
            .expect("embedded English orthography blob is valid (build.rs)")
    }

    /// Allocation-free case-insensitive ASCII suffix check — the single
    /// engine behind the four morpheme predicates.
    #[inline]
    fn ends_with_ci(word: &str, suffix: &str) -> bool {
        word.len() >= suffix.len()
            && word
                .get(word.len() - suffix.len()..)
                .is_some_and(|sfx| sfx.eq_ignore_ascii_case(suffix))
    }

    /// Morpheme predicates (role-named: a language without the category
    /// leaves the spelling to its own blob; the parser never names one).
    #[inline]
    pub fn is_plural_s(&self, word: &str) -> bool {
        Self::ends_with_ci(word, self.plural_s)
    }
    #[inline]
    pub fn is_manner_ly(&self, word: &str) -> bool {
        Self::ends_with_ci(word, self.manner_ly)
    }
    #[inline]
    pub fn is_past_ed(&self, word: &str) -> bool {
        Self::ends_with_ci(word, self.past_ed)
    }
    #[inline]
    pub fn is_part_ing(&self, word: &str) -> bool {
        Self::ends_with_ci(word, self.part_ing)
    }

    /// Single-token punctuation predicates.
    #[inline]
    pub fn is_comma(&self, token: &str) -> bool {
        token == self.comma
    }
    #[inline]
    pub fn is_trailing_punct(&self, token: &str) -> bool {
        self.trailing.iter().any(|p| *p == token)
    }
    #[inline]
    pub fn is_boundary(&self, token: &str) -> bool {
        self.boundary.iter().any(|p| *p == token)
    }
    #[inline]
    pub fn is_bare_follow(&self, token: &str) -> bool {
        self.bare_follow.iter().any(|p| *p == token)
    }

    /// Whether only clause-terminating punctuation follows position `from`.
    #[inline]
    pub fn trailing_only(&self, texts: &[String], from: usize) -> bool {
        texts[from..].iter().all(|t| self.is_trailing_punct(t))
    }

    /// Whether token `i` opens a sentence: position 0 or right after a
    /// sentence-terminating boundary.
    #[inline]
    pub fn starts_sentence(&self, texts: &[String], i: usize) -> bool {
        i == 0 || self.is_boundary(&texts[i - 1])
    }

    /// Blob accessors (audit surface: what each language configures).
    pub fn trailing_punct(&self) -> &[&'a str] {
        &self.trailing
    }
    pub fn sentence_boundary(&self) -> &[&'a str] {
        &self.boundary
    }
    pub fn bare_follow(&self) -> &[&'a str] {
        &self.bare_follow
    }
    pub fn plural_s(&self) -> &'a str {
        self.plural_s
    }
    pub fn manner_ly(&self) -> &'a str {
        self.manner_ly
    }
    pub fn past_ed(&self) -> &'a str {
        self.past_ed
    }
    pub fn part_ing(&self) -> &'a str {
        self.part_ing
    }
    pub fn comma(&self) -> &'a str {
        self.comma
    }
}

#[path = "../tests/ortho.rs"]
#[cfg(test)]
mod tests;

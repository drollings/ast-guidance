//! Taxonomy blob spec — decoupled Lemma (`SLM2`) + YaGO (`YSM1`) artifacts.
//!
//! Sole spec crate per roadmap §1/§2: `src/common-core/src/blob_spec.rs`.
//! All slices `&'static [u8]` zero-copy, little-endian, 8-byte aligned, validated via `rd_u32`/`slice` helpers.

#![forbid(unsafe_code)]

/// Magic `SLM2` = 0x534C_4D32.
pub const LEMMA_MAGIC: u32 = 0x534C_4D32;
/// Magic `YSM1` = 0x5953_4D31.
pub const YAGO_MAGIC: u32 = 0x5953_4D31;
/// Legacy `SLM1` reader kept one release for rollback (`SLM1` = 0x534C_4D31).
pub const LEGACY_LEMMA_MAGIC: u32 = 0x534C_4D31;

/// Header version (format of the header itself).
pub const HEADER_VERSION: u16 = 1;

/// Common header shape (per-artifact).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlobHeader {
    pub magic: u32,
    pub header_version: u16,
    pub section_version: u16,
    pub section_hash: u64,
    pub count: u32,
    pub section_off: u32,
    pub crc32: u32,
    pub sha256: [u8; 16],
}

impl BlobHeader {
    pub const LEN: usize = 44;
    pub fn is_lemma(&self) -> bool {
        self.magic == LEMMA_MAGIC
    }
    pub fn is_yago(&self) -> bool {
        self.magic == YAGO_MAGIC
    }
}

/// Legacy SLM1 magic check.
#[must_use]
pub fn is_legacy_sml1(magic: u32) -> bool {
    magic == LEGACY_LEMMA_MAGIC
}

#[derive(Debug, thiserror::Error)]
pub enum BlobError {
    #[error("truncated header")]
    Truncated,
    #[error("bad magic {0:#010x}")]
    BadMagic(u32),
    #[error("unsupported header version {0}")]
    UnsupportedVersion(u16),
    #[error("crc mismatch")]
    CrcMismatch,
    #[error("sha mismatch")]
    ShaMismatch,
    #[error("out of range")]
    OutOfRange,
}

/// Read little-endian primitives (no unsafe, no mmap).
#[inline]
pub fn rd_u16(b: &[u8], o: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(o..o + 2)?.try_into().ok()?))
}
#[inline]
pub fn rd_u32(b: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?))
}
#[inline]
pub fn rd_u64(b: &[u8], o: usize) -> Option<u64> {
    Some(u64::from_le_bytes(b.get(o..o + 8)?.try_into().ok()?))
}
#[inline]
pub fn slice(b: &[u8], off: usize, len: usize) -> Option<&[u8]> {
    b.get(off..off.checked_add(len)?)
}

pub fn parse_header(data: &[u8]) -> Result<BlobHeader, BlobError> {
    if data.len() < BlobHeader::LEN {
        return Err(BlobError::Truncated);
    }
    let magic = rd_u32(data, 0).ok_or(BlobError::Truncated)?;
    if magic != LEMMA_MAGIC && magic != YAGO_MAGIC && magic != LEGACY_LEMMA_MAGIC {
        return Err(BlobError::BadMagic(magic));
    }
    let header_version = rd_u16(data, 4).ok_or(BlobError::Truncated)?;
    if header_version != HEADER_VERSION {
        return Err(BlobError::UnsupportedVersion(header_version));
    }
    let section_version = rd_u16(data, 6).ok_or(BlobError::Truncated)?;
    let section_hash = rd_u64(data, 8).ok_or(BlobError::Truncated)?;
    let count = rd_u32(data, 16).ok_or(BlobError::Truncated)?;
    let section_off = rd_u32(data, 20).ok_or(BlobError::Truncated)?;
    let crc32 = rd_u32(data, 24).ok_or(BlobError::Truncated)?;
    let mut sha256 = [0u8; 16];
    sha256.copy_from_slice(data.get(28..44).ok_or(BlobError::Truncated)?);
    Ok(BlobHeader {
        magic,
        header_version,
        section_version,
        section_hash,
        count,
        section_off,
        crc32,
        sha256,
    })
}

/// Validate `crc32` (IEEE) of payload slice.
pub fn validate_crc(payload: &[u8], expected: u32) -> Result<(), BlobError> {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(payload);
    if hasher.finalize() != expected {
        return Err(BlobError::CrcMismatch);
    }
    Ok(())
}

/// Validate first 16 bytes of sha256(payload) against header.
pub fn validate_sha(payload: &[u8], expected: &[u8; 16]) -> Result<(), BlobError> {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(payload);
    let out = h.finalize();
    if &out[..16] != expected {
        return Err(BlobError::ShaMismatch);
    }
    Ok(())
}

/// The ISP trait behind which `fst`/`phf` are hidden.
pub trait LemmaView: Send + Sync {
    fn index_contains(&self, key: &str, word: &str) -> bool;
    fn exc_for(&self, key: &str, surface: &str) -> Option<&[u8]>;
    fn rules_for(&self, key: &str) -> &[(String, String)];
    fn pos_keys(&self) -> Vec<String>;
}

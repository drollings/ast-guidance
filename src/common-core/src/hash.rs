//! Hashing utilities: `blake3_*`, `sha256_*`, `fnv1a64`, `HashState`, `hash_file`/`hash_batch`.

use blake3::Hasher;
use sha2::{Digest, Sha256, Sha512};
use std::io::Read;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashAlgorithm {
    Sha256,
    Sha512,
    Blake3,
}

impl HashAlgorithm {
    pub fn digest_length(self) -> usize {
        match self {
            HashAlgorithm::Sha512 => 64,
            HashAlgorithm::Sha256 | HashAlgorithm::Blake3 => 32,
        }
    }
}

pub fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    hex_encode(&result)
}

pub fn blake3_hash(data: &[u8]) -> [u8; 32] {
    blake3::hash(data).into()
}

/// Compute a BLAKE3 content hash and return it as a lowercase hex string.
///
/// # Examples
///
/// ```
/// use common_core::hash::blake3_hex;
///
/// let hash = blake3_hex(b"hello world");
/// assert_eq!(hash.len(), 64); // 32 bytes = 64 hex chars
/// assert!(!hash.is_empty());
/// ```
pub fn blake3_hex(data: &[u8]) -> String {
    blake3::hash(data).to_hex().to_string()
}

pub fn fnv1a64(input: &[u8]) -> u64 {
    const OFFSET: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;
    let mut hash: u64 = OFFSET;
    for &byte in input {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

pub fn content_hash_with_model(content: &str, model: &str) -> [u8; 16] {
    let mut hasher = Sha256::new();
    hasher.update(model.as_bytes());
    hasher.update(b"\x00");
    hasher.update(content.as_bytes());
    let result = hasher.finalize();
    let mut out = [0u8; 16];
    out.copy_from_slice(&result[..16]);
    out
}

enum HashStateInner {
    Sha256(Sha256),
    Sha512(Sha512),
    Blake3(Box<Hasher>),
}

pub struct HashState {
    inner: HashStateInner,
}

impl HashState {
    pub fn new(algorithm: HashAlgorithm) -> Self {
        match algorithm {
            HashAlgorithm::Sha256 => Self {
                inner: HashStateInner::Sha256(Sha256::new()),
            },
            HashAlgorithm::Sha512 => Self {
                inner: HashStateInner::Sha512(Sha512::new()),
            },
            HashAlgorithm::Blake3 => Self {
                inner: HashStateInner::Blake3(Box::new(Hasher::new())),
            },
        }
    }

    pub fn update(&mut self, data: &[u8]) {
        match &mut self.inner {
            HashStateInner::Sha256(h) => h.update(data),
            HashStateInner::Sha512(h) => h.update(data),
            HashStateInner::Blake3(h) => {
                h.update(data);
            }
        }
    }

    pub fn finalize(&self) -> Vec<u8> {
        match &self.inner {
            HashStateInner::Sha256(h) => h.clone().finalize().to_vec(),
            HashStateInner::Sha512(h) => h.clone().finalize().to_vec(),
            HashStateInner::Blake3(h) => h.clone().finalize().as_bytes().to_vec(),
        }
    }

    pub fn digest_hex(&self) -> String {
        hex_encode(&self.finalize())
    }
}

#[derive(Debug, Clone)]
pub struct BatchHashResult {
    pub path: String,
    pub hash: Option<String>,
}

pub fn hash_file(path: &Path, algorithm: HashAlgorithm) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = HashState::new(algorithm);
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.digest_hex())
}

pub fn hash_batch(paths: &[std::path::PathBuf], algorithm: HashAlgorithm) -> Vec<BatchHashResult> {
    paths
        .iter()
        .map(|path| {
            let hash = hash_file(path, algorithm).ok();
            BatchHashResult {
                path: path.to_string_lossy().to_string(),
                hash,
            }
        })
        .collect()
}

pub fn sha256_digest(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

pub fn hex_encode(data: &[u8]) -> String {
    data.iter()
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .concat()
}

/// Generate a time-based pseudo-UUID v4 string.
///
/// Uses `SystemTime` nanoseconds, process ID, and `fastrand` for entropy.
/// Formatted as `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` (36 chars, 4 dashes).
pub fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let nanos = now.as_nanos();
    let pid = u64::from(std::process::id());
    let rng = fastrand::u64(..);
    let time_hi = (nanos >> 32) as u32;
    let time_mid = (nanos >> 16) as u32 & 0xFFFF;
    let time_lo = nanos as u32 & 0xFFFF;
    let version = (u64::from(pid as u32 & 0xFFF)) as u32 | 0x4000;
    let variant = (((pid >> 12) ^ rng) as u32 & 0x3FFF) | 0x8000;
    let node = (rng >> 32) as u32;
    format!("{time_hi:08x}-{time_mid:04x}-{time_lo:04x}-{version:04x}-{variant:04x}{node:08x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_hex_correct_length() {
        let h = sha256_hex(b"hello");
        assert_eq!(h.len(), 64);
    }

    #[test]
    fn sha256_hex_deterministic() {
        assert_eq!(sha256_hex(b"hello"), sha256_hex(b"hello"));
    }

    #[test]
    fn sha256_hex_differs_for_different_inputs() {
        assert_ne!(sha256_hex(b"hello"), sha256_hex(b"world"));
    }

    #[test]
    fn blake3_hash_length() {
        let h = blake3_hash(b"test");
        assert_eq!(h.len(), 32);
    }

    #[test]
    fn blake3_hex_length() {
        let h = blake3_hex(b"test");
        assert_eq!(h.len(), 64);
    }

    #[test]
    fn fnv1a64_basic() {
        let h = fnv1a64(b"hello");
        assert_ne!(h, 0);
        assert_eq!(fnv1a64(b"hello"), fnv1a64(b"hello"));
    }

    #[test]
    fn digest_length_values() {
        assert_eq!(HashAlgorithm::Sha256.digest_length(), 32);
        assert_eq!(HashAlgorithm::Sha512.digest_length(), 64);
        assert_eq!(HashAlgorithm::Blake3.digest_length(), 32);
    }

    #[test]
    fn hash_state_incremental_sha256() {
        let mut state = HashState::new(HashAlgorithm::Sha256);
        state.update(b"hello ");
        state.update(b"world");
        let hex = state.digest_hex();
        assert_eq!(hex.len(), 64);
    }

    #[test]
    fn content_hash_deterministic() {
        let h1 = content_hash_with_model("hello", "model");
        let h2 = content_hash_with_model("hello", "model");
        assert_eq!(h1, h2);
    }

    #[test]
    fn content_hash_model_sensitive() {
        let h1 = content_hash_with_model("hello", "model-a");
        let h2 = content_hash_with_model("hello", "model-b");
        assert_ne!(h1, h2);
    }

    #[test]
    fn hash_file_small() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, b"hello").unwrap();
        let hash = hash_file(&path, HashAlgorithm::Sha256).unwrap();
        assert_eq!(hash.len(), 64);
    }

    #[test]
    fn hash_file_nonexistent() {
        let result = hash_file(Path::new("/nonexistent/file.txt"), HashAlgorithm::Sha256);
        assert!(result.is_err());
    }

    #[test]
    fn hash_batch_all_succeed() {
        let dir = tempfile::TempDir::new().unwrap();
        let p1 = dir.path().join("a.txt");
        let p2 = dir.path().join("b.txt");
        std::fs::write(&p1, b"data1").unwrap();
        std::fs::write(&p2, b"data2").unwrap();
        let results = hash_batch(&[p1, p2], HashAlgorithm::Sha256);
        assert_eq!(results.len(), 2);
        assert!(results[0].hash.is_some());
        assert!(results[1].hash.is_some());
    }

    #[test]
    fn hash_batch_some_fail() {
        let dir = tempfile::TempDir::new().unwrap();
        let p1 = dir.path().join("exists.txt");
        let p2 = dir.path().join("missing.txt");
        std::fs::write(&p1, b"data").unwrap();
        let results = hash_batch(&[p1, p2], HashAlgorithm::Sha256);
        assert_eq!(results.len(), 2);
        assert!(results[0].hash.is_some());
        assert!(results[1].hash.is_none());
    }

    #[test]
    fn hash_state_incremental_sha512() {
        let mut state = HashState::new(HashAlgorithm::Sha512);
        state.update(b"test data");
        let hex = state.digest_hex();
        assert_eq!(hex.len(), 128);
    }

    #[test]
    fn hash_state_incremental_blake3() {
        let mut state = HashState::new(HashAlgorithm::Blake3);
        state.update(b"test data");
        let hex = state.digest_hex();
        assert_eq!(hex.len(), 64);
    }
}

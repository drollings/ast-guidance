use common_core::hash::*;
use std::path::Path;


#[test]
fn sha256_hex_properties() {
        let h = sha256_hex(b"hello");
        assert_eq!(h.len(), 64);
        assert_eq!(h, sha256_hex(b"hello"));
        assert_ne!(h, sha256_hex(b"world"));
}

#[test]
fn blake3_lengths() {
        assert_eq!(blake3_hash(b"test").len(), 32);
        assert_eq!(blake3_hex(b"test").len(), 64);
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
fn hash_state_incremental_various() {
        for (algo, expected_len) in [(HashAlgorithm::Sha256, 64), (HashAlgorithm::Sha512, 128), (HashAlgorithm::Blake3, 64)] {
            let mut state = HashState::new(algo);
            state.update(b"hello ");
            state.update(b"world");
            assert_eq!(state.digest_hex().len(), expected_len);
        }
}

#[test]
fn content_hash_properties() {
        assert_eq!(content_hash_with_model("hello", "model"), content_hash_with_model("hello", "model"));
        assert_ne!(content_hash_with_model("hello", "model-a"), content_hash_with_model("hello", "model-b"));
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
fn hash_batch_mixed_results() {
        let dir = tempfile::TempDir::new().unwrap();
        let p1 = dir.path().join("a.txt");
        let p2 = dir.path().join("b.txt");
        std::fs::write(&p1, b"data1").unwrap();
        std::fs::write(&p2, b"data2").unwrap();
        let results = hash_batch(&[p1.clone(), p2.clone()], HashAlgorithm::Sha256);
        assert!(results.iter().all(|r| r.hash.is_some()));
        let p_missing = dir.path().join("missing.txt");
        let results = hash_batch(&[p1, p_missing], HashAlgorithm::Sha256);
        assert!(results[0].hash.is_some());
        assert!(results[1].hash.is_none());
}



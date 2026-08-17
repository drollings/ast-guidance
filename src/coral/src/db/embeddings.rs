use fluent_db::vector::{try_bytes_to_vec, vec_to_bytes};
use rusqlite::params;

use super::LibraryError;

impl super::Library {
    pub fn cache_embedding(
        &self,
        query_hash: &str,
        query_text: &str,
        embedding: &[f32],
    ) -> Result<(), LibraryError> {
        let blob = vec_to_bytes(embedding);
        self.store.execute(
            "INSERT OR REPLACE INTO embedding_cache (query_hash, query_text, embedding) VALUES (?1, ?2, ?3)",
            params![query_hash, query_text, blob],
        )?;
        Ok(())
    }

    pub fn get_cached_embedding(&self, query_hash: &str) -> Result<Option<Vec<f32>>, LibraryError> {
        let result = self.store.query_row(
            "SELECT embedding FROM embedding_cache WHERE query_hash = ?1",
            params![query_hash],
            |row| {
                let blob: Vec<u8> = row.get(0)?;
                Ok(try_bytes_to_vec(&blob))
            },
        )?;
        Ok(result.flatten())
    }
}

#[cfg(test)]
mod tests {
    fn lib() -> crate::db::Library {
        crate::db::Library::open_in_memory().expect("in-memory db")
    }

    #[test]
    fn cache_embedding_round_trip() {
        let lib = lib();
        lib.cache_embedding("hash1", "what is 2+2", &[0.1, 0.2, 0.3]).expect("cache");
        let back = lib.get_cached_embedding("hash1").expect("get").expect("exists");
        assert_eq!(back, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn missing_hash_returns_none() {
        let lib = lib();
        assert!(lib.get_cached_embedding("nope").expect("get").is_none());
    }

    #[test]
    fn cache_embedding_replaces_existing_row() {
        let lib = lib();
        lib.cache_embedding("k", "q", &[0.0]).expect("first");
        lib.cache_embedding("k", "q2", &[9.0]).expect("replace");
        let back = lib.get_cached_embedding("k").expect("get").expect("exists");
        assert_eq!(back, vec![9.0], "INSERT OR REPLACE overwrites by query_hash");
    }

    #[test]
    fn cache_is_scoped_per_hash() {
        let lib = lib();
        lib.cache_embedding("a", "q", &[1.0]).expect("cache a");
        lib.cache_embedding("b", "q", &[2.0]).expect("cache b");
        assert_eq!(lib.get_cached_embedding("a").expect("a").expect("a"), vec![1.0]);
        assert_eq!(lib.get_cached_embedding("b").expect("b").expect("b"), vec![2.0]);
    }
}

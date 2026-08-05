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

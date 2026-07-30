use rusqlite::params;
use search_vector::math::{try_bytes_to_vec, vec_to_bytes};

use super::LibraryError;

impl super::Library {
    pub fn cache_embedding(
        &self,
        query_hash: &str,
        query_text: &str,
        embedding: &[f32],
    ) -> Result<(), LibraryError> {
        let conn = self.conn.lock().unwrap();
        let blob = vec_to_bytes(embedding);
        conn.execute(
            "INSERT OR REPLACE INTO embedding_cache (query_hash, query_text, embedding) VALUES (?1, ?2, ?3)",
            params![query_hash, query_text, blob],
        )?;
        Ok(())
    }

    pub fn get_cached_embedding(&self, query_hash: &str) -> Result<Option<Vec<f32>>, LibraryError> {
        let conn = self.conn.lock().unwrap();
        let result = conn
            .query_row(
                "SELECT embedding FROM embedding_cache WHERE query_hash = ?1",
                params![query_hash],
                |row| {
                    let blob: Vec<u8> = row.get(0)?;
                    Ok(try_bytes_to_vec(&blob))
                },
            )
            .ok()
            .flatten();
        Ok(result)
    }
}

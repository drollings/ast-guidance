pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let mag = na.sqrt() * nb.sqrt();
    if mag == 0.0 {
        0.0
    } else {
        dot / mag
    }
}

/// Brute-force KNN search: compute cosine distance from `query` to every
/// candidate, sort ascending, and return the top `k` as `(id, distance)`.
///
/// `candidates` yields `(id, embedding_slice)` pairs borrowing the caller's
/// data — no full candidate-list clone. Candidates whose embedding length
/// differs from `query` are silently skipped.
pub fn knn_brute_force<'a, Id: Clone>(
    query: &[f32],
    candidates: impl Iterator<Item = (Id, &'a [f32])>,
    k: usize,
) -> Vec<(Id, f32)> {
    if query.is_empty() || k == 0 {
        return Vec::new();
    }
    let mut results: Vec<(Id, f32)> = candidates
        .filter_map(|(id, emb)| {
            if emb.len() != query.len() {
                return None;
            }
            let distance = 1.0 - cosine_similarity(query, emb);
            Some((id, distance))
        })
        .collect();
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

pub fn vec_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

pub fn bytes_to_vec(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub fn try_bytes_to_vec(b: &[u8]) -> Option<Vec<f32>> {
    if !b.len().is_multiple_of(4) {
        return None;
    }
    Some(bytes_to_vec(b))
}

#[derive(Debug, Clone)]
pub struct QuantizedEmbedding {
    pub values: Vec<i8>,
    pub scale: f32,
    pub dimensions: usize,
}

impl QuantizedEmbedding {
    pub fn from_f32(vec: &[f32]) -> Self {
        if vec.is_empty() {
            return Self {
                values: Vec::new(),
                scale: 1.0,
                dimensions: 0,
            };
        }
        let max_abs = vec.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        let scale = if max_abs > 0.0 { 127.0 / max_abs } else { 1.0 };
        let values: Vec<i8> = vec
            .iter()
            .map(|v| (v * scale).round().clamp(-128.0, 127.0) as i8)
            .collect();
        Self {
            dimensions: vec.len(),
            values,
            scale,
        }
    }
    pub fn to_f32(&self) -> Vec<f32> {
        if self.scale == 0.0 {
            return vec![0.0; self.dimensions];
        }
        let inv = 1.0 / self.scale;
        self.values.iter().map(|&v| f32::from(v) * inv).collect()
    }
}

pub fn cosine_similarity_q8(a: &QuantizedEmbedding, b: &QuantizedEmbedding) -> f32 {
    if a.dimensions != b.dimensions || a.dimensions == 0 {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0i64, 0i64, 0i64);
    for (x, y) in a.values.iter().zip(b.values.iter()) {
        let (xi, yi) = (i64::from(*x), i64::from(*y));
        dot += xi * yi;
        na += xi * xi;
        nb += yi * yi;
    }
    let mag = ((na as f64) * (nb as f64)).sqrt();
    if mag == 0.0 {
        0.0
    } else {
        (dot as f64 / mag) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let sim = cosine_similarity(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0]);
        assert!((sim - 1.0).abs() < 1e-6);
    }
    #[test]
    fn test_cosine_similarity_orthogonal() {
        assert!((cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]) - 0.0).abs() < 1e-6);
    }
    #[test]
    #[allow(clippy::float_cmp)]
    fn test_cosine_similarity_empty() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }
    #[test]
    fn test_vec_bytes_round_trip() {
        let v = vec![1.5, -2.5, 3.0, 0.0, -0.5];
        let restored = bytes_to_vec(&vec_to_bytes(&v));
        for (a, b) in v.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
    #[test]
    fn test_try_bytes_to_vec_valid() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let bytes = vec_to_bytes(&v);
        let restored = try_bytes_to_vec(&bytes).unwrap();
        assert_eq!(restored.len(), 4);
    }
    #[test]
    fn test_try_bytes_to_vec_invalid_length() {
        assert!(try_bytes_to_vec(&[0u8; 3]).is_none());
    }
    #[test]
    fn test_quantize_round_trip() {
        let original = vec![0.5, -0.3, 0.8, -0.1, 0.0, 1.0, -1.0];
        let q = QuantizedEmbedding::from_f32(&original);
        let restored = q.to_f32();
        for (a, b) in original.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 0.02);
        }
    }
    #[test]
    fn test_q8_cosine_similarity() {
        let a = QuantizedEmbedding::from_f32(&[1.0, 0.0, 0.0]);
        let b = QuantizedEmbedding::from_f32(&[1.0, 0.0, 0.0]);
        assert!((cosine_similarity_q8(&a, &b) - 1.0).abs() < 0.02);
    }
    #[test]
    #[allow(clippy::float_cmp)]
    fn test_q8_cosine_empty() {
        assert_eq!(
            cosine_similarity_q8(
                &QuantizedEmbedding::from_f32(&[]),
                &QuantizedEmbedding::from_f32(&[])
            ),
            0.0
        );
    }

    #[test]
    fn knn_brute_force_returns_top_k() {
        let query = vec![1.0, 0.0, 0.0];
        let candidates = vec![
            (1u32, vec![1.0, 0.0, 0.0]),
            (2, vec![0.0, 1.0, 0.0]),
            (3, vec![0.9, 0.1, 0.0]),
        ];
        let results = knn_brute_force(
            &query,
            candidates.iter().map(|(id, e)| (*id, e.as_slice())),
            2,
        );
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1);
        assert!((results[0].1 - 0.0).abs() < 1e-5);
        assert_eq!(results[1].0, 3);
    }

    #[test]
    fn knn_brute_force_skips_mismatched_dimensions() {
        let query = vec![1.0, 0.0];
        let candidates = vec![(1u32, vec![1.0, 0.0]), (2, vec![1.0, 0.0, 0.0])];
        let results = knn_brute_force(
            &query,
            candidates.iter().map(|(id, e)| (*id, e.as_slice())),
            10,
        );
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn knn_brute_force_empty_query_returns_empty() {
        let candidates = vec![(1u32, vec![1.0])];
        let results = knn_brute_force(&[], candidates.iter().map(|(id, e)| (*id, e.as_slice())), 5);
        assert!(results.is_empty());
    }

    #[test]
    fn knn_brute_force_zero_k_returns_empty() {
        let query = vec![1.0, 0.0];
        let candidates = vec![(1u32, vec![1.0, 0.0])];
        let results = knn_brute_force(
            &query,
            candidates.iter().map(|(id, e)| (*id, e.as_slice())),
            0,
        );
        assert!(results.is_empty());
    }

    #[test]
    fn knn_brute_force_empty_candidates() {
        let query = vec![1.0, 0.0];
        let results: Vec<(u32, f32)> = knn_brute_force(&query, std::iter::empty(), 5);
        assert!(results.is_empty());
    }
}

use common_core::vector_math::cosine_similarity_f32;

#[test]
fn equal_vectors_score_one() {
    assert!((cosine_similarity_f32(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
    assert!((cosine_similarity_f32(&[1.0, 2.0], &[2.0, 4.0]) - 1.0).abs() < 1e-6);
}

#[test]
fn orthogonal_and_opposite() {
    assert_eq!(cosine_similarity_f32(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
    assert!((cosine_similarity_f32(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-6);
}

#[test]
fn degenerate_inputs_yield_zero() {
    assert_eq!(cosine_similarity_f32(&[0.0, 0.0], &[0.0, 0.0]), 0.0);
    assert_eq!(cosine_similarity_f32(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    assert_eq!(cosine_similarity_f32(&[], &[]), 0.0);
    assert_eq!(cosine_similarity_f32(&[], &[1.0]), 0.0);
    assert_eq!(cosine_similarity_f32(&[1.0], &[1.0, 2.0]), 0.0);
}

#[test]
fn nan_propagates() {
    assert!(cosine_similarity_f32(&[f32::NAN], &[1.0]).is_nan());
}

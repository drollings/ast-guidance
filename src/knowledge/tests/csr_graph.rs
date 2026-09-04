use super::*;

#[test]
fn serialized_csr_size() {
    assert_eq!(std::mem::size_of::<SerializedCsr>(), SERIALIZED_CSR_SIZE);
}

#[test]
fn empty_graph_roundtrip() {
    let g = CsrGraph::new(0, vec![0], vec![], None);
    assert_eq!(g.node_count, 0);
    assert_eq!(g.edge_count, 0);
}

#[test]
fn manual_construction_and_accessors() {
    let g = CsrGraph::new(
        3,
        vec![0, 2, 3, 3],
        vec![1, 2, 0],
        Some(vec![0.5, 1.0, 0.3]),
    );
    assert_eq!(g.degree(0), 2);
    assert_eq!(g.degree(1), 1);
    assert_eq!(g.degree(2), 0);
    assert_eq!(g.neighbors(0), &[1, 2]);
}

#[test]
fn serialize_deserialize_roundtrip() {
    let g = CsrGraph::new(
        3,
        vec![0, 2, 3, 3],
        vec![1, 2, 0],
        Some(vec![0.5, 1.0, 0.3]),
    );
    let blob = g.serialize();
    let loaded = CsrGraph::deserialize(&blob).unwrap();
    assert_eq!(loaded.node_count, 3);
    assert_eq!(loaded.neighbors(0), &[1, 2]);
    let weights = loaded.weights.unwrap();
    assert!((weights[0] - 0.5).abs() < 0.001);
    assert!((weights[1] - 1.0).abs() < 0.001);
}

#[test]
fn deserialize_bad_magic() {
    let mut blob = vec![0xDE, 0xAD, 0xBE, 0xEF];
    blob.extend_from_slice(&[0; SERIALIZED_CSR_SIZE - 4]);
    assert!(matches!(
        CsrGraph::deserialize(&blob),
        Err(CsrError::InvalidMagic)
    ));
}

#[test]
fn out_of_range_returns_empty() {
    let g = CsrGraph::new(2, vec![0, 1, 2], vec![1], None);
    assert!(g.neighbors(99).is_empty());
    assert_eq!(g.degree(99), 0);
}

#[test]
fn deserialize_blob_too_short() {
    assert!(matches!(
        CsrGraph::deserialize(&[0u8; 4]),
        Err(CsrError::BlobTooShort)
    ));
}

#[test]
fn deserialize_unsupported_version() {
    let mut blob = vec![0u8; SERIALIZED_CSR_SIZE];
    blob[0..4].copy_from_slice(&CSR_MAGIC.to_le_bytes());
    blob[4..8].copy_from_slice(&99u32.to_le_bytes()); // wrong version
    assert!(matches!(
        CsrGraph::deserialize(&blob),
        Err(CsrError::UnsupportedVersion)
    ));
}

#[test]
fn deserialize_without_weights() {
    let g = CsrGraph::new(2, vec![0, 1, 2], vec![1], None);
    let blob = g.serialize();
    let loaded = CsrGraph::deserialize(&blob).unwrap();
    assert!(loaded.weights.is_none());
}

#[test]
fn serialize_deserialize_no_weights_roundtrip() {
    let g = CsrGraph::new(2, vec![0, 1, 2], vec![1], None);
    let blob = g.serialize();
    let loaded = CsrGraph::deserialize(&blob).unwrap();
    assert_eq!(loaded.node_count, 2);
    assert_eq!(loaded.edge_count, 1);
}

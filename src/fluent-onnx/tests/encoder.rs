use super::mean_pool;

#[test]
fn mean_pool_averages_over_non_pad_tokens() {
    let seq = 3;
    let dims = 2;
    let hidden = [
        1.0, 2.0, // token 0
        3.0, 4.0, // token 1
        99.0, 99.0, // token 2 (masked)
    ];
    let mask = [1i64, 1, 0];
    let pooled = mean_pool(&hidden, &mask, seq, dims);
    assert_eq!(pooled, vec![2.0, 3.0]);
}

#[test]
fn mean_pool_ignores_leading_pad() {
    let seq = 4;
    let dims = 3;
    let hidden = [
        99.0, 99.0, 99.0, // pad
        1.0, 2.0, 3.0, // real
        4.0, 5.0, 6.0, // real
        99.0, 99.0, 99.0, // pad
    ];
    let mask = [0i64, 1, 1, 0];
    let pooled = mean_pool(&hidden, &mask, seq, dims);
    assert_eq!(pooled, vec![2.5, 3.5, 4.5]);
}

#[test]
fn mean_pool_fully_masked_yields_zeros() {
    let seq = 2;
    let dims = 4;
    let hidden = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mask = [0i64, 0];
    let pooled = mean_pool(&hidden, &mask, seq, dims);
    assert_eq!(pooled, vec![0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn mean_pool_empty_seq_yields_zeros() {
    let pooled = mean_pool(&[], &[], 0, 3);
    assert_eq!(pooled, vec![0.0, 0.0, 0.0]);
}

#[test]
fn mean_pool_single_token_is_the_token_itself() {
    let seq = 1;
    let dims = 2;
    let hidden = [7.0, -3.0];
    let mask = [1i64];
    assert_eq!(mean_pool(&hidden, &mask, seq, dims), vec![7.0, -3.0]);
}

use super::*;

thread_local_resource!(static COUNTER: usize);

#[test]
fn test_with_tlr() {
    let r1 = with_tlr(&COUNTER, |c| {
        *c += 1;
        *c
    });
    assert_eq!(r1, 1);

    let r2 = with_tlr(&COUNTER, |c| {
        *c += 1;
        *c
    });
    assert_eq!(r2, 2);
}

use super::*;
use fluent_wvr_testutil::tempdir;

#[test]
fn pair_weight_default() {
    let w = pair_weight(b't', b'h');
    assert_eq!(w, 0x0800);
}

#[test]
fn build_frequency_table_basic() {
    let _table = build_frequency_table("hello world");
    let w = pair_weight(b'h', b'e');
    assert!(w > 0);
}

#[test]
fn build_frequency_table_from_map_works() {
    let table = build_frequency_table_from_map("test data");
    let _ = table;
}

#[test]
fn freq_table_roundtrip() {
    let dir = tempdir();
    let path = dir.path().join("freq.bin");
    let table = build_frequency_table("hello world");
    write_frequency_table(&path, &table).unwrap();
    let read = read_frequency_table(&path).unwrap().unwrap();
    assert_eq!(table, read);
}

#[test]
fn freq_table_missing_file() {
    let dir = tempdir();
    let path = dir.path().join("nonexistent.bin");
    let result = read_frequency_table(&path).unwrap();
    assert!(result.is_none());
}

#[test]
fn freq_table_wrong_magic() {
    let dir = tempdir();
    let path = dir.path().join("bad.bin");
    fs::write(&path, b"BADMAGIC").unwrap();
    let result = read_frequency_table(&path).unwrap();
    assert!(result.is_none());
}

#[test]
fn set_and_get_frequency_table() {
    let table = build_frequency_table("custom data");
    set_frequency_table(Box::new(table));
    let w = pair_weight(b'c', b'u');
    assert!(w > 0);
}

#[test]
fn get_frequency_table_returns_default() {
    let table = get_frequency_table();
    assert_eq!(table[b't' as usize][b'h' as usize], 0x0800);
}

#[test]
fn freq_table_read_truncated() {
    let dir = tempdir();
    let path = dir.path().join("truncated.bin");
    let magic: [u8; 4] = 0x4652_4551u32.to_le_bytes();
    let version: [u8; 4] = 1u32.to_le_bytes();
    let mut buf = Vec::new();
    buf.extend_from_slice(&magic);
    buf.extend_from_slice(&version);
    buf.extend_from_slice(&[0u8; 10]); // truncated data
    fs::write(&path, &buf).unwrap();
    let result = read_frequency_table(&path).unwrap();
    assert!(result.is_none());
}

#[test]
fn build_frequency_table_from_map_delegates() {
    let table = build_frequency_table_from_map("abcabc");
    assert!(table[b'a' as usize][b'b' as usize] > table[b'x' as usize][b'y' as usize]);
}

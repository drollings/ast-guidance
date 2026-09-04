use super::*;

#[test]
fn write_read_roundtrip_no_git_head() {
    let h = Header {
        magic: 0x574F5244,
        version: 1,
        git_head: None,
    };
    let mut buf = Vec::new();
    h.write_to(&mut buf);
    buf.extend_from_slice(b"payload");
    let result = Header::read(&buf, 0x574F5244, 1).unwrap();
    assert_eq!(result.offset, 10);
    assert_eq!(result.git_head_len, 0);
    assert_eq!(&buf[result.offset..], b"payload");
}

#[test]
fn write_read_roundtrip_with_git_head() {
    let h = Header {
        magic: 0x574F5244,
        version: 1,
        git_head: Some("abc123".into()),
    };
    let mut buf = Vec::new();
    h.write_to(&mut buf);
    let result = Header::read(&buf, 0x574F5244, 1).unwrap();
    assert!(result.offset > 10);
}

#[test]
fn wrong_magic_returns_none() {
    let buf = [0u8; 10];
    assert!(Header::read(&buf, 0xDEADBEEF, 1).is_none());
}

#[test]
fn wrong_version_returns_none() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&0x574F5244u32.to_le_bytes());
    buf.extend_from_slice(&99u32.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes());
    assert!(Header::read(&buf, 0x574F5244, 1).is_none());
}

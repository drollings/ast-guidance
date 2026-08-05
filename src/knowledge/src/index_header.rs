pub const INDEX_HEADER_SIZE: usize = 10;

#[derive(Debug, Clone)]
pub struct Header {
    pub magic: u32,
    pub version: u32,
    pub git_head: Option<String>,
}

#[derive(Debug)]
pub struct ReadResult {
    pub offset: usize,
    pub git_head_len: u16,
}

impl Header {
    pub fn write_to(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.magic.to_le_bytes());
        buf.extend_from_slice(&self.version.to_le_bytes());
        let git_head_bytes = self.git_head.as_deref().unwrap_or("").as_bytes();
        let git_head_len = git_head_bytes.len() as u16;
        buf.extend_from_slice(&git_head_len.to_le_bytes());
        buf.extend_from_slice(git_head_bytes);
    }

    pub fn read(content: &[u8], expected_magic: u32, expected_version: u32) -> Option<ReadResult> {
        if content.len() < INDEX_HEADER_SIZE {
            return None;
        }
        let magic = u32::from_le_bytes(content[0..4].try_into().ok()?);
        if magic != expected_magic {
            return None;
        }
        let version = u32::from_le_bytes(content[4..8].try_into().ok()?);
        if version != expected_version {
            return None;
        }
        let git_head_len = u16::from_le_bytes(content[8..10].try_into().ok()?);
        Some(ReadResult {
            offset: 10 + git_head_len as usize,
            git_head_len,
        })
    }
}

#[cfg(test)]
mod tests {
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
}

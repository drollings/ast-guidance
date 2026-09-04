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
#[path = "../tests/index_header.rs"]
mod tests;

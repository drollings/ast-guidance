use crate::index_header::Header;
use std::collections::HashMap;

pub type Trigram = u32;

#[derive(Debug, Clone)]
pub struct TrigramHit {
    pub doc_id: u32,
    pub position: u32,
}

pub const MAX_POSTINGS: u16 = 512;
pub const TRIGRAM_INDEX_MAGIC: u32 = 0x5452_4947;
pub const TRIGRAM_INDEX_VERSION: u32 = 1;

fn make_trigram(a: u8, b: u8, c: u8) -> Trigram {
    u32::from(a) | (u32::from(b) << 8) | (u32::from(c) << 16)
}

#[derive(Debug, Clone)]
pub struct TrigramIndex {
    index: HashMap<Trigram, Vec<TrigramHit>>,
    doc_count: u32,
}

impl TrigramIndex {
    pub fn new() -> Self {
        Self {
            index: HashMap::new(),
            doc_count: 0,
        }
    }

    pub fn build_from_content(&mut self, path: &str, content: &str) {
        let bytes = content.as_bytes();
        if bytes.len() < 3 {
            return;
        }
        let doc_id = self.doc_count;
        self.doc_count += 1;
        for i in 0..=bytes.len() - 3 {
            let tri = make_trigram(bytes[i], bytes[i + 1], bytes[i + 2]);
            let hit = TrigramHit {
                doc_id,
                position: i as u32,
            };
            self.index.entry(tri).or_default().push(hit);
        }
        let _ = path;
    }

    pub fn search_bytes(&self, tri_bytes: [u8; 3]) -> &[TrigramHit] {
        let tri = make_trigram(tri_bytes[0], tri_bytes[1], tri_bytes[2]);
        self.index
            .get(&tri)
            .map_or(&[] as &[TrigramHit], |v| v.as_slice())
    }

    pub fn search_trigram(&self, tri: Trigram) -> &[TrigramHit] {
        self.index
            .get(&tri)
            .map_or(&[] as &[TrigramHit], |v| v.as_slice())
    }

    pub fn candidates(&self, query: &str) -> Vec<u32> {
        let bytes = query.as_bytes();
        if bytes.len() < 3 {
            return Vec::new();
        }
        let first = make_trigram(bytes[0], bytes[1], bytes[2]);
        let hits = self
            .index
            .get(&first)
            .map_or(&[] as &[TrigramHit], |v| v.as_slice());
        let mut doc_ids: Vec<u32> = hits.iter().map(|h| h.doc_id).collect();
        doc_ids.sort_unstable();
        doc_ids.dedup();
        doc_ids
    }

    pub fn search(&self, query: &str) -> Vec<u32> {
        self.candidates(query)
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&self.doc_count.to_le_bytes());

        let entries: Vec<(&Trigram, &Vec<TrigramHit>)> = self.index.iter().collect();
        let entry_count = entries.len() as u32;
        payload.extend_from_slice(&entry_count.to_le_bytes());

        for (key, hits) in &entries {
            payload.extend_from_slice(&key.to_le_bytes());
            let hit_count = hits.len() as u32;
            payload.extend_from_slice(&hit_count.to_le_bytes());
            for hit in *hits {
                payload.extend_from_slice(&hit.doc_id.to_le_bytes());
                payload.extend_from_slice(&hit.position.to_le_bytes());
            }
        }

        let header = Header {
            magic: TRIGRAM_INDEX_MAGIC,
            version: TRIGRAM_INDEX_VERSION,
            git_head: None,
        };
        let mut buf = Vec::new();
        header.write_to(&mut buf);
        buf.extend_from_slice(&payload);
        buf
    }

    pub fn deserialize(data: &[u8]) -> Result<Self, &'static str> {
        let read_result = Header::read(data, TRIGRAM_INDEX_MAGIC, TRIGRAM_INDEX_VERSION)
            .ok_or("invalid header")?;
        let payload_start = read_result.offset;
        if data.len() < payload_start + 8 {
            return Err("truncated data");
        }

        let mut offset = payload_start;
        let doc_count = u32::from_le_bytes(
            data[offset..offset + 4]
                .try_into()
                .map_err(|_| "truncated")?,
        );
        offset += 4;
        let entry_count = u32::from_le_bytes(
            data[offset..offset + 4]
                .try_into()
                .map_err(|_| "truncated")?,
        );
        offset += 4;

        let mut index = HashMap::new();
        for _ in 0..entry_count {
            if offset + 8 > data.len() {
                return Err("truncated");
            }
            let key = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
            offset += 4;
            let hit_count = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
            offset += 4;
            let mut hits = Vec::with_capacity(hit_count as usize);
            for _ in 0..hit_count {
                if offset + 8 > data.len() {
                    return Err("truncated");
                }
                let doc_id = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
                offset += 4;
                let position = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
                offset += 4;
                hits.push(TrigramHit { doc_id, position });
            }
            index.insert(key, hits);
        }

        Ok(Self { index, doc_count })
    }
}

impl Default for TrigramIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "../tests/trigram_index.rs"]
mod tests;

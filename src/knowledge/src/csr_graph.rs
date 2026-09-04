pub const CSR_MAGIC: u32 = 0x4752_5343;
pub const CSR_VERSION: u32 = 1;

#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct SerializedCsr {
    pub magic: u32,
    pub version: u32,
    pub node_count: u32,
    pub edge_count: u32,
    pub has_weights: u8,
    _pad: [u8; 3],
}

pub const SERIALIZED_CSR_SIZE: usize = 20;

#[derive(Debug, Clone)]
pub struct CsrGraph {
    pub node_count: u32,
    pub edge_count: u32,
    pub offsets: Vec<usize>,
    pub targets: Vec<u32>,
    pub weights: Option<Vec<f32>>,
}

impl CsrGraph {
    pub fn new(
        node_count: u32,
        offsets: Vec<usize>,
        targets: Vec<u32>,
        weights: Option<Vec<f32>>,
    ) -> Self {
        let edge_count = targets.len() as u32;
        Self {
            node_count,
            edge_count,
            offsets,
            targets,
            weights,
        }
    }

    pub fn neighbors(&self, node_idx: u32) -> &[u32] {
        let i = node_idx as usize;
        if i >= self.offsets.len().saturating_sub(1) {
            return &[];
        }
        let start = self.offsets[i];
        let end = self.offsets[i + 1];
        if end > self.targets.len() || start > end {
            return &[];
        }
        &self.targets[start..end]
    }

    pub fn degree(&self, node_idx: u32) -> u32 {
        let i = node_idx as usize;
        if i >= self.offsets.len().saturating_sub(1) {
            return 0;
        }
        (self.offsets[i + 1] - self.offsets[i]) as u32
    }

    pub fn serialize(&self) -> Vec<u8> {
        let has_weights = u8::from(self.weights.is_some());
        let mut buf = Vec::with_capacity(
            SERIALIZED_CSR_SIZE + self.offsets.len() * 8 + self.targets.len() * 4,
        );
        buf.extend_from_slice(&CSR_MAGIC.to_le_bytes());
        buf.extend_from_slice(&CSR_VERSION.to_le_bytes());
        buf.extend_from_slice(&self.node_count.to_le_bytes());
        buf.extend_from_slice(&self.edge_count.to_le_bytes());
        buf.extend_from_slice(&[has_weights, 0, 0, 0]);
        for &off in &self.offsets {
            buf.extend_from_slice(&(off as u64).to_le_bytes());
        }
        for &t in &self.targets {
            buf.extend_from_slice(&t.to_le_bytes());
        }
        if let Some(ref w) = self.weights {
            for &val in w {
                buf.extend_from_slice(&val.to_le_bytes());
            }
        }
        buf
    }

    pub fn deserialize(data: &[u8]) -> Result<Self, CsrError> {
        if data.len() < SERIALIZED_CSR_SIZE {
            return Err(CsrError::BlobTooShort);
        }
        let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
        if magic != CSR_MAGIC {
            return Err(CsrError::InvalidMagic);
        }
        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if version != CSR_VERSION {
            return Err(CsrError::UnsupportedVersion);
        }
        let node_count = u32::from_le_bytes(data[8..12].try_into().unwrap());
        let edge_count = u32::from_le_bytes(data[12..16].try_into().unwrap());
        let has_weights = data[16];
        let mut offset = SERIALIZED_CSR_SIZE;

        let offsets_len = (node_count + 1) as usize;
        let mut offsets = Vec::with_capacity(offsets_len);
        for _ in 0..offsets_len {
            if offset + 8 > data.len() {
                return Err(CsrError::BlobTooShort);
            }
            let val = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
            offsets.push(val as usize);
            offset += 8;
        }

        let targets_len = edge_count as usize;
        let mut targets = Vec::with_capacity(targets_len);
        for _ in 0..targets_len {
            if offset + 4 > data.len() {
                return Err(CsrError::BlobTooShort);
            }
            let t = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
            targets.push(t);
            offset += 4;
        }

        let weights = if has_weights != 0 {
            let mut w = Vec::with_capacity(targets_len);
            for _ in 0..targets_len {
                if offset + 4 > data.len() {
                    return Err(CsrError::BlobTooShort);
                }
                let val = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
                w.push(val);
                offset += 4;
            }
            Some(w)
        } else {
            None
        };

        Ok(Self {
            node_count,
            edge_count,
            offsets,
            targets,
            weights,
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CsrError {
    #[error("invalid magic")]
    InvalidMagic,
    #[error("unsupported version")]
    UnsupportedVersion,
    #[error("blob too short")]
    BlobTooShort,
}

#[cfg(test)]
#[path = "../tests/csr_graph.rs"]
mod tests;

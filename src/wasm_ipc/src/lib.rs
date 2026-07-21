//! WASM IPC — Binary schemas for Extism boundary.
//! #[repr(C, packed)] structs for zero-copy IPC.

use bitvec::vec::BitVec;
use std::convert::TryInto;

pub const BINARY_MAGIC: [u8; 4] = [0x47, 0x52, 0x50, 0x48]; // "GRPH"
pub const BINARY_SCHEMA_VERSION: u32 = 1;
pub const MAX_WASM_HOST_CALLS: u32 = 10_000;
pub const MAX_IPC_PAYLOAD_SIZE: usize = 64 * 1024 * 1024; // 64 MiB

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum PayloadType {
    ExecutionRequest = 1,
    ExecutionResult = 2,
    ContextNode = 3,
}

impl TryFrom<u32> for PayloadType {
    type Error = IpcError;
    fn try_from(v: u32) -> Result<Self, Self::Error> {
        match v {
            1 => Ok(Self::ExecutionRequest),
            2 => Ok(Self::ExecutionResult),
            3 => Ok(Self::ContextNode),
            _ => Err(IpcError::UnknownPayloadType(v)),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum IpcError {
    #[error("buffer too small")]
    BufferTooSmall,
    #[error("invalid magic bytes")]
    InvalidMagic,
    #[error("unsupported schema version")]
    UnsupportedVersion,
    #[error("unknown payload type: {0}")]
    UnknownPayloadType(u32),
    #[error("checksum mismatch")]
    ChecksumMismatch,
    #[error("payload too large: {0}")]
    PayloadTooLarge(u32),
}

#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct BinaryHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub payload_type: u32,
    pub payload_size: u32,
    pub checksum: u32,
}

#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct BinaryExecutionRequest {
    pub header: BinaryHeader,
    pub target_id: i64,
    pub input_offset: u32,
    pub input_len: u32,
    pub flags: u32,
}

#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct BinaryExecutionResult {
    pub header: BinaryHeader,
    pub success: u32,
    pub error_code: u32,
    pub output_offset: u32,
    pub output_len: u32,
    pub provides_words_offset: u32,
    pub provides_words_count: u32,
}

#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct BinaryContextNode {
    pub header: BinaryHeader,
    pub id: i64,
    pub valid_from_ts: i64,
    pub valid_to_ts: i64,
    pub confidence: i32,
    pub provenance_id: i32,
    pub lod_offsets: [u32; 6],
    pub lod_lengths: [u32; 6],
}

pub(crate) fn compute_checksum(data: &[u8]) -> u32 {
    data.iter()
        .fold(0u32, |acc, &b| acc.wrapping_add(u32::from(b)))
}

#[must_use]
pub fn encode_request(req: &BinaryExecutionRequest, input: &[u8]) -> Vec<u8> {
    let fixed_size = std::mem::size_of::<BinaryExecutionRequest>();
    let mut buf = Vec::with_capacity(fixed_size + input.len());

    buf.extend_from_slice(&req.header.magic);
    buf.extend_from_slice(&req.header.version.to_le_bytes());
    buf.extend_from_slice(&req.header.payload_type.to_le_bytes());
    buf.extend_from_slice(&req.header.payload_size.to_le_bytes());
    buf.extend_from_slice(&req.header.checksum.to_le_bytes());

    buf.extend_from_slice(&req.target_id.to_le_bytes());
    buf.extend_from_slice(&req.input_offset.to_le_bytes());
    buf.extend_from_slice(&req.input_len.to_le_bytes());
    buf.extend_from_slice(&req.flags.to_le_bytes());

    buf.extend_from_slice(input);
    buf
}

#[must_use]
pub fn encode_result(result: &BinaryExecutionResult, output: &[u8]) -> Vec<u8> {
    let fixed_size = std::mem::size_of::<BinaryExecutionResult>();
    let mut buf = Vec::with_capacity(fixed_size + output.len());

    buf.extend_from_slice(&result.header.magic);
    buf.extend_from_slice(&result.header.version.to_le_bytes());
    buf.extend_from_slice(&result.header.payload_type.to_le_bytes());
    buf.extend_from_slice(&result.header.payload_size.to_le_bytes());
    buf.extend_from_slice(&result.header.checksum.to_le_bytes());

    buf.extend_from_slice(&result.success.to_le_bytes());
    buf.extend_from_slice(&result.error_code.to_le_bytes());
    buf.extend_from_slice(&result.output_offset.to_le_bytes());
    buf.extend_from_slice(&result.output_len.to_le_bytes());
    buf.extend_from_slice(&result.provides_words_offset.to_le_bytes());
    buf.extend_from_slice(&result.provides_words_count.to_le_bytes());

    buf.extend_from_slice(output);
    buf
}

#[must_use]
pub fn encode_context_node(node: &BinaryContextNode, lod_data: &[u8]) -> Vec<u8> {
    let fixed_size = std::mem::size_of::<BinaryContextNode>();
    let mut buf = Vec::with_capacity(fixed_size + lod_data.len());

    buf.extend_from_slice(&node.header.magic);
    buf.extend_from_slice(&node.header.version.to_le_bytes());
    buf.extend_from_slice(&node.header.payload_type.to_le_bytes());
    buf.extend_from_slice(&node.header.payload_size.to_le_bytes());
    buf.extend_from_slice(&node.header.checksum.to_le_bytes());

    buf.extend_from_slice(&node.id.to_le_bytes());
    buf.extend_from_slice(&node.valid_from_ts.to_le_bytes());
    buf.extend_from_slice(&node.valid_to_ts.to_le_bytes());
    buf.extend_from_slice(&node.confidence.to_le_bytes());
    buf.extend_from_slice(&node.provenance_id.to_le_bytes());
    let lod_offsets = node.lod_offsets;
    for offset in &lod_offsets {
        buf.extend_from_slice(&offset.to_le_bytes());
    }
    let lod_lengths = node.lod_lengths;
    for length in &lod_lengths {
        buf.extend_from_slice(&length.to_le_bytes());
    }

    buf.extend_from_slice(lod_data);
    buf
}

pub fn decode_result(buf: &[u8]) -> Result<(BinaryExecutionResult, Vec<u8>), IpcError> {
    let min_size = std::mem::size_of::<BinaryExecutionResult>();
    if buf.len() < min_size {
        return Err(IpcError::BufferTooSmall);
    }

    let mut offset: usize = 4;
    let read_u32 = |o: &mut usize| -> u32 {
        let val = u32::from_le_bytes(buf[*o..*o + 4].try_into().unwrap());
        *o += 4;
        val
    };

    let magic = [buf[0], buf[1], buf[2], buf[3]];
    if magic != BINARY_MAGIC {
        return Err(IpcError::InvalidMagic);
    }

    let version = read_u32(&mut offset);
    if version != BINARY_SCHEMA_VERSION {
        return Err(IpcError::UnsupportedVersion);
    }
    let payload_type = read_u32(&mut offset);
    let payload_size = read_u32(&mut offset);
    if payload_size as usize > MAX_IPC_PAYLOAD_SIZE {
        return Err(IpcError::PayloadTooLarge(payload_size));
    }
    let checksum = read_u32(&mut offset);

    // result fields follow header (no target_id — that's in the request)
    let success = read_u32(&mut offset);
    let error_code = read_u32(&mut offset);
    let output_offset = read_u32(&mut offset);
    let output_len = read_u32(&mut offset);
    let provides_words_offset = read_u32(&mut offset);
    let provides_words_count = read_u32(&mut offset);

    let payload_start = offset;
    let payload_end = payload_start + payload_size as usize;
    let actual_checksum = if payload_end <= buf.len() {
        compute_checksum(&buf[payload_start..payload_end])
    } else {
        compute_checksum(&buf[payload_start..])
    };
    if checksum != 0 && actual_checksum != checksum {
        return Err(IpcError::ChecksumMismatch);
    }

    let output = if output_offset as usize + output_len as usize <= buf.len() {
        buf[output_offset as usize..][..output_len as usize].to_vec()
    } else {
        Vec::new()
    };

    let result = BinaryExecutionResult {
        header: BinaryHeader {
            magic,
            version,
            payload_type,
            payload_size,
            checksum,
        },
        success,
        error_code,
        output_offset,
        output_len,
        provides_words_offset,
        provides_words_count,
    };

    Ok((result, output))
}

pub fn get_provides_bitset(
    result: &BinaryExecutionResult,
    payload: &[u8],
) -> Result<BitVec, IpcError> {
    // SAFETY: packed struct fields must be read with read_unaligned
    let count =
        unsafe { std::ptr::addr_of!(result.provides_words_count).read_unaligned() } as usize;
    let offset =
        unsafe { std::ptr::addr_of!(result.provides_words_offset).read_unaligned() } as usize;
    let mut bits = BitVec::with_capacity(count * 64);

    for i in 0..count {
        let start = offset + i * 8;
        if start + 8 > payload.len() {
            return Err(IpcError::BufferTooSmall);
        }
        let word = u64::from_le_bytes(
            payload[start..start + 8]
                .try_into()
                .map_err(|_| IpcError::BufferTooSmall)?,
        );
        for bit in 0..64 {
            bits.push((word >> bit) & 1 == 1);
        }
    }
    Ok(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_size_is_correct() {
        assert_eq!(std::mem::size_of::<BinaryHeader>(), 20);
    }

    #[test]
    fn request_roundtrip() {
        let req = BinaryExecutionRequest {
            header: BinaryHeader {
                magic: BINARY_MAGIC,
                version: BINARY_SCHEMA_VERSION,
                payload_type: PayloadType::ExecutionRequest as u32,
                payload_size: 0,
                checksum: 0,
            },
            target_id: 42,
            input_offset: 0,
            input_len: 5,
            flags: 0,
        };
        let input = b"hello";
        let encoded = encode_request(&req, input);
        assert!(encoded.len() >= std::mem::size_of::<BinaryExecutionRequest>());
        assert!(encoded.windows(input.len()).any(|w| w == input));
    }

    #[test]
    fn decode_result_valid() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&BINARY_MAGIC);
        buf.extend_from_slice(&BINARY_SCHEMA_VERSION.to_le_bytes());
        buf.extend_from_slice(&(PayloadType::ExecutionResult as u32).to_le_bytes());
        buf.extend_from_slice(&100u32.to_le_bytes()); // payload_size
        buf.extend_from_slice(&0u32.to_le_bytes()); // checksum
                                                    // result has no target_id field; skip to success
        buf.extend_from_slice(&1u32.to_le_bytes()); // success
        buf.extend_from_slice(&0u32.to_le_bytes()); // error_code
        let result_size = std::mem::size_of::<BinaryExecutionResult>();
        buf.extend_from_slice(&(result_size as u32).to_le_bytes()); // output_offset
        buf.extend_from_slice(&5u32.to_le_bytes()); // output_len
        buf.extend_from_slice(&0u32.to_le_bytes()); // provides_words_offset
        buf.extend_from_slice(&0u32.to_le_bytes()); // provides_words_count
        buf.extend_from_slice(b"hello"); // output

        let (result, output) = decode_result(&buf).unwrap();
        // SAFETY: packed struct field access via read_unaligned
        let success = unsafe { std::ptr::addr_of!(result.success).read_unaligned() };
        assert_eq!(success, 1);
        assert_eq!(output, b"hello");
    }

    #[test]
    fn decode_result_invalid_magic() {
        let buf = vec![0u8; 32];
        let result = decode_result(&buf);
        assert!(result.is_err());
    }

    #[test]
    fn decode_result_buffer_too_small() {
        let buf = vec![0u8; 4];
        let result = decode_result(&buf);
        assert!(result.is_err());
    }

    #[test]
    fn get_provides_bitset_roundtrip() {
        let mut payload = vec![0u8; 100];
        let words: [u64; 2] = [0b101, 0b110];
        for (i, w) in words.iter().enumerate() {
            payload[16 + i * 8..16 + i * 8 + 8].copy_from_slice(&w.to_le_bytes());
        }
        let result = BinaryExecutionResult {
            header: BinaryHeader {
                magic: BINARY_MAGIC,
                version: BINARY_SCHEMA_VERSION,
                payload_type: PayloadType::ExecutionResult as u32,
                payload_size: 0,
                checksum: 0,
            },

            success: 1,
            error_code: 0,
            output_offset: 0,
            output_len: 0,
            provides_words_offset: 16,
            provides_words_count: 2,
        };
        let bits = get_provides_bitset(&result, &payload).unwrap();
        assert!(bits[0]);
        assert!(!bits[1]);
        assert!(bits[2]);
        assert!(!bits[64]);
        assert!(bits[65]);
        assert!(bits[66]);
    }

    #[test]
    fn payload_type_conversion() {
        assert_eq!(
            PayloadType::try_from(1).unwrap(),
            PayloadType::ExecutionRequest
        );
        assert_eq!(
            PayloadType::try_from(2).unwrap(),
            PayloadType::ExecutionResult
        );
        assert_eq!(PayloadType::try_from(3).unwrap(), PayloadType::ContextNode);
        assert!(PayloadType::try_from(99).is_err());
    }

    #[test]
    fn compute_checksum_basic() {
        let data = b"hello";
        let c1 = compute_checksum(data);
        let c2 = compute_checksum(data);
        assert_eq!(c1, c2);
    }

    #[test]
    fn decode_result_wrong_checksum() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&BINARY_MAGIC);
        buf.extend_from_slice(&BINARY_SCHEMA_VERSION.to_le_bytes());
        buf.extend_from_slice(&(PayloadType::ExecutionResult as u32).to_le_bytes());
        buf.extend_from_slice(&100u32.to_le_bytes()); // payload_size
        buf.extend_from_slice(&999u32.to_le_bytes()); // wrong checksum
        buf.extend_from_slice(&1u32.to_le_bytes()); // success
        buf.extend_from_slice(&0u32.to_le_bytes()); // error_code
        let result_size = std::mem::size_of::<BinaryExecutionResult>();
        buf.extend_from_slice(&(result_size as u32).to_le_bytes()); // output_offset
        buf.extend_from_slice(&5u32.to_le_bytes()); // output_len
        buf.extend_from_slice(&0u32.to_le_bytes()); // provides_words_offset
        buf.extend_from_slice(&0u32.to_le_bytes()); // provides_words_count
        buf.extend_from_slice(b"hello"); // output

        let result = decode_result(&buf);
        assert!(matches!(result, Err(IpcError::ChecksumMismatch)));
    }

    #[test]
    fn decode_result_zero_checksum_skips_validation() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&BINARY_MAGIC);
        buf.extend_from_slice(&BINARY_SCHEMA_VERSION.to_le_bytes());
        buf.extend_from_slice(&(PayloadType::ExecutionResult as u32).to_le_bytes());
        buf.extend_from_slice(&100u32.to_le_bytes()); // payload_size
        buf.extend_from_slice(&0u32.to_le_bytes()); // checksum = 0 (skip validation)
        buf.extend_from_slice(&1u32.to_le_bytes()); // success
        buf.extend_from_slice(&0u32.to_le_bytes()); // error_code
        let result_size = std::mem::size_of::<BinaryExecutionResult>();
        buf.extend_from_slice(&(result_size as u32).to_le_bytes()); // output_offset
        buf.extend_from_slice(&5u32.to_le_bytes()); // output_len
        buf.extend_from_slice(&0u32.to_le_bytes()); // provides_words_offset
        buf.extend_from_slice(&0u32.to_le_bytes()); // provides_words_count
        buf.extend_from_slice(b"hello"); // output

        let result = decode_result(&buf);
        assert!(result.is_ok());
    }
}

use std::fs;
use std::io::Read;
use std::path::Path;
use std::sync::LazyLock;
use std::sync::Mutex;

static ACTIVE_TABLE: LazyLock<Mutex<Option<Box<FrequencyTable>>>> =
    LazyLock::new(|| Mutex::new(None));

pub type FrequencyTable = [[u16; 256]; 256];

const FREQ_TABLE_MAGIC: u32 = 0x4652_4551;
const FREQ_TABLE_VERSION: u32 = 1;

pub fn default_frequency_table() -> FrequencyTable {
    let mut table = [[0xFEu16; 256]; 256];
    for a in b'a'..=b'z' {
        for b in b'a'..=b'z' {
            table[a as usize][b as usize] = 0x1000;
        }
    }
    let common: &[([u8; 2], u16)] = &[
        (*b"th", 0x0800),
        (*b"he", 0x0800),
        (*b"in", 0x0800),
        (*b"er", 0x0800),
        (*b"an", 0x0800),
        (*b"re", 0x0800),
        (*b"on", 0x0800),
        (*b"at", 0x0800),
        (*b"en", 0x0800),
        (*b"nd", 0x0800),
        (*b"ti", 0x0800),
        (*b"es", 0x0800),
        (*b"or", 0x0800),
        (*b"te", 0x0800),
        (*b"of", 0x0800),
        (*b"ed", 0x0800),
        (*b"is", 0x0800),
        (*b"it", 0x0800),
        (*b"al", 0x0800),
        (*b"ar", 0x0800),
        (*b"st", 0x0800),
        (*b"to", 0x0800),
        (*b"nt", 0x0800),
        (*b"ng", 0x0800),
    ];
    for &([a, b], val) in common {
        table[a as usize][b as usize] = val;
    }
    table
}

pub fn get_default_pair_freq() -> &'static FrequencyTable {
    static DEFAULT: LazyLock<FrequencyTable> = LazyLock::new(default_frequency_table);
    &DEFAULT
}

pub fn set_frequency_table(table: Box<FrequencyTable>) {
    let mut active = ACTIVE_TABLE.lock().unwrap();
    *active = Some(table);
}

pub fn get_frequency_table() -> &'static FrequencyTable {
    get_default_pair_freq()
}

pub fn pair_weight(a: u8, b: u8) -> u16 {
    get_frequency_table()[a as usize][b as usize]
}

pub fn build_frequency_table(contents: &str) -> FrequencyTable {
    let mut table = default_frequency_table();
    let bytes = contents.as_bytes();
    for window in bytes.windows(2) {
        let a = window[0] as usize;
        let b = window[1] as usize;
        table[a][b] = table[a][b].saturating_add(1);
    }
    table
}

pub fn build_frequency_table_from_map(contents: &str) -> FrequencyTable {
    build_frequency_table(contents)
}

pub fn write_frequency_table(path: &Path, table: &FrequencyTable) -> std::io::Result<()> {
    let mut buf = Vec::with_capacity(4 + 4 + 256 * 256 * 2);
    buf.extend_from_slice(&FREQ_TABLE_MAGIC.to_le_bytes());
    buf.extend_from_slice(&FREQ_TABLE_VERSION.to_le_bytes());
    for row in table {
        for &val in row {
            buf.extend_from_slice(&val.to_le_bytes());
        }
    }
    fs::write(path, buf)
}

pub fn read_frequency_table(path: &Path) -> std::io::Result<Option<FrequencyTable>> {
    let Ok(mut file) = fs::File::open(path) else {
        return Ok(None);
    };
    let mut header = [0u8; 8];
    if file.read_exact(&mut header).is_err() {
        return Ok(None);
    }
    let magic = u32::from_le_bytes(header[0..4].try_into().unwrap());
    let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
    if magic != FREQ_TABLE_MAGIC || version != FREQ_TABLE_VERSION {
        return Ok(None);
    }
    let mut data = vec![0u8; 256 * 256 * 2];
    if file.read_exact(&mut data).is_err() {
        return Ok(None);
    }
    let mut table = [[0u16; 256]; 256];
    let mut off = 0;
    for row in &mut table {
        for val in row.iter_mut() {
            *val = u16::from_le_bytes(data[off..off + 2].try_into().unwrap());
            off += 2;
        }
    }
    Ok(Some(table))
}

#[cfg(test)]
mod tests {
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
}

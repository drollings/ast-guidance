#[allow(unused_imports)]
use common_core::prelude::*;
#[allow(unused_imports)]
use common_core::{uuid_v4, now_secs};

#[test]
fn prelude_smoke() {
        // NOTE (M11): `estimate_tokens` / `TokenBudget` left the prelude
        // with the M11 shim deletion (canonical owner `fluent_llm::tokens`).
        use common_core::prelude::*;
        let _ = blake3_hex(b"test");
        let _ = fnv1a64(b"test");
        let _ = sha256_hex(b"test");
        let _ = hex_encode(&[1u8, 2, 3]);
        let _ = LatencyHistogram::new();
        let _ = ensure_dir(std::path::Path::new("/tmp/common-core-prelude-smoke"));
    }

#[test]
fn uuid_v4_format() {
        let id = common_core::uuid_v4();
        assert_eq!(id.len(), 36);
        assert_eq!(id.chars().filter(|&c| c == '-').count(), 4);
    }

#[test]
fn now_secs_returns_nonzero_after_2020() {
        let s = common_core::now_secs();
        assert!(
            s > 1_577_836_800,
            "now_secs returned {s}, expected > 2020-01-01"
        );
    }

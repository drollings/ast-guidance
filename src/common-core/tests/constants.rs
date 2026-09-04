use common_core::constants::*;


#[test]
fn constants_match_expected() {
        assert_eq!(MAX_VALUE_LEN, 128);
        assert_eq!(MAX_FILE_SIZE, 100 * 1024 * 1024);
        assert_eq!(MAX_JSON_DEPTH, 100);
}

// NOTE (ROADMAP_20260903_LLM M11): the LLM-domain constant goldens moved to
// `fluent-llm --test constants` (canonical owner `fluent_llm::constants`)
// in M6, and M11 deleted the `common_core::constants` shims (with the
// shim-lock test that pinned them). The generic suites below stay.

#[test]
fn promoted_constants_match_coral_originals() {
        assert_eq!(MAX_KNN_CANDIDATES, 100_000);
        assert_eq!(MAX_MCP_REQUEST_SIZE, 10 * 1024 * 1024);
}

#[test]
fn hnsw_params_default_matches_previous_inline_values() {
        let p = HnswParams::default();
        assert_eq!(p.max_nb_connection, 16);
        assert_eq!(p.max_layer, 16);
        assert_eq!(p.ef_construction, 200);
        assert_eq!(p.initial_capacity, 1024);
}

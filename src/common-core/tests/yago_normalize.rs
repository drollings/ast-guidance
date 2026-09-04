use common_core::yago_normalize::*;


use std::collections::HashSet;

#[test]
fn adult_video_game_q() {
        assert_eq!(normalize_yago_name("http://yago-knowledge.org/resource/Adult_Video_Game_Q3362070"), "adult video game");
}
#[test]
fn city_q() {
        assert_eq!(normalize_yago_name("http://yago-knowledge.org/resource/City_Q515"), "city");
}
#[test]
fn u0028_decode() {
        let n = normalize_yago_name("http://yago-knowledge.org/resource/Remix__U0028_Work_U0029__Q113171270");
        assert!(n.contains("remix"));
        assert!(n.contains('('));
        assert!(n.contains("work"));
}
#[test]
fn token_variants_ies() {
        assert!(token_variants("cities").contains(&"city".to_string()));
}
#[test]
fn matches_lexicon_last_token() {
        let mut lex = HashSet::new();
        lex.insert("game".to_string());
        assert!(matches_lexicon("adult video game", &lex));
        assert!(!matches_lexicon("fakezootopia", &lex));
}

// Ported from `normalize_curie` pytest cases in
// src/ontology/tools/parse_yago_taxonomy_to_json.py
// (`test_normalize_curie_regex`). Prefix-preserving form lives in
// `common_core::yago_taxonomy::normalize_curie`; the local-name cases
// below exercise the shared `yago_normalize` pipeline.
#[test]
fn ported_normalize_curie_regex_cases() {
        use common_core::yago_taxonomy::normalize_curie;
        assert_eq!(normalize_curie("yago:City_Q515"), "yago:city");
        assert_eq!(normalize_curie("yago:City"), "yago:city");
        assert_eq!(normalize_curie("yago:Adult_Video_Game_Q3362070"), "yago:adult video game");
        assert_eq!(normalize_curie("<http://yago-knowledge.org/resource/City_Q515>"), "<http://yago-knowledge.org/resource/city>");
        assert_eq!(normalize_curie("yago:Hello_World"), "yago:hello world");
        assert_eq!(normalize_curie("yago:Video_game"), "yago:video game");
        // Regex `_Q\d+$`: no strip without the underscore.
        assert_eq!(normalize_curie("yago:AQ123"), "yago:aq123");
}

// Ported from `test_normalize_yago_name_regex` in
// src/ontology/tools/prune_yago_taxonomy.py.
#[test]
fn ported_normalize_yago_name_regex_cases() {
        assert_eq!(normalize_yago_name("http://yago-knowledge.org/resource/City_Q515"), "city");
        assert_eq!(normalize_yago_name("http://yago-knowledge.org/resource/Adult_Video_Game_Q3362070"), "adult video game");
}

// Documented Rust delta: the Python shim left `_UXXXX` escapes as
// `uXXXX` text; the unified Rust semantics decode them.
#[test]
fn unicode_escape_delta_vs_python() {
        use common_core::yago_taxonomy::normalize_curie;
        // Python: "yago:remix  u0028 work u0029". Rust decodes to parens.
        assert_eq!(
                normalize_curie("yago:Remix__U0028_Work_U0029__Q113171270"),
                "yago:remix (work)"
        );
}

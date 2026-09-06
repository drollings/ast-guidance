use super::*;

#[test]
fn default_matches_spacy_default() {
    // M6.1 audit: every default lives in the DTO and mirrors spacy's
    // `RefinePolicy::default()` field-for-field. All nine fields are
    // asserted — a drift in any `default_*` fn on either side fails here.
    let dto = RouterRefinePolicy::default();
    let spacy = spacy_rs::RefinePolicy::default();
    assert_eq!(dto.mode as u8, spacy.mode as u8);
    assert!((dto.min_overall - spacy.min_overall).abs() < 1e-9);
    assert!((dto.min_role_coverage - spacy.min_role_coverage).abs() < 1e-9);
    assert!((dto.min_token_score - spacy.min_token_score).abs() < 1e-9);
    assert!((dto.unresolved_token_threshold - spacy.unresolved_token_threshold).abs() < 1e-9);
    assert_eq!(dto.refine_on_ties, spacy.refine_on_ties);
    assert_eq!(
        dto.refine_on_unresolved_critical_role,
        spacy.refine_on_unresolved_critical_role
    );
    assert_eq!(dto.refine_on_unresolved_propn, spacy.refine_on_unresolved_propn);
    assert_eq!(dto.refine_on_collision_note, spacy.refine_on_collision_note);
}

#[test]
fn serde_empty_object_matches_on_both_sides() {
    // M6.1 audit, second leg: the `default_*` serde fns (the `{}` shape —
    // what a config omitting `refine_policy` fields deserializes to) agree
    // with `Default::default()` on both sides, so omitted fields can never
    // diverge from the audited defaults above.
    let dto_empty: RouterRefinePolicy = serde_json::from_str("{}").expect("dto {}");
    let dto_default = RouterRefinePolicy::default();
    assert!((dto_empty.min_overall - dto_default.min_overall).abs() < 1e-9);
    assert!((dto_empty.min_role_coverage - dto_default.min_role_coverage).abs() < 1e-9);
    assert!((dto_empty.min_token_score - dto_default.min_token_score).abs() < 1e-9);
    assert!((dto_empty.unresolved_token_threshold - dto_default.unresolved_token_threshold).abs() < 1e-9);
    assert_eq!(dto_empty.refine_on_ties, dto_default.refine_on_ties);
    assert_eq!(
        dto_empty.refine_on_unresolved_critical_role,
        dto_default.refine_on_unresolved_critical_role
    );
    assert_eq!(dto_empty.refine_on_unresolved_propn, dto_default.refine_on_unresolved_propn);
    assert_eq!(dto_empty.refine_on_collision_note, dto_default.refine_on_collision_note);
    assert_eq!(dto_empty.mode as u8, dto_default.mode as u8);

    let spacy_empty: spacy_rs::RefinePolicy = serde_json::from_str("{}").expect("spacy {}");
    let spacy_default = spacy_rs::RefinePolicy::default();
    assert!((spacy_empty.min_overall - spacy_default.min_overall).abs() < 1e-9);
    assert!((spacy_empty.min_role_coverage - spacy_default.min_role_coverage).abs() < 1e-9);
    assert!((spacy_empty.min_token_score - spacy_default.min_token_score).abs() < 1e-9);
    assert!((spacy_empty.unresolved_token_threshold - spacy_default.unresolved_token_threshold).abs() < 1e-9);
    assert_eq!(spacy_empty.refine_on_ties, spacy_default.refine_on_ties);
    assert_eq!(
        spacy_empty.refine_on_unresolved_critical_role,
        spacy_default.refine_on_unresolved_critical_role
    );
    assert_eq!(spacy_empty.refine_on_unresolved_propn, spacy_default.refine_on_unresolved_propn);
    assert_eq!(spacy_empty.refine_on_collision_note, spacy_default.refine_on_collision_note);
    assert_eq!(spacy_empty.mode as u8, spacy_default.mode as u8);
}

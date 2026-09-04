use common_core::interner::*;


#[test]
fn basic_intern_and_retrieve() {
        let reg = CapabilityRegistry::new();
        assert_eq!(reg.intern("hello"), 0);
        assert_eq!(reg.intern("world"), 1);
        assert_eq!(reg.intern("hello"), 0);
        assert_eq!(reg.count(), 2);
}

#[test]
fn get_index_returns_none_for_unknown() {
        let reg = CapabilityRegistry::new();
        reg.intern("known");
        assert_eq!(reg.get_index("known"), Some(0));
        assert_eq!(reg.get_index("unknown"), None);
}

#[test]
fn get_name_roundtrip() {
        let reg = CapabilityRegistry::new();
        reg.intern("foo");
        reg.intern("bar");
        assert_eq!(reg.get_name(0).as_deref(), Some("foo"));
        assert_eq!(reg.get_name(1).as_deref(), Some("bar"));
}

#[test]
fn to_bitvec_roundtrip() {
        let reg = CapabilityRegistry::new();
        reg.intern_list(&["compile", "link", "test"]);
        let bits = reg.to_bitvec(&["compile", "test"]);
        assert!(bits[0]);
        assert!(!bits[1]);
        assert!(bits[2]);
        assert_eq!(reg.bitvec_to_names(&bits).len(), 2);
}

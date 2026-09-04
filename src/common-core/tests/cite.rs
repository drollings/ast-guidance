use common_core::cite::*;


#[test]
fn extract_two_spans() {
        let text = "src/foo.rs:42 and src/bar.rs:7";
        let cites = extract_citations(text);
        assert_eq!(cites.len(), 2);
        assert_eq!(cites[0].file, "src/foo.rs");
        assert_eq!(cites[0].line, 42);
        assert_eq!(cites[1].file, "src/bar.rs");
        assert_eq!(cites[1].line, 7);
        // spans should cover the substrings
        assert_eq!(&text[cites[0].span.0..cites[0].span.1], "src/foo.rs:42");
        assert_eq!(&text[cites[1].span.0..cites[1].span.1], "src/bar.rs:7");
}

#[test]
fn extract_citation_at_inside_outside() {
        let text = "see src/foo.rs:42 here";
        let cites = extract_citations(text);
        assert_eq!(cites.len(), 1);
        let span = cites[0].span;
        let inside = span.0 + 2;
        let outside = 0;
        assert_eq!(extract_citation_at(text, inside).unwrap().file, "src/foo.rs");
        assert!(extract_citation_at(text, outside).is_none());
        // offset just past the span is outside
        assert!(extract_citation_at(text, span.1).is_none());
}

#[test]
fn no_file_extension_ignored() {
        let text = "See Makefile:42";
        let cites = extract_citations(text);
        assert!(cites.is_empty());
}

use common_core::error_context::*;


#[test]
fn init_with_all_fields() {
        let ctx = ErrorContext::new(
            "parse",
            Some("port"),
            Some("99999"),
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "Overflow"),
        );
        assert_eq!(ctx.operation, "parse");
        assert_eq!(ctx.field.as_deref(), Some("port"));
}

#[test]
fn simple_operation() {
        let ctx = ErrorContext::simple(
            "connect",
            std::io::Error::new(std::io::ErrorKind::TimedOut, "Timeout"),
        );
        assert_eq!(ctx.operation, "connect");
        assert!(ctx.field.is_none());
}

#[test]
fn format_with_all_fields() {
        let ctx = ErrorContext::new(
            "parse",
            Some("port"),
            Some("99999"),
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "Overflow"),
        );
        let s = format!("{ctx}");
        assert!(s.contains("[parse"));
        assert!(s.contains("port=99999"));
        assert!(s.contains("Overflow"));
        assert!(s.ends_with(']'));
}

#[test]
fn format_without_field() {
        let ctx = ErrorContext::simple(
            "connect",
            std::io::Error::new(std::io::ErrorKind::TimedOut, "Timeout"),
        );
        let s = format!("{ctx}");
        assert!(s.contains("[connect"));
        assert!(s.contains("Timeout"));
}

#[test]
fn heap_error_context_chain() {
        let inner =
            HeapErrorContext::new("inner", None, None, std::io::Error::other("inner error"));
        let outer = inner.chain(std::io::Error::other("outer error"));
        let s = format!("{outer}");
        assert!(s.contains("outer error"));
        assert!(s.contains("inner error"));
}

#[test]
fn value_truncation() {
        let long = "a".repeat(200);
        let ctx = ErrorContext::new(
            "test",
            Some("field"),
            Some(&long),
            std::io::Error::other("err"),
        );
        let s = format!("{ctx}");
        assert!(s.len() < 300);
}

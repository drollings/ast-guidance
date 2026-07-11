//! Contextual error wrappers: `ErrorContext` (stack-allocated) and `HeapErrorContext`.

use std::fmt;

use crate::constants::MAX_VALUE_LEN;

#[derive(Debug)]
struct ValueBuffer {
    data: [u8; MAX_VALUE_LEN],
    len: usize,
}

impl ValueBuffer {
    pub fn new(val: &str) -> Self {
        let end = val.len().min(MAX_VALUE_LEN);
        let mut data = [0u8; MAX_VALUE_LEN];
        data[..end].copy_from_slice(&val.as_bytes()[..end]);
        Self { data, len: end }
    }

    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.data[..self.len]).unwrap_or("")
    }
}

#[derive(Debug)]
pub struct ErrorContext {
    pub operation: String,
    pub field: Option<String>,
    value: ValueBuffer,
    pub cause: Box<dyn std::error::Error + Send + Sync + 'static>,
}

impl ErrorContext {
    pub fn new(
        operation: &str,
        field: Option<&str>,
        value: Option<&str>,
        cause: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self {
            operation: operation.to_string(),
            field: field.map(ToString::to_string),
            value: ValueBuffer::new(value.unwrap_or("")),
            cause: Box::new(cause),
        }
    }

    pub fn simple(operation: &str, cause: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::new(operation, None, None, cause)
    }
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}", self.operation)?;
        if let Some(ref field) = self.field {
            let val = self.value.as_str();
            if !val.is_empty() {
                write!(f, " {field}={val}")?;
            }
        }
        write!(f, ": {}]", self.cause)
    }
}

impl std::error::Error for ErrorContext {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.cause.as_ref())
    }
}

#[derive(Debug)]
pub struct HeapErrorContext {
    pub operation: String,
    pub field: Option<String>,
    pub value: Option<String>,
    pub cause: Box<dyn std::error::Error + Send + Sync>,
    pub message: String,
}

impl HeapErrorContext {
    pub fn new(
        operation: &str,
        field: Option<&str>,
        value: Option<&str>,
        cause: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        let message = format!("{cause}");
        Self {
            operation: operation.to_string(),
            field: field.map(ToString::to_string),
            value: value.map(ToString::to_string),
            cause: Box::new(cause),
            message,
        }
    }

    #[must_use]
    pub fn chain(self, parent: impl std::error::Error + Send + Sync + 'static) -> Self {
        let message = format!("{parent}: {}", self.message);
        Self {
            operation: self.operation,
            field: self.field,
            value: self.value,
            cause: Box::new(parent),
            message,
        }
    }
}

impl fmt::Display for HeapErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}", self.operation)?;
        if let Some(ref field) = self.field {
            if let Some(ref val) = self.value {
                write!(f, " {field}={val}")?;
            }
        }
        write!(f, ": {}]", self.message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}

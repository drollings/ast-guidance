use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::error::CopilotError;

/// Append-only JSONL audit log.
///
/// Every record is a single JSON line. All values are pre-hashed (blake3 hex)
/// — the `AuditLog` never sees raw PII. This makes the log safe to share
/// for debugging.
pub struct AuditLog {
    path: PathBuf,
    file: Mutex<File>,
}

impl AuditLog {
    /// Open (or create) the audit log at `path` in append mode.
    pub fn open(path: &Path) -> Result<Self, CopilotError> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| {
                CopilotError::Context(Box::new(common_core::error_context::ErrorContext::new(
                    "open_audit_log",
                    Some("path"),
                    Some(&path.display().to_string()),
                    e,
                )))
            })?;
        Ok(Self {
            path: path.to_path_buf(),
            file: Mutex::new(file),
        })
    }

    /// The path this log is writing to.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Record a successful `page.analyzeForm` call.
    ///
    /// All identifiers must be pre-hashed via `common_core::hash::blake3_hex`.
    pub fn record_analyze(
        &self,
        request_id_hash: &str,
        url_hash: &str,
        prefilled_count: u64,
        skipped_count: u64,
        duration_us: u64,
    ) -> Result<(), CopilotError> {
        let ts = time::OffsetDateTime::now_utc()
            .format(&time::format_description::well_known::Rfc3339)
            .unwrap_or_else(|_| "1970-01-01T00:00:00Z".into());
        let line = serde_json::json!({
            "ts": ts,
            "kind": "analyze",
            "request_id_hash": request_id_hash,
            "url_hash": url_hash,
            "prefilled_count": prefilled_count,
            "skipped_count": skipped_count,
            "duration_us": duration_us,
        });
        self.write_line(&line)
    }

    /// Record a `session.feedback` call.
    ///
    /// `final_value_hash` is already hashed by the client.
    pub fn record_feedback(
        &self,
        request_id_hash: &str,
        field_id_hash: &str,
        action: &str,
        final_value_hash: &str,
    ) -> Result<(), CopilotError> {
        let ts = time::OffsetDateTime::now_utc()
            .format(&time::format_description::well_known::Rfc3339)
            .unwrap_or_else(|_| "1970-01-01T00:00:00Z".into());
        let line = serde_json::json!({
            "ts": ts,
            "kind": "feedback",
            "request_id_hash": request_id_hash,
            "field_id_hash": field_id_hash,
            "action": action,
            "final_value_hash": final_value_hash,
        });
        self.write_line(&line)
    }

    /// Record an error (unknown method, handler panic, etc.).
    pub fn record_error(
        &self,
        request_id_hash: &str,
        kind: &str,
        message: &str,
    ) -> Result<(), CopilotError> {
        let ts = time::OffsetDateTime::now_utc()
            .format(&time::format_description::well_known::Rfc3339)
            .unwrap_or_else(|_| "1970-01-01T00:00:00Z".into());
        let line = serde_json::json!({
            "ts": ts,
            "kind": "error",
            "request_id_hash": request_id_hash,
            "error_kind": kind,
            "message": message,
        });
        self.write_line(&line)
    }

    fn write_line(&self, value: &serde_json::Value) -> Result<(), CopilotError> {
        let mut json_str = serde_json::to_string(value)?;
        json_str.push('\n');
        let mut file = self
            .file
            .lock()
            .map_err(|e| CopilotError::Audit(format!("lock poisoned: {e}")))?;
        file.write_all(json_str.as_bytes())
            .map_err(|e| CopilotError::Audit(format!("write failed: {e}")))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn temp_audit() -> (TempDir, AuditLog) {
        let dir = TempDir::new().unwrap();
        let log_path = dir.path().join("audit.jsonl");
        let log = AuditLog::open(&log_path).unwrap();
        (dir, log)
    }

    #[test]
    fn open_creates_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("new-audit.jsonl");
        assert!(!path.exists());
        let _log = AuditLog::open(&path).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn record_analyze_writes_jsonl() {
        let (_dir, log) = temp_audit();
        log.record_analyze("req-hash", "url-hash", 3, 1, 1234)
            .unwrap();

        let content = std::fs::read_to_string(log.path()).unwrap();
        let line = content.trim();
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(v["kind"], "analyze");
        assert_eq!(v["request_id_hash"], "req-hash");
        assert_eq!(v["url_hash"], "url-hash");
        assert_eq!(v["prefilled_count"], 3);
        assert_eq!(v["skipped_count"], 1);
        assert_eq!(v["duration_us"], 1234);
        assert!(v["ts"].is_string());
    }

    #[test]
    fn record_feedback_writes_jsonl() {
        let (_dir, log) = temp_audit();
        log.record_feedback("req-h", "field-h", "accepted", "val-h")
            .unwrap();

        let content = std::fs::read_to_string(log.path()).unwrap();
        let v: serde_json::Value = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(v["kind"], "feedback");
        assert_eq!(v["action"], "accepted");
        assert_eq!(v["final_value_hash"], "val-h");
    }

    #[test]
    fn record_error_writes_jsonl() {
        let (_dir, log) = temp_audit();
        log.record_error("req-h", "unknown_method", "no such method")
            .unwrap();

        let content = std::fs::read_to_string(log.path()).unwrap();
        let v: serde_json::Value = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(v["kind"], "error");
        assert_eq!(v["error_kind"], "unknown_method");
    }

    #[test]
    fn multiple_records_are_append_only() {
        let (_dir, log) = temp_audit();
        log.record_analyze("a", "b", 1, 0, 100).unwrap();
        log.record_feedback("c", "d", "rejected", "e").unwrap();
        log.record_error("f", "err", "boom").unwrap();

        let content = std::fs::read_to_string(log.path()).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 3);
        // Each line is valid JSON.
        for line in &lines {
            let _: serde_json::Value = serde_json::from_str(line).unwrap();
        }
    }
}

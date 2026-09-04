//! Shared test logging capture.
//!
//! `tracing` caches each callsite's interest the first time it is observed,
//! process-wide, and a thread with no subscriber at all (no scoped
//! `with_default` and no global default) dispatches to the no-op
//! `NoSubscriber`, whose `register_callsite` returns `Interest::never()`
//! (see `tracing-core`'s `subscriber.rs`). A callsite cached as `never()` is
//! silently dropped by *every* later subscriber — including a scoped
//! `capture_logs`. Parallel unit tests that build pipelines without a
//! capture therefore race to poison callsites before the assertion test
//! observes them, making log assertions flaky.
//!
//! `capture_logs` makes assertions deterministic:
//!   * it idempotently installs a global `Registry` subscriber (writing into
//!     a shared buffer) so a subscriber-less thread never hits
//!     `NoSubscriber`, and
//!   * it rebuilds the callsite-interest cache inside the capture scope so
//!     any callsite that *was* poisoned before the global existed is
//!     recomputed against the active capture subscriber.
//!
//! The global buffer is shared with the escalation/audit integration tests
//! (`server_http_tests::install_audit_capture`): whoever installs the global
//! first, every caller keeps reading the same log stream, so the two never
//! starve each other.

use std::io::Write;
use std::sync::{Arc, Mutex, OnceLock};
use tracing_subscriber::layer::SubscriberExt;

/// A `MakeWriter` that appends formatted lines to a shared `Vec<String>`.
#[derive(Clone, Default)]
pub(crate) struct LogCapture(Arc<Mutex<Vec<String>>>);

impl Write for LogCapture {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if let Ok(mut lines) = self.0.lock() {
            lines.push(String::from_utf8_lossy(buf).into_owned());
        }
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl tracing_subscriber::fmt::MakeWriter<'_> for LogCapture {
    type Writer = Self;
    fn make_writer(&self) -> Self::Writer {
        self.clone()
    }
}

/// Process-wide buffer behind the idempotent global subscriber.
static GLOBAL_CAPTURE: OnceLock<Arc<Mutex<Vec<String>>>> = OnceLock::new();

/// Install (once per process) a global `Registry` subscriber that formats
/// every line into `GLOBAL_CAPTURE`, and return the shared buffer.
///
/// Returning the buffer (rather than installing and discarding) means the
/// audit integration tests and the unit `capture_logs` read the same stream
/// regardless of which one wins the once-only `set_global_default`.
pub(crate) fn install_global_subscriber() -> Arc<Mutex<Vec<String>>> {
    GLOBAL_CAPTURE
        .get_or_init(|| {
            let capture = Arc::new(Mutex::new(Vec::<String>::new()));
            let subscriber = tracing_subscriber::registry().with(
                tracing_subscriber::fmt::layer()
                    .with_writer(LogCapture(Arc::clone(&capture)))
                    .with_ansi(false)
                    .with_target(true),
            );
            let _ = tracing::subscriber::set_global_default(subscriber);
            // Un-poison the callsite-interest cache: any `tracing` callsite
            // (e.g. `audit::emit`) first observed on a subscriber-less thread
            // is cached process-wide as `Interest::never()` and silently
            // dropped by every later subscriber. Rebuilding here recomputes
            // interest against the just-installed global, so audit tests that
            // lose the install race still observe emissions. Without this,
            // `wait_for_audit` can poll a full deadline for lines that will
            // never arrive while statuses flip normally.
            tracing::callsite::rebuild_interest_cache();
            capture
        })
        .clone()
}

/// Snapshot the current length of the shared global buffer. Audit tests
/// record this right after install and scan only the suffix: node ids
/// restart in every fresh store, so a whole-buffer scan can match a sibling
/// test's stale lines and unblock a wait (or an assertion) before this
/// test's own worker has emitted anything. Suffix-scoping keeps parallel
/// tests sound without clearing (which would drop a sibling's lines).
pub(crate) fn global_capture_len(capture: &Arc<Mutex<Vec<String>>>) -> usize {
    capture.lock().map(|lines| lines.len()).unwrap_or(0)
}

/// Lines appended to the shared global buffer since `base` (see
/// [`global_capture_len`]).
pub(crate) fn global_capture_since(
    capture: &Arc<Mutex<Vec<String>>>,
    base: usize,
) -> Vec<String> {
    capture
        .lock()
        .map(|lines| lines.iter().skip(base).cloned().collect())
        .unwrap_or_default()
}

/// Run `f` under a scoped subscriber that formats into a fresh capture,
/// returning `f`'s result and the captured lines.
///
/// See the module docs for why this is deterministic against `tracing`'s
/// set-once callsite-interest cache.
pub(crate) fn capture_logs<T>(f: impl FnOnce() -> T) -> (T, Vec<String>) {
    install_global_subscriber();
    let capture = LogCapture::default();
    let subscriber = tracing_subscriber::registry().with(
        tracing_subscriber::fmt::layer()
            .with_writer(capture.clone())
            .with_ansi(false)
            .with_target(true),
    );
    let result = tracing::subscriber::with_default(subscriber, || {
        tracing::callsite::rebuild_interest_cache();
        f()
    });
    let logs = capture
        .0
        .lock()
        .map(|mut lines| std::mem::take(&mut *lines))
        .unwrap_or_default();
    (result, logs)
}

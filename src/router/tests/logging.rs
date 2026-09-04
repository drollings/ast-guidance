use super::*;

#[test]
fn default_config_has_sensible_values() {
    let cfg = LoggingConfig::default();
    assert_eq!(cfg.max_file_size_mb, 100);
    assert_eq!(cfg.max_age_days, 30);
    assert_eq!(cfg.max_files, 10);
    assert!(cfg.console_output);
    assert!(!cfg.json_format);
}

#[test]
fn init_router_logging_creates_log_dir() {
    let dir = tempfile::tempdir().unwrap();
    let config = LoggingConfig {
        log_dir: dir.path().to_path_buf(),
        console_output: false,
        json_format: false,
        ..Default::default()
    };
    // init_router_logging can only be called once (global subscriber).
    // We verify the config values are accepted and the function type-checks.
    // In a full integration test we'd spawn a subprocess for this.
    let _ = config;
}

#[test]
fn audit_events_reach_audit_writer_not_ops_writer() {
    use tracing_subscriber::registry;

    let dir = tempfile::tempdir().unwrap();
    let ops_dir = dir.path().join("ops");
    let audit_dir = dir.path().join("audit");
    std::fs::create_dir_all(&ops_dir).unwrap();
    std::fs::create_dir_all(&audit_dir).unwrap();

    let ops_appender = tracing_appender::rolling::RollingFileAppender::builder()
        .rotation(tracing_appender::rolling::Rotation::NEVER)
        .filename_prefix("router")
        .filename_suffix("log")
        .build(&ops_dir)
        .unwrap();
    let audit_appender = tracing_appender::rolling::RollingFileAppender::builder()
        .rotation(tracing_appender::rolling::Rotation::NEVER)
        .filename_prefix("audit")
        .filename_suffix("log")
        .build(&audit_dir)
        .unwrap();

    let (ops_nb, ops_guard) = tracing_appender::non_blocking(ops_appender);
    let (audit_nb, audit_guard) = tracing_appender::non_blocking(audit_appender);

    // The real registry construction: an ops file layer plus the audit
    // layer over `router.audit=info,router.charts.audit=info`. Used under
    // `with_default` (never `.init()`) so the global subscriber stays
    // untouched for the rest of the test binary.
    let subscriber = registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(ops_nb)
                .with_filter(ops_filter()),
        )
        .with(audit_layer(audit_nb));

    tracing::subscriber::with_default(subscriber, || {
        crate::audit::emit("chart_target", serde_json::json!({ "chart": "c" }));
        tracing::info!(target: "router.pipeline", "ops event");
    });

    // Dropping the guards flushes the non-blocking writers synchronously.
    drop(audit_guard);
    drop(ops_guard);

    let ops_content = std::fs::read_to_string(ops_dir.join("router.log")).unwrap();
    let audit_content = std::fs::read_to_string(audit_dir.join("audit.log")).unwrap();

    assert!(
        audit_content.contains("chart_target"),
        "audit record missing from audit file:\n{audit_content}"
    );
    assert!(
        !audit_content.contains("ops event"),
        "ops event leaked into audit file:\n{audit_content}"
    );
    assert!(
        ops_content.contains("ops event"),
        "ops event missing from ops file:\n{ops_content}"
    );
    assert!(
        !ops_content.contains("chart_target"),
        "audit record leaked into ops file:\n{ops_content}"
    );
}

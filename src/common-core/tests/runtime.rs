use common_core::runtime::*;


    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn block_on_from_multi_thread_worker_completes() {
        // The case a plain `handle.block_on` fails on: a bare async task on a
        // multi-thread worker must not panic ("Cannot start a runtime from
        // within a runtime").
        let value = block_on(async { 40 + 2 });
        assert_eq!(value, 42);
}

    #[tokio::test(flavor = "current_thread")]
    async fn block_on_from_current_thread_runtime_completes() {
        // `block_in_place` panics on a current-thread runtime; the bridge must
        // use a plain `handle.block_on` instead. From the runtime's driver
        // task neither is legal, so exercise the current-thread branch from a
        // blocking thread, where the runtime handle is present and
        // `handle.block_on` is safe.
        let value = tokio::task::spawn_blocking(|| block_on(async { 40 + 2 }))
            .await
            .expect("blocking task must not panic");
        assert_eq!(value, 42);
}

#[test]
fn block_on_with_no_runtime_uses_fallback() {
        // A dedicated std::thread with no tokio runtime active must complete
        // via the process-wide fallback runtime.
        std::thread::spawn(|| {
            let value = block_on(async { 40 + 2 });
            assert_eq!(value, 42);
        })
        .join()
        .expect("thread must not panic");
}

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn nested_block_in_place_does_not_panic() {
        // `block_on` from inside an outer `tokio::task::block_in_place` nests
        // without panicking.
        let value = tokio::task::block_in_place(|| block_on(async { 40 + 2 }));
        assert_eq!(value, 42);
}

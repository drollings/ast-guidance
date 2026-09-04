use super::*;
use fluent_wvr::capability::CURRENT_CAPS;
use crate::tests::common::db_caps;

fn db() -> DbCapability {
    DbCapability::open(":memory:").unwrap()
}

#[tokio::test]
async fn open_and_pool_round_trip() {
    let db = db();
    let caps = db_caps();
    CURRENT_CAPS
        .scope(caps, async {
            let conn = db.pool().acquire().await.unwrap();
            conn.execute_batch("CREATE TABLE t (id INTEGER)").unwrap();
        })
        .await;
}

#[tokio::test]
async fn capability_name_is_db() {
    assert_eq!(db().name(), "db");
}

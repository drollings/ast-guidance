/// A single HNSW index handle — the workflow_library index path owned by the
/// `ChartStore`. Consumers (`main.rs`, the chart store, tests) construct one
/// from a name + SQLite path.
#[derive(Debug, Clone)]
pub struct HnswIndexHandle {
    pub name: String,
    pub path: String,
}

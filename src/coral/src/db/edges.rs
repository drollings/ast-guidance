use fluent_types::{GraphNode, NodeId};
use rusqlite::params;

use super::LibraryError;

impl super::Library {
    pub fn insert_edge(
        &self,
        source: NodeId,
        target: NodeId,
        edge_type: &str,
        weight: f64,
    ) -> Result<(), LibraryError> {
        self.store.execute(
            "INSERT INTO edges (source_node_id, target_node_id, edge_type, weight) VALUES (?1, ?2, ?3, ?4)",
            params![source.as_int(), target.as_int(), edge_type, weight],
        )?;
        Ok(())
    }

    pub fn traverse_from(
        &self,
        node_id: NodeId,
        max_depth: u8,
    ) -> Result<Vec<GraphNode>, LibraryError> {
        Ok(self.store.query_rows(
            "WITH RECURSIVE
                traverse(id, name, depth) AS (
                    SELECT n.id, n.name, 0
                    FROM context_nodes n
                    WHERE n.id = ?1
                    UNION ALL
                    SELECT n.id, n.name, t.depth + 1
                    FROM traverse t
                    JOIN edges e ON e.source_node_id = t.id
                    JOIN context_nodes n ON n.id = e.target_node_id
                    WHERE t.depth < ?2
                )
            SELECT DISTINCT id, name, depth FROM traverse ORDER BY depth, name",
            params![node_id.as_int(), max_depth],
            |row| {
                let id: i64 = row.get(0)?;
                let name: String = row.get(1)?;
                let depth: u32 = row.get(2)?;
                Ok(GraphNode {
                    node_id: NodeId::from_int(id),
                    name: name.as_str().into(),
                    depth,
                })
            },
        )?)
    }
    pub fn traverse_all_nodes(&self, max_depth: u8) -> Result<Vec<GraphNode>, LibraryError> {
        Ok(self.store.query_rows(
            "WITH RECURSIVE
                roots(id) AS (
                    SELECT id FROM context_nodes n
                    WHERE NOT EXISTS (SELECT 1 FROM edges e WHERE e.target_node_id = n.id)
                ),
                traverse(id, name, depth) AS (
                    SELECT n.id, n.name, 0 FROM context_nodes n JOIN roots r ON n.id = r.id
                    UNION ALL
                    SELECT n.id, n.name, t.depth + 1
                    FROM traverse t
                    JOIN edges e ON e.source_node_id = t.id
                    JOIN context_nodes n ON n.id = e.target_node_id
                    WHERE t.depth < ?1
                )
            SELECT DISTINCT id, name, depth FROM traverse ORDER BY depth, name",
            params![max_depth],
            |row| {
                let id: i64 = row.get(0)?;
                let name: String = row.get(1)?;
                let depth: u32 = row.get(2)?;
                Ok(GraphNode {
                    node_id: NodeId::from_int(id),
                    name: name.as_str().into(),
                    depth,
                })
            },
        )?)
    }

    pub fn insert_entity_type(&self, node_id: NodeId, type_iri: &str) -> Result<(), LibraryError> {
        self.store.execute(
            "INSERT INTO entity_types (node_id, type_iri) VALUES (?1, ?2)",
            params![node_id.as_int(), type_iri],
        )?;
        Ok(())
    }

    pub fn insert_entity_hierarchy(
        &self,
        subclass_iri: &str,
        superclass_iri: &str,
    ) -> Result<(), LibraryError> {
        self.store.execute(
            "INSERT INTO entity_hierarchy (subclass_iri, superclass_iri) VALUES (?1, ?2)",
            params![subclass_iri, superclass_iri],
        )?;
        Ok(())
    }

    pub fn is_a(&self, child_id: NodeId, parent_type_iri: &str) -> Result<bool, LibraryError> {
        let result = self.store.query_row(
            "WITH RECURSIVE ancestors(type_iri) AS (
                SELECT type_iri FROM entity_types WHERE node_id = ?1
                UNION
                SELECT eh.superclass_iri FROM ancestors a
                JOIN entity_hierarchy eh ON a.type_iri = eh.subclass_iri
            )
            SELECT COUNT(*) > 0 FROM ancestors WHERE type_iri = ?2",
            params![child_id.as_int(), parent_type_iri],
            |row| row.get::<_, bool>(0),
        )?;
        Ok(result.unwrap_or(false))
    }

    pub fn edge_count(&self) -> Result<i64, LibraryError> {
        let count = self
            .store
            .query_row("SELECT COUNT(*) FROM edges", &[], |row| {
                row.get::<_, i64>(0)
            })?;
        Ok(count.unwrap_or(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::common::make_node;

    fn lib() -> crate::db::Library {
        crate::db::Library::open_in_memory().expect("in-memory db")
    }

    fn insert(lib: &crate::db::Library, name: &str) -> NodeId {
        lib.insert_node(&make_node(name, "src")).expect("insert node")
    }

    #[test]
    fn insert_edge_and_count() {
        let lib = lib();
        let a = insert(&lib, "a");
        let b = insert(&lib, "b");
        lib.insert_edge(a, b, "depends", 1.0).expect("insert edge");
        assert_eq!(lib.edge_count().expect("count"), 1);
    }

    #[test]
    fn traverse_from_respects_depth_and_ordering() {
        let lib = lib();
        let a = insert(&lib, "a");
        let b = insert(&lib, "b");
        let c = insert(&lib, "c");
        lib.insert_edge(a, b, "depends", 1.0).expect("e1");
        lib.insert_edge(b, c, "depends", 1.0).expect("e2");

        // Depth 0: only the seed itself.
        let hops = lib.traverse_from(a, 0).expect("traverse");
        assert_eq!(hops.len(), 1);
        assert_eq!(hops[0].name.as_str(), "a");
        assert_eq!(hops[0].depth, 0);

        // Depth 1: seed + immediate neighbour.
        let hops = lib.traverse_from(a, 1).expect("traverse");
        let names: Vec<&str> = hops.iter().map(|g| g.name.as_str()).collect();
        assert!(names.contains(&"a") && names.contains(&"b"));
        assert!(!names.contains(&"c"));

        // Depth 2: the full chain.
        let hops = lib.traverse_from(a, 2).expect("traverse");
        let names: Vec<&str> = hops.iter().map(|g| g.name.as_str()).collect();
        assert!(names.contains(&"a") && names.contains(&"b") && names.contains(&"c"));
    }

    #[test]
    fn traverse_all_nodes_starts_at_roots() {
        let lib = lib();
        let a = insert(&lib, "a");
        let b = insert(&lib, "b");
        let c = insert(&lib, "c");
        lib.insert_edge(a, b, "depends", 1.0).expect("e1");
        lib.insert_edge(b, c, "depends", 1.0).expect("e2");
        // `a` is the only root (nothing points at it).
        let hops = lib.traverse_all_nodes(3).expect("traverse all");
        let names: Vec<&str> = hops.iter().map(|g| g.name.as_str()).collect();
        assert!(names.contains(&"a") && names.contains(&"b") && names.contains(&"c"));
        assert_eq!(hops.iter().find(|g| g.name.as_str() == "a").expect("a").depth, 0);
    }

    #[test]
    fn entity_type_and_is_a_hierarchy() {
        let lib = lib();
        let node = insert(&lib, "dog");
        lib.insert_entity_type(node, "http://schema/Mammal").expect("type");
        lib.insert_entity_hierarchy("http://schema/Mammal", "http://schema/Animal")
            .expect("hierarchy");
        lib.insert_entity_hierarchy("http://schema/Dog", "http://schema/Mammal")
            .expect("hierarchy");
        assert!(lib.is_a(node, "http://schema/Mammal").expect("direct"));
        assert!(lib.is_a(node, "http://schema/Animal").expect("transitive"));
        assert!(!lib.is_a(node, "http://schema/Plant").expect("unrelated"));
    }
}

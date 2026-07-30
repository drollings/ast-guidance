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
        let conn = self.conn.lock().unwrap();
        conn.execute(
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
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
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
        )?;

        let results = stmt
            .query_map(params![node_id.as_int(), max_depth], |row| {
                let id: i64 = row.get(0)?;
                let name: String = row.get(1)?;
                let depth: u32 = row.get(2)?;
                Ok(GraphNode {
                    node_id: NodeId::from_int(id),
                    name: name.as_str().into(),
                    depth,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(results)
    }

    pub fn traverse_all_nodes(&self, max_depth: u8) -> Result<Vec<GraphNode>, LibraryError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
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
        )?;

        let results = stmt
            .query_map(params![max_depth], |row| {
                let id: i64 = row.get(0)?;
                let name: String = row.get(1)?;
                let depth: u32 = row.get(2)?;
                Ok(GraphNode {
                    node_id: NodeId::from_int(id),
                    name: name.as_str().into(),
                    depth,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(results)
    }

    pub fn insert_entity_type(&self, node_id: NodeId, type_iri: &str) -> Result<(), LibraryError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
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
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO entity_hierarchy (subclass_iri, superclass_iri) VALUES (?1, ?2)",
            params![subclass_iri, superclass_iri],
        )?;
        Ok(())
    }

    pub fn is_a(&self, child_id: NodeId, parent_type_iri: &str) -> Result<bool, LibraryError> {
        let conn = self.conn.lock().unwrap();
        let result: bool = conn.query_row(
            "WITH RECURSIVE ancestors(type_iri) AS (
                SELECT type_iri FROM entity_types WHERE node_id = ?1
                UNION
                SELECT eh.superclass_iri FROM ancestors a
                JOIN entity_hierarchy eh ON a.type_iri = eh.subclass_iri
            )
            SELECT COUNT(*) > 0 FROM ancestors WHERE type_iri = ?2",
            params![child_id.as_int(), parent_type_iri],
            |row| row.get(0),
        )?;
        Ok(result)
    }

    pub fn edge_count(&self) -> Result<i64, LibraryError> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM edges", [], |row| row.get(0))?;
        Ok(count)
    }
}

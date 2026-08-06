//! Route resolution and routing configuration types.
//! `RoutingConfig` is the resolved routing table used by the classifier stage.

use std::cmp::Ordering;
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::ModelEntry;
use crate::config::ModelGroup;
use crate::pipeline::RoutingTarget;
use crate::score_matrix::ScoreMatrix;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteRef {
    pub group: String,
    #[serde(default = "default_pipelines")]
    pub pipelines: Vec<String>,
    #[serde(default)]
    pub description: String,
}

fn default_pipelines() -> Vec<String> {
    vec!["default".into()]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingConfig {
    pub routes: HashMap<String, RouteRef>,
    pub models: HashMap<String, ModelEntry>,
    pub model_groups: HashMap<String, ModelGroup>,
    pub system_prompt: String,
    pub safety_threshold: f64,
    pub default_route: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score_matrix: Option<ScoreMatrix>,
}

impl RoutingConfig {
    /// The `model_group` a route resolves through — the route's own `RouteRef`
    /// group, or the default route's group when the route is unknown (mirrors
    /// the `resolve_route` lookup at the top of [`Self::resolve_route`]).
    ///
    /// The target-matching ladder uses this to find the ordered candidate
    /// list for a resolved route; `None` when neither the route nor the
    /// default route has a group entry.
    pub fn route_group(&self, route: &str) -> Option<&str> {
        self.routes
            .get(route)
            .or_else(|| self.routes.get(&self.default_route))
            .map(|r| r.group.as_str())
    }

    /// Resolve a route name to a fully-populated typed `RoutingTarget`.
    ///
    /// Reuses `resolve_route` (cheapest model in the route's group whose
    /// `intelligence` meets `min_complexity`, falling back to the cheapest in
    /// the group) and attaches the resolved group, the `target_name`, and the
    /// ordered `fallbacks`. This is the canonical target builder shared by the
    /// flat classifier stage and the M4 classification-tree engine — a
    /// terminal node dispatches through it unchanged
    pub fn routing_target(&self, route: &str, min_complexity: Option<u8>) -> Option<RoutingTarget> {
        let (entry, model_name) = self.resolve_route(route, min_complexity)?;
        let group = self
            .routes
            .get(route)
            .or_else(|| self.routes.get(&self.default_route))
            .map_or(String::new(), |r| r.group.clone());

        let mut rt = RoutingTarget::from_model_entry(&model_name, entry);
        rt.group = Some(group);
        rt.target_name = Some(route.to_string());
        rt.fallbacks = self
            .all_dispatch_targets(route, min_complexity)
            .into_iter()
            .skip(1) // skip the primary (already included)
            .map(|(name, entry)| RoutingTarget::from_model_entry(&name, &entry))
            .collect();
        Some(rt)
    }

    pub fn resolve_route(
        &self,
        route_name: &str,
        min_complexity: Option<u8>,
    ) -> Option<(&ModelEntry, String)> {
        let route_ref = self
            .routes
            .get(route_name)
            .or_else(|| self.routes.get(&self.default_route));

        let route_ref =
            match route_ref {
                Some(r) => Some(r),
                None => {
                    return self.models.get(route_name).map(|entry| {
                    let name = entry.name.clone().unwrap_or_else(|| route_name.to_string());
                    tracing::info!(target: "router.config", route = %route_name, model = %name,
                        "route resolved as direct model"
                    );
                    (entry, name)
                }).or_else(|| {
                    tracing::warn!(target: "router.config", route = %route_name,
                        default = %self.default_route,
                        "no route or model found for target"
                    );
                    None
                });
                }
            };

        let route_ref = route_ref?;

        let model_names = self.model_groups.get(route_ref.group.as_str());
        let Some(model_names) = model_names else {
            tracing::warn!(target: "router.config", route = %route_name, group = %route_ref.group, "model group not found for route");
            return None;
        };
        let model_keys = model_names.models();

        tracing::debug!(target: "router.config",
            route = %route_name,
            group = %route_ref.group,
            model_count = model_keys.len(),
            min_complexity = ?min_complexity,
            "resolving route"
        );

        let candidates: Vec<(&String, &ModelEntry)> = model_keys
            .iter()
            .filter_map(|n| self.models.get(n).map(|m| (n, m)))
            .filter(|(_, m)| m.intelligence >= min_complexity.unwrap_or(0))
            .collect();

        if candidates.is_empty() {
            tracing::debug!(target: "router.config", route = %route_name, "no candidates passed complexity filter, falling back to cheapest in group");
            model_keys
                .iter()
                .filter_map(|n| self.models.get(n).map(|m| (n, m)))
                .min_by(|(_, a), (_, b)| {
                    (a.cost_input + a.cost_output)
                        .partial_cmp(&(b.cost_input + b.cost_output))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(entry_key, entry)| {
                    let name = entry.name.clone().unwrap_or_else(|| entry_key.clone());
                    tracing::info!(target: "router.config", route = %route_name, model = %name, "route resolved (cheapest fallback)");
                    (entry, name)
                })
        } else {
            let (entry_key, entry) = candidates.into_iter().min_by(|(_, a), (_, b)| {
                (a.cost_input + a.cost_output)
                    .partial_cmp(&(b.cost_input + b.cost_output))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })?;
            let name = entry.name.clone().unwrap_or_else(|| entry_key.clone());
            tracing::info!(target: "router.config", route = %route_name, model = %name, "route resolved");
            Some((entry, name))
        }
    }

    /// Return ALL available dispatch targets across all model groups ordered by
    /// dispatch preference:
    ///
    /// 1. Models from the resolved route's own group (cheapest first)
    /// 2. Models from other groups sorted by intelligence proximity to the
    ///    target complexity (closest first, cheapest tie-break)
    ///
    /// When the primary target fails (rate-limited, timeout, etc.) the caller
    /// can iterate this list to find a working model.
    pub fn all_dispatch_targets(
        &self,
        route_name: &str,
        min_complexity: Option<u8>,
    ) -> Vec<(String, ModelEntry)> {
        // Resolve the route to find its group
        let route_ref = self
            .routes
            .get(route_name)
            .or_else(|| self.routes.get(&self.default_route));

        let primary_group = route_ref.map(|r| r.group.as_str());

        // Collect all (model_key, model_entry) with resolved names
        let mut seen = std::collections::HashSet::new();
        let mut entries: Vec<(String, ModelEntry, f64)> = Vec::new();

        let target_intelligence = f64::from(min_complexity.unwrap_or(0));

        for (group_key, group) in &self.model_groups {
            let is_primary = primary_group == Some(group_key.as_str());
            for model_key in group.models() {
                if !seen.insert(model_key.clone()) {
                    continue;
                }
                if let Some(entry) = self.models.get(model_key) {
                    let name = entry.name.clone().unwrap_or_else(|| model_key.clone());
                    // Compute distance from target intelligence for cross-group sorting
                    let dist = if is_primary {
                        -f64::from(entry.intelligence) // primary group: negative so they sort first
                    } else {
                        (f64::from(entry.intelligence) - target_intelligence).abs()
                    };
                    entries.push((name, entry.clone(), dist));
                }
            }
        }

        // Sort: primary group first (dist < 0), then by intelligence proximity,
        // then by cost for tie-breaking
        entries.sort_by(|a, b| {
            let a_primary = a.2 < 0.0;
            let b_primary = b.2 < 0.0;
            match (a_primary, b_primary) {
                (true, false) => Ordering::Less,
                (false, true) => Ordering::Greater,
                _ => {
                    let d = a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal);
                    if d == Ordering::Equal {
                        let cost_a = a.1.cost_input + a.1.cost_output;
                        let cost_b = b.1.cost_input + b.1.cost_output;
                        cost_a.partial_cmp(&cost_b).unwrap_or(Ordering::Equal)
                    } else {
                        d
                    }
                }
            }
        });

        entries
            .into_iter()
            .map(|(name, entry, _)| (name, entry))
            .collect()
    }
}

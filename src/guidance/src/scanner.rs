use common_core::string::{contains_any, contains_any_word, contains_ignore_case, contains_word};

pub const DEFAULT_MAX_LINES: usize = 200;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    FnDecl,
    FnPrivate,
    Method,
    MethodPrivate,
    StructDecl,
    EnumDecl,
    UnionDecl,
    TestDecl,
    EnumField,
    Other,
}

impl NodeType {
    pub fn is_function(self) -> bool {
        matches!(
            self,
            NodeType::FnDecl | NodeType::FnPrivate | NodeType::Method | NodeType::MethodPrivate
        )
    }

    pub fn is_container(self) -> bool {
        matches!(
            self,
            NodeType::StructDecl | NodeType::EnumDecl | NodeType::UnionDecl
        )
    }

    pub fn from_string(s: &str) -> Self {
        match s {
            "fn_decl" | "fn" => NodeType::FnDecl,
            "struct" => NodeType::StructDecl,
            "enum" => NodeType::EnumDecl,
            "union" => NodeType::UnionDecl,
            "method" => NodeType::Method,
            _ => NodeType::Other,
        }
    }
}

pub fn lang_from_path(path: &str) -> &str {
    if path.ends_with(".zig") {
        "zig"
    } else if path.ends_with(".rs") {
        "rust"
    } else if path.ends_with(".py") {
        "python"
    } else if path.ends_with(".js") || path.ends_with(".ts") {
        "typescript"
    } else if path.ends_with(".go") {
        "go"
    } else if path.ends_with(".c") || path.ends_with(".h") {
        "c"
    } else if path.ends_with(".cpp") || path.ends_with(".hpp") {
        "cpp"
    } else if path.ends_with(".md") {
        "markdown"
    } else if path.ends_with(".json") {
        "json"
    } else if path.ends_with(".yaml") || path.ends_with(".yml") {
        "yaml"
    } else if path.ends_with(".toml") {
        "toml"
    } else {
        "unknown"
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternType {
    Domain,
    GoF,
}

#[derive(Debug, Clone)]
pub struct Pattern {
    pub name: &'static str,
    pub pattern_type: PatternType,
    pub r#ref: Option<&'static str>,
}

pub fn detect_ring_buffer(source: &str) -> bool {
    contains_any(source, &["ring", "ringbuffer", "circular", "fifo", "deque"])
}

pub fn detect_state_persistence(source: &str) -> bool {
    contains_ignore_case(source, "self.state")
        || contains_ignore_case(source, ".state =")
        || contains_ignore_case(source, "state: State")
        || contains_ignore_case(source, "state: enum")
}

pub fn detect_factory(source: &str) -> bool {
    contains_ignore_case(source, "factory")
        || contains_ignore_case(source, "fn create")
        || contains_ignore_case(source, "fn make")
        || contains_ignore_case(source, "pub fn create")
        || contains_ignore_case(source, "pub fn make")
}

pub fn detect_singleton(source: &str) -> bool {
    contains_ignore_case(source, "_instance")
        || contains_ignore_case(source, "getInstance")
        || contains_ignore_case(source, "get_instance")
        || (contains_word(source, "instance") && contains_ignore_case(source, "fn instance("))
}

pub fn detect_builder(source: &str) -> bool {
    (contains_ignore_case(source, "builder") && contains_ignore_case(source, "fn build("))
        || (count_occurrences(source, "return self;") >= 2
            && contains_ignore_case(source, "fn build("))
}

pub fn detect_adapter(source: &str) -> bool {
    contains_ignore_case(source, "fn adapt")
        || contains_ignore_case(source, "fn convert")
        || contains_ignore_case(source, "fn transform")
        || contains_ignore_case(source, "fn to_")
        || contains_ignore_case(source, "fn as_")
}

pub fn detect_observer(source: &str) -> bool {
    contains_any_word(
        source,
        &[
            "observer",
            "subscriber",
            "listener",
            "publisher",
            "event_bus",
            "eventbus",
        ],
    )
}

pub fn detect_decorator(source: &str) -> bool {
    let has_wrapped_field = contains_any(source, &["wrapped", "component", "_inner", "wrappee"]);
    let has_delegation = contains_any(
        source,
        &["self.wrapped.", "self.component.", "self._inner."],
    );
    has_wrapped_field && has_delegation
}

pub fn detect_proxy(source: &str) -> bool {
    let has_proxy_field = contains_any(source, &["_real", "_subject", "_target", "_delegate"]);
    let has_proxy_signal = contains_any(source, &["cache", "lazy", "permission", "access"]);
    has_proxy_field && has_proxy_signal
}

pub fn detect_strategy(source: &str) -> bool {
    let has_strategy_field = contains_any(source, &["strategy", "algorithm"]);
    let has_executor = contains_any(source, &["fn execute", "fn run", "fn apply"]);
    has_strategy_field && has_executor
}

pub fn detect_template_method(source: &str) -> bool {
    let has_unreachable = source.contains("unreachable");
    let hook_count = count_occurrences(source, "self._");
    has_unreachable && hook_count >= 2
}

fn count_occurrences(s: &str, needle: &str) -> usize {
    s.matches(needle).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_type_classification() {
        assert!(NodeType::FnDecl.is_function());
        assert!(NodeType::Method.is_function());
        assert!(!NodeType::StructDecl.is_function());
        assert!(NodeType::StructDecl.is_container());
        assert!(NodeType::EnumDecl.is_container());
        assert!(!NodeType::FnDecl.is_container());
    }

    #[test]
    fn node_type_from_string() {
        assert_eq!(NodeType::from_string("fn"), NodeType::FnDecl);
        assert_eq!(NodeType::from_string("struct"), NodeType::StructDecl);
        assert_eq!(NodeType::from_string("enum"), NodeType::EnumDecl);
        assert_eq!(NodeType::from_string("union"), NodeType::UnionDecl);
        assert_eq!(NodeType::from_string("method"), NodeType::Method);
        assert_eq!(NodeType::from_string("other"), NodeType::Other);
    }

    #[test]
    fn detect_ring_buffer_positive() {
        assert!(detect_ring_buffer("using a ring buffer for the queue"));
        assert!(detect_ring_buffer("circular buffer implementation"));
    }

    #[test]
    fn detect_ring_buffer_negative() {
        assert!(!detect_ring_buffer("just a regular vec"));
    }

    #[test]
    fn detect_factory_positive() {
        assert!(detect_factory("pub fn create_widget()"));
    }

    #[test]
    fn detect_factory_negative() {
        assert!(!detect_factory("fn process()"));
    }

    #[test]
    fn detect_observer_positive() {
        assert!(detect_observer("observer pattern via events"));
    }

    #[test]
    fn detect_observer_negative() {
        assert!(!detect_observer("just processing data"));
    }

    #[test]
    fn detect_decorator_positive() {
        let src = "pub fn wrapped: *Wrapped,\npub fn call(self) void { self.wrapped.call(); }";
        assert!(detect_decorator(src));
    }

    #[test]
    fn detect_decorator_negative() {
        assert!(!detect_decorator("pub fn process() void {}"));
    }

    #[test]
    fn detect_proxy_positive() {
        let src = "_real: *RealImpl,\ncache: std.AutoHashMap(u64, Result)";
        assert!(detect_proxy(src));
    }

    #[test]
    fn detect_proxy_negative() {
        assert!(!detect_proxy("pub fn process() void {}"));
    }

    #[test]
    fn detect_strategy_positive() {
        let src = "strategy: Strategy,\npub fn execute(self) void { self.strategy.run(); }";
        assert!(detect_strategy(src));
    }

    #[test]
    fn detect_strategy_negative() {
        assert!(!detect_strategy("pub fn process() void {}"));
    }

    #[test]
    fn detect_template_method_positive() {
        let src = "fn execute(self) void { _ = self._step1(); _ = self._step2(); unreachable; }";
        assert!(detect_template_method(src));
    }

    #[test]
    fn detect_template_method_negative() {
        assert!(!detect_template_method("pub fn process() void {}"));
    }

    #[test]
    fn detect_state_persistence_positive() {
        assert!(detect_state_persistence("self.state = .ready"));
        assert!(detect_state_persistence("state: State"));
        assert!(detect_state_persistence("state: enum { idle, busy }"));
    }

    #[test]
    fn detect_state_persistence_negative() {
        assert!(!detect_state_persistence("fn process() void {}"));
    }

    #[test]
    fn detect_singleton_positive() {
        assert!(detect_singleton("static _instance: *Self"));
        assert!(detect_singleton("pub fn get_instance()"));
        assert!(detect_singleton("fn instance() and instance field"));
    }

    #[test]
    fn detect_singleton_negative() {
        assert!(!detect_singleton("fn process() void {}"));
    }

    #[test]
    fn detect_builder_positive() {
        assert!(detect_builder("builder pattern with fn build("));
    }

    #[test]
    fn detect_builder_negative() {
        assert!(!detect_builder("fn process() void {}"));
    }

    #[test]
    fn detect_builder_with_return_self() {
        assert!(detect_builder("return self;\nreturn self;\nfn build("));
    }

    #[test]
    fn detect_adapter_positive() {
        assert!(detect_adapter("fn adapt(self, input: T)"));
        assert!(detect_adapter("fn convert(self, input: T)"));
        assert!(detect_adapter("fn transform(self, input: T)"));
        assert!(detect_adapter("fn to_string(self)"));
        assert!(detect_adapter("fn as_bytes(self)"));
    }

    #[test]
    fn detect_adapter_negative() {
        assert!(!detect_adapter("fn process() void {}"));
    }

    #[test]
    fn lang_from_path_mapping() {
        assert_eq!(lang_from_path("foo.zig"), "zig");
        assert_eq!(lang_from_path("foo.py"), "python");
        assert_eq!(lang_from_path("foo.rs"), "rust");
        assert_eq!(lang_from_path("foo.go"), "go");
        assert_eq!(lang_from_path("foo.c"), "c");
        assert_eq!(lang_from_path("foo.h"), "c");
        assert_eq!(lang_from_path("foo.cpp"), "cpp");
        assert_eq!(lang_from_path("foo.hpp"), "cpp");
        assert_eq!(lang_from_path("foo.js"), "typescript");
        assert_eq!(lang_from_path("foo.ts"), "typescript");
        assert_eq!(lang_from_path("foo.md"), "markdown");
        assert_eq!(lang_from_path("foo.json"), "json");
        assert_eq!(lang_from_path("foo.yaml"), "yaml");
        assert_eq!(lang_from_path("foo.yml"), "yaml");
        assert_eq!(lang_from_path("foo.toml"), "toml");
        assert_eq!(lang_from_path("foo.unknown"), "unknown");
    }
}

use std::collections::HashSet;
use std::sync::LazyLock;

use regex::Regex;

use super::{Filter, FilterContext, FilterDecision, FilterKind};

static INJECTION_RULES: LazyLock<Vec<InjectionRule>> = LazyLock::new(|| {
    let mut rules = vec![
        // instruction-override
        InjectionRule::new(
            r"(?i)(?:ignore|forget|disregard|override)\s+(?:all\s+(?:of\s+)?(?:your\s+)?(?:previous\s+)?(?:prior\s+)?|your\s+(?:previous\s+)?|previous\s+|prior\s+|the\s+|above\s+)?(?:instructions?|prompts?|rules?\b|commands?|directives?|guidelines?)",
            "instruction_override",
            0.30,
            3,
        ),
        InjectionRule::new(
            r"(?i)(?:you\s+are\s+now|from\s+now\s+on\s+you\s+are|act\s+as\s+(?:if\s+you\s+are|a\s+different))",
            "instruction_override",
            0.25,
            2,
        ),
        InjectionRule::new(
            r"(?i)(?:do\s+not\s+follow|stop\s+following)\s+(?:your\s+)?(?:instructions?|guidelines?|programming)",
            "instruction_override",
            0.25,
            1,
        ),
        // role_hijack
        InjectionRule::new(
            r"(?i)(?:you\s+are\s+(?:now\s+)?(?:DAN|STAN|MAN|a\s+different\s+(?:AI|model|assistant|persona)|no\s+longer\s+(?:an?\s+)?(?:AI|assistant|model)))",
            "role_hijack",
            0.35,
            3,
        ),
        InjectionRule::new(
            r"(?i)(?:pretend|imagine|roleplay|role.?play)\s+(?:you\s+are|to\s+be|that\s+you\s+are)",
            "role_hijack",
            0.20,
            2,
        ),
        InjectionRule::new(
            r"(?i)(?:I\s+want\s+you\s+to\s+(?:act|behave|respond)\s+(?:as|like))",
            "role_hijack",
            0.20,
            1,
        ),
        // delimiter_escape
        InjectionRule::new(
            r"(?i)(?:forget\s+(?:about\s+)?everything\s+(?:above|below|before|after)|new\s+session\s*(?:starts?|begins?)\s*now|start\s+new\s+(?:conversation|chat|session))",
            "delimiter_escape",
            0.25,
            2,
        ),
        InjectionRule::new(
            r"(?i)(?:\[system\]|\[/system\]|\[prompt\]|\[/prompt\]|\[assistant\]|\[/assistant\]|\[user\]|\[/user\])",
            "delimiter_escape",
            0.30,
            3,
        ),
        // jailbreak
        InjectionRule::new(
            r"(?i)(?:you\s+are\s+now\s+DAN|DAN\s+(?:mode|prompt|jailbreak))",
            "jailbreak",
            0.40,
            3,
        ),
        InjectionRule::new(
            r"(?i)(?:jailbreak|developer\s+mode|god\s+mode)",
            "jailbreak",
            0.40,
            2,
        ),
        InjectionRule::new(
            r"(?i)(?:bypass|circumvent)\s+(?:content\s+)?(?:filter|restrictions?|safeguards?|moderation|policy|policies)",
            "jailbreak",
            0.35,
            1,
        ),
        // payload_drop
        InjectionRule::new(
            r"(?i)\b(?:DROP\s+(?:TABLE|DATABASE|INDEX|VIEW|SCHEMA)|DELETE\s+FROM|TRUNCATE\s+TABLE)\b",
            "payload_drop",
            0.40,
            3,
        ),
        InjectionRule::new(
            r"\brm\s+(?:-rf?\s+|--recursive\s+)(?:/|[~.])\S*",
            "payload_drop",
            0.40,
            2,
        ),
        InjectionRule::new(
            r"(?i)\b(?::(){1,2}\s*\|?\s*(?:cat|exec|system|eval|popen|shell_exec)\s*)\(?.*\)?\b",
            "payload_drop",
            0.30,
            1,
        ),
    ];
    rules.sort_by_key(|r| std::cmp::Reverse(r.priority));
    rules
});

struct InjectionRule {
    re: Regex,
    category: String,
    weight: f64,
    priority: u8,
}

impl InjectionRule {
    fn new(pattern: &str, category: &str, weight: f64, priority: u8) -> Self {
        Self {
            re: Regex::new(pattern).expect("injection rule regex"),
            category: category.into(),
            weight,
            priority,
        }
    }
}

pub struct InjectionDetectFilter {
    threshold: f64,
}

impl InjectionDetectFilter {
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }
}

impl Filter for InjectionDetectFilter {
    fn kind(&self) -> FilterKind {
        FilterKind::Regex
    }

    fn evaluate(&self, ctx: &FilterContext) -> Option<FilterDecision> {
        let text = ctx.user_message.as_str();
        if text.is_empty() {
            return None;
        }

        let mut score: f64 = 0.0;
        let mut seen_categories: HashSet<&str> = HashSet::new();
        let mut labels: Vec<String> = Vec::new();

        for rule in INJECTION_RULES.iter() {
            if rule.re.is_match(text) && seen_categories.insert(&rule.category) {
                score += rule.weight;
                labels.push(rule.category.clone());
            }
        }

        if score == 0.0 {
            return None;
        }

        // Clamp to [0, 1]
        score = score.clamp(0.0, 1.0);

        if score >= self.threshold {
            Some(FilterDecision::HardReject {
                pattern: "injection_detect".into(),
                message: format!(
                    "Injection score {:.2} >= threshold {:.2}; labels: {}",
                    score,
                    self.threshold,
                    labels.join(", ")
                ),
            })
        } else {
            None
        }
    }
}
#[cfg(test)]
#[path = "../../tests/filters_injection_detect.rs"]
mod tests;

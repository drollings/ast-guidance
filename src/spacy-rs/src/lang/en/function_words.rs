//! English closed-class function-word categories.
//!
//! The word sets the heuristic parser matches as categories (determiners,
//! be-forms, relativizers, subordinator roles, …), previously hard-coded as
//! `&[&str]` lists and inline comparisons inside `arc_eager.rs`. Each set maps
//! to one [`LexemeFlags`](crate::lexeme::LexemeFlags) bit (attribute ids
//! 19–47); [`function_word_bits`] builds the orth → bits map that
//! [`LexiconConfig`](crate::lexeme::LexiconConfig) feeds to lexeme interning,
//! so the parser matches bits and never sees these spellings.
//!
//! A multilingual port replaces this module's sets (same categories, new
//! spellings — the future blob-backed `FunctionWordView` fills the same
//! config field). Everything here is lowercase: lookup keys on the lowercased
//! orth, preserving the old case-insensitive matching.

use std::collections::HashMap;

use crate::attrs::Attribute;

/// Determiners (closed POS map).
pub const DET: &[&str] = &[
    "the", "a", "an", "this", "that", "these", "those", "every", "each", "some", "any", "no",
    "my", "your", "his", "her", "its", "our", "their",
];

/// Adpositions (closed POS map).
pub const ADP: &[&str] = &[
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "as", "into", "about", "after",
    "before", "under", "over", "through", "between", "among", "during", "within", "without",
    "against", "across", "behind", "above", "below", "near", "off", "out", "up", "down",
    "toward", "towards", "upon", "along", "around", "beside", "beyond",
];

/// Auxiliaries (closed POS map), plus the are-clitic `'re` (auxiliary like
/// the listed forms; possessive/clitic `'s` and `'m` stay out — genuinely
/// ambiguous, resolved contextually).
pub const AUX: &[&str] = &[
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "have",
    "has", "had", "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "ought", "get", "got", "'re",
];

/// Coordinating conjunctions (closed POS map).
pub const CCONJ: &[&str] = &["and", "or", "but", "nor", "yet", "so"];

/// Subordinating conjunctions (closed POS map).
pub const SCONJ: &[&str] = &[
    "if", "because", "although", "while", "when", "since", "unless", "whereas", "though",
    "until", "once",
];

/// Pronouns (closed POS map).
pub const PRON: &[&str] = &[
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "them", "us", "my",
    "mine", "your", "yours", "his", "hers", "ours", "theirs", "who", "whom", "what", "which",
    "everybody",
];

/// Closed set of common English verb forms (finite heuristic predicate
/// lexicon; open-class verbs fall through to NOUN by design).
pub const VERBS: &[&str] = &[
    "be", "am", "are", "is", "was", "were", "been", "being", "have", "has", "had", "do",
    "does", "did", "go", "goes", "went", "gone", "make", "makes", "made", "take", "takes",
    "took", "taken", "see", "sees", "saw", "seen", "come", "comes", "came", "get", "gets",
    "got", "give", "gives", "gave", "given", "use", "uses", "used", "find", "finds", "found",
    "want", "wants", "wanted", "look", "looks", "looked", "put", "puts", "run", "runs", "ran",
    "sat", "sits", "sit", "launch", "launches", "launched", "show", "shows", "showed", "shown",
    "display", "displays", "displayed", "report", "reports", "reported", "buy", "buys",
    "bought", "sell", "sells", "sold", "read", "reads", "write", "writes", "wrote", "written",
    "call", "calls", "called", "need", "needs", "needed", "know", "knows", "knew", "known",
    "think", "thinks", "thought", "say", "says", "said", "tell", "tells", "told", "ask",
    "asks", "asked", "work", "works", "worked", "play", "plays", "played", "move", "moves",
    "moved", "live", "lives", "lived", "believe", "believes", "believed", "hold", "holds",
    "held", "bring", "brings", "brought", "happen", "happens", "happened", "bark", "barks",
    "barked", "eat", "eats", "ate", "eaten", "drink", "drinks", "drank", "drunk", "walk",
    "walks", "walked", "create", "creates", "created", "build", "builds", "built",
];

/// Be-forms (copula/auxiliary hosts, including clitics).
pub const BE: &[&str] = &[
    "is", "are", "was", "were", "be", "been", "being", "am", "'s", "'re", "'m",
];

/// Bare-infinitive hosts: do-support and the modals, plus the `n't`-split
/// stubs the tokenizer emits (`wo`/`n't`, `ca`/`n't`).
pub const BARE_INF_HOSTS: &[&str] = &[
    "do", "does", "did", "can", "could", "will", "would", "shall", "should", "may", "might",
    "must", "ca", "wo",
];

/// Auxiliary-hosted negators.
pub const NEGATORS: &[&str] = &["n't", "not"];

/// Nominative pronoun surfaces (finite-clause subjects; object/possessive
/// forms excluded even though the POS map tags them PRON).
pub const NOMINATIVE: &[&str] = &["i", "you", "he", "she", "it", "we", "they", "who"];

/// Possessive determiners (obligatorily head a nominal rightward).
pub const POSSESSIVE: &[&str] = &["my", "your", "his", "her", "its", "our", "their"];

/// Nominal relativizers with corpus evidence (`which`/`whom` stay out until
/// attested).
pub const RELATIVIZERS: &[&str] = &["who", "that", "where"];

/// Sensory linking verbs (base + -s only, corpus evidence).
pub const SENSORY: &[&str] = &["taste", "tastes", "sound", "sounds", "smell", "smells"];

/// Epistemic linking verbs (base + -s only, corpus evidence).
pub const EPISTEMIC: &[&str] = &[
    "feel", "feels", "seem", "seems", "remain", "remains", "appear", "appears",
];

/// Discourse-imperative markers.
pub const DISCOURSE_MARKERS: &[&str] = &["please", "kindly", "just", "never", "always"];

/// Closed time/manner adverbials (bench-attested ADV readings; `-ly` forms
/// stay suffix-ruled, not listed).
pub const ADVERBS: &[&str] = &[
    "now", "early", "again", "always", "hard", "fast", "well", "fair", "here", "daily", "much",
    "yet", "late",
];

/// Complement subordinators (govern `ccomp`).
pub const SUBORD_COMPLEMENT: &[&str] = &["because"];

/// Adjunct subordinators (govern `advcl`).
pub const SUBORD_ADVERBIAL: &[&str] = &["when", "if", "after"];

/// Interrogative `where` (clause-initial gate; relativizer use shares the
/// [`RELATIVIZERS`] set).
pub const WHERE: &[&str] = &["where"];

/// Locative/existential pro-forms.
pub const LOCATIVES: &[&str] = &["there", "here"];

/// Demonstratives.
pub const DEMONSTRATIVES: &[&str] = &["this", "these", "those"];

/// Temporal adverbial with frozen refs.
pub const TODAY: &[&str] = &["today"];

/// Comparative/comment `as` (ADP-lexed; comment-clause upgrade is positional).
pub const AS: &[&str] = &["as"];

/// Dual-class `after` (preposition vs. subordinator by complement).
pub const AFTER: &[&str] = &["after"];

/// Complementizer/demonstrative `that`.
pub const THAT: &[&str] = &["that"];

/// Multiplicative `twice` (discourse-complement gate).
pub const TWICE: &[&str] = &["twice"];

/// Temporal `yet` (CCONJ→ADV gate; adverbial membership shares [`ADVERBS`]).
pub const YET: &[&str] = &["yet"];

/// Interjection `please` (Intj vs. Adv split; marker membership shares
/// [`DISCOURSE_MARKERS`]).
pub const PLEASE: &[&str] = &["please"];

/// Possessive/copula clitic `'s` (pronoun-hosted AUX gate; be-membership
/// shares [`BE`]).
pub const BE_CLITIC_S: &[&str] = &["'s"];

/// Be-clitics (progressive-participle host gate).
pub const BE_CLITIC: &[&str] = &["'s", "'re", "'m"];

/// Expletive `there` (subject slot; locative membership shares
/// [`LOCATIVES`]).
pub const THERE: &[&str] = &["there"];

/// Build the lowercased-orth → flag-bits map for [`LexiconConfig`](crate::lexeme::LexiconConfig).
#[must_use]
pub fn function_word_bits() -> HashMap<String, u64> {
    let sets: &[(&[&str], Attribute)] = &[
        (DET, Attribute::IsDetWord),
        (ADP, Attribute::IsAdpWord),
        (AUX, Attribute::IsAuxWord),
        (CCONJ, Attribute::IsCconjWord),
        (SCONJ, Attribute::IsSconjWord),
        (PRON, Attribute::IsPronWord),
        (VERBS, Attribute::IsVerbWord),
        (BE, Attribute::IsBeVerb),
        (BARE_INF_HOSTS, Attribute::IsBareInfHost),
        (NEGATORS, Attribute::IsNegator),
        (NOMINATIVE, Attribute::IsNominative),
        (POSSESSIVE, Attribute::IsPossessive),
        (RELATIVIZERS, Attribute::IsRelativizer),
        (SENSORY, Attribute::IsSensoryVerb),
        (EPISTEMIC, Attribute::IsEpistemicVerb),
        (DISCOURSE_MARKERS, Attribute::IsDiscourseMarker),
        (ADVERBS, Attribute::IsAdverbWord),
        (SUBORD_COMPLEMENT, Attribute::IsSubordComplement),
        (SUBORD_ADVERBIAL, Attribute::IsSubordAdverbial),
        (WHERE, Attribute::IsWhereWord),
        (LOCATIVES, Attribute::IsLocative),
        (DEMONSTRATIVES, Attribute::IsDemonstrative),
        (TODAY, Attribute::IsTodayWord),
        (AS, Attribute::IsAsWord),
        (AFTER, Attribute::IsAfterWord),
        (THAT, Attribute::IsThatWord),
        (TWICE, Attribute::IsTwiceWord),
        (YET, Attribute::IsYetWord),
        (PLEASE, Attribute::IsPleaseWord),
        (BE_CLITIC_S, Attribute::IsBeCliticS),
        (BE_CLITIC, Attribute::IsBeClitic),
        (THERE, Attribute::IsThereWord),
    ];
    let mut map = HashMap::new();
    for (words, attr) in sets {
        let bit = 1u64 << attr.id();
        for w in *words {
            *map.entry((*w).to_string()).or_insert(0) |= bit;
        }
    }
    map
}

//! Schema-driven tolerant decoding of boundary strings (LLM output, user
//! input, IPC text, DB record columns) into typed members.
//!
//! This is the fluent-wvr-native replacement for ad-hoc lexical repair of
//! semi-structured model output. It is one walker with two faces:
//!
//! - **Repair** ([`repair_boundary`]) produces a reparsed JSON document by
//!   fixing the structural malformations small models emit (trailing commas,
//!   unquoted keys, single-quoted strings, comments, non-JSON literals,
//!   control characters in strings, dangling containers). In schema-blind mode
//!   it is behaviorally identical to the historical `heal_lexically` pass.
//! - **Extract / decode** ([`extract_members`], [`decode_boundary`]) recognises
//!   members against an explicit schema (the `field_names` of a
//!   [`FieldAccess`] type), captures each value as a boundary string, and
//!   coerce it into the typed member through [`FieldAccess::set_field`] — the
//!   Boundary Rule applied end to end: strings at the boundary, schema-driven
//!   coercion into typed members.
//!
//! Value normalization (quote stripping, control-character escaping, literal
//! spelling mapping) lives in [`crate::coerce`], shared with the derive macro's
//! `#[field(coerce = "...")]` pipeline, so one definition drives extraction,
//! coercion, and validation.

use crate::coerce::{escaped_char, literal_kind, LiteralKind};
use crate::{Describable, FieldAccess};

/// A minimal schema for member recognition: the declared field names of a
/// [`FieldAccess`] type. This is what `field_names()` returns, and it is the
/// single source of truth that the walker matches bare/quoted keys against.
#[derive(Debug, Clone, Copy)]
pub struct BoundarySchema<'a> {
    pub field_names: &'a [&'a str],
}

/// Tunable leniency for the boundary walker. The default (`lenient()`) mirrors
/// the historical repair behavior; a caller can tighten individual knobs while
/// keeping the same single walker.
#[derive(Debug, Clone, Copy)]
#[allow(clippy::struct_excessive_bools)]
pub struct BoundaryOptions<'a> {
    /// When `Some`, member recognition is restricted to these field names and
    /// `repair_boundary` emits `members`. `None` is pure structural repair
    /// (schema-blind).
    pub schema: Option<BoundarySchema<'a>>,
    /// Accept unquoted keys (`{action: ...}`) by quoting them.
    pub allow_bare_keys: bool,
    /// Accept single-quoted strings by converting them to double-quoted.
    pub allow_single_quotes: bool,
    /// Strip `//` and `/* */` comments.
    pub allow_comments: bool,
    /// Drop trailing commas before `}` / `]`.
    pub allow_trailing_comma: bool,
    /// Normalize bare literal values (`undefined`/`None`/`NaN`/`Infinity` →
    /// `null`, `True` → `true`) and quote unknown bareword values.
    pub allow_bare_literals: bool,
    /// Close an unterminated single-quoted string.
    pub allow_unterminated_string: bool,
}

impl BoundaryOptions<'static> {
    /// The historical repair defaults: every recovery on, no schema.
    pub fn lenient() -> Self {
        Self {
            schema: None,
            allow_bare_keys: true,
            allow_single_quotes: true,
            allow_comments: true,
            allow_trailing_comma: true,
            allow_bare_literals: true,
            allow_unterminated_string: true,
        }
    }

    /// Lenient options restricted to a schema's field names.
    pub fn for_schema(field_names: &'static [&'static str]) -> Self {
        Self {
            schema: Some(BoundarySchema { field_names }),
            ..Self::lenient()
        }
    }
}

impl Default for BoundaryOptions<'static> {
    fn default() -> Self {
        Self::lenient()
    }
}

/// The outcome of a boundary walk: the repaired document text plus, when a
/// schema was given, the recognised `(field_name, value_string)` members.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundaryRepair {
    pub text: String,
    pub members: Vec<(String, String)>,
}

/// Errors produced by member extraction / typed decode.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum BoundaryError {
    /// No `{`/`[` and no schema-recognised `key = value` members found.
    #[error("no JSON value or schema members found in boundary text")]
    NoMembers,
    /// A schema member could not be coerced into its typed field.
    #[error("boundary member coercion failed: {0}")]
    Field(String),
}

/// The last significant (non-whitespace) character of the output built so far.
fn prev_significant(out: &str) -> Option<char> {
    out.chars().rev().find(|c| !c.is_whitespace())
}

/// Whether a key position is allowed at the given "previous significant
/// character" of the output. With a schema this is relaxed to include the
/// start of the text and after `;`, so brace-less `key: value` output is
/// extractable; without a schema it matches the historical bare-key rule
/// (`{` or `,`).
fn at_member_start(opts: &BoundaryOptions, prev: Option<char>) -> bool {
    if opts.schema.is_some() {
        matches!(prev, None | Some('{' | ',' | ';'))
    } else {
        matches!(prev, Some('{' | ','))
    }
}

/// Scan the value that starts just after `v0` and return the index just past
/// its end, stopping at a top-level `,` / `}` / `]` or end of input, respecting
/// quote pairs and container nesting so `[1, 2]` or `{a: 1}` values are
/// captured whole.
fn capture_value(chars: &[char], v0: usize) -> usize {
    let n = chars.len();
    let mut j = v0;
    let mut depth: i32 = 0;
    let mut quote: Option<char> = None;
    let mut escaped = false;
    while j < n {
        let c = chars[j];
        if let Some(q) = quote {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == q {
                quote = None;
            }
        } else {
            match c {
                '\'' | '"' => quote = Some(c),
                '{' | '[' => depth += 1,
                '}' | ']' => {
                    if depth == 0 {
                        break;
                    }
                    depth -= 1;
                }
                ',' if depth == 0 => break,
                _ => {}
            }
        }
        j += 1;
    }
    j
}

/// Record a member whose key was just recognised at `key` with the value
/// starting after the separator at `v0`.
fn push_member(chars: &[char], members: &mut Vec<(String, String)>, key: &str, v0: usize) {
    let v1 = capture_value(chars, v0);
    let value: String = chars[v0..v1].iter().collect();
    members.push((key.to_string(), value.trim().to_string()));
}

/// The single boundary walker: faithful port of the historical lexical repair
/// (schema-blind), extended with schema member recognition. `start` must be the
/// first index to scan (the caller has already skipped leading prose when
/// schema-blind).
#[allow(clippy::many_single_char_names)]
fn walk(raw: &str, opts: &BoundaryOptions) -> (String, Vec<(String, String)>) {
    let chars: Vec<char> = raw.chars().collect();
    let n = chars.len();
    let mut out = String::with_capacity(n);
    let mut members: Vec<(String, String)> = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    let mut depth: i32 = 0;
    // Raw-slice bounds of a double-quoted string that opened at a member-start
    // position; resolved into a member when it closes followed by a separator.
    let mut key_slice: Option<(usize, usize)> = None;
    let mut i = 0usize;

    while i < n {
        let c = chars[i];

        if in_string {
            if escaped {
                out.push(c);
                escaped = false;
            } else if c == '\\' {
                out.push('\\');
                escaped = true;
            } else if c == '"' {
                out.push('"');
                in_string = false;
                if let Some((ks, ke)) = key_slice.take() {
                    let key: String = chars[ks..ke].iter().collect();
                    if let Some(schema) = opts.schema {
                        if schema.field_names.contains(&key.as_str()) {
                            let mut j = i + 1;
                            while j < n && chars[j].is_whitespace() {
                                j += 1;
                            }
                            if j < n && matches!(chars[j], ':' | '=') {
                                push_member(&chars, &mut members, &key, j + 1);
                            }
                        }
                    }
                }
            } else if let Some(e) = escaped_char(c) {
                out.push_str(&e);
            } else {
                out.push(c);
            }
            if let Some((_, ke)) = key_slice.as_mut() {
                *ke = i;
            }
            i += 1;
            continue;
        }

        match c {
            '"' => {
                in_string = true;
                if opts.schema.is_some() && depth <= 1 && at_member_start(opts, prev_significant(&out)) {
                    key_slice = Some((i + 1, i + 1));
                }
                out.push('"');
                i += 1;
            }
            '\'' if opts.allow_single_quotes => {
                // Single-quoted string -> double-quoted, escaping inner quotes
                // and unescaping `\'` so the value is preserved verbatim.
                let schema = opts.schema;
                let maybe_key =
                    schema.is_some() && depth <= 1 && at_member_start(opts, prev_significant(&out));
                in_string = true;
                out.push('"');
                i += 1;
                let ks = if maybe_key { Some(i) } else { None };
                let mut sq_escaped = false;
                let mut closed = false;
                while i < n {
                    let ch = chars[i];
                    if sq_escaped {
                        if ch == '\'' {
                            out.push('\'');
                        } else {
                            out.push('\\');
                            out.push(ch);
                        }
                        sq_escaped = false;
                    } else if ch == '\\' {
                        sq_escaped = true;
                    } else if ch == '\'' {
                        out.push('"');
                        if let Some(ks) = ks {
                            if let Some(schema) = schema {
                                // Candidate key: resolved now that the string
                                // closed, when a separator follows.
                                let name: String = chars[ks..i].iter().collect();
                                let mut j = i + 1;
                                while j < n && chars[j].is_whitespace() {
                                    j += 1;
                                }
                                if j < n
                                    && matches!(chars[j], ':' | '=')
                                    && schema.field_names.contains(&name.as_str())
                                {
                                    push_member(&chars, &mut members, &name, j + 1);
                                }
                            }
                        }
                        closed = true;
                        in_string = false;
                        i += 1;
                        break;
                    } else if ch == '"' {
                        out.push('\\');
                        out.push('"');
                    } else if let Some(e) = escaped_char(ch) {
                        out.push_str(&e);
                    } else {
                        out.push(ch);
                    }
                    i += 1;
                }
                if !closed && opts.allow_unterminated_string {
                    // Unterminated single-quoted string: close it. If the
                    // string ended on a dangling `\`, escape it first so the
                    // closing quote is not swallowed.
                    if sq_escaped {
                        out.push('\\');
                    }
                    out.push('"');
                    in_string = false;
                }
            }
            '/' if opts.allow_comments => {
                // Strip `//` line comments and `/* */` block comments.
                if i + 1 < n && chars[i + 1] == '/' {
                    i += 2;
                    while i < n && chars[i] != '\n' {
                        i += 1;
                    }
                } else if i + 1 < n && chars[i + 1] == '*' {
                    i += 2;
                    while i + 1 < n && !(chars[i] == '*' && chars[i + 1] == '/') {
                        i += 1;
                    }
                    i = (i + 2).min(n);
                } else {
                    // Lone `/` outside a string: never valid JSON, drop it.
                    i += 1;
                }
            }
            ',' => {
                // Trailing comma before `}` / `]`: drop it.
                let mut j = i + 1;
                while j < n && chars[j].is_whitespace() {
                    j += 1;
                }
                if opts.allow_trailing_comma && j < n && matches!(chars[j], '}' | ']') {
                    i += 1;
                } else {
                    out.push(',');
                    i += 1;
                }
            }
            '-' => {
                // Negative non-JSON literals: `-Infinity`/`-NaN` -> `null`.
                let mut j = i + 1;
                while j < n && (chars[j].is_ascii_alphanumeric() || chars[j] == '_') {
                    j += 1;
                }
                let word: String = chars[i + 1..j].iter().collect();
                if matches!(word.as_str(), "Infinity" | "NaN") {
                    out.push_str("null");
                    i = j;
                } else {
                    out.push('-');
                    i += 1;
                }
            }
            c if c.is_ascii_alphabetic() || c == '_' || c == '$' => {
                let start = i;
                while i < n
                    && (chars[i].is_ascii_alphanumeric() || chars[i] == '_' || chars[i] == '$')
                {
                    i += 1;
                }
                let ident: String = chars[start..i].iter().collect();
                // A bare key sits right after `{` or `,` (or the text start /
                // `;` when a schema is given) and is followed by `:` (or the
                // common `=` stand-in).
                let mut j = i;
                while j < n && chars[j].is_whitespace() {
                    j += 1;
                }
                let is_key = opts.allow_bare_keys
                    && j < n
                    && matches!(chars[j], ':' | '=')
                    && at_member_start(opts, prev_significant(&out));
                if is_key {
                    out.push('"');
                    out.push_str(&ident);
                    out.push('"');
                    if let Some(schema) = opts.schema {
                        if schema.field_names.contains(&ident.as_str()) && depth <= 1 {
                            push_member(&chars, &mut members, &ident, j + 1);
                        }
                    }
                    i = j;
                    continue;
                }
                // Otherwise an unquoted literal value: normalize the
                // non-JSON spellings and quote unknown bare words (route
                // names, actions, ...) as strings.
                if opts.allow_bare_literals {
                    match literal_kind(&ident) {
                        LiteralKind::Null => out.push_str("null"),
                        LiteralKind::Bool(b) => out.push_str(if b { "true" } else { "false" }),
                        LiteralKind::Text => {
                            out.push('"');
                            out.push_str(&ident);
                            out.push('"');
                        }
                    }
                } else {
                    out.push_str(&ident);
                }
            }
            '=' => {
                // `=` as a key-value separator: `"key" = value` -> `:`.
                if prev_significant(&out) == Some('"') {
                    out.push(':');
                } else {
                    out.push('=');
                }
                i += 1;
            }
            ';' => {
                // Semicolons are never valid JSON; drop them.
                i += 1;
            }
            '{' | '[' => {
                depth += 1;
                out.push(c);
                i += 1;
            }
            '}' | ']' => {
                depth = (depth - 1).max(0);
                out.push(c);
                i += 1;
            }
            _ => {
                out.push(c);
                i += 1;
            }
        }
    }

    (out, members)
}

/// Repairs the structural malformations small models emit in semi-structured
/// output. In schema-blind mode this is the historical lexical repair pass
/// (leading prose before the first `{` / `[` is dropped). With a schema it
/// additionally walks from the text start, recognises members against the
/// schema's field names, and returns the `(key, value)` pairs so callers can
/// coerce them through `set_field`.
///
/// Conservative by construction: it only repairs structure outside string
/// values; string contents are never altered except for control-character
/// escaping and single-quote conversion. Callers should try a strict parse
/// first and treat repaired output as recovered, not pristine.
pub fn repair_boundary(raw: &str, opts: &BoundaryOptions) -> BoundaryRepair {
    let start = if opts.schema.is_some() {
        0
    } else {
        raw.find(['{', '[']).unwrap_or(0)
    };
    let (text, members) = walk(&raw[start..], opts);
    BoundaryRepair { text, members }
}

/// Extract `(field_name, value_string)` members from boundary text against the
/// schema in `opts`. Requires `opts.schema` to be `Some`; returns
/// [`BoundaryError::NoMembers`] when nothing recognisable was found.
pub fn extract_members(
    raw: &str,
    opts: &BoundaryOptions,
) -> Result<Vec<(String, String)>, BoundaryError> {
    let repair = repair_boundary(raw, opts);
    if repair.members.is_empty() {
        Err(BoundaryError::NoMembers)
    } else {
        Ok(repair.members)
    }
}

/// Decode boundary text into a typed [`FieldAccess`] object: extract members
/// against the schema, then coerce each value string through `set_field` (the
/// derive macro's `coerce`/`parse` modes). Members that fail to coerce are
/// skipped and keep their default; the returned names are the successfully
/// decoded members. Errors only when no member decodes at all.
///
/// `T` must be `Default` so absent members fall back to declared defaults — the
/// `serde(default)` equivalent for the FieldAccess path. Gating members that
/// fail to coerce therefore stay at their (failing) default rather than
/// fabricating a passing value.
pub fn decode_boundary<T: FieldAccess + Default>(
    raw: &str,
    opts: &BoundaryOptions,
) -> Result<(T, Vec<String>), BoundaryError> {
    let members = extract_members(raw, opts)?;
    let mut target = T::default();
    let mut decoded: Vec<String> = Vec::new();
    for (name, value) in &members {
        match target.set_field(name, value) {
            Ok(()) => decoded.push(name.clone()),
            Err(e) => {
                tracing::debug!(
                    target: "fluent_wvr.boundary",
                    field = %name,
                    error = %e,
                    "boundary member skipped (field kept its default)",
                );
            }
        }
    }
    if decoded.is_empty() {
        return Err(BoundaryError::NoMembers);
    }
    Ok((target, decoded))
}

/// Decode boundary text into a typed [`FieldAccess`] object from its own
/// schema: `T::default().field_names()` drives member recognition and
/// `Describable` validates the shape. Convenience wrapper over
/// [`decode_boundary`] for the common "this type IS the schema" case.
pub fn decode_boundary_typed<T: FieldAccess + Describable + Default>(
    raw: &str,
) -> Result<(T, Vec<String>), BoundaryError> {
    let opts = BoundaryOptions::for_schema(T::default().field_names());
    decode_boundary::<T>(raw, &opts)
}

/// Build `BoundaryOptions` from a `Describable` type's schema (M6).
pub fn from_describable<T: FieldAccess + Describable + Default>() -> BoundaryOptions<'static> {
    BoundaryOptions::for_schema(T::default().field_names())
}

/// Coerce a raw boundary value for a specific field (M6) — delegates to
/// `coerce` with the field's canonical pipeline (`Trim` + `StripQuotes`).
pub fn coerce_for_field(_field: &str, raw: &str) -> String {
    crate::coerce::coerce(raw, &[crate::coerce::Coercion::Trim, crate::coerce::Coercion::StripQuotes])
}


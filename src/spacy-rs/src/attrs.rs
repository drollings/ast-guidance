//! The attribute-id space and the `get_struct_attr` dispatch, mirroring
//! `spacy/attrs.pxd` and `Token.get_struct_attr` (`spacy/tokens/token.pxd`).
//!
//! spaCy reserves ids 0–63 for boolean lexeme flags (bit positions into the
//! `flags_t` bitmask) and ids ≥ 64 for value attributes. A single `match`
//! reproduces the dispatch with zero string comparisons — this is what
//! `Doc::to_array` runs per cell.

use crate::error::SpacyError;

/// A token/lexeme attribute id. Named variants cover the ids spaCy defines;
/// [`Attribute::Other`] catches reserved flag slots (19–63) and unknown ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum Attribute {
    // ── boolean flags (ids 1..=18; bit positions into LexemeFlags) ──
    IsAlpha = 1,
    IsAscii = 2,
    IsDigit = 3,
    IsLower = 4,
    IsPunct = 5,
    IsSpace = 6,
    IsTitle = 7,
    IsUpper = 8,
    LikeUrl = 9,
    LikeNum = 10,
    LikeEmail = 11,
    IsStop = 12,
    IsOovDeprecated = 13,
    IsBracket = 14,
    IsQuote = 15,
    IsLeftPunct = 16,
    IsRightPunct = 17,
    IsCurrency = 18,
    // ── value attributes (ids ≥ 64) ──
    Id = 64,
    Orth = 65,
    Lower = 66,
    Norm = 67,
    Shape = 68,
    Prefix = 69,
    Suffix = 70,
    Length = 71,
    Cluster = 72,
    Lemma = 73,
    Pos = 74,
    Tag = 75,
    Dep = 76,
    EntIob = 77,
    EntType = 78,
    Head = 79,
    SentStart = 80,
    Spacy = 81,
    Prob = 82,
    Lang = 83,
    EntKbId = 84,
    Morph = 85,
    EntId = 86,
    Idx = 87,
    SentEnd = 88,
    /// The token lemma's interlingua id as a u64 (ROADMAP §10.2; 0 = None).
    InterlinguaLemmaId = 89,
    /// The token's entity interlingua id as a u64 (0 = None).
    InterlinguaEntityId = 90,
    /// The per-token annotation confidence as f64 bits (0 = None).
    AnnotationConfidence = 91,
    /// Reserved flag slots 19–63 and any unknown id.
    Other(u16),
}

impl Attribute {
    /// The numeric id (`attr_id_t`).
    #[must_use]
    pub const fn id(self) -> u16 {
        match self {
            Self::IsAlpha => 1,
            Self::IsAscii => 2,
            Self::IsDigit => 3,
            Self::IsLower => 4,
            Self::IsPunct => 5,
            Self::IsSpace => 6,
            Self::IsTitle => 7,
            Self::IsUpper => 8,
            Self::LikeUrl => 9,
            Self::LikeNum => 10,
            Self::LikeEmail => 11,
            Self::IsStop => 12,
            Self::IsOovDeprecated => 13,
            Self::IsBracket => 14,
            Self::IsQuote => 15,
            Self::IsLeftPunct => 16,
            Self::IsRightPunct => 17,
            Self::IsCurrency => 18,
            Self::Id => 64,
            Self::Orth => 65,
            Self::Lower => 66,
            Self::Norm => 67,
            Self::Shape => 68,
            Self::Prefix => 69,
            Self::Suffix => 70,
            Self::Length => 71,
            Self::Cluster => 72,
            Self::Lemma => 73,
            Self::Pos => 74,
            Self::Tag => 75,
            Self::Dep => 76,
            Self::EntIob => 77,
            Self::EntType => 78,
            Self::Head => 79,
            Self::SentStart => 80,
            Self::Spacy => 81,
            Self::Prob => 82,
            Self::Lang => 83,
            Self::EntKbId => 84,
            Self::Morph => 85,
            Self::EntId => 86,
            Self::Idx => 87,
            Self::SentEnd => 88,
            Self::InterlinguaLemmaId => 89,
            Self::InterlinguaEntityId => 90,
            Self::AnnotationConfidence => 91,
            Self::Other(id) => id,
        }
    }

    /// Reconstruct an [`Attribute`] from a numeric id, mapping the reserved
    /// flag slots 19–63 (and anything unknown) to [`Attribute::Other`].
    #[must_use]
    pub const fn from_id(id: u16) -> Self {
        match id {
            1 => Self::IsAlpha,
            2 => Self::IsAscii,
            3 => Self::IsDigit,
            4 => Self::IsLower,
            5 => Self::IsPunct,
            6 => Self::IsSpace,
            7 => Self::IsTitle,
            8 => Self::IsUpper,
            9 => Self::LikeUrl,
            10 => Self::LikeNum,
            11 => Self::LikeEmail,
            12 => Self::IsStop,
            13 => Self::IsOovDeprecated,
            14 => Self::IsBracket,
            15 => Self::IsQuote,
            16 => Self::IsLeftPunct,
            17 => Self::IsRightPunct,
            18 => Self::IsCurrency,
            64 => Self::Id,
            65 => Self::Orth,
            66 => Self::Lower,
            67 => Self::Norm,
            68 => Self::Shape,
            69 => Self::Prefix,
            70 => Self::Suffix,
            71 => Self::Length,
            72 => Self::Cluster,
            73 => Self::Lemma,
            74 => Self::Pos,
            75 => Self::Tag,
            76 => Self::Dep,
            77 => Self::EntIob,
            78 => Self::EntType,
            79 => Self::Head,
            80 => Self::SentStart,
            81 => Self::Spacy,
            82 => Self::Prob,
            83 => Self::Lang,
            84 => Self::EntKbId,
            85 => Self::Morph,
            86 => Self::EntId,
            87 => Self::Idx,
            88 => Self::SentEnd,
            89 => Self::InterlinguaLemmaId,
            90 => Self::InterlinguaEntityId,
            91 => Self::AnnotationConfidence,
            _ => Self::Other(id),
        }
    }

    /// Whether this attribute is a boolean lexeme flag (id < 64).
    #[must_use]
    pub const fn is_flag(self) -> bool {
        self.id() < 64
    }

    /// Parse a name (case-insensitive) into an attribute, spaCy-style
    /// (`attr_ids[i] = IDS[id_.upper()]`).
    pub fn from_name(name: &str) -> Result<Self, SpacyError> {
        let upper = name.to_ascii_uppercase();
        let attr = match upper.as_str() {
            "IS_ALPHA" => Self::IsAlpha,
            "IS_ASCII" => Self::IsAscii,
            "IS_DIGIT" => Self::IsDigit,
            "IS_LOWER" => Self::IsLower,
            "IS_PUNCT" => Self::IsPunct,
            "IS_SPACE" => Self::IsSpace,
            "IS_TITLE" => Self::IsTitle,
            "IS_UPPER" => Self::IsUpper,
            "LIKE_URL" => Self::LikeUrl,
            "LIKE_NUM" => Self::LikeNum,
            "LIKE_EMAIL" => Self::LikeEmail,
            "IS_STOP" => Self::IsStop,
            "IS_OOV_DEPRECATED" => Self::IsOovDeprecated,
            "IS_BRACKET" => Self::IsBracket,
            "IS_QUOTE" => Self::IsQuote,
            "IS_LEFT_PUNCT" => Self::IsLeftPunct,
            "IS_RIGHT_PUNCT" => Self::IsRightPunct,
            "IS_CURRENCY" => Self::IsCurrency,
            "ID" => Self::Id,
            "ORTH" => Self::Orth,
            "LOWER" => Self::Lower,
            "NORM" => Self::Norm,
            "SHAPE" => Self::Shape,
            "PREFIX" => Self::Prefix,
            "SUFFIX" => Self::Suffix,
            "LENGTH" => Self::Length,
            "CLUSTER" => Self::Cluster,
            "LEMMA" => Self::Lemma,
            "POS" => Self::Pos,
            "TAG" => Self::Tag,
            "DEP" => Self::Dep,
            "ENT_IOB" => Self::EntIob,
            "ENT_TYPE" => Self::EntType,
            "ENT_KB_ID" => Self::EntKbId,
            "ENT_ID" => Self::EntId,
            "HEAD" => Self::Head,
            "SENT_START" => Self::SentStart,
            "SPACY" => Self::Spacy,
            "PROB" => Self::Prob,
            "LANG" => Self::Lang,
            "MORPH" => Self::Morph,
            "IDX" => Self::Idx,
            "SENT_END" => Self::SentEnd,
            "INTERLINGUA_LEMMA_ID" => Self::InterlinguaLemmaId,
            "INTERLINGUA_ENTITY_ID" => Self::InterlinguaEntityId,
            "ANNOTATION_CONFIDENCE" => Self::AnnotationConfidence,
            other => return Err(SpacyError::UnknownAttributeText(other.to_string())),
        };
        Ok(attr)
    }
}

#[cfg(test)]
#[path = "../tests/attrs.rs"]
mod tests;

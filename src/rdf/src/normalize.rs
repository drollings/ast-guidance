pub struct BlankNodeScope;

pub fn hash_iri(iri: &str) -> i64 {
    let hash = common_core::hash::blake3_hash(iri.as_bytes());
    i64::from_le_bytes(hash[0..8].try_into().unwrap())
}

pub fn hash_blank_node(scope: &str, id: &str) -> i64 {
    let mut input = Vec::with_capacity(8 + scope.len() + 8 + id.len());
    input.extend_from_slice(&scope.len().to_le_bytes());
    input.extend_from_slice(scope.as_bytes());
    input.extend_from_slice(&id.len().to_le_bytes());
    input.extend_from_slice(id.as_bytes());
    let hash = common_core::hash::blake3_hash(&input);
    i64::from_le_bytes(hash[0..8].try_into().unwrap())
}

use crate::XSD_NS;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum XsdType {
    String,
    LangString,
    Integer,
    Decimal,
    Double,
    Boolean,
    DateTime,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TypedValue {
    String,
    LangString,
    Integer(i64),
    Double(f64),
    Boolean(bool),
    DateTime(i64),
    Other,
}

pub fn detect_xsd_type(datatype: Option<&str>) -> XsdType {
    let Some(dt) = datatype else {
        return XsdType::String;
    };
    if dt == XSD_NS.to_owned() + "string" {
        return XsdType::String;
    }
    if dt == XSD_NS.to_owned() + "integer"
        || dt == XSD_NS.to_owned() + "int"
        || dt == XSD_NS.to_owned() + "long"
        || dt == XSD_NS.to_owned() + "short"
    {
        return XsdType::Integer;
    }
    if dt == XSD_NS.to_owned() + "decimal" {
        return XsdType::Decimal;
    }
    if dt == XSD_NS.to_owned() + "float" || dt == XSD_NS.to_owned() + "double" {
        return XsdType::Double;
    }
    if dt == XSD_NS.to_owned() + "boolean" {
        return XsdType::Boolean;
    }
    if dt == XSD_NS.to_owned() + "dateTime" || dt == XSD_NS.to_owned() + "date" {
        return XsdType::DateTime;
    }
    XsdType::Other
}

pub fn normalize_literal(value: &str, lang: Option<&str>, datatype: Option<&str>) -> TypedValue {
    if lang.is_some() {
        return TypedValue::LangString;
    }
    match detect_xsd_type(datatype) {
        XsdType::Integer => {
            if let Ok(v) = value.trim().parse::<i64>() {
                return TypedValue::Integer(v);
            }
            TypedValue::Other
        }
        XsdType::Decimal | XsdType::Double => {
            if let Ok(v) = value.trim().parse::<f64>() {
                return TypedValue::Double(v);
            }
            TypedValue::Other
        }
        XsdType::Boolean => {
            if value == "true" || value == "1" {
                return TypedValue::Boolean(true);
            }
            if value == "false" || value == "0" {
                return TypedValue::Boolean(false);
            }
            TypedValue::Other
        }
        XsdType::DateTime => TypedValue::DateTime(0),
        XsdType::String => TypedValue::String,
        XsdType::LangString => TypedValue::LangString,
        XsdType::Other => TypedValue::Other,
    }
}

#[cfg(test)]
#[path = "../tests/normalize.rs"]
mod tests;
